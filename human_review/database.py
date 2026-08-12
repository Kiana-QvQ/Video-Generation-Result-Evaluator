#!/usr/bin/env python3
"""SQLite storage for versioned human-review datasets and votes."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    per_ip_quota INTEGER,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS assets (
    dataset_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    media_type TEXT NOT NULL,
    original_name TEXT NOT NULL,
    sha256 TEXT,
    size_bytes INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (dataset_id, asset_id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);

CREATE TABLE IF NOT EXISTS tasks (
    dataset_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ready',
    modality TEXT NOT NULL,
    prompt TEXT NOT NULL DEFAULT '',
    references_json TEXT NOT NULL DEFAULT '[]',
    candidates_json TEXT NOT NULL,
    control_type TEXT,
    task_type TEXT NOT NULL DEFAULT 'ai_real_anchor',
    question TEXT NOT NULL DEFAULT '',
    reveal_mode TEXT NOT NULL DEFAULT 'origin',
    show_context INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (dataset_id, task_id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);

CREATE TABLE IF NOT EXISTS review_votes (
    vote_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    ip_hash TEXT NOT NULL,
    round_id TEXT NOT NULL DEFAULT 'legacy',
    choice TEXT NOT NULL,
    displayed_a_candidate TEXT NOT NULL,
    displayed_b_candidate TEXT NOT NULL,
    response_ms INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE (dataset_id, task_id, ip_hash, round_id),
    FOREIGN KEY (dataset_id, task_id)
        REFERENCES tasks(dataset_id, task_id)
);

"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def hash_ip(ip_address: str, secret: str) -> str:
    """Create a stable, non-reversible reviewer key."""

    return hmac.new(
        secret.encode("utf-8"),
        ip_address.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def deterministic_swap(
    ip_hash: str,
    dataset_id: str,
    task_id: str,
    secret: str,
    round_id: str = "legacy",
) -> bool:
    digest = hmac.new(
        secret.encode("utf-8"),
        f"{ip_hash}:{dataset_id}:{task_id}:{round_id}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return bool(digest[0] & 1)


class ReviewDatabase:
    """Small database facade shared by the dataset builder and Flask app."""

    def __init__(self, path: Path, ip_secret: str) -> None:
        self.path = Path(path)
        self.ip_secret = ip_secret
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.path,
            timeout=30,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript("PRAGMA journal_mode=WAL;\n" + SCHEMA)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_assets_dataset
                    ON assets(dataset_id, asset_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_dataset_status
                    ON tasks(dataset_id, status)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_votes_dataset_ip
                    ON review_votes(dataset_id, ip_hash, round_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_votes_task
                    ON review_votes(dataset_id, task_id)
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(tasks)")
            }
            migrations = {
                "task_type": "ALTER TABLE tasks ADD COLUMN task_type TEXT NOT NULL DEFAULT 'ai_real_anchor'",
                "question": "ALTER TABLE tasks ADD COLUMN question TEXT NOT NULL DEFAULT ''",
                "reveal_mode": "ALTER TABLE tasks ADD COLUMN reveal_mode TEXT NOT NULL DEFAULT 'origin'",
                "show_context": "ALTER TABLE tasks ADD COLUMN show_context INTEGER NOT NULL DEFAULT 0",
            }
            for column, statement in migrations.items():
                if column not in columns:
                    connection.execute(statement)

    def replace_dataset_bundle(
        self,
        dataset: dict[str, Any],
        assets: list[dict[str, Any]],
        tasks: list[dict[str, Any]],
    ) -> None:
        """Replace a vote-free dataset in one SQLite transaction."""

        dataset_id = str(dataset["dataset_id"])
        with self.connect() as connection:
            vote_count = connection.execute(
                "SELECT COUNT(*) AS count FROM review_votes WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()["count"]
            if vote_count:
                raise RuntimeError(
                    f"Refusing to rebuild {dataset_id}: it already has votes."
                )

            connection.execute(
                "DELETE FROM tasks WHERE dataset_id = ?",
                (dataset_id,),
            )
            connection.execute(
                "UPDATE datasets SET status = 'archived' WHERE dataset_id <> ?",
                (dataset_id,),
            )
            connection.execute(
                """
                INSERT INTO datasets (
                    dataset_id, name, version, status,
                    per_ip_quota, created_at, metadata_json
                ) VALUES (?, ?, ?, 'active', ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    name = excluded.name,
                    version = excluded.version,
                    status = 'active',
                    per_ip_quota = excluded.per_ip_quota,
                    metadata_json = excluded.metadata_json
                """,
                (
                    dataset_id,
                    dataset.get("name", dataset_id),
                    dataset.get("version", "v1"),
                    dataset.get("per_ip_quota"),
                    dataset.get("created_at", utc_now()),
                    json.dumps(dataset.get("metadata", {}), ensure_ascii=False),
                ),
            )
            connection.executemany(
                """
                INSERT INTO assets (
                    dataset_id, asset_id, source_path, media_type, original_name,
                    sha256, size_bytes, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id, asset_id) DO UPDATE SET
                    source_path = excluded.source_path,
                    media_type = excluded.media_type,
                    original_name = excluded.original_name,
                    sha256 = excluded.sha256,
                    size_bytes = excluded.size_bytes,
                    metadata_json = excluded.metadata_json
                """,
                [
                    (
                        dataset_id,
                        asset["asset_id"],
                        asset["source_path"],
                        asset.get("media_type", "application/octet-stream"),
                        asset.get(
                            "original_name",
                            Path(asset["source_path"]).name,
                        ),
                        asset.get("sha256"),
                        asset.get("size_bytes"),
                        json.dumps(
                            asset.get("metadata", {}),
                            ensure_ascii=False,
                        ),
                    )
                    for asset in assets
                ],
            )
            connection.executemany(
                """
                INSERT INTO tasks (
                    dataset_id, task_id, case_id, status, modality,
                    prompt, references_json, candidates_json,
                    control_type, task_type, question, reveal_mode,
                    show_context, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        task["dataset_id"],
                        task["task_id"],
                        task.get("case_id", task["task_id"]),
                        task.get("status", "ready"),
                        task.get("modality", "multi_reference"),
                        task.get("prompt", ""),
                        json.dumps(
                            task.get("references", []),
                            ensure_ascii=False,
                        ),
                        json.dumps(task["candidates"], ensure_ascii=False),
                        task.get("control_type"),
                        task.get("task_type", "ai_real_anchor"),
                        task.get("question", ""),
                        task.get("reveal_mode", "origin"),
                        int(bool(task.get("show_context", False))),
                        json.dumps(
                            task.get("metadata", {}),
                            ensure_ascii=False,
                        ),
                    )
                    for task in tasks
                ],
            )

    def get_active_dataset(self, dataset_id: str | None = None) -> sqlite3.Row | None:
        with self.connect() as connection:
            if dataset_id:
                return connection.execute(
                    """
                    SELECT * FROM datasets
                    WHERE dataset_id = ? AND status = 'active'
                    """,
                    (dataset_id,),
                ).fetchone()
            return connection.execute(
                """
                SELECT * FROM datasets
                WHERE status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()

    def get_tasks(self, dataset_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM tasks
                WHERE dataset_id = ?
                  AND status IN ('ready', 'active')
                  AND control_type IS NULL
                ORDER BY task_id
                """,
                (dataset_id,),
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def get_task(self, dataset_id: str, task_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM tasks
                WHERE dataset_id = ? AND task_id = ?
                """,
                (dataset_id, task_id),
            ).fetchone()
        return self._row_to_task(row) if row else None

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "dataset_id": row["dataset_id"],
            "task_id": row["task_id"],
            "case_id": row["case_id"],
            "status": row["status"],
            "modality": row["modality"],
            "prompt": row["prompt"],
            "references": json.loads(row["references_json"]),
            "candidates": json.loads(row["candidates_json"]),
            "control_type": row["control_type"],
            "task_type": row["task_type"],
            "question": row["question"],
            "reveal_mode": row["reveal_mode"],
            "show_context": bool(row["show_context"]),
            "metadata": json.loads(row["metadata_json"]),
        }

    def get_asset(
        self,
        asset_id: str,
        dataset_id: str | None = None,
    ) -> sqlite3.Row | None:
        with self.connect() as connection:
            if dataset_id:
                return connection.execute(
                    """
                    SELECT * FROM assets
                    WHERE dataset_id = ? AND asset_id = ?
                    """,
                    (dataset_id, asset_id),
                ).fetchone()
            return connection.execute(
                """
                SELECT * FROM assets
                WHERE asset_id = ?
                ORDER BY dataset_id DESC
                LIMIT 1
                """,
                (asset_id,),
            ).fetchone()

    def get_unvoted_tasks(
        self,
        dataset_id: str,
        ip_hash: str,
        round_id: str = "legacy",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT t.*, COUNT(v.vote_id) AS vote_count
            FROM tasks t
            LEFT JOIN review_votes v
              ON v.dataset_id = t.dataset_id
             AND v.task_id = t.task_id
             AND v.round_id = ?
            WHERE t.dataset_id = ?
              AND t.status IN ('ready', 'active')
              AND t.control_type IS NULL
              AND t.task_type IN ('ai_real_anchor', 'model_comparison')
              AND NOT EXISTS (
                  SELECT 1 FROM review_votes own_vote
                  WHERE own_vote.dataset_id = t.dataset_id
                    AND own_vote.task_id = t.task_id
                    AND own_vote.ip_hash = ?
                    AND own_vote.round_id = ?
              )
            GROUP BY t.dataset_id, t.task_id
            ORDER BY
                CASE WHEN t.task_type = 'model_comparison' THEN 0 ELSE 1 END,
                vote_count ASC,
                t.task_id ASC
        """
        params: list[Any] = [round_id, dataset_id, ip_hash, round_id]
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_task(row) for row in rows]

    def count_votes(
        self,
        dataset_id: str,
        ip_hash: str,
        round_id: str = "legacy",
    ) -> int:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM review_votes
                WHERE dataset_id = ?
                  AND ip_hash = ?
                  AND round_id = ?
                  AND task_id IN (
                      SELECT task_id
                      FROM tasks
                      WHERE dataset_id = ?
                        AND status IN ('ready', 'active')
                        AND control_type IS NULL
                  )
                """,
                (dataset_id, ip_hash, round_id, dataset_id),
            ).fetchone()
        return int(row["count"])

    def insert_vote(self, vote: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO review_votes (
                    vote_id, dataset_id, task_id, ip_hash, round_id,
                    choice,
                    displayed_a_candidate, displayed_b_candidate,
                    response_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vote.get("vote_id", secrets.token_hex(16)),
                    vote["dataset_id"],
                    vote["task_id"],
                    vote["ip_hash"],
                    vote.get("round_id", "legacy"),
                    vote["choice"],
                    vote["displayed_a_candidate"],
                    vote["displayed_b_candidate"],
                    vote.get("response_ms"),
                    vote.get("created_at", utc_now()),
                ),
            )

    def count_dataset_tasks(self, dataset_id: str) -> int:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM tasks
                WHERE dataset_id = ?
                  AND status IN ('ready', 'active')
                  AND control_type IS NULL
                  AND task_type IN ('ai_real_anchor', 'model_comparison')
                """,
                (dataset_id,),
            ).fetchone()
        return int(row["count"])
