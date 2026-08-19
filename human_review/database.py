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
    per_reviewer_quota INTEGER,
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
    reviewer_id_hash TEXT NOT NULL,
    ip_hash TEXT NOT NULL,
    round_id TEXT NOT NULL DEFAULT 'legacy',
    choice TEXT NOT NULL,
    displayed_a_candidate TEXT NOT NULL,
    displayed_b_candidate TEXT NOT NULL,
    response_ms INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE (dataset_id, task_id, reviewer_id_hash),
    FOREIGN KEY (dataset_id, task_id)
        REFERENCES tasks(dataset_id, task_id)
);

CREATE TABLE IF NOT EXISTS quality_datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    per_reviewer_quota INTEGER,
    per_ip_quota INTEGER,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS quality_assets (
    dataset_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    media_type TEXT NOT NULL,
    original_name TEXT NOT NULL,
    sha256 TEXT,
    size_bytes INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (dataset_id, asset_id),
    FOREIGN KEY (dataset_id) REFERENCES quality_datasets(dataset_id)
);

CREATE TABLE IF NOT EXISTS quality_tasks (
    dataset_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ready',
    asset_id TEXT NOT NULL,
    question TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (dataset_id, task_id),
    FOREIGN KEY (dataset_id, asset_id)
        REFERENCES quality_assets(dataset_id, asset_id)
);

CREATE TABLE IF NOT EXISTS quality_votes (
    vote_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    reviewer_id_hash TEXT NOT NULL,
    ip_hash TEXT NOT NULL,
    round_id TEXT NOT NULL DEFAULT 'legacy',
    rating TEXT NOT NULL,
    response_ms INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE (dataset_id, task_id, reviewer_id_hash),
    FOREIGN KEY (dataset_id, task_id)
        REFERENCES quality_tasks(dataset_id, task_id)
);

"""


class ReviewQuotaExceededError(RuntimeError):
    """Raised when a reviewer reaches the configured dataset quota."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def hash_ip(ip_address: str, secret: str) -> str:
    """Create a stable, non-reversible reviewer key."""

    return hmac.new(
        secret.encode("utf-8"),
        ip_address.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def hash_reviewer_id(reviewer_id: str, secret: str) -> str:
    """Create a stable, non-reversible browser identity key."""

    return hmac.new(
        secret.encode("utf-8"),
        reviewer_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def deterministic_swap(
    reviewer_id_hash: str,
    dataset_id: str,
    task_id: str,
    secret: str,
    round_id: str = "legacy",
) -> bool:
    del round_id
    digest = hmac.new(
        secret.encode("utf-8"),
        f"{reviewer_id_hash}:{dataset_id}:{task_id}".encode("utf-8"),
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
            dataset_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(datasets)")
            }
            if "per_reviewer_quota" not in dataset_columns:
                connection.execute(
                    "ALTER TABLE datasets ADD COLUMN per_reviewer_quota INTEGER"
                )
                connection.execute(
                    """
                    UPDATE datasets
                    SET per_reviewer_quota = per_ip_quota
                    WHERE per_reviewer_quota IS NULL
                    """
                )
            vote_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(review_votes)")
            }
            if "reviewer_id_hash" not in vote_columns:
                self._migrate_vote_identity(connection)
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
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_votes_reviewer_task_unique
                    ON review_votes(dataset_id, task_id, reviewer_id_hash)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_quality_assets_dataset
                    ON quality_assets(dataset_id, asset_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_quality_tasks_dataset_status
                    ON quality_tasks(dataset_id, status)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_quality_votes_dataset_ip
                    ON quality_votes(dataset_id, ip_hash, round_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_quality_votes_task
                    ON quality_votes(dataset_id, task_id)
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_quality_votes_reviewer_task_unique
                    ON quality_votes(dataset_id, task_id, reviewer_id_hash)
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

    @staticmethod
    def _migrate_vote_identity(connection: sqlite3.Connection) -> None:
        """Upgrade IP-keyed votes while preserving their historical identity."""

        connection.execute("DROP INDEX IF EXISTS idx_votes_dataset_ip")
        connection.execute("DROP INDEX IF EXISTS idx_votes_task")
        connection.execute("DROP INDEX IF EXISTS idx_votes_reviewer_task_unique")
        connection.execute("ALTER TABLE review_votes RENAME TO review_votes_ip_legacy")
        connection.execute(
            """
            CREATE TABLE review_votes (
                vote_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                reviewer_id_hash TEXT NOT NULL,
                ip_hash TEXT NOT NULL,
                round_id TEXT NOT NULL DEFAULT 'legacy',
                choice TEXT NOT NULL,
                displayed_a_candidate TEXT NOT NULL,
                displayed_b_candidate TEXT NOT NULL,
                response_ms INTEGER,
                created_at TEXT NOT NULL,
                UNIQUE (dataset_id, task_id, reviewer_id_hash),
                FOREIGN KEY (dataset_id, task_id)
                    REFERENCES tasks(dataset_id, task_id)
            )
            """
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO review_votes (
                vote_id, dataset_id, task_id, reviewer_id_hash, ip_hash,
                round_id, choice, displayed_a_candidate, displayed_b_candidate,
                response_ms, created_at
            )
            SELECT vote_id, dataset_id, task_id, ip_hash, ip_hash,
                   round_id, choice, displayed_a_candidate, displayed_b_candidate,
                   response_ms, created_at
            FROM review_votes_ip_legacy
            ORDER BY created_at, vote_id
            """
        )
        connection.execute("DROP TABLE review_votes_ip_legacy")

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
                "DELETE FROM assets WHERE dataset_id = ?",
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
                    per_reviewer_quota, per_ip_quota, created_at, metadata_json
                ) VALUES (?, ?, ?, 'active', ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    name = excluded.name,
                    version = excluded.version,
                    status = 'active',
                    per_reviewer_quota = excluded.per_reviewer_quota,
                    per_ip_quota = excluded.per_ip_quota,
                    metadata_json = excluded.metadata_json
                """,
                (
                    dataset_id,
                    dataset.get("name", dataset_id),
                    dataset.get("version", "v1"),
                    dataset.get(
                        "per_reviewer_quota",
                        dataset.get("per_ip_quota"),
                    ),
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

    def replace_quality_dataset_bundle(
        self,
        dataset: dict[str, Any],
        assets: list[dict[str, Any]],
        tasks: list[dict[str, Any]],
    ) -> None:
        """Replace a vote-free single-video quality dataset.

        Quality datasets intentionally live in separate tables from the
        pairwise review tables so a rebuild can never alter old A/B votes.
        """

        dataset_id = str(dataset["dataset_id"])
        with self.connect() as connection:
            vote_count = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM quality_votes
                WHERE dataset_id = ?
                """,
                (dataset_id,),
            ).fetchone()["count"]
            if vote_count:
                raise RuntimeError(
                    f"Refusing to rebuild quality dataset {dataset_id}: "
                    "it already has ratings."
                )

            connection.execute(
                "DELETE FROM quality_tasks WHERE dataset_id = ?",
                (dataset_id,),
            )
            connection.execute(
                "DELETE FROM quality_assets WHERE dataset_id = ?",
                (dataset_id,),
            )
            connection.execute(
                """
                INSERT INTO quality_datasets (
                    dataset_id, name, version, status,
                    per_reviewer_quota, per_ip_quota, created_at, metadata_json
                ) VALUES (?, ?, ?, 'active', ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    name = excluded.name,
                    version = excluded.version,
                    status = 'active',
                    per_reviewer_quota = excluded.per_reviewer_quota,
                    per_ip_quota = excluded.per_ip_quota,
                    metadata_json = excluded.metadata_json
                """,
                (
                    dataset_id,
                    dataset.get("name", dataset_id),
                    dataset.get("version", "v1"),
                    dataset.get(
                        "per_reviewer_quota",
                        dataset.get("per_ip_quota"),
                    ),
                    dataset.get("per_ip_quota"),
                    dataset.get("created_at", utc_now()),
                    json.dumps(dataset.get("metadata", {}), ensure_ascii=False),
                ),
            )
            connection.executemany(
                """
                INSERT INTO quality_assets (
                    dataset_id, asset_id, source_path, media_type, original_name,
                    sha256, size_bytes, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                INSERT INTO quality_tasks (
                    dataset_id, task_id, case_id, status, asset_id,
                    question, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        dataset_id,
                        task["task_id"],
                        task.get("case_id", task["task_id"]),
                        task.get("status", "ready"),
                        task["asset_id"],
                        task.get("question", ""),
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

    def get_active_quality_dataset(
        self,
        dataset_id: str | None = None,
    ) -> sqlite3.Row | None:
        with self.connect() as connection:
            if dataset_id:
                return connection.execute(
                    """
                    SELECT * FROM quality_datasets
                    WHERE dataset_id = ? AND status = 'active'
                    """,
                    (dataset_id,),
                ).fetchone()
            return connection.execute(
                """
                SELECT * FROM quality_datasets
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

    def get_quality_asset(
        self,
        asset_id: str,
        dataset_id: str | None = None,
    ) -> sqlite3.Row | None:
        with self.connect() as connection:
            if dataset_id:
                return connection.execute(
                    """
                    SELECT * FROM quality_assets
                    WHERE dataset_id = ? AND asset_id = ?
                    """,
                    (dataset_id, asset_id),
                ).fetchone()
            return connection.execute(
                """
                SELECT * FROM quality_assets
                WHERE asset_id = ?
                ORDER BY dataset_id DESC
                LIMIT 1
                """,
                (asset_id,),
            ).fetchone()

    def get_quality_task(
        self,
        dataset_id: str,
        task_id: str,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM quality_tasks
                WHERE dataset_id = ? AND task_id = ?
                """,
                (dataset_id, task_id),
            ).fetchone()
        return self._row_to_quality_task(row) if row else None

    @staticmethod
    def _row_to_quality_task(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "dataset_id": row["dataset_id"],
            "task_id": row["task_id"],
            "case_id": row["case_id"],
            "status": row["status"],
            "asset_id": row["asset_id"],
            "question": row["question"],
            "metadata": json.loads(row["metadata_json"]),
        }

    def get_unrated_quality_tasks(
        self,
        dataset_id: str,
        reviewer_id_hash: str,
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT t.*
                FROM quality_tasks t
                WHERE t.dataset_id = ?
                  AND t.status IN ('ready', 'active')
                  AND NOT EXISTS (
                      SELECT 1 FROM quality_votes own_vote
                      WHERE own_vote.dataset_id = t.dataset_id
                        AND own_vote.task_id = t.task_id
                        AND own_vote.reviewer_id_hash = ?
                  )
                ORDER BY t.task_id
                """,
                (dataset_id, reviewer_id_hash),
            ).fetchall()
        return [self._row_to_quality_task(row) for row in rows]

    def count_quality_tasks(self, dataset_id: str) -> int:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM quality_tasks
                WHERE dataset_id = ?
                  AND status IN ('ready', 'active')
                """,
                (dataset_id,),
            ).fetchone()
        return int(row["count"])

    def count_quality_votes(
        self,
        dataset_id: str,
        reviewer_id_hash: str,
    ) -> int:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM quality_votes v
                JOIN quality_tasks t
                  ON t.dataset_id = v.dataset_id
                 AND t.task_id = v.task_id
                WHERE v.dataset_id = ?
                  AND v.reviewer_id_hash = ?
                  AND t.status IN ('ready', 'active')
                """,
                (dataset_id, reviewer_id_hash),
            ).fetchone()
        return int(row["count"])

    def insert_quality_vote(
        self,
        vote: dict[str, Any],
        per_reviewer_quota: int | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if per_reviewer_quota and per_reviewer_quota > 0:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM quality_votes v
                    JOIN quality_tasks t
                      ON t.dataset_id = v.dataset_id
                     AND t.task_id = v.task_id
                    WHERE v.dataset_id = ?
                      AND v.reviewer_id_hash = ?
                      AND t.status IN ('ready', 'active')
                    """,
                    (
                        vote["dataset_id"],
                        vote["reviewer_id_hash"],
                    ),
                ).fetchone()
                if int(row["count"]) >= per_reviewer_quota:
                    raise ReviewQuotaExceededError
            connection.execute(
                """
                INSERT INTO quality_votes (
                    vote_id, dataset_id, task_id, reviewer_id_hash, ip_hash,
                    round_id, rating, response_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vote.get("vote_id", secrets.token_hex(16)),
                    vote["dataset_id"],
                    vote["task_id"],
                    vote["reviewer_id_hash"],
                    vote["ip_hash"],
                    vote.get("round_id", "legacy"),
                    vote["rating"],
                    vote.get("response_ms"),
                    vote.get("created_at", utc_now()),
                ),
            )

    def get_unvoted_tasks(
        self,
        dataset_id: str,
        reviewer_id_hash: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT t.*, COUNT(v.vote_id) AS vote_count
            FROM tasks t
            LEFT JOIN review_votes v
              ON v.dataset_id = t.dataset_id
              AND v.task_id = t.task_id
            WHERE t.dataset_id = ?
              AND t.status IN ('ready', 'active')
              AND t.control_type IS NULL
              AND t.task_type IN ('ai_real_anchor', 'model_comparison')
              AND NOT EXISTS (
                SELECT 1 FROM review_votes own_vote
                  WHERE own_vote.dataset_id = t.dataset_id
                    AND own_vote.task_id = t.task_id
                    AND own_vote.reviewer_id_hash = ?
              )
            GROUP BY t.dataset_id, t.task_id
            ORDER BY
                CASE WHEN t.task_type = 'model_comparison' THEN 0 ELSE 1 END,
                vote_count ASC,
                t.task_id ASC
        """
        params: list[Any] = [dataset_id, reviewer_id_hash]
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_task(row) for row in rows]

    def count_votes(
        self,
        dataset_id: str,
        reviewer_id_hash: str,
    ) -> int:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM review_votes
                WHERE dataset_id = ?
                  AND reviewer_id_hash = ?
                  AND task_id IN (
                      SELECT task_id
                      FROM tasks
                      WHERE dataset_id = ?
                        AND status IN ('ready', 'active')
                        AND control_type IS NULL
                  )
                """,
                (dataset_id, reviewer_id_hash, dataset_id),
            ).fetchone()
        return int(row["count"])

    def insert_vote(
        self,
        vote: dict[str, Any],
        per_reviewer_quota: int | None = None,
    ) -> None:
        with self.connect() as connection:
            # Serialize quota checks with the insert so parallel tabs cannot
            # both pass a preflight count before either vote is committed.
            connection.execute("BEGIN IMMEDIATE")
            if per_reviewer_quota and per_reviewer_quota > 0:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM review_votes v
                    JOIN tasks t
                      ON t.dataset_id = v.dataset_id
                     AND t.task_id = v.task_id
                    WHERE v.dataset_id = ?
                      AND v.reviewer_id_hash = ?
                      AND t.status IN ('ready', 'active')
                      AND t.control_type IS NULL
                      AND t.task_type IN ('ai_real_anchor', 'model_comparison')
                    """,
                    (
                        vote["dataset_id"],
                        vote["reviewer_id_hash"],
                    ),
                ).fetchone()
                if int(row["count"]) >= per_reviewer_quota:
                    raise ReviewQuotaExceededError
            connection.execute(
                """
                INSERT INTO review_votes (
                    vote_id, dataset_id, task_id, reviewer_id_hash, ip_hash,
                    round_id,
                    choice,
                    displayed_a_candidate, displayed_b_candidate,
                    response_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vote.get("vote_id", secrets.token_hex(16)),
                    vote["dataset_id"],
                    vote["task_id"],
                    vote["reviewer_id_hash"],
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
