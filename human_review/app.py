#!/usr/bin/env python3
"""Standalone pairwise human review service backed by SQLite."""

from __future__ import annotations

import json
import mimetypes
import os
import secrets
import sqlite3
import uuid
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import quote

from flask import Flask, g, jsonify, request, send_file

try:
    from .database import ReviewDatabase, deterministic_swap, hash_ip
except ImportError:
    from database import ReviewDatabase, deterministic_swap, hash_ip


ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
DATA_DIR = ROOT_DIR / "data"
DEFAULT_DB = DATA_DIR / "review.sqlite3"
LEGACY_MANIFEST = DATA_DIR / "tasks.jsonl"
LEGACY_MEDIA_ROOT = ROOT_DIR / "assets"
DEFAULT_DATASET_ID = "performance_v7"
DEFAULT_IP_SECRET = "human-review-local-v1"
CHOICES = {"A", "B", "tie_or_unrateable"}
READY_STATUSES = {"ready", "active"}


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def normalize_ip(raw_ip: str | None) -> str:
    raw_ip = str(raw_ip or "unknown").strip()
    try:
        return str(ip_address(raw_ip))
    except ValueError:
        return raw_ip[:128]


def detect_mime(path: Path, fallback: str | None = None) -> str:
    guessed = mimetypes.guess_type(path.name)[0]
    if guessed and guessed != "application/octet-stream":
        return guessed
    try:
        with path.open("rb") as handle:
            header = handle.read(32)
    except OSError:
        return fallback or "application/octet-stream"
    if b"ftyp" in header:
        return "video/mp4"
    if header.startswith(b"\x89PNG"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return fallback or "application/octet-stream"


class ReviewStore:
    def __init__(self, db_path: Path, dataset_id: str) -> None:
        ip_secret = os.getenv("HUMAN_REVIEW_IP_SECRET", DEFAULT_IP_SECRET)
        self.database = ReviewDatabase(db_path, ip_secret=ip_secret)
        self.dataset_id = dataset_id
        self._ensure_dataset()

    def _ensure_dataset(self) -> None:
        selected = self.database.get_active_dataset(self.dataset_id)
        if selected and self.database.count_dataset_tasks(selected["dataset_id"]) > 0:
            self.dataset_id = selected["dataset_id"]
            return

        active = self.database.get_active_dataset()
        if active and self.database.count_dataset_tasks(active["dataset_id"]) > 0:
            self.dataset_id = active["dataset_id"]
            return

        self._import_legacy_manifest()

    def _import_legacy_manifest(self) -> None:
        if not LEGACY_MANIFEST.exists():
            self.dataset_id = "empty"
            return
        dataset_id = "demo_v1"
        dataset = {
            "dataset_id": dataset_id,
            "name": "Legacy Demo Dataset",
            "version": "v1",
            "per_ip_quota": None,
            "metadata": {"source": str(LEGACY_MANIFEST)},
        }
        self.database.activate_dataset(dataset)
        try:
            tasks = [
                json.loads(line)
                for line in LEGACY_MANIFEST.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except json.JSONDecodeError:
            self.dataset_id = "empty"
            return

        for task in tasks:
            task["dataset_id"] = dataset_id
            for collection_name in ("references", "candidates"):
                collection = task.get(collection_name, [])
                for asset in collection:
                    path_value = (
                        asset.get("path")
                        or asset.get("video_path")
                        or asset.get("image_path")
                    )
                    if not path_value:
                        continue
                    source = (LEGACY_MEDIA_ROOT / path_value).resolve()
                    asset_id = f"legacy_{uuid.uuid5(uuid.NAMESPACE_URL, str(source)).hex}"
                    asset["asset_id"] = asset_id
                    self.database.upsert_asset(
                        {
                            "asset_id": asset_id,
                            "source_path": str(source),
                            "media_type": detect_mime(source),
                            "original_name": source.name,
                            "metadata": {"legacy_path": path_value},
                        }
                    )
                    if asset.get("poster"):
                        poster_source = (LEGACY_MEDIA_ROOT / asset["poster"]).resolve()
                        poster_id = (
                            f"legacy_{uuid.uuid5(uuid.NAMESPACE_URL, str(poster_source)).hex}"
                        )
                        asset["poster_asset_id"] = poster_id
                        self.database.upsert_asset(
                            {
                                "asset_id": poster_id,
                                "source_path": str(poster_source),
                                "media_type": detect_mime(
                                    poster_source,
                                    "image/svg+xml",
                                ),
                                "original_name": poster_source.name,
                            }
                        )
            self.database.upsert_task(task)
        self.dataset_id = dataset_id

    @property
    def ip_secret(self) -> str:
        return self.database.ip_secret

    def reviewer_hash(self, client_ip: str) -> str:
        return hash_ip(normalize_ip(client_ip), self.ip_secret)

    def client_ip(self) -> str:
        trust_proxy = parse_bool(
            os.getenv("HUMAN_REVIEW_TRUST_PROXY_HEADERS"),
            default=False,
        )
        if trust_proxy:
            forwarded = request.headers.get("X-Forwarded-For", "")
            if forwarded:
                return normalize_ip(forwarded.split(",", 1)[0])
        return normalize_ip(request.remote_addr)

    def dataset_row(self) -> Any:
        return self.database.get_active_dataset(self.dataset_id)

    def progress(self, reviewer_hash: str) -> dict[str, Any]:
        dataset = self.dataset_row()
        total = self.database.count_dataset_tasks(self.dataset_id)
        quota = int(dataset["per_ip_quota"]) if dataset and dataset["per_ip_quota"] else 0
        target = min(total, quota) if quota else total
        completed = self.database.count_votes(self.dataset_id, reviewer_hash)
        return {
            "current": min(completed + 1, target) if target else 0,
            "completed": completed,
            "total": target,
            "dataset_total": total,
            "remaining": max(target - completed, 0),
            "quota": quota or None,
            "done": target > 0 and completed >= target,
        }

    def _asset_url(self, asset: dict[str, Any] | None) -> str | None:
        if not asset:
            return None
        if asset.get("url"):
            return str(asset["url"])
        asset_id = asset.get("asset_id")
        if asset_id:
            return f"/media/asset/{quote(str(asset_id), safe='')}"
        return None

    def _public_reference(self, reference: dict[str, Any]) -> dict[str, Any]:
        public = {
            "type": reference.get("type", "image"),
            "role": reference.get("role", ""),
            "label": reference.get("label", ""),
            "url": self._asset_url(reference),
        }
        poster_asset_id = reference.get("poster_asset_id")
        if poster_asset_id:
            public["poster"] = self._asset_url({"asset_id": poster_asset_id})
        return public

    def _public_task(
        self,
        task: dict[str, Any],
        reviewer_hash: str,
    ) -> dict[str, Any]:
        candidates = list(task["candidates"])
        if len(candidates) != 2:
            raise ValueError(f"Task {task['task_id']} does not contain two candidates.")
        if deterministic_swap(
            reviewer_hash,
            self.dataset_id,
            task["task_id"],
            self.ip_secret,
        ):
            candidates.reverse()

        public_candidates: dict[str, dict[str, Any]] = {}
        for label, candidate in zip(("A", "B"), candidates):
            public_candidates[label] = {
                "url": self._asset_url(candidate),
                "poster": self._asset_url(
                    {"asset_id": candidate.get("poster_asset_id")}
                ),
                "duration": candidate.get("duration"),
                "width": candidate.get("width"),
                "height": candidate.get("height"),
            }

        show_context = bool(task.get("show_context", False))
        return {
            "task_id": task["task_id"],
            "modality": task.get("modality", "multi_reference"),
            "mode": task.get("mode", "random"),
            "prompt": task.get("prompt", "") if show_context else "",
            "question": task.get(
                "question",
                "哪个视频中的人物表演更像真人？",
            ),
            "task_type": task.get("task_type", "ai_real_anchor"),
            "reveal_mode": task.get("reveal_mode", "origin"),
            "show_context": bool(task.get("show_context", False)),
            "focus": task.get("metadata", {}).get("focus", "overall_human_realism"),
            "references": (
                [
                    self._public_reference(reference)
                    for reference in task.get("references", [])
                ]
                if show_context
                else []
            ),
            "candidates": public_candidates,
            "_displayed_candidates": {
                "A": candidates[0]["candidate_id"],
                "B": candidates[1]["candidate_id"],
            },
        }

    def next_task(
        self,
        reviewer_hash: str,
        requested_task_id: str | None = None,
    ) -> dict[str, Any] | None:
        progress = self.progress(reviewer_hash)
        if progress["done"]:
            return None
        requested = (
            self.database.get_task(self.dataset_id, requested_task_id)
            if requested_task_id
            else None
        )
        if requested and requested.get("status") in READY_STATUSES:
            own_votes = self.database.get_unvoted_tasks(
                self.dataset_id,
                reviewer_hash,
            )
            allowed_ids = {task["task_id"] for task in own_votes}
            if requested["task_id"] in allowed_ids:
                return self._public_task(requested, reviewer_hash)

        candidates = self.database.get_unvoted_tasks(
            self.dataset_id,
            reviewer_hash,
        )
        if not candidates:
            return None
        # The database sorts by global vote count; randomize only among the
        # lowest-count window to keep task coverage balanced.
        chosen = secrets.choice(candidates[: min(5, len(candidates))])
        return self._public_task(chosen, reviewer_hash)

    def record_vote(
        self,
        reviewer_hash: str,
        task_id: str,
        choice: str,
        response_ms: int | None,
    ) -> dict[str, Any]:
        progress = self.progress(reviewer_hash)
        if progress["done"]:
            raise ValueError("This reviewer has completed the current dataset quota.")
        if choice not in CHOICES:
            raise ValueError("choice must be A, B, or tie_or_unrateable.")
        task = self.database.get_task(self.dataset_id, task_id)
        if not task or task.get("status") not in READY_STATUSES:
            raise ValueError("Task does not exist or is not available.")
        candidates = list(task["candidates"])
        if len(candidates) != 2:
            raise ValueError("Task must contain exactly two candidates.")
        if deterministic_swap(
            reviewer_hash,
            self.dataset_id,
            task_id,
            self.ip_secret,
        ):
            candidates.reverse()
        reveal_mode = task.get("reveal_mode", "origin")
        reveal = {}
        for label, candidate in zip(("A", "B"), candidates):
            if reveal_mode == "model":
                reveal[label] = {
                    "reveal_mode": "model",
                    "label": candidate.get("reveal_label")
                    or self._model_label(candidate.get("model_id")),
                }
            else:
                reveal[label] = {
                    "reveal_mode": "origin",
                    "origin_type": candidate.get("origin_type", "unknown"),
                    "label": self._origin_label(candidate.get("origin_type")),
                }
        try:
            self.database.insert_vote(
                {
                    "dataset_id": self.dataset_id,
                    "task_id": task_id,
                    "ip_hash": reviewer_hash,
                    "choice": choice,
                    "displayed_a_candidate": candidates[0]["candidate_id"],
                    "displayed_b_candidate": candidates[1]["candidate_id"],
                    "response_ms": response_ms,
                }
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("This task has already been reviewed for this IP.") from exc
        return {
            **self.progress(reviewer_hash),
            "reveal": reveal,
        }

    @staticmethod
    def _origin_label(origin_type: str | None) -> str:
        return {
            "ai": "AI 生成",
            "real": "实拍",
        }.get(origin_type or "unknown", "来源未标注")

    @staticmethod
    def _model_label(model_id: str | None) -> str:
        return {
            "ltx2_3": "LTX2.3",
            "seedance_2_0": "Seedance 2.0",
        }.get(model_id or "", model_id or "模型未标注")

    def media(self, asset_id: str) -> Any:
        asset = self.database.get_asset(asset_id)
        if not asset:
            return jsonify({"error": "Asset not found"}), 404
        path = Path(asset["source_path"])
        if not path.is_file():
            return jsonify({"error": "Asset file not found"}), 404
        return send_file(
            path,
            mimetype=asset["media_type"],
            conditional=True,
        )


def create_app(
    db_path: Path | None = None,
    dataset_id: str | None = None,
) -> Flask:
    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        static_url_path="/assets",
    )
    store = ReviewStore(
        db_path=db_path
        or Path(os.getenv("HUMAN_REVIEW_DB", DEFAULT_DB)),
        dataset_id=dataset_id
        or os.getenv("HUMAN_REVIEW_DATASET", DEFAULT_DATASET_ID),
    )
    app.extensions["review_store"] = store

    def reviewer_hash() -> str:
        if not hasattr(g, "reviewer_hash"):
            g.reviewer_hash = store.reviewer_hash(store.client_ip())
        return g.reviewer_hash

    @app.after_request
    def attach_session_cookie(response: Any) -> Any:
        # The cookie is only a UI convenience. Duplicate prevention is IP-based.
        response.set_cookie(
            "review_session_id",
            request.cookies.get("review_session_id") or uuid.uuid4().hex,
            httponly=False,
            samesite="Lax",
        )
        return response

    @app.get("/")
    def index() -> Any:
        return app.send_static_file("index.html")

    @app.get("/api/review/next")
    def get_next_task() -> Any:
        task = store.next_task(
            reviewer_hash(),
            request.args.get("task_id"),
        )
        payload = {
            "task": task,
            "progress": store.progress(reviewer_hash()),
            "dataset_id": store.dataset_id,
        }
        if task:
            task.pop("_displayed_candidates", None)
        return jsonify(payload)

    @app.get("/api/review/progress")
    def get_progress() -> Any:
        return jsonify(
            {
                "dataset_id": store.dataset_id,
                **store.progress(reviewer_hash()),
            }
        )

    @app.post("/api/review/vote")
    def post_vote() -> Any:
        data = request.get_json(silent=True) or {}
        task_id = str(data.get("task_id", "")).strip()
        choice = str(data.get("choice", "")).strip()
        response_value = data.get("response_ms")
        try:
            response_ms = (
                max(0, min(int(response_value), 86_400_000))
                if response_value is not None
                else None
            )
            progress = store.record_vote(
                reviewer_hash(),
                task_id,
                choice,
                response_ms,
            )
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(
            {
                "status": "recorded",
                "dataset_id": store.dataset_id,
                "progress": progress,
            }
        )

    @app.get("/api/review/health")
    def health() -> Any:
        return jsonify(
            {
                "status": "ok",
                "dataset_id": store.dataset_id,
                "database": str(store.database.path),
                "ip_deduplication": True,
                "task_count": store.database.count_dataset_tasks(store.dataset_id),
            }
        )

    @app.get("/media/asset/<asset_id>")
    def media_asset(asset_id: str) -> Any:
        return store.media(asset_id)

    return app


app = create_app()
