#!/usr/bin/env python3
"""Standalone pairwise human review service backed by SQLite."""

from __future__ import annotations

import os
import re
import secrets
import sqlite3
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import quote

from flask import Flask, g, jsonify, request, send_file

try:
    from .database import (
        ReviewDatabase,
        ReviewQuotaExceededError,
        deterministic_swap,
        hash_ip,
        hash_reviewer_id,
    )
except ImportError:
    from database import (
        ReviewDatabase,
        ReviewQuotaExceededError,
        deterministic_swap,
        hash_ip,
        hash_reviewer_id,
    )


ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
DATA_DIR = ROOT_DIR / "data"
DEFAULT_DB = DATA_DIR / "review.sqlite3"
DEFAULT_DATASET_ID = "performance_v8"
DEFAULT_QUALITY_DATASET_ID = "ai_quality_25plus5_v1"
DEFAULT_IP_SECRET = "human-review-local-v1"
REVIEWER_COOKIE = "human_review_reviewer_id"
REVIEWER_COOKIE_MAX_AGE = 60 * 60 * 24 * 365
CHOICES = {"A", "B", "tie_or_unrateable"}
QUALITY_RATINGS = {"upper", "middle", "lower"}
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


class ReviewStore:
    def __init__(
        self,
        db_path: Path,
        dataset_id: str,
        quality_dataset_id: str | None = None,
    ) -> None:
        ip_secret = os.getenv("HUMAN_REVIEW_IP_SECRET", DEFAULT_IP_SECRET)
        self.database = ReviewDatabase(db_path, ip_secret=ip_secret)
        self.dataset_id = dataset_id
        self.quality_dataset_id = (
            quality_dataset_id
            or os.getenv(
                "HUMAN_REVIEW_QUALITY_DATASET",
                DEFAULT_QUALITY_DATASET_ID,
            )
        )
        self._ensure_dataset()

    def _ensure_dataset(self) -> None:
        selected = self.database.get_active_dataset(self.dataset_id)
        if selected and self.database.count_dataset_tasks(selected["dataset_id"]) > 0:
            self.dataset_id = selected["dataset_id"]
            return

        raise RuntimeError(
            f"Review dataset {self.dataset_id!r} is not available. "
            "Build performance_v8 before starting the review service."
        )

    @property
    def ip_secret(self) -> str:
        return self.database.ip_secret

    def ip_hash(self, client_ip: str) -> str:
        return hash_ip(normalize_ip(client_ip), self.ip_secret)

    def reviewer_id_hash(self, reviewer_id: str) -> str:
        return hash_reviewer_id(reviewer_id, self.ip_secret)

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

    def progress(
        self,
        reviewer_id_hash: str,
        round_id: str = "legacy",
    ) -> dict[str, Any]:
        dataset = self.dataset_row()
        total = self.database.count_dataset_tasks(self.dataset_id)
        quota = self._reviewer_quota(dataset)
        target = min(total, quota) if quota else total
        completed = self.database.count_votes(
            self.dataset_id,
            reviewer_id_hash,
        )
        return {
            "current": min(completed + 1, target) if target else 0,
            "completed": completed,
            "total": target,
            "dataset_total": total,
            "remaining": max(target - completed, 0),
            "quota": quota or None,
            "done": target > 0 and completed >= target,
        }

    @staticmethod
    def _reviewer_quota(dataset: Any) -> int:
        if not dataset:
            return 0
        quota_value = (
            dataset["per_reviewer_quota"]
            if dataset["per_reviewer_quota"] is not None
            else dataset["per_ip_quota"]
        )
        return max(0, int(quota_value)) if quota_value is not None else 0

    def _asset_url(self, asset: dict[str, Any] | None) -> str | None:
        if not asset:
            return None
        if asset.get("url"):
            return str(asset["url"])
        asset_id = asset.get("asset_id")
        if asset_id:
            return (
                f"/media/asset/{quote(self.dataset_id, safe='')}"
                f"/{quote(str(asset_id), safe='')}"
            )
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
        reviewer_id_hash: str,
        round_id: str = "legacy",
    ) -> dict[str, Any]:
        candidates = list(task["candidates"])
        if len(candidates) != 2:
            raise ValueError(f"Task {task['task_id']} does not contain two candidates.")
        self._validate_reviewable_task(task)
        self._validate_task_assets(task)
        if deterministic_swap(
            reviewer_id_hash,
            self.dataset_id,
            task["task_id"],
            self.ip_secret,
            round_id,
        ):
            candidates.reverse()

        public_candidates: dict[str, dict[str, Any]] = {}
        for label, candidate in zip(("A", "B"), candidates):
            asset = self.database.get_asset(
                candidate.get("asset_id", ""),
                self.dataset_id,
            )
            if not asset or asset["media_type"] != "video/mp4":
                raise ValueError(
                    f"Candidate {candidate.get('candidate_id')} is not a video asset."
                )
            public_candidates[label] = {
                "url": self._asset_url(candidate),
                "media_type": asset["media_type"],
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

    def _validate_task_assets(self, task: dict[str, Any]) -> None:
        candidate_ids = {
            candidate.get("asset_id")
            for candidate in task.get("candidates", [])
        }
        for candidate in task.get("candidates", []):
            asset = self.database.get_asset(
                candidate.get("asset_id", ""),
                self.dataset_id,
            )
            if (
                not asset
                or asset["media_type"] != "video/mp4"
                or not Path(asset["source_path"]).is_file()
            ):
                raise ValueError(
                    f"Candidate {candidate.get('candidate_id')} is not an available video asset."
                )
        for reference in task.get("references", []):
            asset_id = reference.get("asset_id")
            if not asset_id or asset_id in candidate_ids:
                raise ValueError("Reference assets must be separate from candidates.")
            asset = self.database.get_asset(asset_id, self.dataset_id)
            if not asset or not Path(asset["source_path"]).is_file():
                raise ValueError(f"Reference asset {asset_id} does not exist.")
            expected_prefix = {
                "image": "image/",
                "video": "video/",
                "audio": "audio/",
            }.get(reference.get("type"))
            if expected_prefix and not asset["media_type"].startswith(expected_prefix):
                raise ValueError(
                    f"Reference asset {asset_id} has an invalid media type."
                )
            poster_id = reference.get("poster_asset_id")
            if poster_id:
                poster = self.database.get_asset(poster_id, self.dataset_id)
                if not poster or not Path(poster["source_path"]).is_file():
                    raise ValueError(f"Reference poster {poster_id} does not exist.")

    @staticmethod
    def _validate_reviewable_task(task: dict[str, Any]) -> None:
        if task.get("control_type"):
            raise ValueError("Control tasks are not user-facing review tasks.")
        candidates = task.get("candidates") or []
        if len(candidates) != 2:
            raise ValueError("Review task must contain exactly two candidates.")
        asset_ids = [candidate.get("asset_id") for candidate in candidates]
        if not all(asset_ids) or asset_ids[0] == asset_ids[1]:
            raise ValueError("Review candidates must be two different video assets.")
        origins = {candidate.get("origin_type") for candidate in candidates}
        if origins == {"ai", "real"}:
            return
        if origins == {"ai"}:
            models = {candidate.get("model_id") for candidate in candidates}
            if len(models) == 2 and all(models):
                return
        raise ValueError(
            "Only AI/real or distinct-model AI/AI pairs can be user-facing."
        )

    def next_task(
        self,
        reviewer_id_hash: str,
        round_id: str = "legacy",
        requested_task_id: str | None = None,
    ) -> dict[str, Any] | None:
        progress = self.progress(reviewer_id_hash, round_id)
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
                reviewer_id_hash,
            )
            allowed_ids = {task["task_id"] for task in own_votes}
            if requested["task_id"] in allowed_ids:
                try:
                    return self._public_task(
                        requested,
                        reviewer_id_hash,
                        round_id,
                    )
                except ValueError:
                    pass

        candidates = self.database.get_unvoted_tasks(
            self.dataset_id,
            reviewer_id_hash,
        )
        valid_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            try:
                self._validate_reviewable_task(candidate)
                self._validate_task_assets(candidate)
            except ValueError:
                continue
            valid_candidates.append(candidate)
        if not valid_candidates:
            raise ValueError("No valid reviewable tasks are available.")
        model_candidates = [
            candidate
            for candidate in valid_candidates
            if candidate.get("task_type") == "model_comparison"
        ]
        pool = model_candidates or valid_candidates
        # Keep model-comparison tasks visible instead of burying them in the
        # random low-vote window.
        chosen = secrets.choice(pool)
        return self._public_task(chosen, reviewer_id_hash, round_id)

    def record_vote(
        self,
        reviewer_id_hash: str,
        ip_hash: str,
        task_id: str,
        choice: str,
        response_ms: int | None,
        round_id: str = "legacy",
    ) -> dict[str, Any]:
        progress = self.progress(reviewer_id_hash, round_id)
        if progress["done"]:
            raise ValueError("This reviewer has completed the current dataset quota.")
        if choice not in CHOICES:
            raise ValueError("choice must be A, B, or tie_or_unrateable.")
        task = self.database.get_task(self.dataset_id, task_id)
        if not task or task.get("status") not in READY_STATUSES:
            raise ValueError("Task does not exist or is not available.")
        self._validate_reviewable_task(task)
        self._validate_task_assets(task)
        candidates = list(task["candidates"])
        if len(candidates) != 2:
            raise ValueError("Task must contain exactly two candidates.")
        if deterministic_swap(
            reviewer_id_hash,
            self.dataset_id,
            task_id,
            self.ip_secret,
            round_id,
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
                    "reviewer_id_hash": reviewer_id_hash,
                    "ip_hash": ip_hash,
                    "round_id": round_id,
                    "choice": choice,
                    "displayed_a_candidate": candidates[0]["candidate_id"],
                    "displayed_b_candidate": candidates[1]["candidate_id"],
                    "response_ms": response_ms,
                },
                per_reviewer_quota=self._reviewer_quota(self.dataset_row()),
            )
        except ReviewQuotaExceededError as exc:
            raise ValueError(
                "This reviewer has completed the current dataset quota."
            ) from exc
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "This task has already been reviewed by this browser."
            ) from exc
        return {
            **self.progress(reviewer_id_hash, round_id),
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

    def media(self, asset_id: str, dataset_id: str | None = None) -> Any:
        selected_dataset = dataset_id or self.dataset_id
        if selected_dataset != self.dataset_id:
            return jsonify({"error": "Dataset not available"}), 404
        asset = self.database.get_asset(asset_id, selected_dataset)
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

    def _quality_dataset_row(self) -> Any:
        selected = self.database.get_active_quality_dataset(
            self.quality_dataset_id,
        )
        if selected and self.database.count_quality_tasks(
            selected["dataset_id"],
        ) > 0:
            return selected
        raise RuntimeError(
            f"Quality dataset {self.quality_dataset_id!r} is not available. "
            "Build the AI quality dataset before opening this section."
        )

    def quality_progress(self, reviewer_id_hash: str) -> dict[str, Any]:
        dataset = self._quality_dataset_row()
        total = self.database.count_quality_tasks(self.quality_dataset_id)
        quota = self._reviewer_quota(dataset)
        target = min(total, quota) if quota else total
        completed = self.database.count_quality_votes(
            self.quality_dataset_id,
            reviewer_id_hash,
        )
        return {
            "current": min(completed + 1, target) if target else 0,
            "completed": completed,
            "total": target,
            "dataset_total": total,
            "remaining": max(target - completed, 0),
            "quota": quota or None,
            "done": target > 0 and completed >= target,
        }

    def _public_quality_task(self, task: dict[str, Any]) -> dict[str, Any]:
        asset = self.database.get_quality_asset(
            task["asset_id"],
            self.quality_dataset_id,
        )
        if (
            not asset
            or asset["media_type"] != "video/mp4"
            or not Path(asset["source_path"]).is_file()
        ):
            raise ValueError(
                f"Quality asset {task['asset_id']} is not an available video."
            )
        cache_version = asset["sha256"] or asset["size_bytes"] or "current"
        return {
            "task_id": task["task_id"],
            "task_type": "ai_quality_rating",
            "question": task.get(
                "question",
                "这段 AI 视频的人物表现属于哪个质量档次？",
            ),
            "video": {
                "url": (
                    f"/media/quality/"
                    f"{quote(self.quality_dataset_id, safe='')}/"
                    f"{quote(str(task['asset_id']), safe='')}"
                    f"?v={quote(str(cache_version), safe='')}"
                ),
                "media_type": asset["media_type"],
                "duration": task.get("metadata", {}).get("duration"),
                "width": task.get("metadata", {}).get("width"),
                "height": task.get("metadata", {}).get("height"),
            },
            "ratings": [
                {"value": "upper", "label": "上档"},
                {"value": "middle", "label": "中档"},
                {"value": "lower", "label": "下档"},
            ],
        }

    def next_quality_task(
        self,
        reviewer_id_hash: str,
        requested_task_id: str | None = None,
    ) -> dict[str, Any] | None:
        progress = self.quality_progress(reviewer_id_hash)
        if progress["done"]:
            return None

        requested = (
            self.database.get_quality_task(
                self.quality_dataset_id,
                requested_task_id,
            )
            if requested_task_id
            else None
        )
        if requested and requested.get("status") in READY_STATUSES:
            own_tasks = self.database.get_unrated_quality_tasks(
                self.quality_dataset_id,
                reviewer_id_hash,
            )
            if requested["task_id"] in {task["task_id"] for task in own_tasks}:
                return self._public_quality_task(requested)

        tasks = self.database.get_unrated_quality_tasks(
            self.quality_dataset_id,
            reviewer_id_hash,
        )
        if not tasks:
            return None
        chosen = secrets.choice(tasks)
        return self._public_quality_task(chosen)

    def record_quality_rating(
        self,
        reviewer_id_hash: str,
        ip_hash: str,
        task_id: str,
        rating: str,
        response_ms: int | None,
        round_id: str = "legacy",
    ) -> dict[str, Any]:
        progress = self.quality_progress(reviewer_id_hash)
        if progress["done"]:
            raise ValueError(
                "This reviewer has completed the current quality dataset quota."
            )
        if rating not in QUALITY_RATINGS:
            raise ValueError("rating must be upper, middle, or lower.")
        task = self.database.get_quality_task(
            self.quality_dataset_id,
            task_id,
        )
        if not task or task.get("status") not in READY_STATUSES:
            raise ValueError("Quality task does not exist or is not available.")
        self._public_quality_task(task)
        try:
            self.database.insert_quality_vote(
                {
                    "dataset_id": self.quality_dataset_id,
                    "task_id": task_id,
                    "reviewer_id_hash": reviewer_id_hash,
                    "ip_hash": ip_hash,
                    "round_id": round_id,
                    "rating": rating,
                    "response_ms": response_ms,
                },
                per_reviewer_quota=self._reviewer_quota(
                    self._quality_dataset_row(),
                ),
            )
        except ReviewQuotaExceededError as exc:
            raise ValueError(
                "This reviewer has completed the current quality dataset quota."
            ) from exc
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "This quality task has already been reviewed by this browser."
            ) from exc
        return self.quality_progress(reviewer_id_hash)

    def quality_media(
        self,
        asset_id: str,
        dataset_id: str | None = None,
    ) -> Any:
        selected_dataset = dataset_id or self.quality_dataset_id
        if selected_dataset != self.quality_dataset_id:
            return jsonify({"error": "Quality dataset not available"}), 404
        asset = self.database.get_quality_asset(asset_id, selected_dataset)
        if not asset:
            return jsonify({"error": "Quality asset not found"}), 404
        path = Path(asset["source_path"])
        if not path.is_file():
            return jsonify({"error": "Quality asset file not found"}), 404
        return send_file(
            path,
            mimetype=asset["media_type"],
            conditional=True,
        )


def create_app(
    db_path: Path | None = None,
    dataset_id: str | None = None,
    quality_dataset_id: str | None = None,
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
        quality_dataset_id=quality_dataset_id,
    )
    app.extensions["review_store"] = store

    def reviewer_id() -> str:
        if not hasattr(g, "reviewer_id"):
            candidate = request.cookies.get(REVIEWER_COOKIE, "")
            if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", candidate):
                candidate = secrets.token_urlsafe(32)
            g.reviewer_id = candidate
        return g.reviewer_id

    def reviewer_id_hash() -> str:
        if not hasattr(g, "reviewer_id_hash"):
            g.reviewer_id_hash = store.reviewer_id_hash(reviewer_id())
        return g.reviewer_id_hash

    def reviewer_ip_hash() -> str:
        if not hasattr(g, "reviewer_ip_hash"):
            g.reviewer_ip_hash = store.ip_hash(store.client_ip())
        return g.reviewer_ip_hash

    def reviewer_round() -> str:
        raw_round = (
            request.headers.get("X-Review-Round")
            or request.args.get("round_id")
            or "legacy"
        )
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", raw_round):
            return "legacy"
        return raw_round

    @app.after_request
    def attach_reviewer_cookie(response: Any) -> Any:
        response.set_cookie(
            REVIEWER_COOKIE,
            reviewer_id(),
            max_age=REVIEWER_COOKIE_MAX_AGE,
            httponly=True,
            samesite="Lax",
            secure=parse_bool(
                os.getenv("HUMAN_REVIEW_COOKIE_SECURE"),
                default=False,
            ),
            path="/",
        )
        return response

    @app.get("/")
    def index() -> Any:
        return app.send_static_file("index.html")

    @app.get("/api/review/next")
    def get_next_task() -> Any:
        try:
            task = store.next_task(
                reviewer_id_hash(),
                reviewer_round(),
                request.args.get("task_id"),
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 503
        payload = {
            "task": task,
            "progress": store.progress(reviewer_id_hash(), reviewer_round()),
            "dataset_id": store.dataset_id,
            "round_id": reviewer_round(),
        }
        if task:
            task.pop("_displayed_candidates", None)
        return jsonify(payload)

    @app.get("/api/review/progress")
    def get_progress() -> Any:
        return jsonify(
            {
                "dataset_id": store.dataset_id,
                "round_id": reviewer_round(),
                **store.progress(reviewer_id_hash(), reviewer_round()),
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
                reviewer_id_hash(),
                reviewer_ip_hash(),
                task_id,
                choice,
                response_ms,
                reviewer_round(),
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
                "reviewer_deduplication": True,
                "ip_audit_hash": True,
                "task_count": store.database.count_dataset_tasks(store.dataset_id),
            }
        )

    @app.get("/api/quality/next")
    def get_next_quality_task() -> Any:
        try:
            task = store.next_quality_task(
                reviewer_id_hash(),
                request.args.get("task_id"),
            )
            progress = store.quality_progress(reviewer_id_hash())
        except (RuntimeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 503
        return jsonify(
            {
                "task": task,
                "progress": progress,
                "dataset_id": store.quality_dataset_id,
                "round_id": reviewer_round(),
            }
        )

    @app.get("/api/quality/progress")
    def get_quality_progress() -> Any:
        try:
            progress = store.quality_progress(reviewer_id_hash())
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503
        return jsonify(
            {
                "dataset_id": store.quality_dataset_id,
                "round_id": reviewer_round(),
                **progress,
            }
        )

    @app.post("/api/quality/rate")
    def post_quality_rating() -> Any:
        data = request.get_json(silent=True) or {}
        task_id = str(data.get("task_id", "")).strip()
        rating = str(data.get("rating", "")).strip()
        response_value = data.get("response_ms")
        try:
            response_ms = (
                max(0, min(int(response_value), 86_400_000))
                if response_value is not None
                else None
            )
            progress = store.record_quality_rating(
                reviewer_id_hash(),
                reviewer_ip_hash(),
                task_id,
                rating,
                response_ms,
                reviewer_round(),
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(
            {
                "status": "recorded",
                "dataset_id": store.quality_dataset_id,
                "progress": progress,
            }
        )

    @app.get("/api/quality/health")
    def quality_health() -> Any:
        try:
            task_count = store.database.count_quality_tasks(
                store.quality_dataset_id,
            )
            available = bool(store._quality_dataset_row())
        except RuntimeError:
            task_count = 0
            available = False
        return jsonify(
            {
                "status": "ok",
                "available": available,
                "dataset_id": store.quality_dataset_id,
                "reviewer_deduplication": True,
                "ip_audit_hash": True,
                "task_count": task_count,
                "ratings": sorted(QUALITY_RATINGS),
            }
        )

    @app.get("/media/asset/<dataset_id>/<asset_id>")
    def media_asset(dataset_id: str, asset_id: str) -> Any:
        return store.media(asset_id, dataset_id)

    @app.get("/media/quality/<dataset_id>/<asset_id>")
    def quality_media_asset(dataset_id: str, asset_id: str) -> Any:
        return store.quality_media(asset_id, dataset_id)

    return app
