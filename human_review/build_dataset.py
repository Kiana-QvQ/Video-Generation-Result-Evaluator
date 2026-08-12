#!/usr/bin/env python3
"""Build a versioned first human-review dataset from raw experiment folders."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import mimetypes
import re
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from database import ReviewDatabase


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR.parent
DEFAULT_RAW_ROOT = ROOT_DIR / "data" / "raw_archive" / "experiments_20260811"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "datasets" / "performance_v7"
DEFAULT_DB = ROOT_DIR / "data" / "review.sqlite3"
DEFAULT_REUSE_ASSETS_DIR = ROOT_DIR / "data" / "datasets" / "performance_v6"
DEFAULT_BASELINE_DATASET_DIR = ROOT_DIR / "data" / "datasets" / "performance_v6"
DEFAULT_FORENSICS_MANIFEST = PROJECT_DIR / "data" / "forensics" / "forensics_manifest.json"
DEFAULT_TARGET_TASK_COUNT = 80
DEFAULT_CONTROL_COUNT = 8
DEFAULT_MAX_VIDEO_SECONDS = 10
DEFAULT_MAX_VIDEO_WIDTH = 720

KNOWN_MODELS = (
    ("seedance", "seedance_2_0"),
    ("seedance2.0", "seedance_2_0"),
    ("ltx2.3", "ltx2_3"),
    ("ltx", "ltx"),
    ("wan", "wan"),
    ("kling", "kling"),
    ("pika", "pika"),
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".url"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_media_type(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return mimetypes.guess_type(path.name)[0] or "image/*"
    if suffix in AUDIO_EXTENSIONS:
        return mimetypes.guess_type(path.name)[0] or "audio/*"
    if suffix in VIDEO_EXTENSIONS:
        try:
            with path.open("rb") as handle:
                header = handle.read(32)
        except OSError:
            return None
        if suffix == ".url" and b"ftyp" not in header:
            return None
        return "video/mp4"
    return None


def probe_video(path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=width,height",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        payload = json.loads(result.stdout or "{}")
        stream = next(
            (
                item
                for item in payload.get("streams", [])
                if item.get("width") and item.get("height")
            ),
            {},
        )
        return {
            "duration": float(payload.get("format", {}).get("duration") or 0),
            "width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0),
        }
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        return {"duration": 0.0, "width": 0, "height": 0}


def infer_model_name(value: str) -> str | None:
    lowered = re.sub(r"[^a-z0-9]+", "", value.strip().lower())
    for marker, model_id in sorted(
        KNOWN_MODELS,
        key=lambda item: len(re.sub(r"[^a-z0-9]+", "", item[0])),
        reverse=True,
    ):
        marker = re.sub(r"[^a-z0-9]+", "", marker.lower())
        if marker in lowered:
            return model_id
    return None


def canonical_model_id(model_id: str | None) -> str | None:
    if not model_id:
        return None
    return infer_model_name(model_id) or model_id


def parse_prompt(txt_path: Path | None) -> tuple[str, str, str | None]:
    if txt_path is None or not txt_path.exists():
        return "请比较两段视频中人物表演的真人感。", "seedance_2_0", None
    text = txt_path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return "请比较两段视频中人物表演的真人感。", "seedance_2_0", str(txt_path)
    lines = text.splitlines()
    first_line = lines[0].strip()
    model_id = None
    for marker, candidate_model in KNOWN_MODELS:
        if marker in first_line.lower():
            model_id = candidate_model
            break
    if model_id:
        prompt = "\n".join(lines[1:]).strip()
    else:
        model_id = "seedance_2_0"
        prompt = text
    return prompt or "请比较两段视频中人物表演的真人感。", model_id, str(txt_path)


def is_video(path: Path) -> bool:
    return detect_media_type(path) == "video/mp4"


def stable_slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return value[:48] or "item"


def focus_for_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    if any(marker in prompt for marker in ("说", "台词", "语音", "音频")):
        return "speech_and_lip_sync"
    if any(marker in prompt for marker in ("眼神", "目光", "盯着", "看向", "转头")):
        return "gaze_and_head_motion"
    if any(marker in prompt for marker in ("走", "跑", "转身", "动作", "身体", "重心")):
        return "body_motion_and_weight"
    if any(marker in prompt for marker in ("笑", "哭", "表情", "皱眉", "惊讶")):
        return "facial_expression"
    if any(marker in lowered for marker in ("smile", "laugh", "speech", "talk")):
        return "facial_expression"
    return "overall_human_realism"


class DatasetBuilder:
    def __init__(
        self,
        raw_root: Path,
        output_dir: Path,
        db_path: Path,
        dataset_id: str,
        per_ip_quota: int,
        target_task_count: int,
        control_count: int,
        max_video_seconds: int,
        max_video_width: int,
        forensics_manifest: Path | None,
        reuse_assets_dir: Path | None = None,
        baseline_dataset_dir: Path | None = None,
    ) -> None:
        self.raw_root = raw_root.resolve()
        self.output_dir = output_dir.resolve()
        self.db_path = db_path.resolve()
        self.dataset_id = dataset_id
        self.per_ip_quota = per_ip_quota
        self.target_task_count = target_task_count
        self.control_count = control_count
        self.max_video_seconds = max_video_seconds
        self.max_video_width = max_video_width
        self.forensics_manifest = (
            forensics_manifest.resolve() if forensics_manifest else None
        )
        self.database = ReviewDatabase(
            self.db_path,
            ip_secret="human-review-local-v1",
        )
        self.assets: dict[str, dict[str, Any]] = {}
        self.tasks: list[dict[str, Any]] = []
        self.skipped_batches: list[dict[str, Any]] = []
        self.seen_hashes: dict[str, str] = {}
        self.signature_cache: dict[str, np.ndarray | None] = {}
        self.reuse_assets_by_key = self._load_reuse_assets(reuse_assets_dir)
        self.asset_reuse_paths: dict[str, Path] = {}
        self.baseline_anchor_count = self._load_baseline_anchor_count(
            baseline_dataset_dir,
        )
        self.candidate_source_names = self._load_candidate_source_names()
        self.candidate_manifest_models = self._load_candidate_manifest_models()

    @staticmethod
    def _load_reuse_assets(
        assets_dir: Path | None,
    ) -> dict[tuple[str, float | None], Path]:
        if not assets_dir:
            return {}
        manifest = assets_dir / "assets.jsonl"
        if not manifest.exists():
            return {}
        reuse: dict[tuple[str, float | None], Path] = {}
        try:
            rows = (
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            )
            for asset in rows:
                source = Path(asset.get("normalized_path", ""))
                if not source.is_file():
                    continue
                metadata = asset.get("metadata") or {}
                clip_seconds = metadata.get("clip_seconds")
                clip_key = round(float(clip_seconds), 3) if clip_seconds else None
                sha256 = str(asset.get("sha256") or "")
                if sha256:
                    reuse[(sha256, clip_key)] = source
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}
        return reuse

    @staticmethod
    def _load_baseline_anchor_count(dataset_dir: Path | None) -> int | None:
        if not dataset_dir:
            return None
        manifest = dataset_dir / "tasks.jsonl"
        if not manifest.exists():
            return None
        try:
            count = 0
            for line in manifest.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                task = json.loads(line)
                if task.get("metadata", {}).get("source_kind") == "ai_real_anchor":
                    count += 1
            return count or None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _load_candidate_source_names(self) -> dict[str, str]:
        try:
            self.raw_root.iterdir()
        except OSError:
            return {}
        names: dict[str, str] = {}
        for batch in self.raw_root.iterdir():
            if not batch.is_dir():
                continue
            batch_manifest = batch / "rename_manifest.json"
            if not batch_manifest.exists():
                continue
            try:
                batch_payload = json.loads(
                    batch_manifest.read_text(encoding="utf-8-sig"),
                )
            except (OSError, json.JSONDecodeError):
                continue
            for item in batch_payload.get("files", []):
                if item.get("role") != "candidate":
                    continue
                normalized_path = item.get("normalized_path")
                original_name = item.get("original_name")
                if normalized_path and original_name:
                    names[str(Path(normalized_path).resolve())] = str(original_name)
        return names

    def _load_candidate_manifest_models(self) -> dict[str, str]:
        models: dict[str, str] = {}
        try:
            batches = list(self.raw_root.iterdir())
        except OSError:
            return models
        for batch in batches:
            if not batch.is_dir():
                continue
            manifest = batch / "rename_manifest.json"
            if not manifest.exists():
                continue
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            for item in payload.get("files", []):
                if item.get("role") != "candidate":
                    continue
                normalized_path = item.get("normalized_path")
                model_id = canonical_model_id(item.get("model_id"))
                if normalized_path and model_id:
                    models[str(Path(normalized_path).resolve())] = model_id
        return models

    def _candidate_source_name(self, path: Path) -> str:
        return self.candidate_source_names.get(str(path.resolve()), path.name)

    def _candidate_manifest_model(self, path: Path) -> str | None:
        return self.candidate_manifest_models.get(str(path.resolve()))

    @staticmethod
    def _candidate_compare_key(source_name: str) -> str:
        stem = Path(source_name).stem.lower()
        stem = re.sub(
            r"(ltx2?\.?3|ltx|seedance2?\.?0|seedance|4k|gen)",
            "",
            stem,
        )
        stem = re.sub(r"[^a-z0-9]+", "", stem)
        return stem

    def asset(
        self,
        path: Path,
        role: str,
        clip_seconds: float | None = None,
    ) -> str:
        path = path.resolve()
        media_type = detect_media_type(path)
        if not media_type:
            raise ValueError(f"Unsupported media asset: {path}")
        file_hash = sha256_file(path)
        clip_key = f"{file_hash}:{round(clip_seconds or 0, 3)}"
        existing = self.seen_hashes.get(clip_key)
        if existing:
            return existing
        asset_id = f"asset_{len(self.assets) + 1:06d}"
        record = {
            "asset_id": asset_id,
            "source_path": str(path),
            "media_type": media_type,
            "original_name": path.name,
            "sha256": file_hash,
            "size_bytes": path.stat().st_size,
            "metadata": {
                "role": role,
                "clip_seconds": round(clip_seconds, 3)
                if clip_seconds
                else None,
                "probe": probe_video(path) if media_type == "video/mp4" else None,
            },
        }
        self.assets[asset_id] = record
        self.seen_hashes[clip_key] = asset_id
        reuse_key = (
            file_hash,
            round(float(clip_seconds), 3) if clip_seconds else None,
        )
        reuse_path = self.reuse_assets_by_key.get(reuse_key)
        if reuse_path:
            self.asset_reuse_paths[asset_id] = reuse_path
        return asset_id

    def _asset_ref(
        self,
        path: Path,
        role: str,
        label: str = "",
    ) -> dict[str, Any]:
        asset_id = self.asset(path, role)
        return {
            "asset_id": asset_id,
            "type": self.assets[asset_id]["media_type"].split("/")[0],
            "role": role,
            "label": label or path.stem,
        }

    def _find_child(self, batch: Path, *names: str) -> Path | None:
        for name in names:
            direct = batch / name
            if direct.is_dir():
                return direct
        for candidate in batch.rglob("*"):
            if candidate.is_dir() and candidate.name in names:
                return candidate
        return None

    def _build_batch(self, batch: Path) -> None:
        use_dir = self._find_child(batch, "使用素材", "references")
        output_dir = self._find_child(batch, "输出结果", "candidates")
        if output_dir is None:
            self.skipped_batches.append(
                {"batch": str(batch), "reason": "missing_output_directory"}
            )
            return

        candidate_paths = sorted(
            path
            for path in output_dir.rglob("*")
            if path.is_file() and is_video(path)
        )
        unique_candidates: list[Path] = []
        candidate_hashes: set[str] = set()
        for path in candidate_paths:
            file_hash = sha256_file(path)
            if file_hash not in candidate_hashes:
                candidate_hashes.add(file_hash)
                unique_candidates.append(path)
        if len(unique_candidates) < 2:
            self.skipped_batches.append(
                {
                    "batch": str(batch),
                    "reason": "fewer_than_two_unique_output_videos",
                    "output_count": len(unique_candidates),
                }
            )
            return

        txt_paths = sorted(batch.rglob("*.txt"))
        prompt, prompt_model, prompt_source = parse_prompt(
            txt_paths[0] if txt_paths else None
        )
        references: list[dict[str, Any]] = []
        if use_dir:
            for path in sorted(use_dir.rglob("*")):
                if not path.is_file():
                    continue
                media_type = detect_media_type(path)
                if media_type == "image/png" or media_type == "image/jpeg":
                    name_lower = path.name.lower()
                    role = "identity_reference"
                    if name_lower.startswith("bs"):
                        role = "appearance_reference"
                    references.append(self._asset_ref(path, role))
                elif media_type == "video/mp4":
                    references.append(
                        self._asset_ref(path, "motion_driver", "动作参考")
                    )
                elif media_type and media_type.startswith("audio/"):
                    references.append(
                        self._asset_ref(path, "speech_audio", "语音参考")
                    )

        modality = "text_to_video"
        if any(item["role"] == "motion_driver" for item in references):
            modality = "multi_reference"
        elif any(item["type"] == "image" for item in references):
            modality = "image_to_video"

        candidate_records: list[dict[str, Any]] = []
        candidate_probes = {
            path: probe_video(path)
            for path in unique_candidates
        }
        common_duration = min(
            [
                probe["duration"]
                for probe in candidate_probes.values()
                if probe["duration"] > 0
            ]
            + [float(self.max_video_seconds)]
        )
        for path in unique_candidates:
            asset_id = self.asset(
                path,
                "candidate",
                clip_seconds=common_duration,
            )
            model_id = canonical_model_id(
                self._candidate_manifest_model(path)
                or infer_model_name(self._candidate_source_name(path))
                or infer_model_name(path.stem)
                or prompt_model,
            )
            candidate_records.append(
                {
                    "candidate_id": (
                        f"{stable_slug(batch.name)}_{stable_slug(path.stem)}"
                    ),
                    "model_id": model_id,
                    "origin_type": "ai",
                    "asset_id": asset_id,
                    "variant": path.stem,
                    "reveal_label": (
                        "LTX2.3"
                        if model_id == "ltx2_3"
                        else "Seedance 2.0"
                        if model_id == "seedance_2_0"
                        else None
                    ),
                    "_source_name": self._candidate_source_name(path),
                    "_compare_key": self._candidate_compare_key(
                        self._candidate_source_name(path),
                    ),
                }
            )

        case_id = f"raw_{stable_slug(batch.name)}"
        focus = focus_for_prompt(prompt)
        model_comparison_pairs = self._model_comparison_pairs(candidate_records)
        for left, right in model_comparison_pairs:
            pair_key = f"{left['candidate_id']}_vs_{right['candidate_id']}"
            pair_hash = hashlib.sha1(pair_key.encode("utf-8")).hexdigest()[:12]
            self.tasks.append(
                {
                    "dataset_id": self.dataset_id,
                    "task_id": f"{case_id}__model_pair_{pair_hash}",
                    "case_id": case_id,
                    "status": "ready",
                    "modality": modality,
                    "prompt": prompt,
                    "question": (
                        "在相同 Prompt 和参考内容下，"
                        "哪个视频中的人物表演更自然、更像真人？"
                    ),
                    "task_type": "model_comparison",
                    "reveal_mode": "model",
                    "show_context": True,
                    "references": references,
                    "candidates": [
                        self._public_candidate(left),
                        self._public_candidate(right),
                    ],
                    "metadata": {
                        "source_batch": str(batch),
                        "prompt_source": prompt_source,
                        "prompt_model": prompt_model,
                        "focus": focus,
                        "source_kind": "model_comparison",
                        "comparison_key": left.get("_compare_key"),
                    },
                }
            )

        comparison_ids = {
            candidate["candidate_id"]
            for pair in model_comparison_pairs
            for candidate in pair
        }
        for left, right in itertools.combinations(candidate_records, 2):
            if (
                left["candidate_id"] in comparison_ids
                and right["candidate_id"] in comparison_ids
                and left.get("model_id") != right.get("model_id")
            ):
                continue
            pair_key = f"{left['candidate_id']}_vs_{right['candidate_id']}"
            pair_hash = hashlib.sha1(pair_key.encode("utf-8")).hexdigest()[:12]
            if left.get("model_id") != right.get("model_id"):
                manual_task = {
                    "dataset_id": self.dataset_id,
                    "task_id": f"{case_id}__pair_{pair_hash}",
                    "case_id": case_id,
                    "status": "needs_manual_review",
                    "modality": modality,
                    "prompt": prompt,
                    "question": "哪个视频中的人物表演更像真人？",
                    "task_type": "model_comparison",
                    "reveal_mode": "model",
                    "show_context": True,
                    "references": references,
                    "candidates": [
                        self._public_candidate(left),
                        self._public_candidate(right),
                    ],
                    "metadata": {
                        "source_batch": str(batch),
                        "prompt_source": prompt_source,
                        "prompt_model": prompt_model,
                        "focus": focus,
                        "source_kind": "needs_manual_review",
                        "review_reason": "cross_model_pair_requires_manual_review",
                    },
                }
                self.tasks.append(manual_task)
                self.skipped_batches.append(
                    {
                        "batch": str(batch),
                        "reason": "cross_model_pair_requires_manual_review",
                        "candidates": [
                            left.get("_source_name"),
                            right.get("_source_name"),
                        ],
                    }
                )
                continue
            task = {
                "dataset_id": self.dataset_id,
                "task_id": f"{case_id}__pair_{pair_hash}",
                "case_id": case_id,
                "status": "ready",
                "modality": modality,
                "prompt": prompt,
                "question": "哪个视频中的人物表演更像真人？",
                "task_type": "ai_real_anchor",
                "reveal_mode": "origin",
                "show_context": True,
                "references": references,
                "candidates": [
                    self._public_candidate(left),
                    self._public_candidate(right),
                ],
                "metadata": {
                    "source_batch": str(batch),
                    "prompt_source": prompt_source,
                    "prompt_model": prompt_model,
                    "focus": focus,
                    "source_kind": "raw_experiment_pair",
                },
            }
            self.tasks.append(task)

    @staticmethod
    def _public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in candidate.items()
            if not key.startswith("_")
        }

    @staticmethod
    def _model_comparison_pairs(
        candidate_records: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        by_key: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidate_records:
            model_id = candidate.get("model_id")
            compare_key = candidate.get("_compare_key")
            if not model_id or not compare_key:
                continue
            by_key.setdefault(str(compare_key), []).append(candidate)

        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for candidates in by_key.values():
            by_model: dict[str, list[dict[str, Any]]] = {}
            for candidate in candidates:
                by_model.setdefault(str(candidate["model_id"]), []).append(candidate)
            if len(by_model) != 2 or any(len(items) != 1 for items in by_model.values()):
                continue
            left, right = sorted(
                (items[0] for items in by_model.values()),
                key=lambda item: str(item["model_id"]),
            )
            pairs.append((left, right))
        return pairs

    def _load_forensics_records(self) -> list[dict[str, Any]]:
        if not self.forensics_manifest or not self.forensics_manifest.exists():
            return []
        payload = json.loads(self.forensics_manifest.read_text(encoding="utf-8-sig"))
        return [
            record
            for record in payload.get("records", [])
            if record.get("domain") in {"real_wangxing", "generated_wangxing"}
        ]

    def _build_anchor_tasks(self, count: int) -> None:
        records = self._load_forensics_records()
        generated = [
            record for record in records if record.get("domain") == "generated_wangxing"
        ]
        real = [record for record in records if record.get("domain") == "real_wangxing"]
        if not generated or not real:
            return

        generated = generated[: min(count, len(generated))]
        if not real:
            return

        used_real_ids: set[str] = set()
        for index, generated_record in enumerate(generated):
            generated_path = Path(generated_record["video_path"])
            generated_probe = probe_video(generated_path)
            candidates_by_metadata = sorted(
                [
                    record
                    for record in real
                    if record.get("sample_id") not in used_real_ids
                ],
                key=lambda record: self._pair_cost(
                    generated_probe,
                    record,
                    None,
                    None,
                ),
            )
            candidates_for_layout = candidates_by_metadata[:8] or real
            generated_signature = self._video_signature(generated_path)
            real_signatures = {
                str(record.get("sample_id")): self._video_signature(
                    Path(record["video_path"]),
                )
                for record in candidates_for_layout
            }
            real_record = min(
                candidates_for_layout,
                key=lambda record: self._pair_cost(
                    generated_probe,
                    record,
                    generated_signature,
                    real_signatures.get(str(record.get("sample_id"))),
                ),
            )
            used_real_ids.add(str(real_record.get("sample_id")))
            real_path = Path(real_record["video_path"])
            if not generated_path.is_file() or not real_path.is_file():
                continue
            real_probe = probe_video(real_path)
            clip_seconds = min(
                generated_probe["duration"] or self.max_video_seconds,
                real_probe["duration"] or self.max_video_seconds,
                float(self.max_video_seconds),
            )
            generated_asset = self.asset(
                generated_path,
                "ai_candidate",
                clip_seconds=clip_seconds,
            )
            real_asset = self.asset(
                real_path,
                "real_candidate",
                clip_seconds=clip_seconds,
            )
            case_id = f"anchor_{index + 1:03d}"
            self.tasks.append(
                {
                    "dataset_id": self.dataset_id,
                    "task_id": f"{case_id}_ai_vs_real",
                    "case_id": case_id,
                    "status": "ready",
                    "modality": "reference_material",
                    "prompt": "",
                    "question": "哪个视频中的人物表演更像真人？",
                    "task_type": "ai_real_anchor",
                    "reveal_mode": "origin",
                    "show_context": False,
                    "references": [],
                    "candidates": [
                        {
                            "candidate_id": f"{case_id}_generated",
                            "model_id": "generated_wangxing",
                            "origin_type": "ai",
                            "asset_id": generated_asset,
                            "variant": generated_path.stem,
                        },
                        {
                            "candidate_id": f"{case_id}_real",
                            "model_id": "real_wangxing",
                            "origin_type": "real",
                            "asset_id": real_asset,
                            "variant": real_path.stem,
                        },
                    ],
                    "metadata": {
                        "source_kind": "ai_real_anchor",
                        "focus": "overall_human_realism",
                        "generated_sample_id": generated_record.get("sample_id"),
                        "real_sample_id": real_record.get("sample_id"),
                        "real_expression": real_record.get("expression_class"),
                        "pair_duration_seconds": round(clip_seconds, 3),
                        "pairing_method": "frame_layout_duration",
                        "frame_layout_cost": round(
                            self._frame_layout_cost(
                                generated_signature,
                                real_signatures.get(
                                    str(real_record.get("sample_id")),
                                ),
                            ),
                            5,
                        ),
                    },
                }
            )

    def _video_signature(self, path: Path) -> np.ndarray | None:
        key = str(path.resolve())
        if key in self.signature_cache:
            return self.signature_cache[key]
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            self.signature_cache[key] = None
            return None
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            capture.release()
            self.signature_cache[key] = None
            return None
        signatures: list[np.ndarray] = []
        for ratio in (0.08, 0.5, 0.92):
            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                min(frame_count - 1, max(0, int(frame_count * ratio))),
            )
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48, 64), interpolation=cv2.INTER_AREA)
            gray = cv2.normalize(gray, None, 0.0, 1.0, cv2.NORM_MINMAX)
            edges = cv2.Canny((gray * 255).astype(np.uint8), 40, 100)
            edges = edges.astype(np.float32) / 255.0
            signatures.append(
                np.concatenate(
                    [
                        gray.reshape(-1).astype(np.float32),
                        edges.reshape(-1),
                    ],
                ),
            )
        capture.release()
        result = np.stack(signatures) if signatures else None
        self.signature_cache[key] = result
        return result

    @staticmethod
    def _frame_layout_cost(
        generated_signature: np.ndarray | None,
        real_signature: np.ndarray | None,
    ) -> float:
        if generated_signature is None or real_signature is None:
            return 0.5
        distances = [
            float(np.mean((generated_frame - real_frame) ** 2))
            for generated_frame in generated_signature
            for real_frame in real_signature
        ]
        return min(distances) if distances else 0.5

    @staticmethod
    def _pair_cost(
        generated_probe: dict[str, Any],
        real_record: dict[str, Any],
        generated_signature: np.ndarray | None,
        real_signature: np.ndarray | None,
    ) -> float:
        real_probe = real_record.get("video", {})
        generated_duration = generated_probe.get("duration") or 5.0
        real_duration = float(real_probe.get("duration_seconds") or 5.0)
        generated_ratio = generated_probe.get("width", 0) / max(
            generated_probe.get("height", 1),
            1,
        )
        real_ratio = float(real_probe.get("width", 0)) / max(
            float(real_probe.get("height", 1)),
            1,
        )
        frame_cost = DatasetBuilder._frame_layout_cost(
            generated_signature,
            real_signature,
        )
        return (
            frame_cost * 18
            + abs(generated_duration - real_duration) * 0.5
            + abs(generated_ratio - real_ratio) * 8
        )

    def _build_controls(self, count: int) -> None:
        records = self._load_forensics_records()
        if not records:
            return
        per_domain = max(1, count // 2)
        selected: list[dict[str, Any]] = []
        for domain in ("real_wangxing", "generated_wangxing"):
            domain_records = [record for record in records if record.get("domain") == domain]
            if domain == "real_wangxing":
                by_expression: dict[str, list[dict[str, Any]]] = {}
                for record in domain_records:
                    expression = str(record.get("expression_class") or "unknown")
                    by_expression.setdefault(expression, []).append(record)
                buckets = [
                    by_expression[key]
                    for key in sorted(by_expression)
                    if by_expression[key]
                ]
                for index in range(per_domain):
                    bucket = buckets[index % len(buckets)]
                    selected.append(bucket[index // len(buckets) % len(bucket)])
            else:
                selected.extend(domain_records[:per_domain])

        for index, record in enumerate(selected[:count], start=1):
            source_path = Path(record["video_path"])
            if not source_path.exists() or not is_video(source_path):
                continue
            asset_id = self.asset(source_path, "control_video")
            candidate_a = {
                "candidate_id": f"control_{index:03d}_a",
                "model_id": "hidden_control",
                "origin_type": (
                    "real"
                    if record.get("domain") == "real_wangxing"
                    else "ai"
                ),
                "asset_id": asset_id,
            }
            candidate_b = {
                "candidate_id": f"control_{index:03d}_b",
                "model_id": "hidden_control",
                "origin_type": candidate_a["origin_type"],
                "asset_id": asset_id,
            }
            self.tasks.append(
                {
                    "dataset_id": self.dataset_id,
                    "task_id": f"control_{index:03d}",
                    "case_id": f"control_{record.get('sample_id', index)}",
                    "status": "ready",
                    "modality": "reference_material",
                    "prompt": "请比较两段视频中人物表演的真人感。",
                    "question": "哪个视频中的人物表演更像真人？",
                    "task_type": "control",
                    "reveal_mode": "origin",
                    "show_context": False,
                    "references": [],
                    "candidates": [candidate_a, candidate_b],
                    "control_type": "duplicate",
                    "metadata": {
                        "source_kind": "hidden_control",
                        "source_domain": record.get("domain"),
                        "source_sample_id": record.get("sample_id"),
                    },
                }
            )

    def _materialize_assets(self) -> None:
        asset_dir = self.output_dir / "assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        extension_by_type = {
            "video/mp4": ".mp4",
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "audio/wav": ".wav",
        }
        for asset in self.assets.values():
            source = Path(asset["source_path"])
            extension = extension_by_type.get(
                asset["media_type"],
                source.suffix.lower() or ".bin",
            )
            target = asset_dir / f"{asset['asset_id']}{extension}"
            if target.exists():
                target.unlink()
            reused_source = self.asset_reuse_paths.get(asset["asset_id"])
            if reused_source and reused_source.is_file():
                try:
                    target.hardlink_to(reused_source)
                except (OSError, NotImplementedError):
                    shutil.copy2(reused_source, target)
                asset["raw_source_path"] = asset["source_path"]
                asset["normalized_path"] = str(target.resolve())
                asset["source_path"] = str(target.resolve())
                continue
            if asset["media_type"] == "video/mp4":
                temp_target = target.with_name(f".{target.name}.tmp.mp4")
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-i",
                            str(source),
                            "-t",
                            str(
                                asset.get("metadata", {}).get("clip_seconds")
                                or self.max_video_seconds
                            ),
                            "-map",
                            "0:v:0?",
                            "-map",
                            "0:a?",
                            "-vf",
                            f"scale={self.max_video_width}:-2",
                            "-c:v",
                            "libx264",
                            "-preset",
                            "ultrafast",
                            "-crf",
                            "28",
                            "-c:a",
                            "aac",
                            "-b:a",
                            "96k",
                            "-movflags",
                            "+faststart",
                            str(temp_target),
                        ],
                        check=True,
                        timeout=90,
                    )
                    temp_target.replace(target)
                except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    if temp_target.exists():
                        temp_target.unlink()
                    try:
                        target.hardlink_to(source)
                    except (OSError, NotImplementedError):
                        shutil.copy2(source, target)
            else:
                try:
                    target.hardlink_to(source)
                except (OSError, NotImplementedError):
                    shutil.copy2(source, target)
            asset["raw_source_path"] = asset["source_path"]
            asset["normalized_path"] = str(target.resolve())
            asset["source_path"] = str(target.resolve())

    def _write_case_directories(self) -> None:
        case_dir = self.output_dir / "cases"
        case_dir.mkdir(parents=True, exist_ok=True)
        by_case: dict[str, list[dict[str, Any]]] = {}
        for task in self.tasks:
            by_case.setdefault(task["case_id"], []).append(task)
        for case_id, tasks in by_case.items():
            current = case_dir / stable_slug(case_id)
            current.mkdir(parents=True, exist_ok=True)
            first = tasks[0]
            (current / "prompt.txt").write_text(
                first.get("prompt", ""),
                encoding="utf-8",
            )
            (current / "metadata.json").write_text(
                json.dumps(
                    {
                        "case_id": case_id,
                        "task_ids": [task["task_id"] for task in tasks],
                        "focus": first.get("metadata", {}).get("focus"),
                        "references": first.get("references", []),
                        "candidates": [
                            candidate["candidate_id"]
                            for candidate in first["candidates"]
                        ],
                        "source": first.get("metadata", {}),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def write_outputs(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._materialize_assets()
        self._write_case_directories()
        version = self.dataset_id.rsplit("_", 1)[-1]
        reviewable_task_count = sum(
            1
            for task in self.tasks
            if task.get("status") in {"ready", "active"}
        )
        dataset = {
            "dataset_id": self.dataset_id,
            "version": version,
            "name": f"Human Performance Review {version}",
            "created_at": utc_now(),
            "per_ip_quota": self.per_ip_quota,
            "task_count": reviewable_task_count,
            "total_task_count": len(self.tasks),
            "asset_count": len(self.assets),
            "content_task_count": len(
                [
                    task
                    for task in self.tasks
                    if not task.get("control_type")
                    and task.get("status") in {"ready", "active"}
                ]
            ),
            "control_task_count": len(
                [task for task in self.tasks if task.get("control_type")]
            ),
            "manual_review_task_count": len(
                [
                    task
                    for task in self.tasks
                    if task.get("status") == "needs_manual_review"
                ]
            ),
            "requested_task_count": self.target_task_count,
            "raw_root": str(self.raw_root),
            "skipped_batch_count": len(self.skipped_batches),
            "max_video_seconds": self.max_video_seconds,
            "max_video_width": self.max_video_width,
        }
        (self.output_dir / "dataset.json").write_text(
            json.dumps(dataset, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (self.output_dir / "assets.jsonl").open("w", encoding="utf-8") as handle:
            for asset in self.assets.values():
                handle.write(json.dumps(asset, ensure_ascii=False) + "\n")
        with (self.output_dir / "tasks.jsonl").open("w", encoding="utf-8") as handle:
            for task in self.tasks:
                handle.write(json.dumps(task, ensure_ascii=False) + "\n")
        (self.output_dir / "skipped_batches.json").write_text(
            json.dumps(self.skipped_batches, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.database.prepare_dataset_rebuild(self.dataset_id)
        self.database.activate_dataset(dataset)
        for asset in self.assets.values():
            self.database.upsert_asset(asset)
        for task in self.tasks:
            self.database.upsert_task(task)
        return dataset

    def build(self) -> dict[str, Any]:
        for batch in sorted(self.raw_root.iterdir()):
            if batch.is_dir():
                self._build_batch(batch)
        reviewable_content_count = sum(
            1
            for task in self.tasks
            if not task.get("control_type")
            and task.get("status") in {"ready", "active"}
        )
        anchor_count = self.baseline_anchor_count
        if anchor_count is None:
            anchor_count = max(
                0,
                self.target_task_count
                - self.control_count
                - reviewable_content_count,
            )
        self._build_anchor_tasks(anchor_count)
        self._build_controls(self.control_count)
        return self.write_outputs()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--dataset-id", default="performance_v7")
    parser.add_argument("--per-ip-quota", type=int, default=80)
    parser.add_argument(
        "--target-task-count",
        type=int,
        default=DEFAULT_TARGET_TASK_COUNT,
    )
    parser.add_argument(
        "--control-count",
        type=int,
        default=DEFAULT_CONTROL_COUNT,
    )
    parser.add_argument(
        "--max-video-seconds",
        type=int,
        default=DEFAULT_MAX_VIDEO_SECONDS,
    )
    parser.add_argument(
        "--max-video-width",
        type=int,
        default=DEFAULT_MAX_VIDEO_WIDTH,
    )
    parser.add_argument(
        "--forensics-manifest",
        type=Path,
        default=DEFAULT_FORENSICS_MANIFEST,
    )
    parser.add_argument(
        "--reuse-assets-from",
        type=Path,
        default=DEFAULT_REUSE_ASSETS_DIR,
        help="Reuse normalized media from a previous dataset when hashes match.",
    )
    parser.add_argument(
        "--baseline-dataset-dir",
        type=Path,
        default=DEFAULT_BASELINE_DATASET_DIR,
        help="Preserve the baseline anchor-task count from an earlier dataset.",
    )
    args = parser.parse_args()

    builder = DatasetBuilder(
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        db_path=args.db,
        dataset_id=args.dataset_id,
        per_ip_quota=max(0, args.per_ip_quota),
        target_task_count=max(1, args.target_task_count),
        control_count=max(0, args.control_count),
        max_video_seconds=max(1, args.max_video_seconds),
        max_video_width=max(320, args.max_video_width),
        forensics_manifest=args.forensics_manifest,
        reuse_assets_dir=args.reuse_assets_from,
        baseline_dataset_dir=args.baseline_dataset_dir,
    )
    summary = builder.build()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "task_status": Counter(task.get("status") for task in builder.tasks),
                "skipped_batches": len(builder.skipped_batches),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
