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
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from database import ReviewDatabase


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR.parent
DEFAULT_RAW_ROOT = ROOT_DIR / "data" / "raw_archive" / "experiments_20260811"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "datasets" / "performance_v3"
DEFAULT_DB = ROOT_DIR / "data" / "review.sqlite3"
DEFAULT_FORENSICS_MANIFEST = PROJECT_DIR / "data" / "forensics" / "forensics_manifest.json"
DEFAULT_TARGET_TASK_COUNT = 80
DEFAULT_CONTROL_COUNT = 8

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


def infer_model_name(value: str) -> str | None:
    lowered = value.strip().lower()
    for marker, model_id in KNOWN_MODELS:
        if marker in lowered:
            return model_id
    return None


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
        forensics_manifest: Path | None,
    ) -> None:
        self.raw_root = raw_root.resolve()
        self.output_dir = output_dir.resolve()
        self.db_path = db_path.resolve()
        self.dataset_id = dataset_id
        self.per_ip_quota = per_ip_quota
        self.target_task_count = target_task_count
        self.control_count = control_count
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

    def asset(self, path: Path, role: str) -> str:
        path = path.resolve()
        media_type = detect_media_type(path)
        if not media_type:
            raise ValueError(f"Unsupported media asset: {path}")
        file_hash = sha256_file(path)
        existing = self.seen_hashes.get(file_hash)
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
            "metadata": {"role": role},
        }
        self.assets[asset_id] = record
        self.seen_hashes[file_hash] = asset_id
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
        for path in unique_candidates:
            asset_id = self.asset(path, "candidate")
            model_id = infer_model_name(path.stem) or prompt_model
            candidate_records.append(
                {
                    "candidate_id": (
                        f"{stable_slug(batch.name)}_{stable_slug(path.stem)}"
                    ),
                    "model_id": model_id,
                    "origin_type": "ai",
                    "asset_id": asset_id,
                    "variant": path.stem,
                }
            )

        case_id = f"raw_{stable_slug(batch.name)}"
        focus = focus_for_prompt(prompt)
        for left, right in itertools.combinations(candidate_records, 2):
            pair_key = f"{left['candidate_id']}_vs_{right['candidate_id']}"
            pair_hash = hashlib.sha1(pair_key.encode("utf-8")).hexdigest()[:12]
            task = {
                "dataset_id": self.dataset_id,
                "task_id": f"{case_id}__pair_{pair_hash}",
                "case_id": case_id,
                "status": "ready",
                "modality": modality,
                "prompt": prompt,
                "references": references,
                "candidates": [left, right],
                "metadata": {
                    "source_batch": str(batch),
                    "prompt_source": prompt_source,
                    "prompt_model": prompt_model,
                    "focus": focus,
                    "source_kind": "raw_experiment_pair",
                },
            }
            self.tasks.append(task)

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
        real_by_expression: dict[str, list[dict[str, Any]]] = {}
        for record in real:
            expression = str(record.get("expression_class") or "unknown")
            real_by_expression.setdefault(expression, []).append(record)
        real_pool = [
            record
            for expression in sorted(real_by_expression)
            for record in real_by_expression[expression]
        ]
        if not real_pool:
            return

        for index, generated_record in enumerate(generated):
            real_record = real_pool[index % len(real_pool)]
            generated_path = Path(generated_record["video_path"])
            real_path = Path(real_record["video_path"])
            if not generated_path.is_file() or not real_path.is_file():
                continue
            generated_asset = self.asset(generated_path, "ai_candidate")
            real_asset = self.asset(real_path, "real_candidate")
            case_id = f"anchor_{index + 1:03d}"
            self.tasks.append(
                {
                    "dataset_id": self.dataset_id,
                    "task_id": f"{case_id}_ai_vs_real",
                    "case_id": case_id,
                    "status": "ready",
                    "modality": "reference_material",
                    "prompt": "",
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
                    },
                }
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
            if not target.exists():
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
        dataset = {
            "dataset_id": self.dataset_id,
            "version": version,
            "name": f"Human Performance Review {version}",
            "created_at": utc_now(),
            "per_ip_quota": self.per_ip_quota,
            "task_count": len(self.tasks),
            "asset_count": len(self.assets),
            "content_task_count": len(
                [task for task in self.tasks if not task.get("control_type")]
            ),
            "control_task_count": len(
                [task for task in self.tasks if task.get("control_type")]
            ),
            "raw_root": str(self.raw_root),
            "skipped_batch_count": len(self.skipped_batches),
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
        content_target = max(0, self.target_task_count - self.control_count)
        raw_tasks = list(self.tasks)
        if len(raw_tasks) < content_target:
            self._build_anchor_tasks(content_target - len(raw_tasks))
        self.tasks = self.tasks[:content_target]
        self._build_controls(self.control_count)
        self.tasks = self.tasks[: self.target_task_count]
        return self.write_outputs()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--dataset-id", default="performance_v3")
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
        "--forensics-manifest",
        type=Path,
        default=DEFAULT_FORENSICS_MANIFEST,
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
        forensics_manifest=args.forensics_manifest,
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
