#!/usr/bin/env python3
"""Build the isolated single-video AI quality review dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from .database import ReviewDatabase
except ImportError:
    from database import ReviewDatabase


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "ai_quality" / "videos"
DEFAULT_MANIFEST = ROOT_DIR / "data" / "ai_quality" / "manifest.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "datasets" / "ai_quality_25plus5_v1"
DEFAULT_DB = ROOT_DIR / "data" / "review.sqlite3"
DEFAULT_QUESTION = "请根据人物的动作、表情和时序连续性，判断这段 AI 视频属于哪个质量档次。"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def stable_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value or "video"


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    items = raw.get("items", []) if isinstance(raw, dict) else raw
    return {
        str(item["file_name"]): dict(item)
        for item in items
        if isinstance(item, dict) and item.get("file_name")
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(
    input_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    db_path: Path,
    dataset_id: str,
    per_reviewer_quota: int,
) -> dict[str, Any]:
    videos = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not videos:
        raise RuntimeError(f"No MP4 videos found in {input_dir}")

    manifest = load_manifest(manifest_path)
    seen_ids: set[str] = set()
    assets: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    created_at = utc_now()

    for index, path in enumerate(videos, start=1):
        annotation = manifest.get(path.name, {})
        sample_id = str(
            annotation.get("sample_id")
            or f"video_{index:03d}_{stable_slug(path.stem)}"
        )
        if sample_id in seen_ids:
            raise RuntimeError(f"Duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)

        asset_id = f"quality_asset_{stable_slug(sample_id)}"
        task_id = f"{dataset_id}__{stable_slug(sample_id)}"
        metadata = {
            "sample_id": sample_id,
            "file_name": path.name,
            "cohort": annotation.get("cohort"),
            "source_domain": annotation.get("source_domain"),
            "program_band": annotation.get("program_band"),
            "human_band": annotation.get("human_band"),
            "source_results": annotation.get("source_results"),
        }
        digest = file_sha256(path)
        assets.append(
            {
                "dataset_id": dataset_id,
                "asset_id": asset_id,
                "source_path": str(path.resolve()),
                "media_type": "video/mp4",
                "original_name": path.name,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "metadata": metadata,
            }
        )
        tasks.append(
            {
                "dataset_id": dataset_id,
                "task_id": task_id,
                "case_id": sample_id,
                "status": "ready",
                "asset_id": asset_id,
                "question": annotation.get("question", DEFAULT_QUESTION),
                "metadata": metadata,
            }
        )

    dataset = {
        "dataset_id": dataset_id,
        "name": "AI Video Quality Rating",
        "version": dataset_id.rsplit("_", 1)[-1],
        "created_at": created_at,
        "per_reviewer_quota": max(0, per_reviewer_quota),
        "per_ip_quota": max(0, per_reviewer_quota),
        "task_count": len(tasks),
        "asset_count": len(assets),
        "rating_values": ["upper", "middle", "lower"],
        "metadata": {
            "purpose": "single_video_ai_quality_rating",
            "input_dir": str(input_dir.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "privacy_note": "Program and human annotations stay server-side.",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "assets.jsonl").open("w", encoding="utf-8") as handle:
        for asset in assets:
            handle.write(json.dumps(asset, ensure_ascii=False) + "\n")
    with (output_dir / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")

    database = ReviewDatabase(db_path, ip_secret="human-review-local-v1")
    database.replace_quality_dataset_bundle(dataset, assets, tasks)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the isolated AI quality rating dataset."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--dataset-id",
        default="ai_quality_25plus5_v1",
        help="Use a new version, such as ai_quality_25plus5_v2, after adding videos.",
    )
    parser.add_argument("--per-reviewer-quota", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_dataset(
        input_dir=args.input_dir,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        db_path=args.db,
        dataset_id=args.dataset_id,
        per_reviewer_quota=max(0, args.per_reviewer_quota),
    )
    print(
        json.dumps(
            {
                "dataset_id": dataset["dataset_id"],
                "task_count": dataset["task_count"],
                "asset_count": dataset["asset_count"],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
