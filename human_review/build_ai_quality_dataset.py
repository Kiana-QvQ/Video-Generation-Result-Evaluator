#!/usr/bin/env python3
"""Build the isolated single-video AI quality review dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
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
DEFAULT_SOURCE_QUEUE = ROOT_DIR / "data" / "ai_quality" / "source_queue.json"
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


def load_source_queue(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Source queue does not exist: {path}")
    queue = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(queue, dict):
        raise RuntimeError("Source queue must be a JSON object.")
    return queue


def collect_videos(
    input_dirs: list[tuple[Path, str | None]],
) -> list[tuple[Path, str | None]]:
    videos: list[tuple[Path, str | None]] = []
    seen_names: set[str] = set()
    for input_dir, source_label in input_dirs:
        if not input_dir.is_dir():
            raise RuntimeError(f"Input directory does not exist: {input_dir}")
        for path in sorted(input_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".mp4":
                continue
            if path.name in seen_names:
                raise RuntimeError(
                    f"Duplicate video filename across input directories: {path.name}"
                )
            seen_names.add(path.name)
            videos.append((path, source_label))
    if not videos:
        raise RuntimeError("No MP4 videos found in the configured input directories.")
    return videos


def source_signature(videos: list[tuple[Path, str | None]]) -> str:
    records = [
        {
            "path": str(path.resolve()),
            "label": label,
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path, label in videos
    ]
    payload = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, str | None]:
    ffprobe = os.getenv("FFPROBE_BIN") or shutil.which("ffprobe")
    if not ffprobe:
        return {"codec_name": None, "pix_fmt": None}
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
        stream = streams[0] if streams else {}
        return {
            "codec_name": stream.get("codec_name"),
            "pix_fmt": stream.get("pix_fmt"),
        }
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return {"codec_name": None, "pix_fmt": None}


def make_browser_playback_copy(
    source: Path,
    destination: Path,
    probe: dict[str, str | None],
) -> bool:
    """Create a browser-compatible H.264 copy when the source needs one."""

    if not probe["codec_name"] or not probe["pix_fmt"]:
        shutil.copy2(source, destination)
        return False

    if probe["codec_name"] == "h264" and probe["pix_fmt"] == "yuv420p":
        shutil.copy2(source, destination)
        return False

    ffmpeg = os.getenv("FFMPEG_BIN") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            f"{source.name} uses {probe['codec_name']}/{probe['pix_fmt']}; "
            "FFMPEG_BIN or ffmpeg is required to create a browser copy."
        )

    temporary = destination.with_name(f".{destination.stem}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def build_dataset(
    input_dir: Path | None,
    manifest_path: Path,
    output_dir: Path,
    db_path: Path,
    dataset_id: str,
    per_reviewer_quota: int,
    input_dirs: list[tuple[Path, str | None]] | None = None,
) -> dict[str, Any]:
    configured_dirs = input_dirs or [(input_dir, None)]
    if any(path is None for path, _ in configured_dirs):
        raise RuntimeError("At least one input directory is required.")
    videos = collect_videos(
        [(path, label) for path, label in configured_dirs if path is not None]
    )
    input_signature = source_signature(videos)

    database = ReviewDatabase(db_path, ip_secret="human-review-local-v1")
    if database.count_quality_dataset_votes(dataset_id):
        raise RuntimeError(
            f"Refusing to rebuild {dataset_id}: it already has ratings. "
            "Use a new dataset ID for a new video snapshot."
        )

    manifest = load_manifest(manifest_path)
    seen_ids: set[str] = set()
    assets: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    created_at = utc_now()
    snapshot_dir = output_dir / "assets"
    originals_dir = output_dir / "originals"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    originals_dir.mkdir(parents=True, exist_ok=True)

    for index, (path, source_label) in enumerate(videos, start=1):
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
        original_snapshot_path = originals_dir / path.name
        playback_path = snapshot_dir / path.name
        shutil.copy2(path, original_snapshot_path)
        probe = probe_video(path)
        transcoded = make_browser_playback_copy(
            original_snapshot_path,
            playback_path,
            probe,
        )
        metadata = {
            "sample_id": sample_id,
            "file_name": path.name,
            "cohort": annotation.get("cohort"),
            "source_domain": annotation.get("source_domain"),
            "source_collection": source_label,
            "program_band": annotation.get("program_band"),
            "human_band": annotation.get("human_band"),
            "expression_score": annotation.get("expression_score"),
            "source_results": annotation.get("source_results"),
            "original_snapshot_path": str(original_snapshot_path.resolve()),
            "source_codec": probe["codec_name"],
            "source_pix_fmt": probe["pix_fmt"],
            "playback_transcoded": transcoded,
        }
        digest = file_sha256(playback_path)
        assets.append(
            {
                "dataset_id": dataset_id,
                "asset_id": asset_id,
                "source_path": str(playback_path.resolve()),
                "media_type": "video/mp4",
                "original_name": path.name,
                "sha256": digest,
                "size_bytes": playback_path.stat().st_size,
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
            "input_dirs": [
                {
                    "path": str(path.resolve()),
                    "label": label,
                }
                for path, label in configured_dirs
                if path is not None
            ],
            "manifest_path": str(manifest_path.resolve()),
            "snapshot_dir": str(snapshot_dir.resolve()),
            "originals_dir": str(originals_dir.resolve()),
            "playback_format": "h264_yuv420p",
            "source_signature": input_signature,
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

    database.replace_quality_dataset_bundle(dataset, assets, tasks)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the isolated AI quality rating dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        default=None,
        help="Input folder; repeat this option to add multiple folders.",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--source-queue", type=Path, default=DEFAULT_SOURCE_QUEUE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Use a new version, such as ai_quality_25plus5_v2, after adding videos.",
    )
    parser.add_argument(
        "--per-reviewer-quota",
        type=int,
        default=0,
        help="0 means every reviewer can rate all tasks; set a positive limit if needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue = load_source_queue(args.source_queue)
    queue_dirs = [
        (
            Path(item["path"]),
            item.get("label"),
        )
        for item in queue.get("input_dirs", [])
        if item.get("enabled", True)
    ]
    if args.input_dir:
        input_dirs = [(path, None) for path in args.input_dir]
    else:
        input_dirs = queue_dirs
    dataset_id = args.dataset_id or queue.get(
        "dataset_id",
        "ai_quality_25plus5_v1",
    )
    manifest_value = args.manifest or queue.get("manifest")
    manifest_path = Path(manifest_value) if manifest_value else DEFAULT_MANIFEST
    output_dir = args.output_dir or Path(
        queue.get("output_dir", str(DEFAULT_OUTPUT_DIR))
    )
    dataset = build_dataset(
        input_dir=None,
        manifest_path=manifest_path,
        output_dir=output_dir,
        db_path=args.db,
        dataset_id=dataset_id,
        per_reviewer_quota=max(0, args.per_reviewer_quota),
        input_dirs=input_dirs,
    )
    print(
        json.dumps(
            {
                "dataset_id": dataset["dataset_id"],
                "task_count": dataset["task_count"],
                "asset_count": dataset["asset_count"],
                "output_dir": str(output_dir.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
