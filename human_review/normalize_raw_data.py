#!/usr/bin/env python3
"""Rename the rough human-review experiment archive into stable case folders."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT_DIR / "data" / "新建文件夹"
DEFAULT_TARGET = ROOT_DIR / "data" / "raw_archive" / "experiments_20260811"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".url"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


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


def infer_model(value: str) -> str:
    lowered = value.lower()
    if "seedance" in lowered:
        return "seedance_2_0"
    if "ltx2.3" in lowered or "ltx2_3" in lowered:
        return "ltx2_3"
    if "wan" in lowered:
        return "wan"
    if "kling" in lowered:
        return "kling"
    return "seedance_2_0"


def prompt_text(batch: Path) -> str:
    candidates = sorted(batch.rglob("*.txt"))
    if not candidates:
        return ""
    return candidates[0].read_text(encoding="utf-8-sig", errors="replace")


def classify_batch(batch: Path) -> str:
    combined = f"{batch.name}\n{prompt_text(batch)[:2400]}".lower()
    rules = [
        (("曹丁元", "头盔视频+王兴"), "helmet_identity_views"),
        (("ue渲染", "nilaiduanhou"), "ue_driver_identity"),
        (("头盔视频", "不上班"), "helmet_happy_dialogue"),
        (("lora", "有效参考"), "lora_reference_test"),
        (("caodingyuan", "替换wangxing"), "identity_replacement"),
        (("跑步", "昏暗走廊"), "running_hallway"),
        (("什么人", "惊恐"), "surprise_question"),
        (("莫挨老子", "烦得很"), "annoyed_head_turn"),
        (("今天不上班", "开心大笑"), "happy_dialogue"),
        (("i miss you", "悲伤"), "sad_english_line"),
        (("乐极生悲", "悲伤"), "happy_to_sad"),
        (("huanrao", "hdr"), "camera_motion_hdr"),
        (("多参", "转头"), "multi_reference_head_turn"),
        (("中景", "平视镜头"), "medium_shot_driver"),
    ]
    for markers, tag in rules:
        if any(marker in combined for marker in markers):
            return tag
    videos = [
        path.stem
        for path in batch.rglob("*")
        if path.is_file() and detect_media_type(path) == "video/mp4"
    ]
    if videos:
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", videos[0]).strip("_").lower()
        return f"unclassified_{safe[:32] or 'video'}"
    return "unclassified_batch"


def unique_name(directory: Path, name: str) -> Path:
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 2
    while True:
        candidate = directory / f"{stem}_{index:02d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def rename_directory(path: Path, name: str) -> Path:
    target = path.with_name(name)
    if target == path:
        return path
    if target.exists():
        raise FileExistsError(f"Rename target already exists: {target}")
    path.rename(target)
    return target


def move_prompt_to_batch(batch: Path, records: list[dict[str, Any]]) -> None:
    txt_files = sorted(batch.rglob("*.txt"))
    if not txt_files:
        return
    source = txt_files[0]
    target = batch / "prompt.txt"
    if source != target:
        target = unique_name(batch, "prompt.txt")
        shutil.move(str(source), str(target))
    records.append(
        {
            "role": "prompt",
            "original_path": str(source),
            "normalized_path": str(target),
            "original_name": source.name,
            "normalized_name": target.name,
            "media_type": "text/plain",
        }
    )


def normalize_reference_files(
    references: Path,
    records: list[dict[str, Any]],
) -> None:
    image_index = 1
    video_index = 1
    audio_index = 1
    for source in sorted(references.iterdir()):
        if not source.is_file() or source.name == "prompt.txt":
            continue
        media_type = detect_media_type(source)
        if media_type in {"image/png", "image/jpeg", "image/webp"}:
            lowered = source.stem.lower()
            if lowered == "front":
                name = "reference_identity_front"
            elif lowered == "left":
                name = "reference_identity_left"
            elif lowered == "right":
                name = "reference_identity_right"
            elif lowered.startswith("bs"):
                name = f"reference_appearance_{image_index:02d}"
                image_index += 1
            else:
                name = f"reference_image_{image_index:02d}"
                image_index += 1
            target = unique_name(references, f"{name}{source.suffix.lower()}")
        elif media_type == "video/mp4":
            target = unique_name(references, f"reference_motion_{video_index:02d}.mp4")
            video_index += 1
        elif media_type and media_type.startswith("audio/"):
            target = unique_name(references, f"reference_audio_{audio_index:02d}{source.suffix.lower()}")
            audio_index += 1
        else:
            continue
        original = source
        source.rename(target)
        records.append(
            {
                "role": "reference",
                "original_path": str(original),
                "normalized_path": str(target),
                "original_name": original.name,
                "normalized_name": target.name,
                "media_type": media_type,
            }
        )


def normalize_candidate_files(
    candidates: Path,
    model_hint: str,
    records: list[dict[str, Any]],
) -> None:
    index = 1
    for source in sorted(candidates.iterdir()):
        if not source.is_file() or detect_media_type(source) != "video/mp4":
            continue
        model_id = infer_model(source.stem)
        if model_id == "seedance_2_0" and "seedance" not in source.stem.lower():
            model_id = model_hint
        extension = ".mp4"
        target = unique_name(
            candidates,
            f"candidate_{index:02d}_{model_id}{extension}",
        )
        index += 1
        original = source
        source.rename(target)
        records.append(
            {
                "role": "candidate",
                "original_path": str(original),
                "normalized_path": str(target),
                "original_name": original.name,
                "normalized_name": target.name,
                "media_type": "video/mp4",
                "model_id": model_id,
            }
        )


def normalize_batch(batch: Path, index: int) -> dict[str, Any]:
    original_name = batch.name
    tag = classify_batch(batch)
    normalized_name = f"exp_{index:03d}_{tag}"
    batch = rename_directory(batch, normalized_name)
    records: list[dict[str, Any]] = []

    reference_dirs = sorted(
        path for path in batch.rglob("使用素材") if path.is_dir()
    )
    candidate_dirs = sorted(
        path for path in batch.rglob("输出结果") if path.is_dir()
    )
    if not reference_dirs:
        reference_dirs = sorted(
            path for path in batch.rglob("references") if path.is_dir()
        )
    if not candidate_dirs:
        candidate_dirs = sorted(
            path for path in batch.rglob("candidates") if path.is_dir()
        )

    references = reference_dirs[0] if reference_dirs else None
    candidates = candidate_dirs[0] if candidate_dirs else None
    if references:
        references = rename_directory(references, "references")
    if candidates:
        candidates = rename_directory(candidates, "candidates")

    move_prompt_to_batch(batch, records)
    prompt_model = infer_model(prompt_text(batch).splitlines()[0] if prompt_text(batch) else "")
    if references:
        normalize_reference_files(references, records)
    if candidates:
        normalize_candidate_files(candidates, prompt_model, records)

    manifest = {
        "original_name": original_name,
        "normalized_name": batch.name,
        "normalized_path": str(batch),
        "prompt_model": prompt_model,
        "files": records,
        "normalized_at": utc_now(),
    }
    (batch / "rename_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    target = args.target.resolve()
    if not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")
    if target.exists():
        raise SystemExit(f"Target directory already exists: {target}")
    if target.parent == source or source in target.parents:
        raise SystemExit("Target must not be inside the source directory.")

    batches = sorted(path for path in source.iterdir() if path.is_dir())
    planned: list[dict[str, str]] = []
    used_names: set[str] = set()
    for index, batch in enumerate(batches, start=1):
        tag = classify_batch(batch)
        name = f"exp_{index:03d}_{tag}"
        if name in used_names:
            name = f"{name}_{index:02d}"
        used_names.add(name)
        planned.append({"original_name": batch.name, "normalized_name": name})

    if args.dry_run:
        print(json.dumps(planned, ensure_ascii=False, indent=2))
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    manifests: list[dict[str, Any]] = []
    for index, batch in enumerate(sorted(target.iterdir()), start=1):
        if batch.is_dir():
            manifests.append(normalize_batch(batch, index))
    (target / "rename_map.json").write_text(
        json.dumps(
            {
                "source_root": str(source),
                "target_root": str(target),
                "renamed_at": utc_now(),
                "batches": manifests,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "target": str(target),
                "batch_count": len(manifests),
                "mapping": str(target / "rename_map.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

