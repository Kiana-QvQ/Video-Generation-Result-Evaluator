"""Build the local 25-real + 25-AI single-video evaluation set.

The AI side is composed of:
- 20 uniformly selected samples from the official Seedance holdout;
- the 5 dedicated data/test/AI Change clips.

This is intentionally marked as overlapping the official holdout. The output
is evaluation-only and is never added to any training manifest.

Pairwise reference-content tasks are exported separately by
``scripts/data_tools/export_human_review_reference_set.py`` into ``with_reference/``.
This builder must not recreate the old 25+25 fake-reference copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "test"
REFERENCE_SOURCE = PROJECT_ROOT / "tests" / "test1"
REFERENCE_MOTION_SOURCE = (
    PROJECT_ROOT
    / "outputs"
    / "web_runs"
    / "20260727_101347_579f59223fc2"
    / "reference_motion.mp4"
)
CHANGE_NAMES = (
    "BaiJunZhiJiang_Change.mp4",
    "Happy_Change.mp4",
    "ImissU_Change.mp4",
    "LeJiShengBei_Change.mp4",
    "YanWu_Change.mp4",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uniform_select(
    items: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if len(items) < count:
        raise ValueError(f"Need {count} items, only have {len(items)}.")
    if count == len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    indexes = [
        round(index * (len(items) - 1) / (count - 1))
        for index in range(count)
    ]
    return [items[index] for index in indexes]


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if _sha256(source) == _sha256(destination):
            return "existing"
        destination.unlink()
    try:
        destination.hardlink_to(source)
        return "hardlink"
    except (OSError, NotImplementedError):
        shutil.copy2(source, destination)
        return "copy"


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_pair(item: dict[str, Any]) -> tuple[Path, Path]:
    video = _resolve_project_path(str(item["video"]))
    au = _resolve_project_path(str(item["au"]))
    if not video.is_file():
        raise FileNotFoundError(f"Video missing: {video}")
    if not au.is_file():
        raise FileNotFoundError(f"AU CSV missing: {au}")
    return video, au


def _copy_reference_pack(destination: Path) -> dict[str, str]:
    required = ("front.png", "Left.png", "Right.png", "prompt.txt")
    for name in required:
        source = REFERENCE_SOURCE / name
        if not source.is_file():
            raise FileNotFoundError(f"Reference asset missing: {source}")
        _link_or_copy(source, destination / name)
    if REFERENCE_MOTION_SOURCE.is_file():
        _link_or_copy(
            REFERENCE_MOTION_SOURCE,
            destination / "reference_motion.mp4",
        )
    return {
        "front": "front.png",
        "left": "Left.png",
        "right": "Right.png",
        "prompt": "prompt.txt",
        "motion": (
            "reference_motion.mp4"
            if (destination / "reference_motion.mp4").is_file()
            else ""
        ),
    }


def _build_sample(
    *,
    root: Path,
    split: str,
    index: int,
    label: str,
    video: Path,
    au: Path,
    source_domain: str,
    overlaps_official_holdout: bool,
    reference: bool,
) -> dict[str, Any]:
    sample_id = f"{label}_{index:02d}"
    sample_root = root / split / label / sample_id
    sample_root.mkdir(parents=True, exist_ok=True)
    video_destination = sample_root / "video.mp4"
    au_destination = sample_root / "au.csv"
    _link_or_copy(video, video_destination)
    _link_or_copy(au, au_destination)
    payload: dict[str, Any] = {
        "sample_id": sample_id,
        "label": label,
        "label_generated": int(label == "ai"),
        "video": f"{label}/{sample_id}/video.mp4",
        "au": f"{label}/{sample_id}/au.csv",
        "source_video": _relative(video),
        "source_au": _relative(au),
        "source_domain": source_domain,
        "overlaps_official_holdout": bool(overlaps_official_holdout),
        "training_allowed": False,
    }
    if reference:
        payload["reference"] = _copy_reference_pack(sample_root / "reference")
        payload["reference_root"] = f"{label}/{sample_id}/reference"
        payload["ground_truth"] = None
    _write_json(sample_root / "sample.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the single-video 25+25 forensics test set."
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT.relative_to(PROJECT_ROOT)),
    )
    parser.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split.json",
    )
    args = parser.parse_args(argv)

    output_root = _resolve_project_path(args.output_root)
    holdout = _load_json(_resolve_project_path(args.holdout_manifest))

    real_items = list(holdout.get("real", []))
    ai_holdout_items = list(holdout.get("seedance", []))
    selected_real = _uniform_select(real_items, 25)
    selected_ai_holdout = _uniform_select(ai_holdout_items, 20)

    selected_change: list[dict[str, Any]] = []
    for name in CHANGE_NAMES:
        video = PROJECT_ROOT / "data" / "test" / "AI" / name
        au = PROJECT_ROOT / "data" / "au" / "test" / "AI" / f"{Path(name).stem}.csv"
        if not video.is_file() or not au.is_file():
            raise FileNotFoundError(f"Change pair missing: {name}")
        selected_change.append(
            {
                "video": _relative(video),
                "au": _relative(au),
            }
        )

    (output_root / "single_video").mkdir(parents=True, exist_ok=True)

    manifests: dict[str, dict[str, Any]] = {}
    for split, include_reference in (
        ("single_video", False),
    ):
        samples: list[dict[str, Any]] = []
        for index, item in enumerate(selected_real, start=1):
            video, au = _validate_pair(item)
            samples.append(
                _build_sample(
                    root=output_root,
                    split=split,
                    index=index,
                    label="real",
                    video=video,
                    au=au,
                    source_domain="MD_CL_official_holdout",
                    overlaps_official_holdout=True,
                    reference=include_reference,
                )
            )
        for index, item in enumerate(selected_ai_holdout, start=1):
            video, au = _validate_pair(item)
            samples.append(
                _build_sample(
                    root=output_root,
                    split=split,
                    index=index,
                    label="ai",
                    video=video,
                    au=au,
                    source_domain="WangXing_Seedance_official_holdout",
                    overlaps_official_holdout=True,
                    reference=include_reference,
                )
            )
        for offset, item in enumerate(selected_change, start=1):
            video, au = _validate_pair(item)
            samples.append(
                _build_sample(
                    root=output_root,
                    split=split,
                    index=20 + offset,
                    label="ai",
                    video=video,
                    au=au,
                    source_domain="Change_OOD",
                    overlaps_official_holdout=False,
                    reference=include_reference,
                )
            )
        manifest = {
            "schema_version": "independent_25x25_test_set_v1",
            "split": split,
            "sample_count": len(samples),
            "real_count": sum(item["label"] == "real" for item in samples),
            "ai_count": sum(item["label"] == "ai" for item in samples),
            "training_allowed": False,
            "official_holdout_overlap": True,
            "reference_content": include_reference,
            "samples": samples,
        }
        manifest_path = output_root / split / "manifest.json"
        _write_json(manifest_path, manifest)
        manifests[split] = manifest

    summary = {
        "output_root": str(output_root),
        "single_video": {
            "real": manifests["single_video"]["real_count"],
            "ai": manifests["single_video"]["ai_count"],
        },
        "with_reference": "exported separately from performance_v8",
        "ai_sources": {
            "seedance_holdout": 20,
            "change_ood": 5,
        },
        "training_allowed": False,
        "official_holdout_overlap": True,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
