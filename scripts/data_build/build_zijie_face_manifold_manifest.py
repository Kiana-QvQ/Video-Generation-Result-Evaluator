"""Build the group-isolated ZiJie H3 face-manifold experiment manifest.

The source dataset contains the same ten capture IDs in nine real motion
groups. Splitting individual clips would leak a capture across train and
test, so this builder holds out whole capture IDs. H3 samples ending in
``_0`` and ``_1`` are also held out as pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT.parent
    / "MiniMax_H3"
    / "data"
    / "zijie_hk_neutralgray_matte_1k_uncropped_v1"
)
CAPTURE_PATTERN = re.compile(r"^DA\d+$", re.IGNORECASE)
PAIR_SUFFIX_PATTERN = re.compile(r"_[01]$")


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_paths(root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in root.rglob("*.mp4")
            if path.is_file()
        ],
        key=lambda path: path.as_posix().casefold(),
    )


def _real_records(source_root: Path, au_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for category in sorted(
        [
            path
            for path in source_root.iterdir()
            if path.is_dir() and path.name.casefold() != "ai_make"
        ],
        key=lambda path: path.name.casefold(),
    ):
        for video in _video_paths(category):
            relative = video.relative_to(source_root)
            if len(relative.parts) < 3:
                continue
            capture_id = relative.parts[1]
            if not CAPTURE_PATTERN.fullmatch(capture_id):
                continue
            records.append(
                {
                    "video": _relative(video),
                    "au": _relative(au_root / relative.with_suffix(".csv")),
                    "label_generated": 0,
                    "sample_id": (
                        f"real_{category.name.casefold()}_"
                        f"{capture_id.casefold()}"
                    ),
                    "group_id": f"capture_{capture_id.casefold()}",
                    "capture_id": capture_id,
                    "motion_group": category.name,
                    "source_kind": "zijie_neutral_gray_real",
                    "sha256": _sha256(video),
                }
            )
    if len(records) != 90:
        raise ValueError(
            f"Expected 90 neutral-gray real clips, found {len(records)}."
        )
    return records


def _ai_pair_id(path: Path, source_root: Path) -> str:
    relative = path.relative_to(source_root / "AI_Make")
    run = relative.parts[0]
    stem = PAIR_SUFFIX_PATTERN.sub("", path.stem)
    return f"{run}/{stem}"


def _ai_records(source_root: Path, au_root: Path) -> list[dict[str, Any]]:
    root = source_root / "AI_Make"
    records: list[dict[str, Any]] = []
    for video in _video_paths(root):
        relative = video.relative_to(source_root)
        pair_id = _ai_pair_id(video, source_root)
        records.append(
            {
                "video": _relative(video),
                "au": _relative(au_root / relative.with_suffix(".csv")),
                "label_generated": 1,
                "sample_id": (
                    "ai_" + re.sub(r"[^a-z0-9]+", "_", pair_id.casefold())
                    + f"_{video.stem[-1:]}"
                ),
                "group_id": f"h3_pair_{pair_id.casefold()}",
                "h3_run": relative.parts[1] if len(relative.parts) > 1 else "",
                "h3_pair_id": pair_id,
                "source_kind": "h3_generated",
                "sha256": _sha256(video),
            }
        )
    if len(records) != 18:
        raise ValueError(f"Expected 18 H3 AI clips, found {len(records)}.")
    return records


def _require_au(items: Iterable[dict[str, Any]]) -> None:
    missing = [
        str(item["au"])
        for item in items
        if not _path(str(item["au"])).is_file()
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"AU extraction is incomplete ({len(missing)} missing): {preview}"
        )


def _partition(
    real: list[dict[str, Any]],
    ai: list[dict[str, Any]],
    *,
    holdout_capture_ids: set[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    real_train = [
        item for item in real
        if str(item["capture_id"]).casefold()
        not in {value.casefold() for value in holdout_capture_ids}
    ]
    real_test = [
        item for item in real
        if str(item["capture_id"]).casefold()
        in {value.casefold() for value in holdout_capture_ids}
    ]
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in ai:
        by_hash[str(item["sha256"])].append(item)
    unique_ai = [
        sorted(items, key=lambda item: str(item["video"]).casefold())[0]
        for _, items in sorted(by_hash.items())
    ]
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in unique_ai:
        runs[str(item["h3_run"])].append(item)
    if len(runs) != 2 or any(len(items) < 2 for items in runs.values()):
        raise ValueError(
            "Expected two H3 runs with at least two unique contents each."
        )
    # The raw H3 export contains eleven byte-identical placeholder copies.
    # Keep one canonical instance for fitting, and hold out one genuinely
    # distinct sample from each run for a content-disjoint AI evaluation.
    holdout_hashes = {
        str(sorted(items, key=lambda item: str(item["video"]).casefold())[-1]["sha256"])
        for items in runs.values()
    }
    ai_train = [
        item for item in unique_ai if str(item["sha256"]) not in holdout_hashes
    ]
    ai_test = [
        item for item in unique_ai if str(item["sha256"]) in holdout_hashes
    ]
    excluded_duplicates = [
        item
        for digest, items in by_hash.items()
        for item in sorted(items, key=lambda item: str(item["video"]).casefold())[1:]
    ]
    return real_train, real_test, ai_train, ai_test, excluded_duplicates


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--au-root", default="data/au/zijie/source")
    parser.add_argument(
        "--output",
        default="data/zijie/manifests/zijie_face_manifold_v1.json",
    )
    parser.add_argument(
        "--holdout-capture-id",
        action="append",
        default=["DA0897678", "DA0897679"],
    )
    parser.add_argument(
        "--require-au",
        action="store_true",
        help="Fail unless every selected train and holdout video has an AU CSV.",
    )
    args = parser.parse_args(argv)

    source_root = _path(args.source_root)
    au_root = _path(args.au_root)
    if not source_root.is_dir():
        raise SystemExit(f"ZiJie source root was not found: {source_root}")
    real = _real_records(source_root, au_root)
    ai = _ai_records(source_root, au_root)
    capture_ids = {str(item["capture_id"]).casefold() for item in real}
    holdout_ids = {str(value).strip() for value in args.holdout_capture_id}
    unknown = {
        value for value in holdout_ids
        if value.casefold() not in capture_ids
    }
    if unknown:
        raise SystemExit(
            "Unknown holdout capture IDs: " + ", ".join(sorted(unknown))
        )
    real_train, real_test, ai_train, ai_test, excluded_duplicates = _partition(
        real,
        ai,
        holdout_capture_ids=holdout_ids,
    )
    if len(real_train) < 36 or not ai_train or not real_test or not ai_test:
        raise SystemExit("The ZiJie group split is unexpectedly incomplete.")
    train_hashes = {str(item["sha256"]) for item in [*real_train, *ai_train]}
    test_hashes = {str(item["sha256"]) for item in [*real_test, *ai_test]}
    if train_hashes & test_hashes:
        raise SystemExit("Train/test SHA-256 overlap detected.")
    if args.require_au:
        _require_au([*real_train, *ai_train, *real_test, *ai_test])

    payload: dict[str, Any] = {
        "schema_version": "zijie_face_manifold_v1_manifest",
        "subject": "zijie",
        "training_allowed": True,
        "pairs": {
            "train": {"real": real_train, "fake": ai_train},
            "test": {"real": real_test, "fake": ai_test},
        },
        "real_manifold_bank": real_train,
        "counts": {
            "real_total": len(real),
            "ai_total": len(ai),
            "ai_unique_content": len(ai_train) + len(ai_test),
            "ai_excluded_duplicate_content": len(excluded_duplicates),
            "train_real": len(real_train),
            "train_ai": len(ai_train),
            "test_real": len(real_test),
            "test_ai": len(ai_test),
        },
        "split_policy": {
            "real_group_key": "capture_id",
            "real_holdout_capture_ids": sorted(holdout_ids),
            "ai_group_key": "sha256",
            "ai_holdout_unique_contents": sorted(
                str(item["video"]) for item in ai_test
            ),
            "sha256_overlap": False,
        },
        "feature_policy": {
            "full_frame_rgb_used": False,
            "full_frame_hsv_used": False,
            "background_used": False,
            "absolute_brightness_used": False,
            "pose_normalized_face_mesh_used": True,
            "au_trajectory_used": True,
            "local_brightness_centered_face_crops_used": True,
            "mouth_priority": True,
        },
        "test_training_allowed": False,
        "notes": [
            "Neutral-gray real videos estimate the real facial-motion range.",
            "H3 byte-identical duplicates are deduplicated before splitting.",
            "Holdout capture IDs and unique H3 contents never enter profile or checkpoint fitting.",
            "This offline candidate is intentionally not registered in the production webpage.",
        ],
    }
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "counts": payload["counts"],
                "split_policy": payload["split_policy"],
                "au_required": bool(args.require_au),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
