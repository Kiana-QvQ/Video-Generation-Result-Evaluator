"""Build a fresh 32-real + 32-AI Wang Xing final test set.

This script only selects and copies/hardlinks evaluation assets. It never
trains a model, builds a profile, or changes the existing 25+25 test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "test" / "wangxing_32x32"
DEFAULT_SPLIT = PROJECT_ROOT / "outputs" / "vedio_pred" / "wangxing_dual_pt_split_res1k.json"
DEFAULT_OLD_TEST = PROJECT_ROOT / "data" / "test" / "single_video" / "manifest.json"
DEFAULT_AI_ROOT = PROJECT_ROOT / "data" / "test" / "xin_AI"
DEFAULT_AI_AU_ROOT = PROJECT_ROOT / "data" / "au" / "test" / "xin_AI"
DEFAULT_PROFILE_EXCLUSION = (
    PROJECT_ROOT / "data" / "forensics" / "wangxing_32x32_profile_exclusion.json"
)
DEFAULT_PT_MANIFEST = (
    PROJECT_ROOT / "outputs" / "vedio_pred" / "wangxing_v3_32x32_holdout_manifest.json"
)
CHANGE_NAMES = {
    "BaiJunZhiJiang_Change.mp4",
    "Happy_Change.mp4",
    "ImissU_Change.mp4",
    "LeJiShengBei_Change.mp4",
    "YanWu_Change.mp4",
}


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


def _norm(path: str | Path) -> str:
    return str(Path(path).resolve()).casefold()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256(source) == _sha256(destination):
        return
    if destination.is_file():
        destination.unlink()
    try:
        destination.hardlink_to(source)
    except (OSError, NotImplementedError):
        shutil.copy2(source, destination)


def _round_robin_select(
    videos: list[Path],
    count: int,
) -> list[Path]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for video in sorted(videos, key=lambda item: _relative(item)):
        groups[video.parent.name].append(video)
    ordered_groups = sorted(groups)
    selected: list[Path] = []
    cursor = 0
    while len(selected) < count:
        added = False
        for group_name in ordered_groups:
            group = groups[group_name]
            if cursor < len(group):
                selected.append(group[cursor])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        cursor += 1
    if len(selected) != count:
        raise ValueError(
            f"Unable to select {count} real videos; got {len(selected)}."
        )
    return selected


def _source_au(video: Path, au_root: Path) -> Path:
    relative = video.resolve().relative_to(
        (PROJECT_ROOT / "data" / "MD_CL").resolve()
    )
    return (au_root / relative).with_suffix(".csv")


def _sample(
    *,
    output_root: Path,
    label: str,
    index: int,
    video: Path,
    au: Path,
    source_domain: str,
    previous_test_overlap: bool,
) -> dict[str, Any]:
    sample_id = f"{label}_{index:02d}"
    sample_root = output_root / "single_video" / label / sample_id
    _link_or_copy(video, sample_root / "video.mp4")
    _link_or_copy(au, sample_root / "au.csv")
    return {
        "sample_id": sample_id,
        "label": label,
        "label_generated": int(label == "ai"),
        "video": f"{label}/{sample_id}/video.mp4",
        "au": f"{label}/{sample_id}/au.csv",
        "source_video": _relative(video),
        "source_au": _relative(au),
        "source_domain": source_domain,
        "training_allowed": False,
        "profile_training_allowed": False,
        "previous_test_overlap": bool(previous_test_overlap),
        "overlaps_pt_train": False,
        "overlaps_official_holdout": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a fresh evaluation-only Wang Xing 32+32 test set."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT))
    parser.add_argument("--old-test-manifest", default=str(DEFAULT_OLD_TEST))
    parser.add_argument("--ai-root", default=str(DEFAULT_AI_ROOT))
    parser.add_argument("--ai-au-root", default=str(DEFAULT_AI_AU_ROOT))
    parser.add_argument(
        "--profile-exclusion",
        default=str(DEFAULT_PROFILE_EXCLUSION),
    )
    parser.add_argument("--pt-manifest", default=str(DEFAULT_PT_MANIFEST))
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root).expanduser().resolve()
    split = _load_json(Path(args.split_manifest).expanduser().resolve())
    old_test = _load_json(Path(args.old_test_manifest).expanduser().resolve())
    ai_root = Path(args.ai_root).expanduser().resolve()
    ai_au_root = Path(args.ai_au_root).expanduser().resolve()

    pt_train_real = {_norm(item) for item in split["train"]["real"]}
    pt_train_fake = {_norm(item) for item in split["train"]["fake"]}
    pt_holdout_real = {_norm(item) for item in split["test"]["real"]}
    pt_holdout_fake = {_norm(item) for item in split["test"]["fake"]}
    old_test_sources = {
        _norm(item["source_video"])
        for item in old_test.get("samples", [])
        if item.get("source_video")
    }
    forbidden_real = pt_train_real | pt_holdout_real | old_test_sources
    forbidden_fake = pt_train_fake | pt_holdout_fake

    real_candidates: list[Path] = []
    for video in (PROJECT_ROOT / "data" / "MD_CL").rglob("*.mp4"):
        key = _norm(video)
        au = _source_au(video, PROJECT_ROOT / "data" / "au" / "MD_CL")
        if key in forbidden_real or not au.is_file():
            continue
        real_candidates.append(video)
    selected_real = _round_robin_select(real_candidates, int(args.count))

    ai_videos = sorted(ai_root.glob("*.mp4"), key=lambda item: item.name.casefold())
    if len(ai_videos) != int(args.count):
        raise ValueError(
            f"Expected {args.count} AI videos in {ai_root}, got {len(ai_videos)}."
        )
    selected_ai: list[tuple[Path, Path, bool]] = []
    for video in ai_videos:
        au = ai_au_root / f"{video.stem}.csv"
        if not au.is_file():
            raise FileNotFoundError(f"Missing AI AU CSV: {au}")
        previous_overlap = video.name in CHANGE_NAMES
        if _norm(video) in forbidden_fake:
            raise ValueError(f"AI final test overlaps PT data: {video}")
        selected_ai.append((video, au, previous_overlap))

    samples: list[dict[str, Any]] = []
    for index, video in enumerate(selected_real, start=1):
        samples.append(
            _sample(
                output_root=output_root,
                label="real",
                index=index,
                video=video,
                au=_source_au(video, PROJECT_ROOT / "data" / "au" / "MD_CL"),
                source_domain="MD_CL_new_real_final",
                previous_test_overlap=False,
            )
        )
    for index, (video, au, previous_overlap) in enumerate(selected_ai, start=1):
        samples.append(
            _sample(
                output_root=output_root,
                label="ai",
                index=index,
                video=video,
                au=au,
                source_domain="xin_AI_final",
                previous_test_overlap=previous_overlap,
            )
        )

    manifest = {
        "schema_version": "wangxing_final_32x32_test_v1",
        "split": "single_video",
        "sample_count": len(samples),
        "real_count": int(args.count),
        "ai_count": int(args.count),
        "training_allowed": False,
        "profile_training_allowed": False,
        "overlaps_pt_train": False,
        "overlaps_official_holdout": False,
        "previous_test_ai_count": sum(
            item["previous_test_overlap"] for item in samples
        ),
        "selection_protocol": {
            "real_excluded_pt_train": True,
            "real_excluded_official_holdout": True,
            "real_excluded_old_25x25": True,
            "ai_source": _relative(ai_root),
            "real_selection": "round_robin_by_parent_folder",
        },
        "samples": samples,
    }
    _write_json(output_root / "single_video" / "manifest.json", manifest)
    for sample in samples:
        _write_json(
            output_root / "single_video" / sample["label"] / sample["sample_id"] / "sample.json",
            sample,
        )

    profile_exclusion = {
        "schema_version": "wangxing_32x32_profile_exclusion_v1",
        "note": (
            "Exclude this final 32+32 set from profile/calibrator fitting. "
            "Do not overwrite the old exclusion files."
        ),
        "real": [
            {"video": item["source_video"], "au": item["source_au"]}
            for item in samples
            if item["label"] == "real"
        ],
        "seedance": [
            {"video": item["source_video"], "au": item["source_au"]}
            for item in samples
            if item["label"] == "ai"
        ],
    }
    _write_json(Path(args.profile_exclusion).expanduser().resolve(), profile_exclusion)

    pt_manifest = {
        "schema_version": "wangxing_final_32x32_pt_manifest_v1",
        "training_allowed": False,
        "real": [
            {"video": item["source_video"], "au": item["source_au"]}
            for item in samples
            if item["label"] == "real"
        ],
        "seedance": [
            {"video": item["source_video"], "au": item["source_au"]}
            for item in samples
            if item["label"] == "ai"
        ],
    }
    _write_json(Path(args.pt_manifest).expanduser().resolve(), pt_manifest)

    (output_root / "README.md").write_text(
        """# Wang Xing Final 32+32 Test Set

This is a new evaluation-only set. It does not overwrite the old 25+25 set.

- 32 real Wang Xing videos;
- 32 AI videos from `data/test/xin_AI`;
- real videos are excluded from PT train, official holdout, and old 25+25;
- all videos have AU CSV files;
- this script only builds data and manifests; it does not train anything.

The five existing Change clips in `xin_AI` are marked with
`previous_test_overlap=true`; they are not training samples.
""",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "single_video_manifest": str(
                    output_root / "single_video" / "manifest.json"
                ),
                "profile_exclusion": str(
                    Path(args.profile_exclusion).expanduser().resolve()
                ),
                "pt_manifest": str(
                    Path(args.pt_manifest).expanduser().resolve()
                ),
                "counts": {
                    "real": int(args.count),
                    "ai": int(args.count),
                    "previous_test_ai": manifest["previous_test_ai_count"],
                },
                "training_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
