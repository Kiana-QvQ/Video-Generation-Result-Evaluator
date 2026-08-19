"""Build the web-forensics v2 test set without the old holdout-50 samples.

Outputs both:
- full set: 25 real + 30 AI (25 WangXing candidates + 5 Change);
- balanced view: 25 real + 25 AI (20 WangXing + 5 Change).

The selected WangXing videos come from the old training candidate pool, but
the new web profile/fusion pipeline must exclude them before fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from wangxing_project.joint_au_pt import is_augmented_video

REFERENCE_SOURCE = PROJECT_ROOT / "tests" / "test1"
REFERENCE_MOTION = (
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


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _uniform(items: list[Path], count: int) -> list[Path]:
    if len(items) < count:
        raise ValueError(f"Need {count} candidates, got {len(items)}.")
    if count == 1:
        return [items[0]]
    indexes = [
        round(index * (len(items) - 1) / (count - 1))
        for index in range(count)
    ]
    return [items[index] for index in indexes]


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _hash(source) == _hash(destination):
        return
    if destination.is_file():
        destination.unlink()
    try:
        destination.hardlink_to(source)
    except (OSError, NotImplementedError):
        shutil.copy2(source, destination)


def _reference_pack(destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("front.png", "Left.png", "Right.png", "prompt.txt"):
        source = REFERENCE_SOURCE / name
        if not source.is_file():
            raise FileNotFoundError(f"Reference file missing: {source}")
        _link_or_copy(source, destination / name)
    if REFERENCE_MOTION.is_file():
        _link_or_copy(REFERENCE_MOTION, destination / "reference_motion.mp4")
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


def _make_sample(
    *,
    split_root: Path,
    label: str,
    index: int,
    video: Path,
    au: Path,
    source_domain: str,
    overlaps_old_training: bool,
    reference: bool,
) -> dict[str, Any]:
    sample_id = f"{label}_{index:02d}"
    sample_root = split_root / label / sample_id
    sample_root.mkdir(parents=True, exist_ok=True)
    _link_or_copy(video, sample_root / "video.mp4")
    _link_or_copy(au, sample_root / "au.csv")
    payload: dict[str, Any] = {
        "sample_id": sample_id,
        "label": label,
        "label_generated": int(label == "ai"),
        "video": f"{label}/{sample_id}/video.mp4",
        "au": f"{label}/{sample_id}/au.csv",
        "source_video": _relative(video),
        "source_au": _relative(au),
        "source_domain": source_domain,
        "overlaps_old_training": bool(overlaps_old_training),
        "overlaps_official_holdout": False,
        "training_allowed": False,
    }
    if reference:
        payload["reference_root"] = f"{label}/{sample_id}/reference"
        payload["reference"] = _reference_pack(sample_root / "reference")
    _write_json(sample_root / "sample.json", payload)
    return payload


def _pair_au(
    video: Path,
    au_root: Path,
    video_root: Path | None = None,
) -> Path:
    if video_root is not None:
        relative = video.resolve().relative_to(video_root.resolve())
        au = (au_root / relative).with_suffix(".csv")
    else:
        au = au_root / f"{video.stem}.csv"
    if not au.is_file():
        raise FileNotFoundError(f"AU CSV missing for {video}: {au}")
    return au


def _write_manifest(
    *,
    split_root: Path,
    name: str,
    samples: list[dict[str, Any]],
) -> None:
    _write_json(
        split_root / name,
        {
            "schema_version": "web_forensics_v2_test_manifest_v1",
            "sample_count": len(samples),
            "real_count": sum(item["label"] == "real" for item in samples),
            "ai_count": sum(item["label"] == "ai" for item in samples),
            "training_allowed": False,
            "official_holdout_overlap": False,
            "samples": samples,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build web-forensics v2 full and balanced test manifests."
    )
    parser.add_argument(
        "--output-root",
        default="data/test/web_forensics_v2",
    )
    parser.add_argument(
        "--split-manifest",
        default="outputs/vedio_pred/wangxing_dual_pt_split_res1k.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    output_root = _resolve(args.output_root)
    manifest = _load_json(_resolve(args.split_manifest))
    train_real = [
        Path(value).resolve()
        for value in manifest["train"]["real"]
        if not is_augmented_video(value)
    ]
    train_fake = [
        Path(value).resolve()
        for value in manifest["train"]["fake"]
        if not is_augmented_video(value)
    ]
    used_real = {
        Path(value).resolve()
        for value in manifest["train"]["real"]
    } | {
        Path(value).resolve()
        for value in manifest["test"]["real"]
    }
    all_real = sorted((PROJECT_ROOT / "data" / "MD_CL").rglob("*.mp4"))
    unused_real = [
        path.resolve()
        for path in all_real
        if path.resolve() not in used_real
        and (
            PROJECT_ROOT
            / "data"
            / "au"
            / "MD_CL"
            / path.resolve().relative_to(
                (PROJECT_ROOT / "data" / "MD_CL").resolve()
            )
        ).with_suffix(".csv").is_file()
    ]
    selected_real = _uniform(sorted(unused_real), 25)
    selected_wangxing = _uniform(sorted(train_fake), 25)

    change_pairs = []
    for name in CHANGE_NAMES:
        video = PROJECT_ROOT / "data" / "test" / "AI" / name
        au = PROJECT_ROOT / "data" / "au" / "test" / "AI" / f"{Path(name).stem}.csv"
        if not video.is_file() or not au.is_file():
            raise FileNotFoundError(f"Change pair missing: {name}")
        change_pairs.append((video.resolve(), au.resolve()))

    pair_specs: list[tuple[str, Path, Path, str, bool]] = []
    for video in selected_real:
        pair_specs.append(
            (
                "real",
                video,
                _pair_au(
                    video,
                    PROJECT_ROOT / "data" / "au" / "MD_CL",
                    PROJECT_ROOT / "data" / "MD_CL",
                ),
                "MD_CL_unused_real",
                False,
            )
        )
    for video in selected_wangxing:
        pair_specs.append(
            (
                "ai",
                video,
                _pair_au(
                    video,
                    PROJECT_ROOT / "data" / "au" / "WangXing_Seedance",
                ),
                "WangXing_Seedance_train_candidate",
                True,
            )
        )
    for video, au in change_pairs:
        pair_specs.append(
            ("ai", video, au, "Change_OOD", False)
        )

    manifests: dict[str, list[dict[str, Any]]] = {}
    for split, reference in (
        ("single_video", False),
        ("with_reference", True),
    ):
        split_root = output_root / split
        samples: list[dict[str, Any]] = []
        real_index = 0
        ai_index = 0
        for label, video, au, domain, overlap in pair_specs:
            if label == "real":
                real_index += 1
                index = real_index
            else:
                ai_index += 1
                index = ai_index
            samples.append(
                _make_sample(
                    split_root=split_root,
                    label=label,
                    index=index,
                    video=video,
                    au=au,
                    source_domain=domain,
                    overlaps_old_training=overlap,
                    reference=reference,
                )
            )
        _write_manifest(
            split_root=split_root,
            name="manifest.json",
            samples=samples,
        )
        balanced = [
            item
            for item in samples
            if item["label"] == "real"
        ] + [
            item
            for item in samples
            if item["label"] == "ai"
            and (
                item["source_domain"] == "Change_OOD"
                or (
                    item["source_domain"]
                    == "WangXing_Seedance_train_candidate"
                    and int(item["sample_id"].split("_")[-1]) <= 20
                )
            )
        ]
        _write_manifest(
            split_root=split_root,
            name="manifest_25x25.json",
            samples=balanced,
        )
        manifests[split] = samples

    exclusion = {
        "schema_version": "web_forensics_v2_profile_exclusion_v1",
        "note": (
            "Exclude selected web-test sources and the old official holdout "
            "when rebuilding web profiles. Do not delete old profiles."
        ),
        "real": [
            {
                "video": _relative(video),
                "au": _relative(
                    _pair_au(
                        video,
                        PROJECT_ROOT / "data" / "au" / "MD_CL",
                        PROJECT_ROOT / "data" / "MD_CL",
                    )
                ),
            }
            for video in selected_real
        ],
        "seedance": [
            {
                "video": _relative(video),
                "au": _relative(
                    _pair_au(
                        video,
                        PROJECT_ROOT / "data" / "au" / "WangXing_Seedance",
                    )
                ),
            }
            for video in selected_wangxing
        ],
    }
    old_holdout = _load_json(
        PROJECT_ROOT / "data" / "forensics" / "holdout_split.json"
    )
    exclusion["real"].extend(old_holdout.get("real", []))
    exclusion["seedance"].extend(old_holdout.get("seedance", []))
    _write_json(
        PROJECT_ROOT
        / "data"
        / "forensics"
        / "web_forensics_v2_profile_exclusion.json",
        exclusion,
    )
    _write_json(
        output_root / "dataset_summary.json",
        {
            "full_real": 25,
            "full_ai": 30,
            "balanced_real": 25,
            "balanced_ai": 25,
            "wangxing_ai": 25,
            "change_ai": 5,
            "old_holdout_used_in_test": False,
            "old_training_overlap_ai": 25,
            "training_allowed": False,
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "full": {"real": 25, "ai": 30},
                "balanced": {"real": 25, "ai": 25},
                "profile_exclusion": (
                    "data/forensics/"
                    "web_forensics_v2_profile_exclusion.json"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
