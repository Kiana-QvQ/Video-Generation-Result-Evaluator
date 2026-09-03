"""Build an isolated XiaoYue 6-train+1-test AI/real experiment.

The experiment intentionally does not modify the existing 104-real/4-AI
dataset. It copies six local AI training clips, one test AI clip, six
accepted public-disk real clips, and one paired real test reference into an
ASCII-path experiment directory. Existing AU CSVs are copied when available;
missing AI AU files are produced by the pipeline's extraction stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(value: str | Path) -> Path:
    target = Path(value).expanduser()
    return (target if target.is_absolute() else PROJECT_ROOT / target).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _copy(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"Missing source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        not destination.is_file()
        or destination.stat().st_size != source.stat().st_size
    ):
        shutil.copy2(source, destination)
    return destination


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _select_real_videos(real_root: Path, count: int) -> list[Path]:
    candidates = sorted(
        real_root.rglob("*.mp4"),
        key=lambda item: item.as_posix().casefold(),
    )
    if len(candidates) < count:
        raise ValueError(
            f"Need {count} accepted real videos under {real_root}, "
            f"found {len(candidates)}."
        )

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in candidates:
        relative = path.relative_to(real_root)
        group = (
            "/".join(relative.parts[:2])
            if len(relative.parts) >= 2
            else relative.parent.as_posix()
        )
        groups[group].append(path)

    selected: list[Path] = []
    for group in sorted(groups):
        if len(selected) >= count:
            break
        selected.append(groups[group][0])

    if len(selected) < count:
        used = set(selected)
        for path in candidates:
            if path not in used:
                selected.append(path)
                if len(selected) >= count:
                    break
    return selected


def _source_quality_index(
    path: Path,
    quality_manifest: Path | None,
) -> dict[str, Any]:
    if quality_manifest is None or not quality_manifest.is_file():
        return {}
    payload = _load_json(quality_manifest)
    target = str(path.resolve()).casefold()
    for row in payload.get("videos") or []:
        if str(Path(str(row.get("video") or "")).resolve()).casefold() == target:
            return {
                "screening_decision": row.get("decision"),
                "sample_keep_ratio": row.get("sample_keep_ratio"),
                "screening_reason_counts": row.get("reason_counts") or {},
            }
    return {}


def _train_item(
    *,
    source_video: Path,
    target_video: Path,
    target_au: Path,
    label: int,
    sample_id: str,
    group_id: str,
    source_kind: str,
    quality_manifest: Path | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "video": _project_relative(target_video),
        "au": _project_relative(target_au),
        "label_generated": int(label),
        "base_label": int(label),
        "sample_id": sample_id,
        "group_id": group_id,
        "source_kind": source_kind,
        "source_video": _project_relative(source_video),
        "sha256": _sha256(target_video),
    }
    record.update(_source_quality_index(source_video, quality_manifest))
    return record


def _test_item(
    *,
    source_video: Path,
    target_video: Path,
    target_au: Path,
    label: int,
    sample_id: str,
    group_id: str,
) -> dict[str, Any]:
    return {
        "video": _project_relative(target_video),
        "au": _project_relative(target_au),
        "label_generated": int(label),
        "base_label": int(label),
        "sample_id": sample_id,
        "group_id": group_id,
        "source_kind": "test_reference" if not label else "test_generated",
        "source_video": _project_relative(source_video),
        "sha256": _sha256(target_video),
    }


def _hashes(items: Iterable[dict[str, Any]]) -> set[str]:
    return {str(item["sha256"]) for item in items}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an isolated XiaoYue 6+1 AI / 6+1 real experiment."
    )
    parser.add_argument("--ai-root", default="data/xiaoyue/AI")
    parser.add_argument("--real-root", default="data/xiaoyue/real")
    parser.add_argument(
        "--ai-test-video",
        default="data/xiaoyue/test/test1/seedance2.5.mp4",
    )
    parser.add_argument(
        "--real-test-video",
        default="data/xiaoyue/test/test1/GT.mp4",
    )
    parser.add_argument("--real-au-root", default="data/au/xiaoyue/real")
    parser.add_argument("--test-au-root", default="data/au/xiaoyue/test")
    parser.add_argument(
        "--quality-manifest",
        default="outputs/xiaoyue_screening/source_quality_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        default="data/xiaoyue/experiment_7x7",
    )
    parser.add_argument("--real-train-count", type=int, default=6)
    parser.add_argument("--ai-train-count", type=int, default=6)
    parser.add_argument(
        "--require-train-au",
        action="store_true",
        help="Fail unless every training video already has an AU CSV.",
    )
    args = parser.parse_args(argv)

    if args.real_train_count <= 0 or args.ai_train_count <= 0:
        raise SystemExit("Train counts must be positive.")

    ai_root = _path(args.ai_root)
    real_root = _path(args.real_root)
    ai_videos = sorted(
        ai_root.glob("*.mp4"),
        key=lambda item: item.name.casefold(),
    )
    if len(ai_videos) != args.ai_train_count:
        raise SystemExit(
            f"Expected exactly {args.ai_train_count} AI training videos in "
            f"{ai_root}, found {len(ai_videos)}."
        )
    real_videos = _select_real_videos(real_root, args.real_train_count)
    ai_test = _path(args.ai_test_video)
    real_test = _path(args.real_test_video)
    if not ai_test.is_file() or not real_test.is_file():
        raise SystemExit("The selected XiaoYue test AI/real pair is incomplete.")

    real_au_root = _path(args.real_au_root)
    test_au_root = _path(args.test_au_root)
    quality_manifest = (
        _path(args.quality_manifest)
        if args.quality_manifest
        else None
    )
    output_root = _path(args.output_root)
    train_real_root = output_root / "train" / "real"
    train_ai_root = output_root / "train" / "ai"
    test_real_root = output_root / "test" / "real"
    test_ai_root = output_root / "test" / "ai"
    train_real_au_root = output_root / "au" / "train" / "real"
    train_ai_au_root = output_root / "au" / "train" / "ai"
    test_real_au_root = output_root / "au" / "test" / "real"
    test_ai_au_root = output_root / "au" / "test" / "ai"

    train_real: list[dict[str, Any]] = []
    for index, source in enumerate(real_videos, start=1):
        target = train_real_root / f"real_{index:02d}.mp4"
        relative = source.relative_to(real_root)
        source_au = real_au_root / relative.with_suffix(".csv")
        target_au = train_real_au_root / relative.with_suffix(".csv")
        _copy(source, target)
        if source_au.is_file():
            _copy(source_au, target_au)
        train_real.append(
            _train_item(
                source_video=source,
                target_video=target,
                target_au=target_au,
                label=0,
                sample_id=f"real_train_{index:02d}",
                group_id=f"real_train_group_{index:02d}",
                source_kind="accepted_public_real",
                quality_manifest=quality_manifest,
            )
        )

    train_ai: list[dict[str, Any]] = []
    old_ai_au_root = _path("data/au/xiaoyue/generated")
    for index, source in enumerate(ai_videos, start=1):
        target = train_ai_root / f"ai_{index:02d}.mp4"
        target_au = train_ai_au_root / f"ai_{index:02d}.csv"
        _copy(source, target)
        old_au = old_ai_au_root / f"ai_{index:02d}.csv"
        if old_au.is_file():
            _copy(old_au, target_au)
        train_ai.append(
            _train_item(
                source_video=source,
                target_video=target,
                target_au=target_au,
                label=1,
                sample_id=f"ai_train_{index:02d}",
                group_id=f"ai_train_{source.stem}",
                source_kind="local_xiaoyue_ai",
                quality_manifest=None,
            )
        )

    real_target = test_real_root / "test_real_01.mp4"
    ai_target = test_ai_root / "test_ai_01.mp4"
    real_target_au = test_real_au_root / "test_real_01.csv"
    ai_target_au = test_ai_au_root / "test_ai_01.csv"
    _copy(real_test, real_target)
    _copy(ai_test, ai_target)
    test_real_au = test_au_root / "test1_real_reference.csv"
    test_ai_au = test_au_root / "test1_seedance25.csv"
    if test_real_au.is_file():
        _copy(test_real_au, real_target_au)
    if test_ai_au.is_file():
        _copy(test_ai_au, ai_target_au)
    test_real = _test_item(
        source_video=real_test,
        target_video=real_target,
        target_au=real_target_au,
        label=0,
        sample_id="real_test_01",
        group_id="xiaoyue_test1_pair",
    )
    test_ai = _test_item(
        source_video=ai_test,
        target_video=ai_target,
        target_au=ai_target_au,
        label=1,
        sample_id="ai_test_01",
        group_id="xiaoyue_test1_pair",
    )

    train = [*train_real, *train_ai]
    test = [test_real, test_ai]
    if _hashes(train) & _hashes(test):
        raise SystemExit("Train/test SHA-256 overlap detected.")
    missing_train_au = [
        item["au"]
        for item in train
        if not _path(item["au"]).is_file()
    ]
    missing_test_au = [
        item["au"]
        for item in test
        if not _path(item["au"]).is_file()
    ]
    if missing_test_au:
        raise SystemExit(
            "The paired test AU files are missing; run test AU extraction "
            "first: " + ", ".join(missing_test_au)
        )
    if args.require_train_au and missing_train_au:
        raise SystemExit(
            "Training AU extraction is incomplete: "
            + ", ".join(missing_train_au)
        )

    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    pt_manifest = {
        "schema_version": "xiaoyue_temporal_au_video_7x7_manifest_v1",
        "subject": "xiaoyue",
        "training_allowed": True,
        "experiment": (
            "six AI train + one AI test paired with six real train + one real test"
        ),
        "pairs": {
            "train": {"real": train_real, "fake": train_ai},
            "test": {"real": [test_real], "fake": [test_ai]},
        },
        "counts": {
            "train_real": len(train_real),
            "train_fake": len(train_ai),
            "test_real": 1,
            "test_fake": 1,
        },
        "quality_notes": {
            "ai_training_inclusion": (
                "All six files under data/xiaoyue/AI are included by explicit "
                "experiment request, including previously reviewed low-coverage candidates."
            ),
            "real_selection": (
                "Six accepted public-disk real clips selected deterministically "
                "across source groups."
            ),
            "test_pair": (
                "test1/GT.mp4 and test1/seedance2.5.mp4 are evaluation-only."
            ),
        },
        "excluded_test_paths": [
            "data/xiaoyue/test",
            "data/xiaoyue/processed/test_reference",
        ],
        "missing_train_au_before_extraction": missing_train_au,
    }
    test_manifest = {
        "schema_version": "xiaoyue_temporal_au_video_7x7_test_manifest_v1",
        "subject": "xiaoyue",
        "training_allowed": False,
        "real": [test_real],
        "fake": [test_ai],
        "seedance": [test_ai],
        "counts": {"real": 1, "fake": 1},
        "train_test_sha256_overlap": False,
        "test_pair": "test1/GT.mp4 vs test1/seedance2.5.mp4",
    }
    specialization_manifest = {
        "schema_version": "xiaoyue_specialization_7x7_manifest_v1",
        "subject": "xiaoyue",
        "training_allowed": True,
        "real": train_real,
        "generated": train_ai,
        "counts": {"real": len(train_real), "generated": len(train_ai)},
        "test_paths_forbidden": [
            "data/xiaoyue/experiment_7x7/test",
            "data/xiaoyue/test",
        ],
    }
    web_test_manifest = {
        "schema_version": "xiaoyue_web_7x7_test_manifest_v1",
        "subject": "xiaoyue",
        "training_allowed": False,
        "real": [test_real],
        "fake": [test_ai],
        "counts": {"real": 1, "fake": 1},
    }
    paths = {
        "pt_manifest": manifest_root / "pt_manifest.json",
        "pt_test_manifest": manifest_root / "pt_test_manifest.json",
        "specialization_manifest": manifest_root / "specialization_manifest.json",
        "web_test_manifest": manifest_root / "web_test_manifest.json",
        "selection_report": output_root / "selection_report.json",
    }
    for path, payload in (
        (paths["pt_manifest"], pt_manifest),
        (paths["pt_test_manifest"], test_manifest),
        (paths["specialization_manifest"], specialization_manifest),
        (paths["web_test_manifest"], web_test_manifest),
    ):
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    paths["selection_report"].write_text(
        json.dumps(
            {
                "schema_version": "xiaoyue_7x7_selection_report_v1",
                "subject": "xiaoyue",
                "counts": pt_manifest["counts"],
                "train_real": [
                    {
                        "sample_id": item["sample_id"],
                        "source_video": item["source_video"],
                        "video": item["video"],
                    }
                    for item in train_real
                ],
                "train_ai": [
                    {
                        "sample_id": item["sample_id"],
                        "source_video": item["source_video"],
                        "video": item["video"],
                    }
                    for item in train_ai
                ],
                "test_pair": {
                    "real": test_real["source_video"],
                    "ai": test_ai["source_video"],
                },
                "missing_train_au_before_extraction": missing_train_au,
                "test_training_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_root": _project_relative(output_root),
                "counts": pt_manifest["counts"],
                "missing_train_au_before_extraction": missing_train_au,
                "manifests": {
                    name: _project_relative(path)
                    for name, path in paths.items()
                    if name != "selection_report"
                },
                "selection_report": _project_relative(paths["selection_report"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
