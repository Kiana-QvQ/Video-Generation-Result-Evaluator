"""Build a real-manifold manifest from the isolated XiaoYue experiment.

The six AI clips remain supervised generated references. All accepted public
real clips are used only to estimate the real face distribution. The 1+1
test pair remains evaluation-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(value: str | Path) -> Path:
    target = Path(value).expanduser()
    return (target if target.is_absolute() else PROJECT_ROOT / target).resolve()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _real_bank(
    video_root: Path,
    au_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for video in sorted(video_root.rglob("*.mp4"), key=lambda item: item.as_posix().casefold()):
        relative = video.relative_to(video_root)
        au = au_root / relative.with_suffix(".csv")
        if not au.is_file():
            continue
        records.append(
            {
                "video": _relative(video),
                "au": _relative(au),
                "label_generated": 0,
                "sample_id": f"real_bank_{len(records):04d}",
                "group_id": f"real_bank_{relative.parent.as_posix()}",
                "source_kind": "accepted_public_real_bank",
            }
        )
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-manifest",
        default="data/xiaoyue/experiment_7x7/manifests/pt_manifest.json",
    )
    parser.add_argument(
        "--real-root",
        default="data/xiaoyue/processed/real_candidates",
    )
    parser.add_argument("--real-au-root", default="data/au/xiaoyue/real")
    parser.add_argument(
        "--output",
        default="data/xiaoyue/experiment_7x7/manifests/face_manifold_manifest.json",
    )
    args = parser.parse_args(argv)

    base_path = _path(args.base_manifest)
    base = _load(base_path)
    pairs = base.get("pairs") or {}
    train = pairs.get("train") or {}
    test = pairs.get("test") or {}
    train_ai = list(train.get("fake") or [])
    test_real = list(test.get("real") or [])
    test_ai = list(test.get("fake") or [])
    if len(train_ai) != 6 or len(test_ai) != 1 or len(test_real) != 1:
        raise SystemExit(
            "Expected base XiaoYue experiment counts of train AI=6, "
            "test real=1, test AI=1."
        )
    real_bank = _real_bank(_path(args.real_root), _path(args.real_au_root))
    if len(real_bank) < 20:
        raise SystemExit(f"Too few real-manifold videos with AU: {len(real_bank)}")
    train_hashes = {
        str(item.get("sha256") or "")
        for item in [*list(train.get("real") or []), *train_ai]
        if item.get("sha256")
    }
    test_hashes = {
        str(item.get("sha256") or "")
        for item in [*test_real, *test_ai]
        if item.get("sha256")
    }
    if train_hashes & test_hashes:
        raise SystemExit("Base train/test SHA-256 overlap detected.")

    payload = {
        "schema_version": "xiaoyue_face_manifold_manifest_v1",
        "subject": "xiaoyue",
        "training_allowed": True,
        "pairs": {
            "train": {
                "real": list(train.get("real") or []),
                "fake": train_ai,
            },
            "test": {
                "real": test_real,
                "fake": test_ai,
            },
        },
        "real_manifold_bank": real_bank,
        "counts": {
            "train_real_core": len(train.get("real") or []),
            "train_ai": len(train_ai),
            "real_manifold_bank": len(real_bank),
            "test_real": len(test_real),
            "test_ai": len(test_ai),
        },
        "feature_policy": {
            "full_frame_rgb_used": False,
            "full_frame_hsv_used": False,
            "background_used": False,
            "absolute_brightness_used": False,
            "mouth_priority": True,
        },
        "test_training_allowed": False,
        "notes": [
            "The real manifold bank is used only to estimate the real face distribution.",
            "The six AI clips are generated-domain training references.",
            "The test pair remains excluded from manifold fitting and threshold fitting.",
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
                "test_training_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
