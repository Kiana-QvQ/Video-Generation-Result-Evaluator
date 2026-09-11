"""Build ZiJie V5 manifests with a checkpoint-disjoint LTX development set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LTX_ROOT = (
    PROJECT_ROOT.parent
    / "MiniMax_H3"
    / "artifacts"
    / "zijie_ltx25_v4_1500_compare"
    / "ai_generated_videos_for_training_20260910"
)
DEFAULT_METRICS = (
    DEFAULT_LTX_ROOT.parent
    / "final_checkpoint_review"
    / "gt_similarity_metrics.tsv"
)
DEFAULT_GT_VIDEO = (
    PROJECT_ROOT.parent / "MiniMax_H3" / "work" / "zijie_hk_gate_v3_review" / "gt_real.mp4"
)
STEP_PATTERN = re.compile(r"step_(\d+)_1$", re.IGNORECASE)
DEFAULT_HOLDOUT_STEPS = {100, 400, 700, 1100}


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _quality_metrics(path: Path) -> dict[int, dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    parsed = {
        int(row["step"]): {
            "ssim": float(row["ssim"]),
            "psnr": float(row["psnr"]),
        }
        for row in rows
    }
    if not parsed:
        raise ValueError(f"No quality metrics found: {path}")
    low = min(value["ssim"] for value in parsed.values())
    high = max(value["ssim"] for value in parsed.values())
    for value in parsed.values():
        # Keep AI scores below the real band while making checkpoint quality
        # differences visible. This target is supervision, never a filename-
        # based runtime rule.
        value["quality_target_0_100"] = 25.0 + 55.0 * (
            (value["ssim"] - low) / max(high - low, 1e-8)
        )
    return parsed


def _ltx_records(
    root: Path,
    au_root: Path,
    metrics: dict[int, dict[str, float]],
) -> list[dict]:
    records: list[dict] = []
    for video in sorted(root.glob("*.mp4")):
        match = STEP_PATTERN.fullmatch(video.stem)
        if not match:
            continue
        step = int(match.group(1))
        quality = metrics.get(step)
        if step == 0:
            quality = {
                "ssim": 0.0,
                "psnr": 0.0,
                "quality_target_0_100": 5.0,
            }
        if quality is None:
            raise ValueError(f"Missing GT quality metric for step {step}.")
        records.append(
            {
                "video": str(video.resolve()),
                "au": str((au_root / f"{video.stem}.csv").resolve()),
                "label_generated": 1,
                "sample_id": f"ai_ltx25_step_{step:06d}",
                "group_id": "ltx25_checkpoint_sweep_20260910",
                "source_domain": "ltx25_v4_checkpoint_sweep",
                "checkpoint_step": step,
                "quality_target_0_100": quality["quality_target_0_100"],
                "quality_reference": {
                    "source": "gt_similarity_metrics.tsv",
                    "ssim": quality["ssim"],
                    "psnr": quality["psnr"],
                },
                "sha256": _sha256(video),
            }
        )
    if len(records) != 16:
        raise ValueError(f"Expected 16 LTX videos, found {len(records)}.")
    if len({record["sha256"] for record in records}) != len(records):
        raise ValueError("Duplicate LTX video content detected.")
    return records


def _require_files(records: list[dict]) -> None:
    missing = [
        str(record["au"])
        for record in records
        if not Path(str(record["au"])).is_file()
    ]
    if missing:
        raise ValueError(
            f"AU extraction is incomplete ({len(missing)} missing): "
            + ", ".join(missing[:4])
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-manifest",
        default="data/zijie/manifests/zijie_face_manifold_v1.json",
    )
    parser.add_argument("--ltx-root", default=str(DEFAULT_LTX_ROOT))
    parser.add_argument(
        "--ltx-au-root",
        default="data/au/zijie/ltx25_v4_1500_20260910",
    )
    parser.add_argument("--quality-metrics", default=str(DEFAULT_METRICS))
    parser.add_argument("--gt-video", default=str(DEFAULT_GT_VIDEO))
    parser.add_argument(
        "--gt-au",
        default="data/au/zijie/ltx25_v4_1500_gt/gt_real.csv",
    )
    parser.add_argument(
        "--output",
        default="data/zijie/manifests/zijie_face_temporal_v5.json",
    )
    parser.add_argument("--require-au", action="store_true")
    args = parser.parse_args()

    base = _load(_path(args.base_manifest))
    ltx = _ltx_records(
        _path(args.ltx_root),
        _path(args.ltx_au_root),
        _quality_metrics(_path(args.quality_metrics)),
    )
    if args.require_au:
        _require_files(ltx)
    gt_video = _path(args.gt_video)
    gt_au = _path(args.gt_au)
    if not gt_video.is_file() or (args.require_au and not gt_au.is_file()):
        raise ValueError("The LTX ranking GT video or AU file is missing.")
    ranking_fit = [
        item for item in ltx
        if int(item["checkpoint_step"]) not in DEFAULT_HOLDOUT_STEPS
    ]
    ranking_holdout = [
        item for item in ltx
        if int(item["checkpoint_step"]) in DEFAULT_HOLDOUT_STEPS
    ]
    base_train = (base.get("pairs") or {}).get("train") or {}
    base_test = (base.get("pairs") or {}).get("test") or {}
    train_real = list(base_train.get("real") or [])
    train_ai = [*list(base_train.get("fake") or []), *ranking_fit]
    final_real = list(base_test.get("real") or [])
    final_ai = list(base_test.get("fake") or [])
    train_hashes = {str(item["sha256"]) for item in [*train_real, *train_ai]}
    final_hashes = {str(item["sha256"]) for item in [*final_real, *final_ai]}
    ltx_holdout_hashes = {str(item["sha256"]) for item in ranking_holdout}
    if train_hashes & final_hashes or train_hashes & ltx_holdout_hashes:
        raise ValueError("V5 train/evaluation SHA-256 overlap detected.")

    payload = {
        "schema_version": "zijie_face_temporal_v5_manifest",
        "subject": "zijie",
        "training_allowed": True,
        "pairs": {
            "train": {"real": train_real, "fake": train_ai},
            "test": {"real": final_real, "fake": final_ai},
        },
        "evaluation_sets": {
            "fixed_h3_final": {"real": final_real, "fake": final_ai},
            "ltx_checkpoint_holdout": {"real": [], "fake": ranking_holdout},
        },
        "ranking": {
            "fit": ranking_fit,
            "holdout": ranking_holdout,
            "reference": {
                "sample_id": "real_ltx25_ranking_gt",
                "video": str(gt_video),
                "au": str(gt_au),
                "label_generated": 0,
                "sha256": _sha256(gt_video),
            },
            "label": "GT similarity calibrated to an AI-only display band",
            "runtime_uses_checkpoint_step": False,
            "development_only": True,
        },
        "counts": {
            "train_real": len(train_real),
            "train_ai": len(train_ai),
            "final_real": len(final_real),
            "final_h3_ai": len(final_ai),
            "ltx_ranking_fit": len(ranking_fit),
            "ltx_ranking_holdout": len(ranking_holdout),
            "ltx_unique": len(ltx),
        },
        "split_policy": {
            "ltx_holdout_steps": sorted(DEFAULT_HOLDOUT_STEPS),
            "fixed_h3_holdout_preserved": True,
            "sha256_overlap": False,
            "same_prompt_warning": True,
        },
        "test_training_allowed": False,
    }
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **payload["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
