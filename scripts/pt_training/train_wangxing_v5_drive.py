"""Train the V5 auxiliary Wang Xing DriveHead without touching V3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wangxing_project.drive_head_v5 import train_drive_head


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the V5 auxiliary DriveHead; frozen V3 is untouched."
    )
    parser.add_argument("command", choices=("train",))
    parser.add_argument(
        "--manifest",
        default="outputs/vedio_pred/wangxing_v3_generalization_manifest_res1k.json",
    )
    parser.add_argument(
        "--cache-dir",
        default="outputs/vedio_pred/cache_wangxing_v5_drive_res1k",
    )
    parser.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_v5_drive.json",
    )
    parser.add_argument(
        "--transition-cache",
        default="outputs/vedio_pred/cache_wangxing_v4_expression_res1k/wangxing_v4_transition.npz",
    )
    parser.add_argument(
        "--blendshape-cache",
        default="outputs/vedio_pred/cache_wangxing_v4_expression_res1k/wangxing_v4_blendshape.npz",
    )
    parser.add_argument(
        "--include-blendshape",
        action="store_true",
        help="Use cached/computed Blendshape features; default is transition-only.",
    )
    parser.add_argument("--compute-blendshape", action="store_true")
    parser.add_argument(
        "--metrics-output",
        default="outputs/vedio_pred/wangxing_v5_drive_metrics_res1k.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    payload = train_drive_head(
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        model_path=args.model_path,
        transition_cache=args.transition_cache,
        blendshape_cache=args.blendshape_cache,
        include_blendshape=args.include_blendshape,
        compute_blendshape=args.compute_blendshape,
        seed=args.seed,
    )
    metrics = {
        "schema_version": "wangxing_v5_drive_metrics_v1",
        "architecture": (
            "grouped logistic DriveHead over AU/landmark transition "
            "features plus optional cached MediaPipe Blendshape descriptors"
        ),
        "decision_model_unchanged": True,
        "frozen_v3_required": True,
        **payload,
    }
    output = Path(args.metrics_output).expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics["validation_metrics"], ensure_ascii=False, indent=2))
    print(f"DriveHead: {Path(args.model_path).expanduser().resolve()}")
    print(f"Metrics: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
