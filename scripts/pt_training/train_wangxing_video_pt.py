"""Build split + train Wang Xing dual-scale video .pt detector.

Protocol:
- Train real: evenly sample 120 clips from MD_CL (holdout excluded)
- Train fake: all non-holdout Seedance videos
- Test: forensics holdout videos (50 real + 50 generated)
- Features: concat(24 frames @ 1024, 8 frames @ 2048), train-only normalize
- Output: one .pt classifier

You run the commands locally (feature extract at 1k/2k is slow).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from evaluator.vedio_pred.wangxing_dual_pt import (
    build_wangxing_split_manifest,
    predict_wangxing_dual_pt,
    train_wangxing_dual_pt,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def cmd_build_split(args: argparse.Namespace) -> int:
    manifest = build_wangxing_split_manifest(
        project_root=PROJECT_ROOT,
        real_root=project_path(args.real_root),
        fake_root=project_path(args.fake_root),
        holdout_manifest=project_path(args.holdout_manifest),
        real_train_count=args.real_train_count,
        seed=args.seed,
    )
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    manifest_path = project_path(args.manifest)
    if not manifest_path.is_file():
        # Auto-build if missing.
        print(f"Manifest missing, building: {manifest_path}")
        try:
            rel_output = str(manifest_path.relative_to(PROJECT_ROOT))
        except ValueError:
            rel_output = str(manifest_path)
        build_args = argparse.Namespace(
            real_root=args.real_root,
            fake_root=args.fake_root,
            holdout_manifest=args.holdout_manifest,
            real_train_count=args.real_train_count,
            seed=args.seed,
            output=rel_output,
        )
        cmd_build_split(build_args)
    manifest = _load_json(manifest_path)
    result = train_wangxing_dual_pt(
        manifest=manifest,
        cache_dir=project_path(args.cache_dir),
        model_path=project_path(args.model_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    metrics_path = project_path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "wangxing_dual_pt_metrics_v1",
        "manifest": str(manifest_path),
        "model_path": result["model_path"],
        "headline": result["headline"],
        "confusion": result["confusion"],
        "counts": result["counts"],
        "temperature": result["temperature"],
        "note": (
            "Dual-scale video .pt (24f@1024 + 8f@2048). "
            "Test set = forensics holdout. Train reals = evenly sampled 120."
        ),
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["headline"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["confusion"], ensure_ascii=False, indent=2))
    print(f"Model: {result['model_path']}")
    print(f"Metrics: {metrics_path}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    result = predict_wangxing_dual_pt(
        video_path=project_path(args.video),
        model_path=project_path(args.model_path),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wang Xing dual-scale video .pt train/eval pipeline."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    split = sub.add_parser(
        "build-split",
        help="Build train/test manifest (120 real train + holdout test).",
    )
    split.add_argument("--real-root", default="data/MD_CL")
    split.add_argument("--fake-root", default="data/WangXing_Seedance")
    split.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split.json",
    )
    split.add_argument("--real-train-count", type=int, default=120)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument(
        "--output",
        default="outputs/vedio_pred/wangxing_dual_pt_split.json",
    )
    split.set_defaults(func=cmd_build_split)

    train = sub.add_parser(
        "train",
        help="Extract dual-scale features, normalize, train .pt, eval holdout.",
    )
    train.add_argument("--real-root", default="data/MD_CL")
    train.add_argument("--fake-root", default="data/WangXing_Seedance")
    train.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split.json",
    )
    train.add_argument("--real-train-count", type=int, default=120)
    train.add_argument(
        "--manifest",
        default="outputs/vedio_pred/wangxing_dual_pt_split.json",
    )
    train.add_argument(
        "--cache-dir",
        default="outputs/vedio_pred/cache",
    )
    train.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_dual_scale_classifier.pt",
    )
    train.add_argument(
        "--metrics-output",
        default="outputs/vedio_pred/wangxing_dual_pt_holdout_metrics.json",
    )
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict", help="Predict one video with trained .pt.")
    predict.add_argument("--video", required=True)
    predict.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_dual_scale_classifier.pt",
    )
    predict.set_defaults(func=cmd_predict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
