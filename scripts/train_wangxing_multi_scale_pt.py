"""Train one Wang Xing .pt from mixed 8/24/32 x 1k/2k features.

Reuses single-scale feature caches from the ablation run when present.
Same holdout split as dual-scale training.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from evaluator.vedio_pred.wangxing_dual_pt import build_wangxing_split_manifest
from wangxing_project.multi_scale_pt import (
    DEFAULT_SCALES,
    predict_wangxing_multi_scale_pt,
    train_wangxing_multi_scale_pt,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * float(value):.1f}%"


def cmd_train(args: argparse.Namespace) -> int:
    manifest_path = project_path(args.manifest)
    if not manifest_path.is_file():
        print(f"Manifest missing, building: {manifest_path}", flush=True)
        manifest = build_wangxing_split_manifest(
            project_root=PROJECT_ROOT,
            real_root=project_path(args.real_root),
            fake_root=project_path(args.fake_root),
            holdout_manifest=project_path(args.holdout_manifest),
            real_train_count=args.real_train_count,
            seed=args.seed,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        manifest = _load_json(manifest_path)

    print("Split counts:", json.dumps(manifest.get("counts"), ensure_ascii=False))
    print(
        "Scales:",
        ", ".join(
            f"{s['num_frames']}f@{s['frame_size']}" for s in DEFAULT_SCALES
        ),
        flush=True,
    )

    result = train_wangxing_multi_scale_pt(
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
        "schema_version": "wangxing_multi_scale_pt_metrics_v1",
        "manifest": str(manifest_path),
        "model_path": result["model_path"],
        "cache_path": result["cache_path"],
        "scales": result["scales"],
        "input_dim": result["input_dim"],
        "headline": result["headline"],
        "confusion": result["confusion"],
        "counts": result["counts"],
        "temperature": result["temperature"],
        "note": (
            "Multi-scale video .pt: concat(8/24/32 @1024 + 8/24/32 @2048). "
            "Test = forensics holdout. Train reals = evenly sampled 120."
        ),
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report_md = project_path(args.report_md)
    report_md.write_text(
        "\n".join(
            [
                "# 视频 `.pt` 六路混合（8/24/32 × 1k/2k）",
                "",
                "## 协议",
                "",
                "- 训练：非 holdout 真拍均匀 120 + 非 holdout Seedance",
                "- 测试：forensics holdout 50 真 + 50 生成",
                "- 特征：六路单尺度特征 concat 后训一个 MLP → 一个 `.pt`",
                "",
                "## Holdout 指标",
                "",
                f"- 生成召回：{_pct(payload['headline']['generated_recall'])}",
                f"- 整体准确率：{_pct(payload['headline']['overall_accuracy'])}",
                f"- 真拍召回：{_pct(payload['headline'].get('real_recall'))}",
                f"- 生成精确率：{_pct(payload['headline'].get('generated_precision'))}",
                "",
                f"- 模型：`{payload['model_path']}`",
                f"- 指标 JSON：`{metrics_path}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("\n======== 六路混合 .pt Holdout ========")
    print(f"生成召回:   {_pct(payload['headline']['generated_recall'])}")
    print(f"整体准确率: {_pct(payload['headline']['overall_accuracy'])}")
    print(json.dumps(payload["confusion"], ensure_ascii=False, indent=2))
    print(f"Model:   {result['model_path']}")
    print(f"Metrics: {metrics_path}")
    print(f"Report:  {report_md}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    result = predict_wangxing_multi_scale_pt(
        video_path=project_path(args.video),
        model_path=project_path(args.model_path),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wang Xing multi-scale (8/24/32 x 1k/2k) video .pt."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser(
        "train",
        help="Concat six scales, train one .pt, eval holdout.",
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
    train.add_argument("--cache-dir", default="outputs/vedio_pred/cache")
    train.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_multi_scale_classifier.pt",
    )
    train.add_argument(
        "--metrics-output",
        default="outputs/vedio_pred/wangxing_multi_scale_holdout_metrics.json",
    )
    train.add_argument(
        "--report-md",
        default="outputs/vedio_pred/wangxing_multi_scale_holdout_metrics.md",
    )
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict", help="Predict one video with multi-scale .pt.")
    predict.add_argument("--video", required=True)
    predict.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_multi_scale_classifier.pt",
    )
    predict.set_defaults(func=cmd_predict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
