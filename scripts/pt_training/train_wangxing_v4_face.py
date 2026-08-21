"""Train/evaluate the optional face-geometry-aware PT v4 model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from wangxing_project.joint_au_pt_v4 import (
    evaluate_holdout_v4,
    predict_wangxing_v4,
    train_wangxing_v4,
)


def _load(path: str) -> dict[str, Any]:
    return json.loads(project_path(path).read_text(encoding="utf-8-sig"))


def _profiles(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    return _load(args.source_profile), _load(args.forensics_profile)


def cmd_train(args: argparse.Namespace) -> int:
    source, profiles = _profiles(args)
    manifest_path = project_path(args.manifest)
    result = train_wangxing_v4(
        manifest=_load(args.manifest),
        cache_dir=project_path(args.cache_dir),
        model_path=project_path(args.model_path),
        source_profile=source,
        forensics_profiles=profiles,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        modality_dropout=args.modality_dropout,
    )
    payload = {
        "schema_version": "wangxing_expression_authenticity_v4_metrics_v1",
        "manifest": str(manifest_path),
        "model_path": result["model_path"],
        "headline": result["headline"],
        "confusion": result["confusion"],
        "counts": result["counts"],
        "temperature": result["temperature"],
        "device": result["device"],
        "architecture": (
            "expression-only primary: facial-motion AU subset, face geometry, "
            "transition windows, and MediaPipe 52-blendshape temporal branch; "
            "full-frame video is auxiliary"
        ),
        "note": (
            "Parallel v4 model. It does not overwrite v3 or v2. "
            "Training is intentionally not run by the coding agent."
        ),
    }
    output = project_path(args.metrics_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["headline"], ensure_ascii=False, indent=2))
    print(f"Model: {result['model_path']}")
    print(f"Metrics: {output}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    source, profiles = _profiles(args)
    output = project_path(args.output)
    payload = evaluate_holdout_v4(
        holdout_manifest=project_path(args.holdout_manifest),
        model_path=project_path(args.model_path),
        source_profile=source,
        forensics_profiles=profiles,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["headline"], ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    source, profiles = _profiles(args)
    result = predict_wangxing_v4(
        video_path=project_path(args.video),
        au_path=project_path(args.au),
        model_path=project_path(args.model_path),
        source_profile=source,
        forensics_profiles=profiles,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expression-authenticity Wang Xing PT v4."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument(
        "--manifest",
        default="outputs/vedio_pred/wangxing_v3_generalization_manifest_res1k.json",
    )
    train.add_argument(
        "--cache-dir",
        default="outputs/vedio_pred/cache_wangxing_v4_expression_res1k",
    )
    train.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_v4_expression_res1k.pt",
    )
    train.add_argument(
        "--metrics-output",
        default="outputs/vedio_pred/wangxing_v4_expression_metrics_res1k.json",
    )
    train.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_holdout_excluded.json",
    )
    train.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles.json",
    )
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--modality-dropout", type=float, default=0.10)
    train.add_argument("--device", default="cuda")
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict")
    predict.add_argument("--video", required=True)
    predict.add_argument("--au", required=True)
    predict.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_v4_expression_res1k.pt",
    )
    predict.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_holdout_excluded.json",
    )
    predict.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles.json",
    )
    predict.set_defaults(func=cmd_predict)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split.json",
    )
    evaluate.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_v4_expression_res1k.pt",
    )
    evaluate.add_argument(
        "--output",
        default="outputs/forensics/wangxing_v4_expression_holdout_metrics.json",
    )
    evaluate.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_holdout_excluded.json",
    )
    evaluate.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles.json",
    )
    evaluate.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
