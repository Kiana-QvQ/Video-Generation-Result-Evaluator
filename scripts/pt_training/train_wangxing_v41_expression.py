"""Train and evaluate the isolated Wang Xing expression-only PT v4.1."""

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
from wangxing_project.joint_au_pt_v41 import (
    evaluate_holdout_v41,
    predict_wangxing_v41,
    train_wangxing_v41,
)


def _load(path: str) -> dict[str, Any]:
    return json.loads(
        project_path(path).read_text(encoding="utf-8-sig")
    )


def cmd_train(args: argparse.Namespace) -> int:
    result = train_wangxing_v41(
        manifest=_load(args.manifest),
        cache_dir=project_path(args.cache_dir),
        model_path=project_path(args.model_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    payload = {
        "schema_version": "wangxing_expression_authenticity_v41_metrics_v1",
        "manifest": str(project_path(args.manifest)),
        **result,
        "architecture": (
            "profile-independent AU/landmark sequence GRU plus "
            "Blendshape temporal branch"
        ),
        "training_labels": ["real", "seedance"],
        "test_sets_are_excluded": True,
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
    output = project_path(args.output)
    payload = evaluate_holdout_v41(
        holdout_manifest=project_path(args.holdout_manifest),
        model_path=project_path(args.model_path),
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
    result = predict_wangxing_v41(
        video_path=project_path(args.video),
        au_path=project_path(args.au),
        model_path=project_path(args.model_path),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wang Xing expression-only PT v4.1."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--manifest", required=True)
    train.add_argument(
        "--cache-dir",
        default="outputs/vedio_pred/cache_wangxing_v41_expression_res1k",
    )
    train.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_v41_expression_res1k.pt",
    )
    train.add_argument(
        "--metrics-output",
        default="outputs/vedio_pred/wangxing_v41_expression_metrics_res1k.json",
    )
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--holdout-manifest", required=True)
    evaluate.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_v41_expression_res1k.pt",
    )
    evaluate.add_argument(
        "--output",
        required=True,
    )
    evaluate.set_defaults(func=cmd_evaluate)

    predict = sub.add_parser("predict")
    predict.add_argument("--video", required=True)
    predict.add_argument("--au", required=True)
    predict.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_v41_expression_res1k.pt",
    )
    predict.set_defaults(func=cmd_predict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
