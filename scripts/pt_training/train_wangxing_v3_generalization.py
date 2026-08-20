"""Train / predict / evaluate the v3 temporal domain-generalization model.

This script expects the manifest produced by
``prepare_wangxing_v3_generalization.py`` and never adds Change clips.
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
from wangxing_project.joint_au_pt_v3 import (
    DEFAULT_MODALITY_DROPOUT,
    evaluate_holdout_v3,
    predict_wangxing_v3,
    train_wangxing_v3,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _default_forensics_profile() -> Path:
    primary = project_path("outputs/forensics/forensics_profiles.json")
    if primary.is_file():
        return primary
    return project_path("outputs/forensics/forensics_profiles_quality_filtered.json")


def _load_profiles(
    *,
    source_profile: str,
    forensics_profile: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        _load_json(project_path(source_profile)),
        _load_json(project_path(forensics_profile)),
    )


def cmd_train(args: argparse.Namespace) -> int:
    manifest_path = project_path(args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"v3 manifest missing: {manifest_path}")
    manifest = _load_json(manifest_path)
    source_profile, forensics_profiles = _load_profiles(
        source_profile=args.source_profile,
        forensics_profile=args.forensics_profile,
    )
    result = train_wangxing_v3(
        manifest=manifest,
        cache_dir=project_path(args.cache_dir),
        model_path=project_path(args.model_path),
        source_profile=source_profile,
        forensics_profiles=forensics_profiles,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        modality_dropout=args.modality_dropout,
    )
    metrics_path = project_path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "wangxing_temporal_au_video_v3_metrics_v1",
        "manifest": str(manifest_path),
        "model_path": result["model_path"],
        "headline": result["headline"],
        "confusion": result["confusion"],
        "counts": result["counts"],
        "validation": result["validation"],
        "temperature": result["temperature"],
        "device": result["device"],
        "architecture": (
            "frame_temporal_sequences + shared BiGRU attention pooling + "
            "AU-conditioned gate + auxiliary heads"
        ),
        "note": (
            "Parallel v3 model. It does not replace v1, v2, default dual, "
            "or the noleak fusion head."
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
    source_profile, forensics_profiles = _load_profiles(
        source_profile=args.source_profile,
        forensics_profile=args.forensics_profile,
    )
    result = predict_wangxing_v3(
        video_path=project_path(args.video),
        au_path=project_path(args.au),
        model_path=project_path(args.model_path),
        source_profile=source_profile,
        forensics_profiles=forensics_profiles,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    source_profile, forensics_profiles = _load_profiles(
        source_profile=args.source_profile,
        forensics_profile=args.forensics_profile,
    )
    output = project_path(args.output)
    payload = evaluate_holdout_v3(
        holdout_manifest=project_path(args.holdout_manifest),
        model_path=project_path(args.model_path),
        source_profile=source_profile,
        forensics_profiles=forensics_profiles,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["headline"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["confusion"], ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return 0


def _add_profiles(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_holdout_excluded.json",
    )
    parser.add_argument(
        "--forensics-profile",
        default=str(
            _default_forensics_profile().relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Temporal v3 AU + video domain-generalization model."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train v3")
    train.add_argument(
        "--manifest",
        default=(
            "outputs/vedio_pred/"
            "wangxing_v3_generalization_manifest_res1k.json"
        ),
    )
    train.add_argument(
        "--cache-dir",
        default="outputs/vedio_pred/cache_wangxing_v3_res1k",
    )
    train.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_v3_res1k.pt",
    )
    train.add_argument(
        "--metrics-output",
        default="outputs/vedio_pred/wangxing_v3_holdout_metrics_res1k.json",
    )
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--modality-dropout",
        type=float,
        default=DEFAULT_MODALITY_DROPOUT,
    )
    train.add_argument(
        "--device",
        default="cuda",
        help="MLP training device: cuda, cpu, auto, or cuda:0.",
    )
    _add_profiles(train)
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict", help="Predict one video + AU pair")
    predict.add_argument("--video", required=True)
    predict.add_argument("--au", required=True)
    predict.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_v3_res1k.pt",
    )
    _add_profiles(predict)
    predict.set_defaults(func=cmd_predict)

    evaluate = sub.add_parser("evaluate", help="Evaluate a holdout manifest")
    evaluate.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split.json",
    )
    evaluate.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_v3_res1k.pt",
    )
    evaluate.add_argument(
        "--output",
        default="outputs/forensics/wangxing_v3_holdout_metrics.json",
    )
    _add_profiles(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
