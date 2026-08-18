"""Train / predict / evaluate the v2 AU + dual-scale video model.

The v2 model is parallel to the v1 early-concat baseline:
- separate video and AU encoders;
- AU-conditioned video gate;
- joint plus video-only and AU-only auxiliary heads;
- no webpage or default dual-model integration.
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
from wangxing_project.joint_au_pt import attach_au_pairs
from wangxing_project.joint_au_pt_v2 import (
    DEFAULT_MODALITY_DROPOUT,
    evaluate_holdout_joint_au_pt_v2,
    predict_wangxing_joint_au_pt_v2,
    train_wangxing_joint_au_pt_v2,
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
    source = _load_json(project_path(source_profile))
    forensics = _load_json(project_path(forensics_profile))
    return source, forensics


def cmd_train(args: argparse.Namespace) -> int:
    manifest_path = project_path(args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest missing: {manifest_path}")
    manifest = _load_json(manifest_path)
    if "pairs" not in manifest:
        manifest = attach_au_pairs(
            manifest,
            project_root=PROJECT_ROOT,
            holdout_manifest=project_path(args.holdout_manifest),
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "rewrote_pairs": True,
                    "counts": manifest.get("counts"),
                    "au_pair_missing": {
                        key: len(value)
                        for key, value in (
                            manifest.get("au_pair_missing") or {}
                        ).items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    source_profile, forensics_profiles = _load_profiles(
        source_profile=args.source_profile,
        forensics_profile=args.forensics_profile,
    )
    result = train_wangxing_joint_au_pt_v2(
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
        "schema_version": "wangxing_joint_au_pt_v2_metrics_v1",
        "manifest": str(manifest_path),
        "model_path": result["model_path"],
        "headline": result["headline"],
        "confusion": result["confusion"],
        "counts": result["counts"],
        "validation": result["validation"],
        "temperature": result["temperature"],
        "input_dim": result["input_dim"],
        "video_dim": result["video_dim"],
        "au_dim": result["au_dim"],
        "device": result["device"],
        "architecture": (
            "video_encoder + au_encoder + au_conditioned_gate + "
            "joint/video/au heads"
        ),
        "note": (
            "Parallel v2 model. It does not replace v1, default dual, "
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
    video = project_path(args.video)
    au = project_path(args.au) if args.au else None
    if au is None:
        from wangxing_project.joint_au_pt import resolve_au_csv_for_video

        au = resolve_au_csv_for_video(video, project_root=PROJECT_ROOT)
    if au is None or not Path(au).is_file():
        raise SystemExit(f"AU CSV not found for video: {video}")
    result = predict_wangxing_joint_au_pt_v2(
        video_path=video,
        au_path=au,
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
    payload = evaluate_holdout_joint_au_pt_v2(
        holdout_manifest=project_path(args.holdout_manifest),
        model_path=project_path(args.model_path),
        source_profile=source_profile,
        forensics_profiles=forensics_profiles,
    )
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["headline"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["confusion"], ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Joint AU + dual-scale video v2 (two-branch gated MLP)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train the v2 joint model")
    train.add_argument(
        "--manifest",
        default="outputs/vedio_pred/wangxing_dual_pt_split_res1k.json",
    )
    train.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split.json",
    )
    train.add_argument(
        "--cache-dir",
        default="outputs/vedio_pred/cache_joint_au_pt_v2_res1k",
    )
    train.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_joint_au_pt_v2_res1k.pt",
    )
    train.add_argument(
        "--metrics-output",
        default=(
            "outputs/vedio_pred/"
            "wangxing_joint_au_pt_v2_holdout_metrics_res1k.json"
        ),
    )
    train.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_holdout_excluded.json",
    )
    train.add_argument(
        "--forensics-profile",
        default=str(
            _default_forensics_profile().relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
    )
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--modality-dropout", type=float, default=DEFAULT_MODALITY_DROPOUT)
    train.add_argument(
        "--device",
        default="cuda",
        help="MLP training device: cuda, cpu, auto, or cuda:0.",
    )
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict", help="Score one video + AU pair")
    predict.add_argument("--video", required=True)
    predict.add_argument("--au", default=None)
    predict.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_joint_au_pt_v2_res1k.pt",
    )
    predict.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_holdout_excluded.json",
    )
    predict.add_argument(
        "--forensics-profile",
        default=str(
            _default_forensics_profile().relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
    )
    predict.set_defaults(func=cmd_predict)

    evaluate = sub.add_parser("evaluate", help="Evaluate a holdout manifest")
    evaluate.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split.json",
    )
    evaluate.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_joint_au_pt_v2_res1k.pt",
    )
    evaluate.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_holdout_excluded.json",
    )
    evaluate.add_argument(
        "--forensics-profile",
        default=str(
            _default_forensics_profile().relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
    )
    evaluate.add_argument(
        "--output",
        default="outputs/forensics/wangxing_joint_au_pt_v2_holdout_metrics.json",
    )
    evaluate.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
