"""Train and evaluate an isolated XiaoYue temporal AU/video classifier."""

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
from wangxing_project.joint_au_pt_v3 import (
    DEFAULT_MODALITY_DROPOUT,
    evaluate_holdout_v3,
    train_wangxing_v3,
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _validate_training_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("subject") != "xiaoyue":
        raise ValueError("The PT manifest subject must be xiaoyue.")
    train = manifest.get("pairs", {}).get("train", {})
    for label in ("real", "fake"):
        for item in train.get(label, []):
            path = str(item.get("video") or "").casefold().replace("\\", "/")
            if "/data/xiaoyue/test/" in path or "/test_reference/" in path:
                raise ValueError(f"Test video entered XiaoYue training: {path}")


def _write_test_manifest(manifest: dict[str, Any], path: Path) -> None:
    test_manifest = {
        "schema_version": "xiaoyue_temporal_au_video_v3_test_manifest_v1",
        "subject": "xiaoyue",
        "training_allowed": False,
        "real": list(manifest["pairs"]["test"]["real"]),
        "fake": list(manifest["pairs"]["test"]["fake"]),
        "seedance": list(manifest["pairs"]["test"]["fake"]),
    }
    path.write_text(
        json.dumps(test_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/xiaoyue/processed/pt_manifest.json",
    )
    parser.add_argument(
        "--source-profile",
        default="data/xiaoyue/profiles/xiaoyue_source_profile.json",
    )
    parser.add_argument(
        "--forensics-profile",
        default="data/xiaoyue/profiles/xiaoyue_forensics_profiles.json",
    )
    parser.add_argument(
        "--cache-dir",
        default="outputs/xiaoyue/pt_v3_cache",
    )
    parser.add_argument(
        "--model-path",
        default="outputs/xiaoyue/models/xiaoyue_temporal_v3.pt",
    )
    parser.add_argument(
        "--metrics-output",
        default="outputs/xiaoyue/xiaoyue_temporal_v3_metrics.json",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modality-dropout",
        type=float,
        default=DEFAULT_MODALITY_DROPOUT,
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    manifest_path = project_path(args.manifest)
    source_path = project_path(args.source_profile)
    forensics_path = project_path(args.forensics_profile)
    for path in (manifest_path, source_path, forensics_path):
        if not path.is_file():
            raise SystemExit(f"Missing XiaoYue training asset: {path}")
    manifest = _load(manifest_path)
    _validate_training_manifest(manifest)
    source_profile = _load(source_path)
    forensics_profile = _load(forensics_path)

    result = train_wangxing_v3(
        manifest=manifest,
        cache_dir=project_path(args.cache_dir),
        model_path=project_path(args.model_path),
        source_profile=source_profile,
        forensics_profiles=forensics_profile,
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
        "schema_version": "xiaoyue_temporal_au_video_v3_metrics_v1",
        "subject": "xiaoyue",
        "manifest": str(manifest_path.resolve()),
        "model_path": result["model_path"],
        "headline": result["headline"],
        "confusion": result["confusion"],
        "counts": result["counts"],
        "validation": result["validation"],
        "temperature": result["temperature"],
        "device": result["device"],
        "architecture": (
            "dual-scale frame temporal features + shared BiGRU attention "
            "+ AU-conditioned gate + video/AU auxiliary heads"
        ),
        "test_policy": "XiaoYue test/reference videos are evaluation-only.",
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    test_path = project_path(args.manifest).with_name("pt_test_manifest.json")
    _write_test_manifest(manifest, test_path)
    evaluation = evaluate_holdout_v3(
        holdout_manifest=test_path,
        model_path=project_path(args.model_path),
        source_profile=source_profile,
        forensics_profiles=forensics_profile,
    )
    evaluation_path = metrics_path.with_name(
        "xiaoyue_temporal_v3_test_metrics.json"
    )
    evaluation["subject"] = "xiaoyue"
    evaluation_path.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["headline"], ensure_ascii=False, indent=2))
    print(f"Model: {result['model_path']}")
    print(f"Train metrics: {metrics_path}")
    print(f"Test metrics: {evaluation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
