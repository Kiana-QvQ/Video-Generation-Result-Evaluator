"""Train/evaluate the isolated XiaoYue face-and-mouth PT model."""

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
from wangxing_project.xiaoyue_face_pt import train_xiaoyue_face


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/xiaoyue/experiment_7x7/manifests/pt_manifest.json",
    )
    parser.add_argument(
        "--cache",
        default="outputs/xiaoyue/experiment_7x7/face_feature_cache.npz",
    )
    parser.add_argument(
        "--model",
        default="outputs/xiaoyue/experiment_7x7/models/xiaoyue_face_mouth_v1.pt",
    )
    parser.add_argument(
        "--metrics",
        default="outputs/xiaoyue/experiment_7x7/xiaoyue_face_mouth_v1_metrics.json",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    manifest_path = project_path(args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"Face PT manifest not found: {manifest_path}")
    payload = train_xiaoyue_face(
        manifest=_load(manifest_path),
        cache_path=project_path(args.cache),
        model_path=project_path(args.model),
        metrics_path=project_path(args.metrics),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(payload["headline"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["confusion"], ensure_ascii=False, indent=2))
    print(f"Model: {project_path(args.model).resolve()}")
    print(f"Metrics: {project_path(args.metrics).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
