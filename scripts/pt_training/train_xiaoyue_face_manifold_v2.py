"""Fit/evaluate the XiaoYue real-manifold face PT checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from wangxing_project.xiaoyue_face_manifold import (
    fit_face_manifold,
    save_pt_checkpoint,
    score_face_manifold,
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/xiaoyue/experiment_7x7/manifests/face_manifold_manifest.json",
    )
    parser.add_argument(
        "--train-cache",
        default="outputs/xiaoyue/experiment_7x7_face_v2/real_bank_features.npz",
    )
    parser.add_argument(
        "--test-cache",
        default="outputs/xiaoyue/experiment_7x7_face_v2/test_features.npz",
    )
    parser.add_argument(
        "--profile",
        default="outputs/xiaoyue/experiment_7x7_face_v2/xiaoyue_face_manifold_profile.json",
    )
    parser.add_argument(
        "--model",
        default="outputs/xiaoyue/experiment_7x7_face_v2/models/xiaoyue_face_manifold_v2.pt",
    )
    parser.add_argument(
        "--metrics",
        default="outputs/xiaoyue/experiment_7x7_face_v2/xiaoyue_face_manifold_v2_metrics.json",
    )
    args = parser.parse_args(argv)

    manifest_path = project_path(args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    manifest = _load(manifest_path)
    profile = fit_face_manifold(
        manifest=manifest,
        cache_path=project_path(args.train_cache),
        output_path=project_path(args.profile),
    )
    save_pt_checkpoint(profile, project_path(args.model))
    result = score_face_manifold(
        manifest=manifest,
        profile=profile,
        cache_path=project_path(args.test_cache),
    )
    result["model_path"] = str(project_path(args.model).resolve())
    result["profile_path"] = str(project_path(args.profile).resolve())
    result["manifest"] = str(manifest_path.resolve())
    output = project_path(args.metrics)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["headline"], ensure_ascii=False, indent=2))
    print(json.dumps(result["confusion"], ensure_ascii=False, indent=2))
    print(f"PT checkpoint: {project_path(args.model).resolve()}")
    print(f"Metrics: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
