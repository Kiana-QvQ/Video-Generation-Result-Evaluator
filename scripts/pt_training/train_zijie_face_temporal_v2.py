"""Fit and evaluate the offline ZiJie V2 fixed-window temporal candidate."""

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
from wangxing_project.zijie_face_temporal_v2 import (
    fit_zijie_face_temporal_v2,
    save_pt_checkpoint,
    score_zijie_face_temporal_v2,
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
        default="data/zijie/manifests/zijie_face_manifold_v1.json",
    )
    parser.add_argument(
        "--train-cache",
        default="outputs/zijie/face_temporal_v2/cache/train_features.npz",
    )
    parser.add_argument(
        "--test-cache",
        default="outputs/zijie/face_temporal_v2/cache/test_features.npz",
    )
    parser.add_argument(
        "--profile",
        default="outputs/zijie/face_temporal_v2/zijie_face_temporal_v2_profile.json",
    )
    parser.add_argument(
        "--model",
        default="outputs/zijie/face_temporal_v2/models/zijie_face_temporal_v2.pt",
    )
    parser.add_argument(
        "--metrics",
        default="outputs/zijie/face_temporal_v2/zijie_face_temporal_v2_metrics.json",
    )
    args = parser.parse_args(argv)
    manifest_path = project_path(args.manifest)
    manifest = _load(manifest_path)
    profile = fit_zijie_face_temporal_v2(
        manifest=manifest,
        cache_path=project_path(args.train_cache),
        output_path=project_path(args.profile),
    )
    save_pt_checkpoint(profile, project_path(args.model))
    result = score_zijie_face_temporal_v2(
        manifest=manifest,
        profile=profile,
        cache_path=project_path(args.test_cache),
    )
    result["model_path"] = str(project_path(args.model).resolve())
    result["profile_path"] = str(project_path(args.profile).resolve())
    result["manifest"] = str(manifest_path.resolve())
    metrics_path = project_path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["headline"], ensure_ascii=False, indent=2))
    print(json.dumps(result["confusion"], ensure_ascii=False, indent=2))
    print(f"PT checkpoint: {project_path(args.model).resolve()}")
    print(f"Metrics: {metrics_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
