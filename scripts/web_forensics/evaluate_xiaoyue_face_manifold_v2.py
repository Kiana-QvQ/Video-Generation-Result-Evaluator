"""Fit/evaluate the XiaoYue real-manifold face web profile."""

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
from wangxing_project.xiaoyue_face_manifold import (
    fit_face_manifold,
    score_face_manifold,
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit")
    fit.add_argument(
        "--manifest",
        default="data/xiaoyue/experiment_7x7/manifests/face_manifold_manifest.json",
    )
    fit.add_argument(
        "--cache",
        default="outputs/xiaoyue/experiment_7x7_face_v2/real_bank_features.npz",
    )
    fit.add_argument(
        "--profile",
        default="outputs/xiaoyue/experiment_7x7_face_v2/xiaoyue_face_manifold_profile.json",
    )

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument(
        "--manifest",
        default="data/xiaoyue/experiment_7x7/manifests/face_manifold_manifest.json",
    )
    evaluate.add_argument(
        "--profile",
        default="outputs/xiaoyue/experiment_7x7_face_v2/xiaoyue_face_manifold_profile.json",
    )
    evaluate.add_argument(
        "--cache",
        default="outputs/xiaoyue/experiment_7x7_face_v2/test_features.npz",
    )
    evaluate.add_argument(
        "--output-root",
        default="outputs/xiaoyue/experiment_7x7_face_v2/web_test",
    )
    args = parser.parse_args(argv)
    manifest_path = project_path(args.manifest)
    manifest = _load(manifest_path)

    if args.command == "fit":
        profile = fit_face_manifold(
            manifest=manifest,
            cache_path=project_path(args.cache),
            output_path=project_path(args.profile),
        )
        print(
            json.dumps(
                {
                    "profile": str(project_path(args.profile).resolve()),
                    "training_counts": profile["training_counts"],
                    "threshold": profile["threshold"],
                    "feature_policy": profile["feature_policy"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    profile_path = project_path(args.profile)
    if not profile_path.is_file():
        raise SystemExit(f"Profile not found: {profile_path}")
    result = score_face_manifold(
        manifest=manifest,
        profile=_load(profile_path),
        cache_path=project_path(args.cache),
    )
    output_root = project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    result["manifest"] = str(manifest_path.resolve())
    result["profile"] = str(profile_path.resolve())
    (output_root / "all_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "xiaoyue_face_real_manifold_v2_summary",
                "subject": "xiaoyue",
                "headline": result["headline"],
                "confusion": result["confusion"],
                "feature_policy": result["feature_policy"],
                "training_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["headline"], ensure_ascii=False, indent=2))
    print(f"All results: {output_root / 'all_results.json'}")
    print(f"Summary: {output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
