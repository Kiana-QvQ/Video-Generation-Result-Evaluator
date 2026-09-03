"""Fit/evaluate an isolated face-only XiaoYue web score."""

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
from wangxing_project.xiaoyue_face_web import (
    evaluate_face_web,
    fit_face_web_profile,
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument(
        "--manifest",
        default="data/xiaoyue/experiment_7x7/manifests/pt_manifest.json",
    )
    fit_parser.add_argument(
        "--cache",
        default="outputs/xiaoyue/experiment_7x7/web_face_train_cache.npz",
    )
    fit_parser.add_argument(
        "--profile",
        default="data/xiaoyue/experiment_7x7/profiles/xiaoyue_face_web_profile.json",
    )

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument(
        "--manifest",
        default="data/xiaoyue/experiment_7x7/manifests/web_test_manifest.json",
    )
    eval_parser.add_argument(
        "--profile",
        default="data/xiaoyue/experiment_7x7/profiles/xiaoyue_face_web_profile.json",
    )
    eval_parser.add_argument(
        "--cache",
        default="outputs/xiaoyue/experiment_7x7/web_face_test_cache.npz",
    )
    eval_parser.add_argument(
        "--output-root",
        default="outputs/xiaoyue/experiment_7x7/web_test",
    )
    args = parser.parse_args(argv)

    manifest_path = project_path(args.manifest)
    manifest = _load(manifest_path)
    if args.command == "fit":
        profile = fit_face_web_profile(
            manifest=manifest,
            cache_path=project_path(args.cache),
            output_path=project_path(args.profile),
        )
        print(
            json.dumps(
                {
                    "profile": str(project_path(args.profile).resolve()),
                    "training_counts": profile["training_counts"],
                    "feature_policy": profile["feature_policy"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    profile_path = project_path(args.profile)
    if not profile_path.is_file():
        raise SystemExit(f"Face web profile not found: {profile_path}")
    profile = _load(profile_path)
    result = evaluate_face_web(
        manifest=manifest,
        profile=profile,
        cache_path=project_path(args.cache),
    )
    output_root = project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        **result,
        "manifest": str(manifest_path.resolve()),
        "profile": str(profile_path.resolve()),
        "training_allowed": False,
    }
    (output_root / "all_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "xiaoyue_face_mouth_web_v1_summary",
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
