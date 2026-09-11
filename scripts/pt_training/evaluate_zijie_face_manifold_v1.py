"""Evaluate the saved ZiJie profile without refitting it.

This is the webpage-equivalent offline result producer. It deliberately does
not import or register any browser route or UI specialization.
"""

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
from wangxing_project.zijie_face_manifold import score_zijie_face_manifold


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
        "--profile",
        default="outputs/zijie/face_manifold_v1/zijie_face_manifold_profile.json",
    )
    parser.add_argument(
        "--cache",
        default="outputs/zijie/face_manifold_v1/cache/offline_holdout_features.npz",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/zijie/face_manifold_v1/offline_profile_test",
    )
    args = parser.parse_args(argv)

    manifest_path = project_path(args.manifest)
    profile_path = project_path(args.profile)
    if not manifest_path.is_file() or not profile_path.is_file():
        raise SystemExit("ZiJie manifest or profile is missing.")
    result = score_zijie_face_manifold(
        manifest=_load(manifest_path),
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
                "schema_version": "zijie_face_manifold_v1_offline_summary",
                "subject": "zijie",
                "headline": result["headline"],
                "confusion": result["confusion"],
                "feature_policy": result["feature_policy"],
                "production_web_changed": False,
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
