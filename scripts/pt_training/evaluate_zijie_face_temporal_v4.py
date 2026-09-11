"""Evaluate the offline webpage-equivalent ZiJie V4 branch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from wangxing_project.zijie_face_temporal_v4 import score_zijie_face_temporal_v4


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    manifest_path = project_path(args.manifest)
    profile_path = project_path(args.profile)
    result = score_zijie_face_temporal_v4(
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
    summary = {
        key: result[key]
        for key in (
            "schema_version",
            "subject",
            "headline",
            "confusion",
            "holdout_separation",
            "feature_policy",
        )
    }
    summary["production_web_changed"] = False
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
