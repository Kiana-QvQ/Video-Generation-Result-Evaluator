"""Evaluate an existing isolated XiaoYue PT checkpoint."""

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
from wangxing_project.joint_au_pt_v3 import evaluate_holdout_v3


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/xiaoyue/processed/pt_test_manifest.json",
    )
    parser.add_argument(
        "--model-path",
        default="outputs/xiaoyue/models/xiaoyue_temporal_v3.pt",
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
        "--output",
        default="outputs/xiaoyue/xiaoyue_temporal_v3_test_metrics.json",
    )
    args = parser.parse_args(argv)
    manifest_path = project_path(args.manifest)
    model_path = project_path(args.model_path)
    source_path = project_path(args.source_profile)
    forensics_path = project_path(args.forensics_profile)
    for path in (manifest_path, model_path, source_path, forensics_path):
        if not path.is_file():
            raise SystemExit(f"Missing XiaoYue evaluation asset: {path}")
    source = _load(source_path)
    forensics = _load(forensics_path)
    source_manifest = _load(manifest_path)
    holdout = {
        "real": list(source_manifest.get("real") or []),
        "seedance": list(
            source_manifest.get("seedance")
            or source_manifest.get("fake")
            or []
        ),
    }
    temp_manifest = manifest_path.with_name(".xiaoyue_eval_manifest.json")
    temp_manifest.write_text(
        json.dumps(holdout, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        result = evaluate_holdout_v3(
            holdout_manifest=temp_manifest,
            model_path=model_path,
            source_profile=source,
            forensics_profiles=forensics,
        )
    finally:
        temp_manifest.unlink(missing_ok=True)
    result["subject"] = "xiaoyue"
    result["model_path"] = str(model_path.resolve())
    result["test_manifest"] = str(manifest_path.resolve())
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["headline"], ensure_ascii=False, indent=2))
    print(json.dumps(result["confusion"], ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
