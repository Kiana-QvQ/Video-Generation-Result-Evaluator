"""Run webpage-equivalent XiaoYue forensics and specialization evaluation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from evaluator.modules.forensics import analyze_forensics
from evaluator.modules.wangxing.wangxing_specialization import (
    evaluate_specialization,
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([int(row["label_generated"]) for row in rows])
    predictions = np.asarray(
        [
            int(
                row.get("forensics", {})
                .get("summary", {})
                .get("predicted_generated")
                == 1
            )
            for row in rows
        ]
    )
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return {
        "sample_count": len(rows),
        "generated_recall": tp / (tp + fn) if tp + fn else None,
        "real_recall": tn / (tn + fp) if tn + fp else None,
        "overall_accuracy": (tp + tn) / len(rows) if rows else None,
        "generated_precision": tp / (tp + fp) if tp + fp else None,
        "confusion": {
            "tp_generated": tp,
            "tn_real": tn,
            "fp_real_as_generated": fp,
            "fn_generated_as_real": fn,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/xiaoyue/processed/pt_test_manifest.json",
    )
    parser.add_argument(
        "--forensics-profile",
        default="data/xiaoyue/profiles/xiaoyue_forensics_profiles.json",
    )
    parser.add_argument(
        "--identity-profile",
        default="data/xiaoyue/profiles/xiaoyue_identity_profile.json",
    )
    parser.add_argument(
        "--expression-profile",
        default="data/xiaoyue/profiles/xiaoyue_expression_profile.json",
    )
    parser.add_argument(
        "--source-profile",
        default="data/xiaoyue/profiles/xiaoyue_source_profile.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/xiaoyue/web_test",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wangxing-device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--sample-fps", type=float, default=8.0)
    args = parser.parse_args(argv)

    manifest_path = project_path(args.manifest)
    profiles_path = project_path(args.forensics_profile)
    identity_path = project_path(args.identity_profile)
    expression_path = project_path(args.expression_profile)
    source_path = project_path(args.source_profile)
    manifest = _load(manifest_path)
    profiles = _load(profiles_path)
    identity = _load(identity_path)
    expression = _load(expression_path)
    source = _load(source_path)
    rows: list[dict[str, Any]] = []
    samples = [
        *[
            {**item, "label_generated": 0}
            for item in manifest.get("real") or []
        ],
        *[
            {**item, "label_generated": 1}
            for item in manifest.get("fake") or []
        ],
    ]
    for index, sample in enumerate(samples, start=1):
        video = project_path(sample["video"])
        au = project_path(sample["au"])
        print(f"[XiaoYue Web] {index}/{len(samples)} {video.name}", flush=True)
        forensics = analyze_forensics(
            facial_motion=au,
            facial_motion_profile=profiles.get("facial_motion"),
            texture_detail=video,
            texture_detail_profile=profiles.get("texture_detail"),
            authenticity_calibrator=profiles.get("authenticity_calibrator"),
            max_frames=args.max_frames,
            sample_fps=args.sample_fps,
            device=args.device,
        )
        specialization = evaluate_specialization(
            video_path=video,
            au_path=au,
            identity_profile_path=identity_path,
            expression_profile_path=expression_path,
            source_profile_path=source_path,
            device=args.wangxing_device,
            max_identity_frames=8,
        )
        rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "label_generated": int(sample["label_generated"]),
                "video": str(video),
                "au": str(au),
                "forensics": forensics,
                "xiaoyue": specialization,
                "source_probability_real": _finite(
                    specialization.get("source", {}).get(
                        "real_probability_0_1"
                    )
                ),
                "forensics_probability_real": _finite(
                    forensics.get("scores", {}).get(
                        "calibrated_real_probability_0_1"
                    )
                ),
            }
        )
    output_root = project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "xiaoyue_web_forensics_results_v1",
        "subject": "xiaoyue",
        "manifest": str(manifest_path.resolve()),
        "forensics_profile": str(profiles_path.resolve()),
        "training_allowed": False,
        "metrics": _metrics(rows),
        "rows": rows,
    }
    (output_root / "all_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "xiaoyue_web_forensics_summary_v1",
                "subject": "xiaoyue",
                "metrics": payload["metrics"],
                "training_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
    print(f"All results: {output_root / 'all_results.json'}")
    print(f"Summary: {output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
