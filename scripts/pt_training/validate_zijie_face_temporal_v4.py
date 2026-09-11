"""Validate ZiJie V4 classification and score separation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-metrics", required=True)
    parser.add_argument("--offline-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-gap", type=float, default=0.03)
    args = parser.parse_args()
    pt = _load(_path(args.pt_metrics))
    offline = _load(_path(args.offline_results))
    headline = pt.get("headline") or {}
    gap = float(
        (pt.get("holdout_separation") or {}).get(
            "minimum_generated_minus_maximum_real",
            float("-inf"),
        )
    )
    checks = [
        {
            "name": "overall_accuracy",
            "actual": headline.get("overall_accuracy"),
            "minimum": 0.95,
            "passed": float(headline.get("overall_accuracy") or 0) >= 0.95,
        },
        {
            "name": "real_recall",
            "actual": headline.get("real_recall"),
            "minimum": 17 / 18,
            "passed": float(headline.get("real_recall") or 0) >= 17 / 18,
        },
        {
            "name": "generated_recall",
            "actual": headline.get("generated_recall"),
            "minimum": 1.0,
            "passed": float(headline.get("generated_recall") or 0) >= 1.0,
        },
        {
            "name": "strict_raw_score_gap",
            "actual": gap,
            "minimum": args.minimum_gap,
            "passed": gap >= args.minimum_gap,
        },
        {
            "name": "pt_offline_match",
            "actual": {
                "pt": pt.get("confusion"),
                "offline": offline.get("confusion"),
            },
            "minimum": "identical",
            "passed": pt.get("confusion") == offline.get("confusion"),
        },
    ]
    failures = [check["name"] for check in checks if not check["passed"]]
    payload = {
        "schema_version": "zijie_face_temporal_v4_acceptance_gate",
        "subject": "zijie",
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "production_web_changed": False,
        "holdout_warning": (
            "Only two unique AI videos are available in the strict holdout; "
            "passing does not prove cross-seed or cross-generator robustness."
        ),
    }
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
