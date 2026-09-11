"""Apply fixed acceptance gates to the ZiJie offline holdout results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MINIMUM_ACCURACY = 0.90
MINIMUM_REAL_RECALL = 0.90
MINIMUM_GENERATED_RECALL = 1.00
EXPECTED_REAL_HOLDOUT = 18
EXPECTED_AI_HOLDOUT = 2


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _headline_gate(
    name: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    headline = payload.get("headline") or {}
    rows = list(payload.get("rows") or [])
    real_count = sum(
        1 for row in rows if int(row.get("label_generated", 0)) == 0
    )
    ai_count = sum(
        1 for row in rows if int(row.get("label_generated", 0)) == 1
    )
    expected = {
        "overall_accuracy": MINIMUM_ACCURACY,
        "real_recall": MINIMUM_REAL_RECALL,
        "generated_recall": MINIMUM_GENERATED_RECALL,
    }
    checks = [
        {
            "name": f"{name}.{metric}",
            "actual": headline.get(metric),
            "minimum": minimum,
            "passed": (
                isinstance(headline.get(metric), (int, float))
                and float(headline[metric]) >= minimum
            ),
        }
        for metric, minimum in expected.items()
    ]
    checks.extend(
        [
            {
                "name": f"{name}.holdout_real_count",
                "actual": real_count,
                "minimum": EXPECTED_REAL_HOLDOUT,
                "passed": real_count == EXPECTED_REAL_HOLDOUT,
            },
            {
                "name": f"{name}.holdout_ai_count",
                "actual": ai_count,
                "minimum": EXPECTED_AI_HOLDOUT,
                "passed": ai_count == EXPECTED_AI_HOLDOUT,
            },
            {
                "name": f"{name}.background_excluded",
                "actual": bool(
                    (payload.get("feature_policy") or {}).get(
                        "background_used",
                        True,
                    )
                ),
                "minimum": False,
                "passed": not bool(
                    (payload.get("feature_policy") or {}).get(
                        "background_used",
                        True,
                    )
                ),
            },
        ]
    )
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pt-metrics",
        default="outputs/zijie/face_manifold_v1/zijie_face_manifold_v1_metrics.json",
    )
    parser.add_argument(
        "--offline-results",
        default="outputs/zijie/face_manifold_v1/offline_profile_test/all_results.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/zijie/face_manifold_v1/acceptance_gate.json",
    )
    parser.add_argument(
        "--schema-version",
        default="zijie_face_manifold_v1_acceptance_gate",
    )
    args = parser.parse_args(argv)

    pt_path = _path(args.pt_metrics)
    offline_path = _path(args.offline_results)
    if not pt_path.is_file() or not offline_path.is_file():
        raise SystemExit("PT or offline holdout result is missing.")
    pt = _load(pt_path)
    offline = _load(offline_path)
    checks = [
        *_headline_gate("pt", pt),
        *_headline_gate("offline_profile", offline),
        {
            "name": "pt_offline_confusion_match",
            "actual": {
                "pt": pt.get("confusion"),
                "offline_profile": offline.get("confusion"),
            },
            "minimum": "identical",
            "passed": pt.get("confusion") == offline.get("confusion"),
        },
    ]
    failures = [check["name"] for check in checks if not check["passed"]]
    payload = {
        "schema_version": str(args.schema_version),
        "subject": "zijie",
        "holdout_protocol": {
            "real": EXPECTED_REAL_HOLDOUT,
            "generated": EXPECTED_AI_HOLDOUT,
            "training_allowed": False,
        },
        "thresholds": {
            "minimum_accuracy": MINIMUM_ACCURACY,
            "minimum_real_recall": MINIMUM_REAL_RECALL,
            "minimum_generated_recall": MINIMUM_GENERATED_RECALL,
        },
        "checks": checks,
        "passed": not failures,
        "failures": failures,
        "pt_metrics": str(pt_path),
        "offline_results": str(offline_path),
        "production_web_changed": False,
        "note": (
            "A passed gate validates this fixed H3 task holdout only; it does "
            "not establish cross-generator generalization."
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
