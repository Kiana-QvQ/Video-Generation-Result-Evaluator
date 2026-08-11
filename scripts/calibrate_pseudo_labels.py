#!/usr/bin/env python
"""Fit authenticity calibrators from source labels / multi-model consensus.

No manual per-clip MOS scores are required. Use known real/generated source
labels when available; otherwise keep only high-agreement pseudo-labels and
fit Platt calibration on declared holdout ids.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.forensics.pseudo_label_calibration import (
    build_pseudo_labeled_samples,
    fit_pseudo_label_calibrator,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Automatic pseudo-label calibration for forensics authenticity."
        )
    )
    parser.add_argument(
        "--scored-manifest",
        required=True,
        help=(
            "JSON list/dict with records containing raw scores and optional "
            "source_label / holdout flags."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output calibrator JSON path.",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--allow-non-holdout",
        action="store_true",
        help="Fit on all accepted samples if holdout flags are absent.",
    )
    args = parser.parse_args()

    payload = _load_json(Path(args.scored_manifest))
    if isinstance(payload, dict):
        records = payload.get("records", payload.get("samples", []))
    else:
        records = payload
    if not isinstance(records, list):
        raise SystemExit("Manifest must contain a list of scored records.")

    labeled = build_pseudo_labeled_samples(records)
    calibrator = fit_pseudo_label_calibrator(
        labeled["accepted"],
        require_holdout=not args.allow_non_holdout,
        min_per_class=args.min_per_class,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "calibrator": calibrator,
                "labeling_summary": {
                    "accepted_count": labeled["accepted_count"],
                    "rejected_count": labeled["rejected_count"],
                    "manual_scores_required": False,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "status": calibrator.get("status"),
                "accepted": labeled["accepted_count"],
                "rejected": labeled["rejected_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if calibrator.get("status") in {"ready", "calibrated"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
