"""Fit a pseudo-label authenticity calibrator without manual clip scores.

Uses known real/generated source labels when present, otherwise multi-model
score consensus, then fits a held-out Platt calibrator.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from evaluator.modules.core.paths import project_path
    from evaluator.modules.forensics.pseudo_label_calibration import (
        apply_pseudo_calibrator,
        build_pseudo_labeled_samples,
        fit_pseudo_label_calibrator,
    )
except ImportError:
    # Also run directly from a flat collaborator host containing modules/.
    from modules.core.paths import project_path
    from modules.forensics.pseudo_label_calibration import (
        apply_pseudo_calibrator,
        build_pseudo_labeled_samples,
        fit_pseudo_label_calibrator,
    )


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "samples", "items", "scored"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        # Single scored report masquerading as a one-item manifest.
        if any(
            key in payload
            for key in (
                "raw_real_domain_evidence_0_1",
                "scores",
                "source_label",
            )
        ):
            record = dict(payload)
            scores = payload.get("scores")
            if isinstance(scores, dict):
                record.update(scores)
            return [record]
    raise SystemExit(
        f"Unsupported scored manifest format: {path}. "
        "Expected a list of records or an object with a records/samples array."
    )


def _maybe_mark_holdout(
    records: list[dict[str, Any]],
    holdout_ids: set[str] | None,
) -> list[dict[str, Any]]:
    if not holdout_ids:
        return records
    updated: list[dict[str, Any]] = []
    for record in records:
        cloned = dict(record)
        sample_id = str(record.get("id", record.get("source", "")))
        if sample_id in holdout_ids or str(record.get("source", "")) in holdout_ids:
            cloned["holdout"] = True
        updated.append(cloned)
    return updated


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a pseudo-label calibrator from automatic forensics scores "
            "(source labels and/or multi-model consensus; no manual scores)."
        )
    )
    parser.add_argument(
        "--scored-manifest",
        required=True,
        help="JSON with scored clips (raw evidence + optional source_label).",
    )
    parser.add_argument(
        "--output",
        default="outputs/forensics/pseudo_calibrator.json",
    )
    parser.add_argument(
        "--holdout-ids",
        help="Optional JSON list / manifest of holdout sample ids.",
    )
    parser.add_argument(
        "--allow-non-holdout",
        action="store_true",
        help="Fit on all accepted pseudo-labels when holdout flags are absent.",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--update-profile",
        help="Optional forensics_profiles.json to embed authenticity_calibrator.",
    )
    args = parser.parse_args(argv)

    manifest_path = project_path(args.scored_manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"Scored manifest not found: {manifest_path}")

    holdout_ids: set[str] | None = None
    if args.holdout_ids:
        holdout_path = project_path(args.holdout_ids)
        holdout_payload = json.loads(holdout_path.read_text(encoding="utf-8-sig"))
        if isinstance(holdout_payload, list):
            holdout_ids = {str(item) for item in holdout_payload}
        elif isinstance(holdout_payload, dict):
            values = holdout_payload.get("ids") or holdout_payload.get("holdout_ids")
            if isinstance(values, list):
                holdout_ids = {str(item) for item in values}
            else:
                raise SystemExit("holdout-ids JSON must be a list or {ids:[...]}.")
        else:
            raise SystemExit("Unsupported holdout-ids format.")

    records = _maybe_mark_holdout(_load_records(manifest_path), holdout_ids)
    built = build_pseudo_labeled_samples(
        records,
        min_confidence=args.min_confidence,
    )
    calibrator = fit_pseudo_label_calibrator(
        built["accepted"],
        require_holdout=not args.allow_non_holdout,
        min_per_class=args.min_per_class,
    )
    payload = {
        "schema_version": calibrator.get("schema_version"),
        "manual_scores_required": False,
        "source_manifest": str(manifest_path),
        "pseudo_label_summary": {
            "accepted_count": built["accepted_count"],
            "rejected_count": built["rejected_count"],
            "rejected_preview": built["rejected"][:20],
        },
        "calibrator": calibrator,
    }
    if calibrator.get("status") == "ready":
        # Smoke-apply on accepted holdout samples for a quick sanity dump.
        preview = []
        for sample in built["accepted"][:12]:
            if args.allow_non_holdout or sample.get("holdout"):
                preview.append(
                    {
                        "id": sample["id"],
                        "raw_score": sample["raw_score"],
                        "label": sample["label"],
                        "calibrated": apply_pseudo_calibrator(
                            sample["raw_score"],
                            calibrator,
                        ),
                    }
                )
        payload["preview"] = preview

    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.update_profile and calibrator.get("status") == "ready":
        profile_path = project_path(args.update_profile)
        profiles = json.loads(profile_path.read_text(encoding="utf-8-sig"))
        if not isinstance(profiles, dict):
            raise SystemExit(f"Profile is not a JSON object: {profile_path}")
        profiles["authenticity_calibrator"] = calibrator
        profiles["pseudo_label_calibration"] = {
            "source_manifest": str(manifest_path),
            "accepted_count": built["accepted_count"],
            "rejected_count": built["rejected_count"],
        }
        profile_path.write_text(
            json.dumps(profiles, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Updated profile calibrator: {profile_path}")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if calibrator.get("status") != "ready":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
