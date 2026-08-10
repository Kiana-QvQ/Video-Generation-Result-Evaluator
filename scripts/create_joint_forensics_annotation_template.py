from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.core.paths import project_path


ANNOTATION_FIELDS = [
    "video_path",
    "au_path",
    "domain",
    "source_label",
    "identity_label",
    "expression_naturalness_1_to_5",
    "expression_support_label",
    "texture_detail_quality_1_to_5",
    "temporal_quality_1_to_5",
    "seedance_artifact_visibility_0_to_3",
    "audio_visual_sync_1_to_5",
    "annotator",
    "notes",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a manual-label template without opening videos or AU "
            "CSVs."
        )
    )
    parser.add_argument(
        "--manifest",
        default="outputs/forensics/joint_forensics_manifest.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/forensics/joint_forensics_annotations.csv",
    )
    parser.add_argument(
        "--domain",
        choices=("real", "seedance", "both"),
        default="seedance",
    )
    args = parser.parse_args()

    manifest_path = project_path(args.manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    records = payload.get("records", [])
    if args.domain != "both":
        records = [
            record
            for record in records
            if record.get("domain") == args.domain
        ]

    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "video_path": record.get("video_path"),
                    "au_path": record.get("au_path"),
                    "domain": record.get("domain"),
                    "source_label": record.get("source_label"),
                    "identity_label": record.get("identity_label"),
                    "expression_naturalness_1_to_5": "",
                    "expression_support_label": "",
                    "texture_detail_quality_1_to_5": "",
                    "temporal_quality_1_to_5": "",
                    "seedance_artifact_visibility_0_to_3": "",
                    "audio_visual_sync_1_to_5": "",
                    "annotator": "",
                    "notes": "",
                }
            )
    print(f"records={len(records)}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
