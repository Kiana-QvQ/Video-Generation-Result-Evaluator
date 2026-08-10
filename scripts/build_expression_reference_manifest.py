from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.wangxing.expression_dataset import (
    build_expression_manifest,
    validate_expression_manifest,
)
from evaluator.core.paths import project_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Wang Xing expression reference manifest."
    )
    parser.add_argument(
        "--root",
        default="data/video",
        help="Directory containing slice_manifest.json and the video folders.",
    )
    parser.add_argument(
        "--output",
        default="data/video/expression_reference_manifest.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any manifest row is missing its local MP4.",
    )
    args = parser.parse_args()

    root = project_path(args.root)
    output = project_path(args.output)
    payload = build_expression_manifest(root)
    if args.strict and payload["source"]["missing_video_rows"]:
        print(
            "ERROR: "
            f"{payload['source']['missing_video_rows']} manifest rows have no MP4."
        )
        return 1
    errors = validate_expression_manifest(payload)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    print(
        "manifest_rows={manifest} usable_rows={usable} "
        "emotion_rows={emotion} missing_video_rows={missing} "
        "filesystem_rows={filesystem}".format(
            manifest=payload["source"]["manifest_rows"],
            usable=payload["source"]["usable_rows"],
            emotion=payload["source"]["emotion_rows"],
            missing=payload["source"]["missing_video_rows"],
            filesystem=payload["source"]["filesystem_rows"],
        )
    )
    print(json.dumps(payload["counts"]["emotion"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
