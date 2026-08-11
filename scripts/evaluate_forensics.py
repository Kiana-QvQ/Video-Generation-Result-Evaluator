from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.forensics import analyze_forensics
from evaluator.modules.core.paths import project_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate facial-motion and texture forensic evidence."
    )
    parser.add_argument(
        "--profile",
        default="outputs/forensics/forensics_profiles.json",
    )
    parser.add_argument("--au-csv")
    parser.add_argument("--video")
    parser.add_argument(
        "--output",
        default="outputs/forensics/forensics_report.json",
    )
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--sample-fps", type=float, default=8.0)
    parser.add_argument(
        "--no-face-detection",
        action="store_true",
        help="Use the full frame for texture features.",
    )
    args = parser.parse_args(argv)
    if not args.au_csv and not args.video:
        raise SystemExit("Specify at least one of --au-csv or --video.")

    profile_path = project_path(args.profile)
    if not profile_path.is_file():
        raise FileNotFoundError(f"Forensics profile was not found: {profile_path}")
    profiles = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    report = analyze_forensics(
        facial_motion=project_path(args.au_csv) if args.au_csv else None,
        facial_motion_profile=profiles.get("facial_motion"),
        texture_detail=project_path(args.video) if args.video else None,
        texture_detail_profile=profiles.get("texture_detail"),
        authenticity_calibrator=profiles.get("authenticity_calibrator"),
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
        detect_faces=not args.no_face_detection,
    )
    report["evaluation_meta"] = {
        "profile": str(profile_path),
        "au_csv": str(project_path(args.au_csv)) if args.au_csv else None,
        "video": str(project_path(args.video)) if args.video else None,
        "face_detection": not args.no_face_detection,
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    output.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
