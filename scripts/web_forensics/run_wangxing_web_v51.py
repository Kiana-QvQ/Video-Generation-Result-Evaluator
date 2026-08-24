"""Offline Web V5.1 evaluator.

It uses the same V5.1 realness contract as PT, while retaining forensics
direction evidence in every row.  It never writes to the online queue.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run offline Web V5.1 using the PT realness evaluator."
    )
    parser.add_argument(
        "--calibrator",
        default="outputs/forensics/wangxing_v5_realness_calibrator.json",
    )
    parser.add_argument(
        "--ranking-root",
        default=r"C:\Users\zhanghaotian\Desktop\ppt_video",
    )
    parser.add_argument("--holdout-group", default="test2")
    parser.add_argument(
        "--test-set",
        dest="test_sets",
        nargs=2,
        action="append",
        metavar=("NAME", "MANIFEST"),
        required=True,
    )
    parser.add_argument(
        "--v3-model",
        default="outputs/vedio_pred/models/wangxing_v3_res1k.pt",
    )
    parser.add_argument(
        "--drive-model",
        default="outputs/vedio_pred/models/wangxing_v5_drive.json",
    )
    parser.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles_web_v3_test_excluded.json",
    )
    parser.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_web_v3_test_excluded.json",
    )
    parser.add_argument(
        "--cache-dir",
        default="outputs/forensics/cache_wangxing_v5_1_web",
    )
    parser.add_argument(
        "--au-output-root",
        default="outputs/forensics/cache_wangxing_v5_1_web/au",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/forensics/wangxing_v5_1_web_results",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wangxing-device", default="cuda")
    parser.add_argument(
        "--min-pairwise",
        type=float,
        default=5.0 / 6.0,
        help="Minimum holdout pairwise ordering rate (default 5/6).",
    )
    parser.add_argument(
        "--enforce-gates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero when holdout/binary gates fail.",
    )
    args = parser.parse_args(argv)
    command = [
        str(PYTHON),
        str(
            PROJECT_ROOT
            / "scripts"
            / "pt_training"
            / "evaluate_wangxing_v5_realness.py"
        ),
        "--calibrator",
        args.calibrator,
        "--ranking-root",
        args.ranking_root,
        "--holdout-group",
        args.holdout_group,
        "--v3-model",
        args.v3_model,
        "--drive-model",
        args.drive_model,
        "--forensics-profile",
        args.forensics_profile,
        "--source-profile",
        args.source_profile,
        "--cache-dir",
        args.cache_dir,
        "--au-output-root",
        args.au_output_root,
        "--output-root",
        args.output_root,
        "--device",
        args.device,
        "--wangxing-device",
        args.wangxing_device,
        "--min-pairwise",
        str(args.min_pairwise),
    ]
    if args.enforce_gates:
        command.append("--enforce-gates")
    else:
        command.append("--no-enforce-gates")
    for name, manifest in args.test_sets:
        command.extend(["--test-set", name, manifest])
    print(">>", subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
