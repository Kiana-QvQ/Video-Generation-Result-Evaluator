"""Offline Web V5.2 evaluator using the shared PT contract."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run offline Web V5.2; online Web remains V3."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rank-policy", required=True)
    parser.add_argument(
        "--calibrator",
        default="outputs/forensics/wangxing_v5_realness_calibrator.json",
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
        default="outputs/forensics/cache_wangxing_v5_2_web",
    )
    parser.add_argument(
        "--au-output-root",
        default="outputs/forensics/cache_wangxing_v5_2_web/au",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/forensics/wangxing_v5_2_web_results",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wangxing-device", default="cuda")
    parser.add_argument("--min-pairwise", type=float, default=5.0 / 6.0)
    parser.add_argument("--enforce-gates", action="store_true")
    parser.add_argument(
        "--test-set",
        dest="test_sets",
        nargs=2,
        action="append",
        required=True,
        metavar=("NAME", "MANIFEST"),
    )
    args = parser.parse_args(argv)
    command = [
        str(PYTHON),
        str(
            PROJECT_ROOT
            / "scripts"
            / "pt_training"
            / "evaluate_wangxing_v5_rank.py"
        ),
        "--manifest",
        args.manifest,
        "--rank-policy",
        args.rank_policy,
        "--calibrator",
        args.calibrator,
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
    for name, manifest in args.test_sets:
        command.extend(["--test-set", name, manifest])
    print(">>", subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
