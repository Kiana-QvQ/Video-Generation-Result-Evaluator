"""One-click V5 PT pipeline: train DriveHead, then evaluate frozen V3."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def _run(command: list[str]) -> None:
    print(">>", subprocess.list2cmdline([str(item) for item in command]), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="V5 PT: frozen V3 + auxiliary DriveHead + independent tests."
    )
    parser.add_argument(
        "--train-manifest",
        default="outputs/vedio_pred/wangxing_v3_generalization_manifest_res1k.json",
    )
    parser.add_argument(
        "--drive-cache",
        default="outputs/vedio_pred/cache_wangxing_v5_drive_res1k",
    )
    parser.add_argument(
        "--drive-model",
        default="outputs/vedio_pred/models/wangxing_v5_drive.json",
    )
    parser.add_argument(
        "--drive-metrics",
        default="outputs/vedio_pred/wangxing_v5_drive_metrics_res1k.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/vedio_pred/wangxing_v5_cascade_results",
    )
    parser.add_argument(
        "--test-set",
        dest="test_sets",
        nargs=2,
        action="append",
        metavar=("NAME", "MANIFEST"),
        required=True,
    )
    parser.add_argument(
        "--transition-cache",
        default="outputs/vedio_pred/cache_wangxing_v4_expression_res1k/wangxing_v4_transition.npz",
    )
    parser.add_argument(
        "--blendshape-cache",
        default="outputs/vedio_pred/cache_wangxing_v4_expression_res1k/wangxing_v4_blendshape.npz",
    )
    parser.add_argument("--include-blendshape", action="store_true")
    parser.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_holdout_excluded.json",
    )
    parser.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles.json",
    )
    parser.add_argument("--rank-policy", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not PYTHON.is_file():
        raise SystemExit(f"Project Python not found: {PYTHON}")

    train_command = [
        str(PYTHON),
        str(PROJECT_ROOT / "scripts" / "pt_training" / "train_wangxing_v5_drive.py"),
        "train",
        "--manifest",
        args.train_manifest,
        "--cache-dir",
        args.drive_cache,
        "--model-path",
        args.drive_model,
        "--transition-cache",
        args.transition_cache,
        "--blendshape-cache",
        args.blendshape_cache,
        "--metrics-output",
        args.drive_metrics,
        "--seed",
        str(args.seed),
    ]
    if args.include_blendshape:
        train_command.append("--include-blendshape")
    _run(train_command)
    command = [
        str(PYTHON),
        str(PROJECT_ROOT / "scripts" / "pt_training" / "evaluate_wangxing_v5_cascade.py"),
        "--v3-model",
        "outputs/vedio_pred/models/wangxing_v3_res1k.pt",
        "--drive-model",
        args.drive_model,
        "--drive-cache",
        args.drive_cache,
        "--transition-cache",
        args.transition_cache,
        "--blendshape-cache",
        args.blendshape_cache,
        "--source-profile",
        args.source_profile,
        "--forensics-profile",
        args.forensics_profile,
        "--output-root",
        args.output_root,
    ]
    if args.include_blendshape:
        command.append("--include-blendshape")
    if args.rank_policy:
        command.extend(["--rank-policy", args.rank_policy])
    for name, manifest in args.test_sets:
        command.extend(["--test-set", name, manifest])
    _run(command)
    print("[PT v5] completed; frozen V3 decision diff must remain zero.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
