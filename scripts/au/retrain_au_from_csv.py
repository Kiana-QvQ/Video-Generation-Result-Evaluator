from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)


def _run(label: str, command: list[str]) -> None:
    print(f"AU_CSV_STAGE|{label}", flush=True)
    print(f"AU_CSV_COMMAND|{' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild AU profiles and the leakage classifier from existing "
            "AU CSV files only. No video extraction or downloads are run."
        )
    )
    parser.add_argument("--target-au-root", default="data/au/MD_CL")
    parser.add_argument("--target-video-root", default="data/MD_CL")
    parser.add_argument("--negative-au-root", default="data/au/negative")
    parser.add_argument("--emotion-au-root", default="data/au/MD_CL")
    parser.add_argument("--output-root", default="data/au")
    args = parser.parse_args(argv)

    output_root = PROJECT_ROOT / args.output_root
    _run(
        "wangxing_profile",
        [
            str(PYTHON),
            str(
                PROJECT_ROOT
                / "scripts"
                / "数据构建"
                / "build_au_profile.py"
            ),
            "--au-root",
            args.target_au_root,
            "--input-root",
            args.target_au_root,
            "--video-root",
            args.target_video_root,
            "--output",
            str(output_root / "wangxing_au_profile.json"),
        ],
    )
    _run(
        "emotion_profile",
        [
            str(PYTHON),
            str(
                PROJECT_ROOT
                / "scripts"
                / "数据构建"
                / "build_original_emotion_au_profile.py"
            ),
            "--au-root",
            args.emotion_au_root,
            "--video-root",
            args.target_video_root,
            "--csv-only",
            "--output",
            str(output_root / "original_emotion_au_profile.json"),
        ],
    )
    _run(
        "leakage_classifier",
        [
            str(PYTHON),
            str(PROJECT_ROOT / "scripts/au/fit_au_leakage_classifier.py"),
            "--positive-root",
            args.target_au_root,
            "--negative-root",
            args.negative_au_root,
            "--output",
            str(output_root / "au_leakage_classifier.json"),
        ],
    )
    print(
        "AU_CSV_RESULT|completed|existing_csv_only|"
        "no_video_extraction|no_download",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
