from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
EXTRACTOR = PROJECT_ROOT / "scripts" / "extract_libreface_au.py"
EVALUATOR = PROJECT_ROOT / "scripts" / "evaluate_au_compliance.py"


def _csv_path(video_path: Path, output_root: Path) -> Path:
    return output_root / (
        video_path.name.rsplit(".", 1)[0] + ".csv"
    )


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _run_extraction(
    video_path: Path,
    output_root: Path,
    *,
    device: str,
    batch_size: int,
    num_workers: int,
    force: bool,
) -> Path:
    command: list[str] = [
        str(PYTHON),
        str(EXTRACTOR),
        "--input",
        str(video_path),
        "--output-root",
        str(output_root),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
    ]
    if force:
        command.append("--force")
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"AU extraction failed for {video_path}. "
            "Check that the video contains a visible, front-facing face."
        ) from exc

    output_path = _csv_path(video_path, output_root)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"AU extraction completed without creating {output_path}"
        )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract AU features and evaluate one generated video using "
            "the trained AU profile and leakage classifier."
        )
    )
    parser.add_argument("--generated-video", required=True)
    parser.add_argument(
        "--output-root",
        default="data/au/generated",
        help="Directory for the generated video's AU CSV.",
    )
    parser.add_argument(
        "--driver-video",
        help="Optional driver video; its AU CSV is extracted automatically.",
    )
    parser.add_argument(
        "--driver-au",
        help="Optional existing driver AU CSV instead of --driver-video.",
    )
    parser.add_argument(
        "--driver-au-root",
        default="data/au/driver",
        help="Directory for an automatically extracted driver AU CSV.",
    )
    parser.add_argument(
        "--au-profile",
        default="data/au/wangxing_au_profile.json",
    )
    parser.add_argument(
        "--leakage-classifier",
        default="data/au/au_leakage_classifier.json",
    )
    parser.add_argument("--expected-class")
    parser.add_argument("--target-video")
    parser.add_argument("--target-image", action="append")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--identity-threshold", type=float, default=0.75)
    parser.add_argument("--personal-au-threshold", type=float, default=0.50)
    parser.add_argument(
        "--driver-expression-threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument("--leakage-threshold", type=float, default=0.50)
    parser.add_argument("--output")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract AU CSV files even when they already exist.",
    )
    return parser


def _existing_file(value: str | None, *, label: str) -> Path | None:
    if not value:
        return None
    path = _project_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not found: {path}")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.driver_video and args.driver_au:
        raise ValueError("Use --driver-video or --driver-au, not both.")

    generated_video = _existing_file(
        args.generated_video,
        label="Generated video",
    )
    assert generated_video is not None
    au_profile = _project_path(args.au_profile)
    leakage_classifier = _project_path(args.leakage_classifier)
    output_root = _project_path(args.output_root)
    driver_au_root = _project_path(args.driver_au_root)
    output = _project_path(args.output) if args.output else None
    _existing_file(str(au_profile), label="AU profile")
    _existing_file(
        str(leakage_classifier),
        label="Leakage classifier",
    )
    target_video = _existing_file(args.target_video, label="Target video")
    target_images = [
        _existing_file(value, label="Target image")
        for value in (args.target_image or [])
    ]
    driver_au = _existing_file(args.driver_au, label="Driver AU CSV")

    generated_au = _run_extraction(
        generated_video,
        output_root,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force=args.force,
    )
    if args.driver_video:
        driver_video = _existing_file(
            args.driver_video,
            label="Driver video",
        )
        assert driver_video is not None
        driver_au = _run_extraction(
            driver_video,
            driver_au_root,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            force=args.force,
        )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)

    command: list[str] = [
        str(PYTHON),
        str(EVALUATOR),
        "--generated-au",
        str(generated_au),
        "--au-profile",
        str(au_profile),
        "--leakage-classifier",
        str(leakage_classifier),
        "--device",
        args.device,
        "--identity-threshold",
        str(args.identity_threshold),
        "--personal-au-threshold",
        str(args.personal_au_threshold),
        "--driver-expression-threshold",
        str(args.driver_expression_threshold),
        "--leakage-threshold",
        str(args.leakage_threshold),
    ]
    if driver_au is not None:
        command.extend(["--driver-au", str(driver_au)])
    if args.expected_class:
        command.extend(["--expected-class", args.expected_class])
    if args.generated_video:
        command.extend(["--generated-video", str(generated_video)])
    if target_video is not None:
        command.extend(["--target-video", str(target_video)])
    for target_image in target_images:
        if target_image is not None:
            command.extend(["--target-image", str(target_image)])
    if output is not None:
        command.extend(["--output", str(output)])

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "AU compliance evaluation failed after feature extraction."
        ) from exc
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
