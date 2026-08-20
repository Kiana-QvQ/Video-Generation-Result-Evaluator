from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)


def _emit_stage(stage: str, progress: float, message: str) -> None:
    print(
        f"AU_STAGE|{stage}|{progress:.3f}|{message}",
        flush=True,
    )


def _run_step(label: str, command: Sequence[str]) -> None:
    print(f"AU_STEP|{label}", flush=True)
    print(f"AU_COMMAND|{' '.join(str(part) for part in command)}", flush=True)
    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip("\r\n"), flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"{label} failed with exit code {return_code}."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete bounded AU leakage training pipeline."
    )
    parser.add_argument(
        "--negative-dataset",
        choices=("RAVDESS", "MetaHuman"),
        default="RAVDESS",
    )
    parser.add_argument("--metahuman-archive", default="")
    parser.add_argument("--metahuman-url", default="")
    parser.add_argument("--ravdess-actors", default="1,2")
    parser.add_argument(
        "--ravdess-source",
        choices=("ZENODO", "HUGGINGFACE"),
        default="ZENODO",
    )
    parser.add_argument(
        "--ravdess-emotions",
        default="1,2,3,4,5,6,7,8",
    )
    parser.add_argument("--ravdess-cache-root", default="data/cache/ravdess")
    parser.add_argument("--max-negative-videos", type=int, default=48)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cuda",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-negative-preparation", action="store_true")
    parser.add_argument("--force-au-extraction", action="store_true")
    parser.add_argument(
        "--original-au-root",
        default="data/au/MD_CL",
        help="Original AU CSV root used for general emotion classification.",
    )
    parser.add_argument(
        "--original-video-root",
        default="data/MD_CL",
        help="Original video root used to verify AU extraction completeness.",
    )
    parser.add_argument(
        "--emotion-profile-output",
        default="data/au/original_emotion_au_profile.json",
    )
    parser.add_argument("--emotion-min-samples-per-class", type=int, default=3)
    return parser


def _prepare_negative(args: argparse.Namespace) -> Path:
    if args.negative_dataset == "RAVDESS":
        _emit_stage(
            "negative_data",
            0.08,
            "Downloading and sampling RAVDESS negative videos",
        )
        _run_step(
            "RAVDESS negative preparation",
            [
                PYTHON,
                PROJECT_ROOT / "scripts/au/download_ravdess_negative.py",
                "--actors",
                args.ravdess_actors,
                "--source",
                args.ravdess_source,
                "--emotions",
                args.ravdess_emotions,
                "--max-videos",
                str(args.max_negative_videos),
                "--output-root",
                "data/negative/ravdess",
                "--cache-root",
                args.ravdess_cache_root,
            ],
        )
        return PROJECT_ROOT / "data/negative/ravdess/negative_manifest.json"

    if not args.metahuman_archive and not args.metahuman_url:
        raise ValueError(
            "MetaHuman requires --metahuman-archive or --metahuman-url."
        )
    _emit_stage(
        "negative_data",
        0.08,
        "Preparing the approved Synthesized MetaHuman subset",
    )
    command: list[object] = [
        PYTHON,
                PROJECT_ROOT
                / "scripts/archive/prepare_synthesized_metahuman_negative.py",
        "--output-root",
        "data/negative/synthesized_metahuman",
        "--max-videos",
        str(args.max_negative_videos),
    ]
    if args.metahuman_archive:
        command.extend(["--archive", args.metahuman_archive])
    else:
        command.extend(["--url", args.metahuman_url])
    _run_step("Synthesized MetaHuman preparation", command)
    return (
        PROJECT_ROOT
        / "data/negative/synthesized_metahuman/negative_manifest.json"
    )


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    if args.max_negative_videos <= 0:
        raise ValueError("--max-negative-videos must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.emotion_min_samples_per_class <= 0:
        raise ValueError("--emotion-min-samples-per-class must be positive.")

    if args.skip_negative_preparation:
        negative_manifest = (
            PROJECT_ROOT
            / (
                "data/negative/ravdess/negative_manifest.json"
                if args.negative_dataset == "RAVDESS"
                else (
                    "data/negative/synthesized_metahuman/"
                    "negative_manifest.json"
                )
            )
        )
    else:
        negative_manifest = _prepare_negative(args)

    if not negative_manifest.is_file():
        raise FileNotFoundError(
            f"Negative manifest was not found: {negative_manifest}"
        )

    force_args = ["--force"] if args.force_au_extraction else []
    _emit_stage("target_au", 0.28, "Extracting target AU sequences")
    _run_step(
        "Wang Xing AU extraction",
        [
            PYTHON,
            PROJECT_ROOT / "scripts/au/extract_libreface_au.py",
            "--input-root",
            "data/MD_CL",
            "--output-root",
            "data/au/MD_CL",
            "--exclude-dir",
            "CL_FACS*",
            "--exclude-dir",
            "CL_HeadMove",
            "--continue-on-error",
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            *force_args,
        ],
    )

    _emit_stage("negative_au", 0.55, "Extracting negative AU sequences")
    _run_step(
        "Negative AU extraction",
        [
            PYTHON,
            PROJECT_ROOT / "scripts/au/extract_libreface_au.py",
            "--manifest",
            negative_manifest,
            "--output-root",
            "data/au/negative",
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            *force_args,
        ],
    )

    _emit_stage("profile", 0.76, "Building the target AU profile")
    _run_step(
        "Wang Xing AU profile",
        [
            PYTHON,
                PROJECT_ROOT
                / "scripts"
                / "数据构建"
                / "build_au_profile.py",
            "--au-root",
            "data/au/MD_CL",
            "--input-root",
            "data/au/MD_CL",
            "--video-root",
            "data/MD_CL",
            "--output",
            "data/au/wangxing_au_profile.json",
        ],
    )

    original_au_root = PROJECT_ROOT / args.original_au_root
    if original_au_root.is_dir() and any(original_au_root.rglob("*.csv")):
        _emit_stage(
            "emotion_profile",
            0.84,
            "Building the original AU emotion profile",
        )
        _run_step(
            "Original AU emotion profile",
            [
                PYTHON,
                PROJECT_ROOT
                / "scripts"
                / "数据构建"
                / "build_original_emotion_au_profile.py",
                "--au-root",
                args.original_au_root,
                "--video-root",
                args.original_video_root,
                "--output",
                args.emotion_profile_output,
                "--min-samples-per-class",
                str(args.emotion_min_samples_per_class),
            ],
        )
    else:
        print(
            "AU_WARNING|Original AU emotion root is missing or has no CSV files; "
            "automatic emotion classification will remain unavailable.",
            flush=True,
        )

    _emit_stage("classifier", 0.90, "Fitting the AU leakage classifier")
    _run_step(
        "AU leakage classifier",
        [
            PYTHON,
            PROJECT_ROOT / "scripts/au/fit_au_leakage_classifier.py",
            "--positive-root",
            "data/au/MD_CL",
            "--negative-root",
            "data/au/negative",
            "--output",
            "data/au/au_leakage_classifier.json",
        ],
    )
    _emit_stage("completed", 1.0, "AU training completed")
    return {
        "status": "completed",
        "negative_manifest": str(negative_manifest),
        "au_profile": str(
            PROJECT_ROOT / "data/au/wangxing_au_profile.json"
        ),
        "leakage_classifier": str(
            PROJECT_ROOT / "data/au/au_leakage_classifier.json"
        ),
        "emotion_profile": str(
            PROJECT_ROOT / args.emotion_profile_output
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = run_pipeline(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
