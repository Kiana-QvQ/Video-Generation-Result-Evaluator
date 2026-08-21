"""One-click v4 PT data preparation, training, and evaluation.

This script is an orchestration entry point only. The coding agent does not
run it automatically because it starts long-running data processing/training.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluator.modules.core.paths import project_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "vedio_pred"
    / "wangxing_v4_expression_pipeline_report.json"
)
DEFAULT_TEST_SETS = (
    ("25+25", "data/test/single_video"),
    ("32+32", "data/test/wangxing_32x32"),
)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _safe_name(value: str) -> str:
    value = value.replace("+", "x")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "test"


def _resolve_test_manifest(
    *,
    name: str,
    folder_or_manifest: str,
    generated_root: Path,
) -> Path:
    requested = project_path(folder_or_manifest)
    candidates = (
        [requested]
        if requested.is_file()
        else [
            requested / "pt_manifest.json",
            requested / "manifest.json",
            requested / "single_video" / "manifest.json",
        ]
    )
    manifest_path = next(
        (path for path in candidates if path.is_file()),
        None,
    )
    if manifest_path is None:
        raise FileNotFoundError(
            f"No manifest.json or pt_manifest.json found under "
            f"{requested}"
        )
    payload = _load_json(manifest_path)
    if "real" in payload and "seedance" in payload:
        return manifest_path
    samples = payload.get("samples") or []
    if not samples:
        raise ValueError(
            f"Unsupported test manifest format: {manifest_path}"
        )
    real: list[dict[str, str]] = []
    seedance: list[dict[str, str]] = []
    for sample in samples:
        video = sample.get("source_video")
        au = sample.get("source_au")
        if not video or not au:
            raise ValueError(
                f"Sample {sample.get('sample_id')} lacks source_video/source_au"
            )
        item = {"video": str(video), "au": str(au)}
        if sample.get("label") == "real":
            real.append(item)
        elif sample.get("label") == "ai":
            seedance.append(item)
        else:
            raise ValueError(
                f"Unsupported test label {sample.get('label')} in {manifest_path}"
            )
    generated_root.mkdir(parents=True, exist_ok=True)
    output = generated_root / f"{_safe_name(name)}_pt_manifest.json"
    _write(
        output,
        {
            "schema_version": "wangxing_v4_generated_test_manifest_v1",
            "source_manifest": str(manifest_path),
            "training_allowed": False,
            "real": real,
            "seedance": seedance,
        },
    )
    return output


def _run_stage(
    name: str,
    command: list[str],
    report: dict[str, Any],
    *,
    dry_run: bool,
) -> None:
    record = {
        "name": name,
        "command": subprocess.list2cmdline(command),
        "status": "dry_run" if dry_run else "running",
        "started_at": datetime.now(UTC).isoformat(),
    }
    report["stages"].append(record)
    if dry_run:
        print(f"[dry-run] {record['command']}")
        return
    started = time.monotonic()
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    record["duration_seconds"] = round(time.monotonic() - started, 2)
    record["returncode"] = int(result.returncode)
    record["finished_at"] = datetime.now(UTC).isoformat()
    if result.returncode != 0:
        record["status"] = "failed"
        raise RuntimeError(f"Stage failed: {name} ({result.returncode})")
    record["status"] = "completed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-click Wang Xing PT v4 pipeline."
    )
    parser.add_argument(
        "--base-manifest",
        default="outputs/vedio_pred/wangxing_v3_generalization_manifest_res1k.json",
    )
    parser.add_argument(
        "--v4-manifest",
        default="outputs/vedio_pred/wangxing_v4_generalization_manifest_res1k.json",
    )
    parser.add_argument(
        "--augmentation-root",
        default="data/_aug/wangxing_v4_photometric",
    )
    parser.add_argument(
        "--cache-dir",
        default="outputs/vedio_pred/cache_wangxing_v4_expression_res1k",
    )
    parser.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_v4_expression_res1k.pt",
    )
    parser.add_argument(
        "--train-metrics",
        default="outputs/vedio_pred/wangxing_v4_expression_metrics_res1k.json",
    )
    parser.add_argument(
        "--official-metrics",
        default="outputs/forensics/wangxing_v4_expression_official_holdout_metrics.json",
    )
    parser.add_argument(
        "--test-set",
        dest="test_sets",
        nargs=2,
        action="append",
        metavar=("NAME", "FOLDER_OR_MANIFEST"),
        help=(
            "Repeatable test set: --test-set 25+25 data/test/single_video. "
            "The folder may contain manifest.json or pt_manifest.json."
        ),
    )
    parser.add_argument(
        "--final-manifest",
        default=None,
        help="Legacy single-test alias; prefer --test-set.",
    )
    parser.add_argument(
        "--final-metrics",
        default=None,
        help="Legacy output alias for --final-manifest.",
    )
    parser.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_holdout_excluded.json",
    )
    parser.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles.json",
    )
    parser.add_argument("--max-photometric-per-class", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--report",
        default="outputs/vedio_pred/wangxing_v4_expression_pipeline_report.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = project_path(args.report)
    report: dict[str, Any] = {
        "schema_version": "wangxing_v4_expression_pipeline_report_v1",
        "status": "dry_run" if args.dry_run else "running",
        "started_at": datetime.now(UTC).isoformat(),
        "config": vars(args),
        "stages": [],
    }

    python = str(Path(sys.executable))
    test_specs = args.test_sets
    if not test_specs:
        if args.final_manifest:
            test_specs = [
                (
                    "final",
                    args.final_manifest,
                )
            ]
        else:
            test_specs = list(DEFAULT_TEST_SETS)
    generated_test_root = (
        PROJECT_ROOT
        / "outputs"
        / "vedio_pred"
        / "wangxing_v4_test_manifests"
    )
    resolved_tests: list[tuple[str, Path, str]] = []
    for name, folder_or_manifest in test_specs:
        manifest_path = _resolve_test_manifest(
            name=name,
            folder_or_manifest=folder_or_manifest,
            generated_root=generated_test_root,
        )
        if args.final_metrics and len(test_specs) == 1 and name == "final":
            output_path = args.final_metrics
        else:
            output_path = (
                f"outputs/forensics/"
                f"wangxing_v4_expression_{_safe_name(name)}_metrics.json"
            )
        resolved_tests.append((name, manifest_path, output_path))

    commands = [
        (
            "prepare_photometric_train_data",
            [
                python,
                str(PROJECT_ROOT / "scripts/pt_training/prepare_wangxing_v4_photometric.py"),
                "--manifest",
                args.base_manifest,
                "--output-manifest",
                args.v4_manifest,
                "--output-root",
                args.augmentation_root,
                "--max-per-class",
                str(args.max_photometric_per_class),
                "--seed",
                str(args.seed),
            ],
        ),
        (
            "train_v4_face_geometry",
            [
                python,
                str(PROJECT_ROOT / "scripts/pt_training/train_wangxing_v4_face.py"),
                "train",
                "--manifest",
                args.v4_manifest,
                "--cache-dir",
                args.cache_dir,
                "--model-path",
                args.model_path,
                "--metrics-output",
                args.train_metrics,
                "--source-profile",
                args.source_profile,
                "--forensics-profile",
                args.forensics_profile,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
            ],
        ),
        (
            "evaluate_official_holdout",
            [
                python,
                str(PROJECT_ROOT / "scripts/pt_training/train_wangxing_v4_face.py"),
                "evaluate",
                "--holdout-manifest",
                "data/forensics/holdout_split.json",
                "--model-path",
                args.model_path,
                "--source-profile",
                args.source_profile,
                "--forensics-profile",
                args.forensics_profile,
                "--output",
                args.official_metrics,
            ],
        ),
    ]
    for name, manifest_path, output_path in resolved_tests:
        commands.append(
            (
                f"evaluate_{name}",
                [
                    python,
                    str(
                        PROJECT_ROOT
                        / "scripts/pt_training/train_wangxing_v4_face.py"
                    ),
                    "evaluate",
                    "--holdout-manifest",
                    _relative(manifest_path),
                    "--model-path",
                    args.model_path,
                    "--source-profile",
                    args.source_profile,
                    "--forensics-profile",
                    args.forensics_profile,
                    "--output",
                    output_path,
                ],
            )
        )
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write(report_path, report)
        for name, command in commands:
            _run_stage(name, command, report, dry_run=args.dry_run)
            _write(report_path, report)
        report["status"] = "dry_run" if args.dry_run else "completed"
        report["finished_at"] = datetime.now(UTC).isoformat()
        _write(report_path, report)
        print(f"Pipeline completed. Report: {report_path}")
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["finished_at"] = datetime.now(UTC).isoformat()
        _write(report_path, report)
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        print(f"Report: {report_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
