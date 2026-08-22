"""One-click isolated PT v4.1 preparation, training, and evaluation."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from scripts.pt_training.run_wangxing_v4_pipeline import (
    _resolve_test_manifest,
)


def _safe_name(value: str) -> str:
    return (
        re.sub(r"[^A-Za-z0-9_.-]+", "_", value.replace("+", "x"))
        .strip("_")
        or "test"
    )


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_stage(
    index: int,
    total: int,
    name: str,
    command: list[str],
    report: dict[str, Any],
) -> None:
    print(f"[v4.1 stage {index}/{total}] START {name}", flush=True)
    started = time.monotonic()
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    duration = round(time.monotonic() - started, 2)
    report["stages"].append(
        {
            "name": name,
            "command": subprocess.list2cmdline(command),
            "returncode": int(result.returncode),
            "duration_seconds": duration,
            "status": "completed" if result.returncode == 0 else "failed",
        }
    )
    if result.returncode != 0:
        print(
            f"[v4.1 stage {index}/{total}] FAILED {name} "
            f"returncode={result.returncode}",
            file=sys.stderr,
            flush=True,
        )
        raise RuntimeError(f"Stage failed: {name}")
    print(
        f"[v4.1 stage {index}/{total}] DONE {name} "
        f"duration={duration:.1f}s",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wang Xing expression-only PT v4.1 pipeline."
    )
    parser.add_argument(
        "--base-manifest",
        default="outputs/vedio_pred/wangxing_v3_generalization_manifest_res1k.json",
    )
    parser.add_argument(
        "--v41-manifest",
        default="outputs/vedio_pred/wangxing_v41_expression_generalization_manifest_res1k.json",
    )
    parser.add_argument(
        "--augmentation-root",
        default="data/_aug/wangxing_v41_expression_photometric",
    )
    parser.add_argument(
        "--cache-dir",
        default="outputs/vedio_pred/cache_wangxing_v41_expression_res1k",
    )
    parser.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_v41_expression_res1k.pt",
    )
    parser.add_argument(
        "--train-metrics",
        default="outputs/vedio_pred/wangxing_v41_expression_metrics_res1k.json",
    )
    parser.add_argument(
        "--official-metrics",
        default="outputs/forensics/wangxing_v41_expression_official_holdout_metrics.json",
    )
    parser.add_argument(
        "--test-manifest-root",
        default="outputs/vedio_pred/wangxing_v41_expression_test_manifests",
    )
    parser.add_argument(
        "--report",
        default="outputs/vedio_pred/wangxing_v41_expression_pipeline_report.json",
    )
    parser.add_argument("--max-photometric-per-class", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--test-set",
        dest="test_sets",
        nargs=2,
        action="append",
        metavar=("NAME", "FOLDER_OR_MANIFEST"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    python = str(Path(sys.executable))
    test_specs = args.test_sets or [
        ("25+25", "data/test/single_video"),
        ("32+32", "data/test/wangxing_32x32"),
    ]
    generated_test_root = project_path(args.test_manifest_root)
    resolved_tests: list[tuple[str, Path, str]] = []
    for name, folder in test_specs:
        manifest = _resolve_test_manifest(
            name=name,
            folder_or_manifest=folder,
            generated_root=generated_test_root,
        )
        resolved_tests.append(
            (
                name,
                manifest,
                f"outputs/forensics/"
                f"wangxing_v41_expression_{_safe_name(name)}_metrics.json",
            )
        )

    commands = [
        [
            python,
            str(PROJECT_ROOT / "scripts/pt_training/prepare_wangxing_v4_photometric.py"),
            "--manifest",
            args.base_manifest,
            "--output-manifest",
            args.v41_manifest,
            "--output-root",
            args.augmentation_root,
            "--max-per-class",
            str(args.max_photometric_per_class),
            "--seed",
            str(args.seed),
        ],
        [
            python,
            str(PROJECT_ROOT / "scripts/pt_training/train_wangxing_v41_expression.py"),
            "train",
            "--manifest",
            args.v41_manifest,
            "--cache-dir",
            args.cache_dir,
            "--model-path",
            args.model_path,
            "--metrics-output",
            args.train_metrics,
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
    ]
    names = ["prepare_expression_photometric_data", "train_expression_v41"]
    if args.evaluate_only:
        commands = []
        names = []

    commands.append(
        [
            python,
            str(PROJECT_ROOT / "scripts/pt_training/train_wangxing_v41_expression.py"),
            "evaluate",
            "--holdout-manifest",
            "data/forensics/holdout_split.json",
            "--model-path",
            args.model_path,
            "--output",
            args.official_metrics,
        ]
    )
    names.append("evaluate_official_holdout")
    for name, manifest, output in resolved_tests:
        commands.append(
            [
                python,
                str(PROJECT_ROOT / "scripts/pt_training/train_wangxing_v41_expression.py"),
                "evaluate",
                "--holdout-manifest",
                str(manifest.relative_to(PROJECT_ROOT).as_posix()),
                "--model-path",
                args.model_path,
                "--output",
                output,
            ]
        )
        names.append(f"evaluate_{name}")

    report_path = project_path(args.report)
    report: dict[str, Any] = {
        "schema_version": "wangxing_v41_expression_pipeline_report_v1",
        "status": "running",
        "config": vars(args),
        "stages": [],
    }
    try:
        _write(report_path, report)
        total = len(commands)
        print(
            f"PT v4.1 pipeline: {total} stages, "
            f"device={args.device}, tests={len(resolved_tests)}",
            flush=True,
        )
        for index, (name, command) in enumerate(
            zip(names, commands),
            start=1,
        ):
            _run_stage(index, total, name, command, report)
            _write(report_path, report)
        report["status"] = "completed"
        report["finished_at"] = datetime.now(UTC).isoformat()
        _write(report_path, report)
        print(f"Pipeline completed. Report: {report_path}", flush=True)
        return 0
    except KeyboardInterrupt:
        report["status"] = "cancelled"
        report["error"] = "KeyboardInterrupt"
        _write(report_path, report)
        return 130
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["finished_at"] = datetime.now(UTC).isoformat()
        _write(report_path, report)
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
