"""One-click PT v4.3 candidate pipeline."""

from __future__ import annotations

import argparse
import json
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


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _safe_name(value: str) -> str:
    return value.replace("+", "x").replace(" ", "_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wang Xing expression/face-crop PT v4.3 pipeline."
    )
    parser.add_argument(
        "--base-manifest",
        default="outputs/vedio_pred/wangxing_v3_generalization_manifest_res1k.json",
    )
    parser.add_argument(
        "--v43-manifest",
        default="outputs/vedio_pred/wangxing_v43_expression_generalization_manifest_res1k.json",
    )
    parser.add_argument(
        "--augmentation-root",
        default="data/_aug/wangxing_v43_expression_photometric",
    )
    parser.add_argument(
        "--cache-dir",
        default="outputs/vedio_pred/cache_wangxing_v43_expression_res1k",
    )
    parser.add_argument(
        "--model-path",
        default="outputs/vedio_pred/models/wangxing_v43_expression_res1k.pt",
    )
    parser.add_argument(
        "--train-metrics",
        default="outputs/vedio_pred/wangxing_v43_expression_metrics_res1k.json",
    )
    parser.add_argument(
        "--official-metrics",
        default="outputs/forensics/wangxing_v43_expression_official_holdout_metrics.json",
    )
    parser.add_argument(
        "--test-manifest-root",
        default="outputs/vedio_pred/wangxing_v43_expression_test_manifests",
    )
    parser.add_argument(
        "--report",
        default="outputs/vedio_pred/wangxing_v43_expression_pipeline_report.json",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--test-set",
        dest="test_sets",
        nargs=2,
        action="append",
        metavar=("NAME", "FOLDER_OR_MANIFEST"),
    )
    parser.add_argument("--evaluate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    python = str(Path(sys.executable))
    test_specs = args.test_sets or [
        ("25+25", "data/test/single_video"),
        ("32+32", "data/test/wangxing_32x32"),
    ]
    test_root = project_path(args.test_manifest_root)
    tests: list[tuple[str, Path, str]] = []
    for name, folder in test_specs:
        tests.append(
            (
                name,
                _resolve_test_manifest(
                    name=name,
                    folder_or_manifest=folder,
                    generated_root=test_root,
                ),
                f"outputs/forensics/"
                f"wangxing_v43_expression_{_safe_name(name)}_metrics.json",
            )
        )
    trainer = str(
        PROJECT_ROOT / "scripts/pt_training/train_wangxing_v43_expression.py"
    )
    commands: list[tuple[str, list[str]]] = [
        (
            "prepare_expression_photometric_data",
            [
                python,
                str(
                    PROJECT_ROOT
                    / "scripts/pt_training/prepare_wangxing_v4_photometric.py"
                ),
                "--manifest",
                args.base_manifest,
                "--output-manifest",
                args.v43_manifest,
                "--output-root",
                args.augmentation_root,
                "--max-per-class",
                "120",
                "--seed",
                str(args.seed),
            ],
        ),
        (
            "train_expression_v43",
            [
                python,
                trainer,
                "train",
                "--manifest",
                args.v43_manifest,
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
        ),
        (
            "evaluate_official_holdout",
            [
                python,
                trainer,
                "evaluate",
                "--holdout-manifest",
                "data/forensics/holdout_split.json",
                "--model-path",
                args.model_path,
                "--output",
                args.official_metrics,
            ],
        ),
    ]
    if args.evaluate_only:
        commands = commands[2:]
    for name, manifest, output in tests:
        commands.append(
            (
                f"evaluate_{name}",
                [
                    python,
                    trainer,
                    "evaluate",
                    "--holdout-manifest",
                    manifest.relative_to(PROJECT_ROOT).as_posix(),
                    "--model-path",
                    args.model_path,
                    "--output",
                    output,
                ],
            )
        )
    report_path = project_path(args.report)
    report: dict[str, Any] = {
        "schema_version": "wangxing_v43_expression_pipeline_report_v1",
        "status": "running",
        "config": vars(args),
        "stages": [],
    }
    try:
        _write(report_path, report)
        print(f"PT v4.3 pipeline: {len(commands)} stages", flush=True)
        for index, (name, command) in enumerate(commands, start=1):
            print(
                f"[v4.3 stage {index}/{len(commands)}] START {name}",
                flush=True,
            )
            started = time.monotonic()
            result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            duration = round(time.monotonic() - started, 2)
            report["stages"].append(
                {
                    "name": name,
                    "returncode": int(result.returncode),
                    "duration_seconds": duration,
                    "status": (
                        "completed" if result.returncode == 0 else "failed"
                    ),
                }
            )
            _write(report_path, report)
            if result.returncode != 0:
                raise RuntimeError(f"Stage failed: {name}")
            print(
                f"[v4.3 stage {index}/{len(commands)}] DONE {name}",
                flush=True,
            )
        report["status"] = "completed"
        report["finished_at"] = datetime.now(UTC).isoformat()
        _write(report_path, report)
        print(f"Pipeline completed. Report: {report_path}")
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
