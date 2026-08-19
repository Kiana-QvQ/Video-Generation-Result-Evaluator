"""Run the complete Wang Xing temporal v3 data/train/evaluation pipeline.

The pipeline intentionally builds a fresh v3-only base split so stale absolute
paths from older workspace locations cannot enter the new manifest.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BASE_MANIFEST = (
    PROJECT_ROOT / "outputs" / "vedio_pred" / "wangxing_v3_base_split.json"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "vedio_pred"
    / "wangxing_v3_generalization_manifest_res1k.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "_aug" / "wangxing_v3_generalization"
)
DEFAULT_CACHE_DIR = (
    PROJECT_ROOT / "outputs" / "vedio_pred" / "cache_wangxing_v3_res1k"
)
DEFAULT_MODEL = (
    PROJECT_ROOT / "outputs" / "vedio_pred" / "models" / "wangxing_v3_res1k.pt"
)
DEFAULT_TRAIN_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "vedio_pred"
    / "wangxing_v3_holdout_metrics_res1k.json"
)
DEFAULT_OFFICIAL_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "forensics"
    / "wangxing_v3_official_holdout_metrics.json"
)
DEFAULT_CHANGE_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "forensics"
    / "wangxing_v3_test_AI_metrics.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "vedio_pred"
    / "wangxing_v3_pipeline_report.json"
)
SOURCE_PROFILE = (
    PROJECT_ROOT
    / "outputs"
    / "forensics"
    / "wangxing_source_profile_holdout_excluded.json"
)
FORENSICS_PROFILE = (
    PROJECT_ROOT / "outputs" / "forensics" / "forensics_profiles.json"
)
HOLDOUT_MANIFEST = PROJECT_ROOT / "data" / "forensics" / "holdout_split.json"
CHANGE_MANIFEST = PROJECT_ROOT / "data" / "forensics" / "holdout_test_AI.json"


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(item) for item in command])


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _ensure_prerequisites() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install it or add it to PATH."
        )
    _require_file(HOLDOUT_MANIFEST, "Official holdout manifest")
    _require_file(SOURCE_PROFILE, "Wang Xing source profile")
    _require_file(FORENSICS_PROFILE, "Forensics profile")
    if not CHANGE_MANIFEST.is_file():
        from scripts.prepare_res1k_au_pt_training import build_test_ai_holdout

        _write_json(CHANGE_MANIFEST, build_test_ai_holdout())


def _run_stage(
    *,
    name: str,
    command: list[str],
    report: dict[str, Any],
) -> None:
    started = time.monotonic()
    record: dict[str, Any] = {
        "name": name,
        "command": _command_text(command),
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
    }
    report["stages"].append(record)
    _write_json(_path(report["report_path"]), report)
    print(f"\n===== {name} =====", flush=True)
    print(record["command"], flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    record["duration_seconds"] = round(time.monotonic() - started, 2)
    record["returncode"] = int(completed.returncode)
    record["finished_at"] = datetime.now(UTC).isoformat()
    if completed.returncode != 0:
        record["status"] = "failed"
        _write_json(_path(report["report_path"]), report)
        raise RuntimeError(
            f"Stage failed: {name} (exit {completed.returncode})"
        )
    record["status"] = "completed"
    _write_json(_path(report["report_path"]), report)


def _validate_base_split(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    for section in ("train", "test"):
        if not isinstance(payload.get(section), dict):
            raise TypeError(f"Base split missing section: {section}")
        for label in ("real", "fake"):
            values = payload[section].get(label)
            if not isinstance(values, list) or not values:
                raise ValueError(f"Base split missing {section}.{label}")
            for value in values:
                if not _path(value).is_file():
                    raise FileNotFoundError(
                        f"Base split video does not exist: {_path(value)}"
                    )
    return payload


def _iter_manifest_pairs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = payload.get("pairs") or {}
    result: list[dict[str, Any]] = []
    for section in ("train", "test"):
        for label in ("real", "fake"):
            values = (pairs.get(section) or {}).get(label) or []
            result.extend(values)
    return result


def _validate_v3_manifest(
    path: Path,
    *,
    base_payload: dict[str, Any],
    max_pseudo_fakes: int,
    max_media_per_class: int,
    allow_partial_media: bool,
) -> dict[str, Any]:
    payload = _load_json(path)
    pairs = _iter_manifest_pairs(payload)
    if not pairs:
        raise ValueError("v3 manifest contains no pairs")

    train = payload["pairs"]["train"]
    test = payload["pairs"]["test"]
    original_real = len(base_payload["train"]["real"])
    original_fake = len(base_payload["train"]["fake"])
    expected_media = min(max_media_per_class, original_real) + min(
        max_media_per_class,
        original_fake,
    )
    expected_pseudo = min(max_pseudo_fakes, original_real)
    counts = payload.get("counts") or {}
    if len(test["real"]) != len(base_payload["test"]["real"]):
        raise ValueError("v3 real holdout count does not match base split")
    if len(test["fake"]) != len(base_payload["test"]["fake"]):
        raise ValueError("v3 fake holdout count does not match base split")
    if int(counts.get("pseudo_fake", -1)) != expected_pseudo:
        raise ValueError(
            "Pseudo-fake generation is incomplete: "
            f"expected {expected_pseudo}, got {counts.get('pseudo_fake')}"
        )
    actual_media = int(counts.get("media_variants", -1))
    if not allow_partial_media and actual_media != expected_media:
        raise ValueError(
            "Media augmentation is incomplete: "
            f"expected {expected_media}, got {actual_media}"
        )

    forbidden_marker = (PROJECT_ROOT / "data" / "test" / "AI").resolve()
    for item in pairs:
        video_value = str(item.get("video", ""))
        au_value = str(item.get("au", ""))
        video = _path(video_value).resolve()
        au = _path(au_value).resolve()
        if Path(video_value).is_absolute() or Path(au_value).is_absolute():
            raise ValueError("v3 manifest must use project-relative paths")
        if not video.is_file():
            raise FileNotFoundError(f"v3 video missing: {video}")
        if not au.is_file():
            raise FileNotFoundError(f"v3 AU CSV missing: {au}")
        if (
            forbidden_marker == video or forbidden_marker in video.parents
        ) and (item in train["real"] or item in train["fake"]):
            raise ValueError(f"Change clip entered training: {video}")

    expected_train_real = original_real + min(max_media_per_class, original_real)
    expected_train_fake = (
        original_fake
        + min(max_media_per_class, original_fake)
        + expected_pseudo
    )
    if len(train["real"]) != expected_train_real:
        raise ValueError(
            f"Unexpected real train count: expected {expected_train_real}, "
            f"got {len(train['real'])}"
        )
    if len(train["fake"]) != expected_train_fake:
        raise ValueError(
            f"Unexpected fake train count: expected {expected_train_fake}, "
            f"got {len(train['fake'])}"
        )
    return {
        "counts": counts,
        "expected_media_variants": expected_media,
        "expected_pseudo_fakes": expected_pseudo,
        "train_real": len(train["real"]),
        "train_fake": len(train["fake"]),
        "test_real": len(test["real"]),
        "test_fake": len(test["fake"]),
    }


def _add_common_profiles(command: list[str]) -> None:
    command.extend(
        [
            "--source-profile",
            _relative(SOURCE_PROFILE),
            "--forensics-profile",
            _relative(FORENSICS_PROFILE),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-click Wang Xing v3 data preparation, training, and testing."
        )
    )
    parser.add_argument("--base-manifest", default=_relative(DEFAULT_BASE_MANIFEST))
    parser.add_argument("--manifest", default=_relative(DEFAULT_MANIFEST))
    parser.add_argument("--output-root", default=_relative(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cache-dir", default=_relative(DEFAULT_CACHE_DIR))
    parser.add_argument("--model-path", default=_relative(DEFAULT_MODEL))
    parser.add_argument(
        "--train-metrics",
        default=_relative(DEFAULT_TRAIN_METRICS),
    )
    parser.add_argument(
        "--official-metrics",
        default=_relative(DEFAULT_OFFICIAL_METRICS),
    )
    parser.add_argument(
        "--change-metrics",
        default=_relative(DEFAULT_CHANGE_METRICS),
    )
    parser.add_argument("--report", default=_relative(DEFAULT_REPORT))
    parser.add_argument("--max-pseudo-fakes", type=int, default=120)
    parser.add_argument("--max-media-per-class", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modality-dropout", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-partial-media",
        action="store_true",
        help="Do not fail when some FFmpeg media variants are missing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pipeline commands without running them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = _path(args.report)
    report: dict[str, Any] = {
        "schema_version": "wangxing_v3_pipeline_report_v1",
        "report_path": _relative(report_path),
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "config": {
            key: value
            for key, value in vars(args).items()
            if key != "dry_run"
        },
        "stages": [],
    }
    _write_json(report_path, report)

    base_manifest = _path(args.base_manifest)
    manifest = _path(args.manifest)
    output_root = _path(args.output_root)
    model_path = _path(args.model_path)
    train_metrics = _path(args.train_metrics)
    official_metrics = _path(args.official_metrics)
    change_metrics = _path(args.change_metrics)

    base_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_wangxing_video_pt.py"),
        "build-split",
        "--real-root",
        "data/MD_CL",
        "--fake-root",
        "data/WangXing_Seedance",
        "--holdout-manifest",
        _relative(HOLDOUT_MANIFEST),
        "--real-train-count",
        "120",
        "--seed",
        str(args.seed),
        "--output",
        _relative(base_manifest),
    ]
    prepare_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "prepare_wangxing_v3_generalization.py"),
        "--manifest",
        _relative(base_manifest),
        "--output-manifest",
        _relative(manifest),
        "--output-root",
        _relative(output_root),
        "--max-pseudo-fakes",
        str(args.max_pseudo_fakes),
        "--max-media-per-class",
        str(args.max_media_per_class),
        "--seed",
        str(args.seed),
    ]
    train_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_wangxing_v3_generalization.py"),
        "train",
        "--manifest",
        _relative(manifest),
        "--cache-dir",
        _relative(_path(args.cache_dir)),
        "--model-path",
        _relative(model_path),
        "--metrics-output",
        _relative(train_metrics),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--modality-dropout",
        str(args.modality_dropout),
        "--device",
        str(args.device),
    ]
    _add_common_profiles(train_command)
    official_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_wangxing_v3_generalization.py"),
        "evaluate",
        "--holdout-manifest",
        _relative(HOLDOUT_MANIFEST),
        "--model-path",
        _relative(model_path),
        "--output",
        _relative(official_metrics),
    ]
    _add_common_profiles(official_command)
    change_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_wangxing_v3_generalization.py"),
        "evaluate",
        "--holdout-manifest",
        _relative(CHANGE_MANIFEST),
        "--model-path",
        _relative(model_path),
        "--output",
        _relative(change_metrics),
    ]
    _add_common_profiles(change_command)

    commands = [
        ("build_current_base_split", base_command),
        ("prepare_v3_dataset", prepare_command),
        ("train_v3", train_command),
        ("evaluate_official_holdout", official_command),
        ("evaluate_change_test", change_command),
    ]
    try:
        if not args.dry_run:
            _ensure_prerequisites()
        for name, command in commands:
            if args.dry_run:
                print(f"[dry-run] {name}: {_command_text(command)}")
                continue
            _run_stage(name=name, command=command, report=report)
            if name == "build_current_base_split":
                base_payload = _validate_base_split(base_manifest)
                report["base_split"] = {
                    "counts": base_payload.get("counts"),
                    "path": _relative(base_manifest),
                }
            elif name == "prepare_v3_dataset":
                report["dataset_validation"] = _validate_v3_manifest(
                    manifest,
                    base_payload=base_payload,
                    max_pseudo_fakes=args.max_pseudo_fakes,
                    max_media_per_class=args.max_media_per_class,
                    allow_partial_media=args.allow_partial_media,
                )
            elif name == "train_v3":
                report["train_result"] = _load_json(train_metrics)
            elif name == "evaluate_official_holdout":
                report["official_result"] = _load_json(official_metrics)
            elif name == "evaluate_change_test":
                report["change_result"] = _load_json(change_metrics)
        report["status"] = "completed"
        report["finished_at"] = datetime.now(UTC).isoformat()
        _write_json(report_path, report)
        print(f"\nPipeline completed. Report: {report_path}")
        return 0
    except Exception as exc:  # noqa: BLE001 - persist failure in report
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["finished_at"] = datetime.now(UTC).isoformat()
        _write_json(report_path, report)
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        print(f"Report: {report_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
