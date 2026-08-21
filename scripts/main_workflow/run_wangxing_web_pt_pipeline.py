"""One-click Wang Xing web + PT final pipeline.

This is an orchestration entry point. It does not run automatically from the
coding agent because it includes profile fitting and GPU training.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "stage"


def _run(
    name: str,
    command: list[str],
    report: dict[str, Any],
    *,
    dry_run: bool,
) -> None:
    record: dict[str, Any] = {
        "name": name,
        "command": subprocess.list2cmdline(command),
        "status": "dry_run" if dry_run else "running",
        "started_at": datetime.now(UTC).isoformat(),
    }
    report["stages"].append(record)
    _write(_path(report["report"]), report)
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
        _write(_path(report["report"]), report)
        raise RuntimeError(f"Stage failed: {name} ({result.returncode})")
    record["status"] = "completed"
    _write(_path(report["report"]), report)


def _merge_exclusions(
    *,
    inputs: list[Path],
    output: Path,
) -> None:
    merged: dict[str, list[dict[str, str]]] = {
        "real": [],
        "seedance": [],
    }
    seen = {"real": set(), "seedance": set()}
    for path in inputs:
        payload = _load(path)
        for domain in ("real", "seedance"):
            for item in payload.get(domain, []):
                video = str(item.get("video", ""))
                if not video:
                    continue
                key = str(Path(video).resolve()).casefold()
                if key in seen[domain]:
                    continue
                seen[domain].add(key)
                merged[domain].append(
                    {
                        "video": video,
                        "au": str(item.get("au", "")),
                    }
                )
    _write(
        output,
        {
            "schema_version": "wangxing_web_v3_profile_exclusion_v1",
            "note": (
                "Union of official holdout, hard development, and final "
                "32+32 exclusions. Do not use final-test sources for fitting."
            ),
            "real": merged["real"],
            "seedance": merged["seedance"],
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-click Wang Xing web + PT training/evaluation pipeline."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report",
        default="outputs/vedio_pred/wangxing_web_pt_pipeline_report.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    python = str(sys.executable)
    report: dict[str, Any] = {
        "schema_version": "wangxing_web_pt_pipeline_v1",
        "report": args.report,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "config": vars(args),
        "stages": [],
    }
    report_path = _path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write(report_path, report)

    final_builder = [
        python,
        str(PROJECT_ROOT / "scripts/data_build/build_wangxing_32x32_final_test.py"),
    ]
    hard_builder = [
        python,
        str(PROJECT_ROOT / "scripts/data_build/build_wangxing_hard_dev_set.py"),
    ]
    _run("build_final_32x32", final_builder, report, dry_run=args.dry_run)
    _run("build_hard_dev", hard_builder, report, dry_run=args.dry_run)

    final_exclusion = _path(
        "data/forensics/wangxing_32x32_profile_exclusion.json"
    )
    hard_exclusion = _path(
        "data/forensics/wangxing_hard_dev_exclusion.json"
    )
    official_exclusion = _path("data/forensics/holdout_split.json")
    old_web_exclusion = _path(
        "data/forensics/web_forensics_v2_profile_exclusion.json"
    )
    merged_exclusion = _path(
        "data/forensics/wangxing_web_v3_profile_exclusion.json"
    )
    if not args.dry_run:
        _merge_exclusions(
            inputs=[
                final_exclusion,
                hard_exclusion,
                official_exclusion,
                old_web_exclusion,
            ],
            output=merged_exclusion,
        )
    else:
        print(f"[dry-run] merge exclusions -> {merged_exclusion}")

    forensics_profile = _path(
        "outputs/forensics/forensics_profiles_web_v3_test_excluded.json"
    )
    source_profile = _path(
        "outputs/forensics/wangxing_source_profile_web_v3_test_excluded.json"
    )
    expression_profile = _path(
        "outputs/forensics/wangxing_expression_profile_web_v3_test_excluded.json"
    )
    profile_command = [
        python,
        str(PROJECT_ROOT / "scripts/data_build/build_forensics_profiles.py"),
        "--real-au-root",
        "data/au/MD_CL",
        "--seedance-au-root",
        "data/au/WangXing_Seedance",
        "--real-video-root",
        "data/MD_CL",
        "--seedance-video-root",
        "data/WangXing_Seedance",
        "--output",
        _relative(forensics_profile),
        "--holdout-manifest",
        _relative(merged_exclusion),
        "--authenticity-calibrator",
        "outputs/forensics/forensics_authenticity_calibrator_web_v2.json",
        "--max-videos",
        "120",
        "--max-motion-videos",
        "120",
        "--min-landmark-ratio",
        "0.45",
        "--min-pose-ratio",
        "0.35",
    ]
    _run(
        "build_web_profiles_excluding_dev_and_final",
        profile_command,
        report,
        dry_run=args.dry_run,
    )

    source_command = [
        python,
        str(PROJECT_ROOT / "scripts/wangxing/train_wangxing_specialization.py"),
        "--skip-identity",
        "--au-root",
        "data/au/MD_CL",
        "--expression-output",
        _relative(expression_profile),
        "--source-profile-output",
        _relative(source_profile),
        "--seedance-label-manifest",
        "data/au/WangXing_Seedance/pseudo_expression_manifest.json",
        "--holdout-manifest",
        _relative(merged_exclusion),
        "--device",
        "cpu",
    ]
    _run(
        "build_web_source_and_expression_profiles",
        source_command,
        report,
        dry_run=args.dry_run,
    )

    fusion_head = _path(
        "outputs/forensics/web_forensics_fusion_v3_transition.json"
    )
    feature_cache = _path(
        "outputs/forensics/web_forensics_transition_feature_cache.npz"
    )
    fusion_train = [
        python,
        str(PROJECT_ROOT / "scripts/web_forensics/run_web_forensics_v2.py"),
        "train",
        "--transition-features",
        "--split-manifest",
        "outputs/vedio_pred/wangxing_dual_pt_split_res1k.json",
        "--profile-exclusion",
        _relative(merged_exclusion),
        "--forensics-profile",
        _relative(forensics_profile),
        "--source-profile",
        _relative(source_profile),
        "--fusion-head",
        _relative(fusion_head),
        "--feature-cache",
        _relative(feature_cache),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    _run(
        "train_web_transition_fusion",
        fusion_train,
        report,
        dry_run=args.dry_run,
    )

    hard_v2_output = _path(
        "outputs/forensics/wangxing_hard_dev_web_v3_results"
    )
    hard_eval = [
        python,
        str(PROJECT_ROOT / "scripts/web_forensics/run_web_forensics_v2.py"),
        "evaluate",
        "--manifest",
        "data/dev/wangxing_hard_cases/single_video/manifest.json",
        "--forensics-profile",
        _relative(forensics_profile),
        "--source-profile",
        _relative(source_profile),
        "--fusion-head",
        _relative(fusion_head),
        "--output-root",
        _relative(hard_v2_output),
        "--device",
        args.device,
        "--wangxing-device",
        args.device,
    ]
    _run(
        "evaluate_web_hard_dev",
        hard_eval,
        report,
        dry_run=args.dry_run,
    )

    policy = _path("outputs/forensics/web_authenticity_policy_web_v3_dev.json")
    policy_fit = [
        python,
        str(PROJECT_ROOT / "scripts/web_forensics/web_authenticity_policy.py"),
        "--results",
        _relative(hard_v2_output / "all_results.json"),
        "--output",
        _relative(policy),
    ]
    _run("fit_web_authenticity_policy", policy_fit, report, dry_run=args.dry_run)

    web_results: list[tuple[str, str, Path]] = [
        (
            "25x25",
            "data/test/single_video/manifest.json",
            _path("outputs/forensics/wangxing_web_v3_25x25"),
        ),
        (
            "32x32",
            "data/test/wangxing_32x32/single_video/manifest.json",
            _path("outputs/forensics/wangxing_web_v3_32x32"),
        ),
    ]
    for name, manifest, output in web_results:
        command = [
            python,
            str(PROJECT_ROOT / "scripts/web_forensics/run_web_forensics_v2.py"),
            "evaluate",
            "--manifest",
            manifest,
            "--forensics-profile",
            _relative(forensics_profile),
            "--source-profile",
            _relative(source_profile),
            "--fusion-head",
            _relative(fusion_head),
            "--authenticity-policy",
            _relative(policy),
            "--output-root",
            _relative(output),
            "--device",
            args.device,
            "--wangxing-device",
            args.device,
        ]
        _run(f"evaluate_web_{name}", command, report, dry_run=args.dry_run)

    pt_pipeline = [
        python,
        str(PROJECT_ROOT / "scripts/pt_training/run_wangxing_v4_pipeline.py"),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--source-profile",
        _relative(source_profile),
        "--forensics-profile",
        _relative(forensics_profile),
        "--test-set",
        "25+25",
        "data/test/single_video",
        "--test-set",
        "32+32",
        "data/test/wangxing_32x32",
    ]
    _run("train_and_evaluate_pt_v4", pt_pipeline, report, dry_run=args.dry_run)

    comparisons = [
        (
            "25x25",
            "data/test/single_video/manifest.json",
            "outputs/forensics/wangxing_web_v3_25x25/all_results.json",
            "outputs/forensics/wangxing_v4_face_25x25_metrics.json",
        ),
        (
            "32x32",
            "data/test/wangxing_32x32/single_video/manifest.json",
            "outputs/forensics/wangxing_web_v3_32x32/all_results.json",
            "outputs/forensics/wangxing_v4_face_32x32_metrics.json",
        ),
    ]
    for name, manifest, web_results, pt_metrics in comparisons:
        output = _path(
            f"outputs/forensics/wangxing_web_pt_comparison_{name}"
        )
        command = [
            python,
            str(PROJECT_ROOT / "scripts/web_forensics/compare_web_pt_fusion.py"),
            "--manifest",
            manifest,
            "--web-results",
            web_results,
            "--pt-metrics",
            pt_metrics,
            "--output",
            _relative(output),
        ]
        _run(f"compare_web_pt_{name}", command, report, dry_run=args.dry_run)

    report["status"] = "completed"
    report["finished_at"] = datetime.now(UTC).isoformat()
    _write(report_path, report)
    print(f"Pipeline completed. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
