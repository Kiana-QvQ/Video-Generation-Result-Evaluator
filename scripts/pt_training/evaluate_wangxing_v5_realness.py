"""Evaluate V5.1 realness on ppt holdout and final binary test sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from wangxing_project.drive_head_v5 import load_drive_head
from wangxing_project.realness_v5 import load_calibrator
from wangxing_project.v51_runtime import (
    build_feature_row,
    collect_videos,
    extract_au_for_video,
    lexicographic_metrics,
    load_json,
    rank_metrics,
)


def _binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray(
        [int(row["label_generated"]) for row in rows],
        dtype=np.int32,
    )
    predictions = np.asarray(
        [int(row["v5"]["decision"] == "generated") for row in rows],
        dtype=np.int32,
    )
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return {
        "generated_recall": tp / (tp + fn) if tp + fn else None,
        "real_recall": tn / (tn + fp) if tn + fp else None,
        "overall_accuracy": (tp + tn) / len(labels) if len(labels) else None,
        "generated_precision": tp / (tp + fp) if tp + fp else None,
        "coverage": 1.0 if len(labels) else 0.0,
        "decision_flip_count": sum(
            not bool(row.get("decision_matches_v3", False))
            for row in rows
        ),
        "confusion": {
            "tp_generated": tp,
            "tn_real": tn,
            "fp_real_as_generated": fp,
            "fn_generated_as_real": fn,
        },
        "realness_mean_real": (
            float(np.mean([
                row["v5"]["s_realness"]
                for row in rows
                if int(row["label_generated"]) == 0
                and row["v5"].get("s_realness") is not None
            ]))
            if any(int(row["label_generated"]) == 0 for row in rows)
            else None
        ),
        "realness_mean_generated": (
            float(np.mean([
                row["v5"]["s_realness"]
                for row in rows
                if int(row["label_generated"]) == 1
                and row["v5"].get("s_realness") is not None
            ]))
            if any(int(row["label_generated"]) == 1 for row in rows)
            else None
        ),
        **lexicographic_metrics(rows),
    }


def _evaluate_group(
    *,
    ranking_root: Path,
    group: str,
    args: argparse.Namespace,
    v3_model: Path,
    drive_model: dict[str, Any] | None,
    source_profile: dict[str, Any],
    forensics_profile: dict[str, Any],
    calibrator: dict[str, Any] | None,
    cache_dir: Path,
    au_root: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, (video, label) in enumerate(
        collect_videos(ranking_root, group),
        start=1,
    ):
        print(
            f"[V5.1 holdout] {group} {index}/4 {video.name}",
            flush=True,
        )
        au = extract_au_for_video(
            video=video,
            au_output_root=au_root,
            cache_dir=cache_dir,
            device=args.wangxing_device,
        )
        rows.append(
            build_feature_row(
                video=video,
                label=label,
                group=group,
                au_path=au,
                v3_model=v3_model,
                drive_model=drive_model,
                drive_cache=cache_dir,
                source_profile=source_profile,
                forensics_profile=forensics_profile,
                device=args.device,
                wangxing_device=args.wangxing_device,
                calibrator=calibrator,
                realness_enabled=True,
            )
        )
    return {
        "metrics": rank_metrics(rows, min_pairwise=args.min_pairwise),
        "rows": rows,
    }


def _manifest_rows(
    *,
    manifest_path: Path,
    args: argparse.Namespace,
    v3_model: Path,
    drive_model: dict[str, Any] | None,
    source_profile: dict[str, Any],
    forensics_profile: dict[str, Any],
    calibrator: dict[str, Any] | None,
    cache_dir: Path,
    au_root: Path,
) -> list[dict[str, Any]]:
    payload = load_json(manifest_path)
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(payload.get("samples") or [], start=1):
        video = (manifest_path.parent / str(sample["video"])).resolve()
        au = (manifest_path.parent / str(sample["au"])).resolve()
        label = "real" if int(sample.get("label_generated", 0)) == 0 else "seedance"
        print(
            f"[V5.1 binary] {manifest_path.name} "
            f"{index}/{len(payload.get('samples') or [])}",
            flush=True,
        )
        row = build_feature_row(
            video=video,
            label=label,
            group=manifest_path.stem,
            au_path=au,
            v3_model=v3_model,
            drive_model=drive_model,
            drive_cache=cache_dir,
            source_profile=source_profile,
            forensics_profile=forensics_profile,
            device=args.device,
            wangxing_device=args.wangxing_device,
            calibrator=calibrator,
            realness_enabled=True,
        )
        row["sample_id"] = sample.get("sample_id")
        row["label_generated"] = int(sample.get("label_generated", 0))
        rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate V5.1 calibrator without fitting on holdout/test."
    )
    parser.add_argument(
        "--calibrator",
        default="outputs/forensics/wangxing_v5_realness_calibrator.json",
    )
    parser.add_argument(
        "--ranking-root",
        default=r"C:\Users\zhanghaotian\Desktop\ppt_video",
    )
    parser.add_argument("--holdout-group", default="test2")
    parser.add_argument(
        "--test-set",
        dest="test_sets",
        nargs=2,
        action="append",
        metavar=("NAME", "MANIFEST"),
        required=True,
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
        default="outputs/forensics/cache_wangxing_v5_1_ppt",
    )
    parser.add_argument(
        "--au-output-root",
        default="outputs/forensics/cache_wangxing_v5_1_ppt/au",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/vedio_pred/wangxing_v5_1_results",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wangxing-device", default="cuda")
    parser.add_argument(
        "--min-pairwise",
        type=float,
        default=5.0 / 6.0,
        help="Minimum holdout pairwise ordering rate (default 5/6).",
    )
    parser.add_argument(
        "--enforce-gates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero when holdout/binary gates fail.",
    )
    args = parser.parse_args(argv)

    calibrator = load_calibrator(project_path(args.calibrator))
    if calibrator is None:
        raise SystemExit(
            "V5.1 calibrator is missing or schema-invalid; run calibration first."
        )
    ranking_root = Path(args.ranking_root).expanduser().resolve()
    v3_model = project_path(args.v3_model)
    drive_model = load_drive_head(project_path(args.drive_model))
    source_profile = load_json(args.source_profile)
    forensics_profile = load_json(args.forensics_profile)
    cache_dir = project_path(args.cache_dir)
    au_root = project_path(args.au_output_root)
    output_root = project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    holdout = _evaluate_group(
        ranking_root=ranking_root,
        group=args.holdout_group,
        args=args,
        v3_model=v3_model,
        drive_model=drive_model,
        source_profile=source_profile,
        forensics_profile=forensics_profile,
        calibrator=calibrator,
        cache_dir=cache_dir,
        au_root=au_root,
    )
    test_payloads: dict[str, Any] = {}
    for name, manifest in args.test_sets:
        rows = _manifest_rows(
            manifest_path=project_path(manifest),
            args=args,
            v3_model=v3_model,
            drive_model=drive_model,
            source_profile=source_profile,
            forensics_profile=forensics_profile,
            calibrator=calibrator,
            cache_dir=cache_dir,
            au_root=au_root,
        )
        test_payloads[name] = {
            "metrics": _binary_metrics(rows),
            "rows": rows,
        }

    payload = {
        "schema_version": "wangxing_v5_1_realness_evaluation_v1",
        "decision_source": "v3_frozen",
        "calibrator": str(project_path(args.calibrator)),
        "development_only": True,
        "min_pairwise_threshold": float(args.min_pairwise),
        "enforce_gates": bool(args.enforce_gates),
        "holdout": holdout,
        "test_sets": test_payloads,
        "test_training_allowed": False,
    }
    gate_failures: list[str] = []
    holdout_metrics = holdout["metrics"]
    pairwise = holdout_metrics.get("pairwise_ordering_rate")
    if pairwise is not None and pairwise + 1e-9 < float(args.min_pairwise):
        gate_failures.append(
            "holdout pairwise "
            f"{pairwise:.4f} < threshold {args.min_pairwise:.4f}"
        )
    for name, test_payload in test_payloads.items():
        metrics = test_payload["metrics"]
        flip_count = int(metrics.get("decision_flip_count") or 0)
        if flip_count != 0:
            gate_failures.append(
                f"{name} decision_flip_count={flip_count} (expected 0)"
            )
        lex_ok = metrics.get("lexicographic_satisfied")
        if lex_ok is False:
            gate_failures.append(
                f"{name} lexicographic violated: "
                f"min_real={metrics.get('min_real_score_display')} "
                f"max_ai={metrics.get('max_ai_score_display')}"
            )
    payload["gate_failures"] = gate_failures
    payload["gates_passed"] = not gate_failures
    (output_root / "all_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "gates_passed": payload["gates_passed"],
            "gate_failures": gate_failures,
            "holdout": holdout["metrics"],
            "test_sets": {
                name: value["metrics"]
                for name, value in test_payloads.items()
            },
        },
        ensure_ascii=False,
        indent=2,
    ))
    print(f"All results: {output_root / 'all_results.json'}")
    if gate_failures and args.enforce_gates:
        raise SystemExit(
            "V5.1 evaluation gates failed:\n- "
            + "\n- ".join(gate_failures)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
