"""Evaluate V5.2 RankHead on holdout and binary regression sets."""

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
from wangxing_project.cascade_v5 import cascade_score_v52
from wangxing_project.drive_head_v5 import load_drive_head
from wangxing_project.rank_head_v52 import (
    load_rank_policy_v52,
    predict_rank_score,
    rank_metrics,
    resolve_disabled_reason,
)
from wangxing_project.realness_v5 import load_calibrator
from wangxing_project.v51_runtime import (
    build_feature_row,
    extract_au_for_video,
    load_json,
)

# V5.1 offline baseline floors; V5.2 must not regress decision quality.
BINARY_ACCURACY_FLOORS = {
    "25+25": 0.98,
    "32+32": 1.0,
}


def _build_context(args: argparse.Namespace) -> dict[str, Any]:
    calibrator = load_calibrator(project_path(args.calibrator))
    if calibrator is None:
        raise SystemExit("V5.1 calibrator is missing or schema-invalid.")
    return {
        "profiles": load_json(args.forensics_profile),
        "source_profile": load_json(args.source_profile),
        "calibrator": calibrator,
        "v3_model": project_path(args.v3_model),
        "drive_model": load_drive_head(project_path(args.drive_model)),
        "cache_dir": project_path(args.cache_dir),
        "au_root": project_path(args.au_output_root),
        "rank_policy": load_rank_policy_v52(
            project_path(args.rank_policy)
        ),
    }


def _ranking_rows(
    *,
    manifest: dict[str, Any],
    split: str,
    args: argparse.Namespace,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in manifest.get("groups") or []:
        if str(group.get("split")) != split:
            continue
        group_id = str(group["group_id"])
        for label, video_value in (group.get("videos") or {}).items():
            if not video_value:
                continue
            video = Path(str(video_value)).expanduser().resolve()
            au = extract_au_for_video(
                video=video,
                au_output_root=context["au_root"],
                cache_dir=context["cache_dir"],
                device=args.wangxing_device,
            )
            row = build_feature_row(
                video=video,
                label=label,
                group=group_id,
                au_path=au,
                v3_model=context["v3_model"],
                drive_model=context["drive_model"],
                drive_cache=context["cache_dir"],
                source_profile=context["source_profile"],
                forensics_profile=context["profiles"],
                device=args.device,
                wangxing_device=args.wangxing_device,
                calibrator=context["calibrator"],
                realness_enabled=True,
            )
            row["group_id"] = group_id
            rank_score, rank_status = predict_rank_score(
                row,
                context["rank_policy"],
            )
            row["rank_prediction"] = rank_status
            row["v5"] = cascade_score_v52(
                p_v3_real=row["v5"]["p_v3_real"],
                p_drive=row["v5"].get("p_drive"),
                p_drive_eff=row["v5"].get("p_drive_eff"),
                realness=row["realness"],
                rank_score=rank_score,
                rank_policy=context["rank_policy"],
                realness_enabled=True,
                rank_enabled=True,
                prior_conflict=bool(row.get("prior_conflict")),
                group_id=group_id,
            )
            row["decision_matches_v3"] = (
                row["v5"]["decision"] == row["v3"]["prediction"]
            )
            rows.append(row)
    return rows


def _binary_rows(
    manifest_path: Path,
    args: argparse.Namespace,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    manifest = load_json(manifest_path)
    rows: list[dict[str, Any]] = []
    for sample in manifest.get("samples") or []:
        video = (manifest_path.parent / str(sample["video"])).resolve()
        au = (manifest_path.parent / str(sample["au"])).resolve()
        label = "real" if int(sample.get("label_generated", 0)) == 0 else "seedance"
        row = build_feature_row(
            video=video,
            label=label,
            group=manifest_path.stem,
            au_path=au,
            v3_model=context["v3_model"],
            drive_model=context["drive_model"],
            drive_cache=context["cache_dir"],
            source_profile=context["source_profile"],
            forensics_profile=context["profiles"],
            device=args.device,
            wangxing_device=args.wangxing_device,
            calibrator=context["calibrator"],
            realness_enabled=True,
        )
        row["sample_id"] = sample.get("sample_id")
        row["label_generated"] = int(sample.get("label_generated", 0))
        rank_score, rank_status = predict_rank_score(
            row,
            context["rank_policy"],
        )
        row["rank_prediction"] = rank_status
        row["v5"] = cascade_score_v52(
            p_v3_real=row["v5"]["p_v3_real"],
            p_drive=row["v5"].get("p_drive"),
            p_drive_eff=row["v5"].get("p_drive_eff"),
            realness=row["realness"],
            rank_score=rank_score,
            rank_policy=context["rank_policy"],
            realness_enabled=True,
            rank_enabled=True,
            prior_conflict=bool(row.get("prior_conflict")),
            group_id=None,
        )
        row["decision_matches_v3"] = (
            row["v5"]["decision"] == row["v3"]["prediction"]
        )
        rows.append(row)
    return rows


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
        "lexicographic": _lexicographic(rows),
        "confusion": {
            "tp_generated": tp,
            "tn_real": tn,
            "fp_real_as_generated": fp,
            "fn_generated_as_real": fn,
        },
    }


def _lexicographic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    real_scores = [
        row["v5"]["score_display"]
        for row in rows
        if row["v5"]["decision"] == "real"
    ]
    ai_scores = [
        row["v5"]["score_display"]
        for row in rows
        if row["v5"]["decision"] == "generated"
    ]
    return {
        "lexicographic_satisfied": (
            min(real_scores) > max(ai_scores)
            if real_scores and ai_scores
            else None
        ),
        "min_real_score_display": min(real_scores) if real_scores else None,
        "max_ai_score_display": max(ai_scores) if ai_scores else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate V5.2 rank policy and final binary regressions."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rank-policy", required=True)
    parser.add_argument(
        "--calibrator",
        default="outputs/forensics/wangxing_v5_realness_calibrator.json",
    )
    parser.add_argument("--test-set", dest="test_sets", nargs=2, action="append", required=True)
    parser.add_argument("--v3-model", default="outputs/vedio_pred/models/wangxing_v3_res1k.pt")
    parser.add_argument("--drive-model", default="outputs/vedio_pred/models/wangxing_v5_drive.json")
    parser.add_argument("--forensics-profile", default="outputs/forensics/forensics_profiles_web_v3_test_excluded.json")
    parser.add_argument("--source-profile", default="outputs/forensics/wangxing_source_profile_web_v3_test_excluded.json")
    parser.add_argument("--cache-dir", default="outputs/forensics/cache_wangxing_v5_2")
    parser.add_argument("--au-output-root", default="outputs/forensics/cache_wangxing_v5_2/au")
    parser.add_argument("--output-root", default="outputs/vedio_pred/wangxing_v5_2_results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wangxing-device", default="cuda")
    parser.add_argument("--min-pairwise", type=float, default=5.0 / 6.0)
    parser.add_argument("--enforce-gates", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_json(args.manifest)
    context = _build_context(args)
    holdout_rows = _ranking_rows(
        manifest=manifest,
        split="holdout",
        args=args,
        context=context,
    )
    holdout_metrics = rank_metrics(
        holdout_rows,
        min_pairwise=args.min_pairwise,
    )
    test_payloads: dict[str, Any] = {}
    for name, manifest_value in args.test_sets:
        rows = _binary_rows(
            project_path(manifest_value),
            args,
            context,
        )
        test_payloads[name] = {
            "metrics": _binary_metrics(rows),
            "rows": rows,
        }
    binary_gate_failures: list[str] = []
    for name, value in test_payloads.items():
        metrics = value["metrics"]
        if int(metrics.get("decision_flip_count") or 0) != 0:
            binary_gate_failures.append(
                f"{name} decision_flip_count="
                f"{metrics.get('decision_flip_count')}"
            )
        lex = (metrics.get("lexicographic") or {}).get(
            "lexicographic_satisfied"
        )
        if lex is False:
            binary_gate_failures.append(f"{name} lexicographic violated")
        accuracy = metrics.get("overall_accuracy")
        floor = BINARY_ACCURACY_FLOORS.get(name)
        if (
            floor is not None
            and accuracy is not None
            and float(accuracy) + 1e-9 < float(floor)
        ):
            binary_gate_failures.append(
                f"{name} accuracy {accuracy:.4f} < floor {floor:.4f}"
            )
    binary_gates = not binary_gate_failures
    holdout_counts = holdout_metrics.get("class_counts") or {}
    model_enabled = bool(
        context["rank_policy"]
        and context["rank_policy"].get("rank_model", {}).get("enabled")
    )
    rank_usable = bool(
        model_enabled
        and holdout_metrics.get("rank_available")
        and holdout_metrics.get("class_ordering_satisfied") is True
        and holdout_metrics.get("pairwise_ordering_rate", 0.0)
        >= float(args.min_pairwise)
        and all(holdout_counts.get(label, 0) > 0 for label in (
            "real",
            "lora",
            "seedance",
            "multiref",
        ))
    )
    output_root = project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    validated_policy = dict(context["rank_policy"] or {})
    validated_policy["usable_for_runtime"] = rank_usable
    validated_policy["ordering_satisfied"] = rank_usable
    validated_policy["holdout_metrics"] = holdout_metrics
    validated_policy["disabled_reason"] = resolve_disabled_reason(
        policy=context["rank_policy"],
        rank_usable=rank_usable,
    )
    validated_policy_path = output_root / "rank_policy_validated.json"
    validated_policy_path.write_text(
        json.dumps(validated_policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": "wangxing_v5_2_evaluation_v1",
        "decision_source": "v3_frozen",
        "manifest": str(project_path(args.manifest)),
        "rank_policy": context["rank_policy"],
        "holdout": {
            "metrics": holdout_metrics,
            "rows": holdout_rows,
        },
        "test_sets": test_payloads,
        "gates": {
            "binary_gates_passed": binary_gates,
            "binary_gate_failures": binary_gate_failures,
            "rank_usable": rank_usable,
            "rank_model_enabled": model_enabled,
            "disabled_reason": validated_policy["disabled_reason"],
            "gates_passed": binary_gates,
        },
        "test_training_allowed": False,
        "validated_rank_policy": str(validated_policy_path),
    }
    (output_root / "all_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["gates"], ensure_ascii=False, indent=2))
    print(json.dumps(holdout_metrics, ensure_ascii=False, indent=2))
    print(f"All results: {output_root / 'all_results.json'}")
    if args.enforce_gates and not binary_gates:
        raise SystemExit(
            "V5.2 binary gates failed:\n- "
            + "\n- ".join(binary_gate_failures)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
