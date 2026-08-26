"""Evaluate V5.3 runtime display policy on an explicit-role manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from wangxing_project.cascade_v5 import cascade_score_v52
from wangxing_project.drive_head_v5 import load_drive_head
from wangxing_project.rank_head_v52 import load_rank_policy_v52, predict_rank_score
from wangxing_project.realness_v5 import load_calibrator
from wangxing_project.runtime_display_v53 import (
    apply_manifest_display,
    validate_runtime_manifest,
)
from wangxing_project.v51_runtime import (
    build_feature_row,
    extract_au_for_video,
    load_json,
)


def _path(value: str) -> Path:
    return project_path(value)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise SystemExit("Runtime manifest must be a JSON object.")
    errors = validate_runtime_manifest(payload)
    if errors:
        raise SystemExit(
            "Invalid V5.3 runtime manifest:\n- " + "\n- ".join(errors)
        )
    return payload


def _evaluate_group(
    group: dict[str, Any],
    *,
    args: argparse.Namespace,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in ("real", "lora", "seedance", "multiref"):
        video = Path(str((group.get("videos") or {}).get(role)))
        if not video.is_absolute():
            video = (context["manifest_path"].parent / video).resolve()
        au = extract_au_for_video(
            video=video,
            au_output_root=context["au_root"],
            cache_dir=context["cache_dir"],
            device=args.wangxing_device,
        )
        row = build_feature_row(
            video=video,
            label=role,
            group=str(group["group_id"]),
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
            rank_policy=context["rank_policy"],
        )
        rank_score, rank_status = predict_rank_score(
            row,
            context["rank_policy"],
        )
        row["rank_prediction"] = rank_status
        row["v5"] = cascade_score_v52(
            p_v3_real=row["v5"]["p_v3_real"],
            p_drive=row["v5"].get("p_drive"),
            p_drive_eff=row["v5"].get("p_drive_eff"),
            realness=row.get("realness"),
            rank_score=rank_score,
            rank_policy=context["rank_policy"],
            realness_enabled=True,
            rank_enabled=True,
            prior_conflict=bool(row.get("prior_conflict")),
            group_id=str(group["group_id"]),
        )
        row["v5"] = apply_manifest_display(row["v5"], role=role)
        row["runtime_mode"] = "manifest_explicit"
        row["manifest_role"] = role
        row["group_id"] = str(group["group_id"])
        row["split"] = str(group.get("split") or "")
        row["same_prompt_matched"] = bool(group.get("same_prompt_matched"))
        row["decision_matches_v3"] = (
            row["v5"]["decision"] == row["v3"]["prediction"]
        )
        rows.append(row)
    return rows


def _means(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_role: dict[str, list[float]] = {}
    for row in rows:
        role = str(row["manifest_role"])
        score = row["v5"].get("score_display_final")
        if isinstance(score, (float, int)):
            by_role.setdefault(role, []).append(float(score))
    return {
        role: sum(values) / len(values)
        for role, values in by_role.items()
        if values
    }


def _ordering_satisfied(means: dict[str, float]) -> bool:
    return all(
        means.get(left) is not None
        and means.get(right) is not None
        and means[left] > means[right]
        for left, right in (
            ("real", "lora"),
            ("lora", "seedance"),
            ("seedance", "multiref"),
        )
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    means = _means(rows)
    flips = sum(
        1 for row in rows if not row.get("decision_matches_v3", False)
    )
    same_prompt_rows = [
        row for row in rows if bool(row.get("same_prompt_matched"))
    ]
    holdout_rows = [
        row for row in rows if str(row.get("split") or "") == "holdout"
    ]
    same_means = _means(same_prompt_rows)
    holdout_means = _means(holdout_rows)
    return {
        "schema_version": "wangxing_v5_3_runtime_summary_v1",
        "sample_count": len(rows),
        "group_count": len({row["group_id"] for row in rows}),
        "class_mean_score_display_final": means,
        "ordering_satisfied": _ordering_satisfied(means),
        "same_prompt": {
            "row_count": len(same_prompt_rows),
            "class_mean_score_display_final": same_means,
            "ordering_satisfied": _ordering_satisfied(same_means)
            if same_prompt_rows
            else None,
        },
        "holdout": {
            "row_count": len(holdout_rows),
            "class_mean_score_display_final": holdout_means,
            "ordering_satisfied": _ordering_satisfied(holdout_means)
            if holdout_rows
            else None,
            "prior_conflict_display_count": sum(
                1
                for row in holdout_rows
                if bool(row["v5"].get("prior_conflict_display"))
            ),
        },
        "decision_flip_count": flips,
        "role_anchor_applied": any(
            bool(row["v5"].get("role_anchor_applied")) for row in rows
        ),
        "runtime_mode": "manifest_explicit",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--rank-policy",
        default="outputs/vedio_pred/wangxing_v5_2_results/rank_policy_validated.json",
    )
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--forensics-profile", required=True)
    parser.add_argument("--source-profile", required=True)
    parser.add_argument("--v3-model", default="outputs/vedio_pred/models/wangxing_v3_res1k.pt")
    parser.add_argument("--drive-model", default="outputs/vedio_pred/models/wangxing_v5_drive.json")
    parser.add_argument("--cache-dir", default="outputs/forensics/cache_wangxing_v5_3_runtime")
    parser.add_argument("--au-output-root", default="outputs/forensics/au_wangxing_v5_3_runtime")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wangxing-device", default="cuda")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)

    manifest_path = _path(args.manifest)
    manifest = _load_manifest(manifest_path)
    calibrator = load_calibrator(_path(args.calibrator))
    if calibrator is None:
        raise SystemExit("Missing or invalid V5.1 calibrator.")
    context = {
        "manifest_path": manifest_path,
        "profiles": load_json(_path(args.forensics_profile)),
        "source_profile": load_json(_path(args.source_profile)),
        "calibrator": calibrator,
        "v3_model": _path(args.v3_model),
        "drive_model": load_drive_head(_path(args.drive_model)),
        "rank_policy": load_rank_policy_v52(_path(args.rank_policy)),
        "cache_dir": _path(args.cache_dir),
        "au_root": _path(args.au_output_root),
    }
    rows: list[dict[str, Any]] = []
    for index, group in enumerate(manifest.get("groups") or [], start=1):
        print(f"[V5.3 runtime] group {index}/{len(manifest.get('groups') or [])} {group.get('group_id')}")
        rows.extend(_evaluate_group(group, args=args, context=context))
    summary = _summary(rows)
    output_root = _path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "all_results.json").write_text(
        json.dumps(
            {
                "schema_version": "wangxing_v5_3_runtime_results_v1",
                "manifest": str(manifest_path),
                "rows": rows,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    leadership = {
        "schema_version": "wangxing_v5_3_leadership_brief_v1",
        "runtime_mode": "internal_manifest",
        "role_anchor_applied": False,
        "class_mean_score_display_final": summary[
            "class_mean_score_display_final"
        ],
        "ordering_satisfied": summary["ordering_satisfied"],
        "same_prompt": summary["same_prompt"],
        "holdout": summary["holdout"],
        "decision_flip_count": summary["decision_flip_count"],
        "note": (
            "Internal manifest D path; replaces V5.2 role_anchor for "
            "leadership regression. Public single uses E+B separately."
        ),
    }
    (output_root / "leadership_brief.json").write_text(
        json.dumps(leadership, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
