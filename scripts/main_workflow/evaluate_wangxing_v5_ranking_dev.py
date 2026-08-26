"""Evaluate the V5 ranking development set with PT and web diagnostics.

This is a development-only report generator.  It never reads or modifies the
25+25 / 32+32 final-test manifests and never trains a ranking model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from evaluator.modules.forensics import analyze_forensics
from evaluator.modules.wangxing.authenticity_score import (
    extract_weighted_components,
)
from evaluator.modules.wangxing.wangxing_specialization import (
    evaluate_specialization,
)
from scripts.web_forensics.evaluate_generated_video import _run_extraction
from wangxing_project.cascade_v5 import cascade_score, load_rank_policy
from wangxing_project.drive_head_v5 import (
    extract_drive_feature_vector,
    load_drive_head,
    predict_drive_head,
)
from wangxing_project.joint_au_pt_v3 import predict_wangxing_v3

ORDER = ("real", "lora", "seedance", "multiref")
RANK = {label: index for index, label in enumerate(ORDER)}


def _load(path: str | Path) -> dict[str, Any]:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return json.loads(value.resolve().read_text(encoding="utf-8-sig"))


def _label(path: Path) -> str | None:
    name = path.name.casefold()
    if "真人" in path.name or "real" in name:
        return "real"
    if "iclora" in name or "lora" in name:
        return "lora"
    if "seedance" in name:
        return "seedance"
    if "多图" in path.name or "multiref" in name:
        return "multiref"
    return None


def _safe(value: Any, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _rank_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        v5 = row.get("v5") or {}
        by_label[str(row["label"])].append(float(v5["score_display"]))
    class_means = {
        label: (
            float(np.mean(by_label[label]))
            if by_label.get(label)
            else None
        )
        for label in ORDER
    }
    class_ordering = all(
        class_means[ORDER[index]] is not None
        and class_means[ORDER[index + 1]] is not None
        and class_means[ORDER[index]] > class_means[ORDER[index + 1]]
        for index in range(len(ORDER) - 1)
    )
    pair_total = 0
    pair_correct = 0
    for left in rows:
        for right in rows:
            if RANK[left["label"]] <= RANK[right["label"]]:
                continue
            pair_total += 1
            left_score = float((left.get("v5") or {})["score_display"])
            right_score = float((right.get("v5") or {})["score_display"])
            if left_score > right_score:
                pair_correct += 1
    pairwise = pair_correct / pair_total if pair_total else 0.0
    return {
        "expected_order": list(ORDER),
        "sample_count": len(rows),
        "class_counts": {
            label: len(by_label.get(label, [])) for label in ORDER
        },
        "class_mean_scores_0_1": class_means,
        "class_ordering_satisfied": class_ordering,
        "pairwise_ordering_rate": pairwise,
        "ordering_satisfied": bool(class_ordering and pairwise == 1.0),
    }


def _web_diagnostics(
    *,
    video: Path,
    au: Path,
    profiles: dict[str, Any],
    source_profile_path: Path,
    identity_profile_path: Path,
    expression_profile_path: Path,
    device: str,
    wangxing_device: str,
) -> dict[str, Any]:
    forensics = analyze_forensics(
        facial_motion=au,
        facial_motion_profile=profiles.get("facial_motion"),
        texture_detail=video,
        texture_detail_profile=profiles.get("texture_detail"),
        authenticity_calibrator=profiles.get("authenticity_calibrator"),
        max_frames=32,
        sample_fps=8.0,
        device=device,
    )
    specialization = evaluate_specialization(
        video_path=video,
        au_path=au,
        identity_profile_path=identity_profile_path,
        expression_profile_path=expression_profile_path,
        source_profile_path=source_profile_path,
        device=wangxing_device,
        max_identity_frames=16,
    )
    components = extract_weighted_components(
        {
            "wangxing_au": {
                **specialization,
                "forensics": forensics,
            },
            "forensics": forensics,
        }
    )
    return {
        "forensics_status": forensics.get("status"),
        "components": components,
        "identity_decision": (specialization.get("identity") or {}).get(
            "decision"
        ),
        "expression_compatibility": (
            specialization.get("expression_profile") or {}
        ).get("compatibility_0_1"),
        "source_real_probability": (
            specialization.get("source") or {}
        ).get("real_probability_0_1"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "V5 ranking-dev PT/Web report; no training and no final-test "
            "manifests are read."
        )
    )
    parser.add_argument(
        "--ranking-root",
        default=r"C:\Users\zhanghaotian\Desktop\ppt_video",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/forensics/wangxing_v5_ranking_dev",
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
        "--drive-cache",
        default="outputs/forensics/cache_wangxing_v5_2_ranking_dev",
    )
    parser.add_argument(
        "--au-output-root",
        default="outputs/forensics/wangxing_v5_ranking_dev/au",
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
        "--identity-profile",
        default="data/au/wangxing_identity_profile.json",
    )
    parser.add_argument(
        "--expression-profile",
        default="data/au/wangxing_expression_profile.json",
    )
    parser.add_argument(
        "--rank-policy",
        default="outputs/forensics/wangxing_authenticity_policy_v5.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wangxing-device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    del args.seed

    ranking_root = Path(args.ranking_root).expanduser().resolve()
    output_root = project_path(args.output_root)
    if not ranking_root.is_dir():
        raise SystemExit(f"Ranking root not found: {ranking_root}")
    if any(
        part.casefold() == "test"
        for part in ranking_root.parts
    ):
        raise SystemExit("Final test directory is not allowed as ranking root.")

    videos = sorted(ranking_root.rglob("*.mp4"))
    labeled = [
        (video, _label(video))
        for video in videos
        if _label(video) is not None
    ]
    labels = {label for _, label in labeled}
    if labels != set(ORDER):
        raise SystemExit(
            f"Expected labels {set(ORDER)}, got {labels}"
        )

    profiles = _load(args.forensics_profile)
    source_profile_path = project_path(args.source_profile)
    source_profile = _load(args.source_profile)
    identity_profile_path = project_path(args.identity_profile)
    expression_profile_path = project_path(args.expression_profile)
    v3_model = project_path(args.v3_model)
    drive_model = load_drive_head(project_path(args.drive_model))
    rank_policy = load_rank_policy(project_path(args.rank_policy))
    transition_cache = project_path(
        "outputs/vedio_pred/cache_wangxing_v4_expression_res1k/"
        "wangxing_v4_transition.npz"
    )
    blendshape_cache = project_path(
        "outputs/vedio_pred/cache_wangxing_v4_expression_res1k/"
        "wangxing_v4_blendshape.npz"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    au_root = project_path(args.au_output_root)
    rows_pt: list[dict[str, Any]] = []
    rows_web: list[dict[str, Any]] = []

    for index, (video, label) in enumerate(labeled, start=1):
        group = video.parent.name
        print(
            f"[V5 ranking dev] {index}/{len(labeled)} "
            f"{group}/{video.name}",
            flush=True,
        )
        au = _run_extraction(
            video,
            au_root,
            device=args.wangxing_device,
            batch_size=32,
            num_workers=0,
            force=False,
            cache_root=project_path(args.drive_cache),
            cache_namespace="wangxing_v5_ranking_dev",
        )
        v3 = predict_wangxing_v3(
            video_path=video,
            au_path=au,
            model_path=v3_model,
            source_profile=source_profile,
            forensics_profiles=profiles,
        )
        vector, drive_details = extract_drive_feature_vector(
            video_path=video,
            au_path=au,
            cache_dir=project_path(args.drive_cache),
            transition_cache=(
                transition_cache if transition_cache.is_file() else None
            ),
            blendshape_cache=(
                blendshape_cache if blendshape_cache.is_file() else None
            ),
        )
        p_drive, drive_prediction = predict_drive_head(
            vector=vector,
            model=drive_model,
        )
        cascade = cascade_score(
            p_v3_real=1.0 - float(v3["generated_probability"]),
            p_drive=p_drive,
            p_drive_eff=p_drive,
            rank_policy=rank_policy,
        )
        base = {
            "video": str(video),
            "au": str(au),
            "group": group,
            "label": label,
            "rank": RANK[label],
            "v3": {
                "prediction": v3["prediction"],
                "p_real": 1.0 - float(v3["generated_probability"]),
                "p_generated": float(v3["generated_probability"]),
            },
            "drive": {
                **drive_prediction,
                **drive_details,
            },
            "v5": cascade,
        }
        rows_pt.append(base)
        web = _web_diagnostics(
            video=video,
            au=au,
            profiles=profiles,
            source_profile_path=source_profile_path,
            identity_profile_path=identity_profile_path,
            expression_profile_path=expression_profile_path,
            device=args.device,
            wangxing_device=args.wangxing_device,
        )
        rows_web.append(
            {
                **base,
                "web": web,
            }
        )

    payload = {
        "schema_version": "wangxing_v5_ranking_dev_report_v1",
        "ranking_root": str(ranking_root),
        "development_only": True,
        "rank_policy": {
            "ordering_satisfied": bool(
                rank_policy.get("ordering_satisfied")
            ),
            "usable_for_runtime": bool(
                rank_policy.get("usable_for_runtime")
            ),
            "minimum_queries": rank_policy.get("minimum_queries"),
            "complete_query_count": (
                rank_policy.get("ranking_inventory") or {}
            ).get("complete_query_count"),
        },
        "test_sets_excluded": [
            "data/test/single_video",
            "data/test/wangxing_32x32",
        ],
        "pt": {
            "metrics": _rank_metrics(rows_pt),
            "rows": rows_pt,
        },
        "web": {
            "metrics": _rank_metrics(rows_web),
            "rows": rows_web,
        },
    }
    (output_root / "pt_ranking.json").write_text(
        json.dumps(payload["pt"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "web_ranking.json").write_text(
        json.dumps(payload["web"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "all_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "pt": payload["pt"]["metrics"],
            "web": payload["web"]["metrics"],
            "rank_policy": payload["rank_policy"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    print(f"All results: {output_root / 'all_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
