"""Shared V5.1 feature extraction for PT and offline Web reports."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from evaluator.modules.core.paths import project_path
from evaluator.modules.forensics import analyze_forensics
from evaluator.modules.wangxing.authenticity_score import (
    extract_weighted_components,
)
from scripts.web_forensics.evaluate_generated_video import _run_extraction
from wangxing_project.cascade_v5 import cascade_score_v51, load_rank_policy
from wangxing_project.drive_head_v5 import (
    extract_drive_feature_vector,
    load_drive_head,
    predict_drive_head,
)
from wangxing_project.joint_au_pt_v3 import predict_wangxing_v3
from wangxing_project.realness_v5 import (
    features_from_components,
    predict_realness,
)

ORDER = ("real", "lora", "seedance", "multiref")
RANK = {label: index for index, label in enumerate(ORDER)}


def label_video(path: Path) -> str | None:
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


def load_json(path: str | Path) -> dict[str, Any]:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = project_path(value)
    return json.loads(value.resolve().read_text(encoding="utf-8-sig"))


def collect_videos(root: str | Path, group: str) -> list[tuple[Path, str]]:
    ranking_root = Path(root).expanduser().resolve()
    group_root = ranking_root / group
    videos = sorted(group_root.rglob("*.mp4"))
    labeled = [
        (video, label_video(video))
        for video in videos
        if label_video(video) is not None
    ]
    labels = {label for _, label in labeled}
    if labels != set(ORDER):
        raise ValueError(
            f"{group} must contain labels {set(ORDER)}, got {labels}"
        )
    return [(video, str(label)) for video, label in labeled]


def extract_au_for_video(
    *,
    video: Path,
    au_output_root: Path,
    cache_dir: Path,
    device: str,
) -> Path:
    return _run_extraction(
        video,
        au_output_root,
        device=device,
        batch_size=32,
        num_workers=0,
        force=False,
        cache_root=cache_dir,
        cache_namespace="wangxing_v5_1_realness",
    )


def build_feature_row(
    *,
    video: Path,
    label: str,
    group: str,
    au_path: Path,
    v3_model: Path,
    drive_model: dict[str, Any] | None,
    drive_cache: Path,
    source_profile: dict[str, Any],
    forensics_profile: dict[str, Any],
    device: str,
    wangxing_device: str,
    calibrator: dict[str, Any] | None = None,
    realness_enabled: bool = True,
    rank_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    v3 = predict_wangxing_v3(
        video_path=video,
        au_path=au_path,
        model_path=v3_model,
        source_profile=source_profile,
        forensics_profiles=forensics_profile,
    )
    vector, drive_details = extract_drive_feature_vector(
        video_path=video,
        au_path=au_path,
        cache_dir=drive_cache,
        transition_cache=project_path(
            "outputs/vedio_pred/cache_wangxing_v4_expression_res1k/"
            "wangxing_v4_transition.npz"
        ),
        blendshape_cache=project_path(
            "outputs/vedio_pred/cache_wangxing_v4_expression_res1k/"
            "wangxing_v4_blendshape.npz"
        ),
    )
    p_drive, drive_prediction = predict_drive_head(
        vector=vector,
        model=drive_model,
    )
    forensics = analyze_forensics(
        facial_motion=au_path,
        facial_motion_profile=forensics_profile.get("facial_motion"),
        texture_detail=video,
        texture_detail_profile=forensics_profile.get("texture_detail"),
        authenticity_calibrator=forensics_profile.get(
            "authenticity_calibrator"
        ),
        max_frames=32,
        sample_fps=8.0,
        device=device,
    )
    components = extract_weighted_components(
        {
            "wangxing_au": {"forensics": forensics},
            "forensics": forensics,
        }
    )
    p_v3_real = 1.0 - float(v3["generated_probability"])
    p_drive_eff = (
        drive_prediction.get("p_drive_eff")
        if drive_prediction
        else p_drive
    )
    realness_features = features_from_components(
        p_drive_eff=p_drive_eff,
        p_v3_real=p_v3_real,
        components=components,
        drive_status=drive_prediction.get("status")
        if drive_prediction
        else None,
    )
    realness = predict_realness(
        features=realness_features,
        calibrator=calibrator,
        enabled=realness_enabled,
    )
    expected_real = label == "real"
    prior_conflict = (v3["prediction"] == "real") != expected_real
    cascade = cascade_score_v51(
        p_v3_real=p_v3_real,
        p_drive=p_drive,
        p_drive_eff=p_drive_eff,
        realness=realness,
        rank_policy=rank_policy,
        realness_enabled=realness_enabled,
        prior_conflict=prior_conflict,
    )
    return {
        "video": str(video),
        "au": str(au_path),
        "group": group,
        "label": label,
        "rank": RANK[label],
        "v3": {
            "prediction": v3["prediction"],
            "p_real": p_v3_real,
            "p_generated": float(v3["generated_probability"]),
        },
        "drive": {
            **drive_prediction,
            **drive_details,
        },
        "forensics": {
            "status": forensics.get("status"),
            "components": components,
        },
        "realness": realness,
        "v5": cascade,
        "prior_conflict": prior_conflict,
        "decision_matches_v3": cascade["decision"] == v3["prediction"],
    }


def lexicographic_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check V3 decision bands: all real display scores above all AI scores."""
    real_scores = [
        float(row["v5"]["score_display"])
        for row in rows
        if row.get("v5", {}).get("decision") == "real"
    ]
    ai_scores = [
        float(row["v5"]["score_display"])
        for row in rows
        if row.get("v5", {}).get("decision") == "generated"
    ]
    if not real_scores or not ai_scores:
        return {
            "lexicographic_satisfied": None,
            "min_real_score_display": (
                min(real_scores) if real_scores else None
            ),
            "max_ai_score_display": (
                max(ai_scores) if ai_scores else None
            ),
            "reason": "missing_decision_band",
        }
    min_real = min(real_scores)
    max_ai = max(ai_scores)
    return {
        "lexicographic_satisfied": min_real > max_ai,
        "min_real_score_display": min_real,
        "max_ai_score_display": max_ai,
        "reason": None,
    }


def rank_metrics(
    rows: list[dict[str, Any]],
    *,
    min_pairwise: float = 5.0 / 6.0,
) -> dict[str, Any]:
    by_label: dict[str, list[float]] = {label: [] for label in ORDER}
    for row in rows:
        by_label[row["label"]].append(float(row["v5"]["score_display"]))
    means = {
        label: (
            sum(values) / len(values)
            if values
            else None
        )
        for label, values in by_label.items()
    }
    class_ordering = all(
        means[ORDER[index]] is not None
        and means[ORDER[index + 1]] is not None
        and means[ORDER[index]] > means[ORDER[index + 1]]
        for index in range(len(ORDER) - 1)
    )
    total = correct = 0
    for left in rows:
        for right in rows:
            if RANK[left["label"]] <= RANK[right["label"]]:
                continue
            total += 1
            if left["v5"]["score_display"] < right["v5"]["score_display"]:
                correct += 1
    pairwise = correct / total if total else 0.0
    return {
        "expected_order": list(ORDER),
        "sample_count": len(rows),
        "class_counts": {
            label: len(by_label[label]) for label in ORDER
        },
        "class_mean_scores_0_1": means,
        "class_ordering_satisfied": class_ordering,
        "pairwise_ordering_rate": pairwise,
        "pairwise_correct": correct,
        "pairwise_total": total,
        "min_pairwise_threshold": float(min_pairwise),
        "ordering_satisfied": bool(
            class_ordering and pairwise + 1e-9 >= float(min_pairwise)
        ),
        "decision_flip_count": sum(
            not bool(row.get("decision_matches_v3", False))
            for row in rows
        ),
        "prior_conflict_count": sum(
            bool(row.get("prior_conflict")) for row in rows
        ),
    }
