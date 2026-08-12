"""Fused hard real-vs-generated detector (no uncertain band).

Combines:
1. Wang Xing ``source`` specialization (AU domain)
2. Forensics facial-motion profile evidence
3. Optional texture / frequency evidence when a video path is provided

Always returns a hard label. Mid-score clips are still labeled; quality only
down-weights fusion confidence, it does not refuse a decision.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .authenticity_decision import decide_real_vs_generated
from .report import analyze_forensics
from .seedance_authenticity import apply_probability_calibrator

FUSED_DETECTOR_SCHEMA = "fused_hard_detector_v1"


def _finite(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def fuse_real_scores(
    *,
    wangxing_real_0_1: float | None,
    forensics_real_0_1: float | None,
    texture_real_0_1: float | None = None,
    wangxing_weight: float = 0.45,
    forensics_weight: float = 0.40,
    texture_weight: float = 0.15,
    quality_0_1: float | None = None,
) -> dict[str, Any]:
    """Weighted fuse of available real-likeness scores into one hard score."""
    terms: list[tuple[str, float, float]] = []
    if wangxing_real_0_1 is not None:
        terms.append(("wangxing_source", float(wangxing_real_0_1), wangxing_weight))
    if forensics_real_0_1 is not None:
        terms.append(("forensics_motion", float(forensics_real_0_1), forensics_weight))
    if texture_real_0_1 is not None:
        terms.append(("forensics_texture", float(texture_real_0_1), texture_weight))
    if not terms:
        return {
            "fused_real_0_1": None,
            "branch_scores": {},
            "branch_weights": {},
            "quality_0_1": _finite(quality_0_1),
        }
    total_w = sum(weight for _, _, weight in terms)
    fused = sum(score * weight for _, score, weight in terms) / max(total_w, 1e-8)
    quality = _finite(quality_0_1)
    # Low quality pulls toward 0.5 mildly, but never refuses a hard label.
    if quality is not None:
        gate = _clamp((quality - 0.25) / 0.55)
        fused = gate * fused + (1.0 - gate) * 0.5
    return {
        "fused_real_0_1": _clamp(fused),
        "branch_scores": {name: score for name, score, _ in terms},
        "branch_weights": {
            name: weight / total_w for name, _, weight in terms
        },
        "quality_0_1": quality,
    }


def score_fused_hard_detector(
    *,
    au_path: str | Path,
    video_path: str | Path | None = None,
    wangxing_source_profile: dict[str, Any] | None = None,
    forensics_profiles: dict[str, Any] | None = None,
    fused_calibrator: dict[str, Any] | None = None,
    learned_head: dict[str, Any] | None = None,
    include_texture: bool = False,
    max_frames: int = 24,
    sample_fps: float = 8.0,
    hard_threshold: float = 0.5,
    wangxing_weight: float = 0.45,
    forensics_weight: float = 0.40,
    texture_weight: float = 0.15,
) -> dict[str, Any]:
    """Score one clip and always emit a hard real/generated decision."""
    if (
        learned_head is not None
        and wangxing_source_profile is not None
        and forensics_profiles is not None
    ):
        from .learned_fusion_head import score_with_learned_head

        scored = score_with_learned_head(
            au_path=au_path,
            wangxing_source_profile=wangxing_source_profile,
            forensics_profiles=forensics_profiles,
            learned_head=learned_head,
            hard_threshold=hard_threshold,
        )
        return {
            "schema_version": FUSED_DETECTOR_SCHEMA,
            "status": scored.get("status", "available"),
            "hard_decision": scored.get("hard_decision"),
            "predicted_generated": scored.get("predicted_generated"),
            "fused_real_0_1": scored.get("decision_score_0_1"),
            "calibrated_real_0_1": scored.get("decision_score_0_1"),
            "decision_score_0_1": scored.get("decision_score_0_1"),
            "fusion": {
                "mode": "learned_fusion_head",
                "features": scored.get("features"),
                "threshold": scored.get("threshold"),
                "quality_0_1": scored.get("quality_0_1"),
            },
            "wangxing_source": None,
            "forensics": None,
            "manual_scores_required": False,
            "uncertain_band_used": False,
            "note": (
                "Hard detector via learned fusion head on Wang Xing source + "
                "forensics motion features (no texture)."
            ),
        }

    from ..wangxing.wangxing_specialization import score_source_profile

    wangxing = None
    wangxing_real = None
    quality = None
    if wangxing_source_profile is not None:
        wangxing = score_source_profile(au_path, wangxing_source_profile)
        wangxing_real = _finite(wangxing.get("real_probability_0_1"))
        quality_blob = wangxing.get("quality") or {}
        if isinstance(quality_blob, dict):
            quality = _finite(quality_blob.get("valid_frame_ratio"))

    forensics = None
    forensics_real = None
    texture_real = None
    if forensics_profiles is not None:
        texture_input = None
        if include_texture and video_path is not None and Path(video_path).is_file():
            texture_input = video_path
        forensics = analyze_forensics(
            facial_motion=au_path,
            facial_motion_profile=forensics_profiles.get("facial_motion"),
            texture_detail=texture_input,
            texture_detail_profile=forensics_profiles.get("texture_detail"),
            authenticity_calibrator=None,
            max_frames=max_frames,
            sample_fps=sample_fps,
            detect_faces=False,
        )
        scores = forensics.get("scores", {})
        forensics_real = _finite(scores.get("raw_real_domain_evidence_0_1"))
        if forensics_real is None:
            forensics_real = _finite(scores.get("facial_expression_muscle_score_0_1"))
        texture_real = _finite(scores.get("texture_detail_score_0_1"))
        facial_metrics = (
            (forensics.get("branches") or {}).get("facial_motion") or {}
        ).get("metrics", {})
        if isinstance(facial_metrics, dict):
            quality = _finite(
                facial_metrics.get("input_quality_gate_0_1"),
                quality,
            )

    fused = fuse_real_scores(
        wangxing_real_0_1=wangxing_real,
        forensics_real_0_1=forensics_real,
        texture_real_0_1=texture_real if include_texture else None,
        wangxing_weight=wangxing_weight,
        forensics_weight=forensics_weight,
        texture_weight=texture_weight,
        quality_0_1=quality,
    )
    raw_fused = fused.get("fused_real_0_1")
    calibrated = apply_probability_calibrator(raw_fused, fused_calibrator)
    score = calibrated if calibrated is not None else raw_fused
    decision = decide_real_vs_generated(
        real_score_0_1=score,
        quality_0_1=quality,
        hard_threshold=hard_threshold,
        allow_uncertain=False,
    )
    return {
        "schema_version": FUSED_DETECTOR_SCHEMA,
        "status": "available" if score is not None else "unavailable",
        "hard_decision": decision,
        "predicted_generated": decision.get("predicted_generated"),
        "fused_real_0_1": raw_fused,
        "calibrated_real_0_1": calibrated,
        "decision_score_0_1": score,
        "fusion": fused,
        "wangxing_source": wangxing,
        "forensics": forensics,
        "manual_scores_required": False,
        "uncertain_band_used": False,
        "note": (
            "Hard detector always labels real/generated. Quality only softens "
            "the fused score toward 0.5; it does not create an uncertain class."
        ),
    }
