from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

CALIBRATOR_SCHEMA = "seedance_authenticity_calibrator_v1"


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def fit_probability_calibrator(
    real_scores: Iterable[float],
    generated_scores: Iterable[float],
    *,
    iterations: int = 300,
    learning_rate: float = 0.25,
) -> dict[str, Any]:
    """Fit a one-dimensional Platt-style calibrator on held-out scores."""
    real_values = [
        float(value)
        for value in real_scores
        if math.isfinite(float(value))
    ]
    generated_values = [
        float(value)
        for value in generated_scores
        if math.isfinite(float(value))
    ]
    x = np.asarray(
        [*real_values, *generated_values],
        dtype=np.float64,
    )
    y = np.asarray(
        [1.0] * len(real_values) + [0.0] * len(generated_values),
        dtype=np.float64,
    )
    if x.size == 0 or np.unique(y).size < 2:
        raise ValueError("Both real and generated scores are required.")
    mean = float(np.mean(x))
    scale = max(float(np.std(x)), 0.05)
    standardized = (x - mean) / scale
    slope = 1.0
    intercept = 0.0
    for _ in range(max(1, iterations)):
        probabilities = 1.0 / (1.0 + np.exp(-(intercept + slope * standardized)))
        error = probabilities - y
        intercept -= learning_rate * float(np.mean(error))
        slope -= learning_rate * float(np.mean(error * standardized))
    return {
        "schema_version": CALIBRATOR_SCHEMA,
        "status": "ready",
        "feature": "raw_real_capture_likelihood_0_1",
        "calibration_method": "platt_logistic_1d",
        "mean": mean,
        "scale": scale,
        "slope": float(slope),
        "intercept": float(intercept),
        "real_count": int(np.sum(y == 1.0)),
        "generated_count": int(np.sum(y == 0.0)),
        "held_out_required": True,
        "split_protocol": "source_video_or_generation_batch_holdout",
    }


def apply_probability_calibrator(
    raw_score: float | None,
    calibrator: dict[str, Any] | None,
) -> float | None:
    if raw_score is None or calibrator is None:
        return None
    if calibrator.get("status", "ready") not in {"ready", "calibrated"}:
        return None
    standardized = (
        float(raw_score) - float(calibrator.get("mean", 0.5))
    ) / max(float(calibrator.get("scale", 0.05)), 0.05)
    return _sigmoid(
        float(calibrator.get("intercept", 0.0))
        + float(calibrator.get("slope", 1.0)) * standardized
    )


def branch_confidence(result: dict[str, Any] | None) -> float:
    if not isinstance(result, dict):
        return 0.0
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return 0.0
    if result.get("status") == "unavailable":
        return 0.0
    coverage = metrics.get("landmark_valid_frame_ratio")
    if coverage is None:
        coverage = metrics.get("face_box_coverage")
    coverage_score = 0.75 if coverage is None else _clamp(float(coverage))
    quality_gate = metrics.get("input_quality_gate_0_1")
    if quality_gate is not None:
        coverage_score = _clamp(
            0.65 * coverage_score + 0.35 * float(quality_gate)
        )
    frame_count = result.get("feature_record", {}).get("frame_count", 0)
    frame_score = _clamp(float(frame_count) / 16.0)
    fit_score = metrics.get("real_domain_fit_0_1")
    fit_confidence = 0.75 if fit_score is None else _clamp(float(fit_score))
    return _clamp(0.45 * coverage_score + 0.25 * frame_score + 0.30 * fit_confidence)


def summarize_window_evidence(
    windows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize window evidence without turning it into source probability."""
    records = [dict(window) for window in windows]
    if not records:
        return {
            "window_count": 0,
            "mean_evidence_score_0_1": None,
            "worst_evidence_score_0_1": None,
            "aggregate_evidence_score_0_1": None,
            "worst_window": None,
        }
    scores = np.asarray(
        [
            _clamp(
                float(
                    record.get(
                        "anomaly_score_0_1",
                        record.get("evidence_score_0_1", 0.0),
                    )
                )
            )
            for record in records
        ],
        dtype=np.float32,
    )
    worst_index = int(np.argmax(scores))
    mean_score = float(np.mean(scores))
    worst_score = float(scores[worst_index])
    return {
        "window_count": len(records),
        "mean_evidence_score_0_1": mean_score,
        "worst_evidence_score_0_1": worst_score,
        "aggregate_evidence_score_0_1": _clamp(
            0.50 * mean_score + 0.50 * worst_score
        ),
        "worst_window": records[worst_index],
    }


def fuse_authenticity_evidence(
    facial_result: dict[str, Any] | None,
    texture_result: dict[str, Any] | None,
    *,
    calibrator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fuse two branches without calling an uncalibrated ratio a probability."""
    branches: list[tuple[str, dict[str, Any], float, float]] = []
    for name, result in (
        ("facial_expression_muscle", facial_result),
        ("texture_detail", texture_result),
    ):
        if not isinstance(result, dict):
            continue
        metrics = result.get("metrics", {})
        raw_score = None
        if isinstance(metrics, dict):
            raw_score = metrics.get("raw_real_domain_evidence_0_1")
            if raw_score is None:
                # Backward-compatible read for profiles produced before the
                # explicit raw-evidence field was added.
                raw_score = metrics.get("real_capture_likelihood_0_1")
        if raw_score is None:
            continue
        confidence = branch_confidence(result)
        branches.append((name, result, float(raw_score), confidence))
    if not branches:
        return {
            "status": "unavailable",
            "decision": "uncertain",
            "binary_decision": "seedance_like",
            "binary_conclusion": "偏向 AI 生成",
            "raw_real_capture_likelihood_0_1": None,
            "raw_real_domain_evidence_0_1": None,
            "calibrated_real_probability_0_1": None,
            "confidence_0_1": 0.0,
            "calibrator_applied": False,
            "branch_weights": {},
            "uncertainty_reasons": ["no_calibrated_branch_score"],
        }
    total_weight = sum(max(confidence, 0.05) for _, _, _, confidence in branches)
    raw_fused = sum(
        raw_score * max(confidence, 0.05)
        for _, _, raw_score, confidence in branches
    ) / total_weight
    prior_terms: list[tuple[float, float]] = []
    for name, result, _, confidence in branches:
        metrics = result.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        prior_key = (
            "training_free_motion_prior_0_1"
            if name == "facial_expression_muscle"
            else "training_free_texture_prior_0_1"
        )
        prior = metrics.get(prior_key)
        if prior is None:
            continue
        prior_terms.append((float(prior), max(confidence, 0.05)))
    training_free_prior = None
    if prior_terms:
        prior_weight = sum(weight for _, weight in prior_terms)
        training_free_prior = sum(
            prior * weight for prior, weight in prior_terms
        ) / prior_weight
    calibrated = apply_probability_calibrator(raw_fused, calibrator)
    confidence = _clamp(
        sum(confidence for _, _, _, confidence in branches) / len(branches)
    )
    raw_direction = (
        "real_like"
        if raw_fused >= 0.60
        else "seedance_like"
        if raw_fused <= 0.40
        else "mixed"
    )
    # After timestamp-aware recalibration, trust the held-out probability as
    # the primary decision signal. Only fall back to uncertain when the raw
    # fused evidence strongly conflicts with that probability (the failure
    # mode seen with the pre-fix steep calibrator).
    decision = "uncertain"
    if calibrated is not None:
        if calibrated >= 0.60:
            decision = (
                "real_capture" if raw_fused >= 0.45 else "uncertain"
            )
        elif calibrated <= 0.40:
            decision = (
                "seedance_like" if raw_fused <= 0.55 else "uncertain"
            )
        else:
            decision = "uncertain"
    uncertainty_reasons: list[str] = []
    if calibrated is None:
        if calibrator is None:
            uncertainty_reasons.append(
                "held_out_probability_calibrator_missing"
            )
        else:
            uncertainty_reasons.append("probability_calibrator_not_ready")
    elif decision == "uncertain" and calibrated >= 0.60 and raw_fused < 0.45:
        uncertainty_reasons.append("calibrated_real_but_raw_evidence_low")
    elif decision == "uncertain" and calibrated <= 0.40 and raw_fused > 0.55:
        uncertainty_reasons.append(
            "calibrated_seedance_but_raw_evidence_high"
        )
    elif decision == "uncertain" and 0.40 < calibrated < 0.60:
        uncertainty_reasons.append("calibrated_probability_in_uncertain_band")
    binary_score = (
        calibrated
        if calibrated is not None
        else raw_fused
    )
    binary_decision = (
        "real_capture"
        if binary_score >= 0.50
        else "seedance_like"
    )
    return {
        "status": "calibrated" if calibrated is not None else "uncalibrated",
        "decision": decision,
        "binary_decision": binary_decision,
        "binary_conclusion": (
            "偏向真实拍摄"
            if binary_decision == "real_capture"
            else "偏向 AI 生成"
        ),
        "raw_evidence_direction": raw_direction,
        "raw_real_capture_likelihood_0_1": float(raw_fused),
        "raw_real_domain_evidence_0_1": float(raw_fused),
        "calibrated_real_probability_0_1": calibrated,
        "confidence_0_1": confidence,
        "calibrator_applied": calibrated is not None,
        "calibrator_status": (
            calibrator.get("status", "ready")
            if isinstance(calibrator, dict)
            else None
        ),
        "branch_weights": {
            name: float(max(branch_confidence_value, 0.05) / total_weight)
            for name, _, _, branch_confidence_value in branches
        },
        "branch_scores": {
            name: float(raw_score)
            for name, _, raw_score, _ in branches
        },
        "training_free_prior_0_1": training_free_prior,
        "uncertainty_reasons": uncertainty_reasons,
    }


def rank_window_evidence(
    windows: Iterable[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked = sorted(
        (dict(window) for window in windows),
        key=lambda window: float(
            window.get("anomaly_score_0_1", window.get("evidence_score_0_1", 0.0))
        ),
        reverse=True,
    )
    return ranked[: max(0, int(limit))]
