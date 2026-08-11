"""Decision helpers: recalibration + uncertainty band + quality gate."""

from __future__ import annotations

import math
from typing import Any

from .seedance_authenticity import apply_probability_calibrator

DECISION_SCHEMA = "authenticity_decision_v1"


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


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


def decide_real_vs_generated(
    *,
    real_score_0_1: float | None,
    quality_0_1: float | None = None,
    calibrator: dict[str, Any] | None = None,
    hard_threshold: float = 0.5,
    uncertain_low: float = 0.35,
    uncertain_high: float = 0.65,
    min_quality: float = 0.45,
    allow_uncertain: bool = True,
) -> dict[str, Any]:
    """Map a real-capture score into real / generated / uncertain.

    ``real_score_0_1`` higher = more real. Generated is predicted when the
    (optionally calibrated) score is low.
    """
    raw = _finite(real_score_0_1)
    if raw is None:
        return {
            "schema_version": DECISION_SCHEMA,
            "decision": "uncertain",
            "predicted_generated": None,
            "score_0_1": None,
            "calibrated_score_0_1": None,
            "quality_0_1": _finite(quality_0_1),
            "reason": "missing_score",
            "manual_scores_required": False,
        }

    calibrated = apply_probability_calibrator(raw, calibrator)
    score = float(calibrated if calibrated is not None else raw)
    quality = _finite(quality_0_1)
    reasons: list[str] = []

    if allow_uncertain and quality is not None and quality < min_quality:
        reasons.append("low_input_quality")
    if allow_uncertain and uncertain_low <= score <= uncertain_high:
        reasons.append("score_in_uncertain_band")

    if reasons:
        decision = "uncertain"
        predicted_generated = None
    elif score < hard_threshold:
        decision = "generated"
        predicted_generated = True
    else:
        decision = "real"
        predicted_generated = False

    return {
        "schema_version": DECISION_SCHEMA,
        "decision": decision,
        "predicted_generated": predicted_generated,
        "score_0_1": float(raw),
        "calibrated_score_0_1": score,
        "quality_0_1": quality,
        "hard_threshold": float(hard_threshold),
        "uncertain_band": [float(uncertain_low), float(uncertain_high)],
        "min_quality": float(min_quality),
        "reasons": reasons,
        "manual_scores_required": False,
        "note": (
            "Uncertain means refuse to hard-label mid-score or low-quality "
            "clips. It is not a third ground-truth class."
        ),
    }


def metrics_from_decisions(
    labels_generated: list[int],
    decisions: list[dict[str, Any]],
    *,
    include_uncertain_as_error: bool = False,
) -> dict[str, Any]:
    """Compute generated recall / precision / accuracy from decisions.

    By default uncertain samples are excluded from the denominator (coverage
    < 1). Set ``include_uncertain_as_error`` to force full coverage with
    uncertain counted wrong.
    """
    if len(labels_generated) != len(decisions):
        raise ValueError("labels and decisions length mismatch")

    used_labels: list[int] = []
    used_preds: list[int] = []
    uncertain = 0
    for label, decision in zip(labels_generated, decisions):
        pred_flag = decision.get("predicted_generated")
        if pred_flag is None:
            uncertain += 1
            if include_uncertain_as_error:
                # Count as wrong prediction opposite of label.
                used_labels.append(int(label))
                used_preds.append(1 - int(label))
            continue
        used_labels.append(int(label))
        used_preds.append(1 if pred_flag else 0)

    tp = sum(1 for y, p in zip(used_labels, used_preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(used_labels, used_preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(used_labels, used_preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(used_labels, used_preds) if y == 1 and p == 0)
    decided = len(used_labels)
    total = len(labels_generated)
    generated = sum(1 for y in used_labels if y == 1)
    real = sum(1 for y in used_labels if y == 0)
    return {
        "coverage": decided / total if total else 0.0,
        "uncertain_count": uncertain,
        "decided_count": decided,
        "total_count": total,
        "generated_recall": tp / generated if generated else None,
        "generated_precision": tp / (tp + fp) if (tp + fp) else None,
        "real_recall": tn / real if real else None,
        "accuracy": (tp + tn) / decided if decided else None,
        "confusion": {
            "tp_generated": tp,
            "tn_real": tn,
            "fp_real_as_generated": fp,
            "fn_generated_as_real": fn,
        },
        "include_uncertain_as_error": bool(include_uncertain_as_error),
    }
