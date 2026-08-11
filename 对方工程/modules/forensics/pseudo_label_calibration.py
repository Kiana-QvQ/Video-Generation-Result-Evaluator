"""Automatic pseudo-label calibration without manual per-clip scores.

Pipeline:
1. Score each clip with multiple automatic evidence sources.
2. Use source labels (real vs generated) and/or multi-model agreement as
   pseudo-labels.
3. Fit a held-out Platt calibrator; never train and calibrate on the same ids.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from .seedance_authenticity import (
    apply_probability_calibrator,
    fit_probability_calibrator,
)

PSEUDO_CALIBRATION_SCHEMA = "pseudo_label_calibration_v1"


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


def consensus_pseudo_label(
    model_scores: Sequence[float],
    *,
    agreement_threshold: float = 0.15,
    high_confidence_threshold: float = 0.62,
) -> dict[str, Any]:
    """Build a pseudo-label from multi-model score agreement."""
    values = [
        float(score)
        for score in model_scores
        if math.isfinite(float(score))
    ]
    if len(values) < 2:
        return {
            "status": "insufficient_models",
            "pseudo_label": None,
            "confidence_0_1": 0.0,
            "mean_score_0_1": values[0] if values else None,
            "score_std": 0.0,
            "agreement": False,
        }
    mean_score = float(np.mean(values))
    score_std = float(np.std(values))
    agreement = score_std <= agreement_threshold
    if mean_score >= high_confidence_threshold:
        label = 1
    elif mean_score <= 1.0 - high_confidence_threshold:
        label = 0
    else:
        label = None
    confidence = _clamp(
        (1.0 - score_std / max(agreement_threshold * 2.0, 1e-6))
        * (0.55 + 0.45 * abs(mean_score - 0.5) * 2.0)
    )
    if not agreement or label is None:
        return {
            "status": "low_agreement",
            "pseudo_label": None,
            "confidence_0_1": confidence,
            "mean_score_0_1": mean_score,
            "score_std": score_std,
            "agreement": agreement,
        }
    return {
        "status": "high_confidence",
        "pseudo_label": int(label),
        "confidence_0_1": confidence,
        "mean_score_0_1": mean_score,
        "score_std": score_std,
        "agreement": True,
    }


def build_pseudo_labeled_samples(
    records: Iterable[dict[str, Any]],
    *,
    score_keys: Sequence[str] = (
        "facial_expression_muscle_score_0_1",
        "texture_detail_score_0_1",
        "nr_vqa_score_0_1",
        "ssl_au_score_0_1",
        "training_free_prior_0_1",
    ),
    source_label_key: str = "source_label",
    raw_score_key: str = "raw_real_domain_evidence_0_1",
    min_confidence: float = 0.55,
) -> dict[str, Any]:
    """Convert scored records into pseudo-labeled calibration candidates.

    ``source_label`` may be ``real`` / ``generated``. When present it is used
    as a hard label. Otherwise multi-model consensus is used.
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record.get("id", record.get("source", "unknown")))
        raw_score = _finite(record.get(raw_score_key))
        if raw_score is None:
            # Fall back to mean of available automatic scores.
            auto_scores = [
                _finite(record.get(key))
                for key in score_keys
            ]
            auto_scores = [score for score in auto_scores if score is not None]
            raw_score = float(np.mean(auto_scores)) if auto_scores else None
        if raw_score is None:
            rejected.append(
                {
                    "id": sample_id,
                    "reason": "missing_raw_score",
                }
            )
            continue

        source_label = record.get(source_label_key)
        if source_label in {"real", "real_capture", 1, "1", True}:
            label = 1
            label_source = "source_label"
            confidence = 1.0
            consensus = None
        elif source_label in {"generated", "seedance", 0, "0", False}:
            label = 0
            label_source = "source_label"
            confidence = 1.0
            consensus = None
        else:
            model_scores = [
                score
                for key in score_keys
                for score in [_finite(record.get(key))]
                if score is not None
            ]
            consensus = consensus_pseudo_label(model_scores)
            if consensus.get("status") != "high_confidence":
                rejected.append(
                    {
                        "id": sample_id,
                        "reason": consensus.get("status", "low_agreement"),
                        "consensus": consensus,
                    }
                )
                continue
            label = int(consensus["pseudo_label"])
            label_source = "multi_model_consensus"
            confidence = float(consensus["confidence_0_1"])

        if confidence < min_confidence:
            rejected.append(
                {
                    "id": sample_id,
                    "reason": "below_min_confidence",
                    "confidence_0_1": confidence,
                }
            )
            continue
        accepted.append(
            {
                "id": sample_id,
                "raw_score": float(raw_score),
                "label": int(label),
                "confidence_0_1": confidence,
                "label_source": label_source,
                "consensus": consensus,
                "holdout": bool(record.get("holdout", False)),
            }
        )
    return {
        "schema_version": PSEUDO_CALIBRATION_SCHEMA,
        "accepted": accepted,
        "rejected": rejected,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
    }


def fit_pseudo_label_calibrator(
    samples: Sequence[dict[str, Any]],
    *,
    require_holdout: bool = True,
    min_per_class: int = 4,
) -> dict[str, Any]:
    """Fit Platt calibration on holdout pseudo-labeled samples."""
    if require_holdout:
        candidates = [sample for sample in samples if sample.get("holdout")]
        split = "declared_holdout"
    else:
        candidates = list(samples)
        split = "all_accepted_pseudo_labels"
    real_scores = [
        float(sample["raw_score"])
        for sample in candidates
        if int(sample.get("label", -1)) == 1
    ]
    generated_scores = [
        float(sample["raw_score"])
        for sample in candidates
        if int(sample.get("label", -1)) == 0
    ]
    if len(real_scores) < min_per_class or len(generated_scores) < min_per_class:
        return {
            "schema_version": PSEUDO_CALIBRATION_SCHEMA,
            "status": "provisional",
            "reason": "insufficient_holdout_class_counts",
            "real_count": len(real_scores),
            "generated_count": len(generated_scores),
            "min_per_class": min_per_class,
            "split_protocol": split,
        }
    calibrator = fit_probability_calibrator(real_scores, generated_scores)
    calibrator = dict(calibrator)
    calibrator.update(
        {
            "schema_version": PSEUDO_CALIBRATION_SCHEMA,
            "labeling_method": "source_labels_or_multi_model_consensus",
            "manual_scores_required": False,
            "split_protocol": split,
            "sample_count": len(candidates),
        }
    )
    # Diagnostics on the same holdout set (not a claim of absolute accuracy).
    labels = [1] * len(real_scores) + [0] * len(generated_scores)
    scores = real_scores + generated_scores
    probabilities = [
        apply_probability_calibrator(score, calibrator) or 0.5
        for score in scores
    ]
    brier = float(
        np.mean(
            [
                (probability - label) ** 2
                for probability, label in zip(probabilities, labels)
            ]
        )
    )
    calibrator["holdout_brier"] = brier
    calibrator["status"] = "ready"
    return calibrator


def apply_pseudo_calibrator(
    raw_score: float | None,
    calibrator: dict[str, Any] | None,
) -> float | None:
    if calibrator is None or calibrator.get("status") not in {"ready", "calibrated"}:
        return None
    return apply_probability_calibrator(raw_score, calibrator)
