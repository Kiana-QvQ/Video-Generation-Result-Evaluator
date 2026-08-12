"""Learned hard-fusion head for real-vs-generated detection.

Trains a compact logistic / gradient-boosting classifier on non-holdout
branch evidence (Wang Xing source + forensics facial-motion metrics), then
emits a calibrated real-probability for hard labeling.

No texture / MUSIQ branch: earlier holdout runs showed it collapses
generated recall.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

LEARNED_HEAD_SCHEMA = "learned_fusion_head_v1"

FEATURE_NAMES: tuple[str, ...] = (
    "wx_real_probability_0_1",
    "wx_generated_probability_0_1",
    "wx_margin_0_1",
    "wx_real_distance",
    "wx_generated_distance",
    "wx_real_score_0_1",
    "wx_generated_score_0_1",
    "wx_valid_frame_ratio",
    "fm_real_domain_fit_0_1",
    "fm_seedance_domain_fit_0_1",
    "fm_raw_real_domain_evidence_0_1",
    "fm_motion_coherence_0_1",
    "fm_au_relation_consistency_0_1",
    "fm_au_dynamics_naturalness_0_1",
    "fm_training_free_motion_prior_0_1",
    "fm_ssl_au_score_0_1",
    "fm_ssl_backbone_score_0_1",
    "fm_ssl_temporal_consistency_0_1",
    "fm_physio_rhythm_score_0_1",
    "fm_input_quality_gate_0_1",
    "fm_landmark_valid_frame_ratio",
    "fm_pose_normalized_frame_ratio",
    "branch_gap_wx_minus_fm",
    "branch_mean_real",
    "quality_min",
)


def _finite(value: Any, default: float = 0.5) -> float:
    if value is None:
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def extract_fusion_feature_dict(
    *,
    au_path: str | Path,
    wangxing_source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
) -> dict[str, float]:
    """Extract compact branch features for one AU CSV."""
    from ..wangxing.wangxing_specialization import score_source_profile
    from .facial_motion import score_facial_motion

    wangxing = score_source_profile(au_path, wangxing_source_profile)
    wx_scores = wangxing.get("scores") or {}
    wx_quality = wangxing.get("quality") or {}
    wx_real = _finite(wangxing.get("real_probability_0_1"), 0.5)
    wx_gen = _finite(wangxing.get("generated_probability_0_1"), 0.5)
    wx_margin = _finite(wangxing.get("margin_0_1"), abs(wx_real - wx_gen))
    wx_real_blob = wx_scores.get("real_wangxing") or {}
    wx_gen_blob = wx_scores.get("generated_wangxing") or {}

    motion_profile = forensics_profiles.get("facial_motion") or {}
    motion = score_facial_motion(au_path, motion_profile)
    metrics = motion.get("metrics") or {}

    fm_real = _finite(metrics.get("raw_real_domain_evidence_0_1"), 0.5)
    quality_fm = _finite(metrics.get("input_quality_gate_0_1"), 0.5)
    quality_wx = _finite(wx_quality.get("valid_frame_ratio"), 0.5)

    return {
        "wx_real_probability_0_1": wx_real,
        "wx_generated_probability_0_1": wx_gen,
        "wx_margin_0_1": wx_margin,
        "wx_real_distance": _finite(wx_real_blob.get("distance"), 1.0),
        "wx_generated_distance": _finite(wx_gen_blob.get("distance"), 1.0),
        "wx_real_score_0_1": _finite(wx_real_blob.get("score_0_1"), wx_real),
        "wx_generated_score_0_1": _finite(wx_gen_blob.get("score_0_1"), wx_gen),
        "wx_valid_frame_ratio": quality_wx,
        "fm_real_domain_fit_0_1": _finite(metrics.get("real_domain_fit_0_1"), 0.5),
        "fm_seedance_domain_fit_0_1": _finite(
            metrics.get("seedance_domain_fit_0_1"), 0.5
        ),
        "fm_raw_real_domain_evidence_0_1": fm_real,
        "fm_motion_coherence_0_1": _finite(
            metrics.get("motion_coherence_0_1"), 0.5
        ),
        "fm_au_relation_consistency_0_1": _finite(
            metrics.get("au_relation_consistency_0_1"), 0.5
        ),
        "fm_au_dynamics_naturalness_0_1": _finite(
            metrics.get("au_dynamics_naturalness_0_1"), 0.5
        ),
        "fm_training_free_motion_prior_0_1": _finite(
            metrics.get("training_free_motion_prior_0_1"), 0.5
        ),
        "fm_ssl_au_score_0_1": _finite(metrics.get("ssl_au_score_0_1"), 0.5),
        "fm_ssl_backbone_score_0_1": _finite(
            metrics.get("ssl_backbone_score_0_1"), 0.5
        ),
        "fm_ssl_temporal_consistency_0_1": _finite(
            metrics.get("ssl_temporal_consistency_0_1"), 0.5
        ),
        "fm_physio_rhythm_score_0_1": _finite(
            metrics.get("physio_rhythm_score_0_1"), 0.5
        ),
        "fm_input_quality_gate_0_1": quality_fm,
        "fm_landmark_valid_frame_ratio": _finite(
            metrics.get("landmark_valid_frame_ratio"), 0.5
        ),
        "fm_pose_normalized_frame_ratio": _finite(
            metrics.get("pose_normalized_frame_ratio"), 0.5
        ),
        "branch_gap_wx_minus_fm": wx_real - fm_real,
        "branch_mean_real": 0.5 * (wx_real + fm_real),
        "quality_min": min(quality_wx, quality_fm),
    }


def feature_vector_from_dict(features: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [_finite(features.get(name), 0.5) for name in FEATURE_NAMES],
        dtype=np.float64,
    )


def extract_fusion_features(
    *,
    au_path: str | Path,
    wangxing_source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    feature_dict = extract_fusion_feature_dict(
        au_path=au_path,
        wangxing_source_profile=wangxing_source_profile,
        forensics_profiles=forensics_profiles,
    )
    return feature_vector_from_dict(feature_dict), feature_dict


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {
        "generated_recall": (tp / (tp + fn)) if tp + fn else 0.0,
        "real_recall": (tn / (tn + fp)) if tn + fp else 0.0,
        "generated_precision": (tp / (tp + fp)) if tp + fp else 0.0,
        "overall_accuracy": ((tp + tn) / len(y_true)) if len(y_true) else 0.0,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def select_threshold(
    y_true: np.ndarray,
    real_probs: np.ndarray,
    *,
    target_metric: float = 0.75,
) -> tuple[float, dict[str, float]]:
    """Pick threshold on P(real): generated when score < threshold.

    Prefer points that hit both generated_recall and accuracy >= target.
    Otherwise maximize min(gen_recall, accuracy).
    """
    best_hit: tuple[float, dict[str, float]] | None = None
    best_soft: tuple[float, dict[str, float], float] | None = None
    for step in range(20, 86):
        threshold = step / 100.0
        pred = (real_probs < threshold).astype(np.int32)
        metrics = _metrics(y_true, pred)
        score = min(metrics["generated_recall"], metrics["overall_accuracy"])
        if (
            metrics["generated_recall"] >= target_metric
            and metrics["overall_accuracy"] >= target_metric
        ):
            # Prefer higher accuracy then higher recall among hits.
            key = (
                metrics["overall_accuracy"],
                metrics["generated_recall"],
                metrics["real_recall"],
            )
            if best_hit is None or key > (
                best_hit[1]["overall_accuracy"],
                best_hit[1]["generated_recall"],
                best_hit[1]["real_recall"],
            ):
                best_hit = (threshold, metrics)
        if best_soft is None or score > best_soft[2]:
            best_soft = (threshold, metrics, score)
    if best_hit is not None:
        return best_hit
    assert best_soft is not None
    return best_soft[0], best_soft[1]


def fit_learned_fusion_head(
    features: np.ndarray,
    labels_generated: np.ndarray,
    *,
    model_type: str = "hist_gbdt",
    random_state: int = 42,
    hard_example_rounds: int = 2,
    target_metric: float = 0.75,
) -> dict[str, Any]:
    """Fit a classifier that predicts P(real)=1-P(generated)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight

    matrix = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels_generated, dtype=np.int32)
    if matrix.ndim != 2 or matrix.shape[0] != labels.shape[0]:
        raise ValueError("features/labels shape mismatch")
    if matrix.shape[0] < 8:
        raise ValueError("Need at least 8 training rows")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    sample_weight = compute_sample_weight("balanced", labels)

    if model_type == "logistic":
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )
        model.fit(scaled, labels, sample_weight=sample_weight)
    else:
        model = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.06,
            max_iter=180,
            l2_regularization=0.2,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=12,
        )
        model.fit(scaled, labels, sample_weight=sample_weight)

    for _ in range(max(0, hard_example_rounds)):
        proba_gen = model.predict_proba(scaled)[:, 1]
        # Hard generated (look too real) and hard real (look generated).
        hard_gen = (labels == 1) & (proba_gen < 0.55)
        hard_real = (labels == 0) & (proba_gen > 0.45)
        if not np.any(hard_gen) and not np.any(hard_real):
            break
        boost = np.ones(len(labels), dtype=np.float64)
        boost[hard_gen] *= 2.5
        boost[hard_real] *= 2.0
        sample_weight = compute_sample_weight("balanced", labels) * boost
        model.fit(scaled, labels, sample_weight=sample_weight)

    proba_gen = model.predict_proba(scaled)[:, 1]
    real_probs = 1.0 - proba_gen
    threshold, train_metrics = select_threshold(
        labels, real_probs, target_metric=target_metric
    )

    payload: dict[str, Any] = {
        "schema_version": LEARNED_HEAD_SCHEMA,
        "model_type": model_type,
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "threshold": float(threshold),
        "target_metric": float(target_metric),
        "train_metrics": train_metrics,
        "train_counts": {
            "n": int(len(labels)),
            "real": int((labels == 0).sum()),
            "generated": int((labels == 1).sum()),
        },
        "manual_scores_required": False,
        "uncertain_band_used": False,
        "include_texture": False,
        "note": (
            "Predicts P(generated) then converts to real_score=1-P(gen). "
            "Hard label generated when real_score < threshold."
        ),
    }

    if model_type == "logistic":
        payload["coef"] = model.coef_.astype(float).reshape(-1).tolist()
        payload["intercept"] = float(model.intercept_.reshape(-1)[0])
    else:
        # Persist via joblib next to JSON metadata when saving.
        payload["_sklearn_model"] = model
        payload["_scaler"] = scaler
    return payload


def _apply_scaler(vector: np.ndarray, head: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(head["scaler_mean"], dtype=np.float64)
    scale = np.asarray(head["scaler_scale"], dtype=np.float64)
    scale = np.where(np.abs(scale) < 1e-8, 1.0, scale)
    return (vector - mean) / scale


def predict_real_probability(
    features: np.ndarray,
    head: dict[str, Any],
) -> float:
    vector = np.asarray(features, dtype=np.float64).reshape(-1)
    names = head.get("feature_names") or list(FEATURE_NAMES)
    if vector.size != len(names):
        raise ValueError(
            f"Feature size mismatch: got {vector.size}, expected {len(names)}"
        )
    scaled = _apply_scaler(vector, head)
    model_type = head.get("model_type", "logistic")
    if model_type == "logistic":
        coef = np.asarray(head["coef"], dtype=np.float64)
        intercept = float(head["intercept"])
        logit = float(np.dot(scaled, coef) + intercept)
        # coef trained for P(generated=1)
        p_gen = float(_sigmoid(np.asarray([logit]))[0])
        return _clamp(1.0 - p_gen)
    model = head.get("_sklearn_model")
    if model is None:
        raise ValueError("GBDT head missing in-memory sklearn model; load via load_learned_head")
    p_gen = float(model.predict_proba(scaled.reshape(1, -1))[0, 1])
    return _clamp(1.0 - p_gen)


def score_with_learned_head(
    *,
    au_path: str | Path,
    wangxing_source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
    learned_head: dict[str, Any],
    hard_threshold: float | None = None,
) -> dict[str, Any]:
    from .authenticity_decision import decide_real_vs_generated

    vector, feature_dict = extract_fusion_features(
        au_path=au_path,
        wangxing_source_profile=wangxing_source_profile,
        forensics_profiles=forensics_profiles,
    )
    real_score = predict_real_probability(vector, learned_head)
    threshold = (
        float(hard_threshold)
        if hard_threshold is not None
        else float(learned_head.get("threshold", 0.5))
    )
    quality = _finite(feature_dict.get("quality_min"), 0.5)
    decision = decide_real_vs_generated(
        real_score_0_1=real_score,
        quality_0_1=quality,
        hard_threshold=threshold,
        allow_uncertain=False,
    )
    return {
        "schema_version": LEARNED_HEAD_SCHEMA,
        "status": "available",
        "decision_score_0_1": real_score,
        "hard_decision": decision,
        "predicted_generated": decision.get("predicted_generated"),
        "threshold": threshold,
        "features": feature_dict,
        "quality_0_1": quality,
        "manual_scores_required": False,
        "uncertain_band_used": False,
    }


def save_learned_head(head: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: value
        for key, value in head.items()
        if not str(key).startswith("_")
    }
    output.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    model = head.get("_sklearn_model")
    if model is not None:
        import joblib

        bundle = {
            "model": model,
            "scaler_mean": head["scaler_mean"],
            "scaler_scale": head["scaler_scale"],
            "feature_names": head["feature_names"],
            "model_type": head["model_type"],
            "threshold": head["threshold"],
        }
        joblib.dump(bundle, output.with_suffix(".joblib"))
    return output


def load_learned_head(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    head = json.loads(path.read_text(encoding="utf-8-sig"))
    if head.get("model_type") != "logistic":
        import joblib

        joblib_path = path.with_suffix(".joblib")
        if not joblib_path.is_file():
            raise FileNotFoundError(
                f"Missing companion model file for GBDT head: {joblib_path}"
            )
        bundle = joblib.load(joblib_path)
        head["_sklearn_model"] = bundle["model"]
        head.setdefault("scaler_mean", bundle["scaler_mean"])
        head.setdefault("scaler_scale", bundle["scaler_scale"])
    return head
