"""Self-supervised AU / temporal-consistency features (TCAE / VideoMAE style).

No manual AU labels are required. Features measure:

1. Temporal reconstruction consistency of AU trajectories (TCAE-like).
2. Frame-to-frame muscle change smoothness and predictability.
3. AU co-activation graph stability across time.
4. Occluded / dropped-frame prediction residual (AU-vMAE style probe).

These are training-free proxies intended to enrich facial-motion evidence.
They do not claim supervised AU detection accuracy.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

AU_SSL_SCHEMA = "au_self_supervised_temporal_v1"


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 3 or right.size != left.size:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / denom)


def _linear_predict_next(series: np.ndarray) -> np.ndarray:
    """One-step linear autoregressive prediction residual proxy."""
    series = np.asarray(series, dtype=np.float64)
    if series.ndim == 1:
        series = series[:, None]
    if series.shape[0] < 4:
        return np.zeros(series.shape[1], dtype=np.float64)
    # X_t predicts X_{t+1} with ridge-free least squares on lagged window.
    previous = series[:-1]
    nxt = series[1:]
    # Per-channel local slope from first differences.
    predicted = previous + np.median(np.diff(series, axis=0), axis=0, keepdims=True)
    residual = np.mean(np.abs(nxt - predicted), axis=0)
    return residual.astype(np.float64)


def _masked_frame_prediction(
    series: np.ndarray,
    *,
    mask_ratio: float = 0.25,
    seed: int = 0,
) -> float:
    """AU-vMAE style probe: reconstruct randomly dropped frames by lerp."""
    series = np.asarray(series, dtype=np.float64)
    if series.ndim == 1:
        series = series[:, None]
    frame_count = series.shape[0]
    if frame_count < 6:
        return 0.0
    rng = np.random.default_rng(seed)
    mask_count = max(1, int(round(frame_count * mask_ratio)))
    # Avoid endpoints so neighbors exist.
    candidates = np.arange(1, frame_count - 1)
    if candidates.size == 0:
        return 0.0
    masked = rng.choice(
        candidates,
        size=min(mask_count, candidates.size),
        replace=False,
    )
    errors: list[float] = []
    for index in masked:
        left = series[index - 1]
        right = series[index + 1]
        prediction = 0.5 * (left + right)
        errors.append(float(np.mean(np.abs(series[index] - prediction))))
    return float(np.mean(errors)) if errors else 0.0


def _tube_masked_prediction(
    series: np.ndarray,
    *,
    tube_ratio: float = 0.20,
    seed: int = 1,
) -> float:
    """VideoMAE-style contiguous tube mask: reconstruct a temporal segment."""
    series = np.asarray(series, dtype=np.float64)
    if series.ndim == 1:
        series = series[:, None]
    frame_count = series.shape[0]
    if frame_count < 8:
        return 0.0
    tube_len = max(2, int(round(frame_count * tube_ratio)))
    tube_len = min(tube_len, frame_count - 2)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(1, max(2, frame_count - tube_len)))
    stop = start + tube_len
    left = series[start - 1]
    right = series[min(stop, frame_count - 1)]
    errors: list[float] = []
    for offset, index in enumerate(range(start, stop)):
        alpha = (offset + 1) / (tube_len + 1)
        prediction = (1.0 - alpha) * left + alpha * right
        errors.append(float(np.mean(np.abs(series[index] - prediction))))
    return float(np.mean(errors)) if errors else 0.0


def _multiscale_reconstruction_error(series: np.ndarray) -> float:
    """TCAE-like multi-scale temporal autoencoder proxy (kernel 3/5/7)."""
    series = np.asarray(series, dtype=np.float64)
    if series.ndim == 1:
        series = series[:, None]
    if series.shape[0] < 3:
        return 0.0
    errors: list[float] = []
    for kernel in (3, 5, 7):
        if series.shape[0] < kernel:
            continue
        pad = kernel // 2
        padded = np.pad(series, ((pad, pad), (0, 0)), mode="edge")
        reconstructed = np.asarray(
            [
                padded[index : index + kernel].mean(axis=0)
                for index in range(series.shape[0])
            ],
            dtype=np.float64,
        )
        errors.append(float(np.mean(np.abs(series - reconstructed))))
    return float(np.mean(errors)) if errors else 0.0


def _au_coactivation_consistency(au_matrix: np.ndarray) -> float:
    """Score whether known synergistic AU pairs move together over time."""
    matrix = np.asarray(au_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 4 or matrix.shape[1] < 2:
        return 0.5
    # Adjacent channel pairs as a label-free co-activation prior.
    pair_scores: list[float] = []
    for left, right in zip(matrix.T, matrix.T[1:]):
        pair_scores.append((_safe_corr(left, right) + 1.0) / 2.0)
    if not pair_scores:
        return 0.5
    # Prefer moderate positive coupling without perfect lockstep.
    mean_agreement = float(np.mean(pair_scores))
    return _clamp(1.0 - abs(mean_agreement - 0.62) / 0.62)


def _coactivation_stability(au_matrix: np.ndarray) -> float:
    """Compare early/late AU correlation graphs (temporal consistency)."""
    matrix = np.asarray(au_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 8 or matrix.shape[1] < 2:
        return 0.5
    split = matrix.shape[0] // 2
    early = matrix[:split]
    late = matrix[split:]

    def corr_matrix(block: np.ndarray) -> np.ndarray:
        centered = block - block.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=0)
        norms = np.where(norms < 1e-8, 1.0, norms)
        normalized = centered / norms
        return normalized.T @ normalized / max(block.shape[0], 1)

    early_corr = corr_matrix(early)
    late_corr = corr_matrix(late)
    upper = np.triu_indices(early_corr.shape[0], k=1)
    return _clamp(( _safe_corr(early_corr[upper], late_corr[upper]) + 1.0) / 2.0)


def extract_self_supervised_au_features(
    au_matrix: np.ndarray,
    *,
    timestamps_seconds: np.ndarray | None = None,
    blendshape_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    """Extract TCAE / VideoMAE-inspired unsupervised temporal AU features."""
    matrix = np.asarray(au_matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    features: dict[str, float] = {}
    if matrix.size == 0 or matrix.shape[0] < 2:
        return {
            "schema_version": AU_SSL_SCHEMA,
            "status": "unavailable",
            "features": {
                "ssl_temporal_consistency_0_1": 0.5,
                "ssl_predictability_0_1": 0.5,
                "ssl_coactivation_stability_0_1": 0.5,
                "ssl_coactivation_consistency_0_1": 0.5,
                "ssl_occlusion_probe_0_1": 0.5,
                "ssl_videomae_tube_probe_0_1": 0.5,
                "ssl_dynamics_naturalness_0_1": 0.5,
                "ssl_au_score_0_1": 0.5,
            },
            "manual_au_labels_required": False,
        }

    # Temporal reconstruction via multi-scale moving-average autoencoder proxy.
    reconstruction_error = _multiscale_reconstruction_error(matrix)
    features["ssl_reconstruction_error"] = reconstruction_error
    features["ssl_temporal_consistency_0_1"] = _clamp(
        1.0 - reconstruction_error * 4.0
    )

    prediction_residual = _linear_predict_next(matrix)
    features["ssl_prediction_residual_mean"] = float(np.mean(prediction_residual))
    features["ssl_predictability_0_1"] = _clamp(
        1.0 - float(np.mean(prediction_residual)) * 5.0
    )

    velocity = np.diff(matrix, axis=0)
    acceleration = np.diff(velocity, axis=0) if velocity.shape[0] > 1 else velocity
    jerk = np.diff(acceleration, axis=0) if acceleration.shape[0] > 1 else acceleration
    velocity_energy = float(np.mean(np.abs(velocity)))
    jerk_energy = float(np.mean(np.abs(jerk))) if jerk.size else 0.0
    features["ssl_velocity_energy"] = velocity_energy
    features["ssl_jerk_energy"] = jerk_energy
    # Natural muscle motion: some velocity, limited jerk spikes.
    features["ssl_dynamics_naturalness_0_1"] = _clamp(
        0.55 * _clamp(velocity_energy * 6.0)
        + 0.45 * _clamp(1.0 - jerk_energy * 12.0)
    )

    coactivation = _coactivation_stability(matrix)
    features["ssl_coactivation_stability_0_1"] = coactivation
    features["ssl_coactivation_consistency_0_1"] = _au_coactivation_consistency(
        matrix
    )

    occlusion_error = _masked_frame_prediction(matrix)
    features["ssl_occlusion_probe_error"] = occlusion_error
    features["ssl_occlusion_probe_0_1"] = _clamp(1.0 - occlusion_error * 5.0)

    tube_error = _tube_masked_prediction(matrix)
    features["ssl_videomae_tube_probe_error"] = tube_error
    features["ssl_videomae_tube_probe_0_1"] = _clamp(1.0 - tube_error * 4.5)

    if timestamps_seconds is not None:
        stamps = np.asarray(timestamps_seconds, dtype=np.float64)
        if stamps.size == matrix.shape[0] and stamps.size >= 2:
            deltas = np.diff(stamps)
            features["ssl_timestamp_irregularity"] = float(
                np.std(deltas) / max(float(np.mean(deltas)), 1e-6)
            )
        else:
            features["ssl_timestamp_irregularity"] = 0.0
    else:
        features["ssl_timestamp_irregularity"] = 0.0

    if blendshape_matrix is not None:
        blend = np.asarray(blendshape_matrix, dtype=np.float64)
        if blend.ndim == 2 and blend.shape[0] == matrix.shape[0] and blend.size:
            # Cross-modal temporal agreement between AU proxy and blendshapes.
            au_energy = np.linalg.norm(matrix, axis=1)
            blend_energy = np.linalg.norm(blend, axis=1)
            agreement = (_safe_corr(au_energy, blend_energy) + 1.0) / 2.0
            features["ssl_blendshape_agreement_0_1"] = _clamp(agreement)
        else:
            features["ssl_blendshape_agreement_0_1"] = 0.5
    else:
        features["ssl_blendshape_agreement_0_1"] = 0.5

    score = _clamp(
        0.22 * features["ssl_temporal_consistency_0_1"]
        + 0.16 * features["ssl_predictability_0_1"]
        + 0.14 * features["ssl_coactivation_stability_0_1"]
        + 0.10 * features["ssl_coactivation_consistency_0_1"]
        + 0.14 * features["ssl_occlusion_probe_0_1"]
        + 0.12 * features["ssl_videomae_tube_probe_0_1"]
        + 0.12 * features["ssl_dynamics_naturalness_0_1"]
    )
    features["ssl_au_score_0_1"] = score
    return {
        "schema_version": AU_SSL_SCHEMA,
        "status": "available",
        "frame_count": int(matrix.shape[0]),
        "channel_count": int(matrix.shape[1]),
        "features": features,
        "manual_au_labels_required": False,
        "note": (
            "Training-free self-supervised AU temporal proxies inspired by "
            "TCAE / VideoMAE / AU-vMAE (multi-scale reconstruction, random "
            "frame mask, contiguous tube mask, co-activation stability). "
            "They do not require manual AU labels and are not a substitute "
            "for supervised AU accuracy claims."
        ),
    }


def merge_ssl_into_motion_features(
    motion_features: dict[str, Any],
    ssl_result: dict[str, Any],
) -> dict[str, Any]:
    """Attach SSL metrics and mildly enrich the training-free motion prior."""
    features = dict(motion_features.get("features", {}))
    ssl_features = dict(ssl_result.get("features", {}))
    features.update(ssl_features)
    prior = _finite(features.get("training_free_motion_prior_0_1"), 0.5)
    ssl_score = _finite(ssl_features.get("ssl_au_score_0_1"), 0.5)
    features["training_free_motion_prior_0_1"] = _clamp(
        0.78 * prior + 0.22 * ssl_score
    )
    features["ssl_enrichment_applied"] = 1.0
    enriched = dict(motion_features)
    enriched["features"] = features
    enriched["self_supervised_au"] = {
        "schema_version": ssl_result.get("schema_version"),
        "status": ssl_result.get("status"),
        "note": ssl_result.get("note"),
    }
    return enriched
