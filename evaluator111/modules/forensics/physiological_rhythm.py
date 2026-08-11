"""Physiological facial rhythm features without manual scores.

Uses pose-normalized eye landmarks (and optional AU45 / blink blendshapes)
to estimate blink rate, asymmetry, and micro-tremor. Generated faces often
show too-uniform or too-rare blinks.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

PHYSIO_SCHEMA = "physiological_rhythm_v1"

# MediaPipe eye aperture landmarks.
LEFT_EYE_UPPER = 159
LEFT_EYE_LOWER = 145
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_UPPER = 386
RIGHT_EYE_LOWER = 374
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def eye_aspect_ratio(
    points: dict[int, np.ndarray],
    *,
    upper: int,
    lower: int,
    outer: int,
    inner: int,
) -> float | None:
    needed = (upper, lower, outer, inner)
    if any(index not in points for index in needed):
        return None
    vertical = float(np.linalg.norm(points[upper] - points[lower]))
    horizontal = float(np.linalg.norm(points[outer] - points[inner]))
    if horizontal <= 1e-8:
        return None
    return vertical / horizontal


def _detect_blinks(
    ear: np.ndarray,
    *,
    fps: float,
    threshold: float = 0.18,
    min_separation_seconds: float = 0.12,
) -> list[int]:
    if ear.size < 5:
        return []
    # Adaptive threshold: below the lower quantile of the series.
    adaptive = min(threshold, float(np.quantile(ear, 0.25)))
    candidates = np.where(ear < adaptive)[0]
    if candidates.size == 0:
        # Fall back to local minima relative to median.
        median = float(np.median(ear))
        candidates = np.where(ear < median * 0.72)[0]
    if candidates.size == 0:
        return []
    min_sep = max(1, int(round(fps * min_separation_seconds)))
    selected: list[int] = []
    cluster = [int(candidates[0])]
    for index in candidates[1:]:
        index = int(index)
        if index - cluster[-1] <= min_sep:
            cluster.append(index)
            continue
        selected.append(int(min(cluster, key=lambda i: ear[i])))
        cluster = [index]
    selected.append(int(min(cluster, key=lambda i: ear[i])))
    # Deduplicate near neighbors once more.
    cleaned: list[int] = []
    for index in selected:
        if cleaned and index - cleaned[-1] < min_sep:
            if ear[index] < ear[cleaned[-1]]:
                cleaned[-1] = index
            continue
        cleaned.append(index)
    return cleaned


def extract_physiological_rhythm_features(
    landmark_frames: Sequence[dict[int, np.ndarray]] | None = None,
    *,
    timestamps_seconds: np.ndarray | None = None,
    blink_signal: np.ndarray | None = None,
    fps_hint: float = 25.0,
) -> dict[str, Any]:
    """Extract blink / eye-rhythm features from landmarks or a blink channel."""
    left_series: list[float] = []
    right_series: list[float] = []
    if landmark_frames:
        for points in landmark_frames:
            left = eye_aspect_ratio(
                points,
                upper=LEFT_EYE_UPPER,
                lower=LEFT_EYE_LOWER,
                outer=LEFT_EYE_OUTER,
                inner=LEFT_EYE_INNER,
            )
            right = eye_aspect_ratio(
                points,
                upper=RIGHT_EYE_UPPER,
                lower=RIGHT_EYE_LOWER,
                outer=RIGHT_EYE_OUTER,
                inner=RIGHT_EYE_INNER,
            )
            if left is not None:
                left_series.append(float(left))
            if right is not None:
                right_series.append(float(right))

    if blink_signal is not None and not left_series:
        signal = np.asarray(blink_signal, dtype=np.float64)
        # AU45 / blink blendshape: high = closed. Convert to EAR-like.
        ear = 1.0 - np.clip(signal, 0.0, 1.0)
        left_series = ear.tolist()
        right_series = ear.tolist()

    if len(left_series) < 4 and len(right_series) < 4:
        return {
            "schema_version": PHYSIO_SCHEMA,
            "status": "unavailable",
            "features": {
                "physio_blink_rate_per_min": 0.0,
                "physio_blink_asymmetry_0_1": 0.5,
                "physio_ear_stability_0_1": 0.5,
                "physio_microtremor_0_1": 0.5,
                "physio_rhythm_score_0_1": 0.5,
            },
            "manual_labels_required": False,
        }

    left = np.asarray(left_series, dtype=np.float64)
    right = np.asarray(
        right_series if right_series else left_series,
        dtype=np.float64,
    )
    n = min(left.size, right.size)
    left = left[:n]
    right = right[:n]
    mean_ear = 0.5 * (left + right)

    if timestamps_seconds is not None and len(timestamps_seconds) >= n:
        stamps = np.asarray(timestamps_seconds[:n], dtype=np.float64)
        duration = max(float(stamps[-1] - stamps[0]), 1e-3)
        fps = (n - 1) / duration
    else:
        fps = float(fps_hint)
        duration = n / max(fps, 1e-3)

    blinks = _detect_blinks(mean_ear, fps=fps)
    blink_rate = (len(blinks) / duration) * 60.0
    # Natural adult blink rate roughly 8–20 / min; mild outside still ok.
    rate_score = _clamp(1.0 - abs(blink_rate - 14.0) / 14.0)

    asymmetry = float(np.mean(np.abs(left - right)))
    asymmetry_score = _clamp(1.0 - asymmetry / 0.08)

    ear_std = float(np.std(mean_ear))
    # Some variation is healthy; frozen EAR is suspicious.
    stability = _clamp(ear_std / 0.05)
    # Too chaotic also bad.
    stability = _clamp(1.0 - abs(stability - 0.55) / 0.55)

    if mean_ear.size >= 3:
        highpass = mean_ear - np.convolve(mean_ear, np.ones(3) / 3.0, mode="same")
        tremor = float(np.std(highpass))
    else:
        tremor = 0.0
    # Natural micro-motion mid-range.
    tremor_score = _clamp(1.0 - abs(tremor - 0.01) / 0.01)

    # Inter-blink interval CV when enough blinks.
    ibi_cv = 0.0
    if len(blinks) >= 3:
        intervals = np.diff(np.asarray(blinks, dtype=np.float64)) / max(fps, 1e-3)
        ibi_cv = float(np.std(intervals) / max(float(np.mean(intervals)), 1e-3))
    ibi_score = _clamp(ibi_cv / 0.45) if len(blinks) >= 3 else 0.45

    score = _clamp(
        0.34 * rate_score
        + 0.22 * asymmetry_score
        + 0.18 * stability
        + 0.14 * tremor_score
        + 0.12 * ibi_score
    )
    return {
        "schema_version": PHYSIO_SCHEMA,
        "status": "available",
        "blink_count": len(blinks),
        "duration_seconds": float(duration),
        "features": {
            "physio_blink_rate_per_min": float(blink_rate),
            "physio_blink_count": float(len(blinks)),
            "physio_blink_asymmetry": asymmetry,
            "physio_blink_asymmetry_0_1": asymmetry_score,
            "physio_ear_mean": float(np.mean(mean_ear)),
            "physio_ear_std": ear_std,
            "physio_ear_stability_0_1": stability,
            "physio_microtremor": tremor,
            "physio_microtremor_0_1": tremor_score,
            "physio_ibi_cv": ibi_cv,
            "physio_rhythm_score_0_1": score,
        },
        "manual_labels_required": False,
        "note": (
            "Blink / eye-aperture physiological rhythm from landmarks "
            "(or blink channel). No manual scores required."
        ),
    }


def merge_physio_into_motion_features(
    motion_features: dict[str, Any],
    physio_result: dict[str, Any],
) -> dict[str, Any]:
    features = dict(motion_features.get("features", {}))
    physio_features = dict(physio_result.get("features", {}))
    features.update(physio_features)
    prior = _finite(features.get("training_free_motion_prior_0_1"), 0.5)
    physio = _finite(physio_features.get("physio_rhythm_score_0_1"), 0.5)
    features["training_free_motion_prior_0_1"] = _clamp(0.84 * prior + 0.16 * physio)
    enriched = dict(motion_features)
    enriched["features"] = features
    enriched["physiological_rhythm"] = {
        "schema_version": physio_result.get("schema_version"),
        "status": physio_result.get("status"),
        "blink_count": physio_result.get("blink_count"),
        "note": physio_result.get("note"),
    }
    return enriched
