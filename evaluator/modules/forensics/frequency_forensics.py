"""Frequency-domain / compression-artifact forensics (no reference video).

Measures 8x8 DCT grid periodicity, radial FFT spectrum shape, and temporal
high-frequency flicker. Useful for generated / re-encoded video cues without
VMAF or human MOS labels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.video_metrics import sample_video_frames

FREQ_FORENSICS_SCHEMA = "frequency_forensics_v1"


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _safe_mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else 0.0


def _frames_from_input(
    frames_or_video: Sequence[np.ndarray] | str | Path,
    *,
    max_frames: int,
    sample_fps: float,
) -> list[np.ndarray]:
    if isinstance(frames_or_video, (str, Path)):
        _, _, _, frames = sample_video_frames(
            frames_or_video,
            max_frames,
            sample_fps,
        )
        return list(frames)
    return list(frames_or_video)[:max_frames]


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float32)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _block_dct_energy(gray: np.ndarray, block: int = 8) -> dict[str, float]:
    height, width = gray.shape
    h = height - (height % block)
    w = width - (width % block)
    if h < block or w < block:
        return {
            "dct_high_ratio": 0.0,
            "dct_grid_periodicity": 0.0,
            "dct_mid_energy": 0.0,
        }
    crop = gray[:h, :w]
    high_vals: list[float] = []
    mid_vals: list[float] = []
    low_vals: list[float] = []
    # Collect AC energy maps on a coarse grid for periodicity.
    ac_map = []
    for y in range(0, h, block):
        row = []
        for x in range(0, w, block):
            patch = crop[y : y + block, x : x + block]
            coeff = cv2.dct(patch)
            power = coeff * coeff
            low = float(power[0, 0])
            mid = float(np.sum(power[0:4, 0:4]) - low)
            high = float(np.sum(power) - np.sum(power[0:4, 0:4]))
            total = max(low + mid + high, 1e-6)
            low_vals.append(low / total)
            mid_vals.append(mid / total)
            high_vals.append(high / total)
            row.append(high / total)
        ac_map.append(row)
    ac = np.asarray(ac_map, dtype=np.float32)
    # Grid periodicity via horizontal/vertical autocorrelation peak at lag=1 block.
    periodicity = 0.0
    if ac.shape[1] >= 3:
        flat = ac - float(np.mean(ac))
        horiz = float(np.mean(flat[:, 1:] * flat[:, :-1]))
        vert = float(np.mean(flat[1:, :] * flat[:-1, :]))
        denom = float(np.mean(flat * flat) + 1e-6)
        periodicity = _clamp(0.5 * (horiz + vert) / denom)
    return {
        "dct_high_ratio": _safe_mean(high_vals),
        "dct_mid_energy": _safe_mean(mid_vals),
        "dct_low_energy": _safe_mean(low_vals),
        "dct_grid_periodicity": periodicity,
    }


def _radial_fft_features(gray: np.ndarray) -> dict[str, float]:
    # Downscale for speed / stability.
    resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    windowed = resized * np.outer(
        np.hanning(128),
        np.hanning(128),
    ).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(windowed))
    magnitude = np.abs(spectrum)
    cy, cx = 64, 64
    yy, xx = np.ogrid[:128, :128]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    bands = [
        ((0.0, 8.0), "fft_low"),
        ((8.0, 24.0), "fft_mid"),
        ((24.0, 64.0), "fft_high"),
    ]
    total = float(np.sum(magnitude) + 1e-6)
    features: dict[str, float] = {}
    for (lo, hi), name in bands:
        mask = (radius >= lo) & (radius < hi)
        features[f"{name}_energy_ratio"] = float(np.sum(magnitude[mask]) / total)
    # Spectrum slope proxy: high vs mid.
    features["fft_high_mid_ratio"] = float(
        features["fft_high_energy_ratio"]
        / max(features["fft_mid_energy_ratio"], 1e-6)
    )
    return features


def extract_frequency_forensics_features(
    frames_or_video: Sequence[np.ndarray] | str | Path,
    *,
    max_frames: int = 24,
    sample_fps: float = 8.0,
) -> dict[str, Any]:
    """Extract no-reference frequency / compression artifact features."""
    frames = _frames_from_input(
        frames_or_video,
        max_frames=max_frames,
        sample_fps=sample_fps,
    )
    if not frames:
        raise ValueError("No frames available for frequency forensics.")

    dct_rows = []
    fft_rows = []
    for frame in frames:
        gray = _to_gray(frame)
        dct_rows.append(_block_dct_energy(gray))
        fft_rows.append(_radial_fft_features(gray))

    features: dict[str, float] = {}
    for key in dct_rows[0]:
        values = [row[key] for row in dct_rows]
        features[f"freq_{key}_mean"] = _safe_mean(values)
        features[f"freq_{key}_std"] = float(np.std(values)) if values else 0.0
    for key in fft_rows[0]:
        values = [row[key] for row in fft_rows]
        features[f"freq_{key}_mean"] = _safe_mean(values)
        features[f"freq_{key}_std"] = float(np.std(values)) if values else 0.0

    # Temporal high-frequency flicker across frames.
    high_series = [row["dct_high_ratio"] for row in dct_rows]
    if len(high_series) >= 2:
        features["freq_hf_temporal_flicker"] = float(np.std(np.diff(high_series)))
    else:
        features["freq_hf_temporal_flicker"] = 0.0

    # Naturalness prior: mid DCT energy, modest grid periodicity, controlled HF flicker.
    mid = _clamp(features["freq_dct_mid_energy_mean"] / 0.35)
    grid_penalty = _clamp(features["freq_dct_grid_periodicity_mean"])
    hf = _clamp(features["freq_dct_high_ratio_mean"] / 0.25)
    flicker_penalty = _clamp(features["freq_hf_temporal_flicker"] / 0.05)
    spectrum = _clamp(
        1.0
        - abs(features.get("freq_fft_high_mid_ratio_mean", 1.0) - 0.55) / 0.55
    )
    score = _clamp(
        0.28 * mid
        + 0.22 * hf
        + 0.20 * spectrum
        + 0.18 * (1.0 - grid_penalty)
        + 0.12 * (1.0 - flicker_penalty)
    )
    features["freq_forensics_score_0_1"] = score
    return {
        "schema_version": FREQ_FORENSICS_SCHEMA,
        "status": "available",
        "score_0_1": score,
        "features": features,
        "manual_reference_required": False,
        "vmaf_used": False,
        "note": (
            "No-reference DCT/FFT compression and spectrum forensics. "
            "Not a human MOS substitute and not VMAF."
        ),
    }


def merge_frequency_into_texture_features(
    texture_result: dict[str, Any],
    freq_result: dict[str, Any],
) -> dict[str, Any]:
    features = dict(texture_result.get("features", {}))
    freq_features = dict(freq_result.get("features", {}))
    features.update(freq_features)
    prior = float(features.get("training_free_texture_prior_0_1", 0.5))
    freq_score = float(freq_features.get("freq_forensics_score_0_1", 0.5))
    if not math.isfinite(prior):
        prior = 0.5
    if not math.isfinite(freq_score):
        freq_score = 0.5
    features["training_free_texture_prior_0_1"] = _clamp(0.80 * prior + 0.20 * freq_score)
    enriched = dict(texture_result)
    enriched["features"] = features
    enriched["frequency_forensics"] = {
        "schema_version": freq_result.get("schema_version"),
        "status": freq_result.get("status"),
        "score_0_1": freq_result.get("score_0_1"),
        "note": freq_result.get("note"),
    }
    return enriched
