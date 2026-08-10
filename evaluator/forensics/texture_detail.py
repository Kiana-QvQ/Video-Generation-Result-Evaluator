from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..face_detection import FaceDetector
from ..video_metrics import sample_video_frames

TEXTURE_DETAIL_SCHEMA = "texture_detail_forensics_v1"
DEFAULT_CROP_SIZE = (192, 192)
REGION_BOXES = {
    "skin_forehead": (0.22, 0.04, 0.78, 0.28),
    "skin_left_cheek": (0.04, 0.34, 0.36, 0.76),
    "skin_right_cheek": (0.64, 0.34, 0.96, 0.76),
    "eyes": (0.14, 0.22, 0.86, 0.50),
    "mouth": (0.24, 0.54, 0.76, 0.88),
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _safe_mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.std(values)) if values else 0.0


def _crop_frame(
    frame: np.ndarray,
    box: tuple[int, int, int, int] | None,
    *,
    margin: float = 0.05,
    crop_size: tuple[int, int] = DEFAULT_CROP_SIZE,
) -> np.ndarray:
    height, width = frame.shape[:2]
    if box is None:
        x1, y1, x2, y2 = 0, 0, width, height
    else:
        x1, y1, x2, y2 = (int(value) for value in box)
        box_width = max(x2 - x1, 1)
        box_height = max(y2 - y1, 1)
        x_margin = int(box_width * margin)
        y_margin = int(box_height * margin)
        x1 -= x_margin
        y1 -= y_margin
        x2 += x_margin
        y2 += y_margin
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, crop_size, interpolation=cv2.INTER_AREA)


def _entropy(values: np.ndarray) -> float:
    histogram = cv2.calcHist([values], [0], None, [32], [0, 256]).reshape(-1)
    total = float(np.sum(histogram))
    if total <= 1e-8:
        return 0.0
    probabilities = histogram / total
    probabilities = probabilities[probabilities > 1e-8]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _basic_texture_features(crop: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    normalized = gray / 255.0
    blur = cv2.GaussianBlur(normalized, (0, 0), 1.2)
    high_pass = normalized - blur
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)

    small = cv2.resize(normalized, (64, 64), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small)
    frequency_mask = np.ones_like(dct, dtype=bool)
    frequency_mask[:8, :8] = False
    total_energy = float(np.mean(np.abs(dct))) + 1e-6
    high_frequency_dct = float(np.mean(np.abs(dct[frequency_mask])))
    return {
        "high_frequency_ratio": float(
            np.mean(np.abs(high_pass))
            / (float(np.mean(np.abs(normalized))) + 1e-6)
        ),
        "laplacian_variance": float(np.var(laplacian)),
        "gradient_mean": float(np.mean(gradient) / 255.0),
        "gradient_std": float(np.std(gradient) / 255.0),
        "intensity_entropy": _entropy(gray.astype(np.uint8)),
        "dct_high_frequency_ratio": high_frequency_dct / total_energy,
    }


def _normalized_region(crop: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    height, width = crop.shape[:2]
    left, top, right, bottom = box
    x1 = max(0, min(width - 1, round(left * width)))
    y1 = max(0, min(height - 1, round(top * height)))
    x2 = max(x1 + 1, min(width, round(right * width)))
    y2 = max(y1 + 1, min(height, round(bottom * height)))
    return crop[y1:y2, x1:x2]


def _frame_texture_features(crop: np.ndarray) -> dict[str, float]:
    features = _basic_texture_features(crop)
    for region_name, region_box in REGION_BOXES.items():
        region = _normalized_region(crop, region_box)
        for name, value in _basic_texture_features(region).items():
            features[f"region_{region_name}_{name}"] = value
    return features


def _warp_previous(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    height, width = previous_gray.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x - flow[:, :, 0]
    map_y = grid_y - flow[:, :, 1]
    warped = cv2.remap(
        previous,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped, flow


def _training_free_texture_prior(features: dict[str, float]) -> dict[str, float]:
    """Heuristic realness prior from optical-flow residuals (no profile needed)."""
    residual_mean = float(features.get("temporal_warp_residual_mean", 0.0))
    residual_std = float(features.get("temporal_warp_residual_std", 0.0))
    residual_cv = residual_std / max(residual_mean, 1e-6)
    # AI residuals are often overly homogeneous (very low CV).
    homogeneity = _clamp(residual_cv / 0.35)
    # Extremely large residuals look unstable; mid-low residuals look natural.
    stability = _clamp(1.0 - residual_mean * 4.0)
    second_order = float(features.get("optical_flow_second_order_mean", 0.0))
    second_order_score = _clamp(1.0 - second_order * 10.0)
    flow_mean = float(features.get("optical_flow_magnitude_mean", 0.0))
    # Still faces are valid; only penalize near-zero flow with high residual.
    flow_support = 0.55 if flow_mean < 0.15 else _clamp(0.55 + flow_mean / 8.0)
    flicker = _clamp(float(features.get("texture_flicker_mean", 0.0)) * 10.0)
    clarity = _clamp(1.0 - flicker)
    prior = _clamp(
        0.30 * homogeneity
        + 0.25 * stability
        + 0.20 * second_order_score
        + 0.15 * clarity
        + 0.10 * flow_support
    )
    return {
        "optical_flow_residual_cv": float(residual_cv),
        "optical_flow_homogeneity_0_1": homogeneity,
        "optical_flow_second_order_score_0_1": second_order_score,
        "micro_temporal_naturalness_0_1": prior,
        "training_free_texture_prior_0_1": prior,
    }


def extract_texture_detail_features(
    frames_or_video: Sequence[np.ndarray] | str | Path,
    *,
    face_boxes: Sequence[tuple[int, int, int, int] | None] | None = None,
    max_frames: int = 64,
    sample_fps: float = 8.0,
    detect_faces: bool = True,
) -> dict[str, Any]:
    """Extract local texture, frequency and frame-to-frame residual features."""
    if isinstance(frames_or_video, (str, Path)):
        video_info, _, timestamps, frames = sample_video_frames(
            frames_or_video,
            max_frames,
            sample_fps,
        )
    else:
        frames = list(frames_or_video)[:max_frames]
        video_info = None
        timestamps = np.arange(len(frames), dtype=np.float64)
    if not frames:
        raise ValueError("At least one readable frame is required.")

    if face_boxes is not None and len(face_boxes) != len(frames):
        raise ValueError("face_boxes must have one entry per sampled frame.")
    detection_backend = "provided"
    if face_boxes is None and detect_faces:
        detector = FaceDetector()
        detected_boxes: list[tuple[int, int, int, int] | None] = []
        for frame in frames:
            detected = detector.detect(frame)
            if detected is None:
                detected_boxes.append(None)
                continue
            x, y, width, height = detected
            detected_boxes.append((x, y, x + width, y + height))
        face_boxes = detected_boxes
        detection_backend = "opencv_haar"
    elif face_boxes is None:
        detection_backend = "full_frame"
    crops = [
        _crop_frame(
            frame,
            face_boxes[index] if face_boxes is not None else None,
        )
        for index, frame in enumerate(frames)
    ]
    per_frame = [_frame_texture_features(crop) for crop in crops]
    feature_names = tuple(per_frame[0])
    features: dict[str, float] = {}
    for name in feature_names:
        values = [record[name] for record in per_frame]
        features[f"{name}_mean"] = _safe_mean(values)
        features[f"{name}_std"] = _safe_std(values)
        features[f"{name}_p95"] = float(np.quantile(values, 0.95))

    residuals: list[float] = []
    raw_differences: list[float] = []
    high_frequency_differences: list[float] = []
    flow_magnitudes: list[float] = []
    region_flicker: dict[str, list[float]] = {
        region_name: [] for region_name in REGION_BOXES
    }
    for previous, current in pairwise(crops):
        warped, flow = _warp_previous(previous, current)
        current_float = current.astype(np.float32) / 255.0
        warped_float = warped.astype(np.float32) / 255.0
        residuals.append(float(np.mean(np.abs(current_float - warped_float))))
        flow_magnitudes.append(
            float(np.mean(np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)))
        )
        raw_differences.append(
            float(
                np.mean(
                    np.abs(
                        current.astype(np.float32)
                        - previous.astype(np.float32)
                    )
                )
                / 255.0
            )
        )
        current_features = _frame_texture_features(current)
        previous_features = _frame_texture_features(previous)
        high_frequency_differences.append(
            abs(
                current_features["high_frequency_ratio"]
                - previous_features["high_frequency_ratio"]
            )
        )
        for region_name in REGION_BOXES:
            key = f"region_{region_name}_high_frequency_ratio"
            region_flicker[region_name].append(
                abs(current_features[key] - previous_features[key])
            )
    features["temporal_warp_residual_mean"] = _safe_mean(residuals)
    features["temporal_warp_residual_std"] = _safe_std(residuals)
    features["temporal_raw_difference_mean"] = _safe_mean(raw_differences)
    features["texture_flicker_mean"] = _safe_mean(high_frequency_differences)
    features["texture_flicker_std"] = _safe_std(high_frequency_differences)
    features["optical_flow_magnitude_mean"] = _safe_mean(flow_magnitudes)
    features["optical_flow_magnitude_std"] = _safe_std(flow_magnitudes)
    if len(residuals) >= 2:
        second_order = [
            abs(current - previous)
            for previous, current in pairwise(residuals)
        ]
        features["optical_flow_second_order_mean"] = _safe_mean(second_order)
        features["optical_flow_second_order_std"] = _safe_std(second_order)
    else:
        features["optical_flow_second_order_mean"] = 0.0
        features["optical_flow_second_order_std"] = 0.0
    features.update(_training_free_texture_prior(features))
    for region_name, values in region_flicker.items():
        features[f"region_{region_name}_flicker_mean"] = _safe_mean(values)
        features[f"region_{region_name}_flicker_std"] = _safe_std(values)
    features["frame_count"] = float(len(crops))
    features["face_box_coverage"] = (
        float(sum(box is not None for box in face_boxes) / len(crops))
        if face_boxes is not None
        else 0.0
    )

    window_records: list[dict[str, Any]] = []
    window_size = 8
    for window_index, start in enumerate(range(0, len(per_frame), window_size)):
        stop = min(len(per_frame), start + window_size)
        high_frequency_values = [
            record["high_frequency_ratio"] for record in per_frame[start:stop]
        ]
        flicker_values = high_frequency_differences[start : max(start, stop - 1)]
        window_records.append(
            {
                "window_index": window_index,
                "start_frame": start,
                "end_frame": stop - 1,
                "high_frequency_mean": _safe_mean(high_frequency_values),
                "texture_flicker_mean": _safe_mean(flicker_values),
                "evidence_score_0_1": _clamp(
                    _safe_mean(flicker_values) * 10.0
                ),
            }
        )

    return {
        "schema_version": TEXTURE_DETAIL_SCHEMA,
        "source": str(frames_or_video) if isinstance(frames_or_video, (str, Path)) else "frames",
        "frame_count": len(crops),
        "timestamps": [float(value) for value in timestamps],
        "video_info": video_info,
        "features": features,
        "window_records": window_records,
        "regions": list(REGION_BOXES),
        "face_detection_backend": detection_backend,
        "note": (
            "These are quality and temporal-texture features. A calibrated "
            "real-versus-Seedance profile is required for authenticity claims."
        ),
    }


def _profile_from_records(
    records: Sequence[dict[str, Any]],
    *,
    domain: str,
) -> dict[str, Any]:
    if not records:
        raise ValueError("At least one texture feature record is required.")
    names = sorted(
        {
            name
            for record in records
            for name in record.get("features", {})
            if isinstance(name, str)
        }
    )
    matrix = np.asarray(
        [
            [float(record.get("features", {}).get(name, 0.0)) for name in names]
            for record in records
        ],
        dtype=np.float32,
    )
    return {
        "schema_version": TEXTURE_DETAIL_SCHEMA,
        "domain": domain,
        "sample_count": len(records),
        "feature_names": names,
        "mean": np.mean(matrix, axis=0).astype(float).tolist(),
        "std": np.maximum(np.std(matrix, axis=0), 0.01).astype(float).tolist(),
        "source_records": [record.get("source") for record in records],
    }


def build_texture_detail_profile(
    records: Iterable[dict[str, Any]],
    *,
    domain: str = "real",
) -> dict[str, Any]:
    return _profile_from_records(list(records), domain=domain)


def _fit_score(
    values: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float],
) -> float:
    expected = np.asarray(mean, dtype=np.float32)
    scale = np.maximum(np.asarray(std, dtype=np.float32), 0.01)
    z = (values - expected) / scale
    distance = float(np.mean(np.minimum(np.abs(z), 8.0)))
    return float(math.exp(-distance / 2.0))


def score_texture_detail(
    features_or_video: dict[str, Any] | Sequence[np.ndarray] | str | Path,
    profile: dict[str, Any] | None = None,
    *,
    face_boxes: Sequence[tuple[int, int, int, int] | None] | None = None,
    max_frames: int = 64,
    sample_fps: float = 8.0,
    detect_faces: bool = True,
) -> dict[str, Any]:
    """Score texture features and optionally calibrate them by domain."""
    if isinstance(features_or_video, dict):
        features = features_or_video
    else:
        features = extract_texture_detail_features(
            features_or_video,
            face_boxes=face_boxes,
            max_frames=max_frames,
            sample_fps=sample_fps,
            detect_faces=detect_faces,
        )
    if profile is None:
        feature_map = features["features"]
        return {
            "status": "features_only",
            "probability_calibrated": False,
            "backend": "aligned_texture_frequency_temporal_features",
            "schema_version": TEXTURE_DETAIL_SCHEMA,
            "metrics": {
                "detail_quality_proxy_0_1": _clamp(
                    feature_map.get("high_frequency_ratio_mean", 0.0)
                    * 8.0
                ),
                "temporal_stability_proxy_0_1": _clamp(
                    1.0
                    - feature_map.get(
                        "temporal_warp_residual_mean",
                        1.0,
                    )
                ),
                "texture_flicker_0_1": _clamp(
                    feature_map.get("texture_flicker_mean", 1.0)
                    * 10.0
                ),
                "optical_flow_homogeneity_0_1": _clamp(
                    float(feature_map.get("optical_flow_homogeneity_0_1", 0.0))
                ),
                "micro_temporal_naturalness_0_1": _clamp(
                    float(feature_map.get("micro_temporal_naturalness_0_1", 0.0))
                ),
                "training_free_texture_prior_0_1": _clamp(
                    float(
                        feature_map.get("training_free_texture_prior_0_1", 0.5)
                    )
                ),
                "raw_real_domain_evidence_0_1": None,
                "real_capture_likelihood_0_1": None,
                "calibrated_real_probability_0_1": None,
            },
            "feature_record": features,
            "warnings": [
                (
                    "No real-versus-Seedance profile was supplied; "
                    "authenticity likelihood is unavailable."
                )
            ],
        }

    names = list(profile.get("feature_names", []))
    values = np.asarray(
        [float(features["features"].get(name, 0.0)) for name in names],
        dtype=np.float32,
    )
    real = profile.get("real")
    if real is None and profile.get("domain") == "real":
        real = profile
    seedance = profile.get("seedance")
    real_fit = (
        _fit_score(values, real.get("mean", []), real.get("std", []))
        if real
        else None
    )
    seedance_fit = (
        _fit_score(values, seedance.get("mean", []), seedance.get("std", []))
        if seedance
        else None
    )
    authenticity = None
    if real_fit is not None and seedance_fit is not None:
        authenticity = real_fit / max(real_fit + seedance_fit, 1e-6)
    feature_map = features["features"]
    training_free_prior = _clamp(
        float(feature_map.get("training_free_texture_prior_0_1", 0.5))
    )
    enriched_evidence = authenticity
    if authenticity is not None:
        enriched_evidence = _clamp(
            0.82 * float(authenticity) + 0.18 * training_free_prior
        )
    return {
        "status": "calibrated" if authenticity is not None else "features_only",
        "probability_calibrated": False,
        "backend": "aligned_texture_frequency_temporal_profile",
        "schema_version": TEXTURE_DETAIL_SCHEMA,
        "metrics": {
            "real_domain_fit_0_1": real_fit,
            "seedance_domain_fit_0_1": seedance_fit,
            "profile_raw_real_domain_evidence_0_1": authenticity,
            "raw_real_domain_evidence_0_1": enriched_evidence,
            # Kept for clients using the initial forensic schema. This field
            # is a profile-distance ratio, not a calibrated probability.
            "real_capture_likelihood_0_1": enriched_evidence,
            "calibrated_real_probability_0_1": None,
            "temporal_stability_proxy_0_1": _clamp(
                1.0
                - feature_map.get("temporal_warp_residual_mean", 1.0)
            ),
            "texture_flicker_0_1": _clamp(
                feature_map.get("texture_flicker_mean", 1.0) * 10.0
            ),
            "optical_flow_homogeneity_0_1": _clamp(
                float(feature_map.get("optical_flow_homogeneity_0_1", 0.0))
            ),
            "micro_temporal_naturalness_0_1": _clamp(
                float(feature_map.get("micro_temporal_naturalness_0_1", 0.0))
            ),
            "training_free_texture_prior_0_1": training_free_prior,
        },
        "feature_record": features,
        "warnings": (
            [
                (
                    "The two-domain score is raw profile evidence only; "
                    "a held-out probability calibrator is required for an "
                    "authenticity decision."
                )
            ]
            if authenticity is not None
            else [
                (
                    "The supplied profile has no real and Seedance domains; "
                    "authenticity likelihood is unavailable."
                )
            ]
        ),
    }
