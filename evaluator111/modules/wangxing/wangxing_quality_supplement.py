"""Additive Wang Xing quality supplement.

Produces yellow-box style facial-expression / muscle and texture-detail
quality scores without changing ordinary five-category scoring or the
existing Wang Xing identity / expression decision logic.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.face_detection import FaceDetector
from ..forensics.facial_motion import (
    FACE_ANCHORS,
    LANDMARK_GROUPS,
    extract_facial_motion_features,
    _column_for_au,
    _finite,
    _frame_landmarks,
    _landmark_columns,
    _normalized_landmark_frame,
    _timestamp_axis,
)
from ..core.video_metrics import sample_video_frames
from .wangxing_specialization import score_expression_profile

SUPPLEMENT_SCHEMA = "wangxing_quality_supplement_v1"
DEFAULT_AU_IDS = (1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 23, 24, 25, 26)
# AUs that typically drive local skin deformation / wrinkle cues.
WRINKLE_AU_IDS = (1, 2, 4, 6, 7, 9, 10, 12, 14, 15, 17, 23, 24)
PERIOCULAR_AU_IDS = (1, 2, 4, 6, 7)
MOUTH_AU_IDS = (10, 12, 14, 15, 17, 23, 24)
WRINKLE_REGION_BOXES = {
    "periocular": (0.10, 0.18, 0.90, 0.48),
    "glabella_forehead": (0.28, 0.06, 0.72, 0.30),
    "mouth": (0.26, 0.52, 0.74, 0.88),
    "cheek_left": (0.04, 0.36, 0.34, 0.74),
    "cheek_right": (0.66, 0.36, 0.96, 0.74),
}
WRINKLE_REGION_WEIGHTS = {
    "periocular": 0.40,
    "glabella_forehead": 0.22,
    "mouth": 0.26,
    "cheek_left": 0.06,
    "cheek_right": 0.06,
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _safe_mean(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else 0.0


def _safe_std(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.std(clean)) if clean else 0.0


def _score_100(score_0_1: float | None) -> float | None:
    if score_0_1 is None:
        return None
    return round(100.0 * _clamp(float(score_0_1)), 2)


def _moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    if len(series) < 3 or window <= 1:
        return series
    window = min(window, max(3, len(series) // 3 * 2 + 1))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(series, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _load_au_rows(au_csv: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(au_csv)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames or not rows:
        raise ValueError(f"AU CSV is empty: {path}")
    return fieldnames, rows


def _au_intensity_series(
    rows: Sequence[dict[str, str]],
    fieldnames: Sequence[str],
    au_ids: Sequence[int] = DEFAULT_AU_IDS,
) -> np.ndarray:
    columns = [
        column
        for au_id in au_ids
        if (column := _column_for_au(fieldnames, au_id)) is not None
    ]
    if not columns:
        return np.zeros((len(rows),), dtype=np.float32)
    matrix = np.asarray(
        [
            [_finite(row.get(column), 0.0) for column in columns]
            for row in rows
        ],
        dtype=np.float32,
    )
    return matrix.mean(axis=1)


def _resample_series(
    values: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(source_times)
    if int(np.count_nonzero(mask)) < 2 or len(target_times) == 0:
        return np.full(len(target_times), np.nan, dtype=np.float64)
    order = np.argsort(source_times[mask])
    timed = source_times[mask][order]
    series = values[mask][order]
    # Drop duplicate timestamps that break interpolation.
    unique_times, unique_index = np.unique(timed, return_index=True)
    series = series[unique_index]
    if len(unique_times) < 2:
        return np.full(len(target_times), float(series[0]), dtype=np.float64)
    return np.interp(
        target_times,
        unique_times,
        series,
        left=float(series[0]),
        right=float(series[-1]),
    )


def _gaze_series(
    rows: Sequence[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, float]:
    yaw = np.asarray(
        [_finite(row.get("gaze_yaw"), math.nan) for row in rows],
        dtype=np.float64,
    )
    pitch = np.asarray(
        [_finite(row.get("gaze_pitch"), math.nan) for row in rows],
        dtype=np.float64,
    )
    valid = np.isfinite(yaw) & np.isfinite(pitch)
    coverage = float(np.mean(valid)) if len(valid) else 0.0
    return yaw, pitch, coverage


def _pose_series(
    rows: Sequence[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    yaw = np.asarray(
        [_finite(row.get("yaw"), math.nan) for row in rows],
        dtype=np.float64,
    )
    pitch = np.asarray(
        [_finite(row.get("pitch"), math.nan) for row in rows],
        dtype=np.float64,
    )
    return yaw, pitch


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(np.count_nonzero(mask)) < 8:
        return 0.0
    a = left[mask]
    b = right[mask]
    if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _smoothness_score(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if int(np.count_nonzero(mask)) < 4:
        return 0.0
    series = values[mask]
    deltas = np.diff(series)
    # Gaze/pose in degrees: small frame-to-frame jumps score higher.
    mean_abs = float(np.mean(np.abs(deltas)))
    return _clamp(1.0 - mean_abs / 8.0)


def _lagged_abs_corr(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_lag: int = 8,
) -> tuple[float, int]:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(np.count_nonzero(mask)) < 12:
        return 0.0, 0
    a = np.asarray(left[mask], dtype=np.float64)
    b = np.asarray(right[mask], dtype=np.float64)
    best = abs(_corr(a, b))
    best_lag = 0
    limit = min(max_lag, max(1, len(a) // 8))
    for lag in range(1, limit + 1):
        forward = abs(_corr(a[lag:], b[:-lag]))
        backward = abs(_corr(a[:-lag], b[lag:]))
        if forward > best:
            best = forward
            best_lag = lag
        if backward > best:
            best = backward
            best_lag = -lag
    return float(best), int(best_lag)


def _iris_gaze_signals(
    rows: Sequence[dict[str, str]],
    fieldnames: Sequence[str],
) -> dict[str, Any]:
    """Build relative and absolute iris gaze signals for coupling checks."""
    landmark_columns = _landmark_columns(fieldnames)
    left_iris, right_iris = 468, 473
    left_eye = (33, 133)
    right_eye = (362, 263)
    needed = {left_iris, right_iris, *left_eye, *right_eye, *FACE_ANCHORS}
    empty = np.full(len(rows), np.nan, dtype=np.float64)
    if not needed.issubset(landmark_columns):
        return {
            "relative_yaw": empty,
            "relative_pitch": empty,
            "absolute_yaw": empty,
            "absolute_pitch": empty,
            "binocular_agreement_0_1": 0.0,
            "coverage": 0.0,
            "source": "iris_landmarks_missing",
        }

    rel_yaw: list[float] = []
    rel_pitch: list[float] = []
    abs_yaw: list[float] = []
    abs_pitch: list[float] = []
    left_yaw: list[float] = []
    right_yaw: list[float] = []
    left_abs_x: list[float] = []
    right_abs_x: list[float] = []
    valid = 0
    for row in rows:
        points = _frame_landmarks(row, landmark_columns)
        if not needed.issubset(points):
            rel_yaw.append(math.nan)
            rel_pitch.append(math.nan)
            abs_yaw.append(math.nan)
            abs_pitch.append(math.nan)
            left_yaw.append(math.nan)
            right_yaw.append(math.nan)
            left_abs_x.append(math.nan)
            right_abs_x.append(math.nan)
            continue
        left_width = float(
            np.linalg.norm(points[left_eye[1]] - points[left_eye[0]])
        )
        right_width = float(
            np.linalg.norm(points[right_eye[1]] - points[right_eye[0]])
        )
        face_width = float(
            np.linalg.norm(points[FACE_ANCHORS[1]] - points[FACE_ANCHORS[0]])
        )
        face_height = float(
            np.linalg.norm(points[FACE_ANCHORS[3]] - points[FACE_ANCHORS[2]])
        )
        if min(left_width, right_width, face_width, face_height) < 1e-6:
            rel_yaw.append(math.nan)
            rel_pitch.append(math.nan)
            abs_yaw.append(math.nan)
            abs_pitch.append(math.nan)
            left_yaw.append(math.nan)
            right_yaw.append(math.nan)
            left_abs_x.append(math.nan)
            right_abs_x.append(math.nan)
            continue
        left_center = 0.5 * (points[left_eye[0]] + points[left_eye[1]])
        right_center = 0.5 * (points[right_eye[0]] + points[right_eye[1]])
        face_center = 0.5 * (
            points[FACE_ANCHORS[0]] + points[FACE_ANCHORS[1]]
        )
        left_rel = (points[left_iris] - left_center) / left_width
        right_rel = (points[right_iris] - right_center) / right_width
        iris_mid = 0.5 * (points[left_iris] + points[right_iris])
        abs_offset = (iris_mid - face_center) / np.asarray(
            [face_width, face_height],
            dtype=np.float32,
        )
        rel = 0.5 * (left_rel + right_rel)
        rel_yaw.append(float(rel[0]) * 30.0)
        rel_pitch.append(float(rel[1]) * 30.0)
        abs_yaw.append(float(abs_offset[0]) * 45.0)
        abs_pitch.append(float(abs_offset[1]) * 45.0)
        left_yaw.append(float(left_rel[0]) * 30.0)
        right_yaw.append(float(right_rel[0]) * 30.0)
        left_abs_x.append(
            float((points[left_iris][0] - face_center[0]) / face_width)
        )
        right_abs_x.append(
            float((points[right_iris][0] - face_center[0]) / face_width)
        )
        valid += 1

    left_arr = np.asarray(left_yaw, dtype=np.float64)
    right_arr = np.asarray(right_yaw, dtype=np.float64)
    left_abs = np.asarray(left_abs_x, dtype=np.float64)
    right_abs = np.asarray(right_abs_x, dtype=np.float64)
    binocular = max(
        abs(_corr(_moving_average(left_arr), _moving_average(right_arr))),
        abs(_corr(_moving_average(left_abs), _moving_average(right_abs))),
    )
    return {
        "relative_yaw": np.asarray(rel_yaw, dtype=np.float64),
        "relative_pitch": np.asarray(rel_pitch, dtype=np.float64),
        "absolute_yaw": np.asarray(abs_yaw, dtype=np.float64),
        "absolute_pitch": np.asarray(abs_pitch, dtype=np.float64),
        "binocular_agreement_0_1": binocular,
        "coverage": valid / max(len(rows), 1),
        "source": "iris_landmark_proxy",
    }


def _score_eye_gaze(
    rows: Sequence[dict[str, str]],
    fieldnames: Sequence[str],
) -> dict[str, Any]:
    head_yaw, head_pitch = _pose_series(rows)
    gaze_yaw, gaze_pitch, coverage = _gaze_series(rows)
    source = "gaze_yaw_pitch"
    binocular = 1.0
    abs_couple = 0.0
    abs_lag = 0
    if coverage >= 0.2:
        # Explicit gaze angles: allow small temporal lag versus head pose.
        yaw_couple, yaw_lag = _lagged_abs_corr(gaze_yaw, head_yaw, max_lag=10)
        pitch_couple, pitch_lag = _lagged_abs_corr(
            gaze_pitch, head_pitch, max_lag=10
        )
        couple = 0.5 * (yaw_couple + pitch_couple)
        abs_lag = yaw_lag if yaw_couple >= pitch_couple else pitch_lag
        smooth = 0.5 * (
            _smoothness_score(gaze_yaw) + _smoothness_score(gaze_pitch)
        )
    else:
        iris = _iris_gaze_signals(rows, fieldnames)
        coverage = float(iris["coverage"])
        source = str(iris["source"])
        if coverage < 0.2:
            return {
                "score_0_1": None,
                "status": "unavailable",
                "reason": "gaze_signal_sparse",
                "coverage": coverage,
                "source": source,
            }
        # Relative iris-in-socket often fights head motion (VOR). Prefer
        # absolute iris-in-face coupling plus binocular agreement.
        abs_couple, abs_lag = _lagged_abs_corr(
            iris["absolute_yaw"],
            head_yaw,
            max_lag=10,
        )
        abs_pitch_couple, _ = _lagged_abs_corr(
            iris["absolute_pitch"],
            head_pitch,
            max_lag=10,
        )
        abs_couple = 0.5 * (abs_couple + abs_pitch_couple)
        rel_yaw = iris["relative_yaw"]
        rel_pitch = iris["relative_pitch"]
        # Residual relative gaze should still be smooth, not thrashing.
        smooth = 0.5 * (
            _smoothness_score(rel_yaw) + _smoothness_score(rel_pitch)
        )
        binocular = float(iris["binocular_agreement_0_1"])
        couple = _clamp(0.65 * abs_couple + 0.35 * binocular)

    score = _clamp(
        0.20 * coverage
        + 0.25 * smooth
        + 0.40 * couple
        + 0.15 * binocular
    )
    return {
        "score_0_1": score,
        "status": "ready",
        "coverage": coverage,
        "head_coupling_0_1": couple,
        "absolute_head_coupling_0_1": abs_couple if source != "gaze_yaw_pitch" else couple,
        "smoothness_0_1": smooth,
        "binocular_agreement_0_1": binocular,
        "best_lag_frames": abs_lag,
        "source": source,
    }


def _score_muscle_geometry(
    rows: Sequence[dict[str, str]],
    fieldnames: Sequence[str],
    motion_features: dict[str, Any],
) -> dict[str, Any]:
    landmark_columns = _landmark_columns(fieldnames)
    apertures: list[float] = []
    valid = 0
    for row in rows:
        points = _normalized_landmark_frame(
            _frame_landmarks(row, landmark_columns)
        )
        if not points:
            continue
        valid += 1
        mouth = [
            points[index]
            for index in LANDMARK_GROUPS["mouth"]
            if index in points
        ]
        eye_left = [
            points[index]
            for index in LANDMARK_GROUPS["eye_left"]
            if index in points
        ]
        eye_right = [
            points[index]
            for index in LANDMARK_GROUPS["eye_right"]
            if index in points
        ]
        if len(mouth) >= 2:
            apertures.append(float(np.linalg.norm(mouth[0] - mouth[1])))
        if len(eye_left) >= 2:
            apertures.append(float(np.linalg.norm(eye_left[0] - eye_left[1])))
        if len(eye_right) >= 2:
            apertures.append(
                float(np.linalg.norm(eye_right[0] - eye_right[1]))
            )
    coverage = valid / max(len(rows), 1)
    features = motion_features.get("features") or {}
    coherence = _finite(features.get("motion_coherence_0_1"), 0.0)
    landmark_ratio = _finite(
        features.get("landmark_valid_frame_ratio"), coverage
    )
    aperture_std = _safe_std(apertures)
    dynamic = _clamp(aperture_std / 0.08)
    score = _clamp(
        0.40 * coherence + 0.35 * landmark_ratio + 0.25 * dynamic
    )
    return {
        "score_0_1": score if coverage >= 0.2 else None,
        "status": "ready" if coverage >= 0.2 else "unavailable",
        "landmark_frame_ratio": coverage,
        "motion_coherence_0_1": coherence,
        "aperture_std": aperture_std,
    }


def _region_box_from_face(
    face_box: tuple[int, int, int, int],
    region: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = face_box
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    rx1, ry1, rx2, ry2 = region
    return (
        int(x1 + rx1 * width),
        int(y1 + ry1 * height),
        int(x1 + rx2 * width),
        int(y1 + ry2 * height),
    )


def _crop_feature_dict(crop: np.ndarray) -> dict[str, float]:
    if crop.size == 0:
        return {
            "high_frequency_ratio": 0.0,
            "laplacian_variance": 0.0,
            "edge_density": 0.0,
        }
    crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    high_pass = gray - blur
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    edges = cv2.Canny(gray.astype(np.uint8), 50, 120)
    return {
        "high_frequency_ratio": float(
            np.mean(np.abs(high_pass)) / (float(np.mean(np.abs(gray))) + 1e-6)
        ),
        "laplacian_variance": float(np.var(laplacian)),
        "edge_density": float(np.mean(edges > 0)),
    }


def _face_region_features(
    frame: np.ndarray,
    box: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    height, width = frame.shape[:2]
    if box is None:
        face_box = (0, 0, width, height)
    else:
        x1, y1, x2, y2 = box
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        face_box = (
            max(0, x1 - int(0.04 * bw)),
            max(0, y1 - int(0.04 * bh)),
            min(width, x2 + int(0.04 * bw)),
            min(height, y2 + int(0.04 * bh)),
        )
    region_scores: dict[str, float] = {}
    region_hf: dict[str, float] = {}
    region_edge: dict[str, float] = {}
    for name, region in WRINKLE_REGION_BOXES.items():
        rx1, ry1, rx2, ry2 = _region_box_from_face(face_box, region)
        rx1 = max(0, min(rx1, width - 1))
        ry1 = max(0, min(ry1, height - 1))
        rx2 = max(rx1 + 1, min(rx2, width))
        ry2 = max(ry1 + 1, min(ry2, height))
        feats = _crop_feature_dict(frame[ry1:ry2, rx1:rx2])
        region_hf[name] = feats["high_frequency_ratio"]
        region_edge[name] = feats["edge_density"]
        region_scores[name] = _map_hf_to_score(
            feats["high_frequency_ratio"],
            feats["laplacian_variance"],
            feats["edge_density"],
        )
    weighted = sum(
        WRINKLE_REGION_WEIGHTS[name] * region_scores[name]
        for name in WRINKLE_REGION_WEIGHTS
    )
    full = _crop_feature_dict(
        frame[face_box[1] : face_box[3], face_box[0] : face_box[2]]
    )
    return {
        "wrinkle_score_0_1": float(weighted),
        "full_face": full,
        "region_scores": region_scores,
        "region_high_frequency": region_hf,
        "region_edge_density": region_edge,
        "periocular_hf": region_hf["periocular"],
        "mouth_hf": region_hf["mouth"],
        "weighted_hf": float(
            sum(
                WRINKLE_REGION_WEIGHTS[name] * region_hf[name]
                for name in WRINKLE_REGION_WEIGHTS
            )
        ),
    }


def _map_hf_to_score(hf_mean: float, lap_mean: float, edge_mean: float) -> float:
    # Local skin patches are softer than full-face crops; use gentler floors.
    hf_score = _clamp((hf_mean - 0.008) / 0.055)
    lap_score = _clamp((math.log1p(lap_mean) - 2.0) / 4.0)
    edge_score = _clamp((edge_mean - 0.02) / 0.16)
    return _clamp(0.40 * hf_score + 0.35 * lap_score + 0.25 * edge_score)


def _sample_face_texture(
    video_path: str | Path,
    *,
    max_frames: int = 24,
) -> dict[str, Any]:
    _info, _indices, timestamps, frames = sample_video_frames(
        video_path,
        max_frames=max_frames,
        sample_fps=4.0,
    )
    if not frames:
        return {"status": "unavailable", "reason": "no_frames"}
    detector = FaceDetector()
    records: list[dict[str, Any]] = []
    for frame in frames:
        detection = detector.detect(frame)
        box = (
            tuple(int(value) for value in detection)
            if detection is not None
            else None
        )
        records.append(_face_region_features(frame, box))

    wrinkle_scores = [float(item["wrinkle_score_0_1"]) for item in records]
    weighted_hf = [float(item["weighted_hf"]) for item in records]
    periocular_hf = [float(item["periocular_hf"]) for item in records]
    mouth_hf = [float(item["mouth_hf"]) for item in records]
    lap = [
        float(item["full_face"]["laplacian_variance"]) for item in records
    ]
    edge = [float(item["full_face"]["edge_density"]) for item in records]
    score = _safe_mean(wrinkle_scores)
    stability = _clamp(
        1.0 - _safe_std(weighted_hf) / max(_safe_mean(weighted_hf), 1e-6)
    )
    region_means = {
        name: _safe_mean(
            [float(item["region_scores"][name]) for item in records]
        )
        for name in WRINKLE_REGION_BOXES
    }
    return {
        "status": "ready",
        "frame_count": len(records),
        "sample_timestamps_seconds": [
            float(value) for value in np.asarray(timestamps, dtype=np.float64)
        ],
        "high_frequency_mean": _safe_mean(weighted_hf),
        "laplacian_variance_mean": _safe_mean(lap),
        "edge_density_mean": _safe_mean(edge),
        "score_0_1": score,
        "temporal_stability_0_1": stability,
        "region_score_means": region_means,
        "per_frame_high_frequency": weighted_hf,
        "per_frame_periocular_hf": periocular_hf,
        "per_frame_mouth_hf": mouth_hf,
        "per_frame_wrinkle_score": wrinkle_scores,
        "backend": "local_wrinkle_regions",
    }


def _muscle_wrinkle_sync(
    *,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, str]],
    au_timestamps: np.ndarray | None,
    texture: dict[str, Any],
) -> dict[str, Any]:
    if texture.get("status") != "ready":
        return {
            "score_0_1": None,
            "status": "degraded",
            "reason": "texture_series_unavailable",
        }
    sample_times = np.asarray(
        texture.get("sample_timestamps_seconds") or [],
        dtype=np.float64,
    )
    weighted_hf = np.asarray(
        texture.get("per_frame_high_frequency") or [],
        dtype=np.float64,
    )
    periocular_hf = np.asarray(
        texture.get("per_frame_periocular_hf") or [],
        dtype=np.float64,
    )
    mouth_hf = np.asarray(
        texture.get("per_frame_mouth_hf") or [],
        dtype=np.float64,
    )
    if len(sample_times) < 6 or len(weighted_hf) < 6:
        return {
            "score_0_1": None,
            "status": "unavailable",
            "reason": "too_few_aligned_samples",
        }

    if au_timestamps is None or len(au_timestamps) != len(rows):
        au_timestamps = np.linspace(0.0, 1.0, len(rows), dtype=np.float64)
        align_mode = "index_normalized"
    else:
        # Stretch AU timeline onto the sampled video span when units differ.
        au_timestamps = np.asarray(au_timestamps, dtype=np.float64)
        if float(np.nanmax(au_timestamps) - np.nanmin(au_timestamps)) > 1e-6:
            au_span = float(np.nanmax(au_timestamps) - np.nanmin(au_timestamps))
            vid_span = float(np.nanmax(sample_times) - np.nanmin(sample_times))
            if vid_span > 1e-6 and abs(au_span - vid_span) / max(vid_span, 1e-6) > 0.35:
                # Rebase both to [0, duration] using their own spans.
                au_timestamps = (
                    (au_timestamps - np.nanmin(au_timestamps))
                    / au_span
                    * vid_span
                    + float(np.nanmin(sample_times))
                )
            align_mode = "timestamp_seconds"
        else:
            align_mode = "timestamp_degenerate"

    wrinkle_au = _au_intensity_series(rows, fieldnames, WRINKLE_AU_IDS)
    peri_au = _au_intensity_series(rows, fieldnames, PERIOCULAR_AU_IDS)
    mouth_au = _au_intensity_series(rows, fieldnames, MOUTH_AU_IDS)

    wrinkle_on_video = _resample_series(wrinkle_au, au_timestamps, sample_times)
    peri_on_video = _resample_series(peri_au, au_timestamps, sample_times)
    mouth_on_video = _resample_series(mouth_au, au_timestamps, sample_times)

    # Compare both level and first-difference (activation bursts).
    corr_level, lag_level = _lagged_abs_corr(
        _moving_average(wrinkle_on_video),
        _moving_average(weighted_hf),
        max_lag=4,
    )
    corr_delta, lag_delta = _lagged_abs_corr(
        np.diff(_moving_average(wrinkle_on_video)),
        np.diff(_moving_average(weighted_hf)),
        max_lag=4,
    )
    peri_corr, _ = _lagged_abs_corr(
        _moving_average(peri_on_video),
        _moving_average(periocular_hf),
        max_lag=4,
    )
    mouth_corr, _ = _lagged_abs_corr(
        _moving_average(mouth_on_video),
        _moving_average(mouth_hf),
        max_lag=4,
    )
    regional = 0.5 * (peri_corr + mouth_corr)
    sync_strength = _clamp(
        0.20 * corr_level + 0.35 * corr_delta + 0.45 * regional
    )
    stability = float(texture.get("temporal_stability_0_1") or 0.0)
    score = _clamp(0.85 * sync_strength + 0.15 * stability)
    return {
        "score_0_1": score,
        "status": "ready",
        "align_mode": align_mode,
        "level_abs_corr": corr_level,
        "delta_abs_corr": corr_delta,
        "periocular_abs_corr": peri_corr,
        "mouth_abs_corr": mouth_corr,
        "best_lag_frames_level": lag_level,
        "best_lag_frames_delta": lag_delta,
        "aligned_samples": int(len(sample_times)),
    }


def _findings_facial(
    metrics: dict[str, float | None],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    gaze = metrics.get("eye_gaze_match_0_1")
    if gaze is not None and gaze >= 0.7:
        findings.append(
            {
                "type": "positive",
                "text": "眼神/视线变化相对平滑，与头部姿态耦合较自然。",
            }
        )
    wrinkle = metrics.get("wrinkle_high_frequency_0_1")
    if wrinkle is not None and wrinkle < 0.55:
        findings.append(
            {
                "type": "issue",
                "text": "肌肉形变对应的皱纹和局部皮肤高频细节不足或不稳定。",
            }
        )
    sync = metrics.get("muscle_wrinkle_sync_0_1")
    if sync is not None and sync < 0.5:
        findings.append(
            {
                "type": "issue",
                "text": "肌肉动作与局部高频纹理变化同步偏弱。",
            }
        )
    proto = metrics.get("motion_prototype_match_0_1")
    if proto is not None and proto >= 0.6:
        findings.append(
            {
                "type": "positive",
                "text": "面部动作与王兴表情画像原型较接近。",
            }
        )
    if not findings:
        findings.append(
            {
                "type": "note",
                "text": "表情与肌肉细项已计算，未发现极端异常。",
            }
        )
    return findings


def _findings_texture(metrics: dict[str, float | None]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    score = metrics.get("score_0_1")
    edge = metrics.get("edge_clarity_0_1")
    hf = metrics.get("local_texture_0_1")
    if hf is not None and hf >= 0.55:
        findings.append(
            {
                "type": "positive",
                "text": "局部纹理信息较充足，没有明显的整体糊化。",
            }
        )
    if edge is not None and edge < 0.55:
        findings.append(
            {
                "type": "issue",
                "text": "边缘清晰度偏弱，近景中可能出现糊边或细节软化。",
            }
        )
        findings.append(
            {
                "type": "suggestion",
                "text": "优先恢复人脸五官、发丝边缘和服装褶皱的局部清晰度，避免对整幅画面统一锐化。",
            }
        )
    if score is not None and score >= 0.75:
        findings.append(
            {
                "type": "note",
                "text": f"本项得分为 {_score_100(score):.2f} 分，整体表现较好。",
            }
        )
    elif score is not None:
        findings.append(
            {
                "type": "note",
                "text": f"本项得分为 {_score_100(score):.2f} 分，整体表现一般。",
            }
        )
    return findings


def evaluate_quality_supplement(
    *,
    au_csv: str | Path,
    video_path: str | Path | None = None,
    expression_profile: dict[str, Any] | None = None,
    expression_profile_path: str | Path | None = None,
    expected_class: str | None = None,
    max_texture_frames: int = 24,
) -> dict[str, Any]:
    """Compute additive quality metrics for one video / AU CSV pair."""
    fieldnames, rows = _load_au_rows(au_csv)
    timestamps, timebase = _timestamp_axis(rows)
    motion = extract_facial_motion_features(
        au_csv,
        time_aware_derivatives=True,
    )

    if expression_profile is None and expression_profile_path is not None:
        expression_profile = _load_json(expression_profile_path)
    prototype = {
        "score_0_1": None,
        "status": "skipped",
        "reason": "expression_profile_not_provided",
    }
    if expression_profile is not None:
        expression = score_expression_profile(
            au_csv,
            expression_profile,
            expected_class=expected_class,
        )
        prototype = {
            "score_0_1": expression.get("compatibility_0_1"),
            "status": expression.get("status", "ready"),
            "selected_profile": expression.get("selected_profile"),
            "selected_profile_display_name": expression.get(
                "selected_profile_display_name"
            ),
            "top_profiles": expression.get("top_profiles", [])[:2],
            "decision": expression.get("decision"),
        }

    muscle = _score_muscle_geometry(rows, fieldnames, motion)
    gaze = _score_eye_gaze(rows, fieldnames)
    texture = (
        _sample_face_texture(video_path, max_frames=max_texture_frames)
        if video_path is not None and Path(video_path).is_file()
        else {"status": "skipped", "reason": "video_not_provided"}
    )
    wrinkle_score = (
        float(texture["score_0_1"])
        if texture.get("status") == "ready"
        else None
    )
    sync = _muscle_wrinkle_sync(
        fieldnames=fieldnames,
        rows=rows,
        au_timestamps=timestamps,
        texture=texture,
    )

    facial_metrics = {
        "motion_prototype_match_0_1": prototype.get("score_0_1"),
        "muscle_geometry_0_1": muscle.get("score_0_1"),
        "eye_gaze_match_0_1": gaze.get("score_0_1"),
        "wrinkle_high_frequency_0_1": wrinkle_score,
        "muscle_wrinkle_sync_0_1": sync.get("score_0_1"),
    }
    facial_values = [
        float(value)
        for value in facial_metrics.values()
        if value is not None and math.isfinite(float(value))
    ]
    facial_score = _safe_mean(facial_values) if facial_values else None

    local_texture = wrinkle_score
    edge_clarity = None
    if texture.get("status") == "ready":
        edge_clarity = _clamp(
            0.55
            * _clamp(
                (math.log1p(float(texture["laplacian_variance_mean"])) - 3.0)
                / 4.0
            )
            + 0.45
            * _clamp((float(texture["edge_density_mean"]) - 0.04) / 0.18)
        )
    texture_stability = texture.get("temporal_stability_0_1")
    texture_components = [
        value
        for value in (local_texture, edge_clarity, texture_stability)
        if value is not None
    ]
    texture_score = _safe_mean(texture_components) if texture_components else None
    texture_metrics = {
        "score_0_1": texture_score,
        "local_texture_0_1": local_texture,
        "edge_clarity_0_1": edge_clarity,
        "temporal_stability_0_1": texture_stability,
    }

    return {
        "schema_version": SUPPLEMENT_SCHEMA,
        "status": "ready" if facial_score is not None else "partial",
        "scope": "additive_quality_supplement",
        "does_not_modify": [
            "ordinary_five_category_scores",
            "wangxing_identity_decision",
            "wangxing_expression_decision",
            "web_ui",
        ],
        "inputs": {
            "au_csv": str(Path(au_csv).resolve()),
            "video_path": (
                str(Path(video_path).resolve())
                if video_path is not None
                else None
            ),
            "expression_profile": (
                str(Path(expression_profile_path).resolve())
                if expression_profile_path is not None
                else None
            ),
            "timebase": timebase,
            "frame_count": len(rows),
        },
        "facial_expression_muscle": {
            "title": "人脸表情与肌肉运动",
            "subtitle": "人脸专项：表情、眼神与肌肉皱纹",
            "backend": "AU + MediaPipe Face Mesh landmarks + gaze columns",
            "score_0_1": facial_score,
            "score_100": _score_100(facial_score),
            "metrics": facial_metrics,
            "metric_labels": {
                "motion_prototype_match_0_1": "动作原型匹配",
                "muscle_geometry_0_1": "肌肉几何",
                "eye_gaze_match_0_1": "眼神匹配",
                "wrinkle_high_frequency_0_1": "皱纹/高频纹理",
                "muscle_wrinkle_sync_0_1": "肌肉-皱纹同步",
            },
            "details": {
                "motion_prototype": prototype,
                "muscle_geometry": muscle,
                "eye_gaze": gaze,
                "wrinkle_regions": (
                    {
                        "region_score_means": texture.get("region_score_means"),
                        "backend": texture.get("backend"),
                        "score_0_1": wrinkle_score,
                    }
                    if texture.get("status") == "ready"
                    else {"status": texture.get("status")}
                ),
                "muscle_wrinkle_sync": sync,
            },
            "findings": _findings_facial(facial_metrics),
        },
        "texture_detail_quality": {
            "title": "质感和细节",
            "subtitle": "质量估计（专项旁路，不改普通五项）",
            "backend": "face-crop high-frequency / Laplacian / edge density",
            "score_0_1": texture_score,
            "score_100": _score_100(texture_score),
            "metrics": texture_metrics,
            "details": {
                key: value
                for key, value in texture.items()
                if not str(key).startswith("per_frame_")
            },
            "findings": _findings_texture(texture_metrics),
        },
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload
