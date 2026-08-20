"""Expression-transition and local face temporal features.

The extractor is deliberately compact and deterministic. It uses AU CSV
landmarks to define eye, mouth, tear-region, brow, and chin crops, then
measures local appearance continuity together with AU/landmark derivatives.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluator.vedio_pred.real_video_detector import _read_sampled_frames

TRANSITION_FEATURE_NAMES: tuple[str, ...] = (
    "transition_event_ratio_0_1",
    "transition_longest_event_ratio_0_1",
    "transition_event_count_norm_0_1",
    "transition_peak_speed_0_1",
    "transition_peak_acceleration_0_1",
    "transition_onset_offset_asymmetry_0_1",
    "au_speed_mean_0_1",
    "au_speed_p95_0_1",
    "au_acceleration_mean_0_1",
    "au_acceleration_p95_0_1",
    "landmark_speed_mean_0_1",
    "landmark_speed_p95_0_1",
    "landmark_acceleration_mean_0_1",
    "landmark_acceleration_p95_0_1",
    "face_geometry_valid_ratio_0_1",
    "face_detection_confidence_0_1",
    "face_detection_missing_mask",
)
for _region in ("eyes", "mouth", "tear", "brow", "chin"):
    for _metric in (
        "motion_mean_0_1",
        "motion_p95_0_1",
        "acceleration_mean_0_1",
        "temporal_residual_0_1",
        "continuity_0_1",
        "missing_mask",
    ):
        TRANSITION_FEATURE_NAMES += (f"{_region}_{_metric}",)

_REGION_LANDMARKS = {
    "eyes": (33, 133, 159, 145, 263, 362, 386, 374),
    "mouth": (61, 291, 13, 14, 78, 308),
    "tear": (133, 145, 362, 374),
    "brow": (70, 107, 300, 336),
    "chin": (152, 148, 377),
}


def _float(value: str | None) -> float:
    try:
        parsed = float(value or "")
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def _sample_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if not rows:
        return []
    if len(rows) <= count:
        return rows
    indices = np.linspace(0, len(rows) - 1, count).round().astype(int)
    return [rows[int(index)] for index in indices]


def _series_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    if len(values) < 2:
        return 0.0, 0.0, 0.0, 0.0
    velocity = np.diff(values, axis=0)
    speed = np.linalg.norm(velocity, axis=-1)
    acceleration = np.diff(velocity, axis=0) if len(velocity) > 1 else velocity
    accel = np.linalg.norm(acceleration, axis=-1)
    return (
        float(np.mean(speed)),
        float(np.quantile(speed, 0.95)),
        float(np.mean(accel)) if len(accel) else 0.0,
        float(np.quantile(accel, 0.95)) if len(accel) else 0.0,
    )


def _normalize(value: float, scale: float = 1.0) -> float:
    return float(np.clip(value / max(scale, 1e-6), 0.0, 1.0))


def _region_crop_stats(
    frame: np.ndarray,
    row: dict[str, str],
    indices: tuple[int, ...],
) -> tuple[np.ndarray, bool]:
    height, width = frame.shape[:2]
    points: list[tuple[float, float]] = []
    for index in indices:
        x = _float(row.get(f"lm_mp_{index}_x"))
        y = _float(row.get(f"lm_mp_{index}_y"))
        if math.isfinite(x) and math.isfinite(y):
            points.append(
                (
                    float(np.clip(x, 0.0, 1.0) * width),
                    float(np.clip(y, 0.0, 1.0) * height),
                )
            )
    if len(points) < 2:
        return np.zeros(4, dtype=np.float32), False
    values = np.asarray(points, dtype=np.float32)
    low = values.min(axis=0)
    high = values.max(axis=0)
    margin = max(float(max(high - low)) * 0.8, 8.0)
    x0 = max(0, int(low[0] - margin))
    y0 = max(0, int(low[1] - margin))
    x1 = min(width, int(high[0] + margin))
    y1 = min(height, int(high[1] + margin))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros(4, dtype=np.float32), False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
    return (
        np.asarray(
            [
                float(np.mean(gray)),
                float(np.std(gray)),
                float(np.mean(edges > 0)),
                float(np.clip(np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var()) / 8.0, 0.0, 1.0)),
            ],
            dtype=np.float32,
        ),
        True,
    )


def extract_transition_features(
    *,
    video_path: str | Path,
    au_path: str | Path,
    max_frames: int = 24,
    frame_size: int = 512,
) -> dict[str, float]:
    rows: list[dict[str, str]] = []
    with Path(au_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = _sample_rows(rows, max_frames)
    frames = _read_sampled_frames(
        video_path=Path(video_path),
        num_frames=max_frames,
        frame_size=frame_size,
    )
    count = min(len(rows), len(frames))
    rows = rows[:count]
    frames = frames[:count]
    result = {name: 0.0 for name in TRANSITION_FEATURE_NAMES}
    if count < 2:
        for region in _REGION_LANDMARKS:
            result[f"{region}_missing_mask"] = 1.0
        return result

    intensity_names = [
        name
        for name in (rows[0].keys() if rows else [])
        if name.startswith("au_") and name.endswith("_intensity")
    ]
    intensity = np.asarray(
        [
            [_float(row.get(name)) for name in intensity_names]
            for row in rows
        ],
        dtype=np.float32,
    )
    intensity = np.nan_to_num(intensity, nan=0.0)
    au_velocity = np.abs(np.diff(intensity, axis=0))
    au_acceleration = (
        np.abs(np.diff(au_velocity, axis=0))
        if len(au_velocity) > 1
        else au_velocity
    )
    au_signal = (
        np.mean(au_velocity, axis=1)
        if len(au_velocity)
        else np.zeros(1, dtype=np.float32)
    )
    threshold = float(np.mean(au_signal) + np.std(au_signal))
    active = au_signal >= max(threshold, 0.03)
    longest = 0
    current = 0
    for value in active:
        current = current + 1 if value else 0
        longest = max(longest, current)
    speed = float(np.max(au_signal)) if len(au_signal) else 0.0
    accel_signal = (
        np.mean(au_acceleration, axis=1)
        if len(au_acceleration)
        else np.zeros(1, dtype=np.float32)
    )
    result.update(
        {
            "transition_event_ratio_0_1": float(np.mean(active)),
            "transition_longest_event_ratio_0_1": float(
                longest / max(len(active), 1)
            ),
            "transition_event_count_norm_0_1": float(
                min(np.sum(np.diff(active.astype(np.int8)) == 1) / 4.0, 1.0)
            ),
            "transition_peak_speed_0_1": _normalize(speed, 1.0),
            "transition_peak_acceleration_0_1": _normalize(
                float(np.max(accel_signal)), 1.0
            ),
            "transition_onset_offset_asymmetry_0_1": _normalize(
                abs(float(np.mean(au_signal[: max(1, len(au_signal) // 2)]))
                    - float(np.mean(au_signal[len(au_signal) // 2 :])))
            ),
            "au_speed_mean_0_1": _normalize(float(np.mean(au_velocity)), 1.0),
            "au_speed_p95_0_1": _normalize(float(np.quantile(au_velocity, 0.95)), 1.0),
            "au_acceleration_mean_0_1": _normalize(float(np.mean(au_acceleration)), 1.0),
            "au_acceleration_p95_0_1": _normalize(float(np.quantile(au_acceleration, 0.95)), 1.0),
        }
    )

    all_points: list[np.ndarray] = []
    valid_rows = 0
    detections: list[float] = []
    for row in rows:
        points = []
        for key, value in row.items():
            if key.startswith("lm_mp_") and key.endswith("_x"):
                index = key[len("lm_mp_") : -2]
                x = _float(value)
                y = _float(row.get(f"lm_mp_{index}_y"))
                if math.isfinite(x) and math.isfinite(y):
                    points.append((x, y))
        if points:
            all_points.append(np.asarray(points, dtype=np.float32))
            valid_rows += 1
        score = _float(row.get("face_detection_score"))
        if math.isfinite(score):
            detections.append(float(np.clip(score, 0.0, 1.0)))
    if all_points:
        common = min(len(item) for item in all_points)
        points = np.stack([item[:common] for item in all_points])
        lm_velocity = np.linalg.norm(np.diff(points, axis=0), axis=-1)
        lm_accel = (
            np.linalg.norm(np.diff(np.diff(points, axis=0), axis=0), axis=-1)
            if len(points) > 2
            else lm_velocity
        )
        result.update(
            {
                "landmark_speed_mean_0_1": _normalize(float(np.mean(lm_velocity)), 0.15),
                "landmark_speed_p95_0_1": _normalize(float(np.quantile(lm_velocity, 0.95)), 0.25),
                "landmark_acceleration_mean_0_1": _normalize(float(np.mean(lm_accel)), 0.15),
                "landmark_acceleration_p95_0_1": _normalize(float(np.quantile(lm_accel, 0.95)), 0.25),
            }
        )
    result["face_geometry_valid_ratio_0_1"] = float(valid_rows / max(count, 1))
    result["face_detection_confidence_0_1"] = (
        float(np.mean(detections)) if detections else 0.5
    )
    result["face_detection_missing_mask"] = float(not detections)

    for region, indices in _REGION_LANDMARKS.items():
        stats: list[np.ndarray] = []
        valid: list[bool] = []
        for frame, row in zip(frames, rows):
            stat, ok = _region_crop_stats(frame, row, indices)
            stats.append(stat)
            valid.append(ok)
        array = np.stack(stats)
        motion = np.linalg.norm(np.diff(array, axis=0), axis=-1)
        acceleration = (
            np.abs(np.diff(motion))
            if len(motion) > 1
            else motion
        )
        result[f"{region}_motion_mean_0_1"] = _normalize(float(np.mean(motion)), 0.5)
        result[f"{region}_motion_p95_0_1"] = _normalize(float(np.quantile(motion, 0.95)), 1.0)
        result[f"{region}_acceleration_mean_0_1"] = _normalize(float(np.mean(acceleration)), 0.5)
        result[f"{region}_temporal_residual_0_1"] = _normalize(
            float(np.mean(np.abs(np.diff(array, axis=0)))), 0.5
        )
        result[f"{region}_continuity_0_1"] = float(np.mean(valid))
        result[f"{region}_missing_mask"] = float(not any(valid))
    return {
        name: float(result.get(name, 0.0))
        for name in TRANSITION_FEATURE_NAMES
    }


def transition_feature_vector(
    *,
    video_path: str | Path,
    au_path: str | Path,
) -> np.ndarray:
    values = extract_transition_features(
        video_path=video_path,
        au_path=au_path,
    )
    return np.asarray(
        [values[name] for name in TRANSITION_FEATURE_NAMES],
        dtype=np.float32,
    )
