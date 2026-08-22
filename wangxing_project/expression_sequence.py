"""Profile-independent facial expression sequence features for PT v4.1."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

AU_INTENSITY_IDS = (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)
AU_PRESENCE_IDS = (1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24)
LANDMARK_IDS = (
    33, 133, 159, 145, 263, 362, 386, 374,
    61, 291, 13, 14, 70, 107, 300, 336, 152,
)
REGION_SLICES = {
    "left_eye": slice(0, 4),
    "right_eye": slice(4, 8),
    "mouth": slice(8, 12),
    "brow": slice(12, 16),
    "chin": slice(16, 17),
}
SEQUENCE_FRAME_DIM = (
    len(AU_INTENSITY_IDS)
    + len(AU_PRESENCE_IDS)
    + len(LANDMARK_IDS) * 2
    + 3
    + 1
)
SEQUENCE_MAX_FRAMES = 24
SEQUENCE_SUMMARY_NAMES = (
    "au_speed_mean",
    "au_speed_p95",
    "au_acceleration_mean",
    "au_acceleration_p95",
    "landmark_speed_mean",
    "landmark_speed_p95",
    "landmark_acceleration_mean",
    "landmark_acceleration_p95",
    "eye_mouth_sync",
    "left_right_symmetry",
    "valid_frame_ratio",
    "timestamp_regularity",
)
SEQUENCE_SUMMARY_DIM = len(SEQUENCE_SUMMARY_NAMES)


def _float(value: str | None, default: float = 0.0) -> float:
    try:
        parsed = float(value or "")
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _sample_rows(
    rows: list[dict[str, str]],
    max_frames: int,
) -> list[dict[str, str]]:
    if len(rows) <= max_frames:
        return rows
    indices = np.linspace(0, len(rows) - 1, max_frames).round().astype(int)
    return [rows[int(index)] for index in indices]


def _landmark_frame(row: dict[str, str]) -> tuple[np.ndarray, bool]:
    points = []
    for index in LANDMARK_IDS:
        x = _float(row.get(f"lm_mp_{index}_x"), math.nan)
        y = _float(row.get(f"lm_mp_{index}_y"), math.nan)
        points.append((x, y))
    points_array = np.asarray(points, dtype=np.float32)
    valid = np.isfinite(points_array).all()
    if not valid:
        return np.zeros((len(LANDMARK_IDS), 2), dtype=np.float32), False
    left_eye = points_array[0:4].mean(axis=0)
    right_eye = points_array[4:8].mean(axis=0)
    center = 0.5 * (left_eye + right_eye)
    scale = float(np.linalg.norm(right_eye - left_eye))
    scale = max(scale, 1e-3)
    normalized = (points_array - center[None, :]) / scale
    return np.clip(normalized, -8.0, 8.0), True


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or len(right) < 3:
        return 0.0
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-6:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def extract_expression_sequence_features(
    au_path: str | Path,
    *,
    max_frames: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a fixed sequence and profile-independent temporal summary."""
    with Path(au_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = _sample_rows(list(csv.DictReader(handle)), max_frames)
    if not rows:
        return (
            np.zeros((max_frames, SEQUENCE_FRAME_DIM), dtype=np.float32),
            np.zeros(SEQUENCE_SUMMARY_DIM, dtype=np.float32),
        )

    frames: list[np.ndarray] = []
    landmarks: list[np.ndarray] = []
    valid_flags: list[float] = []
    timestamps: list[float] = []
    for row in rows:
        intensities = np.asarray(
            [
                np.clip(
                    _float(row.get(f"au_{index}_intensity")) / 5.0,
                    0.0,
                    1.0,
                )
                for index in AU_INTENSITY_IDS
            ],
            dtype=np.float32,
        )
        presence = np.asarray(
            [
                np.clip(_float(row.get(f"au_{index}")), 0.0, 1.0)
                for index in AU_PRESENCE_IDS
            ],
            dtype=np.float32,
        )
        landmark, valid = _landmark_frame(row)
        pose = np.asarray(
            [
                np.clip(_float(row.get("pitch")) / 180.0, -1.0, 1.0),
                np.clip(_float(row.get("yaw")) / 180.0, -1.0, 1.0),
                np.clip(_float(row.get("roll")) / 180.0, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        frame = np.concatenate(
            [
                intensities,
                presence,
                landmark.reshape(-1),
                pose,
                np.asarray([float(valid)], dtype=np.float32),
            ]
        )
        frames.append(np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0))
        landmarks.append(landmark)
        valid_flags.append(float(valid))
        timestamps.append(_float(row.get("frame_time_in_ms"), float(len(frames))))

    sequence = np.stack(frames).astype(np.float32)
    if len(sequence) < max_frames:
        pad_count = max_frames - len(sequence)
        pad_value = sequence[-1:] if len(sequence) else np.zeros(
            (1, SEQUENCE_FRAME_DIM),
            dtype=np.float32,
        )
        sequence = np.concatenate(
            [sequence, np.repeat(pad_value, pad_count, axis=0)],
            axis=0,
        )

    landmark_array = np.stack(landmarks).astype(np.float32)
    valid = np.asarray(valid_flags, dtype=np.float32)
    intensities = sequence[: len(rows), : len(AU_INTENSITY_IDS)]
    au_velocity = np.diff(intensities, axis=0)
    au_acceleration = np.diff(au_velocity, axis=0)
    landmark_velocity = np.linalg.norm(
        np.diff(landmark_array, axis=0),
        axis=-1,
    )
    landmark_acceleration = np.diff(landmark_velocity, axis=0)

    mouth = landmark_array[:, REGION_SLICES["mouth"], :]
    left_eye = landmark_array[:, REGION_SLICES["left_eye"], :]
    right_eye = landmark_array[:, REGION_SLICES["right_eye"], :]
    mouth_motion = (
        np.linalg.norm(np.diff(mouth, axis=0), axis=-1).mean(axis=1)
        if len(mouth) > 1
        else np.zeros(1, dtype=np.float32)
    )
    eye_motion = (
        0.5
        * (
            np.linalg.norm(np.diff(left_eye, axis=0), axis=-1).mean(axis=1)
            + np.linalg.norm(np.diff(right_eye, axis=0), axis=-1).mean(axis=1)
        )
        if len(left_eye) > 1
        else np.zeros(1, dtype=np.float32)
    )
    left_right = np.linalg.norm(
        np.mean(left_eye, axis=1) - np.mean(right_eye, axis=1),
        axis=-1,
    )
    timestamp_diff = np.diff(np.asarray(timestamps, dtype=np.float32))
    timestamp_regularity = 1.0
    if len(timestamp_diff) >= 2 and np.mean(np.abs(timestamp_diff)) > 1e-6:
        timestamp_regularity = float(
            1.0
            - np.clip(
                np.std(timestamp_diff)
                / max(float(np.mean(np.abs(timestamp_diff))), 1e-6),
                0.0,
                1.0,
            )
        )
    summary = np.asarray(
        [
            float(np.mean(np.abs(au_velocity))) if au_velocity.size else 0.0,
            float(np.quantile(np.abs(au_velocity), 0.95))
            if au_velocity.size
            else 0.0,
            float(np.mean(np.abs(au_acceleration)))
            if au_acceleration.size
            else 0.0,
            float(np.quantile(np.abs(au_acceleration), 0.95))
            if au_acceleration.size
            else 0.0,
            float(np.mean(landmark_velocity)) if landmark_velocity.size else 0.0,
            float(np.quantile(landmark_velocity, 0.95))
            if landmark_velocity.size
            else 0.0,
            float(np.mean(np.abs(landmark_acceleration)))
            if landmark_acceleration.size
            else 0.0,
            float(np.quantile(np.abs(landmark_acceleration), 0.95))
            if landmark_acceleration.size
            else 0.0,
            float((1.0 + _safe_corr(eye_motion, mouth_motion)) / 2.0),
            float(1.0 - np.clip(np.std(left_right) / 0.25, 0.0, 1.0)),
            float(np.mean(valid)),
            timestamp_regularity,
        ],
        dtype=np.float32,
    )
    return (
        np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0),
    )
