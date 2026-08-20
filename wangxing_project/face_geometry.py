"""Small face-geometry descriptors derived from LibreFace AU CSV files."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

FACE_GEOMETRY_DIM = 16


def _float(value: str | None) -> float:
    try:
        parsed = float(value or "")
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def extract_face_geometry_features(path: str | Path) -> np.ndarray:
    """Return quality-aware landmark/pose statistics for one AU CSV."""
    rows: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    detection: list[float] = []
    landmark_valid: list[float] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        landmark_columns = sorted(
            {
                name[:-2]
                for name in (reader.fieldnames or [])
                if name.startswith("lm_mp_") and name.endswith("_x")
            }
        )
        for record in reader:
            points: list[tuple[float, float]] = []
            for prefix in landmark_columns:
                x = _float(record.get(prefix + "_x"))
                y = _float(record.get(prefix + "_y"))
                if math.isfinite(x) and math.isfinite(y):
                    points.append((x, y))
            if points:
                array = np.asarray(points, dtype=np.float32)
                center = array.mean(axis=0)
                spread = array.std(axis=0)
                extent = array.max(axis=0) - array.min(axis=0)
                radius = np.linalg.norm(array - center[None, :], axis=1)
                rows.append(
                    np.asarray(
                        [
                            center[0],
                            center[1],
                            spread[0],
                            spread[1],
                            extent[0],
                            extent[1],
                            float(np.mean(radius)),
                            float(np.std(radius)),
                        ],
                        dtype=np.float32,
                    )
                )
                landmark_valid.append(1.0)
            else:
                landmark_valid.append(0.0)
            poses.append(
                np.asarray(
                    [
                        _float(record.get("pitch")),
                        _float(record.get("yaw")),
                        _float(record.get("roll")),
                    ],
                    dtype=np.float32,
                )
            )
            score = _float(record.get("face_detection_score"))
            if math.isfinite(score):
                detection.append(float(np.clip(score, 0.0, 1.0)))

    if not rows:
        return np.zeros(FACE_GEOMETRY_DIM, dtype=np.float32)
    landmark_array = np.stack(rows)
    pose_array = np.asarray(poses, dtype=np.float32)
    pose_mean = np.nanmean(pose_array, axis=0)
    pose_std = np.nanstd(pose_array, axis=0)
    pose_mean = np.nan_to_num(pose_mean, nan=0.0)
    pose_std = np.nan_to_num(pose_std, nan=0.0)
    values = np.concatenate(
        [
            landmark_array.mean(axis=0),
            landmark_array.std(axis=0),
            pose_mean,
            pose_std,
        ]
    ).astype(np.float32)
    # Keep the descriptor compact: 8 landmark means + 8 pose/quality values.
    descriptor = np.asarray(
        [
            *values[:8],
            *values[8:14],
            float(np.mean(landmark_valid)),
            float(np.mean(detection)) if detection else 0.5,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(descriptor, nan=0.0, posinf=1.0, neginf=0.0)
