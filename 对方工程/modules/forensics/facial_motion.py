from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..core.face_landmarker import normalize_csv_landmark_frame
from .au_ssl import (
    extract_self_supervised_au_features,
    merge_ssl_into_motion_features,
)
from .au_ssl_backbone import (
    extract_backbone_features,
    merge_backbone_into_ssl,
)
from .physiological_rhythm import (
    extract_physiological_rhythm_features,
    merge_physio_into_motion_features,
)

FACIAL_MOTION_SCHEMA = "facial_motion_forensics_v1"
DEFAULT_AU_IDS = (
    1,
    2,
    4,
    5,
    6,
    7,
    9,
    10,
    12,
    14,
    15,
    17,
    20,
    23,
    24,
    25,
    26,
)
DEFAULT_ACTIVE_THRESHOLD = 0.20
DEFAULT_MAX_LAG = 4

# The groups are intentionally compact for the initial implementation. They
# can later be expanded to all 478 landmarks without changing the output API.
LANDMARK_GROUPS: dict[str, tuple[int, ...]] = {
    "brow_left": (70, 105, 107),
    "brow_right": (300, 334, 336),
    "eye_left": (33, 133, 145, 159),
    "eye_right": (263, 362, 374, 386),
    "mouth": (13, 14, 61, 291),
    "cheek_left": (116, 123, 147),
    "cheek_right": (345, 352, 376),
    "jaw": (172, 397, 152),
}
FACE_ANCHORS = (234, 454, 10, 152)
COACTIVATION_PAIRS = ((1, 2), (4, 7), (6, 12), (12, 25), (17, 26))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _au_id(name: str) -> int | None:
    match = re.search(r"\bau[\s_-]*0*(\d{1,2})(?!\d)", str(name), re.IGNORECASE)
    if not match:
        return None
    value = int(match.group(1))
    return value if 1 <= value <= 45 else None


def _is_intensity_column(name: str) -> bool:
    normalized = str(name).lower().replace(" ", "")
    return (
        "intensity" in normalized
        or normalized.endswith("_r")
        or bool(re.fullmatch(r"au[_-]*0*\d{1,2}", normalized))
    )


def _column_for_au(fieldnames: Sequence[str], au_id: int) -> str | None:
    candidates = [
        name
        for name in fieldnames
        if _au_id(name) == au_id and _is_intensity_column(name)
    ]
    return candidates[0] if candidates else None


def _read_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        delimiter = "\t" if "\t" in first_line else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _landmark_columns(
    fieldnames: Iterable[str],
) -> dict[int, tuple[str, str]]:
    names = list(fieldnames)
    pairs: dict[int, tuple[str, str]] = {}
    for name in names:
        match = re.fullmatch(r"lm_mp_(\d+)_x", str(name), re.IGNORECASE)
        if not match:
            continue
        index = int(match.group(1))
        y_name = next(
            (
                candidate
                for candidate in names
                if str(candidate).lower() == f"lm_mp_{index}_y"
            ),
            None,
        )
        if y_name is not None:
            pairs[index] = (name, y_name)
    return pairs


def _landmark_z_columns(
    fieldnames: Iterable[str],
) -> dict[int, str]:
    columns: dict[int, str] = {}
    for name in fieldnames:
        match = re.fullmatch(r"lm_mp_(\d+)_z", str(name), re.IGNORECASE)
        if match:
            columns[int(match.group(1))] = name
    return columns


def _blendshape_columns(fieldnames: Iterable[str]) -> list[str]:
    return [
        name
        for name in fieldnames
        if str(name).lower().startswith(("blendshape_", "bs_"))
    ]


def _frame_landmarks(
    row: dict[str, str],
    landmark_columns: dict[int, tuple[str, str]],
) -> dict[int, np.ndarray]:
    points: dict[int, np.ndarray] = {}
    for index, (x_name, y_name) in landmark_columns.items():
        x = _finite(row.get(x_name), math.nan)
        y = _finite(row.get(y_name), math.nan)
        if math.isfinite(x) and math.isfinite(y):
            points[index] = np.asarray([x, y], dtype=np.float32)
    return points


def _frame_landmark_z(
    row: dict[str, str],
    z_columns: dict[int, str],
) -> dict[int, float]:
    values: dict[int, float] = {}
    for index, name in z_columns.items():
        value = _finite(row.get(name), math.nan)
        if math.isfinite(value):
            values[index] = value
    return values


def _normalized_landmark_frame(
    points: dict[int, np.ndarray],
    *,
    points_z: dict[int, float] | None = None,
    pose_normalize: bool = True,
) -> dict[int, np.ndarray]:
    if pose_normalize:
        pose_normalized = normalize_csv_landmark_frame(
            points,
            points_z=points_z,
        )
        if pose_normalized:
            return pose_normalized
    if not all(index in points for index in FACE_ANCHORS):
        return {}
    left, right, top, bottom = (points[index] for index in FACE_ANCHORS)
    width = float(np.linalg.norm(right - left))
    height = float(np.linalg.norm(bottom - top))
    if width < 1e-6 or height < 1e-6:
        return {}
    center = (left + right + top + bottom) / 4.0
    normalized: dict[int, np.ndarray] = {}
    for index, point in points.items():
        normalized[index] = np.asarray(
            [(point[0] - center[0]) / width, (point[1] - center[1]) / height],
            dtype=np.float32,
        )
    return normalized


def _group_centroids(
    normalized_points: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for name, indexes in LANDMARK_GROUPS.items():
        values = [normalized_points[index] for index in indexes if index in normalized_points]
        if values:
            centroids[name] = np.mean(np.stack(values), axis=0)
    return centroids


def _fill_missing(sequence: np.ndarray) -> np.ndarray:
    result = np.asarray(sequence, dtype=np.float32).copy()
    if result.ndim != 2:
        raise ValueError("Expected a two-dimensional sequence.")
    for column in range(result.shape[1]):
        values = result[:, column]
        valid = values[np.isfinite(values)]
        replacement = float(np.median(valid)) if valid.size else 0.0
        values[~np.isfinite(values)] = replacement
    return result


def _safe_statistic(values: np.ndarray, statistic: str) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    if statistic == "median":
        return float(np.median(finite))
    if statistic == "q25":
        return float(np.quantile(finite, 0.25))
    if statistic == "q75":
        return float(np.quantile(finite, 0.75))
    if statistic == "std":
        return float(np.std(finite))
    if statistic == "p95":
        return float(np.quantile(finite, 0.95))
    if statistic == "max":
        return float(np.max(finite))
    raise ValueError(f"Unsupported statistic: {statistic}")


def _sequence_summary(
    values: np.ndarray,
    prefix: str,
    *,
    timestamps_seconds: np.ndarray | None = None,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    if timestamps_seconds is None or len(timestamps_seconds) != len(values):
        velocity = np.diff(values) if len(values) > 1 else np.zeros(1)
        acceleration = (
            np.diff(values, n=2) if len(values) > 2 else np.zeros(1)
        )
        jerk = np.diff(values, n=3) if len(values) > 3 else np.zeros(1)
    else:
        timestamps = np.asarray(timestamps_seconds, dtype=np.float32)
        deltas = np.diff(timestamps)
        positive_deltas = deltas[deltas > 1e-6]
        fallback_delta = (
            float(np.median(positive_deltas))
            if positive_deltas.size
            else 1.0
        )
        deltas = np.where(deltas > 1e-6, deltas, fallback_delta)
        velocity = (
            np.diff(values) / deltas
            if len(values) > 1
            else np.zeros(1)
        )
        if len(velocity) > 1:
            velocity_deltas = np.maximum(
                (deltas[:-1] + deltas[1:]) / 2.0,
                1e-6,
            )
            acceleration = np.diff(velocity) / velocity_deltas
        else:
            acceleration = np.zeros(1)
        if len(acceleration) > 1:
            acceleration_deltas = np.maximum(
                velocity_deltas[1:],
                1e-6,
            )
            jerk = np.diff(acceleration) / acceleration_deltas
        else:
            jerk = np.zeros(1)
    result = {
        f"{prefix}_median": _safe_statistic(values, "median"),
        f"{prefix}_q25": _safe_statistic(values, "q25"),
        f"{prefix}_q75": _safe_statistic(values, "q75"),
        f"{prefix}_std": _safe_statistic(values, "std"),
        f"{prefix}_p95": _safe_statistic(values, "p95"),
        f"{prefix}_max": _safe_statistic(values, "max"),
        f"{prefix}_velocity_median": _safe_statistic(np.abs(velocity), "median"),
        f"{prefix}_velocity_p95": _safe_statistic(np.abs(velocity), "p95"),
        f"{prefix}_acceleration_p95": _safe_statistic(
            np.abs(acceleration),
            "p95",
        ),
        f"{prefix}_jerk_p95": _safe_statistic(np.abs(jerk), "p95"),
    }
    return result


def _event_features(values: np.ndarray, threshold: float) -> dict[str, float]:
    active = np.asarray(values >= threshold, dtype=bool)
    if active.size == 0:
        return {
            "event_count": 0.0,
            "active_ratio": 0.0,
            "longest_event_ratio": 0.0,
            "mean_event_ratio": 0.0,
            "peak_intensity": 0.0,
        }
    lengths: list[int] = []
    current = 0
    for item in active:
        if item:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    frame_count = max(len(active), 1)
    return {
        "event_count": float(len(lengths)),
        "active_ratio": float(np.mean(active)),
        "longest_event_ratio": float(max(lengths, default=0) / frame_count),
        "mean_event_ratio": (
            float(np.mean(lengths) / frame_count) if lengths else 0.0
        ),
        "peak_intensity": float(np.max(values)),
    }


def _timestamp_axis(
    rows: Sequence[dict[str, str]],
) -> tuple[np.ndarray | None, str]:
    for field_name, unit in (
        ("frame_time_in_ms", "milliseconds"),
        ("timestamp_ms", "milliseconds"),
        ("timestamp", "seconds"),
    ):
        if not any(field_name in row for row in rows):
            continue
        values = np.asarray(
            [_finite(row.get(field_name), math.nan) for row in rows],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            continue
        if unit == "milliseconds":
            # Older LibreFace exports label the column as milliseconds but
            # actually write seconds (e.g. 0.033 for the second frame).
            # Match au_compliance: treat tiny maxima as already-seconds.
            max_value = float(np.max(values))
            if max_value < max(100.0, len(values) * 2.0):
                timebase = f"{field_name}_seconds_legacy"
            else:
                values /= 1000.0
                timebase = f"{field_name}_seconds"
        else:
            timebase = f"{field_name}_seconds"
        values -= values[0]
        if len(values) < 2 or np.any(np.diff(values) <= 0.0):
            continue
        return values.astype(np.float32), timebase
    return None, "row_index"


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if len(left) < 2 or len(right) != len(left):
        return 0.0
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _training_free_motion_prior(
    features: dict[str, Any],
    au_matrix: np.ndarray,
) -> dict[str, float]:
    """Heuristic realness prior from AU relations / rhythm (no profile needed)."""
    pair_scores: list[float] = []
    for key, value in features.items():
        if not str(key).startswith("au_pair_corr_"):
            continue
        # Expected co-activation pairs should lean non-negative for natural faces.
        pair_scores.append(_clamp((float(value) + 1.0) / 2.0))
    coactivation = (
        float(np.mean(pair_scores)) if pair_scores else 0.50
    )
    motion = _clamp(_finite(features.get("motion_coherence_0_1"), 0.5))
    phase = _clamp(_finite(features.get("landmark_phase_coherence_0_1"), 0.5))
    active = _clamp(_finite(features.get("au_event_active_ratio"), 0.0))
    # Talking-head clips are rarely fully idle or fully saturated.
    activity = _clamp(1.0 - abs(active - 0.55) / 0.55)

    dynamics_scores: list[float] = []
    velocities: list[float] = []
    for au_id in DEFAULT_AU_IDS:
        velocity = abs(
            _finite(features.get(f"au_{au_id:02d}_velocity_p95"), 0.0)
        )
        acceleration = abs(
            _finite(features.get(f"au_{au_id:02d}_acceleration_p95"), 0.0)
        )
        if velocity <= 1e-8 and acceleration <= 1e-8:
            continue
        velocities.append(velocity)
        # Use acceleration/velocity on a wide log scale; time-aware
        # derivatives make raw jerk ratios too extreme to be useful.
        log_ratio = math.log1p(acceleration) - math.log1p(max(velocity, 1e-8))
        dynamics_scores.append(_clamp(1.0 - abs(log_ratio - 2.5) / 6.0))
    if dynamics_scores:
        dynamics = float(np.mean(dynamics_scores))
    elif velocities:
        dynamics = 0.55
    else:
        dynamics = 0.50

    richness = 0.50
    if au_matrix.size:
        active_channels = float(
            np.mean(np.max(au_matrix, axis=0) > 0.08)
        )
        richness = _clamp(0.25 + 0.75 * active_channels)

    prior = _clamp(
        0.28 * coactivation
        + 0.24 * motion
        + 0.18 * phase
        + 0.15 * dynamics
        + 0.10 * activity
        + 0.05 * richness
    )
    return {
        "au_relation_consistency_0_1": coactivation,
        "au_dynamics_naturalness_0_1": dynamics,
        "au_activation_richness_0_1": richness,
        "training_free_motion_prior_0_1": prior,
    }


def _max_lag_correlation(
    left: np.ndarray,
    right: np.ndarray,
    max_lag: int = DEFAULT_MAX_LAG,
) -> tuple[float, float]:
    if len(left) < 3 or len(right) != len(left):
        return 0.0, 0.0
    candidates: list[tuple[float, int]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            score = _correlation(left[:lag], right[-lag:])
        elif lag > 0:
            score = _correlation(left[lag:], right[:-lag])
        else:
            score = _correlation(left, right)
        candidates.append((score, lag))
    return max(candidates, key=lambda item: abs(item[0]))


def _feature_vector(features: dict[str, Any], names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [_finite(features.get(name), 0.0) for name in names],
        dtype=np.float32,
    )


def _is_landmark_feature(name: str) -> bool:
    return (
        name.startswith("landmark_")
        or name in {
            "motion_coherence_0_1",
        }
    )


def _scoring_feature_names(
    profile_names: Sequence[str],
    feature_record: dict[str, Any],
) -> tuple[list[str], str]:
    """Avoid treating unavailable Face Mesh values as measured zeros."""
    landmark_available = bool(feature_record.get("landmark_available"))
    coverage = _finite(
        feature_record.get("features", {}).get(
            "landmark_valid_frame_ratio",
            0.0,
        )
    )
    if landmark_available and coverage >= 0.5:
        return list(profile_names), "au_plus_landmark"
    au_names = [
        name for name in profile_names if not _is_landmark_feature(name)
    ]
    return (au_names or list(profile_names)), "au_only"


def extract_facial_motion_features(
    csv_path: str | Path,
    *,
    au_ids: Iterable[int] = DEFAULT_AU_IDS,
    active_threshold: float = DEFAULT_ACTIVE_THRESHOLD,
    time_aware_derivatives: bool = False,
    pose_normalize: bool = True,
    include_ssl: bool = True,
    include_physio: bool = True,
    include_ssl_backbone: bool = True,
) -> dict[str, Any]:
    """Extract AU and pose-normalized Face Mesh motion features from one CSV."""
    rows, fieldnames = _read_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in AU CSV: {csv_path}")

    requested_au_ids = tuple(int(value) for value in au_ids)
    timestamps_seconds, timebase = _timestamp_axis(rows)
    derivative_timestamps = (
        timestamps_seconds if time_aware_derivatives else None
    )
    au_columns = {
        au_id: _column_for_au(fieldnames, au_id) for au_id in requested_au_ids
    }
    au_matrix = np.full((len(rows), len(requested_au_ids)), np.nan, dtype=np.float32)
    for row_index, row in enumerate(rows):
        for column_index, au_id in enumerate(requested_au_ids):
            column = au_columns[au_id]
            if column is None:
                continue
            value = _finite(row.get(column), math.nan)
            if math.isfinite(value) and value > 1.0:
                value /= 5.0
            au_matrix[row_index, column_index] = _clamp(value, 0.0, 1.0)
    au_matrix = _fill_missing(au_matrix)

    landmark_columns = _landmark_columns(fieldnames)
    landmark_z_columns = _landmark_z_columns(fieldnames)
    blendshape_names = _blendshape_columns(fieldnames)
    blendshape_matrix = None
    if blendshape_names:
        blendshape_matrix = np.full(
            (len(rows), len(blendshape_names)),
            np.nan,
            dtype=np.float32,
        )
        for row_index, row in enumerate(rows):
            for column_index, name in enumerate(blendshape_names):
                blendshape_matrix[row_index, column_index] = _clamp(
                    _finite(row.get(name), math.nan),
                    0.0,
                    1.0,
                )
        blendshape_matrix = _fill_missing(blendshape_matrix)
    group_names = tuple(LANDMARK_GROUPS)
    landmark_matrix = np.full(
        (len(rows), len(group_names) * 2),
        np.nan,
        dtype=np.float32,
    )
    valid_landmark_frames = 0
    pose_normalized_frames = 0
    physio_landmark_frames: list[dict[int, np.ndarray]] = []
    for row_index, row in enumerate(rows):
        raw_points = _frame_landmarks(row, landmark_columns)
        z_values = _frame_landmark_z(row, landmark_z_columns)
        pose_points = (
            normalize_csv_landmark_frame(
                raw_points,
                points_z=z_values or None,
            )
            if pose_normalize
            else {}
        )
        if pose_points:
            pose_normalized_frames += 1
            points = pose_points
        else:
            points = _normalized_landmark_frame(
                raw_points,
                points_z=z_values or None,
                pose_normalize=False,
            )
        if points:
            physio_landmark_frames.append(
                {
                    index: np.asarray(coord, dtype=np.float32)
                    for index, coord in points.items()
                }
            )
        groups = _group_centroids(points)
        if groups:
            valid_landmark_frames += 1
        for group_index, group_name in enumerate(group_names):
            if group_name not in groups:
                continue
            landmark_matrix[row_index, group_index * 2 : group_index * 2 + 2] = (
                groups[group_name]
            )
    landmark_available = valid_landmark_frames >= max(3, len(rows) // 3)
    if landmark_available:
        landmark_matrix = _fill_missing(landmark_matrix)
    else:
        landmark_matrix = np.zeros_like(landmark_matrix)

    features: dict[str, Any] = {}
    for column_index, au_id in enumerate(requested_au_ids):
        features.update(
            _sequence_summary(
                au_matrix[:, column_index],
                f"au_{au_id:02d}",
                timestamps_seconds=derivative_timestamps,
            )
        )
    event_signal = np.max(au_matrix, axis=1) if au_matrix.size else np.zeros(1)
    features.update(
        {
            f"au_event_{key}": value
            for key, value in _event_features(event_signal, active_threshold).items()
        }
    )

    for group_index, group_name in enumerate(group_names):
        if not landmark_available:
            continue
        x_values = landmark_matrix[:, group_index * 2]
        y_values = landmark_matrix[:, group_index * 2 + 1]
        radial = np.sqrt(x_values * x_values + y_values * y_values)
        features.update(
            _sequence_summary(
                radial,
                f"landmark_{group_name}",
                timestamps_seconds=derivative_timestamps,
            )
        )

    for left_id, right_id in COACTIVATION_PAIRS:
        if left_id not in requested_au_ids or right_id not in requested_au_ids:
            continue
        left_index = requested_au_ids.index(left_id)
        right_index = requested_au_ids.index(right_id)
        features[f"au_pair_corr_{left_id:02d}_{right_id:02d}"] = _correlation(
            au_matrix[:, left_index],
            au_matrix[:, right_index],
        )

    phase_values: list[float] = []
    lag_values: list[float] = []
    motion_columns: dict[str, np.ndarray] = {}
    for group_index, group_name in enumerate(group_names):
        if not landmark_available:
            continue
        x_values = landmark_matrix[:, group_index * 2]
        y_values = landmark_matrix[:, group_index * 2 + 1]
        motion_columns[group_name] = np.sqrt(
            np.diff(x_values, prepend=x_values[0]) ** 2
            + np.diff(y_values, prepend=y_values[0]) ** 2
        )
    for left_name, right_name in (
        ("brow_left", "eye_left"),
        ("brow_right", "eye_right"),
        ("mouth", "jaw"),
        ("cheek_left", "mouth"),
        ("cheek_right", "mouth"),
    ):
        if left_name not in motion_columns or right_name not in motion_columns:
            continue
        correlation, lag = _max_lag_correlation(
            motion_columns[left_name],
            motion_columns[right_name],
        )
        phase_values.append(abs(correlation))
        lag_values.append(float(abs(lag) / max(DEFAULT_MAX_LAG, 1)))
    features["landmark_phase_coherence_0_1"] = (
        float(np.mean(phase_values)) if phase_values else 0.0
    )
    features["landmark_mean_phase_lag_0_1"] = (
        float(np.mean(lag_values)) if lag_values else 0.0
    )
    features["landmark_valid_frame_ratio"] = valid_landmark_frames / max(
        len(rows),
        1,
    )
    features["motion_coherence_0_1"] = _clamp(
        0.7 * float(features["landmark_phase_coherence_0_1"])
        + 0.3 * (1.0 - float(features["landmark_mean_phase_lag_0_1"])),
    ) if landmark_available else 0.0
    features["pose_normalized_frame_ratio"] = pose_normalized_frames / max(
        len(rows),
        1,
    )
    features["blendshape_channel_count"] = float(len(blendshape_names))
    features.update(_training_free_motion_prior(features, au_matrix))
    window_records: list[dict[str, Any]] = []
    window_size = 32
    for window_index, start in enumerate(range(0, len(event_signal), window_size)):
        stop = min(len(event_signal), start + window_size)
        window_signal = event_signal[start:stop]
        event = _event_features(window_signal, active_threshold)
        window_records.append(
            {
                "window_index": window_index,
                "start_frame": start,
                "end_frame": stop - 1,
                "start_time_seconds": (
                    float(timestamps_seconds[start])
                    if timestamps_seconds is not None
                    else None
                ),
                "end_time_seconds": (
                    float(timestamps_seconds[stop - 1])
                    if timestamps_seconds is not None
                    else None
                ),
                "active_ratio": event["active_ratio"],
                "peak_intensity": event["peak_intensity"],
                "evidence_score_0_1": _clamp(
                    0.60 * event["active_ratio"]
                    + 0.40 * event["peak_intensity"]
                ),
            }
        )

    result = {
        "schema_version": FACIAL_MOTION_SCHEMA,
        "source": str(csv_path),
        "frame_count": len(rows),
        "timestamps_seconds": (
            timestamps_seconds.astype(float).tolist()
            if timestamps_seconds is not None
            else None
        ),
        "timebase": timebase,
        "time_aware_derivatives": bool(time_aware_derivatives),
        "pose_normalize": bool(pose_normalize),
        "au_ids": list(requested_au_ids),
        "supported_au_ids": [
            au_id for au_id, column in au_columns.items() if column is not None
        ],
        "landmark_available": landmark_available,
        "features": features,
        "window_records": window_records,
    }
    if include_ssl:
        ssl_result = extract_self_supervised_au_features(
            au_matrix,
            timestamps_seconds=timestamps_seconds,
            blendshape_matrix=blendshape_matrix,
        )
        if include_ssl_backbone:
            backbone_result = extract_backbone_features(au_matrix)
            ssl_result = merge_backbone_into_ssl(ssl_result, backbone_result)
        result = merge_ssl_into_motion_features(result, ssl_result)
    if include_physio:
        blink_signal = None
        # Prefer AU45 intensity if present among columns.
        for field in fieldnames:
            lowered = str(field).lower().replace(" ", "")
            if "au45" in lowered or "au_45" in lowered:
                blink_signal = np.asarray(
                    [_clamp(_finite(row.get(field), 0.0), 0.0, 1.0) for row in rows],
                    dtype=np.float64,
                )
                break
        physio_result = extract_physiological_rhythm_features(
            physio_landmark_frames or None,
            timestamps_seconds=timestamps_seconds,
            blink_signal=blink_signal,
        )
        result = merge_physio_into_motion_features(result, physio_result)
    return result


def _profile_from_feature_records(
    records: Sequence[dict[str, Any]],
    *,
    domain: str,
) -> dict[str, Any]:
    if not records:
        raise ValueError("At least one feature record is required.")
    names = sorted(
        {
            name
            for record in records
            for name in record.get("features", {})
            if isinstance(name, str)
        }
    )
    matrix = np.stack(
        [_feature_vector(record.get("features", {}), names) for record in records]
    )
    mean = np.mean(matrix, axis=0)
    std = np.maximum(np.std(matrix, axis=0), 0.05)
    return {
        "schema_version": FACIAL_MOTION_SCHEMA,
        "domain": domain,
        "sample_count": len(records),
        "feature_names": names,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "source_records": [record.get("source") for record in records],
    }


def build_facial_motion_profile(
    csv_paths: Iterable[str | Path],
    *,
    domain: str = "real",
) -> dict[str, Any]:
    records = [
        extract_facial_motion_features(
            path,
            time_aware_derivatives=True,
        )
        for path in csv_paths
    ]
    return _profile_from_feature_records(records, domain=domain)


def build_two_domain_facial_motion_profile(
    real_csv_paths: Iterable[str | Path],
    seedance_csv_paths: Iterable[str | Path],
) -> dict[str, Any]:
    def _extract_all(
        paths: Sequence[str | Path],
        *,
        label: str,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        total = len(paths)
        for index, path in enumerate(paths, start=1):
            records.append(
                extract_facial_motion_features(
                    path,
                    time_aware_derivatives=True,
                )
            )
            if index == 1 or index == total or index % 25 == 0:
                print(
                    f"  [{label}] {index}/{total} {Path(path).name}",
                    flush=True,
                )
        return records

    real_path_list = list(real_csv_paths)
    seedance_path_list = list(seedance_csv_paths)
    real_records = _extract_all(real_path_list, label="real")
    seedance_records = _extract_all(seedance_path_list, label="seedance")
    if not real_records or not seedance_records:
        raise ValueError("Both real and Seedance records are required.")
    feature_names = sorted(
        {
            name
            for record in [*real_records, *seedance_records]
            for name in record.get("features", {})
        }
    )

    def domain_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        matrix = np.stack(
            [_feature_vector(record["features"], feature_names) for record in records]
        )
        return {
            "sample_count": len(records),
            "mean": np.mean(matrix, axis=0).astype(float).tolist(),
            "std": np.maximum(np.std(matrix, axis=0), 0.05).astype(float).tolist(),
            "source_records": [record.get("source") for record in records],
        }

    return {
        "schema_version": FACIAL_MOTION_SCHEMA,
        "domain": "real_vs_seedance",
        "feature_names": feature_names,
        "real": domain_stats(real_records),
        "seedance": domain_stats(seedance_records),
        "feature_protocol": {
            "time_aware_derivatives": True,
            "derivative_time_unit": "seconds",
        },
        "note": (
            "Initial calibrated profile. It is not a universal detector and "
            "must be evaluated on held-out videos."
        ),
    }


def _fit_score(
    values: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float],
) -> float:
    expected = np.asarray(mean, dtype=np.float32)
    scale = np.maximum(np.asarray(std, dtype=np.float32), 0.05)
    z = (values - expected) / scale
    distance = float(np.mean(np.minimum(np.abs(z), 8.0)))
    return float(math.exp(-distance / 2.0))


def score_facial_motion(
    features_or_csv: dict[str, Any] | str | Path,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Score real-domain fit and optional Seedance-domain fit."""
    features = (
        extract_facial_motion_features(
            features_or_csv,
            time_aware_derivatives=bool(
                profile.get("feature_protocol", {}).get(
                    "time_aware_derivatives",
                    False,
                )
            ),
        )
        if isinstance(features_or_csv, (str, Path))
        else features_or_csv
    )
    profile_names = list(profile.get("feature_names", []))
    names, feature_mode = _scoring_feature_names(
        profile_names,
        features,
    )
    profile_indexes = [profile_names.index(name) for name in names]
    values = _feature_vector(features.get("features", {}), names)
    real = profile.get("real")
    if real is None:
        real = profile if profile.get("domain") == "real" else None
    seedance = profile.get("seedance")
    def _profile_stats(
        domain: dict[str, Any] | None,
    ) -> tuple[list[float], list[float]]:
        if not domain:
            return [], []
        mean = domain.get("mean", [])
        std = domain.get("std", [])
        return (
            [float(mean[index]) for index in profile_indexes],
            [float(std[index]) for index in profile_indexes],
        )

    real_mean, real_std = _profile_stats(real)
    seedance_mean, seedance_std = _profile_stats(seedance)
    real_fit = (
        _fit_score(values, real_mean, real_std)
        if real
        else None
    )
    seedance_fit = (
        _fit_score(values, seedance_mean, seedance_std)
        if seedance
        else None
    )
    authenticity = None
    if real_fit is not None and seedance_fit is not None:
        authenticity = real_fit / max(real_fit + seedance_fit, 1e-6)
    motion_coherence = _finite(
        features.get("features", {}).get("motion_coherence_0_1"),
        0.0,
    )
    feature_map = features.get("features", {})
    training_free_prior = _clamp(
        _finite(
            feature_map.get("training_free_motion_prior_0_1"),
            motion_coherence,
        )
    )
    profile_evidence = authenticity
    enriched_evidence = authenticity
    if authenticity is not None:
        enriched_evidence = _clamp(
            0.82 * float(authenticity) + 0.18 * training_free_prior
        )
    return {
        "status": "calibrated" if authenticity is not None else "features_only",
        "probability_calibrated": False,
        "backend": "au_landmark_temporal_profile",
        "schema_version": FACIAL_MOTION_SCHEMA,
        "metrics": {
            "real_domain_fit_0_1": real_fit,
            "seedance_domain_fit_0_1": seedance_fit,
            "profile_raw_real_domain_evidence_0_1": profile_evidence,
            "raw_real_domain_evidence_0_1": enriched_evidence,
            # Kept for clients using the initial forensic schema. This field
            # is a profile-distance ratio, not a calibrated probability.
            "real_capture_likelihood_0_1": enriched_evidence,
            "calibrated_real_probability_0_1": None,
            "motion_coherence_0_1": _clamp(motion_coherence),
            "au_relation_consistency_0_1": _clamp(
                _finite(feature_map.get("au_relation_consistency_0_1"), 0.0)
            ),
            "au_dynamics_naturalness_0_1": _clamp(
                _finite(feature_map.get("au_dynamics_naturalness_0_1"), 0.0)
            ),
            "training_free_motion_prior_0_1": training_free_prior,
            "ssl_au_score_0_1": _clamp(
                _finite(feature_map.get("ssl_au_score_0_1"), 0.5)
            ),
            "ssl_backbone_score_0_1": _clamp(
                _finite(feature_map.get("ssl_backbone_score_0_1"), 0.5)
            ),
            "ssl_temporal_consistency_0_1": _clamp(
                _finite(feature_map.get("ssl_temporal_consistency_0_1"), 0.5)
            ),
            "physio_rhythm_score_0_1": _clamp(
                _finite(feature_map.get("physio_rhythm_score_0_1"), 0.5)
            ),
            "pose_normalized_frame_ratio": _clamp(
                _finite(feature_map.get("pose_normalized_frame_ratio"), 0.0)
            ),
            "feature_mode": feature_mode,
            "scored_feature_count": len(names),
            "landmark_valid_frame_ratio": _clamp(
                _finite(
                    feature_map.get("landmark_valid_frame_ratio")
                )
            ),
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
                    "No two-domain profile was supplied; this result is not "
                    "a real-versus-Seedance decision."
                )
            ]
        ),
    }
