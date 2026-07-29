from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np


AU_PROFILE_SCHEMA = "wangxing_au_profile_v2"
AU_CLASSIFIER_SCHEMA = "au_leakage_classifier_v2"
AU_EVALUATOR_VERSION = "wangxing_au_eval_v3"
AU_QUALITY_SCHEMA = "face_quality_gate_v1"
DEFAULT_INTENSITY_AU_IDS = (
    1,
    2,
    4,
    5,
    6,
    9,
    12,
    15,
    17,
    20,
    25,
    26,
)
DEFAULT_PRESENCE_AU_IDS = (
    1,
    2,
    4,
    6,
    7,
    10,
    12,
    14,
    15,
    17,
    23,
    24,
)
LEGACY_AU_IDS = (1, 4, 6, 12, 15, 25)
# Backwards-compatible alias for callers that mean the primary intensity set.
DEFAULT_AU_IDS = DEFAULT_INTENSITY_AU_IDS
DEFAULT_ACTIVE_THRESHOLD = 0.20
DEFAULT_WINDOW_SIZE = 32
DEFAULT_WINDOW_STRIDE = 16
DEFAULT_FACE_QUALITY_THRESHOLD = 0.30
DEFAULT_FACE_VALID_RATIO_THRESHOLD = 0.35
DEFAULT_COACTIVATION_PAIRS = (
    (1, 2),
    (1, 4),
    (1, 6),
    (2, 4),
    (4, 6),
    (4, 7),
    (6, 12),
    (9, 12),
    (12, 15),
    (12, 25),
    (17, 20),
    (20, 26),
)


def _canonical_au_id(value: str) -> int | None:
    match = re.search(
        r"\bau[\s_-]*0*(\d{1,2})(?!\d)",
        str(value),
        re.IGNORECASE,
    )
    if not match:
        return None
    au_id = int(match.group(1))
    return au_id if 1 <= au_id <= 45 else None


def _column_priority(name: str) -> int:
    normalized = str(name).lower().replace(" ", "")
    if _column_kind(name) == "intensity":
        return 0
    if _column_kind(name) == "presence":
        return 0
    return 1


def _column_kind(name: str) -> Literal["intensity", "presence", "unknown"]:
    normalized = str(name).lower().replace(" ", "")
    if "intensity" in normalized or normalized.endswith("_r"):
        return "intensity"
    if (
        normalized.endswith("_c")
        or "presence" in normalized
        or re.fullmatch(r"au[_-]*0*\d{1,2}", normalized)
    ):
        return "presence"
    return "unknown"


def _parse_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _landmark_columns(fieldnames: Iterable[str]) -> tuple[list[str], list[str]]:
    x_columns: list[str] = []
    y_columns: list[str] = []
    for name in fieldnames:
        normalized = str(name).lower()
        if normalized.startswith("lm_mp_") and normalized.endswith("_x"):
            x_columns.append(name)
        elif normalized.startswith("lm_mp_") and normalized.endswith("_y"):
            y_columns.append(name)
    return sorted(x_columns), sorted(y_columns)


def _frame_quality(
    row: dict[str, Any],
    x_columns: list[str],
    y_columns: list[str],
) -> float:
    points = [
        (_parse_float(row.get(x_name)), _parse_float(row.get(y_name)))
        for x_name, y_name in zip(x_columns, y_columns)
    ]
    valid_points = [
        (x, y)
        for x, y in points
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    ]
    if len(valid_points) < 20:
        return 0.0

    xs = np.asarray([point[0] for point in valid_points], dtype=np.float32)
    ys = np.asarray([point[1] for point in valid_points], dtype=np.float32)
    width = float(np.ptp(xs))
    height = float(np.ptp(ys))
    face_area = width * height
    if width < 0.05 or height < 0.05:
        return 0.0

    size_score = max(0.0, min(1.0, face_area / 0.08))
    pose_values = [
        abs(_parse_float(row.get("pitch"))),
        abs(_parse_float(row.get("yaw"))),
    ]
    pose_score = max(
        0.0,
        min(1.0, 1.0 - max(pose_values) / 60.0),
    )
    return float(size_score * pose_score)


def _quality_metadata(
    frame_quality: np.ndarray,
    *,
    available: bool,
) -> dict[str, Any]:
    frame_quality = np.asarray(frame_quality, dtype=np.float32)
    valid = frame_quality >= DEFAULT_FACE_QUALITY_THRESHOLD
    valid_ratio = float(np.mean(valid)) if len(valid) else 0.0
    mean_quality = float(np.mean(frame_quality)) if len(frame_quality) else 0.0
    if not available:
        status = "not_available"
    elif valid_ratio < DEFAULT_FACE_VALID_RATIO_THRESHOLD:
        status = "uncertain"
    elif valid_ratio < 0.60:
        status = "partial"
    else:
        status = "pass"
    return {
        "schema_version": AU_QUALITY_SCHEMA,
        "available": available,
        "status": status,
        "frame_count": int(len(frame_quality)),
        "usable_frame_count": int(np.sum(valid)),
        "valid_frame_ratio": valid_ratio,
        "mean_frame_quality": mean_quality,
        "quality_threshold": DEFAULT_FACE_QUALITY_THRESHOLD,
        "valid_ratio_threshold": DEFAULT_FACE_VALID_RATIO_THRESHOLD,
        "low_quality_frame_indices": [
            int(index)
            for index in np.flatnonzero(~valid)[:40]
        ],
    }


def load_au_table(
    path: str | Path,
    au_ids: Iterable[int] | None = None,
    *,
    feature_type: Literal["intensity", "presence"] = "intensity",
    strict: bool = False,
    intensity_scale: float = 5.0,
) -> tuple[np.ndarray, tuple[int, ...], dict[str, Any]]:
    """Load one AU task while preserving unsupported columns as NaN."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        delimiter = "\t" if "\t" in sample.splitlines()[0] else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        landmark_x_columns, landmark_y_columns = _landmark_columns(fieldnames)
        quality_available = bool(
            landmark_x_columns
            and landmark_y_columns
            and "pitch" in fieldnames
            and "yaw" in fieldnames
        )
        candidates: dict[str, dict[int, list[tuple[int, str]]]] = {
            "intensity": {},
            "presence": {},
        }
        for name in fieldnames:
            au_id = _canonical_au_id(name)
            kind = _column_kind(name)
            if au_id is None or kind == "unknown":
                continue
            candidates[kind].setdefault(au_id, []).append(
                (_column_priority(name), name)
            )

        available_candidates = candidates[feature_type]
        requested = (
            tuple(int(au_id) for au_id in au_ids)
            if au_ids is not None
            else tuple(sorted(available_candidates))
        )
        selected: dict[int, str] = {}
        for au_id in requested:
            choices = sorted(available_candidates.get(au_id, []))
            if choices:
                selected[au_id] = choices[0][1]
        supported = tuple(
            au_id for au_id in requested if au_id in selected
        )
        missing = tuple(
            au_id for au_id in requested if au_id not in selected
        )
        if not supported:
            raise ValueError(
                f"No AU {feature_type} columns found in {path}."
            )
        if strict and missing:
            raise ValueError(
                f"Missing AU {feature_type} columns in {path}: "
                f"{', '.join(map(str, missing))}. "
                f"Supported: {', '.join(map(str, supported))}."
            )

        rows: list[list[float]] = []
        frame_quality: list[float] = []
        for row in reader:
            values = [
                (
                    _parse_float(row.get(selected[au_id]))
                    if au_id in selected
                    else float("nan")
                )
                for au_id in requested
            ]
            rows.append(values)
            frame_quality.append(
                _frame_quality(
                    row,
                    landmark_x_columns,
                    landmark_y_columns,
                )
                if quality_available
                else 1.0
            )

    if not rows:
        raise ValueError(f"AU file contains no rows: {path}")
    matrix = np.asarray(rows, dtype=np.float32)
    finite_values = matrix[np.isfinite(matrix)]
    if (
        feature_type == "intensity"
        and intensity_scale > 1.0
        and len(finite_values)
        and float(np.max(finite_values)) > 1.0 + 1e-6
    ):
        matrix = matrix / float(intensity_scale)
    matrix = np.clip(matrix, 0.0, 1.0)
    frame_quality_array = np.asarray(frame_quality, dtype=np.float32)
    return matrix, requested, {
        "path": str(path),
        "delimiter": delimiter,
        "selected_columns": {
            str(au_id): selected.get(au_id) for au_id in requested
        },
        "feature_type": feature_type,
        "requested_au_ids": list(requested),
        "supported_au_ids": list(supported),
        "missing_au_ids": list(missing),
        "available_au_ids": sorted(available_candidates),
        "quality": _quality_metadata(
            frame_quality_array,
            available=quality_available,
        ),
        "_feature_mask": np.asarray(
            [au_id in selected for au_id in requested],
            dtype=bool,
        ),
        "_frame_quality": frame_quality_array,
    }


def _summary_pairs(au_ids: Iterable[int]) -> list[tuple[int, int]]:
    au_ids = tuple(int(value) for value in au_ids)
    supported = set(au_ids)
    pairs = [
        pair
        for pair in DEFAULT_COACTIVATION_PAIRS
        if pair[0] in supported and pair[1] in supported
    ]
    if len(pairs) < min(12, len(au_ids) - 1):
        for left, right in zip(au_ids, au_ids[1:]):
            pair = (left, right)
            if pair not in pairs:
                pairs.append(pair)
            if len(pairs) >= min(12, len(au_ids) - 1):
                break
    return pairs


def _summary_feature_indices(
    full_au_ids: Iterable[int],
    supported_au_ids: Iterable[int],
) -> list[int]:
    full_au_ids = tuple(int(value) for value in full_au_ids)
    supported = set(int(value) for value in supported_au_ids)
    positions = {
        au_id: index
        for index, au_id in enumerate(full_au_ids)
        if au_id in supported
    }
    indices: list[int] = []
    block_size = len(full_au_ids)
    for block in range(3):
        indices.extend(block * block_size + positions[au_id] for au_id in full_au_ids if au_id in positions)
    pairs = _summary_pairs(full_au_ids)
    for pair_index, (left, right) in enumerate(pairs):
        if left in supported and right in supported:
            indices.append(block_size * 3 + pair_index)
    return indices


def au_summary(
    sequence: np.ndarray,
    *,
    au_ids: Iterable[int] | None = None,
    active_threshold: float = DEFAULT_ACTIVE_THRESHOLD,
) -> np.ndarray:
    """Return robust low-dimensional AU distribution features."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[0] == 0:
        raise ValueError("AU sequence must have shape [frames, aus].")
    au_ids = (
        tuple(int(value) for value in au_ids)
        if au_ids is not None
        else tuple(range(sequence.shape[1]))
    )
    if len(au_ids) != sequence.shape[1]:
        raise ValueError("AU ids do not match the sequence width.")
    active = sequence >= float(active_threshold)
    pairs = _summary_pairs(au_ids)
    cooccurrence = np.asarray(
        [
            float(np.mean(active[:, left] & active[:, right]))
            for left, right in (
                (
                    au_ids.index(left),
                    au_ids.index(right),
                )
                for left, right in pairs
            )
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            np.median(sequence, axis=0),
            np.median(
                np.abs(sequence - np.median(sequence, axis=0)),
                axis=0,
            ),
            active.mean(axis=0),
            cooccurrence,
        ]
    ).astype(np.float32)


def _shrink_covariance(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    dimension = samples.shape[1]
    if samples.shape[0] < 2:
        variance = np.ones(dimension, dtype=np.float64) * 0.05
        return np.diag(variance)
    if samples.shape[0] < 8:
        variance = np.maximum(np.var(samples, axis=0), 0.01)
        return np.diag(variance)
    covariance = np.cov(samples, rowvar=False)
    covariance = np.atleast_2d(covariance)
    diagonal = np.diag(np.diag(covariance))
    shrinkage = min(0.75, 8.0 / float(samples.shape[0]))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    minimum_variance = 0.01 if samples.shape[0] < 8 else 1e-4
    covariance += np.eye(dimension, dtype=np.float64) * minimum_variance
    return covariance


def _mahalanobis(
    samples: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    delta = samples - np.asarray(mean, dtype=np.float64)
    inverse = np.linalg.pinv(np.asarray(covariance, dtype=np.float64))
    squared = np.einsum("ni,ij,nj->n", delta, inverse, delta)
    return np.sqrt(np.maximum(squared, 0.0)).astype(np.float32)


def _json_float_list(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value).reshape(-1)]


def _json_matrix(value: np.ndarray) -> list[list[float]]:
    return [
        [float(item) for item in row]
        for row in np.asarray(value).tolist()
    ]


def _fit_distribution(samples: list[np.ndarray]) -> dict[str, Any]:
    matrix = np.stack(samples).astype(np.float32)
    mean = matrix.mean(axis=0)
    covariance = _shrink_covariance(matrix)
    distances = _mahalanobis(matrix, mean, covariance)
    threshold = max(
        3.0,
        float(np.quantile(distances, 0.95)) * 1.25,
    )
    return {
        "count": int(matrix.shape[0]),
        "mean": _json_float_list(mean),
        "covariance": _json_matrix(covariance),
        "distance_threshold": threshold,
    }


def fit_au_profile(
    labeled_sequences: Iterable[tuple[str, np.ndarray]],
    output_path: str | Path,
    *,
    au_ids: Iterable[int] = DEFAULT_AU_IDS,
    presence_labeled_sequences: Iterable[tuple[str, np.ndarray]] | None = None,
    presence_au_ids: Iterable[int] = DEFAULT_PRESENCE_AU_IDS,
    active_threshold: float = DEFAULT_ACTIVE_THRESHOLD,
) -> dict[str, Any]:
    """Fit target AU distributions from real target videos only."""
    labeled_sequences = list(labeled_sequences)
    au_ids = tuple(int(value) for value in au_ids)
    if (
        au_ids == DEFAULT_AU_IDS
        and labeled_sequences
        and np.asarray(labeled_sequences[0][1]).ndim == 2
        and np.asarray(labeled_sequences[0][1]).shape[1] == len(LEGACY_AU_IDS)
    ):
        au_ids = LEGACY_AU_IDS
    grouped_frames: dict[str, list[np.ndarray]] = {}
    grouped_summaries: dict[str, list[np.ndarray]] = {}
    sample_paths: dict[str, list[str]] = {}
    for expression_class, sequence in labeled_sequences:
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape[1] != len(au_ids):
            raise ValueError(
                f"AU sequence for {expression_class} has invalid shape "
                f"{sequence.shape}; expected [frames, {len(au_ids)}]."
            )
        grouped_frames.setdefault(expression_class, []).extend(sequence)
        grouped_summaries.setdefault(expression_class, []).append(
            au_summary(
                sequence,
                au_ids=au_ids,
                active_threshold=active_threshold,
            )
        )
        sample_paths.setdefault(expression_class, []).append("")

    if not grouped_frames:
        raise ValueError("No AU sequences were provided for profile fitting.")

    classes = sorted(grouped_frames)
    models: dict[str, Any] = {}
    for expression_class in classes:
        frame_model = _fit_distribution(grouped_frames[expression_class])
        summary_model = _fit_distribution(grouped_summaries[expression_class])
        models[expression_class] = {
            "frame": frame_model,
            "summary": summary_model,
            "sample_count": len(sample_paths[expression_class]),
        }

    presence_au_ids = tuple(int(value) for value in presence_au_ids)
    presence_by_class: dict[str, list[np.ndarray]] = {}
    for expression_class, sequence in presence_labeled_sequences or []:
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape[1] != len(presence_au_ids):
            raise ValueError(
                f"Presence sequence for {expression_class} has invalid shape "
                f"{sequence.shape}; expected [frames, {len(presence_au_ids)}]."
            )
        if np.isnan(sequence).all(axis=0).any():
            continue
        presence_by_class.setdefault(expression_class, []).append(sequence)

    presence_models: dict[str, Any] = {}
    for expression_class, sequences in presence_by_class.items():
        matrix = np.concatenate(sequences, axis=0)
        presence_models[expression_class] = {
            "sequence_count": len(sequences),
            "mean_activation": _json_float_list(np.nanmean(matrix, axis=0)),
            "active_ratio": _json_float_list(
                np.nanmean(matrix >= float(active_threshold), axis=0)
            ),
        }

    profile = {
        "schema_version": AU_PROFILE_SCHEMA,
        "au_ids": list(au_ids),
        "feature_type": "intensity",
        "presence_au_ids": list(presence_au_ids),
        "presence_classes": presence_models,
        "summary_layout": {
            "blocks": ["median", "mad", "active_ratio"],
            "coactivation_pairs": [
                list(pair) for pair in _summary_pairs(au_ids)
            ],
        },
        "active_threshold": float(active_threshold),
        "classes": models,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return profile


def _downsample(sequence: np.ndarray, max_points: int = 64) -> np.ndarray:
    if len(sequence) <= max_points:
        return sequence
    indices = np.rint(
        np.linspace(0, len(sequence) - 1, max_points)
    ).astype(np.int64)
    return sequence[indices]


def dtw_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_points: int = 64,
    band_ratio: float = 0.20,
) -> float:
    """Compute constrained, normalized DTW distance for AU trajectories."""
    left = _downsample(np.asarray(left, dtype=np.float32), max_points)
    right = _downsample(np.asarray(right, dtype=np.float32), max_points)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("DTW inputs must be [frames, same_au_count].")
    costs = np.full(
        (len(left) + 1, len(right) + 1),
        np.inf,
        dtype=np.float64,
    )
    costs[0, 0] = 0.0
    band = max(
        abs(len(left) - len(right)),
        int(math.ceil(max(len(left), len(right)) * band_ratio)),
    )
    for i in range(1, len(left) + 1):
        start = max(1, i - band)
        end = min(len(right), i + band)
        for j in range(start, end + 1):
            local = float(np.mean(np.abs(left[i - 1] - right[j - 1])))
            costs[i, j] = local + min(
                costs[i - 1, j],
                costs[i, j - 1],
                costs[i - 1, j - 1],
            )
    return float(costs[-1, -1] / max(len(left), len(right), 1))


def dtw_similarity(left: np.ndarray, right: np.ndarray) -> float:
    distance = dtw_distance(left, right)
    return float(math.exp(-distance / 0.25))


def velocity_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if len(left) < 2 or len(right) < 2:
        return 1.0
    return dtw_similarity(np.diff(left, axis=0), np.diff(right, axis=0))


def _smooth_signal(signal: np.ndarray, window: int = 3) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if len(signal) < 2 or window <= 1:
        return signal
    window = min(int(window), len(signal))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(
        signal,
        (window // 2, window - 1 - window // 2),
        mode="edge",
    )
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _event_summary(
    signal: np.ndarray,
    *,
    active_threshold: float,
) -> dict[str, Any]:
    signal = _smooth_signal(signal)
    frame_count = len(signal)
    if frame_count == 0:
        return {
            "active_ratio": 0.0,
            "event_count": 0,
            "longest_event_ratio": 0.0,
            "mean_event_ratio": 0.0,
            "onset_position": None,
            "peak_position": None,
            "peak_intensity": 0.0,
        }

    active = signal >= float(active_threshold)
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    durations = (
        ends - starts + 1
        if len(starts)
        else np.asarray([], dtype=np.int64)
    )
    peak_index = int(np.argmax(signal))
    return {
        "active_ratio": float(np.mean(active)),
        "event_count": int(len(durations)),
        "longest_event_ratio": (
            float(np.max(durations) / frame_count)
            if len(durations)
            else 0.0
        ),
        "mean_event_ratio": (
            float(np.mean(durations) / frame_count)
            if len(durations)
            else 0.0
        ),
        "onset_position": (
            float(starts[0] / max(frame_count - 1, 1))
            if len(starts)
            else None
        ),
        "peak_position": float(peak_index / max(frame_count - 1, 1)),
        "peak_intensity": float(signal[peak_index]),
    }


def temporal_event_features(
    sequence: np.ndarray,
    *,
    au_ids: Iterable[int] = DEFAULT_AU_IDS,
    active_threshold: float = DEFAULT_ACTIVE_THRESHOLD,
) -> dict[str, Any]:
    """Summarize AU onset, peak, duration and active periods."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[0] == 0:
        raise ValueError("AU sequence must have shape [frames, aus].")
    au_ids = tuple(int(value) for value in au_ids)
    if (
        au_ids == DEFAULT_AU_IDS
        and sequence.shape[1] == len(LEGACY_AU_IDS)
    ):
        au_ids = LEGACY_AU_IDS
    if sequence.shape[1] != len(au_ids):
        raise ValueError("AU ids do not match the sequence width.")
    per_au = {
        str(au_id): _event_summary(
            sequence[:, index],
            active_threshold=active_threshold,
        )
        for index, au_id in enumerate(au_ids)
    }
    return {
        "frame_count": int(sequence.shape[0]),
        "active_threshold": float(active_threshold),
        "aggregate": _event_summary(
            sequence.mean(axis=1),
            active_threshold=active_threshold,
        ),
        "per_au": per_au,
    }


def _event_similarity(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, float]:
    def difference(name: str, default: float = 0.0) -> float:
        left_value = left.get(name)
        right_value = right.get(name)
        if left_value is None and right_value is None:
            return 0.0
        if left_value is None or right_value is None:
            return default
        return abs(float(left_value) - float(right_value))

    active_similarity = 1.0 - difference("active_ratio")
    duration_similarity = 1.0 - difference("longest_event_ratio")
    onset_similarity = 1.0 - difference("onset_position", 0.5)
    peak_similarity = 1.0 - difference("peak_position")
    return {
        "active_ratio_similarity_0_1": max(0.0, min(1.0, active_similarity)),
        "duration_similarity_0_1": max(0.0, min(1.0, duration_similarity)),
        "onset_similarity_0_1": max(0.0, min(1.0, onset_similarity)),
        "peak_position_similarity_0_1": max(0.0, min(1.0, peak_similarity)),
    }


def compare_temporal_events(
    generated: dict[str, Any],
    driver: dict[str, Any],
) -> dict[str, Any]:
    """Compare event timing without reducing the whole clip to one DTW score."""
    aggregate = _event_similarity(
        generated.get("aggregate", {}),
        driver.get("aggregate", {}),
    )
    generated_per_au = generated.get("per_au", {})
    driver_per_au = driver.get("per_au", {})
    per_au: dict[str, Any] = {}
    scores: list[float] = []
    for au_id in sorted(set(generated_per_au) & set(driver_per_au)):
        similarity = _event_similarity(
            generated_per_au[au_id],
            driver_per_au[au_id],
        )
        similarity["event_count_difference"] = abs(
            int(generated_per_au[au_id].get("event_count", 0))
            - int(driver_per_au[au_id].get("event_count", 0))
        )
        per_au[au_id] = similarity
        scores.append(
            float(
                np.mean(
                    [
                        similarity["active_ratio_similarity_0_1"],
                        similarity["duration_similarity_0_1"],
                        similarity["onset_similarity_0_1"],
                        similarity["peak_position_similarity_0_1"],
                    ]
                )
            )
        )
    aggregate_score = float(np.mean(list(aggregate.values())))
    overall_score = float(np.mean([aggregate_score, *scores]))
    return {
        "event_alignment_score_0_1": max(0.0, min(1.0, overall_score)),
        "aggregate": aggregate,
        "per_au": per_au,
    }


def _load_profile(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        profile = json.load(handle)
    if profile.get("schema_version") != AU_PROFILE_SCHEMA:
        raise ValueError("Unsupported AU profile schema.")
    return profile


def _profile_model_score(
    sequence: np.ndarray,
    model: dict[str, Any],
    *,
    full_au_ids: tuple[int, ...],
    supported_au_ids: tuple[int, ...],
) -> dict[str, Any]:
    frame_model = model["frame"]
    summary_model = model["summary"]
    frame_indices = [
        full_au_ids.index(au_id)
        for au_id in supported_au_ids
    ]
    frame_mean = np.asarray(frame_model["mean"], dtype=np.float32)[
        frame_indices
    ]
    frame_covariance = np.asarray(
        frame_model["covariance"],
        dtype=np.float64,
    )[
        np.ix_(frame_indices, frame_indices)
    ]
    frame_distances = _mahalanobis(
        sequence,
        frame_mean,
        frame_covariance,
    )
    summary = au_summary(sequence, au_ids=supported_au_ids)
    summary_indices = _summary_feature_indices(
        full_au_ids,
        supported_au_ids,
    )
    summary_distance = float(
        _mahalanobis(
            summary[None, :],
            np.asarray(summary_model["mean"], dtype=np.float32)[
                summary_indices
            ],
            np.asarray(summary_model["covariance"], dtype=np.float64)[
                np.ix_(summary_indices, summary_indices)
            ],
        )[0]
    )
    frame_threshold = float(frame_model["distance_threshold"])
    summary_threshold = float(summary_model["distance_threshold"])
    frame_anomaly_ratio = float(
        np.mean(frame_distances > frame_threshold)
    )
    summary_anomaly = float(summary_distance > summary_threshold)
    personal_score = float(
        math.exp(-summary_distance / max(summary_threshold, 1e-6))
        * (1.0 - 0.5 * frame_anomaly_ratio)
    )
    return {
        "personal_au_score_0_1": max(0.0, min(1.0, personal_score)),
        "frame_anomaly_ratio": frame_anomaly_ratio,
        "summary_distance": summary_distance,
        "summary_threshold": summary_threshold,
        "summary_anomaly": bool(summary_anomaly),
        "max_frame_distance": float(np.max(frame_distances)),
        "anomalous_frame_indices": [
            int(index)
            for index in np.flatnonzero(frame_distances > frame_threshold)
        ],
    }


def _public_au_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    metadata.pop("_frame_quality", None)
    metadata.pop("_feature_mask", None)
    return metadata


def _quality_filtered_sequence(
    sequence: np.ndarray,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    frame_quality = np.asarray(
        metadata.pop("_frame_quality", np.ones(len(sequence))),
        dtype=np.float32,
    )
    quality = metadata.get("quality", {})
    usable = frame_quality >= DEFAULT_FACE_QUALITY_THRESHOLD
    if (
        bool(quality.get("available"))
        and int(np.sum(usable)) >= 3
    ):
        return sequence[usable], frame_quality
    return sequence, frame_quality


def _presence_report(
    sequence: np.ndarray,
    au_ids: tuple[int, ...],
    metadata: dict[str, Any],
    profile: dict[str, Any],
    selected_class: str,
    *,
    active_threshold: float,
) -> dict[str, Any]:
    supported = tuple(
        int(value) for value in metadata.get("supported_au_ids", [])
    )
    activation_ratio: dict[str, float | None] = {}
    for index, au_id in enumerate(au_ids):
        values = sequence[:, index]
        finite = values[np.isfinite(values)]
        activation_ratio[str(au_id)] = (
            float(np.mean(finite >= active_threshold))
            if len(finite)
            else None
        )

    fit_score: float | None = None
    target = (
        profile.get("presence_classes", {})
        .get(selected_class, {})
        .get("mean_activation")
    )
    profile_ids = tuple(
        int(value) for value in profile.get(
            "presence_au_ids",
            DEFAULT_PRESENCE_AU_IDS,
        )
    )
    if target:
        common = tuple(
            au_id
            for au_id in profile_ids
            if au_id in supported
        )
        if common:
            generated_values = np.asarray(
                [
                    activation_ratio[str(au_id)]
                    for au_id in common
                ],
                dtype=np.float32,
            )
            target_values = np.asarray(
                [
                    target[profile_ids.index(au_id)]
                    for au_id in common
                ],
                dtype=np.float32,
            )
            fit_score = float(
                max(0.0, min(1.0, 1.0 - np.mean(
                    np.abs(generated_values - target_values)
                )))
            )

    return {
        "feature_type": "presence",
        "supported_au_ids": list(supported),
        "missing_au_ids": list(
            au_id for au_id in au_ids if au_id not in supported
        ),
        "activation_ratio": activation_ratio,
        "fit_score_0_1": fit_score,
        "quality": metadata.get("quality"),
    }


def _au_time_curve(
    sequence: np.ndarray,
    au_ids: tuple[int, ...],
    *,
    max_points: int = 96,
) -> list[dict[str, Any]]:
    sequence = np.asarray(sequence, dtype=np.float32)
    if len(sequence) <= max_points:
        indices = np.arange(len(sequence), dtype=np.int64)
    else:
        indices = np.rint(
            np.linspace(0, len(sequence) - 1, max_points)
        ).astype(np.int64)
    return [
        {
            "frame_index": int(index),
            "position": float(index / max(len(sequence) - 1, 1)),
            "values": {
                str(au_id): float(sequence[index, column])
                for column, au_id in enumerate(au_ids)
            },
        }
        for index in indices
    ]


def _legacy_score_au_compliance(
    profile_path: str | Path,
    generated_au_path: str | Path,
    *,
    expected_class: str | None = None,
    driver_au_path: str | Path | None = None,
    leakage_classifier_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score target-specific AU compliance and driver expression fidelity."""
    profile = _load_profile(profile_path)
    au_ids = tuple(int(value) for value in profile["au_ids"])
    generated, generated_ids, generated_meta = load_au_table(
        generated_au_path,
        au_ids,
    )
    if generated_ids != au_ids:
        raise ValueError("Generated AU columns do not match the profile.")

    generated_frame_quality = np.asarray(
        generated_meta.pop("_frame_quality", np.ones(len(generated))),
        dtype=np.float32,
    )
    generated_quality = generated_meta.get("quality", {})
    generated_usable = (
        generated_frame_quality >= DEFAULT_FACE_QUALITY_THRESHOLD
    )
    generated_scored = (
        generated[generated_usable]
        if bool(generated_quality.get("available"))
        and int(np.sum(generated_usable)) >= 3
        else generated
    )
    generated_temporal = temporal_event_features(
        generated_scored,
        au_ids=au_ids,
        active_threshold=float(
            profile.get("active_threshold", DEFAULT_ACTIVE_THRESHOLD)
        ),
    )

    classes = profile["classes"]
    class_scores = {
        class_name: _profile_model_score(generated_scored, model)
        for class_name, model in classes.items()
    }
    if expected_class:
        if expected_class not in class_scores:
            raise ValueError(
                f"Unknown expected expression class: {expected_class}"
            )
        selected_class = expected_class
    else:
        selected_class = max(
            class_scores,
            key=lambda name: class_scores[name]["personal_au_score_0_1"],
        )
    selected = class_scores[selected_class]

    driver_expression_score: float | None = None
    driver_temporal_alignment_score: float | None = None
    driver_temporal_alignment: dict[str, Any] | None = None
    driver_meta: dict[str, Any] | None = None
    driver_similarity_proxy: float | None = None
    if driver_au_path:
        driver, driver_ids, driver_meta = load_au_table(
            driver_au_path,
            au_ids,
        )
        if driver_ids != au_ids:
            raise ValueError("Driver AU columns do not match the profile.")
        driver_frame_quality = np.asarray(
            driver_meta.pop("_frame_quality", np.ones(len(driver))),
            dtype=np.float32,
        )
        driver_quality = driver_meta.get("quality", {})
        driver_usable = driver_frame_quality >= DEFAULT_FACE_QUALITY_THRESHOLD
        driver_scored = (
            driver[driver_usable]
            if bool(driver_quality.get("available"))
            and int(np.sum(driver_usable)) >= 3
            else driver
        )
        driver_expression_score = dtw_similarity(
            generated_scored,
            driver_scored,
        )
        driver_temporal = temporal_event_features(
            driver_scored,
            au_ids=au_ids,
            active_threshold=float(
                profile.get("active_threshold", DEFAULT_ACTIVE_THRESHOLD)
            ),
        )
        driver_temporal_alignment = compare_temporal_events(
            generated_temporal,
            driver_temporal,
        )
        driver_temporal_alignment_score = float(
            driver_temporal_alignment["event_alignment_score_0_1"]
        )
        generated_summary = au_summary(generated_scored)
        driver_summary = au_summary(driver_scored)
        denominator = (
            float(np.linalg.norm(generated_summary))
            * float(np.linalg.norm(driver_summary))
        )
        if denominator > 1e-8:
            driver_similarity_proxy = float(
                max(
                    0.0,
                    min(
                        1.0,
                        (
                            float(
                                np.dot(
                                    generated_summary,
                                    driver_summary,
                                )
                            )
                            / denominator
                            + 1.0
                        )
                        / 2.0,
                    )
                )
            )

    classifier_risk: float | None = None
    if leakage_classifier_path:
        classifier = json.loads(
            Path(leakage_classifier_path).read_text(
                encoding="utf-8-sig"
            )
        )
        classifier_risk = score_leakage_classifier(
            classifier,
            au_summary(generated_scored),
        )

    personal_score = float(selected["personal_au_score_0_1"])
    if classifier_risk is not None:
        leakage_risk = classifier_risk
        leakage_backend = "trained_au_leakage_classifier"
    elif driver_similarity_proxy is not None:
        leakage_risk = max(
            0.0,
            min(
                1.0,
                driver_similarity_proxy * (1.0 - personal_score),
            ),
        )
        leakage_backend = "driver_style_overlap_proxy"
    else:
        leakage_risk = float(
            max(
                selected["frame_anomaly_ratio"],
                1.0 - personal_score,
            )
        )
        leakage_backend = "target_au_anomaly_proxy"

    quality_status = str(generated_quality.get("status", "not_available"))
    evidence_quality_status = (
        "uncertain"
        if quality_status == "uncertain"
        else "available"
    )
    quality_confidence = (
        float(
            max(
                0.0,
                min(
                    1.0,
                    0.5 * float(generated_quality.get("valid_frame_ratio", 1.0))
                    + 0.5 * float(
                        generated_quality.get("mean_frame_quality", 1.0)
                    ),
                ),
            )
        )
        if generated_quality.get("available")
        else None
    )

    return {
        "status": "available",
        "backend": "au_personal_profile",
        "evaluator_version": AU_EVALUATOR_VERSION,
        "profile_schema_version": profile.get("schema_version"),
        "au_ids": list(au_ids),
        "selected_expression_class": selected_class,
        "expected_expression_class": expected_class,
        "class_scores": class_scores,
        "personal_au_score_0_1": personal_score,
        "driver_expression_score_0_1": driver_expression_score,
        "driver_temporal_alignment_score_0_1": (
            driver_temporal_alignment_score
        ),
        "driver_similarity_proxy_0_1": driver_similarity_proxy,
        "driver_identity_leakage_risk_0_1": leakage_risk,
        "leakage_backend": leakage_backend,
        "evidence_quality_status": evidence_quality_status,
        "evidence_confidence_0_1": quality_confidence,
        "uncertainty_reasons": (
            ["face_quality_low"]
            if evidence_quality_status == "uncertain"
            else []
        ),
        "quality": {
            "generated": generated_quality,
            "driver": (
                driver_meta.get("quality")
                if driver_meta is not None
                else None
            ),
        },
        "temporal_events": generated_temporal,
        "driver_temporal_alignment": driver_temporal_alignment,
        "generated_au": generated_meta,
        "driver_au": driver_meta,
    }


def score_au_compliance(
    profile_path: str | Path,
    generated_au_path: str | Path,
    *,
    expected_class: str | None = None,
    driver_au_path: str | Path | None = None,
    leakage_classifier_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score intensity AU compliance with presence and timing evidence."""
    profile = _load_profile(profile_path)
    au_ids = tuple(int(value) for value in profile["au_ids"])
    generated, generated_ids, generated_meta = load_au_table(
        generated_au_path,
        au_ids,
        feature_type="intensity",
        strict=False,
    )
    if generated_ids != au_ids:
        raise ValueError("Generated AU columns do not match the profile.")

    generated_supported = tuple(
        int(value) for value in generated_meta["supported_au_ids"]
    )
    if not generated_supported:
        raise ValueError(
            "Generated AU file does not contain any supported intensity AU."
        )
    generated_indices = [
        au_ids.index(au_id) for au_id in generated_supported
    ]
    generated_sequence = generated[:, generated_indices]
    generated_scored, _ = _quality_filtered_sequence(
        generated_sequence,
        generated_meta,
    )
    generated_quality = generated_meta.get("quality", {})
    active_threshold = float(
        profile.get("active_threshold", DEFAULT_ACTIVE_THRESHOLD)
    )
    generated_temporal = temporal_event_features(
        generated_scored,
        au_ids=generated_supported,
        active_threshold=active_threshold,
    )

    classes = profile["classes"]
    class_scores = {
        class_name: _profile_model_score(
            generated_scored,
            model,
            full_au_ids=au_ids,
            supported_au_ids=generated_supported,
        )
        for class_name, model in classes.items()
    }
    if expected_class:
        if expected_class not in class_scores:
            raise ValueError(
                f"Unknown expected expression class: {expected_class}"
            )
        selected_class = expected_class
    else:
        selected_class = max(
            class_scores,
            key=lambda name: class_scores[name]["personal_au_score_0_1"],
        )
    selected = class_scores[selected_class]

    presence_au_ids = tuple(
        int(value) for value in profile.get(
            "presence_au_ids",
            DEFAULT_PRESENCE_AU_IDS,
        )
    )
    try:
        generated_presence, _, generated_presence_meta = load_au_table(
            generated_au_path,
            presence_au_ids,
            feature_type="presence",
            strict=False,
            intensity_scale=1.0,
        )
        generated_presence_report = _presence_report(
            generated_presence,
            presence_au_ids,
            generated_presence_meta,
            profile,
            selected_class,
            active_threshold=active_threshold,
        )
    except ValueError:
        generated_presence_report = {
            "feature_type": "presence",
            "supported_au_ids": [],
            "missing_au_ids": list(presence_au_ids),
            "activation_ratio": {},
            "fit_score_0_1": None,
            "quality": None,
            "status": "unavailable",
        }

    driver_expression_score: float | None = None
    driver_dtw_score: float | None = None
    driver_velocity_score: float | None = None
    driver_temporal_alignment_score: float | None = None
    driver_temporal_alignment: dict[str, Any] | None = None
    driver_meta: dict[str, Any] | None = None
    driver_similarity_proxy: float | None = None
    driver_sequence: np.ndarray | None = None
    driver_scored: np.ndarray | None = None
    driver_supported: tuple[int, ...] = ()
    if driver_au_path:
        driver, driver_ids, driver_meta = load_au_table(
            driver_au_path,
            au_ids,
            feature_type="intensity",
            strict=False,
        )
        if driver_ids != au_ids:
            raise ValueError("Driver AU columns do not match the profile.")
        driver_supported = tuple(
            int(value) for value in driver_meta["supported_au_ids"]
        )
        common_driver_au_ids = tuple(
            au_id
            for au_id in generated_supported
            if au_id in driver_supported
        )
        if common_driver_au_ids:
            driver_indices = [
                au_ids.index(au_id) for au_id in common_driver_au_ids
            ]
            driver_sequence = driver[:, driver_indices]
            driver_scored, _ = _quality_filtered_sequence(
                driver_sequence,
                driver_meta,
            )
            driver_dtw_score = dtw_similarity(
                generated_scored,
                driver_scored,
            )
            driver_velocity_score = velocity_similarity(
                generated_scored,
                driver_scored,
            )
            driver_expression_score = float(
                np.mean([driver_dtw_score, driver_velocity_score])
            )
            driver_temporal = temporal_event_features(
                driver_scored,
                au_ids=common_driver_au_ids,
                active_threshold=active_threshold,
            )
            driver_temporal_alignment = compare_temporal_events(
                generated_temporal,
                driver_temporal,
            )
            driver_temporal_alignment_score = float(
                driver_temporal_alignment["event_alignment_score_0_1"]
            )
            generated_summary = au_summary(
                generated_scored,
                au_ids=common_driver_au_ids,
            )
            driver_summary = au_summary(
                driver_scored,
                au_ids=common_driver_au_ids,
            )
            denominator = (
                float(np.linalg.norm(generated_summary))
                * float(np.linalg.norm(driver_summary))
            )
            if denominator > 1e-8:
                driver_similarity_proxy = float(
                    max(
                        0.0,
                        min(
                            1.0,
                            (
                                float(
                                    np.dot(
                                        generated_summary,
                                        driver_summary,
                                    )
                                )
                                / denominator
                                + 1.0
                            )
                            / 2.0,
                        ),
                    )
                )

    classifier_risk: float | None = None
    if leakage_classifier_path:
        classifier = json.loads(
            Path(leakage_classifier_path).read_text(
                encoding="utf-8-sig"
            )
        )
        classifier_au_ids = tuple(
            int(value) for value in classifier.get("au_ids", au_ids)
        )
        common_classifier_au_ids = tuple(
            au_id
            for au_id in generated_supported
            if au_id in classifier_au_ids
        )
        if common_classifier_au_ids:
            classifier_indices = [
                generated_supported.index(au_id)
                for au_id in common_classifier_au_ids
            ]
            classifier_sequence = generated_scored[:, classifier_indices]
            classifier_risk = score_leakage_classifier(
                classifier,
                au_summary(
                    classifier_sequence,
                    au_ids=common_classifier_au_ids,
                ),
                feature_indices=_summary_feature_indices(
                    classifier_au_ids,
                    common_classifier_au_ids,
                ),
            )

    personal_score = float(selected["personal_au_score_0_1"])
    if classifier_risk is not None:
        leakage_risk = classifier_risk
        leakage_backend = "trained_au_leakage_classifier"
    elif driver_similarity_proxy is not None:
        leakage_risk = max(
            0.0,
            min(
                1.0,
                driver_similarity_proxy * (1.0 - personal_score),
            ),
        )
        leakage_backend = "driver_style_overlap_proxy"
    else:
        leakage_risk = float(
            max(
                selected["frame_anomaly_ratio"],
                1.0 - personal_score,
            )
        )
        leakage_backend = "target_au_anomaly_proxy"

    quality_status = str(generated_quality.get("status", "not_available"))
    missing_intensity_au_ids = [
        au_id for au_id in au_ids if au_id not in generated_supported
    ]
    uncertainty_reasons: list[str] = []
    if quality_status == "uncertain":
        uncertainty_reasons.append("face_quality_low")
    if missing_intensity_au_ids:
        uncertainty_reasons.append("missing_intensity_au")
    evidence_quality_status = (
        "uncertain" if uncertainty_reasons else "available"
    )
    support_ratio = len(generated_supported) / max(len(au_ids), 1)
    base_confidence = (
        0.5 * float(generated_quality.get("valid_frame_ratio", 1.0))
        + 0.5 * float(generated_quality.get("mean_frame_quality", 1.0))
        if generated_quality.get("available")
        else 1.0
    )
    quality_confidence = max(
        0.0,
        min(1.0, base_confidence * support_ratio),
    )

    driver_curve = (
        _au_time_curve(
            driver_sequence,
            tuple(
                au_id
                for au_id in generated_supported
                if au_id in driver_supported
            ),
        )
        if driver_sequence is not None
        else None
    )
    _public_au_metadata(generated_meta)
    if driver_meta is not None:
        _public_au_metadata(driver_meta)
    return {
        "status": "available",
        "backend": "au_personal_profile",
        "evaluator_version": AU_EVALUATOR_VERSION,
        "profile_schema_version": profile.get("schema_version"),
        "feature_type": "intensity",
        "au_ids": list(au_ids),
        "supported_au_ids": list(generated_supported),
        "missing_au_ids": missing_intensity_au_ids,
        "presence_au_ids": list(presence_au_ids),
        "selected_expression_class": selected_class,
        "expected_expression_class": expected_class,
        "class_scores": class_scores,
        "personal_au_score_0_1": personal_score,
        "driver_expression_score_0_1": driver_expression_score,
        "driver_dtw_score_0_1": driver_dtw_score,
        "driver_velocity_score_0_1": driver_velocity_score,
        "driver_temporal_alignment_score_0_1": (
            driver_temporal_alignment_score
        ),
        "driver_similarity_proxy_0_1": driver_similarity_proxy,
        "driver_identity_leakage_risk_0_1": leakage_risk,
        "leakage_backend": leakage_backend,
        "evidence_quality_status": evidence_quality_status,
        "evidence_confidence_0_1": quality_confidence,
        "uncertainty_reasons": uncertainty_reasons,
        "generated_presence": generated_presence_report,
        "quality": {
            "generated": generated_quality,
            "driver": (
                driver_meta.get("quality")
                if driver_meta is not None
                else None
            ),
        },
        "temporal_events": generated_temporal,
        "driver_temporal_alignment": driver_temporal_alignment,
        "time_curve": {
            "au_ids": list(generated_supported),
            "generated": _au_time_curve(
                generated_sequence,
                generated_supported,
            ),
            "driver": driver_curve,
        },
        "generated_au": generated_meta,
        "driver_au": driver_meta,
    }


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def fit_leakage_classifier(
    positive_au_paths: Iterable[str | Path],
    negative_au_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    au_ids: Iterable[int] = DEFAULT_AU_IDS,
) -> dict[str, Any]:
    """Fit a small logistic classifier; label 0=target, 1=leakage/anomaly."""
    au_ids = tuple(int(value) for value in au_ids)
    paths = [
        (Path(path), 0)
        for path in positive_au_paths
    ] + [
        (Path(path), 1)
        for path in negative_au_paths
    ]
    if not paths or not any(label == 0 for _, label in paths):
        raise ValueError("At least one positive target AU file is required.")
    if not any(label == 1 for _, label in paths):
        raise ValueError(
            "Negative AU files are required to train leakage classifier. "
            "Use target-only profile scoring when they are unavailable."
        )

    features: list[np.ndarray] = []
    labels: list[int] = []
    for path, label in paths:
        sequence, _, _ = load_au_table(
            path,
            au_ids,
            feature_type="intensity",
            strict=True,
        )
        features.append(au_summary(sequence, au_ids=au_ids))
        labels.append(label)
    matrix = np.stack(features).astype(np.float64)
    target = np.asarray(labels, dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), 1e-4)
    normalized = (matrix - mean) / scale

    weights = np.zeros(matrix.shape[1], dtype=np.float64)
    intercept = 0.0
    regularization = 0.01
    learning_rate = 0.1
    for _ in range(1200):
        logits = normalized @ weights + intercept
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
        error = probabilities - target
        weights -= learning_rate * (
            (normalized.T @ error) / len(matrix)
            + regularization * weights
        )
        intercept -= learning_rate * float(np.mean(error))

    model = {
        "schema_version": AU_CLASSIFIER_SCHEMA,
        "au_ids": list(au_ids),
        "feature_type": "intensity",
        "summary_layout": {
            "blocks": ["median", "mad", "active_ratio"],
            "coactivation_pairs": [
                list(pair) for pair in _summary_pairs(au_ids)
            ],
        },
        "feature_mean": _json_float_list(mean),
        "feature_scale": _json_float_list(scale),
        "weights": _json_float_list(weights),
        "intercept": float(intercept),
        "positive_count": int(np.sum(target == 0)),
        "negative_count": int(np.sum(target == 1)),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(model, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return model


def score_leakage_classifier(
    classifier: dict[str, Any],
    summary: np.ndarray,
    *,
    feature_indices: list[int] | None = None,
) -> float:
    if classifier.get("schema_version") != AU_CLASSIFIER_SCHEMA:
        raise ValueError("Unsupported AU leakage classifier schema.")
    mean = np.asarray(classifier["feature_mean"], dtype=np.float64)
    scale = np.asarray(classifier["feature_scale"], dtype=np.float64)
    weights = np.asarray(classifier["weights"], dtype=np.float64)
    if feature_indices is not None:
        mean = mean[feature_indices]
        scale = scale[feature_indices]
        weights = weights[feature_indices]
    normalized = (np.asarray(summary, dtype=np.float64) - mean) / scale
    return _sigmoid(float(normalized @ weights) + classifier["intercept"])


def fuse_compliance_scores(
    *,
    identity_score_0_1: float | None,
    personal_au_score_0_1: float | None,
    driver_expression_score_0_1: float | None,
    leakage_risk_0_1: float | None,
    identity_threshold: float = 0.75,
    personal_au_threshold: float = 0.50,
    driver_expression_threshold: float = 0.50,
    leakage_threshold: float = 0.50,
) -> dict[str, Any]:
    """Fuse scores without hiding unavailable evidence."""
    components = [
        (0.40, identity_score_0_1),
        (0.40, personal_au_score_0_1),
        (0.20, driver_expression_score_0_1),
    ]
    valid = [
        (weight, float(score))
        for weight, score in components
        if score is not None and math.isfinite(float(score))
    ]
    weight_sum = sum(weight for weight, _ in valid)
    likeness = (
        sum(weight * score for weight, score in valid) / weight_sum
        if weight_sum
        else None
    )
    reasons: list[str] = []
    if (
        identity_score_0_1 is not None
        and identity_score_0_1 < identity_threshold
    ):
        reasons.append("identity_below_threshold")
    if (
        personal_au_score_0_1 is not None
        and personal_au_score_0_1 < personal_au_threshold
    ):
        reasons.append("personal_au_below_threshold")
    if (
        driver_expression_score_0_1 is not None
        and driver_expression_score_0_1 < driver_expression_threshold
    ):
        reasons.append("driver_expression_below_threshold")
    if (
        leakage_risk_0_1 is not None
        and leakage_risk_0_1 >= leakage_threshold
    ):
        reasons.append("driver_identity_leakage")
    missing_evidence = [
        name
        for name, value in (
            ("identity", identity_score_0_1),
            ("personal_au", personal_au_score_0_1),
            ("driver_expression", driver_expression_score_0_1),
            ("leakage", leakage_risk_0_1),
        )
        if value is None
    ]
    if reasons:
        decision = "block"
    elif missing_evidence:
        decision = "review"
    else:
        decision = "allow"
    return {
        "person_likeness_score_0_1": likeness,
        "score_weight_coverage": weight_sum,
        "leakage_risk_0_1": leakage_risk_0_1,
        "decision": decision,
        "decision_reasons": reasons,
        "missing_evidence": missing_evidence,
        "thresholds": {
            "identity": identity_threshold,
            "personal_au": personal_au_threshold,
            "driver_expression": driver_expression_threshold,
            "leakage": leakage_threshold,
        },
    }


def fuse_wangxing_targeted_scores(
    *,
    personal_au_score_0_1: float | None,
    driver_expression_score_0_1: float | None,
    leakage_risk_0_1: float | None,
    temporal_alignment_score_0_1: float | None = None,
    evidence_quality_status: str = "available",
    evidence_confidence_0_1: float | None = None,
    uncertainty_reasons: Iterable[str] = (),
    personal_au_threshold: float = 0.50,
    driver_expression_threshold: float = 0.50,
    leakage_threshold: float = 0.50,
) -> dict[str, Any]:
    """Judge Wang Xing-specific expression fit without requiring identity evidence."""
    driver_scores = [
        float(score)
        for score in (
            driver_expression_score_0_1,
            temporal_alignment_score_0_1,
        )
        if score is not None and math.isfinite(float(score))
    ]
    driver_fit = (
        sum(driver_scores) / len(driver_scores)
        if driver_scores
        else None
    )
    expression_scores = [
        float(score)
        for score in (personal_au_score_0_1, driver_fit)
        if score is not None and math.isfinite(float(score))
    ]
    expression_fit = (
        sum(expression_scores) / len(expression_scores)
        if expression_scores
        else None
    )
    reasons: list[str] = []
    if (
        personal_au_score_0_1 is not None
        and personal_au_score_0_1 < personal_au_threshold
    ):
        reasons.append("wangxing_au_below_threshold")
    if (
        driver_expression_score_0_1 is not None
        and driver_expression_score_0_1 < driver_expression_threshold
    ):
        reasons.append("driver_expression_below_threshold")
    if (
        temporal_alignment_score_0_1 is not None
        and temporal_alignment_score_0_1 < driver_expression_threshold
    ):
        reasons.append("temporal_alignment_below_threshold")
    if (
        leakage_risk_0_1 is not None
        and leakage_risk_0_1 >= leakage_threshold
    ):
        reasons.append("identity_leakage_risk")
    for reason in uncertainty_reasons:
        if reason not in reasons:
            reasons.append(str(reason))
    if evidence_quality_status == "uncertain":
        if "evidence_quality_low" not in reasons:
            reasons.append("evidence_quality_low")

    if leakage_risk_0_1 is not None and leakage_risk_0_1 >= leakage_threshold:
        decision = "block"
    elif personal_au_score_0_1 is None:
        decision = "review"
    elif evidence_quality_status == "uncertain":
        decision = "review"
    elif reasons:
        decision = "review"
    else:
        decision = "allow"

    return {
        "wangxing_expression_fit_score_0_1": expression_fit,
        "decision": decision,
        "decision_reasons": reasons,
        "evidence": {
            "personal_au": personal_au_score_0_1,
            "driver_expression": driver_expression_score_0_1,
            "temporal_alignment": temporal_alignment_score_0_1,
            "driver_fit": driver_fit,
            "leakage_risk": leakage_risk_0_1,
        },
        "evidence_quality_status": evidence_quality_status,
        "evidence_confidence_0_1": evidence_confidence_0_1,
        "thresholds": {
            "personal_au": personal_au_threshold,
            "driver_expression": driver_expression_threshold,
            "leakage": leakage_threshold,
        },
    }
