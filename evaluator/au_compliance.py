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
AU_EVALUATOR_VERSION = "wangxing_au_eval_v6"
AU_QUALITY_SCHEMA = "face_quality_gate_v1"
AUTO_EMOTION_MIN_CLASSES = 2
AUTO_EMOTION_MIN_SAMPLES_PER_CLASS = 3
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
INTENSITY_PERSONAL_SCORE_WEIGHT = 0.55
PRESENCE_PERSONAL_SCORE_WEIGHT = 0.45
AUTO_NEUTRAL_INTENSITY_THRESHOLD = 0.35
FACE_MESH_MOUTH_OPEN_THRESHOLD = 0.08
FACE_MESH_MOUTH_CHANGE_THRESHOLD = 0.015
FACE_MESH_BROW_CHANGE_THRESHOLD = 0.012
FACE_MESH_EYE_CHANGE_THRESHOLD = 0.012
FACE_MESH_GLOBAL_MOTION_THRESHOLD = 0.008
FACE_MESH_SALIENT_DURATION_RATIO = 0.05
COMPLIANCE_COMPONENT_WEIGHTS = {
    "identity": 0.40,
    "personal_au": 0.40,
    "driver_expression": 0.20,
}
WANGXING_TARGETED_COMPONENT_WEIGHTS = {
    "personal_au": 0.40,
    "driver_expression": 0.35,
    "temporal_alignment": 0.25,
}
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


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_int(value: Any, default: int) -> int:
    parsed = _optional_float(value)
    if parsed is None:
        return int(default)
    return int(round(parsed))


def _field_name(
    fieldnames: Iterable[str],
    *candidates: str,
) -> str | None:
    wanted = {candidate.lower() for candidate in candidates}
    for name in fieldnames:
        if str(name).lower() in wanted:
            return name
    return None


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


def _landmark_pairs(
    fieldnames: Iterable[str],
) -> list[tuple[int, str, str]]:
    pairs: list[tuple[int, str, str]] = []
    names = list(fieldnames)
    for name in names:
        match = re.fullmatch(r"lm_mp_(\d+)_x", str(name), re.IGNORECASE)
        if not match:
            continue
        index = int(match.group(1))
        y_name = _field_name(names, f"lm_mp_{index}_y")
        if y_name is not None:
            pairs.append((index, str(name), y_name))
    return sorted(pairs)


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
        frame_index_field = _field_name(
            fieldnames,
            "frame_idx",
            "frame_index",
            "frame",
        )
        frame_time_field = _field_name(
            fieldnames,
            "frame_time_in_ms",
            "timestamp_ms",
            "frame_time_ms",
        )
        landmark_x_columns, landmark_y_columns = _landmark_columns(fieldnames)
        landmark_pairs = _landmark_pairs(fieldnames)
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
        landmark_rows: list[list[list[float]]] = []
        frame_quality: list[float] = []
        frame_indices: list[int] = []
        frame_times_seconds: list[float | None] = []
        for row_index, row in enumerate(reader):
            values = [
                (
                    _parse_float(row.get(selected[au_id]))
                    if au_id in selected
                    else float("nan")
                )
                for au_id in requested
            ]
            rows.append(values)
            if landmark_pairs:
                landmark_rows.append(
                    [
                        [
                            _parse_float(row.get(x_name)),
                            _parse_float(row.get(y_name)),
                        ]
                        for _, x_name, y_name in landmark_pairs
                    ]
                )
            frame_indices.append(
                _parse_int(
                    row.get(frame_index_field) if frame_index_field else None,
                    row_index,
                )
            )
            raw_time = (
                _optional_float(row.get(frame_time_field))
                if frame_time_field
                else None
            )
            frame_times_seconds.append(
                raw_time / 1000.0 if raw_time is not None else None
            )
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
        "frame_indices": frame_indices,
        "frame_times_seconds": frame_times_seconds,
        "landmark_indices": [index for index, _, _ in landmark_pairs],
        "quality": _quality_metadata(
            frame_quality_array,
            available=quality_available,
        ),
        "_feature_mask": np.asarray(
            [au_id in selected for au_id in requested],
            dtype=bool,
        ),
        "_frame_quality": frame_quality_array,
        "_landmarks_2d": (
            np.asarray(landmark_rows, dtype=np.float32)
            if landmark_rows
            else None
        ),
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
    full_pairs = _summary_pairs(full_au_ids)
    supported_pairs = [
        pair
        for pair in full_pairs
        if pair[0] in supported and pair[1] in supported
    ]
    indices.extend(
        block_size * 3 + full_pairs.index(pair)
        for pair in supported_pairs
    )
    return indices


def au_summary(
    sequence: np.ndarray,
    *,
    au_ids: Iterable[int] | None = None,
    active_threshold: float = DEFAULT_ACTIVE_THRESHOLD,
    coactivation_pairs: Iterable[tuple[int, int]] | None = None,
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
    pairs = (
        [
            (int(left), int(right))
            for left, right in coactivation_pairs
        ]
        if coactivation_pairs is not None
        else _summary_pairs(au_ids)
    )
    if any(
        left not in au_ids or right not in au_ids
        for left, right in pairs
    ):
        raise ValueError("Coactivation pairs must use the supplied AU ids.")
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
        "supported_au_ids": list(au_ids),
        "missing_au_ids": [],
        "feature_type": "intensity",
        "presence_au_ids": list(presence_au_ids),
        "supported_presence_au_ids": list(presence_au_ids),
        "missing_presence_au_ids": [],
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


def _smooth_signal_on_valid_runs(
    signal: np.ndarray,
    valid_mask: np.ndarray,
    window: int = 3,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if len(signal) != len(valid_mask):
        raise ValueError("Signal and valid mask must have the same length.")
    smoothed = np.full(len(signal), np.nan, dtype=np.float32)
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) == 0:
        return smoothed

    run_start = int(valid_indices[0])
    previous = run_start
    for current in valid_indices[1:]:
        current = int(current)
        if current != previous + 1:
            smoothed[run_start:previous + 1] = _smooth_signal(
                signal[run_start:previous + 1],
                window=window,
            )
            run_start = current
        previous = current
    smoothed[run_start:previous + 1] = _smooth_signal(
        signal[run_start:previous + 1],
        window=window,
    )
    return smoothed


def _event_summary(
    signal: np.ndarray,
    *,
    active_threshold: float,
    valid_mask: np.ndarray | None = None,
    frame_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    frame_count = len(signal)
    if valid_mask is None:
        valid_mask = np.ones(frame_count, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if len(valid_mask) != frame_count:
            raise ValueError("Signal and valid mask must have the same length.")
    if frame_indices is None:
        frame_indices = np.arange(frame_count, dtype=np.int64)
    else:
        frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        if len(frame_indices) != frame_count:
            raise ValueError(
                "Signal and frame indices must have the same length."
            )

    smoothed = (
        _smooth_signal(signal)
        if bool(np.all(valid_mask))
        else _smooth_signal_on_valid_runs(signal, valid_mask)
    )
    signal = np.nan_to_num(
        smoothed,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    if frame_count == 0:
        return {
            "active_ratio": 0.0,
            "event_count": 0,
            "longest_event_ratio": 0.0,
            "mean_event_ratio": 0.0,
            "onset_position": None,
            "peak_position": None,
            "peak_intensity": 0.0,
            "events": [],
        }

    active = (signal >= float(active_threshold)) & valid_mask
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    durations = (
        ends - starts + 1
        if len(starts)
        else np.asarray([], dtype=np.int64)
    )
    valid_indices = np.flatnonzero(valid_mask)
    peak_index = (
        int(valid_indices[np.argmax(signal[valid_indices])])
        if len(valid_indices)
        else None
    )
    frame_start = int(frame_indices[0]) if frame_count else 0
    frame_span = (
        max(int(frame_indices[-1]) - frame_start, 1)
        if frame_count
        else 1
    )

    def position(index: int) -> float:
        return float(
            (int(frame_indices[index]) - frame_start) / frame_span
        )

    events: list[dict[str, Any]] = []
    for start, end in zip(starts, ends):
        start = int(start)
        end = int(end)
        segment = signal[start:end + 1]
        peak_offset = int(np.argmax(segment))
        event_peak_frame = start + peak_offset
        duration_frames = int(
            frame_indices[end] - frame_indices[start] + 1
        )
        events.append(
            {
                "start_frame": int(frame_indices[start]),
                "end_frame": int(frame_indices[end]),
                "start_position": position(start),
                "end_position": position(end),
                "duration_frames": duration_frames,
                "duration_ratio": float(
                    duration_frames / max(frame_span + 1, 1)
                ),
                "peak_frame": int(frame_indices[event_peak_frame]),
                "peak_position": position(event_peak_frame),
                "peak_intensity": float(signal[event_peak_frame]),
                "mean_intensity": float(np.mean(segment)),
                "salient": bool(
                    float(signal[event_peak_frame]) >= 0.50
                    and duration_frames / max(frame_span + 1, 1) >= 0.05
                ),
            }
        )
    return {
        "active_ratio": float(np.mean(active)),
        "event_count": int(len(durations)),
        "longest_event_ratio": (
            float(np.max(durations) / max(frame_span + 1, 1))
            if len(durations)
            else 0.0
        ),
        "mean_event_ratio": (
            float(
                np.mean(
                    [
                        int(frame_indices[end] - frame_indices[start] + 1)
                        for start, end in zip(starts, ends)
                    ]
                )
                / max(frame_span + 1, 1)
            )
            if len(durations)
            else 0.0
        ),
        "onset_position": (
            position(int(starts[0]))
            if len(starts)
            else None
        ),
        "peak_position": (
            position(peak_index) if peak_index is not None else None
        ),
        "peak_intensity": (
            float(signal[peak_index]) if peak_index is not None else 0.0
        ),
        "events": events,
    }


def temporal_event_features(
    sequence: np.ndarray,
    *,
    au_ids: Iterable[int] = DEFAULT_AU_IDS,
    active_threshold: float = DEFAULT_ACTIVE_THRESHOLD,
    valid_mask: np.ndarray | None = None,
    frame_indices: Iterable[int] | None = None,
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
    if valid_mask is None:
        valid_mask = np.ones(sequence.shape[0], dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if len(valid_mask) != len(sequence):
            raise ValueError(
                "Sequence and valid mask must have the same length."
            )
    if frame_indices is None:
        frame_indices = np.arange(sequence.shape[0], dtype=np.int64)
    else:
        frame_indices = np.asarray(
            tuple(int(value) for value in frame_indices),
            dtype=np.int64,
        )
        if len(frame_indices) != len(sequence):
            raise ValueError(
                "Sequence and frame indices must have the same length."
            )
    per_au = {
        str(au_id): _event_summary(
            sequence[:, index],
            active_threshold=active_threshold,
            valid_mask=valid_mask,
            frame_indices=frame_indices,
        )
        for index, au_id in enumerate(au_ids)
    }
    return {
        "frame_count": int(sequence.shape[0]),
        "active_threshold": float(active_threshold),
        "aggregate": _event_summary(
            np.nan_to_num(
                np.nanmean(sequence, axis=1),
                nan=0.0,
            ),
            active_threshold=active_threshold,
            valid_mask=valid_mask,
            frame_indices=frame_indices,
        ),
        "per_au": per_au,
        "valid_frame_count": int(np.sum(valid_mask)),
        "valid_frame_ratio": float(np.mean(valid_mask)),
    }


def _face_mesh_action_features(
    metadata: dict[str, Any],
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    points = metadata.get("_landmarks_2d")
    landmark_indices = [
        int(value) for value in metadata.get("landmark_indices", [])
    ]
    if not isinstance(points, np.ndarray) or points.ndim != 3:
        return {
            "status": "unavailable",
            "backend": "mediapipe_face_mesh_csv",
            "reason": "Face Mesh landmarks are not available in the AU output.",
        }
    positions = {index: column for column, index in enumerate(landmark_indices)}
    required = (13, 14, 61, 291, 105, 334, 159, 145, 386, 374, 234, 454)
    if any(index not in positions for index in required):
        return {
            "status": "unavailable",
            "backend": "mediapipe_face_mesh_csv",
            "reason": "Face Mesh output is missing expression landmarks.",
        }
    if valid_mask is None:
        valid_mask = np.ones(len(points), dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if len(valid_mask) != len(points):
        raise ValueError("Face Mesh landmarks and valid mask must have the same length.")
    if not np.any(valid_mask):
        return {
            "status": "unavailable",
            "backend": "mediapipe_face_mesh_csv",
            "reason": "No valid Face Mesh frames passed the face-quality gate.",
        }

    def point(index: int) -> np.ndarray:
        return points[:, positions[index], :2]

    def distance(left: int, right: int) -> np.ndarray:
        return np.linalg.norm(point(left) - point(right), axis=1)

    face_width = distance(234, 454)
    mouth_width = distance(61, 291)
    mouth_aspect = distance(13, 14) / (mouth_width + 1e-6)
    brow_eye_gap = (
        distance(105, 159) + distance(334, 386)
    ) / (2.0 * (face_width + 1e-6))
    eye_opening = (
        distance(159, 145) + distance(386, 374)
    ) / (2.0 * (face_width + 1e-6))

    center = (point(234) + point(454)) / 2.0
    normalized_points = (points[:, :, :2] - center[:, None, :]) / (
        face_width[:, None, None] + 1e-6
    )
    global_motion = np.concatenate(
        [
            np.zeros((1,), dtype=np.float32),
            np.mean(
                np.linalg.norm(
                    np.diff(normalized_points, axis=0),
                    axis=2,
                ),
                axis=1,
            ),
        ]
    )

    def event(signal: np.ndarray, threshold: float) -> dict[str, Any]:
        result = _event_summary(
            signal,
            active_threshold=threshold,
            valid_mask=valid_mask,
        )
        for item in result["events"]:
            item["salient"] = bool(
                float(item["peak_intensity"]) >= threshold * 1.25
                and float(item["duration_ratio"])
                >= FACE_MESH_SALIENT_DURATION_RATIO
            )
        result["salient_event_count"] = sum(
            1 for item in result["events"] if item["salient"]
        )
        return result

    mouth_open = event(mouth_aspect, FACE_MESH_MOUTH_OPEN_THRESHOLD)
    mouth_change = event(
        np.r_[0.0, np.abs(np.diff(mouth_aspect))],
        FACE_MESH_MOUTH_CHANGE_THRESHOLD,
    )
    brow_change = event(
        np.r_[0.0, np.abs(np.diff(brow_eye_gap))],
        FACE_MESH_BROW_CHANGE_THRESHOLD,
    )
    eye_change = event(
        np.r_[0.0, np.abs(np.diff(eye_opening))],
        FACE_MESH_EYE_CHANGE_THRESHOLD,
    )
    global_motion_events = event(
        global_motion,
        FACE_MESH_GLOBAL_MOTION_THRESHOLD,
    )

    def percentile(signal: np.ndarray, value: float) -> float:
        finite = signal[valid_mask]
        return float(np.quantile(finite, value)) if len(finite) else 0.0

    mouth_evidence = min(
        1.0,
        max(
            0.0,
            (percentile(mouth_aspect, 0.95) - FACE_MESH_MOUTH_OPEN_THRESHOLD)
            / FACE_MESH_MOUTH_OPEN_THRESHOLD,
        ),
    )
    motion_evidence = max(
        1.0 if any(
            item["salient"]
            for summary in (
                mouth_open,
                mouth_change,
                brow_change,
                eye_change,
                global_motion_events,
            )
            for item in summary["events"]
        ) else 0.0,
        min(
            1.0,
            max(
                0.0,
                (
                    percentile(global_motion, 0.95)
                    - FACE_MESH_GLOBAL_MOTION_THRESHOLD
                )
                / FACE_MESH_GLOBAL_MOTION_THRESHOLD,
            ),
        ),
    )
    return {
        "status": "available",
        "backend": "mediapipe_face_mesh_csv",
        "metrics": {
            "mouth_aspect_peak": float(np.max(mouth_aspect[valid_mask])),
            "mouth_aspect_p95": percentile(mouth_aspect, 0.95),
            "brow_eye_gap_mean": float(np.mean(brow_eye_gap[valid_mask])),
            "eye_opening_mean": float(np.mean(eye_opening[valid_mask])),
            "global_motion_p95": percentile(global_motion, 0.95),
            "mouth_evidence_0_1": mouth_evidence,
            "motion_evidence_0_1": motion_evidence,
            "expression_confidence_0_1": max(mouth_evidence, motion_evidence),
        },
        "mouth_open": mouth_open,
        "mouth_change": mouth_change,
        "brow_change": brow_change,
        "eye_change": eye_change,
        "global_motion": global_motion_events,
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


def _event_sequence_similarity(
    generated_events: list[dict[str, Any]],
    driver_events: list[dict[str, Any]],
) -> dict[str, Any]:
    if not generated_events and not driver_events:
        return {
            "event_count_difference": 0,
            "matched_event_count": 0,
            "event_count_similarity_0_1": 1.0,
            "event_sequence_similarity_0_1": 1.0,
            "event_pairs": [],
        }
    if not generated_events or not driver_events:
        return {
            "event_count_difference": abs(
                len(generated_events) - len(driver_events)
            ),
            "matched_event_count": 0,
            "event_count_similarity_0_1": 0.0,
            "event_sequence_similarity_0_1": 0.0,
            "event_pairs": [],
        }

    pair_count = min(len(generated_events), len(driver_events))
    event_pairs: list[dict[str, Any]] = []
    pair_scores: list[float] = []
    for index in range(pair_count):
        generated_event = generated_events[index]
        driver_event = driver_events[index]
        similarities = {
            "start_position_similarity_0_1": max(
                0.0,
                1.0
                - abs(
                    float(generated_event["start_position"])
                    - float(driver_event["start_position"])
                ),
            ),
            "end_position_similarity_0_1": max(
                0.0,
                1.0
                - abs(
                    float(generated_event["end_position"])
                    - float(driver_event["end_position"])
                ),
            ),
            "duration_similarity_0_1": max(
                0.0,
                1.0
                - abs(
                    float(generated_event["duration_ratio"])
                    - float(driver_event["duration_ratio"])
                ),
            ),
            "peak_position_similarity_0_1": max(
                0.0,
                1.0
                - abs(
                    float(generated_event["peak_position"])
                    - float(driver_event["peak_position"])
                ),
            ),
            "peak_intensity_similarity_0_1": max(
                0.0,
                1.0
                - abs(
                    float(generated_event["peak_intensity"])
                    - float(driver_event["peak_intensity"])
                ),
            ),
        }
        score = float(np.mean(list(similarities.values())))
        pair_scores.append(score)
        event_pairs.append(
            {
                "generated_index": index,
                "driver_index": index,
                "score_0_1": score,
                **similarities,
            }
        )

    count_similarity = float(
        pair_count / max(len(generated_events), len(driver_events))
    )
    sequence_similarity = float(
        np.mean(pair_scores) * count_similarity
    )
    return {
        "event_count_difference": abs(
            len(generated_events) - len(driver_events)
        ),
        "matched_event_count": pair_count,
        "event_count_similarity_0_1": count_similarity,
        "event_sequence_similarity_0_1": sequence_similarity,
        "event_pairs": event_pairs,
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
    aggregate.update(
        _event_sequence_similarity(
            generated.get("aggregate", {}).get("events", []),
            driver.get("aggregate", {}).get("events", []),
        )
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
        similarity.update(
            _event_sequence_similarity(
                generated_per_au[au_id].get("events", []),
                driver_per_au[au_id].get("events", []),
            )
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
                        similarity["event_sequence_similarity_0_1"],
                    ]
                )
            )
        )
    aggregate_score = float(
        np.mean(
            [
                aggregate["active_ratio_similarity_0_1"],
                aggregate["duration_similarity_0_1"],
                aggregate["onset_similarity_0_1"],
                aggregate["peak_position_similarity_0_1"],
                aggregate["event_sequence_similarity_0_1"],
            ]
        )
    )
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
    summary_pairs = [
        pair
        for pair in _summary_pairs(full_au_ids)
        if pair[0] in supported_au_ids and pair[1] in supported_au_ids
    ]
    summary = au_summary(
        sequence,
        au_ids=supported_au_ids,
        coactivation_pairs=summary_pairs,
    )
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
    metadata.pop("_landmarks_2d", None)
    return metadata


def _quality_mask(
    sequence: np.ndarray,
    metadata: dict[str, Any],
) -> np.ndarray:
    frame_quality = np.asarray(
        metadata.get("_frame_quality", np.ones(len(sequence))),
        dtype=np.float32,
    )
    if len(frame_quality) != len(sequence):
        raise ValueError(
            "AU frame-quality metadata does not match the sequence length."
        )
    quality = metadata.get("quality", {})
    if not bool(quality.get("available")):
        return np.ones(len(sequence), dtype=bool)
    return frame_quality >= DEFAULT_FACE_QUALITY_THRESHOLD


def _frame_indices(metadata: dict[str, Any], length: int) -> np.ndarray:
    values = metadata.get("frame_indices")
    if values is None or len(values) != length:
        return np.arange(length, dtype=np.int64)
    return np.asarray(values, dtype=np.int64)


def _quality_filtered_sequence(
    sequence: np.ndarray,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    frame_quality = np.asarray(
        metadata.get("_frame_quality", np.ones(len(sequence))),
        dtype=np.float32,
    )
    usable = _quality_mask(sequence, metadata)
    if (
        bool(metadata.get("quality", {}).get("available"))
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
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    supported = tuple(
        int(value) for value in metadata.get("supported_au_ids", [])
    )
    if valid_mask is None:
        valid_mask = np.ones(len(sequence), dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if len(valid_mask) != len(sequence):
            raise ValueError(
                "Presence sequence and valid mask must have the same length."
            )
    activation_ratio: dict[str, float | None] = {}
    for index, au_id in enumerate(au_ids):
        values = sequence[valid_mask, index]
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
        "valid_frame_count": int(np.sum(valid_mask)),
        "valid_frame_ratio": float(np.mean(valid_mask)),
    }


def _combine_personal_au_scores(
    class_scores: dict[str, dict[str, Any]],
    presence_scores: dict[str, float | None],
) -> None:
    """Fuse intensity and presence evidence when both are available."""
    for class_name, score in class_scores.items():
        intensity_score = float(score["personal_au_score_0_1"])
        presence_score = presence_scores.get(class_name)
        score["intensity_personal_au_score_0_1"] = intensity_score
        score["presence_fit_score_0_1"] = presence_score
        if presence_score is None or not math.isfinite(float(presence_score)):
            continue
        score["personal_au_score_0_1"] = float(
            max(
                0.0,
                min(
                    1.0,
                    INTENSITY_PERSONAL_SCORE_WEIGHT * intensity_score
                    + PRESENCE_PERSONAL_SCORE_WEIGHT * float(presence_score),
                ),
            )
        )
        score["personal_au_score_aggregation"] = (
            f"{INTENSITY_PERSONAL_SCORE_WEIGHT:.2f} * intensity + "
            f"{PRESENCE_PERSONAL_SCORE_WEIGHT:.2f} * presence"
        )


def _add_auto_selection_scores(
    class_scores: dict[str, dict[str, Any]],
) -> None:
    """Create a cross-class score that is not biased by class thresholds."""
    if not class_scores:
        return
    logits = np.asarray(
        [
            -float(score["summary_distance"])
            - 0.5 * float(score["frame_anomaly_ratio"])
            for score in class_scores.values()
        ],
        dtype=np.float64,
    )
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    probabilities /= max(float(np.sum(probabilities)), 1e-12)
    has_presence = any(
        score.get("presence_fit_score_0_1") is not None
        for score in class_scores.values()
    )
    for probability, score in zip(probabilities, class_scores.values()):
        intensity_rank = float(probability)
        presence_score = score.get("presence_fit_score_0_1")
        score["auto_intensity_rank_score_0_1"] = intensity_rank
        if has_presence and presence_score is not None:
            score["auto_selection_score_0_1"] = float(
                INTENSITY_PERSONAL_SCORE_WEIGHT * intensity_rank
                + PRESENCE_PERSONAL_SCORE_WEIGHT * float(presence_score)
            )
        else:
            score["auto_selection_score_0_1"] = intensity_rank


def _score_auto_emotion_profile(
    *,
    generated_au_path: str | Path,
    generated_scored: np.ndarray,
    generated_supported: tuple[int, ...],
    profile: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Score emotion classes from a separate, general AU reference profile."""
    classes = profile.get("classes", {})
    if not isinstance(classes, dict):
        return {}, "emotion profile has no class models"
    if len(classes) < AUTO_EMOTION_MIN_CLASSES:
        return {}, "emotion profile needs at least two emotion classes"
    sample_counts = [
        int(model.get("sample_count", 0))
        for model in classes.values()
        if isinstance(model, dict)
    ]
    if (
        len(sample_counts) < AUTO_EMOTION_MIN_CLASSES
        or min(sample_counts) < AUTO_EMOTION_MIN_SAMPLES_PER_CLASS
    ):
        return {}, (
            "emotion profile has too few labeled samples per class "
            f"(minimum {AUTO_EMOTION_MIN_SAMPLES_PER_CLASS})"
        )
    if profile.get("auto_classification_ready") is False:
        return {}, str(
            profile.get(
                "auto_classification_reason",
                "emotion profile is not ready for automatic classification",
            )
        )

    profile_au_ids = tuple(int(value) for value in profile.get("au_ids", ()))
    supported = tuple(
        au_id for au_id in generated_supported if au_id in profile_au_ids
    )
    if not supported:
        return {}, "emotion profile has no AU columns in common with the result"
    generated_indices = [
        generated_supported.index(au_id) for au_id in supported
    ]
    sequence = generated_scored[:, generated_indices]
    scores = {
        class_name: _profile_model_score(
            sequence,
            model,
            full_au_ids=profile_au_ids,
            supported_au_ids=supported,
        )
        for class_name, model in classes.items()
    }

    presence_scores: dict[str, float | None] = {}
    presence_au_ids = tuple(
        int(value)
        for value in profile.get("presence_au_ids", DEFAULT_PRESENCE_AU_IDS)
    )
    try:
        presence, _, presence_meta = load_au_table(
            generated_au_path,
            presence_au_ids,
            feature_type="presence",
            strict=False,
            intensity_scale=1.0,
        )
        presence_mask = _quality_mask(presence, presence_meta)
        for class_name in classes:
            presence_scores[class_name] = _presence_report(
                presence,
                presence_au_ids,
                presence_meta,
                profile,
                class_name,
                active_threshold=float(
                    profile.get("active_threshold", DEFAULT_ACTIVE_THRESHOLD)
                ),
                valid_mask=presence_mask,
            ).get("fit_score_0_1")
    except ValueError:
        pass

    _combine_personal_au_scores(scores, presence_scores)
    _add_auto_selection_scores(scores)
    return scores, None


def _au_time_curve(
    sequence: np.ndarray,
    au_ids: tuple[int, ...],
    *,
    max_points: int = 96,
    frame_indices: Iterable[int] | None = None,
    valid_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    sequence = np.asarray(sequence, dtype=np.float32)
    if frame_indices is None:
        frame_indices_array = np.arange(len(sequence), dtype=np.int64)
    else:
        frame_indices_array = np.asarray(
            tuple(int(value) for value in frame_indices),
            dtype=np.int64,
        )
    if len(frame_indices_array) != len(sequence):
        raise ValueError(
            "Sequence and time-curve frame indices must have the same length."
        )
    if valid_mask is None:
        valid_mask_array = np.ones(len(sequence), dtype=bool)
    else:
        valid_mask_array = np.asarray(valid_mask, dtype=bool)
    if len(valid_mask_array) != len(sequence):
        raise ValueError(
            "Sequence and time-curve valid mask must have the same length."
        )
    if len(sequence) <= max_points:
        indices = np.arange(len(sequence), dtype=np.int64)
    else:
        indices = np.rint(
            np.linspace(0, len(sequence) - 1, max_points)
        ).astype(np.int64)
    return [
        {
            "frame_index": int(frame_indices_array[index]),
            "position": float(
                (
                    int(frame_indices_array[index])
                    - int(frame_indices_array[0])
                )
                / max(
                    int(frame_indices_array[-1])
                    - int(frame_indices_array[0]),
                    1,
                )
            ),
            "valid": bool(valid_mask_array[index]),
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

    generated_quality = generated_meta.get("quality", {})
    generated_quality_mask = _quality_mask(
        generated,
        generated_meta,
    )
    generated_scored = (
        generated[generated_quality_mask]
        if bool(generated_quality.get("available"))
        and int(np.sum(generated_quality_mask)) >= 3
        else generated
    )
    generated_temporal = temporal_event_features(
        generated,
        au_ids=au_ids,
        active_threshold=float(
            profile.get("active_threshold", DEFAULT_ACTIVE_THRESHOLD)
        ),
        valid_mask=generated_quality_mask,
        frame_indices=_frame_indices(generated_meta, len(generated)),
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
        driver_quality = driver_meta.get("quality", {})
        driver_quality_mask = _quality_mask(driver, driver_meta)
        driver_scored = (
            driver[driver_quality_mask]
            if bool(driver_quality.get("available"))
            and int(np.sum(driver_quality_mask)) >= 3
            else driver
        )
        driver_expression_score = dtw_similarity(
            generated_scored,
            driver_scored,
        )
        driver_temporal = temporal_event_features(
            driver,
            au_ids=au_ids,
            active_threshold=float(
                profile.get("active_threshold", DEFAULT_ACTIVE_THRESHOLD)
            ),
            valid_mask=driver_quality_mask,
            frame_indices=_frame_indices(driver_meta, len(driver)),
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
        quality_status
        if quality_status in {"pass", "partial", "uncertain"}
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
            if evidence_quality_status in {"partial", "uncertain"}
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
    emotion_profile_path: str | Path | None = None,
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
    generated_quality_mask = _quality_mask(
        generated_sequence,
        generated_meta,
    )
    generated_frame_indices = _frame_indices(
        generated_meta,
        len(generated_sequence),
    )
    generated_scored, _ = _quality_filtered_sequence(
        generated_sequence,
        generated_meta,
    )
    generated_quality = generated_meta.get("quality", {})
    active_threshold = float(
        profile.get("active_threshold", DEFAULT_ACTIVE_THRESHOLD)
    )
    generated_temporal = temporal_event_features(
        generated_sequence,
        au_ids=generated_supported,
        active_threshold=active_threshold,
        valid_mask=generated_quality_mask,
        frame_indices=generated_frame_indices,
    )
    face_mesh = _face_mesh_action_features(
        generated_meta,
        generated_quality_mask,
    )
    generated_temporal["face_mesh"] = face_mesh

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
    presence_au_ids = tuple(
        int(value) for value in profile.get(
            "presence_au_ids",
            DEFAULT_PRESENCE_AU_IDS,
        )
    )
    presence_reports: dict[str, dict[str, Any]] = {}
    try:
        generated_presence, _, generated_presence_meta = load_au_table(
            generated_au_path,
            presence_au_ids,
            feature_type="presence",
            strict=False,
            intensity_scale=1.0,
        )
        generated_presence_quality_mask = _quality_mask(
            generated_presence,
            generated_presence_meta,
        )
        for class_name in classes:
            presence_reports[class_name] = _presence_report(
                generated_presence,
                presence_au_ids,
                generated_presence_meta,
                profile,
                class_name,
                active_threshold=active_threshold,
                valid_mask=generated_presence_quality_mask,
            )
    except ValueError:
        presence_reports = {}

    _combine_personal_au_scores(
        class_scores,
        {
            class_name: report.get("fit_score_0_1")
            for class_name, report in presence_reports.items()
        },
    )
    _add_auto_selection_scores(class_scores)
    best_expression_class = max(
        class_scores,
        key=lambda name: class_scores[name]["auto_selection_score_0_1"],
    )
    best_intensity_score = float(
        class_scores[best_expression_class]["intensity_personal_au_score_0_1"]
    )
    face_mesh_confidence = (
        float(face_mesh.get("metrics", {}).get("expression_confidence_0_1"))
        if face_mesh.get("status") == "available"
        else None
    )
    neutral_selection_reason = (
        "no_clear_expression"
        if best_intensity_score < AUTO_NEUTRAL_INTENSITY_THRESHOLD
        else "face_mesh_no_salient_motion"
    )

    auto_class_scores: dict[str, dict[str, Any]] = {}
    auto_classification_status = "legacy_target_profile"
    auto_classification_reason = (
        "No separate original AU emotion profile was supplied."
    )
    if emotion_profile_path is not None:
        auto_classification_status = "unavailable"
        emotion_path = Path(emotion_profile_path)
        if not emotion_path.is_file():
            auto_classification_reason = (
                f"Original AU emotion profile was not found: {emotion_path}"
            )
        else:
            try:
                emotion_profile = _load_profile(emotion_path)
                auto_class_scores, profile_reason = _score_auto_emotion_profile(
                    generated_au_path=generated_au_path,
                    generated_scored=generated_scored,
                    generated_supported=generated_supported,
                    profile=emotion_profile,
                )
                if auto_class_scores:
                    auto_classification_status = "available"
                    auto_classification_reason = (
                        "Automatic emotion class comes from the original AU "
                        "emotion profile."
                    )
                else:
                    auto_classification_reason = profile_reason or (
                        "Original AU emotion profile could not be scored."
                    )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                auto_classification_reason = (
                    f"Original AU emotion profile is invalid: {exc}"
                )

    if expected_class:
        if expected_class not in class_scores:
            raise ValueError(
                f"Unknown expected expression class: {expected_class}"
            )
        selected_class = expected_class
    elif emotion_profile_path is not None:
        if auto_class_scores:
            selected_class = max(
                auto_class_scores,
                key=lambda name: auto_class_scores[name][
                    "auto_selection_score_0_1"
                ],
            )
        else:
            selected_class = "unknown"
    elif (
        best_intensity_score < AUTO_NEUTRAL_INTENSITY_THRESHOLD
        or (
            face_mesh_confidence is not None
            and face_mesh_confidence < 0.25
        )
    ):
        class_scores["neutral"] = {
            "personal_au_score_0_1": None,
            "intensity_personal_au_score_0_1": best_intensity_score,
            "presence_fit_score_0_1": None,
            "auto_selection_score_0_1": 0.0,
            "selection_reason": neutral_selection_reason,
            "face_mesh_confidence_0_1": face_mesh_confidence,
            "frame_anomaly_ratio": 0.0,
            "summary_distance": None,
            "summary_threshold": None,
            "summary_anomaly": False,
            "max_frame_distance": None,
            "anomalous_frame_indices": [],
        }
        selected_class = "neutral"
    else:
        selected_class = best_expression_class

    target_evidence_class = (
        selected_class
        if selected_class in class_scores
        and selected_class != "neutral"
        else best_expression_class
    )
    selected = (
        class_scores[selected_class]
        if selected_class in class_scores
        else class_scores[target_evidence_class]
    )
    generated_presence_report = presence_reports.get(target_evidence_class)
    if generated_presence_report is None:
        generated_presence_report = {
            "feature_type": "presence",
            "supported_au_ids": [],
            "missing_au_ids": list(presence_au_ids),
            "activation_ratio": {},
            "fit_score_0_1": None,
            "quality": None,
            "status": "unavailable",
        }
    if selected_class == "neutral":
        generated_presence_report["status"] = "not_applicable"
        generated_presence_report["reason"] = (
            "No clear expression signal passed the automatic selection threshold."
        )

    driver_expression_score: float | None = None
    driver_dtw_score: float | None = None
    driver_velocity_score: float | None = None
    driver_temporal_alignment_score: float | None = None
    driver_temporal_alignment: dict[str, Any] | None = None
    driver_meta: dict[str, Any] | None = None
    driver_similarity_proxy: float | None = None
    driver_sequence: np.ndarray | None = None
    driver_scored: np.ndarray | None = None
    driver_quality_mask: np.ndarray | None = None
    driver_frame_indices: np.ndarray | None = None
    driver_supported: tuple[int, ...] = ()
    common_driver_au_ids: tuple[int, ...] = ()
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
            generated_driver_indices = [
                generated_supported.index(au_id)
                for au_id in common_driver_au_ids
            ]
            driver_indices = [
                au_ids.index(au_id) for au_id in common_driver_au_ids
            ]
            generated_driver_sequence = generated_sequence[
                :, generated_driver_indices
            ]
            generated_driver_scored = generated_scored[
                :, generated_driver_indices
            ]
            generated_driver_quality_mask = generated_quality_mask
            driver_sequence = driver[:, driver_indices]
            driver_quality_mask = _quality_mask(driver_sequence, driver_meta)
            driver_frame_indices = _frame_indices(
                driver_meta,
                len(driver_sequence),
            )
            driver_scored, _ = _quality_filtered_sequence(
                driver_sequence,
                driver_meta,
            )
            driver_dtw_score = dtw_similarity(
                generated_driver_scored,
                driver_scored,
            )
            driver_velocity_score = velocity_similarity(
                generated_driver_scored,
                driver_scored,
            )
            driver_expression_score = float(
                np.mean([driver_dtw_score, driver_velocity_score])
            )
            driver_temporal = temporal_event_features(
                driver_sequence,
                au_ids=common_driver_au_ids,
                active_threshold=active_threshold,
                valid_mask=driver_quality_mask,
                frame_indices=driver_frame_indices,
            )
            generated_driver_temporal = temporal_event_features(
                generated_driver_sequence,
                au_ids=common_driver_au_ids,
                active_threshold=active_threshold,
                valid_mask=generated_driver_quality_mask,
                frame_indices=generated_frame_indices,
            )
            driver_temporal_alignment = compare_temporal_events(
                generated_driver_temporal,
                driver_temporal,
            )
            driver_temporal_alignment_score = float(
                driver_temporal_alignment["event_alignment_score_0_1"]
            )
            generated_summary = au_summary(
                generated_driver_scored,
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
            classifier_pairs = [
                pair
                for pair in _summary_pairs(classifier_au_ids)
                if pair[0] in common_classifier_au_ids
                and pair[1] in common_classifier_au_ids
            ]
            classifier_risk = score_leakage_classifier(
                classifier,
                au_summary(
                    classifier_sequence,
                    au_ids=common_classifier_au_ids,
                    coactivation_pairs=classifier_pairs,
                ),
                feature_indices=_summary_feature_indices(
                    classifier_au_ids,
                    common_classifier_au_ids,
                ),
            )

    selected_personal_score = selected.get("personal_au_score_0_1")
    personal_score = (
        float(selected_personal_score)
        if selected_personal_score is not None
        else None
    )
    if selected_class == "neutral":
        leakage_risk = None
        leakage_backend = "not_applicable_no_clear_expression"
    elif classifier_risk is not None:
        leakage_risk = classifier_risk
        leakage_backend = "trained_au_leakage_classifier"
    elif driver_similarity_proxy is not None:
        leakage_risk = max(
            0.0,
            min(
                1.0,
                driver_similarity_proxy * (1.0 - (personal_score or 0.0)),
            ),
        )
        leakage_backend = "driver_style_overlap_proxy"
    else:
        leakage_risk = float(
            max(
                selected["frame_anomaly_ratio"],
                1.0 - (personal_score or 0.0),
            )
        )
        leakage_backend = "target_au_anomaly_proxy"

    quality_status = str(generated_quality.get("status", "not_available"))
    missing_intensity_au_ids = [
        au_id for au_id in au_ids if au_id not in generated_supported
    ]
    uncertainty_reasons: list[str] = []
    if quality_status in {"partial", "uncertain"}:
        uncertainty_reasons.append("face_quality_low")
    if missing_intensity_au_ids:
        uncertainty_reasons.append("missing_intensity_au")
    evidence_quality_status = (
        quality_status
        if quality_status in {"pass", "partial", "uncertain"}
        else ("uncertain" if uncertainty_reasons else "available")
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
        "target_evidence_class": target_evidence_class,
        "expected_expression_class": expected_class,
        "class_scores": class_scores,
        "auto_classification_status": auto_classification_status,
        "auto_classification_reason": auto_classification_reason,
        "auto_class_scores": auto_class_scores,
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
        "face_mesh": face_mesh,
        "driver_temporal_alignment": driver_temporal_alignment,
        "time_curve": {
            "au_ids": list(generated_supported),
            "driver_supported_au_ids": list(driver_supported),
            "comparison_au_ids": list(common_driver_au_ids),
            "generated": _au_time_curve(
                generated_sequence,
                generated_supported,
                frame_indices=generated_frame_indices,
                valid_mask=generated_quality_mask,
            ),
            "driver": (
                _au_time_curve(
                    driver_sequence,
                    tuple(
                        au_id
                        for au_id in generated_supported
                        if au_id in driver_supported
                    ),
                    frame_indices=driver_frame_indices,
                    valid_mask=driver_quality_mask,
                )
                if driver_sequence is not None
                and driver_frame_indices is not None
                and driver_quality_mask is not None
                else driver_curve
            ),
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
        "supported_au_ids": list(au_ids),
        "missing_au_ids": [],
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
        (
            COMPLIANCE_COMPONENT_WEIGHTS["identity"],
            identity_score_0_1,
        ),
        (
            COMPLIANCE_COMPONENT_WEIGHTS["personal_au"],
            personal_au_score_0_1,
        ),
        (
            COMPLIANCE_COMPONENT_WEIGHTS["driver_expression"],
            driver_expression_score_0_1,
        ),
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
    if reasons or missing_evidence:
        decision = "review"
    else:
        decision = "allow"
    return {
        "person_likeness_score_0_1": likeness,
        "score_weight_coverage": weight_sum,
        "leakage_risk_0_1": leakage_risk_0_1,
        "weights": COMPLIANCE_COMPONENT_WEIGHTS,
        "decision": decision,
        "decision_reasons": reasons,
        "missing_evidence": missing_evidence,
        "decision_policy": (
            "score_and_review_only: threshold misses lower the score or "
            "request review; this report does not block uploads"
        ),
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
    """Judge Wang Xing-specific expression fit with explicit evidence coverage."""
    components = {
        "personal_au": personal_au_score_0_1,
        "driver_expression": driver_expression_score_0_1,
        "temporal_alignment": temporal_alignment_score_0_1,
    }
    valid_components = {
        name: float(score)
        for name, score in components.items()
        if score is not None and math.isfinite(float(score))
    }
    score_weight_coverage = sum(
        WANGXING_TARGETED_COMPONENT_WEIGHTS[name]
        for name in valid_components
    )
    expression_fit = (
        sum(
            WANGXING_TARGETED_COMPONENT_WEIGHTS[name] * score
            for name, score in valid_components.items()
        )
        / score_weight_coverage
        if score_weight_coverage
        else None
    )
    driver_scores = [
        valid_components[name]
        for name in ("driver_expression", "temporal_alignment")
        if name in valid_components
    ]
    driver_fit = (
        sum(driver_scores) / len(driver_scores)
        if driver_scores
        else None
    )
    missing_evidence = [
        name for name, score in components.items() if score is None
    ]
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
    if personal_au_score_0_1 is None:
        reasons.append("missing_personal_au")
    if driver_expression_score_0_1 is None:
        reasons.append("missing_driver_expression")
    if temporal_alignment_score_0_1 is None:
        reasons.append("missing_temporal_alignment")
    for reason in uncertainty_reasons:
        if reason not in reasons:
            reasons.append(str(reason))
    if evidence_quality_status in {"partial", "uncertain"}:
        if "evidence_quality_low" not in reasons:
            reasons.append("evidence_quality_low")

    status = (
        "complete"
        if (
            score_weight_coverage >= 1.0
            and evidence_quality_status == "pass"
            and not reasons
        )
        else ("partial" if expression_fit is not None else "unavailable")
    )
    decision = "allow" if status == "complete" else "review"

    return {
        "wangxing_expression_fit_score_0_1": expression_fit,
        "status": status,
        "decision": decision,
        "decision_reasons": reasons,
        "required_evidence": list(WANGXING_TARGETED_COMPONENT_WEIGHTS),
        "missing_evidence": missing_evidence,
        "score_weight_coverage": score_weight_coverage,
        "evidence_coverage_0_1": score_weight_coverage,
        "decision_policy": (
            "A complete allow decision requires personal AU, reference "
            "driver trajectory, and temporal alignment. Missing evidence "
            "keeps the score partial and requests review."
        ),
        "evidence": {
            "personal_au": personal_au_score_0_1,
            "driver_expression": driver_expression_score_0_1,
            "temporal_alignment": temporal_alignment_score_0_1,
            "driver_fit": driver_fit,
            "leakage_risk": leakage_risk_0_1,
        },
        "evidence_quality_status": evidence_quality_status,
        "evidence_confidence_0_1": evidence_confidence_0_1,
        "aggregation": (
            "weighted mean of personal AU (40%), driver expression (35%), "
            "and temporal alignment (25%) over available evidence"
        ),
        "thresholds": {
            "personal_au": personal_au_threshold,
            "driver_expression": driver_expression_threshold,
            "leakage": leakage_threshold,
        },
    }
