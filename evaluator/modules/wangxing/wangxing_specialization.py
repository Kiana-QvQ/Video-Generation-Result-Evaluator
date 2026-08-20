from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SPECIALIZATION_SCHEMA = "wangxing_specialization_v1"
SPECIALIZATION_EVALUATOR_VERSION = "wangxing_specialization_eval_v3"
IDENTITY_PROFILE_SCHEMA = "wangxing_identity_profile_v2"
EXPRESSION_PROFILE_SCHEMA = "wangxing_expression_profile_v2"

EXPRESSION_LABELS = {
    "kaixin": "smile",
    "fennu": "anger",
    "shengqi": "anger",
    "jingya": "surprise",
    "kongju": "fear",
    "beishang": "sadness",
    "yanwu": "disgust",
}

EXPRESSION_DISPLAY_NAMES = {
    "smile": "微笑 / smile",
    "anger": "愤怒 / anger",
    "surprise": "惊讶 / surprise",
    "fear": "紧张 / fear",
    "sadness": "悲伤 / sadness",
    "disgust": "厌恶 / disgust",
}

INTENSITY_AU_IDS = (
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
PRESENCE_AU_IDS = (
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

LANDMARK_INDEXES = (
    1,
    10,
    13,
    14,
    33,
    61,
    105,
    133,
    145,
    152,
    159,
    234,
    263,
    291,
    334,
    362,
    374,
    386,
    454,
)

GEOMETRY_FEATURE_NAMES = (
    "mouth_open",
    "mouth_width",
    "mouth_corner_balance",
    "left_eye_open",
    "right_eye_open",
    "eye_asymmetry",
    "left_brow_eye_gap",
    "right_brow_eye_gap",
    "nose_mouth_distance",
    "nose_chin_distance",
    "jaw_width",
    "pitch",
    "yaw",
    "roll",
)

SUMMARY_STATISTICS = ("median", "q25", "q75", "std", "p95", "max")
DYNAMIC_STATISTICS = ("velocity_median", "velocity_p95", "acceleration_p95")
IDENTITY_FEATURE_NAMES = (
    "real_prototype_similarity",
    "generated_prototype_similarity",
    "positive_similarity",
    "negative_similarity",
    "identity_gap",
    "frame_consistency",
    "valid_frame_ratio",
    "quality_weight_mean",
)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _canonical_au_id(name: str) -> int | None:
    match = re.search(
        r"\bau[\s_-]*0*(\d{1,2})(?!\d)",
        str(name),
        re.IGNORECASE,
    )
    if not match:
        return None
    value = int(match.group(1))
    return value if 1 <= value <= 45 else None


def _column_kind(name: str) -> str:
    normalized = str(name).casefold().replace(" ", "")
    if "intensity" in normalized or normalized.endswith("_r"):
        return "intensity"
    if (
        "presence" in normalized
        or normalized.endswith("_c")
        or re.fullmatch(r"au[_-]*0*\d{1,2}", normalized)
    ):
        return "presence"
    return "unknown"


def _landmark_row(row: dict[str, Any]) -> dict[int, tuple[float, float]]:
    points: dict[int, tuple[float, float]] = {}
    for index in LANDMARK_INDEXES:
        x = _finite_float(row.get(f"lm_mp_{index}_x"), math.nan)
        y = _finite_float(row.get(f"lm_mp_{index}_y"), math.nan)
        if math.isfinite(x) and math.isfinite(y):
            points[index] = (x, y)
    return points


def _landmark_distance(
    points: dict[int, tuple[float, float]],
    left: int,
    right: int,
) -> float:
    if left not in points or right not in points:
        return 0.0
    return float(np.linalg.norm(np.asarray(points[left]) - points[right]))


def _geometry_vector(row: dict[str, Any]) -> tuple[np.ndarray, bool]:
    points = _landmark_row(row)
    face_width = _landmark_distance(points, 234, 454)
    face_height = _landmark_distance(points, 10, 152)
    if face_width < 1e-4 or face_height < 1e-4:
        return np.zeros(len(GEOMETRY_FEATURE_NAMES), dtype=np.float32), False

    def y(index: int) -> float:
        return points.get(index, (0.0, 0.0))[1]

    eye_left = _landmark_distance(points, 159, 145) / face_height
    eye_right = _landmark_distance(points, 386, 374) / face_height
    left_brow_gap = abs(y(105) - (y(159) + y(145)) / 2.0) / face_height
    right_brow_gap = abs(y(334) - (y(386) + y(374)) / 2.0) / face_height
    mouth_corner_balance = (y(61) - y(291)) / face_height
    values = np.asarray(
        [
            _landmark_distance(points, 13, 14) / face_height,
            _landmark_distance(points, 61, 291) / face_width,
            mouth_corner_balance,
            eye_left,
            eye_right,
            abs(eye_left - eye_right),
            left_brow_gap,
            right_brow_gap,
            _landmark_distance(points, 1, 13) / face_height,
            _landmark_distance(points, 1, 152) / face_height,
            _landmark_distance(points, 172, 397) / face_width,
            _finite_float(row.get("pitch")) / 45.0,
            _finite_float(row.get("yaw")) / 45.0,
            _finite_float(row.get("roll")) / 45.0,
        ],
        dtype=np.float32,
    )
    return values, True


def _read_rows(path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def _select_au_columns(
    fieldnames: Iterable[str],
    au_ids: Iterable[int],
    kind: str,
) -> dict[int, str]:
    selected: dict[int, str] = {}
    for name in fieldnames:
        au_id = _canonical_au_id(name)
        if au_id is None or au_id not in set(au_ids):
            continue
        if _column_kind(name) != kind:
            continue
        selected.setdefault(au_id, name)
    return selected


def _fill_missing(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    output = matrix.copy()
    for index in range(output.shape[1]):
        column = output[:, index]
        valid = column[np.isfinite(column)]
        replacement = float(np.median(valid)) if valid.size else 0.0
        column[~np.isfinite(column)] = replacement
    return output


def _quantile(values: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, quantile))


def sequence_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    base_names = [
        *(f"au_intensity_{au_id}" for au_id in INTENSITY_AU_IDS),
        *(f"au_presence_{au_id}" for au_id in PRESENCE_AU_IDS),
        *(f"geometry_{name}" for name in GEOMETRY_FEATURE_NAMES),
    ]
    for name in base_names:
        names.extend(f"{name}_{statistic}" for statistic in SUMMARY_STATISTICS)
    for name in base_names:
        names.extend(f"{name}_{statistic}" for statistic in DYNAMIC_STATISTICS)
    names.extend(
        [
            "event_count",
            "event_active_ratio",
            "event_longest_ratio",
            "event_mean_ratio",
            "event_peak",
            "valid_frame_ratio",
        ]
    )
    return tuple(names)


def extract_sequence_features(
    path: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract AU, mesh geometry, pose and temporal summary features."""
    rows, fieldnames = _read_rows(path)
    if not rows:
        raise ValueError(f"No rows found in AU CSV: {path}")

    intensity_columns = _select_au_columns(
        fieldnames,
        INTENSITY_AU_IDS,
        "intensity",
    )
    presence_columns = _select_au_columns(
        fieldnames,
        PRESENCE_AU_IDS,
        "presence",
    )
    intensity = np.asarray(
        [
            [
                _finite_float(row.get(intensity_columns.get(au_id, "")), math.nan)
                for au_id in INTENSITY_AU_IDS
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    presence = np.asarray(
        [
            [
                _finite_float(row.get(presence_columns.get(au_id, "")), math.nan)
                for au_id in PRESENCE_AU_IDS
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    geometry_rows = [_geometry_vector(row) for row in rows]
    geometry = np.stack([item[0] for item in geometry_rows])
    geometry_valid = np.asarray([item[1] for item in geometry_rows], dtype=bool)
    intensity = np.clip(_fill_missing(intensity), 0.0, None)
    presence = np.clip(_fill_missing(presence), 0.0, 1.0)
    base = _fill_missing(np.concatenate([intensity, presence, geometry], axis=1))

    summaries: list[float] = []
    dynamic: list[float] = []
    for index in range(base.shape[1]):
        signal = base[:, index]
        summaries.extend(
            [
                float(np.median(signal)),
                _quantile(signal, 0.25),
                _quantile(signal, 0.75),
                float(np.std(signal)),
                _quantile(signal, 0.95),
                float(np.max(signal)),
            ]
        )
        velocity = np.diff(signal) if len(signal) > 1 else np.zeros(1)
        acceleration = (
            np.diff(signal, n=2) if len(signal) > 2 else np.zeros(1)
        )
        dynamic.extend(
            [
                float(np.median(np.abs(velocity))),
                _quantile(np.abs(velocity), 0.95),
                _quantile(np.abs(acceleration), 0.95),
            ]
        )

    au_signal = np.max(intensity, axis=1) if intensity.size else np.zeros(1)
    active = au_signal >= 0.20
    active_ratio = float(np.mean(active))
    event_count = 0
    event_lengths: list[int] = []
    current = 0
    for value in active:
        if value:
            current += 1
        elif current:
            event_count += 1
            event_lengths.append(current)
            current = 0
    if current:
        event_count += 1
        event_lengths.append(current)
    frame_count = max(len(rows), 1)
    event_features = [
        float(event_count),
        active_ratio,
        float(max(event_lengths, default=0) / frame_count),
        float(np.mean(event_lengths) / frame_count) if event_lengths else 0.0,
        float(np.max(au_signal)) if au_signal.size else 0.0,
        float(np.mean(geometry_valid)),
    ]
    feature_vector = np.asarray(
        [*summaries, *dynamic, *event_features],
        dtype=np.float32,
    )
    names = sequence_feature_names()
    if feature_vector.size != len(names):
        raise RuntimeError(
            f"Feature layout mismatch: {feature_vector.size} != {len(names)}"
        )
    return feature_vector, {
        "frame_count": len(rows),
        "valid_frame_ratio": float(np.mean(geometry_valid)),
        "supported_intensity_au_ids": [
            au_id for au_id in INTENSITY_AU_IDS if au_id in intensity_columns
        ],
        "supported_presence_au_ids": [
            au_id for au_id in PRESENCE_AU_IDS if au_id in presence_columns
        ],
        "event_statistics": {
            "event_count": event_count,
            "active_ratio": active_ratio,
            "longest_event_ratio": event_features[2],
            "mean_event_ratio": event_features[3],
            "peak_intensity": event_features[4],
        },
        "feature_source": "AU intensity/presence + Face Mesh geometry + pose + temporal derivatives",
    }


def _robust_location_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    location = np.median(matrix, axis=0)
    q25 = np.quantile(matrix, 0.25, axis=0)
    q75 = np.quantile(matrix, 0.75, axis=0)
    scale = np.maximum((q75 - q25) / 1.349, 1e-3)
    return location, scale


def _robust_distance(
    vector: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
) -> float:
    standardized = (np.asarray(vector, dtype=np.float64) - location) / scale
    return float(np.sqrt(np.mean(np.square(standardized))))


def _safe_score(distance: float, threshold: float) -> float:
    return float(math.exp(-max(distance, 0.0) / max(threshold, 1e-6)))


def _expression_class_from_path(path: Path) -> str | None:
    directory = re.sub(r"^cl_", "", path.parent.name.casefold())
    for prefix, expression_class in EXPRESSION_LABELS.items():
        if directory.startswith(prefix):
            return expression_class
    return None


def build_expression_profile(
    au_root: str | Path,
    output_path: str | Path,
    *,
    pseudo_label_manifest: str | Path | None = None,
    max_pseudo_per_class: int = 40,
    real_paths: Iterable[str | Path] | None = None,
    exclude_au_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Fit class support domains from real data plus trusted pseudo-labels."""
    au_root = Path(au_root)
    grouped: dict[str, list[np.ndarray]] = {}
    metadata: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    real_sample_count = 0
    pseudo_sample_count = 0
    source_paths = (
        sorted(Path(path) for path in real_paths)
        if real_paths is not None
        else sorted(au_root.rglob("*.csv"))
    )
    excluded = {
        str(Path(path).resolve()).casefold()
        for path in (exclude_au_paths or set())
    }
    for path in source_paths:
        if str(path.resolve()).casefold() in excluded:
            continue
        expression_class = _expression_class_from_path(path)
        if expression_class is None:
            continue
        try:
            vector, quality = extract_sequence_features(path)
        except (OSError, ValueError, RuntimeError) as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
            continue
        grouped.setdefault(expression_class, []).append(vector)
        metadata.append(
            {
                "source_id": path.relative_to(au_root).with_suffix("").as_posix(),
                "expression_class": expression_class,
                "path": str(path),
                "frame_count": quality["frame_count"],
                "valid_frame_ratio": quality["valid_frame_ratio"],
            }
        )
        real_sample_count += 1

    pseudo_label_source = None
    pseudo_counts: Counter[str] = Counter()
    if pseudo_label_manifest is not None:
        pseudo_label_source = Path(pseudo_label_manifest)
        if not pseudo_label_source.is_file():
            raise FileNotFoundError(
                f"Seedance pseudo-label manifest was not found: "
                f"{pseudo_label_source}"
            )
        try:
            manifest_payload = json.loads(
                pseudo_label_source.read_text(encoding="utf-8-sig")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid Seedance pseudo-label manifest: {exc}"
            ) from exc
        manifest_records = (
            manifest_payload.get("records", [])
            if isinstance(manifest_payload, dict)
            else []
        )
        for record in manifest_records:
            if not isinstance(record, dict):
                continue
            expression_class = str(
                record.get("pseudo_label")
                or record.get("expression_class")
                or ""
            ).strip().lower()
            if expression_class not in EXPRESSION_DISPLAY_NAMES:
                continue
            if str(record.get("label_status")) != "high_confidence":
                continue
            if pseudo_counts[expression_class] >= max_pseudo_per_class:
                continue
            au_path_value = record.get("au_path")
            if not au_path_value:
                continue
            au_path = Path(str(au_path_value))
            if str(au_path.resolve()).casefold() in excluded:
                continue
            if not au_path.is_file():
                skipped.append(
                    {
                        "path": str(au_path),
                        "reason": "pseudo-label AU path does not exist",
                    }
                )
                continue
            try:
                vector, quality = extract_sequence_features(au_path)
            except (OSError, ValueError, RuntimeError) as exc:
                skipped.append({"path": str(au_path), "reason": str(exc)})
                continue
            grouped.setdefault(expression_class, []).append(vector)
            pseudo_counts[expression_class] += 1
            pseudo_sample_count += 1
            metadata.append(
                {
                    "source_id": str(
                        record.get("video_path")
                        or record.get("source_video")
                        or au_path
                    ),
                    "expression_class": expression_class,
                    "path": str(au_path),
                    "source_type": "generated_wangxing",
                    "label_source": "seedance_content_pseudo_label",
                    "label_status": "high_confidence",
                    "label_confidence_0_1": record.get(
                        "confidence_0_1",
                        record.get("compatibility_0_1"),
                    ),
                    "frame_count": quality["frame_count"],
                    "valid_frame_ratio": quality["valid_frame_ratio"],
                }
            )

    if len(grouped) < 2:
        raise ValueError("At least two expression classes are required.")

    classes: dict[str, Any] = {}
    class_counts: Counter[str] = Counter()
    for expression_class, vectors in sorted(grouped.items()):
        matrix = np.stack(vectors).astype(np.float64)
        location, scale = _robust_location_scale(matrix)
        distances = np.asarray(
            [_robust_distance(row, location, scale) for row in matrix],
            dtype=np.float64,
        )
        threshold = max(
            0.75,
            _quantile(distances, 0.95) * 1.20,
        )
        classes[expression_class] = {
            "display_name": EXPRESSION_DISPLAY_NAMES.get(
                expression_class,
                expression_class,
            ),
            "sample_count": len(vectors),
            "location": location.tolist(),
            "scale": scale.tolist(),
            "distance_threshold": float(threshold),
            "training_distance_summary": {
                "median": float(np.median(distances)),
                "p95": _quantile(distances, 0.95),
                "max": float(np.max(distances)),
            },
        }
        class_counts[expression_class] = len(vectors)

    profile = {
        "schema_version": EXPRESSION_PROFILE_SCHEMA,
        "specialization_schema": SPECIALIZATION_SCHEMA,
        "evaluator_version": SPECIALIZATION_EVALUATOR_VERSION,
        "feature_names": list(sequence_feature_names()),
        "feature_blocks": {
            "au_intensity": list(INTENSITY_AU_IDS),
            "au_presence": list(PRESENCE_AU_IDS),
            "face_mesh_geometry": list(GEOMETRY_FEATURE_NAMES),
            "summary_statistics": list(SUMMARY_STATISTICS),
            "dynamic_statistics": list(DYNAMIC_STATISTICS),
        },
        "classes": classes,
        "class_counts": dict(class_counts),
        "provenance": {
            "source": "data/au/MD_CL",
            "real_paths_explicit": real_paths is not None,
            "real_paths_count": len(source_paths),
            "source_role": (
                "real_wangxing_expression_distribution"
                if pseudo_sample_count == 0
                else "real_wangxing_plus_trusted_seedance_expression_distribution"
            ),
            "class_mapping": EXPRESSION_LABELS,
            "real_sample_count": real_sample_count,
            "pseudo_sample_count": pseudo_sample_count,
            "sample_count": len(metadata),
            "pseudo_label_manifest": (
                str(pseudo_label_source)
                if pseudo_label_source is not None
                else None
            ),
            "pseudo_class_counts": dict(pseudo_counts),
            "skipped_count": len(skipped),
            "skipped_preview": skipped[:20],
            "metadata": metadata,
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return profile


def build_source_profile(
    *,
    real_au_root: str | Path,
    seedance_label_manifest: str | Path,
    output_path: str | Path,
    exclude_au_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Fit a real-vs-generated Wang Xing source-domain profile.

    ``exclude_au_paths`` should contain resolved absolute path strings to keep
    holdout AU CSVs out of the fitted centroids (leakage control).
    """
    real_au_root = Path(real_au_root)
    manifest_path = Path(seedance_label_manifest)
    excluded = {
        str(Path(path).resolve()).casefold()
        for path in (exclude_au_paths or set())
    }
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Seedance pseudo-label manifest was not found: {manifest_path}"
        )
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8-sig")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Seedance pseudo-label manifest: {exc}") from exc

    grouped_paths: dict[str, list[Path]] = {
        "real_wangxing": sorted(real_au_root.rglob("*.csv")),
        "generated_wangxing": [],
    }
    seen_generated: set[str] = set()
    for record in payload.get("records", []) if isinstance(payload, dict) else []:
        if not isinstance(record, dict) or not record.get("au_path"):
            continue
        path = Path(str(record["au_path"]))
        key = str(path.resolve()).casefold()
        if path.is_file() and key not in seen_generated:
            grouped_paths["generated_wangxing"].append(path)
            seen_generated.add(key)

    if excluded:
        for source_type in list(grouped_paths):
            grouped_paths[source_type] = [
                path
                for path in grouped_paths[source_type]
                if str(path.resolve()).casefold() not in excluded
            ]

    grouped: dict[str, list[np.ndarray]] = {}
    metadata: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for source_type, paths in grouped_paths.items():
        for path in paths:
            try:
                vector, quality = extract_sequence_features(path)
            except (OSError, ValueError, RuntimeError) as exc:
                skipped.append({"path": str(path), "reason": str(exc)})
                continue
            grouped.setdefault(source_type, []).append(vector)
            metadata.append(
                {
                    "source_type": source_type,
                    "path": str(path),
                    "frame_count": quality["frame_count"],
                    "valid_frame_ratio": quality["valid_frame_ratio"],
                }
            )

    if not grouped.get("real_wangxing") or not grouped.get("generated_wangxing"):
        raise ValueError(
            "Source profile needs both real Wang Xing and generated Wang Xing "
            "AU sequences."
        )

    sources: dict[str, Any] = {}
    for source_type, vectors in sorted(grouped.items()):
        matrix = np.stack(vectors).astype(np.float64)
        location, scale = _robust_location_scale(matrix)
        distances = np.asarray(
            [_robust_distance(row, location, scale) for row in matrix],
            dtype=np.float64,
        )
        sources[source_type] = {
            "location": location.tolist(),
            "scale": scale.tolist(),
            "sample_count": len(vectors),
            "distance_threshold": max(0.75, _quantile(distances, 0.95) * 1.20),
            "training_distance_summary": {
                "median": float(np.median(distances)),
                "p95": _quantile(distances, 0.95),
                "max": float(np.max(distances)),
            },
        }

    profile = {
        "schema_version": "wangxing_source_profile_v1",
        "specialization_schema": SPECIALIZATION_SCHEMA,
        "evaluator_version": SPECIALIZATION_EVALUATOR_VERSION,
        "feature_names": list(sequence_feature_names()),
        "sources": sources,
        "provenance": {
            "real_au_root": str(real_au_root),
            "seedance_label_manifest": str(manifest_path),
            "holdout_excluded_au_count": len(excluded),
            "sample_counts": {
                source_type: len(vectors)
                for source_type, vectors in grouped.items()
            },
            "skipped_count": len(skipped),
            "skipped_preview": skipped[:20],
            "metadata": metadata,
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return profile


def score_source_profile(
    au_path: str | Path,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Score whether a Wang Xing video is closer to real or generated data."""
    vector, quality = extract_sequence_features(au_path)
    scores: dict[str, dict[str, float]] = {}
    for source_type, model in profile.get("sources", {}).items():
        location = np.asarray(model.get("location", []), dtype=np.float64)
        scale = np.asarray(model.get("scale", []), dtype=np.float64)
        if location.shape != vector.shape or scale.shape != vector.shape:
            continue
        distance = _robust_distance(vector, location, scale)
        threshold = float(model.get("distance_threshold", 1.0))
        scores[source_type] = {
            "distance": distance,
            "score_0_1": _safe_score(distance, threshold),
        }
    if not scores:
        return {
            "status": "unavailable",
            "decision": "uncertain",
            "source_type": "unknown",
            "generated_probability_0_1": None,
            "scores": {},
            "quality": quality,
            "uncertainty_reasons": ["source_profile_unavailable"],
        }

    real_score = scores.get("real_wangxing", {}).get("score_0_1", 0.0)
    generated_score = scores.get(
        "generated_wangxing",
        {},
    ).get("score_0_1", 0.0)
    total = real_score + generated_score
    generated_probability = generated_score / total if total > 1e-8 else 0.5
    margin = abs(real_score - generated_score)
    reasons: list[str] = []
    if quality["valid_frame_ratio"] < 0.35:
        reasons.append("low_face_mesh_coverage")
    if margin < 0.08:
        reasons.append("source_margin_small")
    if reasons:
        decision = "uncertain"
        source_type = "uncertain"
    elif generated_score > real_score:
        decision = "generated_wangxing"
        source_type = "generated_wangxing"
    else:
        decision = "real_wangxing"
        source_type = "real_wangxing"
    return {
        "status": "available",
        "decision": decision,
        "source_type": source_type,
        "generated_probability_0_1": float(generated_probability),
        "real_probability_0_1": float(1.0 - generated_probability),
        "margin_0_1": float(margin),
        "scores": scores,
        "quality": quality,
        "uncertainty_reasons": reasons,
    }


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = _normalize(left)
    right = _normalize(right)
    return float(np.dot(left, right))


def _sample_frames(path: str | Path, max_frames: int) -> tuple[list[int], list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            frame_count = max_frames
        indexes = np.linspace(
            0,
            max(frame_count - 1, 0),
            num=min(max_frames, frame_count),
            dtype=np.int64,
        ).tolist()
        frames: list[np.ndarray] = []
        valid_indexes: list[int] = []
        for index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            valid_indexes.append(int(index))
        return valid_indexes, frames, fps
    finally:
        capture.release()


def _identity_frame_embeddings(
    path: str | Path,
    backend: Any,
    max_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    _indexes, frames, fps = _sample_frames(path, max_frames)
    embeddings: list[np.ndarray] = []
    quality_weights: list[float] = []
    backends: Counter[str] = Counter()
    for frame in frames:
        embedding, bbox, backend_name = backend.embedding(frame)
        backends[backend_name] += 1
        if embedding is None:
            continue
        embeddings.append(_normalize(np.asarray(embedding, dtype=np.float32)))
        quality_weights.append(_face_quality_weight(frame, bbox))
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), {
            "frame_count": len(frames),
            "valid_frame_count": 0,
            "valid_frame_ratio": 0.0,
            "backend": "+".join(sorted(backends)) or backend.backend,
            "fps": fps,
            "quality_weights": [],
            "quality_weight_mean": 0.0,
        }
    return np.stack(embeddings), {
        "frame_count": len(frames),
        "valid_frame_count": len(embeddings),
        "valid_frame_ratio": len(embeddings) / max(len(frames), 1),
        "backend": "+".join(sorted(backends)),
        "fps": fps,
        "quality_weights": quality_weights,
        "quality_weight_mean": float(np.mean(quality_weights)),
    }


def _face_quality_weight(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> float:
    """Estimate a conservative per-frame quality weight from face geometry."""
    if bbox is None or frame.size == 0:
        return 0.25
    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = (float(value) for value in bbox)
    if width <= 0.0 or height <= 0.0:
        return 0.25
    area_ratio = (width * height) / max(frame_width * frame_height, 1.0)
    size_score = math.sqrt(max(area_ratio, 0.0) / 0.08)
    size_score = max(0.25, min(1.0, size_score))
    center_x = (x + width / 2.0) / max(frame_width, 1.0)
    center_y = (y + height / 2.0) / max(frame_height, 1.0)
    center_distance = math.sqrt(
        (center_x - 0.5) ** 2 + (center_y - 0.5) ** 2
    )
    center_score = max(0.50, 1.0 - center_distance)
    return float(max(0.25, min(1.0, 0.70 * size_score + 0.30 * center_score)))


def _weighted_prototype(
    embeddings: np.ndarray,
    weights: Iterable[float] | None = None,
) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if weights is None:
        weights_array = np.ones(matrix.shape[0], dtype=np.float32)
    else:
        weights_array = np.asarray(list(weights), dtype=np.float32)
        if weights_array.shape != (matrix.shape[0],):
            weights_array = np.ones(matrix.shape[0], dtype=np.float32)
    weights_array = np.clip(weights_array, 0.25, 1.0)
    return _normalize(np.average(matrix, axis=0, weights=weights_array))


def _video_record(
    path: Path,
    label: str,
    backend: Any,
    max_frames: int,
) -> dict[str, Any] | None:
    embeddings, metadata = _identity_frame_embeddings(path, backend, max_frames)
    if embeddings.size == 0:
        return None
    prototype = _weighted_prototype(embeddings, metadata["quality_weights"])
    frame_scores = embeddings @ prototype
    consistency = float(
        np.average(frame_scores, weights=metadata["quality_weights"])
    )
    return {
        "path": str(path),
        "label": label,
        "prototype": prototype.tolist(),
        "frame_count": metadata["frame_count"],
        "valid_frame_count": metadata["valid_frame_count"],
        "valid_frame_ratio": metadata["valid_frame_ratio"],
        "frame_consistency": consistency,
        "quality_weight_mean": metadata["quality_weight_mean"],
        "backend": metadata["backend"],
    }


def _profile_prototype(records: list[dict[str, Any]]) -> np.ndarray:
    return _weighted_prototype(
        np.asarray([record["prototype"] for record in records], dtype=np.float32),
        [
            max(
                0.25,
                float(record.get("quality_weight_mean", 0.5))
                * float(record.get("valid_frame_ratio", 1.0)),
            )
            for record in records
        ],
    )


def _limit_paths_balanced(paths: list[Path], limit: int | None) -> list[Path]:
    if limit is None or limit <= 0 or len(paths) <= limit:
        return paths
    groups: dict[str, list[Path]] = {}
    for path in paths:
        groups.setdefault(path.parent.name.casefold(), []).append(path)
    queues = [groups[name] for name in sorted(groups)]
    selected: list[Path] = []
    while len(selected) < limit:
        progressed = False
        for queue in queues:
            if queue and len(selected) < limit:
                selected.append(queue.pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _score_identity_features(
    record: dict[str, Any],
    real_prototype: np.ndarray,
    generated_prototype: np.ndarray,
    negative_prototypes: list[np.ndarray],
) -> tuple[float, float, float]:
    prototype = np.asarray(record["prototype"], dtype=np.float32)
    real_similarity = _cosine(prototype, real_prototype)
    generated_similarity = _cosine(prototype, generated_prototype)
    positive_similarity = 0.65 * real_similarity + 0.35 * generated_similarity
    negative_similarity = max(
        (_cosine(prototype, negative) for negative in negative_prototypes),
        default=-1.0,
    )
    return positive_similarity, negative_similarity, positive_similarity - negative_similarity


def _identity_feature_vector(
    record: dict[str, Any],
    real_prototype: np.ndarray,
    generated_prototype: np.ndarray,
    negative_prototypes: list[np.ndarray],
) -> np.ndarray:
    positive_similarity, negative_similarity, gap = _score_identity_features(
        record,
        real_prototype,
        generated_prototype,
        negative_prototypes,
    )
    prototype = np.asarray(record["prototype"], dtype=np.float32)
    return np.asarray(
        [
            _cosine(prototype, real_prototype),
            _cosine(prototype, generated_prototype),
            positive_similarity,
            negative_similarity,
            gap,
            float(record.get("frame_consistency", 0.0)),
            float(record.get("valid_frame_ratio", 0.0)),
            float(record.get("quality_weight_mean", 0.0)),
        ],
        dtype=np.float64,
    )


def _fit_logistic_calibrator(
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Fit a small dependency-free logistic calibrator for open-set scores."""
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != target.shape[0]:
        raise ValueError("Identity calibration features and labels must align.")
    if matrix.shape[0] < 4 or len(np.unique(target)) < 2:
        raise ValueError("Identity calibration needs both positive and negative rows.")

    location = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.maximum(scale, 1e-4)
    standardized = (matrix - location) / scale
    design = np.concatenate(
        [np.ones((standardized.shape[0], 1), dtype=np.float64), standardized],
        axis=1,
    )
    weights = np.zeros(design.shape[1], dtype=np.float64)
    learning_rate = 0.20
    regularization = 0.01
    for _ in range(800):
        logits = np.clip(design @ weights, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = (design.T @ (probabilities - target)) / len(target)
        gradient[1:] += regularization * weights[1:]
        weights -= learning_rate * gradient
        learning_rate *= 0.998
    return {
        "feature_names": list(IDENTITY_FEATURE_NAMES),
        "location": location.tolist(),
        "scale": scale.tolist(),
        "intercept": float(weights[0]),
        "coefficients": weights[1:].tolist(),
        "training_rows": int(matrix.shape[0]),
    }


def _calibrator_probability(
    feature: np.ndarray,
    calibrator: dict[str, Any] | None,
    *,
    fallback_gap: float | None = None,
) -> float:
    if not calibrator:
        gap = float(fallback_gap or 0.0)
        return float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, gap * 8.0)))))
    location = np.asarray(calibrator.get("location", []), dtype=np.float64)
    scale = np.maximum(
        np.asarray(calibrator.get("scale", []), dtype=np.float64),
        1e-4,
    )
    coefficients = np.asarray(
        calibrator.get("coefficients", []),
        dtype=np.float64,
    )
    vector = np.asarray(feature, dtype=np.float64).reshape(-1)
    if (
        location.shape != vector.shape
        or scale.shape != vector.shape
        or coefficients.shape != vector.shape
    ):
        gap = float(fallback_gap or 0.0)
        return float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, gap * 8.0)))))
    standardized = (vector - location) / scale
    logit = float(calibrator.get("intercept", 0.0)) + float(
        standardized @ coefficients
    )
    return float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit)))))


def _identity_calibration_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | None]:
    target = np.asarray(labels, dtype=np.int32).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(np.sum(target == 1))
    negatives = int(np.sum(target == 0))
    if len(values) == 0 or positives == 0 or negatives == 0:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "eer": None,
            "recall_at_1pct_fpr": None,
            "recall_at_5pct_fpr": None,
        }

    order = np.argsort(-values, kind="mergesort")
    sorted_labels = target[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    tpr = tp / positives
    fpr = fp / negatives
    roc_x = np.concatenate([[0.0], fpr, [1.0]])
    roc_y = np.concatenate([[0.0], tpr, [1.0]])
    roc_auc = float(np.trapz(roc_y, roc_x))

    precision = tp / np.maximum(tp + fp, 1)
    recall = tpr
    previous_recall = np.concatenate([[0.0], recall[:-1]])
    pr_auc = float(np.sum((recall - previous_recall) * precision))

    fnr = 1.0 - tpr
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)

    def recall_at_fpr(target_fpr: float) -> float:
        valid = recall[fpr <= target_fpr]
        return float(np.max(valid)) if valid.size else 0.0

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "eer": eer,
        "recall_at_1pct_fpr": recall_at_fpr(0.01),
        "recall_at_5pct_fpr": recall_at_fpr(0.05),
    }


def build_identity_profile(
    *,
    real_root: str | Path,
    generated_root: str | Path,
    negative_root: str | Path,
    output_path: str | Path,
    device: str = "cpu",
    max_frames: int = 8,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build an open-set identity profile from videos without manual labels."""
    from ..core.holistic_evaluator import _FaceDetector, _IdentityBackend

    real_paths = sorted(Path(real_root).rglob("*.mp4"))
    generated_paths = sorted(Path(generated_root).rglob("*.mp4"))
    negative_paths = sorted(Path(negative_root).rglob("*.mp4"))
    if limit is not None:
        real_paths = _limit_paths_balanced(real_paths, limit)
        generated_paths = _limit_paths_balanced(generated_paths, limit)
        negative_paths = _limit_paths_balanced(
            negative_paths,
            max(1, limit),
        )

    backend = _IdentityBackend(_FaceDetector(), device=device)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for label, paths in (
        ("wangxing_real", real_paths),
        ("wangxing_generated", generated_paths),
        ("negative", negative_paths),
    ):
        for path in paths:
            try:
                record = _video_record(path, label, backend, max_frames)
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append({"path": str(path), "reason": str(exc)})
                continue
            if record is None:
                failures.append({"path": str(path), "reason": "no usable face embedding"})
                continue
            records.append(record)

    real_records = [record for record in records if record["label"] == "wangxing_real"]
    generated_records = [
        record for record in records if record["label"] == "wangxing_generated"
    ]
    negative_records = [record for record in records if record["label"] == "negative"]
    if not real_records or not generated_records or not negative_records:
        raise ValueError(
            "Identity profile needs real Wang Xing, generated Wang Xing and "
            "negative video embeddings."
        )

    real_prototype = _profile_prototype(real_records)
    generated_prototype = _profile_prototype(generated_records)
    negative_groups: dict[str, list[dict[str, Any]]] = {}
    for record in negative_records:
        path = Path(record["path"])
        group = path.parent.name or "negative"
        negative_groups.setdefault(group, []).append(record)
    negative_prototypes = [
        _profile_prototype(group_records)
        for group_records in negative_groups.values()
    ]

    feature_rows: list[np.ndarray] = []
    feature_labels: list[int] = []
    calibrated_positive_scores: list[float] = []
    calibrated_negative_scores: list[float] = []
    positive_gaps: list[float] = []
    negative_gaps: list[float] = []
    for record in real_records + generated_records:
        feature = _identity_feature_vector(
            record,
            real_prototype,
            generated_prototype,
            negative_prototypes,
        )
        feature_rows.append(feature)
        feature_labels.append(1)
        positive_gaps.append(float(feature[4]))
    for record in negative_records:
        feature = _identity_feature_vector(
            record,
            real_prototype,
            generated_prototype,
            negative_prototypes,
        )
        feature_rows.append(feature)
        feature_labels.append(0)
        negative_gaps.append(float(feature[4]))

    calibrator = _fit_logistic_calibrator(
        np.stack(feature_rows),
        np.asarray(feature_labels, dtype=np.int32),
    )
    for feature, label in zip(feature_rows, feature_labels):
        probability = _calibrator_probability(feature, calibrator)
        if label:
            calibrated_positive_scores.append(probability)
        else:
            calibrated_negative_scores.append(probability)
    positive_gap_floor = _quantile(np.asarray(positive_gaps), 0.05)
    negative_gap_ceiling = _quantile(np.asarray(negative_gaps), 0.95)
    positive_probability_floor = _quantile(
        np.asarray(calibrated_positive_scores),
        0.05,
    )
    negative_probability_ceiling = _quantile(
        np.asarray(calibrated_negative_scores),
        0.95,
    )
    midpoint = (positive_probability_floor + negative_probability_ceiling) / 2.0
    scale = max(
        abs(positive_probability_floor - negative_probability_ceiling) / 4.0,
        0.02,
    )
    calibration_metrics = _identity_calibration_metrics(
        np.asarray(feature_labels, dtype=np.int32),
        np.asarray(
            [
                _calibrator_probability(feature, calibrator)
                for feature in feature_rows
            ],
            dtype=np.float64,
        ),
    )
    profile = {
        "schema_version": IDENTITY_PROFILE_SCHEMA,
        "specialization_schema": SPECIALIZATION_SCHEMA,
        "evaluator_version": SPECIALIZATION_EVALUATOR_VERSION,
        "backend": backend.backend,
        "identity_feature_names": list(IDENTITY_FEATURE_NAMES),
        "real_prototype": real_prototype.tolist(),
        "generated_prototype": generated_prototype.tolist(),
        "negative_prototypes": [
            {
                "name": name,
                "prototype": prototype.tolist(),
                "sample_count": len(negative_groups[name]),
            }
            for name, prototype in zip(negative_groups, negative_prototypes)
        ],
        "records": records,
        "calibrator": calibrator,
        "thresholds": {
            "positive_gap_floor": positive_gap_floor,
            "negative_gap_ceiling": negative_gap_ceiling,
            "positive_probability_floor": positive_probability_floor,
            "negative_probability_ceiling": negative_probability_ceiling,
            "probability_midpoint": midpoint,
            "probability_scale": scale,
            "min_valid_frame_count": 3,
            "min_valid_frame_ratio": 0.35,
            "min_frame_consistency": 0.70,
        },
        "calibration": {
            "positive_count": len(real_records) + len(generated_records),
            "negative_count": len(negative_records),
            "positive_gap_p05": positive_gap_floor,
            "positive_gap_median": float(np.median(positive_gaps)),
            "negative_gap_p95": negative_gap_ceiling,
            "negative_gap_median": float(np.median(negative_gaps)),
            "positive_probability_p05": positive_probability_floor,
            "negative_probability_p95": negative_probability_ceiling,
            "metrics": calibration_metrics,
        },
        "calibration_metrics": calibration_metrics,
        "provenance": {
            "real_root": str(real_root),
            "generated_root": str(generated_root),
            "negative_root": str(negative_root),
            "negative_sources": sorted(negative_groups),
            "failed_count": len(failures),
            "failed_preview": failures[:50],
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return profile


def _identity_record_features(
    embeddings: np.ndarray,
    profile: dict[str, Any],
    *,
    quality_weights: Iterable[float] | None = None,
    valid_frame_ratio: float = 1.0,
) -> dict[str, Any]:
    real_prototype = np.asarray(profile["real_prototype"], dtype=np.float32)
    generated_prototype = np.asarray(
        profile["generated_prototype"],
        dtype=np.float32,
    )
    negative_prototypes = [
        np.asarray(item["prototype"], dtype=np.float32)
        for item in profile.get("negative_prototypes", [])
    ]
    frame_similarities = np.asarray(
        [
            0.65 * _cosine(embedding, real_prototype)
            + 0.35 * _cosine(embedding, generated_prototype)
            for embedding in embeddings
        ],
        dtype=np.float32,
    )
    weights = (
        list(quality_weights)
        if quality_weights is not None
        else [1.0] * len(embeddings)
    )
    prototype = _weighted_prototype(embeddings, weights)
    positive_similarity, negative_similarity, gap = _score_identity_features(
        {"prototype": prototype.tolist()},
        real_prototype,
        generated_prototype,
        negative_prototypes,
    )
    feature = _identity_feature_vector(
        {
            "prototype": prototype.tolist(),
            "frame_consistency": float(
                np.average(embeddings @ prototype, weights=weights)
            ),
            "valid_frame_ratio": valid_frame_ratio,
            "quality_weight_mean": float(np.mean(weights)),
        },
        real_prototype,
        generated_prototype,
        negative_prototypes,
    )
    probability = _calibrator_probability(
        feature,
        profile.get("calibrator"),
        fallback_gap=gap,
    )
    consistency = float(np.average(embeddings @ prototype, weights=weights))
    return {
        "probability_0_1": probability,
        "negative_class_probability_0_1": 1.0 - probability,
        "positive_similarity": positive_similarity,
        "real_prototype_similarity": _cosine(prototype, real_prototype),
        "generated_prototype_similarity": _cosine(prototype, generated_prototype),
        "negative_similarity": negative_similarity,
        "gap": gap,
        "quality_weight_mean": float(np.mean(weights)),
        "frame_consistency": consistency,
        "frame_similarity_median": float(np.median(frame_similarities)),
        "frame_similarity_p10": _quantile(frame_similarities, 0.10),
    }


def evaluate_identity_profile(
    video_path: str | Path,
    profile: dict[str, Any],
    backend: Any,
    *,
    max_frames: int = 16,
) -> dict[str, Any]:
    embeddings, metadata = _identity_frame_embeddings(
        video_path,
        backend,
        max_frames,
    )
    if embeddings.size == 0:
        return {
            "status": "uncertain",
            "decision": "uncertain",
            "probability_0_1": None,
            "backend": metadata["backend"],
            "valid_frame_count": 0,
            "valid_frame_ratio": metadata["valid_frame_ratio"],
            "frame_consistency": None,
            "uncertainty_reasons": ["no_face_embedding"],
        }
    scores = _identity_record_features(
        embeddings,
        profile,
        quality_weights=metadata.get("quality_weights"),
        valid_frame_ratio=float(metadata["valid_frame_ratio"]),
    )
    thresholds = profile.get("thresholds", {})
    reasons: list[str] = []
    if metadata["valid_frame_count"] < int(thresholds.get("min_valid_frame_count", 3)):
        reasons.append("too_few_face_frames")
    if metadata["valid_frame_ratio"] < float(
        thresholds.get("min_valid_frame_ratio", 0.35)
    ):
        reasons.append("low_face_frame_ratio")
    if scores["frame_consistency"] < float(
        thresholds.get("min_frame_consistency", 0.70)
    ):
        reasons.append("low_face_consistency")

    positive_floor = float(thresholds.get("positive_gap_floor", 0.0))
    negative_ceiling = float(thresholds.get("negative_gap_ceiling", 0.0))
    positive_probability_floor = float(
        thresholds.get(
            "positive_probability_floor",
            0.5,
        )
    )
    negative_probability_ceiling = float(
        thresholds.get(
            "negative_probability_ceiling",
            0.5,
        )
    )
    if reasons:
        decision = "uncertain"
    elif (
        scores["probability_0_1"] >= positive_probability_floor
        and scores["gap"] >= positive_floor
    ):
        decision = "wangxing"
    elif (
        scores["probability_0_1"] <= negative_probability_ceiling
        and scores["gap"] <= negative_ceiling
    ):
        decision = "not_wangxing"
    else:
        decision = "uncertain"
        reasons.append("identity_margin_small")
    return {
        "status": "available",
        "decision": decision,
        "probability_0_1": scores["probability_0_1"],
        "negative_class_probability_0_1": scores[
            "negative_class_probability_0_1"
        ],
        "backend": metadata["backend"],
        "valid_frame_count": metadata["valid_frame_count"],
        "frame_count": metadata["frame_count"],
        "valid_frame_ratio": metadata["valid_frame_ratio"],
        "quality_weight_mean": scores["quality_weight_mean"],
        "frame_consistency": scores["frame_consistency"],
        "positive_similarity": scores["positive_similarity"],
        "real_prototype_similarity": scores["real_prototype_similarity"],
        "generated_prototype_similarity": scores["generated_prototype_similarity"],
        "negative_similarity": scores["negative_similarity"],
        "identity_gap": scores["gap"],
        "frame_similarity_median": scores["frame_similarity_median"],
        "frame_similarity_p10": scores["frame_similarity_p10"],
        "uncertainty_reasons": reasons,
        "thresholds": {
            "positive_gap_floor": positive_floor,
            "negative_gap_ceiling": negative_ceiling,
            "positive_probability_floor": positive_probability_floor,
            "negative_probability_ceiling": negative_probability_ceiling,
            "min_valid_frame_count": int(
                thresholds.get("min_valid_frame_count", 3)
            ),
            "min_valid_frame_ratio": float(
                thresholds.get("min_valid_frame_ratio", 0.35)
            ),
            "min_frame_consistency": float(
                thresholds.get("min_frame_consistency", 0.70)
            ),
        },
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def score_expression_profile(
    au_path: str | Path,
    profile: dict[str, Any],
    *,
    expected_class: str | None = None,
) -> dict[str, Any]:
    vector, quality = extract_sequence_features(au_path)
    class_scores: dict[str, dict[str, Any]] = {}
    for expression_class, model in profile.get("classes", {}).items():
        location = np.asarray(model.get("location", []), dtype=np.float64)
        scale = np.asarray(model.get("scale", []), dtype=np.float64)
        if location.shape != vector.shape or scale.shape != vector.shape:
            continue
        distance = _robust_distance(vector, location, scale)
        threshold = float(model.get("distance_threshold", 1.0))
        class_scores[expression_class] = {
            "score_0_1": _safe_score(distance, threshold),
            "distance": distance,
            "distance_threshold": threshold,
            "display_name": model.get(
                "display_name",
                EXPRESSION_DISPLAY_NAMES.get(expression_class, expression_class),
            ),
            "sample_count": model.get("sample_count"),
        }
    if not class_scores:
        return {
            "status": "unavailable",
            "decision": "uncertain",
            "compatibility_0_1": None,
            "expression_compatibility_0_1": None,
            "class_scores": {},
            "top_profiles": [],
            "most_compatible_profiles": [],
            "profile_winner": None,
            "event_statistics": quality["event_statistics"],
            "quality": quality,
            "uncertainty_reasons": ["expression_profile_unavailable"],
        }

    ranked = sorted(
        class_scores.items(),
        key=lambda item: item[1]["score_0_1"],
        reverse=True,
    )
    profile_winner = ranked[0][0]
    profile_winner_score = ranked[0][1]["score_0_1"]
    selected_class = (
        expected_class if expected_class in class_scores else profile_winner
    )
    selected_score = class_scores[selected_class]["score_0_1"]
    top_score = ranked[0][1]["score_0_1"]
    second_score = ranked[1][1]["score_0_1"] if len(ranked) > 1 else 0.0
    margin = top_score - second_score
    reasons: list[str] = []
    if quality["valid_frame_ratio"] < 0.35:
        reasons.append("low_face_mesh_coverage")
    if top_score < 0.35:
        reasons.append("profile_distance_large")
    if margin < 0.05:
        reasons.append("expression_margin_small")
    if expected_class in class_scores and selected_score < 0.35:
        reasons.append("expected_expression_profile_mismatch")
    if quality["valid_frame_ratio"] < 0.35:
        decision = "uncertain"
    elif top_score < 0.35 or "expected_expression_profile_mismatch" in reasons:
        decision = "incompatible"
    else:
        decision = "compatible"
    return {
        "status": "available",
        "decision": decision,
        "compatibility_0_1": float(top_score),
        "expression_compatibility_0_1": float(top_score),
        "selected_profile": selected_class,
        "selected_profile_display_name": class_scores[selected_class]["display_name"],
        "profile_winner": profile_winner,
        "profile_winner_display_name": class_scores[profile_winner][
            "display_name"
        ],
        "profile_winner_score_0_1": float(profile_winner_score),
        "expected_profile_score_0_1": float(selected_score)
        if expected_class in class_scores
        else None,
        "top_profiles": [
            {
                "class": expression_class,
                **score,
            }
            for expression_class, score in ranked[:2]
        ],
        "most_compatible_profiles": [
            {
                "class": expression_class,
                "score_0_1": float(score["score_0_1"]),
                "display_name": score["display_name"],
            }
            for expression_class, score in ranked[:2]
        ],
        "class_scores": class_scores,
        "margin_0_1": float(margin),
        "emotion_uncertain": bool(margin < 0.05),
        "expression_class_uncertain": bool(margin < 0.05),
        "expected_profile": expected_class,
        "compatibility_basis": (
            "max_similarity_to_real_wangxing_expression_support_domain"
        ),
        "event_statistics": quality["event_statistics"],
        "quality": quality,
        "uncertainty_reasons": reasons,
        "severe_deviation": bool(
            selected_score < 0.25 or (
                decision == "incompatible" and margin >= 0.10
            )
        ),
    }


def evaluate_specialization(
    *,
    video_path: str | Path,
    au_path: str | Path,
    identity_profile_path: str | Path,
    expression_profile_path: str | Path,
    source_profile_path: str | Path | None = None,
    expected_class: str | None = None,
    device: str = "cpu",
    max_identity_frames: int = 16,
) -> dict[str, Any]:
    from ..core.holistic_evaluator import _FaceDetector, _IdentityBackend

    identity_profile = _load_json(identity_profile_path)
    expression_profile = _load_json(expression_profile_path)
    backend = _IdentityBackend(_FaceDetector(), device=device)
    identity = evaluate_identity_profile(
        video_path,
        identity_profile,
        backend,
        max_frames=max_identity_frames,
    )
    expression: dict[str, Any]
    if identity["decision"] == "wangxing":
        expression = score_expression_profile(
            au_path,
            expression_profile,
            expected_class=expected_class,
        )
    else:
        expression = {
            "status": "not_evaluated",
            "decision": "not_evaluated_due_to_identity",
            "compatibility_0_1": None,
            "expression_compatibility_0_1": None,
            "top_profiles": [],
            "most_compatible_profiles": [],
            "profile_winner": None,
            "event_statistics": None,
            "quality": None,
            "uncertainty_reasons": ["identity_gate_not_passed"],
        }
    source: dict[str, Any] = {
        "status": "not_available",
        "decision": "not_evaluated",
        "reason": "source_profile_not_supplied",
    }
    if source_profile_path is not None:
        source_path = Path(source_profile_path)
        if source_path.is_file():
            source = score_source_profile(au_path, _load_json(source_path))
            source["role"] = (
                "secondary_real_vs_generated_domain_evidence; "
                "not_used_for_identity_gate"
            )
        else:
            source = {
                "status": "unavailable",
                "decision": "uncertain",
                "reason": "source_profile_missing",
                "path": str(source_path),
            }
    if identity["decision"] == "not_wangxing":
        final_decision = "not_wangxing"
    elif identity["decision"] == "uncertain":
        final_decision = "uncertain_identity"
    elif expression["decision"] == "compatible":
        final_decision = "wangxing_expression_compatible"
    elif expression["decision"] == "incompatible":
        final_decision = "wangxing_expression_incompatible"
    else:
        final_decision = "uncertain_expression"

    return {
        "status": "available",
        "schema_version": SPECIALIZATION_SCHEMA,
        "evaluator_version": SPECIALIZATION_EVALUATOR_VERSION,
        "identity": identity,
        "expression_profile": expression,
        "source": source,
        "decision": final_decision,
        "decision_policy": (
            "Identity is gated before expression compatibility. Non-Wang Xing "
            "or uncertain identity does not receive an expression conclusion."
        ),
        "scope": "wangxing_specialization_only",
        "normal_evaluation_unchanged": True,
        "sources": {
            "identity_profile": str(identity_profile_path),
            "expression_profile": str(expression_profile_path),
            "source_profile": (
                str(source_profile_path)
                if source_profile_path is not None
                else None
            ),
            "generated_au": str(au_path),
            "generated_video": str(video_path),
        },
    }
