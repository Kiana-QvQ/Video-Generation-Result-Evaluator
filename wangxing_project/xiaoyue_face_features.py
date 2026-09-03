"""Face-only temporal features for the isolated XiaoYue experiment.

This module deliberately excludes full-frame RGB/HSV/gray descriptors. It
uses pose-normalized Face Mesh geometry, AU trajectories, and grayscale local
face crops anchored by the AU landmarks. The mouth sequence is exposed as a
separate modality so the classifier cannot hide it inside a global descriptor.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from evaluator.modules.core.face_landmarker import normalize_csv_landmark_frame
from evaluator.modules.core.paths import project_path
from evaluator.vedio_pred.real_video_detector import _file_signature

from .face_crop_temporal import extract_face_crop_temporal_features

MAX_FRAMES = 24

AU_INTENSITY_IDS = (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)
AU_PRESENCE_IDS = (1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24)
MOUTH_INTENSITY_IDS = (12, 15, 17, 20, 23, 24, 25, 26)
MOUTH_PRESENCE_IDS = (12, 14, 15, 17, 23, 24)

GEOMETRY_NAMES = (
    "mouth_open",
    "mouth_width",
    "mouth_corner_balance",
    "mouth_center_y",
    "jaw_drop",
    "lip_nose_distance",
    "left_eye_open",
    "right_eye_open",
    "eye_asymmetry",
    "left_brow_eye_gap",
    "right_brow_eye_gap",
    "eye_mouth_sync_proxy",
)

FACE_SEQUENCE_DIM = (
    len(GEOMETRY_NAMES)
    + len(AU_INTENSITY_IDS)
    + len(AU_PRESENCE_IDS)
    + 5 * 5
    + 2
)
MOUTH_SEQUENCE_DIM = (
    6
    + len(MOUTH_INTENSITY_IDS)
    + len(MOUTH_PRESENCE_IDS)
    + 5
    + 2
)


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _sample_rows(
    rows: list[dict[str, str]],
    max_frames: int,
) -> list[dict[str, str]]:
    if not rows:
        return []
    if len(rows) <= max_frames:
        return rows
    indexes = np.linspace(0, len(rows) - 1, max_frames).round().astype(int)
    return [rows[int(index)] for index in indexes]


def _points_from_row(
    row: dict[str, str],
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    points: dict[int, np.ndarray] = {}
    z_values: dict[int, float] = {}
    for index in range(478):
        x = _finite(row.get(f"lm_mp_{index}_x"))
        y = _finite(row.get(f"lm_mp_{index}_y"))
        z = _finite(row.get(f"lm_mp_{index}_z"), 0.0)
        if math.isfinite(x) and math.isfinite(y):
            points[index] = np.asarray([x, y], dtype=np.float32)
            z_values[index] = float(z)
    return points, z_values


def _distance(points: dict[int, np.ndarray], left: int, right: int) -> float:
    if left not in points or right not in points:
        return math.nan
    return float(np.linalg.norm(points[left] - points[right]))


def _mean_point(
    points: dict[int, np.ndarray],
    indexes: Sequence[int],
) -> np.ndarray | None:
    available = [points[index] for index in indexes if index in points]
    if not available:
        return None
    return np.mean(np.stack(available), axis=0)


def _geometry_from_row(row: dict[str, str]) -> tuple[np.ndarray, bool]:
    raw_points, z_values = _points_from_row(row)
    normalized = normalize_csv_landmark_frame(raw_points, points_z=z_values)
    if not normalized:
        return np.zeros(len(GEOMETRY_NAMES), dtype=np.float32), False

    face_width = _distance(normalized, 33, 263)
    face_height = _distance(normalized, 10, 152)
    if not math.isfinite(face_width) or not math.isfinite(face_height):
        return np.zeros(len(GEOMETRY_NAMES), dtype=np.float32), False
    face_width = max(face_width, 1e-5)
    face_height = max(face_height, 1e-5)

    mouth_center = _mean_point(normalized, (13, 14))
    nose = normalized.get(1)
    if mouth_center is None or nose is None:
        return np.zeros(len(GEOMETRY_NAMES), dtype=np.float32), False

    mouth_open = _distance(normalized, 13, 14) / face_height
    mouth_width = _distance(normalized, 61, 291) / face_width
    corner_balance = (
        float(normalized[61][1] - normalized[291][1]) / face_height
        if 61 in normalized and 291 in normalized
        else 0.0
    )
    mouth_center_y = float(mouth_center[1] - nose[1]) / face_height
    jaw_drop = _distance(normalized, 1, 152) / face_height
    lip_nose_distance = float(np.linalg.norm(mouth_center - nose)) / face_height
    left_eye = _distance(normalized, 159, 145) / face_height
    right_eye = _distance(normalized, 386, 374) / face_height
    eye_asymmetry = abs(left_eye - right_eye)
    left_brow_gap = _distance(normalized, 70, 159) / face_height
    right_brow_gap = _distance(normalized, 300, 386) / face_height
    eye_mouth_sync = (left_eye + right_eye) * 0.5 - mouth_open
    values = np.asarray(
        [
            mouth_open,
            mouth_width,
            corner_balance,
            mouth_center_y,
            jaw_drop,
            lip_nose_distance,
            left_eye,
            right_eye,
            eye_asymmetry,
            left_brow_gap,
            right_brow_gap,
            eye_mouth_sync,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), True


def _au_row(
    row: dict[str, str],
    ids: Sequence[int],
    suffix: str,
) -> np.ndarray:
    values: list[float] = []
    for au_id in ids:
        value = _finite(row.get(f"au_{au_id}_{suffix}"), 0.0)
        values.append(float(np.clip(value, 0.0, 1.0)))
    return np.asarray(values, dtype=np.float32)


def _quality_row(row: dict[str, str], geometry_valid: bool) -> np.ndarray:
    detection = _finite(row.get("face_detection_score"), 0.5)
    return np.asarray(
        [
            float(geometry_valid),
            float(np.clip(detection, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def extract_face_sequences(
    video_path: str | Path,
    au_path: str | Path,
    *,
    max_frames: int = MAX_FRAMES,
) -> dict[str, Any]:
    """Extract face and mouth sequences without full-frame appearance input."""
    with Path(au_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = _sample_rows(list(csv.DictReader(handle)), max_frames)
    if len(rows) < 2:
        raise ValueError(f"AU sequence is too short: {au_path}")

    crop_sequence, crop_summary = extract_face_crop_temporal_features(
        video_path=video_path,
        au_path=au_path,
        max_frames=max_frames,
        frame_size=512,
    )
    crop_sequence = np.asarray(crop_sequence, dtype=np.float32).reshape(
        max_frames,
        5,
        6,
    )
    # Drop absolute crop brightness. Keep local contrast, edges, Laplacian,
    # residual and validity so lighting/background cannot dominate the model.
    crop_sequence = crop_sequence[:, :, 1:]

    geometry_rows: list[np.ndarray] = []
    geometry_valid: list[bool] = []
    face_rows: list[np.ndarray] = []
    mouth_rows: list[np.ndarray] = []
    for index, row in enumerate(rows[:max_frames]):
        geometry, valid = _geometry_from_row(row)
        geometry_rows.append(geometry)
        geometry_valid.append(valid)
        intensity = _au_row(row, AU_INTENSITY_IDS, "intensity")
        presence = _au_row(row, AU_PRESENCE_IDS, "presence")
        quality = _quality_row(row, valid)
        local = crop_sequence[min(index, len(crop_sequence) - 1)]
        face_rows.append(
            np.concatenate(
                [geometry, intensity, presence, local.reshape(-1), quality]
            )
        )

        mouth_geometry = geometry[:6]
        mouth_intensity = _au_row(row, MOUTH_INTENSITY_IDS, "intensity")
        mouth_presence = _au_row(row, MOUTH_PRESENCE_IDS, "presence")
        mouth_local = local[2]
        mouth_rows.append(
            np.concatenate(
                [
                    mouth_geometry,
                    mouth_intensity,
                    mouth_presence,
                    mouth_local,
                    quality,
                ]
            )
        )

    face = np.stack(face_rows).astype(np.float32)
    mouth = np.stack(mouth_rows).astype(np.float32)
    if len(face) < max_frames:
        face_padding = np.repeat(face[-1:], max_frames - len(face), axis=0)
        mouth_padding = np.repeat(mouth[-1:], max_frames - len(mouth), axis=0)
        face_padding[:, -2:] = 0.0
        mouth_padding[:, -2:] = 0.0
        face = np.concatenate([face, face_padding], axis=0)
        mouth = np.concatenate([mouth, mouth_padding], axis=0)

    if face.shape != (max_frames, FACE_SEQUENCE_DIM):
        raise RuntimeError(
            f"Face feature layout mismatch: {face.shape} != "
            f"({max_frames}, {FACE_SEQUENCE_DIM})"
        )
    if mouth.shape != (max_frames, MOUTH_SEQUENCE_DIM):
        raise RuntimeError(
            f"Mouth feature layout mismatch: {mouth.shape} != "
            f"({max_frames}, {MOUTH_SEQUENCE_DIM})"
        )
    return {
        "face": np.nan_to_num(face, nan=0.0, posinf=0.0, neginf=0.0),
        "mouth": np.nan_to_num(mouth, nan=0.0, posinf=0.0, neginf=0.0),
        "mouth_summary": np.asarray(crop_summary, dtype=np.float32),
        "geometry_valid_ratio": float(np.mean(geometry_valid)),
        "mouth_quality_ratio": float(np.mean(mouth[:, -2])),
        "feature_source": (
            "pose-normalized AU/Face Mesh sequences + "
            "brightness-invariant local face crops"
        ),
    }


def _all_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(manifest.get("pairs"), dict):
        pairs = manifest["pairs"]
        return [
            *list(pairs.get("train", {}).get("real") or []),
            *list(pairs.get("train", {}).get("fake") or []),
            *list(pairs.get("test", {}).get("real") or []),
            *list(pairs.get("test", {}).get("fake") or []),
        ]
    return [*list(manifest.get("real") or []), *list(manifest.get("fake") or [])]


def build_feature_table(
    manifest: dict[str, Any],
    *,
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract all manifest samples, reusing a signature-checked NPZ cache."""
    items = _all_items(manifest)
    paths = [str(project_path(str(item["video"])).resolve()) for item in items]
    signatures = [
        _file_signature(project_path(str(item["video"])).resolve())
        for item in items
    ]
    cache = Path(cache_path).expanduser().resolve() if cache_path else None
    if cache is not None and cache.is_file():
        try:
            with np.load(str(cache), allow_pickle=False) as payload:
                cached_paths = [str(value) for value in payload["paths"].tolist()]
                cached_signatures = [
                    str(value) for value in payload["signatures"].tolist()
                ]
                cached_indexes = {
                    path: index for index, path in enumerate(cached_paths)
                }
                cache_matches = all(
                    path in cached_indexes
                    and cached_signatures[cached_indexes[path]]
                    == signature
                    for path, signature in zip(paths, signatures)
                )
                if cache_matches:
                    features = {
                        path: {
                            "face": payload["face"][cached_indexes[path]].astype(
                                np.float32
                            ),
                            "mouth": payload["mouth"][
                                cached_indexes[path]
                            ].astype(np.float32),
                            "mouth_summary": payload["mouth_summary"][
                                cached_indexes[path]
                            ].astype(np.float32),
                            "geometry_valid_ratio": float(
                                payload["geometry_valid_ratio"][
                                    cached_indexes[path]
                                ]
                            ),
                            "mouth_quality_ratio": float(
                                payload["mouth_quality_ratio"][
                                    cached_indexes[path]
                                ]
                            ),
                            "feature_source": "cache",
                        }
                        for path in paths
                    }
                    return {
                        "items": items,
                        "features": features,
                        "paths": paths,
                        "signatures": signatures,
                    }
        except (OSError, KeyError, ValueError):
            pass

    features: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(items, start=1):
        video = project_path(str(item["video"])).resolve()
        au = project_path(str(item["au"])).resolve()
        features[str(video)] = extract_face_sequences(video, au)
        if index % 5 == 0 or index == len(items):
            print(f"[face feature] {index}/{len(items)}", flush=True)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        ordered = [features[path] for path in paths]
        np.savez_compressed(
            str(cache),
            paths=np.asarray(paths),
            signatures=np.asarray(signatures),
            face=np.stack([item["face"] for item in ordered]),
            mouth=np.stack([item["mouth"] for item in ordered]),
            mouth_summary=np.stack(
                [item["mouth_summary"] for item in ordered]
            ),
            geometry_valid_ratio=np.asarray(
                [item["geometry_valid_ratio"] for item in ordered],
                dtype=np.float32,
            ),
            mouth_quality_ratio=np.asarray(
                [item["mouth_quality_ratio"] for item in ordered],
                dtype=np.float32,
            ),
        )
    return {
        "items": items,
        "features": features,
        "paths": paths,
        "signatures": signatures,
    }


def sequence_summary(face: np.ndarray, mouth: np.ndarray) -> np.ndarray:
    """Create a compact profile vector from face and mouth trajectories."""
    values: list[np.ndarray] = []
    for sequence in (face, mouth):
        sequence = np.asarray(sequence, dtype=np.float32)
        velocity = np.diff(sequence, axis=0)
        values.extend(
            [
                np.mean(sequence, axis=0),
                np.std(sequence, axis=0),
                np.quantile(np.abs(velocity), 0.95, axis=0)
                if len(velocity)
                else np.zeros(sequence.shape[1], dtype=np.float32),
            ]
        )
    return np.concatenate(values).astype(np.float32)


def feature_layout() -> dict[str, int]:
    return {
        "max_frames": MAX_FRAMES,
        "face_sequence_dim": FACE_SEQUENCE_DIM,
        "mouth_sequence_dim": MOUTH_SEQUENCE_DIM,
        "mouth_summary_dim": 8,
        "face_summary_dim": FACE_SEQUENCE_DIM * 3,
        "mouth_profile_dim": MOUTH_SEQUENCE_DIM * 3,
    }


def json_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
