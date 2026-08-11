"""Synthesize Wang Xing-compatible AU CSVs from video frames.

When the host app uploads a video without a side-car AU table, this module
extracts MediaPipe blendshapes / mesh landmarks and writes a temporary CSV in
the same schema expected by ``score_expression_profile`` and facial_motion.
"""

from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# FACS intensity IDs used by wangxing_specialization / facial_motion.
AU_IDS = (
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

# Landmarks required by expression geometry features (+ jaw width anchors).
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
    172,
    234,
    263,
    291,
    334,
    362,
    374,
    386,
    397,
    454,
)

# MediaPipe Face Landmarker blendshape → AU intensity (approximate).
BLENDSHAPE_TO_AU: dict[int, tuple[str, ...]] = {
    1: ("browInnerUp",),
    2: ("browOuterUpLeft", "browOuterUpRight"),
    4: ("browDownLeft", "browDownRight"),
    5: ("eyeWideLeft", "eyeWideRight"),
    6: ("cheekSquintLeft", "cheekSquintRight"),
    7: ("eyeSquintLeft", "eyeSquintRight"),
    9: ("noseSneerLeft", "noseSneerRight"),
    10: ("mouthUpperUpLeft", "mouthUpperUpRight"),
    12: ("mouthSmileLeft", "mouthSmileRight"),
    14: ("mouthDimpleLeft", "mouthDimpleRight"),
    15: ("mouthFrownLeft", "mouthFrownRight"),
    17: ("mouthShrugLower", "jawForward"),
    20: ("mouthStretchLeft", "mouthStretchRight"),
    23: ("mouthPressLeft", "mouthPressRight"),
    24: ("mouthPressLeft", "mouthPressRight"),
    25: ("jawOpen", "mouthClose"),
    26: ("jawOpen",),
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _mean_blend(
    blendshapes: dict[str, float],
    names: Sequence[str],
) -> float:
    values = [float(blendshapes.get(name, 0.0)) for name in names]
    if not values:
        return 0.0
    return _clamp(float(np.mean(values)))


def _au_from_blendshapes(blendshapes: dict[str, float]) -> dict[int, float]:
    intensities: dict[int, float] = {}
    for au_id, names in BLENDSHAPE_TO_AU.items():
        if au_id == 25:
            # lips part ≈ jaw opening and not fully closed
            jaw = float(blendshapes.get("jawOpen", 0.0))
            closed = float(blendshapes.get("mouthClose", 0.0))
            intensities[au_id] = _clamp(max(jaw, 1.0 - closed) * 0.85)
            continue
        intensities[au_id] = _mean_blend(blendshapes, names)
    return intensities


def _au_from_landmarks(landmarks: np.ndarray) -> dict[int, float]:
    """Geometry proxy when blendshapes are unavailable (Face Mesh fallback)."""
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 455:
        return {au_id: 0.0 for au_id in AU_IDS}
    face_w = float(np.linalg.norm(points[454, :2] - points[234, :2])) + 1e-6
    face_h = float(np.linalg.norm(points[152, :2] - points[10, :2])) + 1e-6
    mouth_open = float(np.linalg.norm(points[14, :2] - points[13, :2])) / face_h
    mouth_width = float(np.linalg.norm(points[291, :2] - points[61, :2])) / face_w
    eye_l = float(np.linalg.norm(points[159, :2] - points[145, :2])) / face_h
    eye_r = float(np.linalg.norm(points[386, :2] - points[374, :2])) / face_h
    brow_l = float(np.linalg.norm(points[105, :2] - points[159, :2])) / face_h
    brow_r = float(np.linalg.norm(points[334, :2] - points[386, :2])) / face_h
    smile = _clamp((mouth_width - 0.35) / 0.25)
    frown = _clamp((0.38 - mouth_width) / 0.20)
    jaw = _clamp((mouth_open - 0.02) / 0.18)
    brow_up = _clamp((brow_l + brow_r) / 2.0 / 0.12)
    brow_down = _clamp(1.0 - brow_up)
    eye_wide = _clamp(((eye_l + eye_r) / 2.0 - 0.03) / 0.08)
    eye_squint = _clamp((0.05 - (eye_l + eye_r) / 2.0) / 0.04)
    return {
        1: brow_up * 0.7,
        2: brow_up,
        4: brow_down,
        5: eye_wide,
        6: smile * 0.45,
        7: eye_squint,
        9: frown * 0.35,
        10: jaw * 0.35,
        12: smile,
        14: smile * 0.4,
        15: frown,
        17: frown * 0.5,
        20: _clamp((mouth_width - 0.42) / 0.2),
        23: _clamp((0.34 - mouth_width) / 0.15),
        24: _clamp((0.34 - mouth_width) / 0.15),
        25: jaw,
        26: jaw,
    }


def _frame_to_bgr(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError("Expected HxWxC frame")
    return np.ascontiguousarray(array[:, :, :3])


def synthesize_au_csv_from_frames(
    frames: Sequence[Any],
    *,
    indices: Sequence[int] | None = None,
    sample_fps: float = 8.0,
    video_path: str | Path | None = None,
    download_model: bool = True,
) -> str | None:
    """Write a temporary AU CSV and return its path, or None on failure."""
    if not frames:
        return None

    from .face_landmarker import FacePoseNormalizer

    normalizer = None
    try:
        try:
            candidate = FacePoseNormalizer(download_model=download_model)
        except Exception:
            candidate = None
        if candidate is not None and getattr(candidate, "available", False):
            normalizer = candidate
        elif candidate is not None:
            candidate.close()
    except Exception:
        normalizer = None

    if normalizer is None:
        return None

    rows: list[dict[str, Any]] = []
    try:
        for offset, frame in enumerate(frames):
            try:
                frame_bgr = _frame_to_bgr(frame)
            except Exception:
                continue
            # MediaPipe expects RGB.
            frame_rgb = frame_bgr[:, :, ::-1].copy()
            timestamp_ms = int(round(offset * 1000.0 / max(sample_fps, 0.1)))
            result = normalizer.process_frame(
                frame_rgb,
                timestamp_ms=timestamp_ms,
            )
            if result is None or result.landmarks_xyz.size == 0:
                continue
            landmarks = result.landmarks_pose_normalized
            if landmarks.size == 0:
                landmarks = result.landmarks_xyz
            if result.blendshapes:
                intensities = _au_from_blendshapes(result.blendshapes)
            else:
                intensities = _au_from_landmarks(landmarks)

            frame_idx = (
                int(indices[offset])
                if indices is not None and offset < len(indices)
                else offset
            )
            row: dict[str, Any] = {
                "frame_idx": frame_idx,
                "frame_time_in_ms": float(timestamp_ms),
                "face_alignment_method": result.backend,
                "face_detection_score": float(result.face_score or 1.0),
            }
            for au_id in AU_IDS:
                intensity = _clamp(float(intensities.get(au_id, 0.0)))
                row[f"au_{au_id}"] = 1 if intensity >= 0.20 else 0
                row[f"au_{au_id}_intensity"] = intensity
            for index in LANDMARK_INDEXES:
                if index >= len(landmarks):
                    continue
                row[f"lm_mp_{index}_x"] = float(landmarks[index, 0])
                row[f"lm_mp_{index}_y"] = float(landmarks[index, 1])
                if landmarks.shape[1] > 2:
                    row[f"lm_mp_{index}_z"] = float(landmarks[index, 2])
            for name, value in sorted(result.blendshapes.items()):
                row[f"blendshape_{name}"] = _clamp(float(value))
            rows.append(row)
    finally:
        normalizer.close()

    if len(rows) < 2:
        return None

    fieldnames: list[str] = [
        "frame_idx",
        "frame_time_in_ms",
        "face_alignment_method",
        "face_detection_score",
    ]
    for au_id in AU_IDS:
        fieldnames.extend([f"au_{au_id}", f"au_{au_id}_intensity"])
    for index in LANDMARK_INDEXES:
        fieldnames.extend(
            [f"lm_mp_{index}_x", f"lm_mp_{index}_y", f"lm_mp_{index}_z"]
        )
    blend_keys = sorted(
        {
            key
            for row in rows
            for key in row
            if str(key).startswith("blendshape_")
        }
    )
    fieldnames.extend(blend_keys)

    stem = "synthesized_au"
    if video_path:
        stem = f"{Path(video_path).stem}_synthesized_au"
    # ASCII temp dir avoids OpenCV/MediaPipe path issues on Chinese roots.
    out_dir = Path(tempfile.gettempdir()) / "video_evaluator_au"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_{len(rows)}f.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(out_path.resolve())
