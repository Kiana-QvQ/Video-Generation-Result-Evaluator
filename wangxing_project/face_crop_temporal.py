"""Face-crop temporal descriptors shared by PT v4.3 and web v4.3."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np

from evaluator.vedio_pred.real_video_detector import _read_sampled_frames

CROP_MAX_FRAMES = 24
REGIONS = {
    "left_eye": (33, 133, 159, 145),
    "right_eye": (263, 362, 386, 374),
    "mouth": (61, 291, 13, 14, 78, 308),
    "brow": (70, 107, 300, 336),
    "lower_face": (61, 291, 152, 172, 397),
}
REGION_NAMES = tuple(REGIONS)
# mean, contrast, edge density, Laplacian energy, frame residual, valid mask.
CROP_FRAME_DIM = len(REGIONS) * 6
CROP_SUMMARY_NAMES = (
    "crop_eye_residual_0_1",
    "crop_mouth_residual_0_1",
    "crop_brow_residual_0_1",
    "crop_lower_face_residual_0_1",
    "crop_eye_continuity_0_1",
    "crop_mouth_continuity_0_1",
    "crop_local_flicker_0_1",
    "crop_valid_ratio_0_1",
)
CROP_SUMMARY_DIM = len(CROP_SUMMARY_NAMES)


def _float(value: str | None, default: float = math.nan) -> float:
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
    indexes = np.linspace(0, len(rows) - 1, max_frames).round().astype(int)
    return [rows[int(index)] for index in indexes]


def _crop_frame(
    frame: np.ndarray,
    row: dict[str, str],
    landmark_ids: tuple[int, ...],
) -> tuple[np.ndarray, bool]:
    height, width = frame.shape[:2]
    points: list[tuple[float, float]] = []
    for landmark_id in landmark_ids:
        x = _float(row.get(f"lm_mp_{landmark_id}_x"))
        y = _float(row.get(f"lm_mp_{landmark_id}_y"))
        if math.isfinite(x) and math.isfinite(y):
            points.append(
                (
                    float(np.clip(x, 0.0, 1.0) * width),
                    float(np.clip(y, 0.0, 1.0) * height),
                )
            )
    if len(points) < 2:
        return np.zeros((32, 32), dtype=np.float32), False
    values = np.asarray(points, dtype=np.float32)
    low = values.min(axis=0)
    high = values.max(axis=0)
    span = max(float(np.max(high - low)), 4.0)
    margin = max(0.75 * span, 8.0)
    x0 = max(0, int(low[0] - margin))
    y0 = max(0, int(low[1] - margin))
    x1 = min(width, int(high[0] + margin))
    y1 = min(height, int(high[1] + margin))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((32, 32), dtype=np.float32), False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA), True


def _crop_stats(
    crop: np.ndarray,
    previous: np.ndarray | None,
    valid: bool,
) -> np.ndarray:
    if not valid:
        return np.zeros(6, dtype=np.float32)
    centered = crop - float(np.mean(crop))
    edges = cv2.Canny(
        np.clip(crop * 255.0, 0.0, 255.0).astype(np.uint8),
        40,
        120,
    )
    laplacian = float(cv2.Laplacian(centered, cv2.CV_32F).var())
    residual = (
        float(np.mean(np.abs(crop - previous)))
        if previous is not None
        else 0.0
    )
    return np.asarray(
        [
            float(np.mean(crop)),
            float(np.std(centered)),
            float(np.mean(edges > 0)),
            float(np.clip(laplacian / 0.25, 0.0, 1.0)),
            float(np.clip(residual * 8.0, 0.0, 1.0)),
            1.0,
        ],
        dtype=np.float32,
    )


def extract_face_crop_temporal_features(
    video_path: str | Path,
    au_path: str | Path,
    *,
    max_frames: int = CROP_MAX_FRAMES,
    frame_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed local crop sequence and a compact summary."""
    with Path(au_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = _sample_rows(list(csv.DictReader(handle)), max_frames)
    frames = _read_sampled_frames(
        video_path=Path(video_path),
        num_frames=max_frames,
        frame_size=frame_size,
    )
    count = min(len(rows), len(frames))
    rows = rows[:count]
    frames = frames[:count]
    if count == 0:
        return (
            np.zeros((max_frames, CROP_FRAME_DIM), dtype=np.float32),
            np.zeros(CROP_SUMMARY_DIM, dtype=np.float32),
        )

    previous: dict[str, np.ndarray | None] = {
        name: None for name in REGION_NAMES
    }
    records: list[np.ndarray] = []
    for frame, row in zip(frames, rows):
        frame_values: list[np.ndarray] = []
        for name, landmark_ids in REGIONS.items():
            crop, valid = _crop_frame(frame, row, landmark_ids)
            frame_values.append(_crop_stats(crop, previous[name], valid))
            previous[name] = crop if valid else None
        records.append(np.concatenate(frame_values))
    sequence = np.stack(records).astype(np.float32)
    if len(sequence) < max_frames:
        padding = np.repeat(sequence[-1:], max_frames - len(sequence), axis=0)
        padding[:, 5::6] = 0.0
        sequence = np.concatenate([sequence, padding], axis=0)

    def region(index: int) -> np.ndarray:
        return sequence[:count, index * 6 : index * 6 + 6]

    eye_residual = np.mean(
        np.concatenate([region(0)[:, 4], region(1)[:, 4]])
    )
    mouth_residual = float(np.mean(region(2)[:, 4]))
    brow_residual = float(np.mean(region(3)[:, 4]))
    lower_residual = float(np.mean(region(4)[:, 4]))
    eye_continuity = float(
        np.mean(np.concatenate([region(0)[:, 5], region(1)[:, 5]]))
    )
    mouth_continuity = float(np.mean(region(2)[:, 5]))
    residuals = np.concatenate(
        [region(index)[:, 4] for index in range(len(REGIONS))]
    )
    summary = np.asarray(
        [
            float(np.clip(1.0 - eye_residual, 0.0, 1.0)),
            float(np.clip(1.0 - mouth_residual, 0.0, 1.0)),
            float(np.clip(1.0 - brow_residual, 0.0, 1.0)),
            float(np.clip(1.0 - lower_residual, 0.0, 1.0)),
            eye_continuity,
            mouth_continuity,
            float(np.clip(np.std(residuals) * 8.0, 0.0, 1.0)),
            float(np.mean(sequence[:count, 5::6])),
        ],
        dtype=np.float32,
    )
    return (
        np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0),
    )
