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
FLOW_FEATURE_NAMES = tuple(
    f"{region}_{metric}"
    for region in REGION_NAMES
    for metric in (
        "flow_mean",
        "flow_p90",
        "flow_spatial_std",
        "flow_jerk",
    )
)
FLOW_FEATURE_DIM = len(FLOW_FEATURE_NAMES)
FORENSIC_FEATURE_NAMES = tuple(
    f"{region}_{metric}"
    for region in REGION_NAMES
    for metric in (
        "normalized_gradient_mean",
        "normalized_gradient_p90",
        "normalized_laplacian_energy",
        "high_frequency_ratio",
        "motion_compensated_residual",
        "residual_instability",
    )
)
FORENSIC_FEATURE_DIM = len(FORENSIC_FEATURE_NAMES)
FACE_OVAL_LANDMARKS = (10, 127, 234, 93, 132, 152, 361, 323, 454, 356)


def _float(value: str | None, default: float = math.nan) -> float:
    try:
        parsed = float(value or "")
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _sample_rows(
    rows: list[dict[str, str]],
    max_frames: int,
    *,
    window_start_seconds: float | None = None,
    window_duration_seconds: float | None = None,
) -> list[dict[str, str]]:
    if (
        window_start_seconds is not None
        and window_duration_seconds is not None
        and rows
    ):
        origin_ms = _float(rows[0].get("frame_time_in_ms"), 0.0)
        start_ms = origin_ms + max(0.0, float(window_start_seconds)) * 1000.0
        end_ms = start_ms + max(0.0, float(window_duration_seconds)) * 1000.0
        rows = [
            row
            for row in rows
            if start_ms - 1e-3
            <= _float(row.get("frame_time_in_ms"), origin_ms)
            <= end_ms + 1e-3
        ]
    if len(rows) <= max_frames:
        return rows
    indexes = np.linspace(0, len(rows) - 1, max_frames).round().astype(int)
    return [rows[int(index)] for index in indexes]


def _read_window_frames(
    video_path: str | Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    num_frames: int,
    frame_size: int,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = max(float(capture.get(cv2.CAP_PROP_FPS) or 0.0), 1.0)
    frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0), 1)
    start = max(0.0, float(start_seconds))
    stop = min(
        start + max(float(duration_seconds), 0.0),
        max((frame_count - 1) / fps, 0.0),
    )
    times = np.linspace(start, max(start, stop), max(2, int(num_frames)))
    frames: list[np.ndarray] = []
    for timestamp in times:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(timestamp * fps)))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        frames.append(
            cv2.resize(
                frame,
                (int(frame_size), int(frame_size)),
                interpolation=cv2.INTER_AREA,
            )
        )
    capture.release()
    if not frames:
        raise RuntimeError(f"Video contains no readable frames: {video_path}")
    target = max(2, int(num_frames))
    if len(frames) < target:
        source = list(frames)
        frames = [
            source[
                min(
                    int(round(index * (len(source) - 1) / max(target - 1, 1))),
                    len(source) - 1,
                )
            ]
            for index in range(target)
        ]
    return frames[:target]


def _crop_frame(
    frame: np.ndarray,
    row: dict[str, str],
    landmark_ids: tuple[int, ...],
    *,
    output_size: int = 32,
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
        return np.zeros((output_size, output_size), dtype=np.float32), False
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
        return np.zeros((output_size, output_size), dtype=np.float32), False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return cv2.resize(
        gray,
        (output_size, output_size),
        interpolation=cv2.INTER_AREA,
    ), True


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
    window_start_seconds: float | None = None,
    window_duration_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed local crop sequence and a compact summary."""
    with Path(au_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = _sample_rows(
            list(csv.DictReader(handle)),
            max_frames,
            window_start_seconds=window_start_seconds,
            window_duration_seconds=window_duration_seconds,
        )
    if (
        window_start_seconds is not None
        and window_duration_seconds is not None
    ):
        frames = _read_window_frames(
            video_path=Path(video_path),
            start_seconds=window_start_seconds,
            duration_seconds=window_duration_seconds,
            num_frames=max_frames,
            frame_size=frame_size,
        )
    else:
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


def extract_face_crop_flow_features(
    video_path: str | Path,
    au_path: str | Path,
    *,
    max_frames: int = CROP_MAX_FRAMES,
    frame_size: int = 512,
    window_start_seconds: float | None = None,
    window_duration_seconds: float | None = None,
) -> np.ndarray:
    """Measure local facial optical-flow continuity on an aligned time window.

    Every crop is face-landmark anchored before flow is computed, so the
    descriptor captures local eye/mouth/brow deformation rather than camera
    translation, background motion or absolute brightness.
    """
    with Path(au_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = _sample_rows(
            list(csv.DictReader(handle)),
            max_frames,
            window_start_seconds=window_start_seconds,
            window_duration_seconds=window_duration_seconds,
        )
    if (
        window_start_seconds is not None
        and window_duration_seconds is not None
    ):
        frames = _read_window_frames(
            video_path=Path(video_path),
            start_seconds=window_start_seconds,
            duration_seconds=window_duration_seconds,
            num_frames=max_frames,
            frame_size=frame_size,
        )
    else:
        frames = _read_sampled_frames(
            video_path=Path(video_path),
            num_frames=max_frames,
            frame_size=frame_size,
        )
    count = min(len(rows), len(frames))
    if count < 2:
        return np.zeros(FLOW_FEATURE_DIM, dtype=np.float32)

    crops: dict[str, list[np.ndarray | None]] = {
        name: [] for name in REGION_NAMES
    }
    for frame, row in zip(frames[:count], rows[:count]):
        for name, landmark_ids in REGIONS.items():
            crop, valid = _crop_frame(frame, row, landmark_ids)
            crops[name].append(crop if valid else None)

    values: list[float] = []
    for name in REGION_NAMES:
        pair_mean: list[float] = []
        pair_p90: list[float] = []
        pair_spatial_std: list[float] = []
        sequence = crops[name]
        for previous, current in zip(sequence[:-1], sequence[1:]):
            if previous is None or current is None:
                continue
            flow = cv2.calcOpticalFlowFarneback(
                previous,
                current,
                None,
                pyr_scale=0.5,
                levels=2,
                winsize=11,
                iterations=2,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )
            magnitude = np.linalg.norm(flow, axis=2)
            pair_mean.append(float(np.mean(magnitude)))
            pair_p90.append(float(np.quantile(magnitude, 0.90)))
            pair_spatial_std.append(float(np.std(magnitude)))
        if not pair_mean:
            values.extend([0.0, 0.0, 0.0, 0.0])
            continue
        values.extend(
            [
                float(np.mean(pair_mean)),
                float(np.mean(pair_p90)),
                float(np.mean(pair_spatial_std)),
                float(
                    np.quantile(np.abs(np.diff(pair_mean)), 0.90)
                    if len(pair_mean) > 1
                    else 0.0
                ),
            ]
        )
    return np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _normalized_patch(crop: np.ndarray) -> np.ndarray:
    centered = crop.astype(np.float32) - float(np.mean(crop))
    return centered / max(float(np.std(centered)), 1.0 / 255.0)


def _spatial_forensic_values(crop: np.ndarray) -> tuple[float, float, float, float]:
    normalized = _normalized_patch(crop)
    gradient_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gradient_x, gradient_y)
    laplacian = cv2.Laplacian(normalized, cv2.CV_32F)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(normalized))) ** 2
    height, width = spectrum.shape
    low = np.zeros_like(spectrum, dtype=bool)
    low[
        height // 2 - height // 8 : height // 2 + height // 8 + 1,
        width // 2 - width // 8 : width // 2 + width // 8 + 1,
    ] = True
    total_energy = float(np.sum(spectrum))
    high_ratio = (
        float(np.sum(spectrum[~low])) / max(total_energy, 1e-8)
    )
    return (
        float(np.mean(gradient)),
        float(np.quantile(gradient, 0.90)),
        float(np.mean(np.abs(laplacian))),
        high_ratio,
    )


def _translation_compensated_residual(
    previous: np.ndarray,
    current: np.ndarray,
) -> float:
    previous_normalized = _normalized_patch(previous)
    current_normalized = _normalized_patch(current)
    try:
        shift, _ = cv2.phaseCorrelate(previous_normalized, current_normalized)
        matrix = np.asarray(
            [[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]],
            dtype=np.float32,
        )
        aligned = cv2.warpAffine(
            previous_normalized,
            matrix,
            (current.shape[1], current.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
    except cv2.error:
        aligned = previous_normalized
    return float(np.mean(np.abs(current_normalized - aligned)))


def extract_face_crop_forensic_features(
    video_path: str | Path,
    au_path: str | Path,
    *,
    max_frames: int = CROP_MAX_FRAMES,
    frame_size: int = 512,
    window_start_seconds: float | None = None,
    window_duration_seconds: float | None = None,
) -> np.ndarray:
    """Return brightness-invariant forensic evidence from local face regions.

    The descriptor never observes the background or full frame. Static values
    measure normalized local detail, while temporal residuals compensate small
    crop translations before measuring eye, mouth, brow and lower-face changes.
    """
    with Path(au_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = _sample_rows(
            list(csv.DictReader(handle)),
            max_frames,
            window_start_seconds=window_start_seconds,
            window_duration_seconds=window_duration_seconds,
        )
    if window_start_seconds is not None and window_duration_seconds is not None:
        frames = _read_window_frames(
            video_path=Path(video_path),
            start_seconds=window_start_seconds,
            duration_seconds=window_duration_seconds,
            num_frames=max_frames,
            frame_size=frame_size,
        )
    else:
        frames = _read_sampled_frames(
            video_path=Path(video_path),
            num_frames=max_frames,
            frame_size=frame_size,
        )
    count = min(len(rows), len(frames))
    if count < 2:
        return np.zeros(FORENSIC_FEATURE_DIM, dtype=np.float32)

    values: list[float] = []
    for landmark_ids in REGIONS.values():
        crops: list[np.ndarray] = []
        for frame, row in zip(frames[:count], rows[:count]):
            crop, valid = _crop_frame(
                frame,
                row,
                landmark_ids,
                output_size=64,
            )
            if valid:
                crops.append(crop)
        if len(crops) < 2:
            values.extend([0.0] * 6)
            continue
        spatial = np.asarray(
            [_spatial_forensic_values(crop) for crop in crops],
            dtype=np.float64,
        )
        residuals = np.asarray(
            [
                _translation_compensated_residual(previous, current)
                for previous, current in zip(crops[:-1], crops[1:])
            ],
            dtype=np.float64,
        )
        values.extend(
            [
                float(np.mean(spatial[:, 0])),
                float(np.mean(spatial[:, 1])),
                float(np.mean(spatial[:, 2])),
                float(np.mean(spatial[:, 3])),
                float(np.mean(residuals)),
                float(np.std(residuals)),
            ]
        )
    return np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def extract_face_chroma_features(
    video_path: str | Path,
    au_path: str | Path,
    *,
    max_frames: int = 7,
    frame_size: int = 512,
) -> np.ndarray:
    """Measure face-only chroma without observing background or brightness."""
    with Path(au_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = _sample_rows(list(csv.DictReader(handle)), max_frames)
    frames = _read_sampled_frames(
        video_path=Path(video_path),
        num_frames=max_frames,
        frame_size=frame_size,
    )
    saturation_values: list[float] = []
    chroma_values: list[float] = []
    for frame, row in zip(frames, rows):
        height, width = frame.shape[:2]
        points = []
        for landmark_id in FACE_OVAL_LANDMARKS:
            x = _float(row.get(f"lm_mp_{landmark_id}_x"))
            y = _float(row.get(f"lm_mp_{landmark_id}_y"))
            if math.isfinite(x) and math.isfinite(y):
                points.append(
                    [
                        int(np.clip(x, 0.0, 1.0) * width),
                        int(np.clip(y, 0.0, 1.0) * height),
                    ]
                )
        if len(points) < 6:
            continue
        hull = cv2.convexHull(np.asarray(points, dtype=np.int32))
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        pixels = mask > 0
        if int(np.sum(pixels)) < 64:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        saturation_values.append(float(np.mean(hsv[:, :, 1][pixels])) / 255.0)
        ab = lab[:, :, 1:].astype(np.float32) - 128.0
        chroma_values.append(
            float(np.mean(np.linalg.norm(ab[pixels], axis=1))) / 181.0
        )
    if not saturation_values:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(
        [
            float(np.mean(saturation_values)),
            float(np.quantile(saturation_values, 0.10)),
            float(np.mean(chroma_values)),
        ],
        dtype=np.float32,
    )
