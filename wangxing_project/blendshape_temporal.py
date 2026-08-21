"""MediaPipe 52-blendshape temporal descriptors for Wang Xing PT v4."""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from evaluator.modules.core.face_landmarker import FacePoseNormalizer
from evaluator.modules.core.face_landmarker import default_model_path
from evaluator.vedio_pred.real_video_detector import _read_sampled_frames

BLENDSHAPE_NAMES: tuple[str, ...] = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft",
    "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
)

BLENDSHAPE_DIM = len(BLENDSHAPE_NAMES)
# Mean, standard deviation, p95 velocity, endpoint delta, valid ratio, mask.
BLENDSHAPE_FEATURE_DIM = BLENDSHAPE_DIM * 4 + 2
_EXTRACTOR: FacePoseNormalizer | None = None
_TIMESTAMP_BASE_MS = 0


def _finite(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _ascii_model_path() -> Path:
    """Copy the task model to an ASCII path for MediaPipe on Windows."""
    source = default_model_path()
    if not source.is_file():
        raise FileNotFoundError(f"FaceLandmarker model was not found: {source}")
    target = Path(tempfile.gettempdir()) / "video_evaluator_face_landmarker.task"
    if (
        not target.is_file()
        or target.stat().st_size != source.stat().st_size
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return target


def extract_blendshape_temporal_features(
    video_path: str | Path,
    *,
    max_frames: int = 24,
    frame_size: int = 512,
) -> np.ndarray:
    """Extract finite Blendshape state and motion features."""
    frames = _read_sampled_frames(
        video_path=Path(video_path),
        num_frames=max_frames,
        frame_size=frame_size,
    )
    global _EXTRACTOR, _TIMESTAMP_BASE_MS
    if _EXTRACTOR is None or not _EXTRACTOR.available:
        _EXTRACTOR = FacePoseNormalizer(
            model_path=_ascii_model_path(),
            download_model=False,
            prefer_tasks=True,
        )
    extractor = _EXTRACTOR
    timestamp_step_ms = max(1, int(round(1000.0 / 8.0)))
    timestamp_base_ms = _TIMESTAMP_BASE_MS
    _TIMESTAMP_BASE_MS += (len(frames) + 1) * timestamp_step_ms
    rows: list[np.ndarray] = []
    try:
        for index, frame_bgr in enumerate(frames):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = extractor.process_frame(
                frame_rgb,
                timestamp_ms=(
                    timestamp_base_ms + index * timestamp_step_ms
                ),
            )
            if result is None or not result.blendshapes:
                rows.append(np.full(BLENDSHAPE_DIM, np.nan, dtype=np.float32))
            else:
                rows.append(
                    np.asarray(
                        [
                            float(result.blendshapes.get(name, math.nan))
                            for name in BLENDSHAPE_NAMES
                        ],
                        dtype=np.float32,
                    )
                )
    finally:
        # The MediaPipe extractor is intentionally reused across videos.
        pass
    if not rows:
        return np.zeros(BLENDSHAPE_FEATURE_DIM, dtype=np.float32)
    matrix = np.stack(rows)
    valid = np.isfinite(matrix).all(axis=1)
    usable = matrix[valid]
    if usable.size == 0:
        return np.zeros(BLENDSHAPE_FEATURE_DIM, dtype=np.float32)

    mean = np.mean(usable, axis=0)
    std = np.std(usable, axis=0)
    if len(usable) >= 2:
        velocity = np.abs(np.diff(usable, axis=0))
        velocity_p95 = np.quantile(velocity, 0.95, axis=0)
        endpoint_delta = usable[-1] - usable[0]
    else:
        velocity_p95 = np.zeros(BLENDSHAPE_DIM, dtype=np.float32)
        endpoint_delta = np.zeros(BLENDSHAPE_DIM, dtype=np.float32)
    return _finite(
        np.concatenate(
            [
                mean,
                std,
                velocity_p95,
                endpoint_delta,
                np.asarray(
                    [float(np.mean(valid)), float(not np.all(valid))],
                    dtype=np.float32,
                ),
            ]
        )
    )


def blendshape_temporal_vector(video_path: str | Path) -> np.ndarray:
    return extract_blendshape_temporal_features(video_path)
