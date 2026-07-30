from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .face_detection import FaceDetector
from .runtime import MODEL_CACHE_DIR

try:
    from skimage.metrics import peak_signal_noise_ratio as _skimage_psnr
    from skimage.metrics import structural_similarity as _skimage_ssim
except Exception:
    _skimage_psnr = None
    _skimage_ssim = None


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".m4v"}
DEFAULT_SAMPLE_FPS = 8.0
SEMANTIC_WINDOW_FRAMES = 8
SEMANTIC_WINDOW_SECONDS = 1.0
SEMANTIC_WINDOW_OVERLAP = 0.5
DEFAULT_MAX_FRAME_DIMENSION = 1920


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_video_path(path: str | Path | None) -> bool:
    return bool(path) and Path(path).suffix.lower() in VIDEO_SUFFIXES


def resolve_path(value: Any) -> str | None:
    """Accept paths returned by different Gradio versions."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, dict):
        for key in ("path", "name", "orig_name"):
            if value.get(key):
                return str(value[key])
    for attribute in ("path", "name"):
        candidate = getattr(value, attribute, None)
        if candidate:
            return str(candidate)
    return None


def probe_video(path: str | Path) -> VideoInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0))
    finally:
        capture.release()

    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Video has no readable frames: {path}")

    duration = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(
        path=str(path),
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_seconds=duration,
    )


def _sample_indices(frame_count: int, sample_count: int) -> np.ndarray:
    if frame_count <= 0 or sample_count <= 0:
        return np.array([], dtype=np.int64)
    if sample_count == 1:
        return np.array([0], dtype=np.int64)
    values = np.linspace(0, frame_count - 1, sample_count)
    return np.rint(values).astype(np.int64)


def _video_end_seconds(info: VideoInfo) -> float:
    if info.fps > 0 and info.frame_count > 0:
        return max(0.0, (info.frame_count - 1) / info.fps)
    return max(0.0, float(info.duration_seconds))


def _sample_timestamps(
    end_seconds: float,
    max_frames: int,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
) -> np.ndarray:
    if max_frames <= 0:
        return np.array([], dtype=np.float64)
    end_seconds = max(0.0, float(end_seconds))
    if end_seconds <= 1e-8:
        return np.array([0.0], dtype=np.float64)

    if sample_fps > 0:
        step = 1.0 / float(sample_fps)
        timestamps = np.arange(
            0.0,
            end_seconds + step * 0.5,
            step,
            dtype=np.float64,
        )
        timestamps = np.minimum(timestamps, end_seconds)
        if timestamps.size == 0 or timestamps[-1] < end_seconds - 1e-8:
            timestamps = np.append(timestamps, end_seconds)
    else:
        timestamps = np.array([0.0, end_seconds], dtype=np.float64)

    if timestamps.size > max_frames:
        timestamps = np.linspace(
            0.0,
            end_seconds,
            int(max_frames),
            dtype=np.float64,
        )
    return timestamps


def _time_sample_indices(
    info: VideoInfo,
    max_frames: int,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
) -> tuple[np.ndarray, np.ndarray]:
    count = min(int(max_frames), int(info.frame_count))
    if count < 1:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    if info.fps <= 0:
        timestamps = np.arange(count, dtype=np.float64)
        return _sample_indices(info.frame_count, count), timestamps

    timestamps = _sample_timestamps(
        _video_end_seconds(info),
        count,
        sample_fps,
    )
    indices = np.rint(timestamps * info.fps).astype(np.int64)
    indices = np.clip(indices, 0, max(info.frame_count - 1, 0))
    return indices, timestamps


def _aligned_sample_indices(
    result_info: VideoInfo,
    ground_truth_info: VideoInfo,
    max_frames: int,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    sample_count = min(int(max_frames), result_info.frame_count, ground_truth_info.frame_count)
    if sample_count < 1:
        raise ValueError("The videos do not contain a common readable frame.")

    if result_info.fps > 0 and ground_truth_info.fps > 0:
        result_end = _video_end_seconds(result_info)
        ground_truth_end = _video_end_seconds(ground_truth_info)
        overlap_seconds = min(result_end, ground_truth_end)
        timestamps = _sample_timestamps(
            overlap_seconds,
            sample_count,
            sample_fps,
        )
        sample_count = int(len(timestamps))
        result_indices = np.rint(timestamps * result_info.fps).astype(np.int64)
        ground_truth_indices = np.rint(
            timestamps * ground_truth_info.fps
        ).astype(np.int64)
        result_indices = np.clip(
            result_indices,
            0,
            max(result_info.frame_count - 1, 0),
        )
        ground_truth_indices = np.clip(
            ground_truth_indices,
            0,
            max(ground_truth_info.frame_count - 1, 0),
        )
        return sample_count, result_indices, ground_truth_indices, timestamps

    result_indices = _sample_indices(result_info.frame_count, sample_count)
    ground_truth_indices = _sample_indices(
        ground_truth_info.frame_count,
        sample_count,
    )
    timestamps = np.arange(sample_count, dtype=np.float64)
    return sample_count, result_indices, ground_truth_indices, timestamps


def sample_video_frames(
    path: str | Path,
    max_frames: int,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[np.ndarray]]:
    info = probe_video(path)
    indices, timestamps = _time_sample_indices(info, max_frames, sample_fps)
    frames = _read_frames(info.path, indices)
    return info.to_dict(), indices, timestamps, frames


def _window_starts(
    end_seconds: float,
    max_windows: int,
    window_seconds: float = SEMANTIC_WINDOW_SECONDS,
    overlap: float = SEMANTIC_WINDOW_OVERLAP,
) -> np.ndarray:
    if max_windows <= 0:
        return np.array([], dtype=np.float64)
    end_seconds = max(0.0, float(end_seconds))
    window_seconds = max(1e-6, float(window_seconds))
    last_start = max(0.0, end_seconds - window_seconds)
    if last_start <= 1e-8:
        return np.array([0.0], dtype=np.float64)

    stride = window_seconds * max(1e-3, 1.0 - float(overlap))
    starts = np.arange(
        0.0,
        last_start + stride * 0.5,
        stride,
        dtype=np.float64,
    )
    if starts.size == 0 or starts[-1] < last_start - 1e-8:
        starts = np.append(starts, last_start)
    if starts.size > max_windows:
        starts = np.linspace(0.0, last_start, int(max_windows))
    return starts


def _window_timestamps(
    start_seconds: float,
    end_seconds: float,
    frame_count: int,
) -> np.ndarray:
    count = max(1, int(frame_count))
    return np.linspace(
        float(start_seconds),
        max(float(start_seconds), float(end_seconds)),
        count,
        dtype=np.float64,
    )


def sample_video_windows(
    path: str | Path,
    max_frames: int,
    window_frames: int = SEMANTIC_WINDOW_FRAMES,
    window_seconds: float = SEMANTIC_WINDOW_SECONDS,
    overlap: float = SEMANTIC_WINDOW_OVERLAP,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Sample a video into bounded temporal windows.

    ``max_frames`` controls the total window budget. Each model window keeps
    its own fixed temporal resolution instead of collapsing the whole clip
    into one sparse sample.
    """
    info = probe_video(path)
    max_windows = max(1, int(math.ceil(max_frames / max(window_frames, 1))))
    end_seconds = _video_end_seconds(info)
    starts = _window_starts(
        end_seconds,
        max_windows,
        window_seconds=window_seconds,
        overlap=overlap,
    )
    windows: list[dict[str, Any]] = []
    for window_index, start_seconds in enumerate(starts):
        stop_seconds = min(
            end_seconds,
            float(start_seconds) + max(float(window_seconds), 1e-6),
        )
        timestamps = _window_timestamps(
            float(start_seconds),
            stop_seconds,
            window_frames,
        )
        if info.fps > 0:
            indices = np.rint(timestamps * info.fps).astype(np.int64)
        else:
            indices = np.rint(
                timestamps / max(end_seconds, 1e-6) * (info.frame_count - 1)
            ).astype(np.int64)
        indices = np.clip(indices, 0, max(info.frame_count - 1, 0))
        windows.append(
            {
                "window_index": window_index,
                "start_seconds": round(float(start_seconds), 6),
                "end_seconds": round(float(stop_seconds), 6),
                "timestamps": timestamps,
                "indices": indices,
                "frames": _read_frames(info.path, indices),
            }
        )
    return info.to_dict(), windows


def sample_aligned_video_windows(
    result_path: str | Path,
    reference_path: str | Path,
    max_frames: int,
    window_frames: int = SEMANTIC_WINDOW_FRAMES,
    window_seconds: float = SEMANTIC_WINDOW_SECONDS,
    overlap: float = SEMANTIC_WINDOW_OVERLAP,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Sample result/reference videos in matching timestamp windows."""
    result_info = probe_video(result_path)
    reference_info = probe_video(reference_path)
    max_windows = max(1, int(math.ceil(max_frames / max(window_frames, 1))))
    overlap_end = min(
        _video_end_seconds(result_info),
        _video_end_seconds(reference_info),
    )
    starts = _window_starts(
        overlap_end,
        max_windows,
        window_seconds=window_seconds,
        overlap=overlap,
    )
    windows: list[dict[str, Any]] = []
    for window_index, start_seconds in enumerate(starts):
        stop_seconds = min(
            overlap_end,
            float(start_seconds) + max(float(window_seconds), 1e-6),
        )
        timestamps = _window_timestamps(
            float(start_seconds),
            stop_seconds,
            window_frames,
        )
        result_indices = np.rint(timestamps * result_info.fps).astype(np.int64)
        reference_indices = np.rint(
            timestamps * reference_info.fps
        ).astype(np.int64)
        result_indices = np.clip(
            result_indices,
            0,
            max(result_info.frame_count - 1, 0),
        )
        reference_indices = np.clip(
            reference_indices,
            0,
            max(reference_info.frame_count - 1, 0),
        )
        windows.append(
            {
                "window_index": window_index,
                "start_seconds": round(float(start_seconds), 6),
                "end_seconds": round(float(stop_seconds), 6),
                "timestamps": timestamps,
                "result_indices": result_indices,
                "reference_indices": reference_indices,
                "result_frames": _read_frames(result_info.path, result_indices),
                "reference_frames": _read_frames(
                    reference_info.path,
                    reference_indices,
                ),
            }
        )
    return result_info.to_dict(), reference_info.to_dict(), windows


def _max_frame_dimension() -> int:
    raw_value = os.environ.get(
        "EVALUATOR_MAX_FRAME_DIMENSION",
        str(DEFAULT_MAX_FRAME_DIMENSION),
    ).strip()
    try:
        return max(0, int(raw_value))
    except ValueError:
        return DEFAULT_MAX_FRAME_DIMENSION


def _resize_frame_for_evaluation(frame: np.ndarray) -> np.ndarray:
    max_dimension = _max_frame_dimension()
    if max_dimension <= 0 or max(frame.shape[:2]) <= max_dimension:
        return frame
    height, width = frame.shape[:2]
    scale = max_dimension / max(height, width)
    target_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def _read_frames(path: str, indices: Iterable[int]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}")

    frames: list[np.ndarray] = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"Unable to read frame {index} from {path}")
            frame = _resize_frame_for_evaluation(frame)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


NormalizedFaceBox = tuple[float, float, float, float]


@lru_cache(maxsize=1)
def _default_face_detector() -> FaceDetector:
    return FaceDetector()


def _normalized_face_box(
    frame: np.ndarray,
    detector: FaceDetector,
) -> NormalizedFaceBox | None:
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return None
    bbox = detector.detect(frame)
    if bbox is None:
        return None
    x, y, box_width, box_height = bbox
    x0 = max(0.0, min(1.0, x / width))
    y0 = max(0.0, min(1.0, y / height))
    x1 = max(x0, min(1.0, (x + box_width) / width))
    y1 = max(y0, min(1.0, (y + box_height) / height))
    return x0, y0, x1, y1


def _aggregate_face_box(
    frames: Iterable[np.ndarray],
    detector: FaceDetector,
) -> NormalizedFaceBox | None:
    boxes = [
        box
        for frame in frames
        if (box := _normalized_face_box(frame, detector)) is not None
    ]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _resize_like(frame: np.ndarray, target: np.ndarray) -> np.ndarray:
    if frame.shape[:2] == target.shape[:2]:
        return frame
    return cv2.resize(
        frame,
        (target.shape[1], target.shape[0]),
        interpolation=cv2.INTER_AREA,
    )


def _center_crop_to_aspect(
    frame: np.ndarray,
    target_aspect_ratio: float,
) -> np.ndarray:
    """Crop only the excess border so aspect ratios are never stretched."""
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0 or target_aspect_ratio <= 0:
        return frame
    current_aspect_ratio = width / height
    if abs(current_aspect_ratio - target_aspect_ratio) <= 0.01:
        return frame
    if current_aspect_ratio > target_aspect_ratio:
        cropped_width = max(1, int(round(height * target_aspect_ratio)))
        left = max(0, (width - cropped_width) // 2)
        return frame[:, left : left + cropped_width]
    cropped_height = max(1, int(round(width / target_aspect_ratio)))
    top = max(0, (height - cropped_height) // 2)
    return frame[top : top + cropped_height, :]


def _face_protected_crop_to_aspect(
    frame: np.ndarray,
    target_aspect_ratio: float,
    face_box: NormalizedFaceBox | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = frame.shape[:2]
    current_aspect_ratio = width / max(height, 1)
    metadata: dict[str, Any] = {
        "face_protection_status": "not_needed",
        "face_box_normalized": list(face_box) if face_box else None,
        "crop": {
            "left": 0,
            "top": 0,
            "width": width,
            "height": height,
        },
    }
    if (
        height <= 0
        or width <= 0
        or target_aspect_ratio <= 0
        or abs(current_aspect_ratio - target_aspect_ratio) <= 0.01
    ):
        return frame, metadata

    face_margin = 0.20
    if current_aspect_ratio > target_aspect_ratio:
        crop_width = max(1, int(round(height * target_aspect_ratio)))
        max_left = max(0, width - crop_width)
        left = max(0, (width - crop_width) // 2)
        protected = False
        if face_box is not None:
            face_left = face_box[0] * width
            face_right = face_box[2] * width
            padding = (face_right - face_left) * face_margin
            keep_left = max(0.0, face_left - padding)
            keep_right = min(float(width), face_right + padding)
            lower = max(0, int(math.ceil(keep_right - crop_width)))
            upper = min(max_left, int(math.floor(keep_left)))
            desired = int(round((keep_left + keep_right) / 2 - crop_width / 2))
            if lower <= upper:
                left = max(lower, min(upper, desired))
                protected = True
        crop = frame[:, left : left + crop_width]
        metadata["crop"] = {
            "left": left,
            "top": 0,
            "width": crop_width,
            "height": height,
        }
    else:
        crop_height = max(1, int(round(width / target_aspect_ratio)))
        max_top = max(0, height - crop_height)
        top = max(0, (height - crop_height) // 2)
        protected = False
        if face_box is not None:
            face_top = face_box[1] * height
            face_bottom = face_box[3] * height
            padding = (face_bottom - face_top) * face_margin
            keep_top = max(0.0, face_top - padding)
            keep_bottom = min(float(height), face_bottom + padding)
            lower = max(0, int(math.ceil(keep_bottom - crop_height)))
            upper = min(max_top, int(math.floor(keep_top)))
            desired = int(round((keep_top + keep_bottom) / 2 - crop_height / 2))
            if lower <= upper:
                top = max(lower, min(upper, desired))
                protected = True
        crop = frame[top : top + crop_height, :]
        metadata["crop"] = {
            "left": 0,
            "top": top,
            "width": width,
            "height": crop_height,
        }

    metadata["face_protection_status"] = (
        "applied"
        if protected
        else ("fallback_center" if face_box is None else "face_not_fully_contained")
    )
    return crop, metadata


def _align_ground_truth_frame(
    frame: np.ndarray,
    target: np.ndarray,
    *,
    face_box: NormalizedFaceBox | None = None,
    detect_face: bool = True,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    if face_box is None and detect_face:
        face_box = _normalized_face_box(frame, _default_face_detector())
    cropped, metadata = _face_protected_crop_to_aspect(
        frame,
        target.shape[1] / max(target.shape[0], 1),
        face_box,
    )
    aligned = _resize_like(cropped, target)
    if return_metadata:
        return aligned, metadata
    return aligned


def _ssim(result: np.ndarray, ground_truth: np.ndarray) -> float:
    if _skimage_ssim is not None:
        try:
            value = _skimage_ssim(
                result,
                ground_truth,
                channel_axis=-1,
                data_range=255,
            )
            return float(value)
        except (TypeError, ValueError):
            try:
                value = _skimage_ssim(
                    result,
                    ground_truth,
                    multichannel=True,
                    data_range=255,
                )
                return float(value)
            except (TypeError, ValueError):
                pass

    # This global SSIM fallback keeps the MVP usable when binary wheels in
    # the host environment are temporarily incompatible with NumPy.
    result_float = result.astype(np.float64)
    ground_truth_float = ground_truth.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    values = []
    for channel in range(result.shape[2]):
        result_channel = result_float[:, :, channel]
        ground_truth_channel = ground_truth_float[:, :, channel]
        mean_result = result_channel.mean()
        mean_ground_truth = ground_truth_channel.mean()
        variance_result = result_channel.var()
        variance_ground_truth = ground_truth_channel.var()
        covariance = np.mean(
            (result_channel - mean_result)
            * (ground_truth_channel - mean_ground_truth)
        )
        numerator = (
            (2 * mean_result * mean_ground_truth + c1)
            * (2 * covariance + c2)
        )
        denominator = (
            (mean_result**2 + mean_ground_truth**2 + c1)
            * (variance_result + variance_ground_truth + c2)
        )
        values.append(numerator / denominator if denominator else 1.0)
    return float(np.mean(values))


def _psnr(result: np.ndarray, ground_truth: np.ndarray) -> float:
    mse = _mse(result, ground_truth)
    if mse == 0:
        return float("inf")
    if _skimage_psnr is not None:
        return float(
            _skimage_psnr(
                ground_truth,
                result,
                data_range=255,
            )
        )

    return float(10 * np.log10((255**2) / mse))


def _mse(result: np.ndarray, ground_truth: np.ndarray) -> float:
    difference = ground_truth.astype(np.float64) - result.astype(np.float64)
    return float(np.mean(np.square(difference)))


def _resolve_device(device: str) -> str:
    forced_device = os.environ.get("EVALUATOR_IQA_DEVICE", "").lower()
    if forced_device in {"cpu", "cuda"}:
        device = forced_device
    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return "cpu"


def _compute_lpips(
    result_frames: list[np.ndarray],
    ground_truth_frames: list[np.ndarray],
    device: str,
    batch_size: int = 8,
) -> tuple[list[float] | None, str | None]:
    try:
        import torch
        cache_dir = MODEL_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(cache_dir))
        import lpips
    except ImportError as exc:
        return None, f"LPIPS dependency is not installed: {exc}"

    try:
        model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception as exc:
        return None, (
            "LPIPS model could not be loaded. The first run may need model "
            f"weights to be downloaded or cached: {exc}"
        )

    values: list[float] = []
    with torch.no_grad():
        for start in range(0, len(result_frames), batch_size):
            result_batch = np.stack(result_frames[start : start + batch_size])
            gt_batch = np.stack(ground_truth_frames[start : start + batch_size])
            result_tensor = (
                torch.from_numpy(result_batch)
                .permute(0, 3, 1, 2)
                .float()
                .div(127.5)
                .sub(1.0)
                .to(device)
            )
            gt_tensor = (
                torch.from_numpy(gt_batch)
                .permute(0, 3, 1, 2)
                .float()
                .div(127.5)
                .sub(1.0)
                .to(device)
            )
            distances = model(result_tensor, gt_tensor).flatten().detach().cpu()
            values.extend(float(value) for value in distances)

    del model
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return values, None


def _mean(values: list[float | None]) -> float | None:
    valid = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(valid)) if valid else None


def evaluate_full_reference(
    result_path: str | Path,
    ground_truth_path: str | Path,
    max_frames: int = 8,
    calculate_lpips: bool = True,
    device: str = "auto",
) -> dict[str, Any]:
    """Evaluate a generated video against a time-corresponding GT video."""
    result_info = probe_video(result_path)
    ground_truth_info = probe_video(ground_truth_path)

    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    aspect_ratio_delta = abs(
        (result_info.width / result_info.height)
        - (ground_truth_info.width / ground_truth_info.height)
    )
    sample_count, result_indices, ground_truth_indices, timestamps = (
        _aligned_sample_indices(result_info, ground_truth_info, max_frames)
    )
    result_frames = _read_frames(result_info.path, result_indices)
    ground_truth_frames = _read_frames(
        ground_truth_info.path,
        ground_truth_indices,
    )
    face_box = (
        _aggregate_face_box(
            ground_truth_frames,
            _default_face_detector(),
        )
        if aspect_ratio_delta > 0.01
        else None
    )
    aligned_with_metadata = [
        _align_ground_truth_frame(
            frame,
            result_frames[i],
            face_box=face_box,
            detect_face=False,
            return_metadata=True,
        )
        for i, frame in enumerate(ground_truth_frames)
    ]
    aligned_ground_truth_frames = [
        aligned for aligned, _ in aligned_with_metadata
    ]
    alignment_metadata = (
        aligned_with_metadata[0][1] if aligned_with_metadata else {}
    )
    alignment_mode = (
        (
            "face_protected_crop_gt_to_result_aspect"
            if alignment_metadata.get("face_protection_status") == "applied"
            else "center_crop_gt_to_result_aspect"
        )
        if aspect_ratio_delta > 0.01
        else "resize_gt_to_result"
    )

    psnr_values: list[float] = []
    mse_values: list[float] = []
    ssim_values: list[float] = []
    for result_frame, ground_truth_frame in zip(
        result_frames,
        aligned_ground_truth_frames,
    ):
        mse_values.append(_mse(result_frame, ground_truth_frame))
        psnr_values.append(_psnr(result_frame, ground_truth_frame))
        ssim_values.append(_ssim(result_frame, ground_truth_frame))

    resolved_device = _resolve_device(device)
    lpips_values: list[float | None] = [None] * sample_count
    lpips_error: str | None = None
    if calculate_lpips:
        raw_lpips, lpips_error = _compute_lpips(
            result_frames,
            aligned_ground_truth_frames,
            resolved_device,
        )
        if raw_lpips is not None:
            lpips_values = raw_lpips

    warnings: list[str] = []
    if aspect_ratio_delta > 0.01:
        face_status = alignment_metadata.get(
            "face_protection_status",
            "fallback_center",
        )
        if face_status == "applied":
            crop_note = "已对 GT 做人脸保护裁剪"
        elif face_status == "face_not_fully_contained":
            crop_note = "人脸保护窗口无法完整容纳人脸，已回退居中裁剪"
        else:
            crop_note = "未检测到可靠人脸，已回退居中裁剪"
        warnings.append(
            f"GT 与结果视频宽高比不同（结果 {result_info.width}:{result_info.height} / "
            f"GT {ground_truth_info.width}:{ground_truth_info.height}）；"
            f"未拉伸画面，{crop_note}后再比较，指标仅代表裁剪后的共同区域。"
        )
    elif result_info.width != ground_truth_info.width or result_info.height != ground_truth_info.height:
        warnings.append(
            "GT 分辨率与结果视频不同，计算前已将 GT 帧缩放到结果视频分辨率。"
        )
    if result_info.fps > 0 and ground_truth_info.fps > 0:
        fps_delta = abs(result_info.fps - ground_truth_info.fps)
        if fps_delta > 0.01:
            warnings.append(
                f"FPS 不同（结果 {result_info.fps:.3f} / GT {ground_truth_info.fps:.3f}），"
                "当前按共同时间区间采样并对齐时间戳。"
            )
    duration_delta = abs(
        result_info.duration_seconds - ground_truth_info.duration_seconds
    )
    if duration_delta > 0.05:
        warnings.append(
            f"时长不同（结果 {result_info.duration_seconds:.3f}s / "
            f"GT {ground_truth_info.duration_seconds:.3f}s）。"
        )
    if lpips_error:
        warnings.append(lpips_error)

    records: list[dict[str, Any]] = []
    for i in range(sample_count):
        records.append(
            {
                "sample_index": i,
                "result_frame": int(result_indices[i]),
                "gt_frame": int(ground_truth_indices[i]),
                "timestamp_seconds": round(float(timestamps[i]), 4),
                "psnr_db": round(psnr_values[i], 6),
                "ssim": round(ssim_values[i], 6),
                "lpips": (
                    round(lpips_values[i], 6)
                    if lpips_values[i] is not None
                    else None
                ),
            }
        )

    return {
        "result_video": result_info.to_dict(),
        "ground_truth_video": ground_truth_info.to_dict(),
        "sample_count": sample_count,
        "device": resolved_device,
        "alignment": {
            "mode": alignment_mode,
            "aspect_ratio_delta": float(aspect_ratio_delta),
            "comparison_region": (
                alignment_mode
                if aspect_ratio_delta > 0.01
                else "full_frame"
            ),
            "face_protection": {
                "status": alignment_metadata.get(
                    "face_protection_status",
                    "not_needed",
                ),
                "face_box_normalized": alignment_metadata.get(
                    "face_box_normalized"
                ),
                "crop": alignment_metadata.get("crop"),
            },
        },
        "metrics": {
            "psnr_db": (
                float(10 * np.log10((255**2) / float(np.mean(mse_values))))
                if float(np.mean(mse_values)) > 0
                else float("inf")
            ),
            "ssim": _mean(ssim_values),
            "lpips": _mean(lpips_values),
        },
        "warnings": warnings,
        "records": records,
    }
