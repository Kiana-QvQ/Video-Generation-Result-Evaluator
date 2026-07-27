from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .runtime import MODEL_CACHE_DIR

try:
    from skimage.metrics import peak_signal_noise_ratio as _skimage_psnr
    from skimage.metrics import structural_similarity as _skimage_ssim
except Exception:
    _skimage_psnr = None
    _skimage_ssim = None


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".m4v"}


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


def _aligned_sample_indices(
    result_info: VideoInfo,
    ground_truth_info: VideoInfo,
    max_frames: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    sample_count = min(
        int(max_frames),
        result_info.frame_count,
        ground_truth_info.frame_count,
    )
    if sample_count < 1:
        raise ValueError("The videos do not contain a common readable frame.")

    if result_info.fps > 0 and ground_truth_info.fps > 0:
        result_end = (result_info.frame_count - 1) / result_info.fps
        ground_truth_end = (ground_truth_info.frame_count - 1) / ground_truth_info.fps
        overlap_seconds = min(result_end, ground_truth_end)
        timestamps = np.linspace(0.0, overlap_seconds, sample_count)
        result_indices = np.rint(timestamps * result_info.fps).astype(np.int64)
        ground_truth_indices = np.rint(
            timestamps * ground_truth_info.fps
        ).astype(np.int64)
        return sample_count, result_indices, ground_truth_indices, timestamps

    result_indices = _sample_indices(result_info.frame_count, sample_count)
    ground_truth_indices = _sample_indices(
        ground_truth_info.frame_count,
        sample_count,
    )
    timestamps = np.arange(sample_count, dtype=np.float64)
    return sample_count, result_indices, ground_truth_indices, timestamps


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
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def _resize_like(frame: np.ndarray, target: np.ndarray) -> np.ndarray:
    if frame.shape[:2] == target.shape[:2]:
        return frame
    return cv2.resize(
        frame,
        (target.shape[1], target.shape[0]),
        interpolation=cv2.INTER_AREA,
    )


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
    valid = [value for value in values if value is not None and math.isfinite(value)]
    if any(value == float("inf") for value in values if value is not None):
        return float("inf")
    return float(np.mean(valid)) if valid else None


def evaluate_full_reference(
    result_path: str | Path,
    ground_truth_path: str | Path,
    max_frames: int = 64,
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
    if aspect_ratio_delta > 0.01:
        raise ValueError(
            "Result and GT videos have different aspect ratios; "
            "automatic stretching would make full-reference metrics invalid."
        )

    sample_count, result_indices, ground_truth_indices, timestamps = (
        _aligned_sample_indices(result_info, ground_truth_info, max_frames)
    )
    result_frames = _read_frames(result_info.path, result_indices)
    ground_truth_frames = _read_frames(
        ground_truth_info.path,
        ground_truth_indices,
    )

    psnr_values: list[float] = []
    mse_values: list[float] = []
    ssim_values: list[float] = []
    for result_frame, ground_truth_frame in zip(
        result_frames,
        ground_truth_frames,
    ):
        ground_truth_frame = _resize_like(ground_truth_frame, result_frame)
        mse_values.append(_mse(result_frame, ground_truth_frame))
        psnr_values.append(_psnr(result_frame, ground_truth_frame))
        ssim_values.append(_ssim(result_frame, ground_truth_frame))

    resolved_device = _resolve_device(device)
    lpips_values: list[float | None] = [None] * sample_count
    lpips_error: str | None = None
    if calculate_lpips:
        raw_lpips, lpips_error = _compute_lpips(
            result_frames,
            [_resize_like(frame, result_frames[i]) for i, frame in enumerate(ground_truth_frames)],
            resolved_device,
        )
        if raw_lpips is not None:
            lpips_values = raw_lpips

    warnings: list[str] = []
    if result_info.width != ground_truth_info.width or result_info.height != ground_truth_info.height:
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
