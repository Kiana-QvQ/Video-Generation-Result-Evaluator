"""No-reference video quality adapters (not VMAF).

Preferred external backends (optional): DOVER / FAST-VQA / RAPIQUE when the
user installs a compatible package. Always-available fallback: a lightweight
spatial-temporal NR-VQA proxy built from OpenCV statistics.

VMAF is intentionally unsupported here because it is a full-reference metric.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.video_metrics import sample_video_frames

NR_VQA_SCHEMA = "no_reference_vqa_v1"
SUPPORTED_BACKENDS = (
    "builtin_nr_vqa",
    "pyiqa_musiq",
    "pyiqa_brisque",
    "external_dover",
    "external_fast_vqa",
    "external_rapique",
)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _safe_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else 0.0


def _frames_from_input(
    frames_or_video: Sequence[np.ndarray] | str | Path,
    *,
    max_frames: int,
    sample_fps: float,
) -> list[np.ndarray]:
    if isinstance(frames_or_video, (str, Path)):
        _, _, _, frames = sample_video_frames(
            frames_or_video,
            max_frames,
            sample_fps,
        )
        return list(frames)
    return list(frames_or_video)[:max_frames]


def _spatial_quality(frame: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    laplacian_var = float(np.var(cv2.Laplacian(gray, cv2.CV_32F)))
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    contrast = float(np.std(gray) / 255.0)
    # Natural Image Quality-ish: mid contrast + mid-high sharpness.
    sharpness = _clamp(laplacian_var / 500.0)
    detail = _clamp(float(np.mean(gradient) / 80.0))
    contrast_score = _clamp(1.0 - abs(contrast - 0.22) / 0.22)
    score = _clamp(0.45 * sharpness + 0.35 * detail + 0.20 * contrast_score)
    return {
        "laplacian_variance": laplacian_var,
        "contrast": contrast,
        "spatial_score_0_1": score,
    }


def _temporal_quality(frames: Sequence[np.ndarray]) -> dict[str, float]:
    if len(frames) < 2:
        return {
            "flicker_mean": 0.0,
            "motion_jitter": 0.0,
            "temporal_score_0_1": 0.55,
        }
    diffs: list[float] = []
    sharp_series: list[float] = []
    for previous, current in zip(frames, frames[1:]):
        prev_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY).astype(np.float32)
        curr_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY).astype(np.float32)
        diffs.append(float(np.mean(np.abs(curr_gray - prev_gray)) / 255.0))
        sharp_series.append(
            float(np.var(cv2.Laplacian(curr_gray, cv2.CV_32F)))
        )
    flicker = float(np.std(diffs)) if diffs else 0.0
    motion = float(np.mean(diffs)) if diffs else 0.0
    sharp_jitter = (
        float(np.std(sharp_series) / max(float(np.mean(sharp_series)), 1e-6))
        if sharp_series
        else 0.0
    )
    # Penalize flicker and sharpness pumping; mild motion is fine.
    temporal = _clamp(
        1.0
        - 2.5 * flicker
        - 0.35 * sharp_jitter
        - max(0.0, motion - 0.12) * 1.5
    )
    return {
        "flicker_mean": flicker,
        "motion_mean": motion,
        "sharpness_jitter": sharp_jitter,
        "temporal_score_0_1": temporal,
    }


def _builtin_nr_vqa(frames: Sequence[np.ndarray]) -> dict[str, Any]:
    if not frames:
        raise ValueError("At least one frame is required for NR-VQA.")
    spatial_rows = [_spatial_quality(frame) for frame in frames]
    temporal = _temporal_quality(frames)
    spatial_score = _safe_mean(
        [row["spatial_score_0_1"] for row in spatial_rows]
    )
    score = _clamp(0.62 * spatial_score + 0.38 * temporal["temporal_score_0_1"])
    return {
        "backend": "builtin_nr_vqa",
        "status": "available",
        "score_0_1": score,
        "metrics": {
            "spatial_score_0_1": spatial_score,
            "temporal_score_0_1": temporal["temporal_score_0_1"],
            "flicker_mean": temporal["flicker_mean"],
            "laplacian_variance_mean": _safe_mean(
                [row["laplacian_variance"] for row in spatial_rows]
            ),
        },
        "note": (
            "Built-in no-reference spatial-temporal VQA proxy. "
            "Not VMAF (full-reference) and not a human MOS substitute."
        ),
    }


def _try_pyiqa(frames: Sequence[np.ndarray], metric_name: str) -> dict[str, Any] | None:
    try:
        import torch
        import pyiqa
    except Exception:
        return None
    try:
        metric = pyiqa.create_metric(metric_name, device="cpu")
    except Exception:
        return None
    scores: list[float] = []
    for frame in frames[: min(8, len(frames))]:
        tensor = (
            torch.from_numpy(frame.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        with torch.no_grad():
            value = float(metric(tensor).item())
        scores.append(value)
    if not scores:
        return None
    raw = float(np.mean(scores))
    # Heuristic normalization across common pyiqa ranges.
    if metric_name == "brisque":
        # Lower is better for BRISQUE.
        score = _clamp(1.0 - raw / 100.0)
    else:
        score = _clamp(raw / 100.0 if raw > 1.5 else raw)
    return {
        "backend": f"pyiqa_{metric_name}",
        "status": "available",
        "score_0_1": score,
        "raw_score": raw,
        "metrics": {"pyiqa_raw_mean": raw},
        "note": f"Optional pyiqa backend ({metric_name}).",
    }


def _try_external(module_name: str, backend: str, frames: Sequence[np.ndarray]) -> dict[str, Any] | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    predict = getattr(module, "predict_video_quality", None) or getattr(
        module,
        "evaluate",
        None,
    )
    if not callable(predict):
        return None
    try:
        raw = predict(frames)
        if isinstance(raw, dict):
            score = float(raw.get("score_0_1", raw.get("score", 0.0)))
        else:
            score = float(raw)
    except Exception:
        return None
    return {
        "backend": backend,
        "status": "available",
        "score_0_1": _clamp(score if score <= 1.5 else score / 100.0),
        "metrics": {},
        "note": f"External optional backend loaded from {module_name}.",
    }


def extract_nr_vqa_features(
    frames_or_video: Sequence[np.ndarray] | str | Path,
    *,
    max_frames: int = 24,
    sample_fps: float = 8.0,
    prefer_backends: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Extract no-reference VQA features with graceful backend fallback."""
    frames = _frames_from_input(
        frames_or_video,
        max_frames=max_frames,
        sample_fps=sample_fps,
    )
    if not frames:
        raise ValueError("No frames available for NR-VQA.")

    order = list(
        prefer_backends
        or (
            "external_dover",
            "external_fast_vqa",
            "external_rapique",
            "pyiqa_musiq",
            "pyiqa_brisque",
            "builtin_nr_vqa",
        )
    )
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for backend in order:
        result = None
        if backend == "builtin_nr_vqa":
            result = _builtin_nr_vqa(frames)
        elif backend == "pyiqa_musiq":
            result = _try_pyiqa(frames, "musiq")
        elif backend == "pyiqa_brisque":
            result = _try_pyiqa(frames, "brisque")
        elif backend == "external_dover":
            result = _try_external("dover", "external_dover", frames)
        elif backend == "external_fast_vqa":
            result = _try_external("fastvqa", "external_fast_vqa", frames)
        elif backend == "external_rapique":
            result = _try_external("rapique", "external_rapique", frames)
        if result is None:
            attempts.append({"backend": backend, "status": "unavailable"})
            continue
        attempts.append({"backend": backend, "status": "available"})
        selected = result
        break
    if selected is None:
        selected = _builtin_nr_vqa(frames)

    features = {
        "nr_vqa_score_0_1": float(selected["score_0_1"]),
        "nr_vqa_backend_code": float(
            SUPPORTED_BACKENDS.index(selected["backend"])
            if selected["backend"] in SUPPORTED_BACKENDS
            else -1.0
        ),
    }
    for key, value in selected.get("metrics", {}).items():
        features[f"nr_vqa_{key}"] = float(value)

    return {
        "schema_version": NR_VQA_SCHEMA,
        "status": selected.get("status", "available"),
        "backend": selected.get("backend"),
        "score_0_1": float(selected["score_0_1"]),
        "features": features,
        "attempts": attempts,
        "vmaf_used": False,
        "note": selected.get(
            "note",
            "No-reference VQA only; VMAF is intentionally not used.",
        ),
    }
