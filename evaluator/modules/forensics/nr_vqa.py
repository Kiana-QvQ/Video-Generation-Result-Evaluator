"""No-reference video quality adapters (not VMAF).

Preferred external backends (optional): DOVER / FAST-VQA / RAPIQUE / SLEEQ when
the user installs a compatible package. Always-available fallback: a lightweight
spatial-temporal NR-VQA proxy built from OpenCV statistics.

VMAF is intentionally unsupported here because it is a full-reference metric.
"""

from __future__ import annotations

import importlib
import math
import os
import threading
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
    "external_sleeq",
)

DEFAULT_BACKEND_ORDER = (
    "external_dover",
    "external_fast_vqa",
    "external_rapique",
    "external_sleeq",
    "pyiqa_musiq",
    "pyiqa_brisque",
    "builtin_nr_vqa",
)

_EXTERNAL_MODULE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "external_dover": ("dover", "DOVER", "dover_vqa"),
    "external_fast_vqa": ("fastvqa", "fast_vqa", "FASTVQA"),
    "external_rapique": ("rapique", "RAPIQUE"),
    "external_sleeq": ("sleeq", "SLEEQ"),
}

_PYIQA_METRIC_CACHE: dict[tuple[str, str], Any] = {}
_PYIQA_METRIC_LOCK = threading.Lock()


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _safe_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else 0.0


def resolve_nr_vqa_backend_order(
    prefer_backends: Sequence[str] | None = None,
) -> list[str]:
    """Resolve backend preference: explicit arg → env → package default."""
    if prefer_backends:
        return [str(item).strip() for item in prefer_backends if str(item).strip()]
    env_value = os.environ.get("EVALUATOR_NR_VQA_BACKENDS", "").strip()
    if env_value:
        return [item.strip() for item in env_value.split(",") if item.strip()]
    return list(DEFAULT_BACKEND_ORDER)


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


def _normalize_external_score(raw: float) -> float:
    if raw <= 1.5:
        return _clamp(raw)
    if raw <= 5.0:
        # Common 1-5 MOS style.
        return _clamp((raw - 1.0) / 4.0)
    return _clamp(raw / 100.0)


def _resolve_torch_device(requested_device: str) -> str:
    requested = str(requested_device or "auto").strip().lower()
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except Exception:
        return "cpu"
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _try_pyiqa(
    frames: Sequence[np.ndarray],
    metric_name: str,
    *,
    device: str = "auto",
) -> dict[str, Any] | None:
    try:
        import torch
        import pyiqa
    except Exception:
        return None
    requested_device = _resolve_torch_device(device)
    devices = (
        (requested_device, "cpu")
        if requested_device != "cpu"
        else ("cpu",)
    )
    for resolved_device in devices:
        try:
            cache_key = (metric_name, resolved_device)
            metric = _PYIQA_METRIC_CACHE.get(cache_key)
            if metric is None:
                with _PYIQA_METRIC_LOCK:
                    metric = _PYIQA_METRIC_CACHE.get(cache_key)
                    if metric is None:
                        metric = pyiqa.create_metric(
                            metric_name,
                            device=resolved_device,
                        )
                        _PYIQA_METRIC_CACHE[cache_key] = metric
            scores: list[float] = []
            for frame in frames[: min(8, len(frames))]:
                tensor = (
                    torch.from_numpy(frame.astype(np.float32) / 255.0)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(resolved_device)
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
                score = _normalize_external_score(raw)
            return {
                "backend": f"pyiqa_{metric_name}",
                "status": "available",
                "score_0_1": score,
                "raw_score": raw,
                "device": resolved_device,
                "metrics": {"pyiqa_raw_mean": raw},
                "note": f"Optional pyiqa backend ({metric_name}).",
            }
        except Exception:
            if resolved_device == "cuda":
                _PYIQA_METRIC_CACHE.pop((metric_name, resolved_device), None)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            continue
    return None


def _call_predict(predict: Any, frames: Sequence[np.ndarray]) -> float:
    raw = predict(frames)
    if isinstance(raw, dict):
        for key in ("score_0_1", "score", "quality", "mos"):
            if key in raw and raw[key] is not None:
                return float(raw[key])
        raise ValueError("External predictor returned a dict without a score key.")
    return float(raw)


def _try_external(backend: str, frames: Sequence[np.ndarray]) -> dict[str, Any] | None:
    module_names = _EXTERNAL_MODULE_CANDIDATES.get(backend, ())
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        predict = None
        for attr in (
            "predict_video_quality",
            "evaluate",
            "infer",
            "score",
            "predict",
        ):
            candidate = getattr(module, attr, None)
            if callable(candidate):
                predict = candidate
                break
        if predict is None:
            continue
        try:
            score = _call_predict(predict, frames)
        except Exception:
            continue
        return {
            "backend": backend,
            "status": "available",
            "score_0_1": _normalize_external_score(score),
            "raw_score": float(score),
            "metrics": {},
            "note": (
                f"External optional NR-VQA backend loaded from {module_name}. "
                "VMAF is not used."
            ),
        }
    return None


def extract_nr_vqa_features(
    frames_or_video: Sequence[np.ndarray] | str | Path,
    *,
    max_frames: int = 24,
    sample_fps: float = 8.0,
    prefer_backends: Sequence[str] | None = None,
    ensemble: bool = False,
    device: str = "auto",
) -> dict[str, Any]:
    """Extract no-reference VQA features with graceful backend fallback.

    Set ``EVALUATOR_NR_VQA_BACKENDS=external_dover,builtin_nr_vqa`` to override
    the default preference order. ``ensemble=True`` averages every available
    backend in the preference list (still never uses VMAF).
    """
    frames = _frames_from_input(
        frames_or_video,
        max_frames=max_frames,
        sample_fps=sample_fps,
    )
    if not frames:
        raise ValueError("No frames available for NR-VQA.")

    order = resolve_nr_vqa_backend_order(prefer_backends)
    attempts: list[dict[str, Any]] = []
    available: list[dict[str, Any]] = []
    for backend in order:
        result = None
        if backend == "builtin_nr_vqa":
            result = _builtin_nr_vqa(frames)
        elif backend == "pyiqa_musiq":
            result = _try_pyiqa(frames, "musiq", device=device)
        elif backend == "pyiqa_brisque":
            result = _try_pyiqa(frames, "brisque", device=device)
        elif backend in _EXTERNAL_MODULE_CANDIDATES:
            result = _try_external(backend, frames)
        if result is None:
            attempts.append({"backend": backend, "status": "unavailable"})
            continue
        attempts.append({"backend": backend, "status": "available"})
        available.append(result)
        if not ensemble:
            break

    if not available:
        available = [_builtin_nr_vqa(frames)]
        attempts.append({"backend": "builtin_nr_vqa", "status": "available_fallback"})

    if ensemble and len(available) > 1:
        score = float(np.mean([item["score_0_1"] for item in available]))
        selected = {
            "backend": "ensemble_" + "+".join(item["backend"] for item in available),
            "status": "available",
            "score_0_1": _clamp(score),
            "metrics": {
                "ensemble_member_count": float(len(available)),
                "ensemble_score_std": float(
                    np.std([item["score_0_1"] for item in available])
                ),
            },
            "note": (
                "Ensemble of available no-reference VQA backends. "
                "VMAF is intentionally excluded."
            ),
        }
    else:
        selected = available[0]

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
        "available_backends": [item["backend"] for item in available],
        "device": selected.get("device", "cpu"),
        "vmaf_used": False,
        "manual_reference_required": False,
        "note": selected.get(
            "note",
            "No-reference VQA only; VMAF is intentionally not used.",
        ),
    }
