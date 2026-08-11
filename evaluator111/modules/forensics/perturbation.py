"""Automatic perturbation probes for score-stability checks.

Inject blur / noise / flicker / frame-drop / temporal shuffle / landmark jitter
and verify that authenticity / quality scores move in the expected direction.

These tests validate automatic consistency and robustness. They do not prove
equivalence to human perception.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import cv2
import numpy as np

PERTURBATION_SCHEMA = "perturbation_robustness_v1"


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def perturb_frames_blur(
    frames: Sequence[np.ndarray],
    *,
    kernel_size: int = 11,
) -> list[np.ndarray]:
    size = max(3, int(kernel_size) | 1)
    return [cv2.GaussianBlur(frame, (size, size), 0) for frame in frames]


def perturb_frames_noise(
    frames: Sequence[np.ndarray],
    *,
    sigma: float = 28.0,
    seed: int = 0,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    outputs: list[np.ndarray] = []
    for frame in frames:
        noise = rng.normal(0.0, sigma, size=frame.shape)
        noisy = np.clip(frame.astype(np.float32) + noise, 0, 255)
        outputs.append(noisy.astype(np.uint8))
    return outputs


def perturb_frames_flicker(
    frames: Sequence[np.ndarray],
    *,
    amplitude: float = 0.35,
) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        scale = 1.0 + amplitude * ((-1.0) ** index)
        outputs.append(
            np.clip(frame.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        )
    return outputs


def perturb_frames_drop(
    frames: Sequence[np.ndarray],
    *,
    drop_ratio: float = 0.35,
) -> list[np.ndarray]:
    if not frames:
        return []
    keep = max(2, int(round(len(frames) * (1.0 - drop_ratio))))
    indexes = np.linspace(0, len(frames) - 1, keep)
    return [frames[int(round(index))] for index in indexes]


def perturb_frames_temporal_shuffle(
    frames: Sequence[np.ndarray],
    *,
    seed: int = 0,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(frames))
    rng.shuffle(order)
    return [frames[int(index)] for index in order]


def perturb_landmark_csv_rows(
    rows: list[dict[str, str]],
    *,
    sigma: float = 0.02,
    seed: int = 0,
) -> list[dict[str, str]]:
    """Jitter ``lm_mp_*`` landmark coordinates in-place-copied rows."""
    rng = np.random.default_rng(seed)
    outputs: list[dict[str, str]] = []
    for row in rows:
        cloned = dict(row)
        for key, value in row.items():
            key_l = str(key).lower()
            if not (
                key_l.startswith("lm_mp_")
                and (key_l.endswith("_x") or key_l.endswith("_y"))
            ):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            cloned[key] = str(numeric + float(rng.normal(0.0, sigma)))
        outputs.append(cloned)
    return outputs


DEFAULT_FRAME_PERTURBATIONS: dict[str, Callable[..., list[np.ndarray]]] = {
    "blur": perturb_frames_blur,
    "noise": perturb_frames_noise,
    "flicker": perturb_frames_flicker,
    "frame_drop": perturb_frames_drop,
    "temporal_shuffle": perturb_frames_temporal_shuffle,
}


def evaluate_score_response(
    clean_score: float,
    perturbed_score: float,
    *,
    expect_decrease: bool = True,
    min_drop: float = 0.02,
) -> dict[str, Any]:
    delta = float(perturbed_score) - float(clean_score)
    if expect_decrease:
        passed = delta <= -abs(min_drop)
        expected = "score_should_decrease"
    else:
        passed = abs(delta) <= abs(min_drop)
        expected = "score_should_stay_stable"
    return {
        "clean_score": float(clean_score),
        "perturbed_score": float(perturbed_score),
        "delta": delta,
        "expected": expected,
        "passed": bool(passed),
        "min_drop": float(min_drop),
    }


def run_frame_perturbation_battery(
    frames: Sequence[np.ndarray],
    score_fn: Callable[[Sequence[np.ndarray]], float],
    *,
    perturbations: dict[str, Callable[..., list[np.ndarray]]] | None = None,
    min_drop: float = 0.02,
) -> dict[str, Any]:
    """Score clean frames and each perturbation; expect quality to drop."""
    if len(frames) < 2:
        raise ValueError("Need at least two frames for perturbation tests.")
    clean_score = float(score_fn(frames))
    battery = perturbations or DEFAULT_FRAME_PERTURBATIONS
    results: dict[str, Any] = {}
    for name, perturb in battery.items():
        perturbed = perturb(frames)
        perturbed_score = float(score_fn(perturbed))
        results[name] = evaluate_score_response(
            clean_score,
            perturbed_score,
            expect_decrease=True,
            min_drop=min_drop,
        )
    passed = sum(1 for item in results.values() if item["passed"])
    return {
        "schema_version": PERTURBATION_SCHEMA,
        "clean_score": clean_score,
        "results": results,
        "passed_count": passed,
        "total_count": len(results),
        "pass_ratio": passed / max(len(results), 1),
        "note": (
            "Automatic perturbation robustness probe. Passing means scores "
            "move in the expected direction under synthetic degradations; it "
            "does not prove human-MOS equivalence."
        ),
    }
