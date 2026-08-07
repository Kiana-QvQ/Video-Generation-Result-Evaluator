from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .facial_motion import score_facial_motion
from .texture_detail import score_texture_detail

FORENSICS_REPORT_SCHEMA = "video_forensics_report_v1"


def _available(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def analyze_forensics(
    *,
    facial_motion: dict[str, Any] | str | Path | None = None,
    facial_motion_profile: dict[str, Any] | None = None,
    texture_detail: dict[str, Any] | Sequence[np.ndarray] | str | Path | None = None,
    texture_detail_profile: dict[str, Any] | None = None,
    face_boxes: Sequence[tuple[int, int, int, int] | None] | None = None,
    max_frames: int = 64,
    sample_fps: float = 8.0,
) -> dict[str, Any]:
    """Run either branch or both and keep their evidence separate."""
    facial_result = None
    if facial_motion is not None:
        facial_result = score_facial_motion(
            facial_motion,
            facial_motion_profile or {},
        )

    texture_result = None
    if texture_detail is not None:
        texture_result = score_texture_detail(
            texture_detail,
            texture_detail_profile,
            face_boxes=face_boxes,
            max_frames=max_frames,
            sample_fps=sample_fps,
        )

    likelihoods = []
    if facial_result is not None:
        likelihoods.append(
            facial_result["metrics"].get("real_capture_likelihood_0_1")
        )
    if texture_result is not None:
        likelihoods.append(
            texture_result["metrics"].get("real_capture_likelihood_0_1")
        )
    available_likelihoods = _available(likelihoods)
    fused = (
        float(np.mean(available_likelihoods))
        if available_likelihoods
        else None
    )
    calibrated = len(available_likelihoods) == len(likelihoods) and bool(
        likelihoods
    )
    return {
        "schema_version": FORENSICS_REPORT_SCHEMA,
        "status": "calibrated" if calibrated else "features_only",
        "branches": {
            "facial_motion": facial_result,
            "texture_detail": texture_result,
        },
        "fusion": {
            "real_capture_likelihood_0_1": fused,
            "branch_count": len([item for item in (facial_result, texture_result) if item]),
            "calibrated_branch_count": len(available_likelihoods),
            "method": (
                "mean_of_calibrated_branch_likelihoods"
                if calibrated
                else "not_calibrated"
            ),
            "warning": (
                None
                if calibrated
                else (
                    "At least one branch lacks a held-out real-versus-Seedance "
                    "profile. Do not interpret the fused result as a detector."
                )
            ),
        },
    }
