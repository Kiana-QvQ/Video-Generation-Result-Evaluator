"""ZiJie offline face-and-mouth real-manifold scorer.

The reusable feature extractor is subject-neutral: it keeps pose-normalized
Face Mesh/AU trajectories and local brightness-centered face crops while
excluding full-frame appearance and background information.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .xiaoyue_face_manifold import (
    fit_face_manifold,
    save_pt_checkpoint,
    score_face_manifold,
)

MODEL_TYPE = "zijie_face_mouth_real_manifold_v1"
SUBJECT = "zijie"


def fit_zijie_face_manifold(
    *,
    manifest: dict[str, Any],
    cache_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Fit from real capture IDs and H3-generated training videos only."""
    return fit_face_manifold(
        manifest=manifest,
        cache_path=cache_path,
        output_path=output_path,
        subject=SUBJECT,
        model_type=MODEL_TYPE,
        minimum_real_bank=36,
        expected_ai_train=None,
    )


def score_zijie_face_manifold(
    *,
    manifest: dict[str, Any],
    profile: dict[str, Any],
    cache_path: Path,
) -> dict[str, Any]:
    """Score the capture-ID and H3-pair holdout without fitting it."""
    return score_face_manifold(
        manifest=manifest,
        profile=profile,
        cache_path=cache_path,
        subject=SUBJECT,
        model_type=MODEL_TYPE,
    )


__all__ = [
    "MODEL_TYPE",
    "SUBJECT",
    "fit_zijie_face_manifold",
    "save_pt_checkpoint",
    "score_zijie_face_manifold",
]
