"""Core evaluation helpers (video IO, holistic five-category scoring)."""

from .holistic_evaluator import WEIGHTS, evaluate_all
from .paths import (
    PACKAGE_ROOT,
    PROJECT_ROOT,
    profile_path,
    project_path,
    resolve_profile,
    verify_bundled_profiles,
)
from .video_metrics import evaluate_full_reference, probe_video

__all__ = [
    "WEIGHTS",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "evaluate_all",
    "evaluate_full_reference",
    "probe_video",
    "profile_path",
    "project_path",
    "resolve_profile",
    "verify_bundled_profiles",
]
