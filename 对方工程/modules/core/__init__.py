"""Core helpers: paths, video IO, and optional holistic five-category scoring."""

from __future__ import annotations

from typing import Any

from .paths import (
    ASSETS_DIR,
    MODULES_ROOT,
    PACKAGE_ROOT,
    PROFILE_FILES,
    PROJECT_ROOT,
    PROFILES_DIR,
    WORKSPACE_ROOT,
    modules_path,
    package_path,
    profile_path,
    project_path,
    resolve_profile,
    verify_bundled_profiles,
)

__all__ = [
    "ASSETS_DIR",
    "MODULES_ROOT",
    "PACKAGE_ROOT",
    "PROFILE_FILES",
    "PROJECT_ROOT",
    "PROFILES_DIR",
    "WEIGHTS",
    "WORKSPACE_ROOT",
    "evaluate_all",
    "evaluate_full_reference",
    "modules_path",
    "package_path",
    "probe_video",
    "profile_path",
    "project_path",
    "resolve_profile",
    "verify_bundled_profiles",
]


def __getattr__(name: str) -> Any:
    if name in {"WEIGHTS", "evaluate_all"}:
        from .holistic_evaluator import WEIGHTS, evaluate_all

        return WEIGHTS if name == "WEIGHTS" else evaluate_all
    if name in {"evaluate_full_reference", "probe_video"}:
        from .video_metrics import evaluate_full_reference, probe_video

        return (
            evaluate_full_reference
            if name == "evaluate_full_reference"
            else probe_video
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
