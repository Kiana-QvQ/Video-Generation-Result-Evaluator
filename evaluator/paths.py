"""Backward-compatible import path for scripts that still use evaluator.paths."""

from .core.paths import (
    ASSETS_DIR,
    PACKAGE_ROOT,
    PROFILE_FILES,
    PROJECT_ROOT,
    PROFILES_DIR,
    package_path,
    profile_path,
    project_path,
    resolve_profile,
    verify_bundled_profiles,
)

__all__ = [
    "ASSETS_DIR",
    "PACKAGE_ROOT",
    "PROFILE_FILES",
    "PROJECT_ROOT",
    "PROFILES_DIR",
    "package_path",
    "profile_path",
    "project_path",
    "resolve_profile",
    "verify_bundled_profiles",
]
