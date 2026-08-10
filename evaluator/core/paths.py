from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PACKAGE_ROOT / "assets"
PROFILES_DIR = ASSETS_DIR / "profiles"


def project_path(value: str | Path) -> Path:
    """Resolve project-relative values without depending on the shell cwd."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def package_path(value: str | Path) -> Path:
    """Resolve paths relative to the ``evaluator`` package root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PACKAGE_ROOT / path


def resolve_profile(
    *candidates: str | Path,
    required: bool = False,
) -> Path | None:
    """Resolve a profile file for package and repo layouts.

    Search order for each candidate:
    1. absolute path, if it already exists
    2. project-relative path with directories, if it already exists
       (keeps training scripts on live ``data/`` / ``outputs/`` copies)
    3. ``evaluator/assets/profiles/<basename>`` (bundled collaborator assets)
    4. remaining project / package relative fallbacks
    """
    searched: list[Path] = []
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_absolute():
            searched.append(path)
            if path.is_file():
                return path
        elif len(path.parts) > 1:
            project = PROJECT_ROOT / path
            searched.append(project)
            if project.is_file():
                return project

        asset = PROFILES_DIR / path.name
        searched.append(asset)
        if asset.is_file():
            return asset

        if not path.is_absolute():
            project = PROJECT_ROOT / path
            if project not in searched:
                searched.append(project)
                if project.is_file():
                    return project
            packaged = PACKAGE_ROOT / path
            searched.append(packaged)
            if packaged.is_file():
                return packaged
        else:
            asset_fallback = PROFILES_DIR / path.name
            if asset_fallback not in searched and asset_fallback.is_file():
                return asset_fallback
    if required:
        preview = ", ".join(str(item) for item in searched[:8])
        raise FileNotFoundError(
            f"Profile not found. Looked for: {preview}"
        )
    return None


# Canonical profile keys used by packaging / collaborators.
PROFILE_FILES = {
    "wangxing_expression_profile": "wangxing_expression_profile.json",
    "wangxing_identity_profile": "wangxing_identity_profile.json",
    "wangxing_source_profile": "wangxing_source_profile.json",
    "wangxing_au_profile": "wangxing_au_profile.json",
    "forensics_profiles": "forensics_profiles.json",
    "forensics_authenticity_calibrator": (
        "forensics_authenticity_calibrator.json"
    ),
    "holdout_split": "holdout_split.json",
    "model_profile": "model_profile.json",
}


def profile_path(key: str, *, required: bool = False) -> Path | None:
    """Resolve a named bundled profile by key from ``PROFILE_FILES``."""
    filename = PROFILE_FILES.get(key)
    if filename is None:
        raise KeyError(f"Unknown profile key: {key}")
    return resolve_profile(filename, required=required)


def verify_bundled_profiles() -> dict[str, bool]:
    """Return existence map for all packaged profile assets."""
    return {
        key: (PROFILES_DIR / filename).is_file()
        for key, filename in PROFILE_FILES.items()
    }
