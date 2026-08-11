from __future__ import annotations

from pathlib import Path


# ``evaluator/modules/core/paths.py`` -> parents[1]=modules, parents[2]=evaluator
MODULES_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
# Parent of the shipped ``evaluator/`` folder (full repo root, or unzip parent).
WORKSPACE_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = WORKSPACE_ROOT
ASSETS_DIR = MODULES_ROOT / "assets"
PROFILES_DIR = ASSETS_DIR / "profiles"


def project_path(value: str | Path) -> Path:
    """Resolve values relative to the workspace that contains ``evaluator/``."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def package_path(value: str | Path) -> Path:
    """Resolve paths relative to the ``evaluator`` package root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PACKAGE_ROOT / path


def modules_path(value: str | Path) -> Path:
    """Resolve paths relative to ``evaluator/modules``."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else MODULES_ROOT / path


def resolve_profile(
    *candidates: str | Path,
    required: bool = False,
) -> Path | None:
    """Resolve a profile file using package-relative assets first.

    Search order for each candidate:
    1. absolute path, if it already exists
    2. ``evaluator/modules/assets/profiles/<basename>``
    3. workspace-relative path (full-repo ``data/`` / ``outputs/`` copies)
    4. path under ``PACKAGE_ROOT`` / ``MODULES_ROOT``
    """
    searched: list[Path] = []
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_absolute():
            searched.append(path)
            if path.is_file():
                return path

        asset = PROFILES_DIR / path.name
        searched.append(asset)
        if asset.is_file():
            return asset

        if not path.is_absolute():
            workspace = WORKSPACE_ROOT / path
            searched.append(workspace)
            if workspace.is_file():
                return workspace
            packaged = PACKAGE_ROOT / path
            searched.append(packaged)
            if packaged.is_file():
                return packaged
            modular = MODULES_ROOT / path
            searched.append(modular)
            if modular.is_file():
                return modular
    if required:
        preview = ", ".join(str(item) for item in searched[:8])
        raise FileNotFoundError(
            f"Profile not found. Looked for: {preview}"
        )
    return None


PROFILE_FILES = {
    "wangxing_expression_profile": "wangxing_expression_profile.json",
    "wangxing_identity_profile": "wangxing_identity_profile.json",
    "wangxing_source_profile": "wangxing_source_profile.json",
    "wangxing_au_profile": "wangxing_au_profile.json",
    "original_emotion_au_profile": "original_emotion_au_profile.json",
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


def verify_bundled_profiles(
    package_root: str | Path | None = None,
) -> dict[str, bool]:
    """Return an existence map for profile assets in a package layout."""
    if package_root is not None:
        profiles_dir = (
            Path(package_root).expanduser() / "modules" / "assets" / "profiles"
        )
    else:
        profiles_dir = PROFILES_DIR
    return {
        key: (profiles_dir / filename).is_file()
        for key, filename in PROFILE_FILES.items()
    }
