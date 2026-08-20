#!/usr/bin/env python3
"""Sync and optionally zip a collaborator-ready evaluator bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import ntpath
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import (  # noqa: E402
    PACKAGE_ROOT,
    PROFILE_FILES,
    PROFILES_DIR,
    verify_bundled_profiles,
)

# Canonical source locations inside this repository.
SOURCE_PROFILES = {
    "wangxing_expression_profile.json": "data/au/wangxing_expression_profile.json",
    "wangxing_identity_profile.json": "data/au/wangxing_identity_profile.json",
    "wangxing_source_profile.json": "data/au/wangxing_source_profile.json",
    "wangxing_au_profile.json": "data/au/wangxing_au_profile.json",
    "original_emotion_au_profile.json": (
        "data/au/original_emotion_au_profile.json"
    ),
    "forensics_profiles.json": "outputs/forensics/forensics_profiles.json",
    "holdout_split.json": "data/forensics/holdout_split.json",
    "model_profile.json": "config/model_profile.json",
}

CALIBRATOR_REPORT = "outputs/forensics/forensics_authenticity_calibrator.json"
CALIBRATOR_ASSET = "forensics_authenticity_calibrator.json"
RUNTIME_CALIBRATOR_KEYS = (
    "schema_version",
    "status",
    "feature",
    "calibration_method",
    "mean",
    "scale",
    "slope",
    "intercept",
)
RUNTIME_CALIBRATOR_NUMERIC_KEYS = ("mean", "scale", "slope", "intercept")
ALLOWED_TOP_LEVEL = frozenset(
    {
        "__init__.py",
        "detail_expression_metrics.py",
        "README.md",
        "modules",
    }
)
IGNORED_PACKAGE_ARTIFACTS = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".cache",
        ".docker",
        "model_cache",
    }
)
REQUIRED_PACKAGE_FILES = (
    "__init__.py",
    "detail_expression_metrics.py",
    "README.md",
    "modules/__init__.py",
    "modules/core/detail_expression_runtime.py",
    "modules/core/paths.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _portable_profile_value(value: Any) -> Any:
    """Remove machine-specific absolute paths from profile provenance."""
    if isinstance(value, dict):
        return {
            key: _portable_profile_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portable_profile_value(item) for item in value]
    if not isinstance(value, str) or not ntpath.isabs(value):
        return value

    root = str(PROJECT_ROOT.resolve())
    try:
        relative = ntpath.relpath(value, root)
    except ValueError:
        return ntpath.basename(value)
    if relative == ".." or relative.startswith(".." + ntpath.sep):
        return ntpath.basename(value)
    return relative.replace("\\", "/")


def _write_portable_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _portable_profile_value(payload),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _is_runtime_calibrator(payload: dict[str, Any]) -> bool:
    status = payload.get("status")
    if status not in {"ready", "calibrated"}:
        return False
    if any(key not in payload for key in RUNTIME_CALIBRATOR_KEYS):
        return False
    try:
        return all(
            math.isfinite(float(payload[key]))
            for key in RUNTIME_CALIBRATOR_NUMERIC_KEYS
        )
    except (TypeError, ValueError):
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_metadata(
    filename: str,
    source: str | None,
    path: Path,
) -> dict[str, Any]:
    return {
        "filename": filename,
        "source": source,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _profile_source(filename: str) -> str | None:
    return SOURCE_PROFILES.get(filename) or (
        "extracted runtime calibrator"
        if filename == CALIBRATOR_ASSET
        else None
    )


def _validate_package_layout(package_root: Path) -> None:
    missing = [
        relative
        for relative in REQUIRED_PACKAGE_FILES
        if not (package_root / relative).is_file()
    ]
    if missing:
        raise SystemExit(
            "Bundle is missing required public-entrypoint files:\n- "
            + "\n- ".join(missing)
        )
    unexpected = sorted(
        path.name
        for path in package_root.iterdir()
        if path.name not in ALLOWED_TOP_LEVEL
        and path.name not in IGNORED_PACKAGE_ARTIFACTS
    )
    if unexpected:
        raise SystemExit(
            "evaluator/ top-level must only contain "
            f"{sorted(ALLOWED_TOP_LEVEL)}. Unexpected:\n- "
            + "\n- ".join(unexpected)
        )


def _validate_profile_assets(profiles_dir: Path) -> None:
    missing = [
        filename
        for filename in PROFILE_FILES.values()
        if not (profiles_dir / filename).is_file()
    ]
    if missing:
        raise SystemExit(
            "Missing bundled profile assets:\n- " + "\n- ".join(missing)
        )

    for filename in PROFILE_FILES.values():
        try:
            _read_json(profiles_dir / filename)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(
                f"Invalid bundled profile JSON: {profiles_dir / filename}: {exc}"
            ) from exc

    calibrator = _read_json(
        profiles_dir / "forensics_authenticity_calibrator.json"
    )
    if not _is_runtime_calibrator(calibrator):
        raise SystemExit(
            "Bundled forensics_authenticity_calibrator.json is not a "
            "complete ready runtime calibrator."
        )

    forensics = _read_json(profiles_dir / "forensics_profiles.json")
    embedded = forensics.get("authenticity_calibrator")
    if not isinstance(embedded, dict) or not _is_runtime_calibrator(embedded):
        raise SystemExit(
            "Bundled forensics_profiles.json is missing a complete ready "
            "authenticity_calibrator payload."
        )
    for key in RUNTIME_CALIBRATOR_KEYS:
        if embedded.get(key) != calibrator.get(key):
            raise SystemExit(
                "Bundled calibrator asset does not match "
                f"forensics_profiles.authenticity_calibrator.{key}"
            )


def _verify_manifest(package_root: Path) -> dict[str, bool]:
    _validate_package_layout(package_root)
    profiles_dir = package_root / "modules" / "assets" / "profiles"
    _validate_profile_assets(profiles_dir)
    manifest_path = package_root / "modules" / "assets" / "MANIFEST.json"
    manifest = _read_json(manifest_path)
    entries = manifest.get("profiles")
    if not isinstance(entries, dict):
        raise SystemExit(f"Invalid profile manifest: {manifest_path}")

    for key, filename in PROFILE_FILES.items():
        entry = entries.get(key)
        path = profiles_dir / filename
        if not isinstance(entry, dict):
            raise SystemExit(
                f"Profile manifest is missing entry {key}: {manifest_path}"
            )
        if entry.get("filename") != filename:
            raise SystemExit(
                f"Profile manifest filename mismatch for {key}: "
                f"{manifest_path}"
            )
        if entry.get("bytes") != path.stat().st_size:
            raise SystemExit(
                f"Profile manifest byte count mismatch for {filename}: "
                f"{manifest_path}"
            )
        if entry.get("sha256") != _sha256_file(path):
            raise SystemExit(
                f"Profile manifest SHA-256 mismatch for {filename}: "
                f"{manifest_path}"
            )
    return verify_bundled_profiles(package_root)


def _prune_staged_profile_assets(package_root: Path) -> None:
    profiles_dir = package_root / "modules" / "assets" / "profiles"
    allowed = set(PROFILE_FILES.values())
    for path in profiles_dir.glob("*.json"):
        if path.name not in allowed:
            path.unlink()


def _verify_archive(archive: Path) -> dict[str, bool]:
    with tempfile.TemporaryDirectory(prefix="evaluator_bundle_verify_") as root:
        extract_root = Path(root)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extract_root)
        return _verify_manifest(extract_root / "evaluator")


def _remove_existing_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _safe_staging_path(path: Path) -> Path:
    resolved = path.resolve()
    protected = {
        PROJECT_ROOT.resolve(),
        PACKAGE_ROOT.resolve(),
        PROFILES_DIR.resolve(),
    }
    if resolved in protected or resolved.is_relative_to(PACKAGE_ROOT.resolve()):
        raise SystemExit(
            "Refusing to use a bundle output inside the evaluator source "
            f"package: {resolved}"
        )
    return resolved


def extract_runtime_calibrator() -> dict[str, Any]:
    """Prefer the calibrator embedded in forensics_profiles, else report.calibrator."""
    profile_path = PROJECT_ROOT / SOURCE_PROFILES["forensics_profiles.json"]
    report_path = PROJECT_ROOT / CALIBRATOR_REPORT
    if profile_path.is_file():
        embedded = _read_json(profile_path).get("authenticity_calibrator")
        if isinstance(embedded, dict) and _is_runtime_calibrator(embedded):
            return embedded
    if report_path.is_file():
        report = _read_json(report_path)
        nested = report.get("calibrator")
        if isinstance(nested, dict) and _is_runtime_calibrator(nested):
            return nested
        if _is_runtime_calibrator(report):
            return report
    raise SystemExit(
        "Missing runtime authenticity calibrator. "
        "Run calibrate_forensics.py --update-profile first."
    )


def sync_profiles() -> dict[str, str]:
    _validate_package_layout(PACKAGE_ROOT)
    expected_sources = set(PROFILE_FILES.values()) - {CALIBRATOR_ASSET}
    declared_sources = set(SOURCE_PROFILES)
    if declared_sources != expected_sources:
        raise SystemExit(
            "Profile source mapping is out of sync with PROFILE_FILES. "
            f"Missing={sorted(expected_sources - declared_sources)}, "
            f"extra={sorted(declared_sources - expected_sources)}"
        )

    source_paths = {
        filename: PROJECT_ROOT / relative
        for filename, relative in SOURCE_PROFILES.items()
    }
    missing = [
        str(path)
        for path in source_paths.values()
        if not path.is_file()
    ]
    if missing:
        raise SystemExit(
            "Missing source profiles:\n- " + "\n- ".join(missing)
        )

    # Preflight every source before changing the checked-in asset directory.
    for path in source_paths.values():
        _read_json(path)
    calibrator = extract_runtime_calibrator()
    source_forensics = _read_json(source_paths["forensics_profiles.json"])
    embedded = source_forensics.get("authenticity_calibrator")
    if not isinstance(embedded, dict) or not _is_runtime_calibrator(embedded):
        raise SystemExit(
            "Source forensics_profiles.json is missing a complete ready "
            "authenticity_calibrator payload."
        )
    for key in RUNTIME_CALIBRATOR_KEYS:
        if embedded.get(key) != calibrator.get(key):
            raise SystemExit(
                "Source calibrator does not match "
                f"forensics_profiles.authenticity_calibrator.{key}"
            )

    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for filename, source in source_paths.items():
        target = PROFILES_DIR / filename
        _write_portable_json(target, _read_json(source))
        copied[filename] = SOURCE_PROFILES[filename]

    calibrator_target = PROFILES_DIR / CALIBRATOR_ASSET
    calibrator_target.write_text(
        json.dumps(calibrator, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    copied[CALIBRATOR_ASSET] = (
        "extracted from forensics_profiles.authenticity_calibrator "
        f"(fallback: {CALIBRATOR_REPORT})"
    )

    _validate_profile_assets(PROFILES_DIR)

    manifest = {
        "schema_version": "evaluator_assets_manifest_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "profiles": {
            key: _profile_metadata(
                filename,
                _profile_source(filename),
                PROFILES_DIR / filename,
            )
            for key, filename in PROFILE_FILES.items()
        },
        "calibrator_runtime_keys": list(RUNTIME_CALIBRATOR_KEYS),
        "verify": verify_bundled_profiles(),
    }
    (PROFILES_DIR.parent / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return copied


def build_flat_host(output: Path) -> Path:
    """Write collaborator flat layout: detail_expression_metrics.py + modules/.

    Does not touch host app files (main.py / app.py / Expression / checkpoints).
    """
    # Profiles should already be bundled; sync when sources exist, otherwise
    # reuse the checked-in modules/assets/profiles copy.
    try:
        sync_profiles()
    except SystemExit as exc:
        profiles_dir = PACKAGE_ROOT / "modules" / "assets" / "profiles"
        if not profiles_dir.is_dir():
            raise
        print(f"skip sync_profiles ({exc}); using bundled profiles")
    output = output.resolve()
    package_root = PACKAGE_ROOT.resolve()
    # Windows paths are case-insensitive: ``Evaluator`` == ``evaluator``.
    if output == package_root or str(output).casefold() == str(package_root).casefold():
        raise SystemExit(
            "Refusing --flat-host into the evaluator package itself. "
            "On Windows, Evaluator/ and evaluator/ are the same folder. "
            "Use a distinct host directory such as 对方工程/."
        )
    output.mkdir(parents=True, exist_ok=True)

    modules_dst = output / "modules"
    if modules_dst.exists():
        shutil.rmtree(modules_dst)
    shutil.copytree(
        PACKAGE_ROOT / "modules",
        modules_dst,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".cache",
        ),
    )
    # Flat host imports ``from modules.core...`` (no evaluator. prefix).
    entry_src = (PACKAGE_ROOT / "detail_expression_metrics.py").read_text(
        encoding="utf-8"
    )
    entry_flat = entry_src.replace(
        "from .modules.core.detail_expression_runtime import (",
        "from modules.core.detail_expression_runtime import (",
    )
    (output / "detail_expression_metrics.py").write_text(
        entry_flat,
        encoding="utf-8",
    )
    readme = output / "EVALUATOR_FLAT_README.md"
    readme.write_text(
        "\n".join(
            [
                "# 扁平接入说明（对方宿主工程）",
                "",
                "已覆盖到对方项目根目录：",
                "",
                "- `detail_expression_metrics.py`",
                "- `modules/`",
                "",
                "不要改对方的 `main.py` / `app.py` / `Expression` / `checkpoints`。",
                "",
                "无 AU CSV 时：优先从视频自动合成 AU，对照王兴表情 profile（约 648 样本）；",
                "仅合成失败才回退 Expression/ 动作原型图。",
                "",
                "详见 modules/assets/ASSET_USAGE.md。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    profiles_dir = modules_dst / "assets" / "profiles"
    _validate_profile_assets(profiles_dir)
    required = (
        "detail_expression_metrics.py",
        "modules/__init__.py",
        "modules/core/detail_expression_runtime.py",
        "modules/core/expression_prototype_fallback.py",
        "modules/core/paths.py",
    )
    missing = [item for item in required if not (output / item).is_file()]
    if missing:
        raise SystemExit(
            "Flat host overlay missing files:\n- " + "\n- ".join(missing)
        )
    return output


def _copy_clean_package(destination: Path) -> Path:
    """Copy only the public package top-level into ``destination/evaluator``."""
    package_dst = destination / "evaluator"
    package_dst.mkdir(parents=True, exist_ok=True)
    for name in sorted(ALLOWED_TOP_LEVEL):
        source = PACKAGE_ROOT / name
        if not source.exists():
            raise SystemExit(f"Missing required package entry: {source}")
        target = package_dst / name
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    ".pytest_cache",
                    ".cache",
                    ".docker",
                    "model_cache",
                ),
            )
        else:
            shutil.copy2(source, target)
    return package_dst


def build_bundle(output: Path) -> Path:
    try:
        sync_profiles()
    except SystemExit as exc:
        print(f"skip sync_profiles ({exc}); packing bundled profiles as-is")
    output = output.resolve()
    if output.suffix.lower() == ".zip":
        staging = output.with_suffix("")
    else:
        staging = output
    staging = _safe_staging_path(staging)
    _remove_existing_path(staging)
    staging.mkdir(parents=True)
    # Zip / folder root is exactly ``evaluator/`` with the allowed top-level
    # entries. Ignore any host files that may share this folder on Windows.
    bundle_package_root = _copy_clean_package(staging)
    _prune_staged_profile_assets(bundle_package_root)
    _validate_package_layout(bundle_package_root)
    _verify_manifest(bundle_package_root)
    if output.suffix.lower() == ".zip":
        if output.exists():
            _remove_existing_path(output)
        # Archive so extracting yields ``evaluator/`` directly.
        archive = shutil.make_archive(
            str(staging),
            "zip",
            root_dir=staging,
            base_dir="evaluator",
        )
        try:
            _verify_archive(Path(archive))
        finally:
            shutil.rmtree(staging)
        return Path(archive)
    return staging / "evaluator"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync evaluator/modules/assets/profiles and optionally build a zip."
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only refresh evaluator/modules/assets/profiles from the repo.",
    )
    parser.add_argument(
        "--output",
        default="outputs/evaluator_bundle",
        help="Bundle directory or .zip path.",
    )
    parser.add_argument(
        "--flat-host",
        default="",
        help=(
            "Write flat collaborator layout (detail_expression_metrics.py + "
            "modules/) into this host project directory."
        ),
    )
    args = parser.parse_args()
    if args.sync_only:
        copied = sync_profiles()
        print(
            json.dumps(
                {"copied": copied, "verify": verify_bundled_profiles()},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.flat_host:
        path = build_flat_host(PROJECT_ROOT / args.flat_host)
        print(f"Wrote flat host overlay to {path}")
        print(
            json.dumps(
                {
                    "detail_expression_metrics": (
                        path / "detail_expression_metrics.py"
                    ).is_file(),
                    "expression_fallback": (
                        path
                        / "modules/core/expression_prototype_fallback.py"
                    ).is_file(),
                    "profiles": verify_bundled_profiles(path),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    path = build_bundle(PROJECT_ROOT / args.output)
    print(f"Wrote {path}")
    verification = (
        _verify_manifest(path)
        if path.is_dir()
        else _verify_archive(path)
    )
    print(json.dumps(verification, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
