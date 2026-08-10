#!/usr/bin/env python3
"""Sync and optionally zip a collaborator-ready evaluator bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.core.paths import (  # noqa: E402
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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


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
    profiles_dir = package_root / "assets" / "profiles"
    _validate_profile_assets(profiles_dir)
    manifest_path = package_root / "assets" / "MANIFEST.json"
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
    profiles_dir = package_root / "assets" / "profiles"
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
        shutil.copy2(source, target)
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


def build_bundle(output: Path) -> Path:
    sync_profiles()
    output = output.resolve()
    if output.suffix.lower() == ".zip":
        staging = output.with_suffix("")
    else:
        staging = output
    staging = _safe_staging_path(staging)
    _remove_existing_path(staging)
    staging.mkdir(parents=True)
    shutil.copytree(
        PACKAGE_ROOT,
        staging / "evaluator",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )
    bundle_package_root = staging / "evaluator"
    _prune_staged_profile_assets(bundle_package_root)
    _verify_manifest(bundle_package_root)
    readme = staging / "README_BUNDLE.md"
    readme.write_text(
        (
            "# Evaluator collaborator bundle\n\n"
            "- Entry for yellow-box metrics: "
            "`evaluator/detail_expression_metrics.py`\n"
            "- Bundled profiles: `evaluator/assets/profiles/`\n"
            "- Includes the original AU emotion profile for automatic "
            "emotion classification\n"
            "- Runtime authenticity calibrator is both embedded in "
            "`forensics_profiles.json` and mirrored as "
            "`forensics_authenticity_calibrator.json` (payload only, "
            "not the full calibration report)\n"
            "- Does not include `web/`, `web_app.py`, or repo-root "
            "`backends/` (ViCLIP/ETVA/VBench)\n"
            "- Resolve profiles via `evaluator.core.paths.resolve_profile` / "
            "`profile_path`\n"
        ),
        encoding="utf-8",
    )
    if output.suffix.lower() == ".zip":
        if output.exists():
            _remove_existing_path(output)
        archive = shutil.make_archive(str(staging), "zip", root_dir=staging)
        try:
            _verify_archive(Path(archive))
        finally:
            shutil.rmtree(staging)
        return Path(archive)
    return staging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync evaluator/assets/profiles and optionally build a zip."
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only refresh evaluator/assets/profiles from the repo.",
    )
    parser.add_argument(
        "--output",
        default="outputs/evaluator_bundle",
        help="Bundle directory or .zip path.",
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
    path = build_bundle(PROJECT_ROOT / args.output)
    print(f"Wrote {path}")
    bundle_root = path.parent / "evaluator" if path.is_dir() else None
    verification = (
        _verify_manifest(bundle_root)
        if bundle_root is not None
        else _verify_archive(path)
    )
    print(json.dumps(verification, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
