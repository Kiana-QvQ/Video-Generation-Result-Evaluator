#!/usr/bin/env python3
"""Sync and optionally zip a collaborator-ready evaluator bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _is_runtime_calibrator(payload: dict[str, Any]) -> bool:
    status = payload.get("status")
    return status in {"ready", "calibrated"} and (
        "mean" in payload or "slope" in payload
    )


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
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    missing: list[str] = []
    for filename, relative in SOURCE_PROFILES.items():
        source = PROJECT_ROOT / relative
        target = PROFILES_DIR / filename
        if not source.is_file():
            missing.append(relative)
            continue
        shutil.copy2(source, target)
        copied[filename] = relative
    if missing:
        raise SystemExit(
            "Missing source profiles:\n- " + "\n- ".join(missing)
        )

    calibrator = extract_runtime_calibrator()
    calibrator_target = PROFILES_DIR / CALIBRATOR_ASSET
    calibrator_target.write_text(
        json.dumps(calibrator, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    copied[CALIBRATOR_ASSET] = (
        "extracted from forensics_profiles.authenticity_calibrator "
        f"(fallback: {CALIBRATOR_REPORT})"
    )

    profile_asset = _read_json(PROFILES_DIR / "forensics_profiles.json")
    embedded = profile_asset.get("authenticity_calibrator")
    if not isinstance(embedded, dict) or not _is_runtime_calibrator(embedded):
        raise SystemExit(
            "Bundled forensics_profiles.json is missing a ready "
            "authenticity_calibrator payload."
        )
    for key in ("mean", "scale", "slope", "intercept", "feature", "status"):
        if embedded.get(key) != calibrator.get(key):
            raise SystemExit(
                "Bundled calibrator asset does not match "
                f"forensics_profiles.authenticity_calibrator.{key}"
            )

    manifest = {
        "schema_version": "evaluator_assets_manifest_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "profiles": {
            key: {
                "filename": filename,
                "source": (
                    SOURCE_PROFILES.get(filename)
                    or (
                        "extracted runtime calibrator"
                        if filename == CALIBRATOR_ASSET
                        else None
                    )
                ),
                "bytes": (PROFILES_DIR / filename).stat().st_size,
            }
            for key, filename in PROFILE_FILES.items()
            if (PROFILES_DIR / filename).is_file()
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
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copytree(
        PACKAGE_ROOT,
        staging / "evaluator",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )
    readme = staging / "README_BUNDLE.md"
    readme.write_text(
        (
            "# Evaluator collaborator bundle\n\n"
            "- Entry for yellow-box metrics: "
            "`evaluator/detail_expression_metrics.py`\n"
            "- Bundled profiles: `evaluator/assets/profiles/`\n"
            "- Runtime authenticity calibrator is both embedded in "
            "`forensics_profiles.json` and mirrored as "
            "`forensics_authenticity_calibrator.json` (payload only, "
            "not the full calibration report)\n"
            "- Does not include `web/` or `web_app.py`\n"
            "- Resolve profiles via `evaluator.core.paths.resolve_profile` / "
            "`profile_path`\n"
        ),
        encoding="utf-8",
    )
    if output.suffix.lower() == ".zip":
        archive = shutil.make_archive(str(staging), "zip", root_dir=staging)
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
    print(json.dumps(verify_bundled_profiles(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
