from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.forensics import (
    build_texture_detail_profile,
    build_two_domain_facial_motion_profile,
    extract_texture_detail_features,
)
from evaluator.paths import project_path

VIDEO_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".webm",
    ".wmv",
}


def _files(root: Path, suffixes: set[str]) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _limit(paths: list[Path], limit: int) -> list[Path]:
    return paths if limit <= 0 else paths[:limit]


def _relative_sources(paths: list[Path], root: Path) -> list[str]:
    return [
        path.relative_to(root).as_posix()
        for path in paths
        if path.is_relative_to(root)
    ]


def _build_texture_domain(
    paths: list[Path],
    *,
    domain: str,
    max_frames: int,
    sample_fps: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for path in paths:
        try:
            records.append(
                extract_texture_detail_features(
                    path,
                    max_frames=max_frames,
                    sample_fps=sample_fps,
                )
            )
        except (OSError, ValueError, RuntimeError) as exc:
            skipped.append({"path": str(path), "error": str(exc)})
    if not records:
        return None, {
            "domain": domain,
            "processed_count": 0,
            "skipped_count": len(skipped),
            "skipped": skipped[:50],
        }
    return build_texture_detail_profile(records, domain=domain), {
        "domain": domain,
        "processed_count": len(records),
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build initial real-versus-Seedance forensic profiles for facial "
            "motion and texture detail."
        )
    )
    parser.add_argument("--real-au-root", required=True)
    parser.add_argument("--seedance-au-root", required=True)
    parser.add_argument(
        "--real-video-root",
        help="Optional real-video root for the texture-detail profile.",
    )
    parser.add_argument(
        "--seedance-video-root",
        help="Optional Seedance-video root for the texture-detail profile.",
    )
    parser.add_argument(
        "--output",
        default="outputs/forensics/forensics_profiles.json",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Maximum videos per texture domain; 0 means all videos.",
    )
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--sample-fps", type=float, default=8.0)
    args = parser.parse_args()

    real_au_root = project_path(args.real_au_root)
    seedance_au_root = project_path(args.seedance_au_root)
    output = project_path(args.output)
    real_au_paths = _files(real_au_root, {".csv", ".tsv"})
    seedance_au_paths = _files(seedance_au_root, {".csv", ".tsv"})
    if not real_au_paths:
        print(f"ERROR: no AU files found under {real_au_root}")
        return 1
    if not seedance_au_paths:
        print(f"ERROR: no AU files found under {seedance_au_root}")
        return 1

    try:
        facial_motion_profile = build_two_domain_facial_motion_profile(
            real_au_paths,
            seedance_au_paths,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: facial-motion profile failed: {exc}")
        return 1
    facial_motion_profile["provenance"] = {
        "real_au_root": str(real_au_root),
        "seedance_au_root": str(seedance_au_root),
        "real_au_count": len(real_au_paths),
        "seedance_au_count": len(seedance_au_paths),
        "real_au_sources": _relative_sources(real_au_paths, real_au_root),
        "seedance_au_sources": _relative_sources(
            seedance_au_paths,
            seedance_au_root,
        ),
    }

    payload: dict[str, Any] = {
        "schema_version": "forensics_profiles_v1",
        "facial_motion": facial_motion_profile,
        "texture_detail": None,
        "texture_provenance": None,
        "warnings": [],
    }
    if bool(args.real_video_root) != bool(args.seedance_video_root):
        payload["warnings"].append(
            "Both --real-video-root and --seedance-video-root are required "
            "to build the texture profile."
        )
    elif args.real_video_root and args.seedance_video_root:
        real_video_root = project_path(args.real_video_root)
        seedance_video_root = project_path(args.seedance_video_root)
        real_video_paths = _limit(
            _files(real_video_root, VIDEO_SUFFIXES),
            args.max_videos,
        )
        seedance_video_paths = _limit(
            _files(seedance_video_root, VIDEO_SUFFIXES),
            args.max_videos,
        )
        real_texture, real_report = _build_texture_domain(
            real_video_paths,
            domain="real",
            max_frames=args.max_frames,
            sample_fps=args.sample_fps,
        )
        seedance_texture, seedance_report = _build_texture_domain(
            seedance_video_paths,
            domain="seedance",
            max_frames=args.max_frames,
            sample_fps=args.sample_fps,
        )
        if real_texture and seedance_texture:
            payload["texture_detail"] = {
                "schema_version": "texture_detail_forensics_v1",
                "domain": "real_vs_seedance",
                "feature_names": real_texture["feature_names"],
                "real": {
                    key: real_texture[key]
                    for key in ("sample_count", "mean", "std", "source_records")
                },
                "seedance": {
                    key: seedance_texture[key]
                    for key in (
                        "sample_count",
                        "mean",
                        "std",
                        "source_records",
                    )
                },
            }
        else:
            payload["warnings"].append(
                "Texture profile was not built because one domain had no "
                "readable videos."
            )
        payload["texture_provenance"] = {
            "real_video_root": str(real_video_root),
            "seedance_video_root": str(seedance_video_root),
            "real_video_count": len(real_video_paths),
            "seedance_video_count": len(seedance_video_paths),
            "real_report": real_report,
            "seedance_report": seedance_report,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    print(
        "facial_motion_real={real} facial_motion_seedance={seedance} "
        "texture_profile={texture}".format(
            real=len(real_au_paths),
            seedance=len(seedance_au_paths),
            texture=payload["texture_detail"] is not None,
        )
    )
    for warning in payload["warnings"]:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
