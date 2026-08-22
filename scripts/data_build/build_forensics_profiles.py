from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.forensics import (
    build_texture_detail_profile,
    build_two_domain_facial_motion_profile,
    extract_texture_detail_features,
)
from evaluator.modules.forensics.holdout import holdout_paths
from evaluator.modules.core.paths import project_path

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
    if limit <= 0 or len(paths) <= limit:
        return paths
    if limit == 1:
        return [paths[0]]
    indexes = [
        round(index * (len(paths) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [paths[index] for index in indexes]


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
    parser.add_argument("--real-au-root", default="data/au/MD_CL")
    parser.add_argument(
        "--seedance-au-root",
        default="data/au/WangXing_Seedance",
    )
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
        "--holdout-manifest",
        help=(
            "Optional source-video holdout manifest. Listed AU/video files "
            "are excluded from profile training."
        ),
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Maximum videos per texture domain; 0 means all videos.",
    )
    parser.add_argument(
        "--max-motion-videos",
        type=int,
        default=120,
        help=(
            "Maximum AU files per facial-motion domain; 0 means all files."
        ),
    )
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--sample-fps", type=float, default=8.0)
    parser.add_argument(
        "--skip-motion",
        action="store_true",
        help="Reuse the facial_motion profile already stored in --output.",
    )
    parser.add_argument(
        "--motion-only",
        action="store_true",
        help=(
            "Rebuild facial_motion with the current feature protocol while "
            "preserving texture_detail from the existing output."
        ),
    )
    parser.add_argument(
        "--facial-only",
        action="store_true",
        help="Build only facial-motion profile; do not decode texture videos.",
    )
    parser.add_argument(
        "--authenticity-calibrator",
        help=(
            "Optional held-out calibrator JSON. Provisional calibrators are "
            "stored but ignored by the runtime scorer."
        ),
    )
    parser.add_argument(
        "--min-landmark-ratio",
        type=float,
        default=0.0,
        help=(
            "Drop AU clips below this landmark_valid_frame_ratio when "
            "building the facial-motion profile (e.g. 0.45)."
        ),
    )
    parser.add_argument(
        "--min-pose-ratio",
        type=float,
        default=0.0,
        help=(
            "Drop AU clips below this pose_normalized_frame_ratio when "
            "building the facial-motion profile (e.g. 0.35)."
        ),
    )
    args = parser.parse_args()
    if args.skip_motion and (args.motion_only or args.facial_only):
        print("ERROR: --skip-motion cannot be combined with motion-only/facial-only.")
        return 1
    if args.motion_only and args.facial_only:
        print("ERROR: --motion-only and --facial-only cannot be combined.")
        return 1

    output = project_path(args.output)
    existing_payload: dict[str, Any] = {}
    if output.is_file():
        try:
            loaded = json.loads(output.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                existing_payload = loaded
        except (OSError, json.JSONDecodeError):
            existing_payload = {}
    real_au_paths: list[Path] = []
    seedance_au_paths: list[Path] = []
    excluded_real_au: set[str] = set()
    excluded_seedance_au: set[str] = set()
    excluded_real_videos: set[str] = set()
    excluded_seedance_videos: set[str] = set()
    if args.holdout_manifest:
        excluded_real_au = holdout_paths(
            args.holdout_manifest,
            domain="real",
            kind="au",
        )
        excluded_seedance_au = holdout_paths(
            args.holdout_manifest,
            domain="seedance",
            kind="au",
        )
        excluded_real_videos = holdout_paths(
            args.holdout_manifest,
            domain="real",
            kind="video",
        )
        excluded_seedance_videos = holdout_paths(
            args.holdout_manifest,
            domain="seedance",
            kind="video",
        )
    if args.skip_motion:
        if not output.is_file():
            print(f"ERROR: --skip-motion requires an existing output: {output}")
            return 1
        try:
            existing_payload = json.loads(
                output.read_text(encoding="utf-8-sig")
            )
            facial_motion_profile = existing_payload["facial_motion"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            print(f"ERROR: unable to reuse facial-motion profile: {exc}")
            return 1
    else:
        real_au_root = project_path(args.real_au_root)
        seedance_au_root = project_path(args.seedance_au_root)
        real_au_paths = _files(real_au_root, {".csv", ".tsv"})
        seedance_au_paths = _files(seedance_au_root, {".csv", ".tsv"})
        if args.holdout_manifest:
            real_au_paths = [
                path
                for path in real_au_paths
                if str(path.resolve()) not in excluded_real_au
            ]
            seedance_au_paths = [
                path
                for path in seedance_au_paths
                if str(path.resolve()) not in excluded_seedance_au
            ]
        if not real_au_paths:
            print(f"ERROR: no AU files found under {real_au_root}")
            return 1
        if not seedance_au_paths:
            print(f"ERROR: no AU files found under {seedance_au_root}")
            return 1
        real_au_paths = _limit(real_au_paths, args.max_motion_videos)
        seedance_au_paths = _limit(
            seedance_au_paths,
            args.max_motion_videos,
        )

        print(
            f"Building facial-motion profile from "
            f"{len(real_au_paths)} real + {len(seedance_au_paths)} seedance AU files...",
            flush=True,
        )
        try:
            facial_motion_profile = build_two_domain_facial_motion_profile(
                real_au_paths,
                seedance_au_paths,
                min_landmark_ratio=args.min_landmark_ratio,
                min_pose_ratio=args.min_pose_ratio,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"ERROR: facial-motion profile failed: {exc}")
            return 1
        facial_motion_profile["provenance"] = {
            "real_au_root": str(real_au_root),
            "seedance_au_root": str(seedance_au_root),
            "real_au_count": len(real_au_paths),
            "seedance_au_count": len(seedance_au_paths),
            "holdout_manifest": (
                str(project_path(args.holdout_manifest))
                if args.holdout_manifest
                else None
            ),
            "holdout_excluded_real_au_count": len(excluded_real_au),
            "holdout_excluded_seedance_au_count": len(
                excluded_seedance_au
            ),
            "min_landmark_ratio": float(args.min_landmark_ratio),
            "min_pose_ratio": float(args.min_pose_ratio),
            "max_motion_videos": int(args.max_motion_videos),
            "real_au_sources": _relative_sources(real_au_paths, real_au_root),
            "seedance_au_sources": _relative_sources(
                seedance_au_paths,
                seedance_au_root,
            ),
        }

    if args.motion_only or args.facial_only:
        args.real_video_root = None
        args.seedance_video_root = None

    payload: dict[str, Any] = {
        "schema_version": "forensics_profiles_v1",
        "facial_motion": facial_motion_profile,
        "texture_detail": (
            existing_payload.get("texture_detail")
            if args.motion_only
            else None
        ),
        "texture_provenance": (
            existing_payload.get("texture_provenance")
            if args.motion_only
            else None
        ),
        "warnings": (
            list(existing_payload.get("warnings", []))
            if args.motion_only
            and isinstance(existing_payload.get("warnings"), list)
            else []
        ),
    }
    if args.motion_only:
        payload["warnings"].append(
            "Facial-motion profile rebuilt independently; texture profile "
            "was preserved from the previous output."
        )
    if args.facial_only:
        payload["warnings"].append(
            "Facial-only profile: texture branch intentionally omitted."
        )
    if args.authenticity_calibrator:
        calibrator_path = project_path(args.authenticity_calibrator)
        if not calibrator_path.is_file():
            payload["warnings"].append(
                f"Authenticity calibrator was not found: {calibrator_path}"
            )
        else:
            calibrator_payload = json.loads(
                calibrator_path.read_text(encoding="utf-8-sig")
            )
            calibrator = calibrator_payload.get(
                "calibrator",
                calibrator_payload,
            )
            if isinstance(calibrator, dict):
                payload["authenticity_calibrator"] = calibrator
            else:
                payload["warnings"].append(
                    "Authenticity calibrator JSON did not contain an object."
                )
    if bool(args.real_video_root) != bool(args.seedance_video_root):
        payload["warnings"].append(
            "Both --real-video-root and --seedance-video-root are required "
            "to build the texture profile."
        )
    elif args.real_video_root and args.seedance_video_root:
        real_video_root = project_path(args.real_video_root)
        seedance_video_root = project_path(args.seedance_video_root)
        real_video_candidates = _files(real_video_root, VIDEO_SUFFIXES)
        seedance_video_candidates = _files(
            seedance_video_root,
            VIDEO_SUFFIXES,
        )
        if args.holdout_manifest:
            real_video_candidates = [
                path
                for path in real_video_candidates
                if str(path.resolve()) not in excluded_real_videos
            ]
            seedance_video_candidates = [
                path
                for path in seedance_video_candidates
                if str(path.resolve()) not in excluded_seedance_videos
            ]
        real_video_paths = _limit(
            real_video_candidates,
            args.max_videos,
        )
        seedance_video_paths = _limit(
            seedance_video_candidates,
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
            "max_videos_per_domain": args.max_videos,
            "max_frames_per_video": args.max_frames,
            "sample_fps": args.sample_fps,
            "holdout_manifest": (
                str(project_path(args.holdout_manifest))
                if args.holdout_manifest
                else None
            ),
            "holdout_excluded_real_video_count": len(
                excluded_real_videos
            )
            if args.holdout_manifest
            else 0,
            "holdout_excluded_seedance_video_count": len(
                excluded_seedance_videos
            )
            if args.holdout_manifest
            else 0,
            "real_report": real_report,
            "seedance_report": seedance_report,
        }
        if args.max_videos > 0:
            payload["warnings"].append(
                "Texture profile is a sampled preliminary calibration set; "
                "expand it before production use."
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    payload["warnings"] = list(dict.fromkeys(payload["warnings"]))
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
