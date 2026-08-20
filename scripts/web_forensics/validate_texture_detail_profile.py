from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.forensics import score_texture_detail
from evaluator.modules.core.paths import project_path

VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".wmv"}


def _files(root: Path) -> list[Path]:
    wanted = {suffix.lower() for suffix in VIDEO_SUFFIXES}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in wanted
    )


def _uniform_limit(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    if limit == 1:
        return [paths[0]]
    indexes = [
        round(index * (len(paths) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [paths[index] for index in indexes]


def _roc_auc(labels: Iterable[int], scores: Iterable[float]) -> float | None:
    labels_array = np.asarray(list(labels), dtype=np.int32)
    scores_array = np.asarray(list(scores), dtype=np.float64)
    positive = scores_array[labels_array == 1]
    negative = scores_array[labels_array == 0]
    if positive.size == 0 or negative.size == 0:
        return None
    differences = positive[:, None] - negative[None, :]
    return float(
        (
            np.sum(differences > 0.0)
            + 0.5 * np.sum(differences == 0.0)
        )
        / differences.size
    )


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
    }


def _score_domain(
    paths: list[Path],
    profile: dict[str, Any],
    *,
    label: str,
    max_frames: int,
    sample_fps: float,
) -> tuple[list[float], list[dict[str, Any]]]:
    scores: list[float] = []
    failures: list[dict[str, Any]] = []
    for path in paths:
        try:
            result = score_texture_detail(
                path,
                profile,
                max_frames=max_frames,
                sample_fps=sample_fps,
                detect_faces=True,
            )
            score = result["metrics"].get(
                "raw_real_domain_evidence_0_1"
            )
            if score is None:
                score = result["metrics"].get(
                    "real_capture_likelihood_0_1"
                )
            if score is None:
                failures.append(
                    {"path": str(path), "error": "score unavailable"}
                )
                continue
            scores.append(float(score))
        except (OSError, ValueError, RuntimeError) as exc:
            failures.append({"path": str(path), "error": str(exc)})
    return scores, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate texture/detail scores on videos excluded from the "
            "texture profile calibration set."
        )
    )
    parser.add_argument(
        "--profile",
        default="outputs/forensics/forensics_profiles.json",
    )
    parser.add_argument("--real-video-root", default="data/MD_CL")
    parser.add_argument(
        "--seedance-video-root",
        default="data/WangXing_Seedance",
    )
    parser.add_argument("--max-videos-per-domain", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument(
        "--output",
        default="outputs/forensics/texture_validation.json",
    )
    args = parser.parse_args()

    profile_path = project_path(args.profile)
    profiles = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    profile = profiles.get("texture_detail")
    if not isinstance(profile, dict):
        raise SystemExit("The profile has no texture_detail two-domain profile.")

    train_real = {
        str(Path(path).resolve())
        for path in profile.get("real", {}).get("source_records", [])
    }
    train_seedance = {
        str(Path(path).resolve())
        for path in profile.get("seedance", {}).get("source_records", [])
    }
    real_candidates = [
        path
        for path in _files(project_path(args.real_video_root))
        if str(path.resolve()) not in train_real
    ]
    seedance_candidates = [
        path
        for path in _files(project_path(args.seedance_video_root))
        if str(path.resolve()) not in train_seedance
    ]
    real_paths = _uniform_limit(real_candidates, args.max_videos_per_domain)
    seedance_paths = _uniform_limit(
        seedance_candidates,
        args.max_videos_per_domain,
    )
    real_scores, real_failures = _score_domain(
        real_paths,
        profile,
        label="real",
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
    )
    seedance_scores, seedance_failures = _score_domain(
        seedance_paths,
        profile,
        label="seedance",
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
    )
    labels = [1] * len(real_scores) + [0] * len(seedance_scores)
    scores = real_scores + seedance_scores
    result = {
        "schema_version": "texture_detail_validation_v1",
        "data_split": "video_holdout_excluding_profile_sources",
        "profile": str(profile_path),
        "real": {
            **_summary(real_scores),
            "raw_evidence_semantics": True,
        },
        "seedance": {
            **_summary(seedance_scores),
            "raw_evidence_semantics": True,
        },
        "separation_mean_real_minus_seedance": (
            float(np.mean(real_scores) - np.mean(seedance_scores))
            if real_scores and seedance_scores
            else None
        ),
        "roc_auc_real_vs_seedance": _roc_auc(labels, scores),
        "sampling": {
            "real_candidates": len(real_candidates),
            "seedance_candidates": len(seedance_candidates),
            "real_selected": len(real_paths),
            "seedance_selected": len(seedance_paths),
            "max_frames": args.max_frames,
            "sample_fps": args.sample_fps,
        },
        "failures": (real_failures + seedance_failures)[:20],
        "warning": (
            "This is a small video holdout. Increase the sample size and "
            "repeat by generation batch before production use."
        ),
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    output.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
