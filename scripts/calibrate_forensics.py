from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.forensics import (
    analyze_forensics,
    apply_probability_calibrator,
    fit_probability_calibrator,
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
AU_SUFFIXES = {".csv", ".tsv"}


def _files(root: Path, suffixes: set[str]) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
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


def _resolved_sources(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {
        str(Path(value).resolve())
        for value in values
        if isinstance(value, str) and value.strip()
    }


def _complete_generated_sources(manifest_path: Path) -> tuple[set[str], set[str]]:
    if not manifest_path.is_file():
        raise SystemExit(
            f"Missing forensic manifest required for calibration: "
            f"{manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    records = payload.get("records", [])
    complete_videos: set[str] = set()
    complete_aus: set[str] = set()
    if not isinstance(records, list):
        return complete_videos, complete_aus
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("domain") != "generated_wangxing":
            continue
        metadata = record.get("generation_metadata", {})
        if not isinstance(metadata, dict) or not metadata.get(
            "metadata_complete",
            False,
        ):
            continue
        video_path = record.get("video_path")
        au_path = record.get("au_path")
        if isinstance(video_path, str):
            complete_videos.add(str(Path(video_path).resolve()))
        if isinstance(au_path, str):
            complete_aus.add(str(Path(au_path).resolve()))
    return complete_videos, complete_aus


def _holdout(
    paths: list[Path],
    excluded: set[str],
    limit: int,
) -> list[Path]:
    candidates = [
        path
        for path in paths
        if str(path.resolve()) not in excluded
    ]
    return _uniform_limit(candidates, limit)


def _sample_key(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(path.name)
    return relative.with_suffix("").as_posix().lower()


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    labels_array = np.asarray(labels, dtype=np.int32)
    scores_array = np.asarray(scores, dtype=np.float64)
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


def _brier_score(
    labels: list[int],
    probabilities: list[float],
) -> float | None:
    if not labels or len(labels) != len(probabilities):
        return None
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    return float(np.mean((values - targets) ** 2))


def _expected_calibration_error(
    labels: list[int],
    probabilities: list[float],
    *,
    bins: int = 10,
) -> float | None:
    if not labels or len(labels) != len(probabilities):
        return None
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    error = 0.0
    for index in range(len(edges) - 1):
        lower = edges[index]
        upper = edges[index + 1]
        selected = (
            (values >= lower)
            & (
                (values < upper)
                if index < len(edges) - 2
                else (values <= upper)
            )
        )
        if not np.any(selected):
            continue
        weight = float(np.mean(selected))
        error += weight * abs(
            float(np.mean(values[selected]))
            - float(np.mean(targets[selected]))
        )
    return float(error)


def _cross_fitted_probabilities(
    real_scores: list[float],
    seedance_scores: list[float],
) -> list[float | None]:
    """Estimate calibration metrics without scoring each row with its own fit."""
    scores = [*real_scores, *seedance_scores]
    predictions: list[float | None] = []
    for index, score in enumerate(scores):
        if index < len(real_scores):
            train_real = real_scores[:index] + real_scores[index + 1 :]
            train_seedance = seedance_scores
        else:
            seedance_index = index - len(real_scores)
            train_real = real_scores
            train_seedance = (
                seedance_scores[:seedance_index]
                + seedance_scores[seedance_index + 1 :]
            )
        if not train_real or not train_seedance:
            predictions.append(None)
            continue
        fold_calibrator = fit_probability_calibrator(
            train_real,
            train_seedance,
        )
        predictions.append(
            apply_probability_calibrator(
                score,
                {**fold_calibrator, "status": "ready"},
            )
        )
    return predictions


def _manifest_paths(
    manifest_path: Path,
    *,
    domain: str,
    kind: str,
) -> list[Path]:
    return sorted(
        Path(value)
        for value in holdout_paths(
            manifest_path,
            domain=domain,
            kind=kind,
        )
    )


def _domain_samples(
    *,
    domain: str,
    au_paths: list[Path],
    video_paths: list[Path],
    au_root: Path,
    video_root: Path,
) -> list[dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    for path in au_paths:
        sample = samples.setdefault(
            _sample_key(path, au_root),
            {"domain": domain},
        )
        sample["au_path"] = path
    for path in video_paths:
        sample = samples.setdefault(
            _sample_key(path, video_root),
            {"domain": domain},
        )
        sample["video_path"] = path
    return list(samples.values())


def _score_samples(
    samples: list[dict[str, Any]],
    *,
    facial_profile: dict[str, Any] | None,
    texture_profile: dict[str, Any] | None,
    max_frames: int,
    sample_fps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    scored: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for sample in samples:
        au_path = sample.get("au_path")
        video_path = sample.get("video_path")
        try:
            report = analyze_forensics(
                facial_motion=au_path,
                facial_motion_profile=facial_profile,
                texture_detail=video_path,
                texture_detail_profile=texture_profile,
                max_frames=max_frames,
                sample_fps=sample_fps,
            )
            authenticity = report.get("authenticity", {})
            raw_score = authenticity.get(
                "raw_real_domain_evidence_0_1"
            )
            if raw_score is None:
                failures.append(
                    {
                        "sample": str(au_path or video_path),
                        "error": "raw authenticity evidence unavailable",
                    }
                )
                continue
            scored.append(
                {
                    "domain": sample["domain"],
                    "au_path": str(au_path) if au_path else None,
                    "video_path": str(video_path) if video_path else None,
                    "raw_real_domain_evidence_0_1": float(raw_score),
                    "confidence_0_1": float(
                        authenticity.get("confidence_0_1", 0.0)
                    ),
                }
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            failures.append(
                {
                    "sample": str(au_path or video_path),
                    "error": str(exc),
                }
            )
    return scored, failures


def _cached_scored_samples(
    path: Path,
    *,
    real_samples: list[dict[str, Any]],
    seedance_samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reuse scores only when they match the current explicit holdout pairs."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    cached = payload.get("samples")
    if not isinstance(cached, list):
        raise SystemExit(f"Cached calibration has no samples list: {path}")

    def key(sample: dict[str, Any]) -> tuple[str, str | None, str | None]:
        def resolved(value: Any) -> str | None:
            if isinstance(value, Path):
                value = str(value)
            if not isinstance(value, str) or not value.strip():
                return None
            return str(Path(value).resolve())

        return (
            str(sample.get("domain", "")),
            resolved(sample.get("au_path")),
            resolved(sample.get("video_path")),
        )

    allowed = {
        key(
            {
                "domain": sample["domain"],
                "au_path": sample.get("au_path"),
                "video_path": sample.get("video_path"),
            }
        )
        for sample in [*real_samples, *seedance_samples]
    }
    real: list[dict[str, Any]] = []
    seedance: list[dict[str, Any]] = []
    for sample in cached:
        if not isinstance(sample, dict):
            continue
        sample_key = key(sample)
        if sample_key not in allowed:
            raise SystemExit(
                "Cached calibration sample does not match the current "
                f"holdout manifest: {sample_key}"
            )
        raw = sample.get("raw_real_domain_evidence_0_1")
        if not isinstance(raw, (int, float)) or not np.isfinite(raw):
            raise SystemExit(
                f"Cached calibration sample has invalid raw evidence: {sample}"
            )
        normalized = {
            **sample,
            "raw_real_domain_evidence_0_1": float(raw),
            "confidence_0_1": float(sample.get("confidence_0_1", 0.0)),
        }
        if sample_key[0] == "real":
            real.append(normalized)
        elif sample_key[0] == "seedance":
            seedance.append(normalized)
    expected_real = {
        key(
            {
                "domain": "real",
                "au_path": sample.get("au_path"),
                "video_path": sample.get("video_path"),
            }
        )
        for sample in real_samples
    }
    expected_seedance = {
        key(
            {
                "domain": "seedance",
                "au_path": sample.get("au_path"),
                "video_path": sample.get("video_path"),
            }
        )
        for sample in seedance_samples
    }
    if {key(sample) for sample in real} != expected_real:
        raise SystemExit(
            "Cached calibration does not contain exactly the current real "
            "holdout samples."
        )
    if {key(sample) for sample in seedance} != expected_seedance:
        raise SystemExit(
            "Cached calibration does not contain exactly the current "
            "Seedance holdout samples."
        )
    return real, seedance


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a held-out real-versus-Seedance probability calibrator "
            "without using profile training sources."
        )
    )
    parser.add_argument(
        "--profile",
        default="outputs/forensics/forensics_profiles.json",
    )
    parser.add_argument("--real-au-root", default="data/au/MD_CL")
    parser.add_argument(
        "--seedance-au-root",
        default="data/au/WangXing_Seedance",
    )
    parser.add_argument("--real-video-root", default="data/MD_CL")
    parser.add_argument(
        "--seedance-video-root",
        default="data/WangXing_Seedance",
    )
    parser.add_argument("--max-samples-per-domain", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--sample-fps", type=float, default=8.0)
    parser.add_argument("--min-samples-per-domain", type=int, default=25)
    parser.add_argument(
        "--manifest",
        default="data/forensics/forensics_manifest.json",
    )
    parser.add_argument(
        "--holdout-manifest",
        default="data/forensics/holdout_split.json",
        help=(
            "Explicit source-video holdout manifest. If present, only "
            "these paired AU/video files are used for calibration."
        ),
    )
    parser.add_argument(
        "--allow-unknown-seedance-metadata",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Metadata is not required for "
            "the dataset-specific held-out calibrator."
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/forensics/forensics_authenticity_calibrator.json",
    )
    parser.add_argument(
        "--update-profile",
        help="Also add the calibrator payload to this profile JSON.",
    )
    parser.add_argument(
        "--reuse-samples-from",
        help=(
            "Reuse scored samples from a prior calibration JSON after "
            "verifying they match the current explicit holdout manifest. "
            "This avoids decoding the same videos again."
        ),
    )
    args = parser.parse_args()

    profile_path = project_path(args.profile)
    if not profile_path.is_file():
        raise SystemExit(f"Missing forensic profile: {profile_path}")
    profiles = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    facial_profile = profiles.get("facial_motion")
    texture_profile = profiles.get("texture_detail")

    real_au_root = project_path(args.real_au_root)
    seedance_au_root = project_path(args.seedance_au_root)
    real_video_root = project_path(args.real_video_root)
    seedance_video_root = project_path(args.seedance_video_root)

    facial_real = facial_profile.get("real", {}) if isinstance(
        facial_profile, dict
    ) else {}
    facial_seedance = facial_profile.get("seedance", {}) if isinstance(
        facial_profile, dict
    ) else {}
    texture_real = texture_profile.get("real", {}) if isinstance(
        texture_profile, dict
    ) else {}
    texture_seedance = texture_profile.get("seedance", {}) if isinstance(
        texture_profile, dict
    ) else {}

    holdout_manifest_path = project_path(args.holdout_manifest)
    if holdout_manifest_path.is_file():
        real_au_paths = _manifest_paths(
            holdout_manifest_path,
            domain="real",
            kind="au",
        )
        seedance_au_paths = _manifest_paths(
            holdout_manifest_path,
            domain="seedance",
            kind="au",
        )
        real_video_paths = _manifest_paths(
            holdout_manifest_path,
            domain="real",
            kind="video",
        )
        seedance_video_paths = _manifest_paths(
            holdout_manifest_path,
            domain="seedance",
            kind="video",
        )
    else:
        real_au_paths = _holdout(
            _files(real_au_root, AU_SUFFIXES),
            _resolved_sources(facial_real.get("source_records")),
            args.max_samples_per_domain,
        )
        seedance_au_paths = _holdout(
            _files(seedance_au_root, AU_SUFFIXES),
            _resolved_sources(facial_seedance.get("source_records")),
            args.max_samples_per_domain,
        )
        real_video_paths = _holdout(
            _files(real_video_root, VIDEO_SUFFIXES),
            _resolved_sources(texture_real.get("source_records")),
            args.max_samples_per_domain,
        )
        seedance_video_paths = _holdout(
            _files(seedance_video_root, VIDEO_SUFFIXES),
            _resolved_sources(texture_seedance.get("source_records")),
            args.max_samples_per_domain,
        )

    real_samples = _domain_samples(
        domain="real",
        au_paths=real_au_paths,
        video_paths=real_video_paths,
        au_root=real_au_root,
        video_root=real_video_root,
    )
    seedance_samples = _domain_samples(
        domain="seedance",
        au_paths=seedance_au_paths,
        video_paths=seedance_video_paths,
        au_root=seedance_au_root,
        video_root=seedance_video_root,
    )
    if args.reuse_samples_from:
        cache_path = project_path(args.reuse_samples_from)
        if not cache_path.is_file():
            raise SystemExit(f"Cached calibration was not found: {cache_path}")
        scored_real, scored_seedance = _cached_scored_samples(
            cache_path,
            real_samples=real_samples,
            seedance_samples=seedance_samples,
        )
        real_failures: list[dict[str, str]] = []
        seedance_failures: list[dict[str, str]] = []
    else:
        scored_real, real_failures = _score_samples(
            real_samples,
            facial_profile=facial_profile,
            texture_profile=texture_profile,
            max_frames=args.max_frames,
            sample_fps=args.sample_fps,
        )
        scored_seedance, seedance_failures = _score_samples(
            seedance_samples,
            facial_profile=facial_profile,
            texture_profile=texture_profile,
            max_frames=args.max_frames,
            sample_fps=args.sample_fps,
        )
    real_scores = [
        item["raw_real_domain_evidence_0_1"] for item in scored_real
    ]
    seedance_scores = [
        item["raw_real_domain_evidence_0_1"] for item in scored_seedance
    ]
    if not real_scores or not seedance_scores:
        raise SystemExit(
            "Both domains need at least one scored held-out sample; "
            "check profile sources and input roots."
        )

    calibrator = fit_probability_calibrator(real_scores, seedance_scores)
    minimum = max(2, int(args.min_samples_per_domain))
    ready = (
        len(real_scores) >= minimum
        and len(seedance_scores) >= minimum
    )
    calibrator["status"] = "ready" if ready else "provisional"
    if not ready:
        calibrator["warning"] = (
            f"Need at least {minimum} scored held-out samples per domain "
            "before this calibrator can produce probabilities."
        )
    labels = [1] * len(real_scores) + [0] * len(seedance_scores)
    scores = real_scores + seedance_scores
    calibrated_scores = [
        value
        for score in scores
        if (
            value := apply_probability_calibrator(
                score,
                {**calibrator, "status": "ready"},
            )
        )
        is not None
    ]
    brier_score = (
        _brier_score(labels, calibrated_scores)
        if len(calibrated_scores) == len(labels)
        else None
    )
    expected_calibration_error = (
        _expected_calibration_error(labels, calibrated_scores)
        if len(calibrated_scores) == len(labels)
        else None
    )
    cross_fitted_scores = _cross_fitted_probabilities(
        real_scores,
        seedance_scores,
    )
    cross_fitted_labels = [
        label
        for label, probability in zip(labels, cross_fitted_scores)
        if probability is not None
    ]
    cross_fitted_values = [
        probability
        for probability in cross_fitted_scores
        if probability is not None
    ]
    cross_fitted_brier = (
        _brier_score(cross_fitted_labels, cross_fitted_values)
        if len(cross_fitted_labels) == len(labels)
        else None
    )
    cross_fitted_ece = (
        _expected_calibration_error(
            cross_fitted_labels,
            cross_fitted_values,
        )
        if len(cross_fitted_labels) == len(labels)
        else None
    )
    payload = {
        "schema_version": "forensics_authenticity_calibration_v1",
        "status": calibrator["status"],
        "calibrator": calibrator,
        "profile": str(profile_path),
        "data_split": (
            "explicit_source_video_holdout"
            if holdout_manifest_path.is_file()
            else "held_out_excluding_profile_sources"
        ),
        "protocol": {
            "split_unit": "source video or generation batch",
            "profile_sources_excluded": True,
            "metadata_required_for_calibration": False,
            "metadata_used": False,
            "minimum_scored_samples_per_domain": minimum,
        },
        "validation": {
            "real_count": len(real_scores),
            "seedance_count": len(seedance_scores),
            "real_mean_raw_evidence": float(np.mean(real_scores)),
            "seedance_mean_raw_evidence": float(np.mean(seedance_scores)),
            "roc_auc_real_vs_seedance": _roc_auc(labels, scores),
            "brier_score": brier_score,
            "expected_calibration_error_10": expected_calibration_error,
            "cross_fitted_brier_score": cross_fitted_brier,
            "cross_fitted_expected_calibration_error_10": cross_fitted_ece,
            "warning": (
                "In-sample Brier/ECE use the final fit; cross-fitted "
                "Brier/ECE are less optimistic but still describe only this "
                "source-video holdout split, not independent final "
                "generalization."
            ),
        },
        "sampling": {
            "real_au_candidates": len(_files(real_au_root, AU_SUFFIXES)),
            "seedance_au_candidates": len(
                _files(seedance_au_root, AU_SUFFIXES)
            ),
            "real_video_candidates": len(
                _files(real_video_root, VIDEO_SUFFIXES)
            ),
            "seedance_video_candidates": len(
                _files(seedance_video_root, VIDEO_SUFFIXES)
            ),
            "holdout_manifest": str(holdout_manifest_path),
            "reused_scored_samples": bool(args.reuse_samples_from),
            "reused_samples_from": (
                str(project_path(args.reuse_samples_from))
                if args.reuse_samples_from
                else None
            ),
            "real_selected_samples": len(real_samples),
            "seedance_selected_samples": len(seedance_samples),
            "max_samples_per_domain": args.max_samples_per_domain,
            "max_frames": args.max_frames,
            "sample_fps": args.sample_fps,
        },
        "failures": (real_failures + seedance_failures)[:100],
        "samples": scored_real + scored_seedance,
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.update_profile:
        update_path = project_path(args.update_profile)
        updated = json.loads(
            update_path.read_text(encoding="utf-8-sig")
        )
        updated["authenticity_calibrator"] = calibrator
        update_path.write_text(
            json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload["validation"], ensure_ascii=False, indent=2))
    print(f"status={calibrator['status']}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
