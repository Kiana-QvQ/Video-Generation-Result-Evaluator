from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.forensics import score_facial_motion
from evaluator.paths import project_path


def _files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.csv"))


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
    positives = scores_array[labels_array == 1]
    negatives = scores_array[labels_array == 0]
    if positives.size == 0 or negatives.size == 0:
        return None
    comparisons = positives[:, None] - negatives[None, :]
    return float(
        (np.sum(comparisons > 0.0) + 0.5 * np.sum(comparisons == 0.0))
        / comparisons.size
    )


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _validate_motion(
    profile: dict[str, Any],
    *,
    real_paths: list[Path],
    seedance_paths: list[Path],
) -> dict[str, Any]:
    labels: list[int] = []
    scores: list[float] = []
    domain_values: dict[str, list[float]] = {
        "real": [],
        "seedance": [],
    }
    domain_landmark_coverage: dict[str, list[float]] = {
        "real": [],
        "seedance": [],
    }
    failures: list[dict[str, str]] = []
    for label, paths in (("real", real_paths), ("seedance", seedance_paths)):
        for path in paths:
            try:
                result = score_facial_motion(
                    path,
                    profile,
                )
                metrics = result["metrics"]
                score = metrics.get("raw_real_domain_evidence_0_1")
                if score is None:
                    score = metrics.get("real_capture_likelihood_0_1")
                if score is None:
                    continue
                domain_values[label].append(float(score))
                domain_landmark_coverage[label].append(
                    float(metrics.get("landmark_valid_frame_ratio", 0.0))
                )
                labels.append(1 if label == "real" else 0)
                scores.append(float(score))
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append({"path": str(path), "error": str(exc)})
    real_values = domain_values["real"]
    seedance_values = domain_values["seedance"]
    return {
        "status": "diagnostic_train_domain_only",
        "real": {
            "count": len(real_values),
            "raw_real_domain_evidence": _summary(real_values),
            "landmark_valid_frame_ratio": _summary(
                domain_landmark_coverage["real"]
            ),
        },
        "seedance": {
            "count": len(seedance_values),
            "raw_real_domain_evidence": _summary(seedance_values),
            "landmark_valid_frame_ratio": _summary(
                domain_landmark_coverage["seedance"]
            ),
        },
        "separation_mean_real_minus_seedance": (
            float(np.mean(real_values) - np.mean(seedance_values))
            if real_values and seedance_values
            else None
        ),
        "roc_auc_real_vs_seedance": _roc_auc(labels, scores),
        "failed_count": len(failures),
        "failed_preview": failures[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose facial-motion real-versus-Seedance profile separation "
            "on existing AU CSV data."
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
    parser.add_argument(
        "--max-samples-per-domain",
        type=int,
        default=20,
        help="Uniformly sampled diagnostic files per domain; 0 means all.",
    )
    parser.add_argument(
        "--output",
        default="outputs/forensics/forensics_validation.json",
    )
    args = parser.parse_args()

    profile_path = project_path(args.profile)
    if not profile_path.is_file():
        raise SystemExit(f"Missing profile: {profile_path}")
    profiles = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    real_paths = _uniform_limit(
        _files(project_path(args.real_au_root)),
        args.max_samples_per_domain,
    )
    seedance_paths = _uniform_limit(
        _files(project_path(args.seedance_au_root)),
        args.max_samples_per_domain,
    )
    result = {
        "schema_version": "forensics_validation_v1",
        "profile": str(profile_path),
        "data_split": "not_held_out",
        "warning": (
            "This is a diagnostic run on profile-domain data. It is not a "
            "cross-batch generalization result."
        ),
        "facial_motion": _validate_motion(
            profiles["facial_motion"],
            real_paths=real_paths,
            seedance_paths=seedance_paths,
        ),
        "sampling": {
            "real_csv_count": len(real_paths),
            "seedance_csv_count": len(seedance_paths),
            "max_samples_per_domain": args.max_samples_per_domain,
        },
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    output.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
