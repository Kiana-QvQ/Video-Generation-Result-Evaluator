from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


AU_PROFILE_SCHEMA = "wangxing_au_profile_v1"
AU_CLASSIFIER_SCHEMA = "au_leakage_classifier_v1"
DEFAULT_AU_IDS = (1, 4, 6, 12, 15, 25)
DEFAULT_ACTIVE_THRESHOLD = 0.20
DEFAULT_WINDOW_SIZE = 32
DEFAULT_WINDOW_STRIDE = 16


def _canonical_au_id(value: str) -> int | None:
    match = re.search(
        r"\bau[\s_-]*0*(\d{1,2})(?!\d)",
        str(value),
        re.IGNORECASE,
    )
    if not match:
        return None
    au_id = int(match.group(1))
    return au_id if 1 <= au_id <= 45 else None


def _column_priority(name: str) -> int:
    normalized = str(name).lower().replace(" ", "")
    if "_r" in normalized or "intensity" in normalized:
        return 0
    if normalized.endswith("_c") or "presence" in normalized:
        return 2
    return 1


def _parse_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def load_au_table(
    path: str | Path,
    au_ids: Iterable[int] = DEFAULT_AU_IDS,
    *,
    intensity_scale: float = 5.0,
) -> tuple[np.ndarray, tuple[int, ...], dict[str, Any]]:
    """Load AU intensities from LibreFace/OpenFace-like CSV or TSV output."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        delimiter = "\t" if "\t" in sample.splitlines()[0] else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        candidates: dict[int, list[tuple[int, str]]] = {}
        for name in fieldnames:
            au_id = _canonical_au_id(name)
            if au_id is None:
                continue
            candidates.setdefault(au_id, []).append(
                (_column_priority(name), name)
            )

        requested = tuple(int(au_id) for au_id in au_ids)
        selected: dict[int, str] = {}
        for au_id in requested:
            choices = sorted(candidates.get(au_id, []))
            if choices:
                selected[au_id] = choices[0][1]
        if not selected:
            raise ValueError(
                f"No AU intensity columns found in {path}. "
                "Expected names such as AU01_r, AU12_r, AU25."
            )

        rows: list[list[float]] = []
        for row in reader:
            values = [
                _parse_float(row.get(selected[au_id], 0.0))
                for au_id in requested
            ]
            rows.append(values)

    if not rows:
        raise ValueError(f"AU file contains no rows: {path}")
    matrix = np.asarray(rows, dtype=np.float32)
    if intensity_scale > 1.0 and float(np.max(matrix)) > 1.0 + 1e-6:
        matrix = matrix / float(intensity_scale)
    matrix = np.clip(matrix, 0.0, 1.0)
    return matrix, requested, {
        "path": str(path),
        "delimiter": delimiter,
        "selected_columns": {
            str(au_id): selected.get(au_id) for au_id in requested
        },
        "available_au_ids": sorted(candidates),
    }


def _upper_triangle_pairs(count: int) -> list[tuple[int, int]]:
    return [
        (left, right)
        for left in range(count)
        for right in range(left + 1, count)
    ]


def au_summary(
    sequence: np.ndarray,
    *,
    active_threshold: float = DEFAULT_ACTIVE_THRESHOLD,
) -> np.ndarray:
    """Return distribution and AU co-occurrence features for one clip."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[0] == 0:
        raise ValueError("AU sequence must have shape [frames, aus].")
    active = sequence >= float(active_threshold)
    pairs = _upper_triangle_pairs(sequence.shape[1])
    cooccurrence = np.asarray(
        [
            float(np.mean(active[:, left] & active[:, right]))
            for left, right in pairs
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            sequence.mean(axis=0),
            sequence.std(axis=0),
            sequence.max(axis=0),
            active.mean(axis=0),
            cooccurrence,
        ]
    ).astype(np.float32)


def _shrink_covariance(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    dimension = samples.shape[1]
    if samples.shape[0] < 2:
        variance = np.ones(dimension, dtype=np.float64) * 0.05
        return np.diag(variance)
    covariance = np.cov(samples, rowvar=False)
    covariance = np.atleast_2d(covariance)
    diagonal = np.diag(np.diag(covariance))
    shrinkage = min(0.75, 8.0 / float(samples.shape[0]))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    minimum_variance = 0.01 if samples.shape[0] < 8 else 1e-4
    covariance += np.eye(dimension, dtype=np.float64) * minimum_variance
    return covariance


def _mahalanobis(
    samples: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    delta = samples - np.asarray(mean, dtype=np.float64)
    inverse = np.linalg.pinv(np.asarray(covariance, dtype=np.float64))
    squared = np.einsum("ni,ij,nj->n", delta, inverse, delta)
    return np.sqrt(np.maximum(squared, 0.0)).astype(np.float32)


def _json_float_list(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value).reshape(-1)]


def _json_matrix(value: np.ndarray) -> list[list[float]]:
    return [
        [float(item) for item in row]
        for row in np.asarray(value).tolist()
    ]


def _fit_distribution(samples: list[np.ndarray]) -> dict[str, Any]:
    matrix = np.stack(samples).astype(np.float32)
    mean = matrix.mean(axis=0)
    covariance = _shrink_covariance(matrix)
    distances = _mahalanobis(matrix, mean, covariance)
    threshold = max(
        3.0,
        float(np.quantile(distances, 0.95)) * 1.25,
    )
    return {
        "count": int(matrix.shape[0]),
        "mean": _json_float_list(mean),
        "covariance": _json_matrix(covariance),
        "distance_threshold": threshold,
    }


def fit_au_profile(
    labeled_sequences: Iterable[tuple[str, np.ndarray]],
    output_path: str | Path,
    *,
    au_ids: Iterable[int] = DEFAULT_AU_IDS,
    active_threshold: float = DEFAULT_ACTIVE_THRESHOLD,
) -> dict[str, Any]:
    """Fit target AU distributions from real target videos only."""
    au_ids = tuple(int(value) for value in au_ids)
    grouped_frames: dict[str, list[np.ndarray]] = {}
    grouped_summaries: dict[str, list[np.ndarray]] = {}
    sample_paths: dict[str, list[str]] = {}
    for expression_class, sequence in labeled_sequences:
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape[1] != len(au_ids):
            raise ValueError(
                f"AU sequence for {expression_class} has invalid shape "
                f"{sequence.shape}; expected [frames, {len(au_ids)}]."
            )
        grouped_frames.setdefault(expression_class, []).extend(sequence)
        grouped_summaries.setdefault(expression_class, []).append(
            au_summary(
                sequence,
                active_threshold=active_threshold,
            )
        )
        sample_paths.setdefault(expression_class, []).append("")

    if not grouped_frames:
        raise ValueError("No AU sequences were provided for profile fitting.")

    classes = sorted(grouped_frames)
    models: dict[str, Any] = {}
    for expression_class in classes:
        frame_model = _fit_distribution(grouped_frames[expression_class])
        summary_model = _fit_distribution(grouped_summaries[expression_class])
        models[expression_class] = {
            "frame": frame_model,
            "summary": summary_model,
            "sample_count": len(sample_paths[expression_class]),
        }

    profile = {
        "schema_version": AU_PROFILE_SCHEMA,
        "au_ids": list(au_ids),
        "active_threshold": float(active_threshold),
        "classes": models,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return profile


def _downsample(sequence: np.ndarray, max_points: int = 64) -> np.ndarray:
    if len(sequence) <= max_points:
        return sequence
    indices = np.rint(
        np.linspace(0, len(sequence) - 1, max_points)
    ).astype(np.int64)
    return sequence[indices]


def dtw_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_points: int = 64,
) -> float:
    """Compute normalized DTW distance for AU intensity trajectories."""
    left = _downsample(np.asarray(left, dtype=np.float32), max_points)
    right = _downsample(np.asarray(right, dtype=np.float32), max_points)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("DTW inputs must be [frames, same_au_count].")
    costs = np.full(
        (len(left) + 1, len(right) + 1),
        np.inf,
        dtype=np.float64,
    )
    costs[0, 0] = 0.0
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            local = float(np.mean(np.abs(left[i - 1] - right[j - 1])))
            costs[i, j] = local + min(
                costs[i - 1, j],
                costs[i, j - 1],
                costs[i - 1, j - 1],
            )
    return float(costs[-1, -1] / max(len(left), len(right), 1))


def dtw_similarity(left: np.ndarray, right: np.ndarray) -> float:
    distance = dtw_distance(left, right)
    return float(math.exp(-distance / 0.25))


def _load_profile(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        profile = json.load(handle)
    if profile.get("schema_version") != AU_PROFILE_SCHEMA:
        raise ValueError("Unsupported AU profile schema.")
    return profile


def _profile_model_score(
    sequence: np.ndarray,
    model: dict[str, Any],
) -> dict[str, Any]:
    frame_model = model["frame"]
    summary_model = model["summary"]
    frame_distances = _mahalanobis(
        sequence,
        np.asarray(frame_model["mean"], dtype=np.float32),
        np.asarray(frame_model["covariance"], dtype=np.float64),
    )
    summary = au_summary(sequence)
    summary_distance = float(
        _mahalanobis(
            summary[None, :],
            np.asarray(summary_model["mean"], dtype=np.float32),
            np.asarray(summary_model["covariance"], dtype=np.float64),
        )[0]
    )
    frame_threshold = float(frame_model["distance_threshold"])
    summary_threshold = float(summary_model["distance_threshold"])
    frame_anomaly_ratio = float(
        np.mean(frame_distances > frame_threshold)
    )
    summary_anomaly = float(summary_distance > summary_threshold)
    personal_score = float(
        math.exp(-summary_distance / max(summary_threshold, 1e-6))
        * (1.0 - 0.5 * frame_anomaly_ratio)
    )
    return {
        "personal_au_score_0_1": max(0.0, min(1.0, personal_score)),
        "frame_anomaly_ratio": frame_anomaly_ratio,
        "summary_distance": summary_distance,
        "summary_threshold": summary_threshold,
        "summary_anomaly": bool(summary_anomaly),
        "max_frame_distance": float(np.max(frame_distances)),
        "anomalous_frame_indices": [
            int(index)
            for index in np.flatnonzero(frame_distances > frame_threshold)
        ],
    }


def score_au_compliance(
    profile_path: str | Path,
    generated_au_path: str | Path,
    *,
    expected_class: str | None = None,
    driver_au_path: str | Path | None = None,
    leakage_classifier_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score target-specific AU compliance and driver expression fidelity."""
    profile = _load_profile(profile_path)
    au_ids = tuple(int(value) for value in profile["au_ids"])
    generated, generated_ids, generated_meta = load_au_table(
        generated_au_path,
        au_ids,
    )
    if generated_ids != au_ids:
        raise ValueError("Generated AU columns do not match the profile.")

    classes = profile["classes"]
    class_scores = {
        class_name: _profile_model_score(generated, model)
        for class_name, model in classes.items()
    }
    if expected_class:
        if expected_class not in class_scores:
            raise ValueError(
                f"Unknown expected expression class: {expected_class}"
            )
        selected_class = expected_class
    else:
        selected_class = max(
            class_scores,
            key=lambda name: class_scores[name]["personal_au_score_0_1"],
        )
    selected = class_scores[selected_class]

    driver_expression_score: float | None = None
    driver_meta: dict[str, Any] | None = None
    driver_similarity_proxy: float | None = None
    if driver_au_path:
        driver, driver_ids, driver_meta = load_au_table(
            driver_au_path,
            au_ids,
        )
        if driver_ids != au_ids:
            raise ValueError("Driver AU columns do not match the profile.")
        driver_expression_score = dtw_similarity(generated, driver)
        generated_summary = au_summary(generated)
        driver_summary = au_summary(driver)
        denominator = (
            float(np.linalg.norm(generated_summary))
            * float(np.linalg.norm(driver_summary))
        )
        if denominator > 1e-8:
            driver_similarity_proxy = float(
                max(
                    0.0,
                    min(
                        1.0,
                        (
                            float(
                                np.dot(
                                    generated_summary,
                                    driver_summary,
                                )
                            )
                            / denominator
                            + 1.0
                        )
                        / 2.0,
                    )
                )
            )

    classifier_risk: float | None = None
    if leakage_classifier_path:
        classifier = json.loads(
            Path(leakage_classifier_path).read_text(
                encoding="utf-8-sig"
            )
        )
        classifier_risk = score_leakage_classifier(
            classifier,
            au_summary(generated),
        )

    personal_score = float(selected["personal_au_score_0_1"])
    if classifier_risk is not None:
        leakage_risk = classifier_risk
        leakage_backend = "trained_au_leakage_classifier"
    elif driver_similarity_proxy is not None:
        leakage_risk = max(
            0.0,
            min(
                1.0,
                driver_similarity_proxy * (1.0 - personal_score),
            ),
        )
        leakage_backend = "driver_style_overlap_proxy"
    else:
        leakage_risk = float(
            max(
                selected["frame_anomaly_ratio"],
                1.0 - personal_score,
            )
        )
        leakage_backend = "target_au_anomaly_proxy"

    return {
        "status": "available",
        "backend": "au_personal_profile",
        "au_ids": list(au_ids),
        "selected_expression_class": selected_class,
        "expected_expression_class": expected_class,
        "class_scores": class_scores,
        "personal_au_score_0_1": personal_score,
        "driver_expression_score_0_1": driver_expression_score,
        "driver_similarity_proxy_0_1": driver_similarity_proxy,
        "driver_identity_leakage_risk_0_1": leakage_risk,
        "leakage_backend": leakage_backend,
        "generated_au": generated_meta,
        "driver_au": driver_meta,
    }


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def fit_leakage_classifier(
    positive_au_paths: Iterable[str | Path],
    negative_au_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    au_ids: Iterable[int] = DEFAULT_AU_IDS,
) -> dict[str, Any]:
    """Fit a small logistic classifier; label 0=target, 1=leakage/anomaly."""
    au_ids = tuple(int(value) for value in au_ids)
    paths = [
        (Path(path), 0)
        for path in positive_au_paths
    ] + [
        (Path(path), 1)
        for path in negative_au_paths
    ]
    if not paths or not any(label == 0 for _, label in paths):
        raise ValueError("At least one positive target AU file is required.")
    if not any(label == 1 for _, label in paths):
        raise ValueError(
            "Negative AU files are required to train leakage classifier. "
            "Use target-only profile scoring when they are unavailable."
        )

    features: list[np.ndarray] = []
    labels: list[int] = []
    for path, label in paths:
        sequence, _, _ = load_au_table(path, au_ids)
        features.append(au_summary(sequence))
        labels.append(label)
    matrix = np.stack(features).astype(np.float64)
    target = np.asarray(labels, dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), 1e-4)
    normalized = (matrix - mean) / scale

    weights = np.zeros(matrix.shape[1], dtype=np.float64)
    intercept = 0.0
    regularization = 0.01
    learning_rate = 0.1
    for _ in range(1200):
        logits = normalized @ weights + intercept
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
        error = probabilities - target
        weights -= learning_rate * (
            (normalized.T @ error) / len(matrix)
            + regularization * weights
        )
        intercept -= learning_rate * float(np.mean(error))

    model = {
        "schema_version": AU_CLASSIFIER_SCHEMA,
        "au_ids": list(au_ids),
        "feature_mean": _json_float_list(mean),
        "feature_scale": _json_float_list(scale),
        "weights": _json_float_list(weights),
        "intercept": float(intercept),
        "positive_count": int(np.sum(target == 0)),
        "negative_count": int(np.sum(target == 1)),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(model, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return model


def score_leakage_classifier(
    classifier: dict[str, Any],
    summary: np.ndarray,
) -> float:
    if classifier.get("schema_version") != AU_CLASSIFIER_SCHEMA:
        raise ValueError("Unsupported AU leakage classifier schema.")
    mean = np.asarray(classifier["feature_mean"], dtype=np.float64)
    scale = np.asarray(classifier["feature_scale"], dtype=np.float64)
    weights = np.asarray(classifier["weights"], dtype=np.float64)
    normalized = (np.asarray(summary, dtype=np.float64) - mean) / scale
    return _sigmoid(float(normalized @ weights) + classifier["intercept"])


def fuse_compliance_scores(
    *,
    identity_score_0_1: float | None,
    personal_au_score_0_1: float | None,
    driver_expression_score_0_1: float | None,
    leakage_risk_0_1: float | None,
    identity_threshold: float = 0.75,
    personal_au_threshold: float = 0.50,
    driver_expression_threshold: float = 0.50,
    leakage_threshold: float = 0.50,
) -> dict[str, Any]:
    """Fuse scores without hiding unavailable evidence."""
    components = [
        (0.40, identity_score_0_1),
        (0.40, personal_au_score_0_1),
        (0.20, driver_expression_score_0_1),
    ]
    valid = [
        (weight, float(score))
        for weight, score in components
        if score is not None and math.isfinite(float(score))
    ]
    weight_sum = sum(weight for weight, _ in valid)
    likeness = (
        sum(weight * score for weight, score in valid) / weight_sum
        if weight_sum
        else None
    )
    reasons: list[str] = []
    if (
        identity_score_0_1 is not None
        and identity_score_0_1 < identity_threshold
    ):
        reasons.append("identity_below_threshold")
    if (
        personal_au_score_0_1 is not None
        and personal_au_score_0_1 < personal_au_threshold
    ):
        reasons.append("personal_au_below_threshold")
    if (
        driver_expression_score_0_1 is not None
        and driver_expression_score_0_1 < driver_expression_threshold
    ):
        reasons.append("driver_expression_below_threshold")
    if (
        leakage_risk_0_1 is not None
        and leakage_risk_0_1 >= leakage_threshold
    ):
        reasons.append("driver_identity_leakage")
    missing_evidence = [
        name
        for name, value in (
            ("identity", identity_score_0_1),
            ("personal_au", personal_au_score_0_1),
            ("driver_expression", driver_expression_score_0_1),
            ("leakage", leakage_risk_0_1),
        )
        if value is None
    ]
    if reasons:
        decision = "block"
    elif missing_evidence:
        decision = "review"
    else:
        decision = "allow"
    return {
        "person_likeness_score_0_1": likeness,
        "score_weight_coverage": weight_sum,
        "leakage_risk_0_1": leakage_risk_0_1,
        "decision": decision,
        "decision_reasons": reasons,
        "missing_evidence": missing_evidence,
        "thresholds": {
            "identity": identity_threshold,
            "personal_au": personal_au_threshold,
            "driver_expression": driver_expression_threshold,
            "leakage": leakage_threshold,
        },
    }
