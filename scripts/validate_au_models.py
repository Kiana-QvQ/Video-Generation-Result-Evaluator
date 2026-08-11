from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.wangxing.au_compliance import (  # noqa: E402
    AU_CLASSIFIER_SCHEMA,
    DEFAULT_AU_IDS,
    DEFAULT_PRESENCE_AU_IDS,
    _downsample,
    _profile_model_score,
    _summary_pairs,
    au_summary,
    fit_au_profile,
    load_au_profile_tables,
)
from scripts.build_au_profile import (  # noqa: E402
    FULL_DATASET_CLASS_PREFIXES,
    _full_dataset_class,
)


INTENSITY_WEIGHT = 0.55
PRESENCE_WEIGHT = 0.45
DEFAULT_ACTIVE_THRESHOLD = 0.20


@dataclass
class Sample:
    path: Path
    label: str
    group: str
    intensity: np.ndarray
    presence: np.ndarray | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate AU emotion and leakage models using CSV features only. "
            "Reference images, reference videos, and GT videos are not read."
        )
    )
    parser.add_argument("--au-root", default="data/au/MD_CL")
    parser.add_argument("--negative-root", default="data/au/negative")
    parser.add_argument(
        "--output",
        default="outputs/au_validation/au_validation_report.json",
    )
    parser.add_argument(
        "--leakage-output",
        default="data/au/au_leakage_classifier.json",
    )
    parser.add_argument("--max-points", type=int, default=128)
    parser.add_argument("--max-errors", type=int, default=200)
    return parser.parse_args()


def _canonical_label(path: Path) -> str | None:
    return _full_dataset_class(path)


def _clean_sequence(sequence: np.ndarray, max_points: int) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError(f"Expected a 2D sequence, got {sequence.shape}.")
    valid_rows = np.all(np.isfinite(sequence), axis=1)
    sequence = sequence[valid_rows]
    if len(sequence) < 2:
        raise ValueError("Sequence has fewer than two finite frames.")
    return _downsample(sequence, max_points=max_points).astype(np.float32)


def _load_target_samples(
    root: Path,
    *,
    max_points: int,
) -> tuple[list[Sample], dict[str, Any]]:
    samples: list[Sample] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.csv")):
        label = _canonical_label(path)
        if label is None:
            continue
        try:
            intensity, _, presence, presence_supported = load_au_profile_tables(
                path,
                intensity_au_ids=DEFAULT_AU_IDS,
                presence_au_ids=DEFAULT_PRESENCE_AU_IDS,
            )
            intensity = _clean_sequence(intensity, max_points)
            if presence is not None and presence_supported:
                presence = _clean_sequence(presence, max_points)
            else:
                presence = None
            samples.append(
                Sample(
                    path=path,
                    label=label,
                    group=path.parent.name,
                    intensity=intensity,
                    presence=presence,
                )
            )
        except (OSError, ValueError) as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
    return samples, {
        "sample_count": len(samples),
        "class_counts": dict(sorted(Counter(item.label for item in samples).items())),
        "group_counts": dict(sorted(Counter(item.group for item in samples).items())),
        "skipped_count": len(skipped),
        "skipped_preview": skipped[:20],
    }


def _load_negative_samples(
    root: Path,
    *,
    max_points: int,
) -> tuple[list[Sample], dict[str, Any]]:
    samples: list[Sample] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.csv")):
        try:
            intensity, _, _, _ = load_au_profile_tables(
                path,
                intensity_au_ids=DEFAULT_AU_IDS,
                presence_au_ids=DEFAULT_PRESENCE_AU_IDS,
            )
            intensity = _clean_sequence(intensity, max_points)
            samples.append(
                Sample(
                    path=path,
                    label="negative_identity",
                    group=path.parent.name,
                    intensity=intensity,
                    presence=None,
                )
            )
        except (OSError, ValueError) as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
    return samples, {
        "sample_count": len(samples),
        "group_counts": dict(sorted(Counter(item.group for item in samples).items())),
        "skipped_count": len(skipped),
        "skipped_preview": skipped[:20],
    }


def _presence_fit(
    sequence: np.ndarray | None,
    profile: dict[str, Any],
    label: str,
) -> float | None:
    if sequence is None:
        return None
    target = (
        profile.get("presence_classes", {})
        .get(label, {})
        .get("mean_activation")
    )
    if not isinstance(target, list):
        return None
    finite = np.isfinite(sequence)
    if not np.any(finite):
        return None
    ratios = []
    for index, expected in enumerate(target):
        values = sequence[:, index]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        ratios.append(abs(float(np.mean(values >= DEFAULT_ACTIVE_THRESHOLD)) - float(expected)))
    if not ratios:
        return None
    return max(0.0, min(1.0, 1.0 - float(np.mean(ratios))))


def _score_sample(
    sample: Sample,
    profile: dict[str, Any],
) -> tuple[str, dict[str, float], dict[str, Any]]:
    au_ids = tuple(int(value) for value in profile["au_ids"])
    class_scores: dict[str, float] = {}
    detailed: dict[str, Any] = {}
    for label, model in profile.get("classes", {}).items():
        intensity = _profile_model_score(
            sample.intensity,
            model,
            full_au_ids=au_ids,
            supported_au_ids=au_ids,
        )
        intensity_score = float(intensity["personal_au_score_0_1"])
        presence_score = _presence_fit(sample.presence, profile, label)
        fused = (
            INTENSITY_WEIGHT * intensity_score
            + PRESENCE_WEIGHT * presence_score
            if presence_score is not None
            else intensity_score
        )
        class_scores[label] = max(0.0, min(1.0, float(fused)))
        detailed[label] = {
            "intensity": intensity_score,
            "presence": presence_score,
            "fused": class_scores[label],
            "summary_distance": float(intensity["summary_distance"]),
        }
    predicted = max(class_scores, key=class_scores.get)
    return predicted, class_scores, detailed


def _metric_summary(
    true_labels: list[str],
    predicted_labels: list[str],
    labels: Iterable[str],
) -> dict[str, Any]:
    labels = list(labels)
    confusion = {
        true_label: {predicted: 0 for predicted in labels}
        for true_label in labels
    }
    for true_label, predicted in zip(true_labels, predicted_labels):
        confusion.setdefault(true_label, {value: 0 for value in labels})
        confusion[true_label].setdefault(predicted, 0)
        confusion[true_label][predicted] += 1

    per_class: dict[str, Any] = {}
    total = len(true_labels)
    correct = sum(
        int(true_label == predicted)
        for true_label, predicted in zip(true_labels, predicted_labels)
    )
    for label in labels:
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(
            confusion.get(other, {}).get(label, 0)
            for other in labels
            if other != label
        )
        fn = sum(
            confusion.get(label, {}).get(other, 0)
            for other in labels
            if other != label
        )
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    macro = {
        key: float(np.mean([item[key] for item in per_class.values()]))
        if per_class
        else 0.0
        for key in ("precision", "recall", "f1")
    }
    weighted = {}
    for key in ("precision", "recall", "f1"):
        weighted[key] = (
            sum(item[key] * item["support"] for item in per_class.values())
            / max(total, 1)
        )
    return {
        "sample_count": total,
        "accuracy": correct / max(total, 1),
        "balanced_accuracy": macro["recall"],
        "macro": macro,
        "weighted": weighted,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _rank_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _multiclass_auc(
    true_labels: list[str],
    score_rows: list[dict[str, float]],
    labels: list[str],
) -> dict[str, Any]:
    per_class: dict[str, float | None] = {}
    for label in labels:
        binary = [int(value == label) for value in true_labels]
        scores = [float(row.get(label, 0.0)) for row in score_rows]
        per_class[label] = _rank_auc(binary, scores)
    values = [value for value in per_class.values() if value is not None]
    return {
        "ovr_auc_per_class": per_class,
        "macro_ovr_auc": float(np.mean(values)) if values else None,
    }


def _fit_binary_logistic(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-4)
    normalized = (features - mean) / scale
    weights = np.zeros(features.shape[1], dtype=np.float64)
    intercept = 0.0
    for _ in range(1200):
        logits = normalized @ weights + intercept
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
        error = probabilities - labels
        weights -= 0.1 * (
            (normalized.T @ error) / len(features)
            + 0.01 * weights
        )
        intercept -= 0.1 * float(np.mean(error))
    return weights, intercept, mean, scale


def _binary_metrics(
    true_labels: list[int],
    probabilities: list[float],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    predicted = [int(value >= threshold) for value in probabilities]
    tp = sum(int(actual == 1 and guess == 1) for actual, guess in zip(true_labels, predicted))
    tn = sum(int(actual == 0 and guess == 0) for actual, guess in zip(true_labels, predicted))
    fp = sum(int(actual == 0 and guess == 1) for actual, guess in zip(true_labels, predicted))
    fn = sum(int(actual == 1 and guess == 0) for actual, guess in zip(true_labels, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "sample_count": len(true_labels),
        "threshold": threshold,
        "accuracy": (tp + tn) / max(len(true_labels), 1),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": _rank_auc(true_labels, probabilities),
        "confusion_matrix": {
            "target_true_target_pred": tp,
            "target_true_negative_pred": fn,
            "negative_true_target_pred": fp,
            "negative_true_negative_pred": tn,
        },
    }


def _threshold_sweep(
    true_labels: list[int],
    probabilities: list[float],
) -> dict[str, Any]:
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = _binary_metrics(
            true_labels,
            probabilities,
            threshold=float(threshold),
        )
        rows.append(metrics)
    best_f1 = max(rows, key=lambda item: (item["f1"], item["balanced_accuracy"]))
    best_balanced = max(
        rows,
        key=lambda item: (
            item["balanced_accuracy"],
            item["f1"],
            item["recall"],
        ),
    )
    return {
        "best_f1": best_f1,
        "best_balanced_accuracy": best_balanced,
        "recommended_threshold": best_f1["threshold"],
    }


def _validate_emotion(
    samples: list[Sample],
    *,
    max_errors: int,
) -> dict[str, Any]:
    labels = sorted({sample.label for sample in samples})
    groups = sorted({sample.group for sample in samples})
    predictions: list[str] = []
    truths: list[str] = []
    score_rows: list[dict[str, float]] = []
    errors: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="au_validation_") as directory:
        temp_root = Path(directory)
        for group in groups:
            train = [sample for sample in samples if sample.group != group]
            test = [sample for sample in samples if sample.group == group]
            if not test or {sample.label for sample in train} != set(labels):
                continue
            profile_path = temp_root / f"{group}.json"
            profile = fit_au_profile(
                [(sample.label, sample.intensity) for sample in train],
                profile_path,
                au_ids=DEFAULT_AU_IDS,
                presence_labeled_sequences=[
                    (sample.label, sample.presence)
                    for sample in train
                    if sample.presence is not None
                ],
                presence_au_ids=DEFAULT_PRESENCE_AU_IDS,
                sample_metadata=[
                    {
                        "source_id": str(sample.path),
                        "au_path": str(sample.path),
                    }
                    for sample in train
                ],
            )
            fold_records.append(
                {
                    "group": group,
                    "train_count": len(train),
                    "test_count": len(test),
                }
            )
            for sample in test:
                predicted, scores, detailed = _score_sample(sample, profile)
                truths.append(sample.label)
                predictions.append(predicted)
                score_rows.append(scores)
                if predicted != sample.label:
                    ordered = sorted(scores.values(), reverse=True)
                    errors.append(
                        {
                            "path": str(sample.path),
                            "group": sample.group,
                            "true": sample.label,
                            "predicted": predicted,
                            "margin": (
                                ordered[0] - ordered[1]
                                if len(ordered) > 1
                                else ordered[0]
                            ),
                            "scores": detailed,
                        }
                    )

    metrics = _metric_summary(truths, predictions, labels)
    metrics["auc"] = _multiclass_auc(truths, score_rows, labels)
    metrics["folds"] = fold_records
    metrics["misclassifications"] = sorted(
        errors,
        key=lambda item: (-float(item["margin"]), item["path"]),
    )[:max_errors]
    confusion_pairs = Counter(
        (item["true"], item["predicted"]) for item in errors
    )
    metrics["top_confusions"] = [
        {
            "true": true_label,
            "predicted": predicted,
            "count": count,
        }
        for (true_label, predicted), count in confusion_pairs.most_common()
    ]
    metrics["group_error_rate"] = {
        group: (
            sum(1 for item in errors if item["group"] == group)
            / max(sum(1 for sample in samples if sample.group == group), 1)
        )
        for group in groups
    }
    return metrics


def _validate_leakage(
    positive: list[Sample],
    negative: list[Sample],
    *,
    max_errors: int,
) -> dict[str, Any]:
    samples = positive + negative
    features = np.stack(
        [au_summary(sample.intensity, au_ids=DEFAULT_AU_IDS) for sample in samples]
    ).astype(np.float64)
    labels = np.asarray([0] * len(positive) + [1] * len(negative), dtype=np.float64)
    groups = [
        f"target::{sample.group}" for sample in positive
    ] + [
        f"negative::{sample.group}" for sample in negative
    ]
    predictions: list[int] = []
    truths: list[int] = []
    probabilities: list[float] = []
    errors: list[dict[str, Any]] = []
    fold_count = 0
    for holdout in sorted(set(groups)):
        test_mask = np.asarray([group == holdout for group in groups])
        train_mask = ~test_mask
        if len(np.unique(labels[train_mask])) < 2 or not np.any(test_mask):
            continue
        weights, intercept, mean, scale = _fit_binary_logistic(
            features[train_mask],
            labels[train_mask],
        )
        normalized = (features[test_mask] - mean) / scale
        scores = 1.0 / (
            1.0 + np.exp(
                -np.clip(normalized @ weights + intercept, -60, 60)
            )
        )
        test_indices = np.flatnonzero(test_mask)
        for index, probability in zip(test_indices, scores):
            actual = int(labels[index])
            predicted = int(float(probability) >= 0.5)
            truths.append(actual)
            predictions.append(predicted)
            probabilities.append(float(probability))
            if actual != predicted and len(errors) < max_errors:
                errors.append(
                    {
                        "path": str(samples[index].path),
                        "group": groups[index],
                        "true": actual,
                        "predicted": predicted,
                        "negative_probability": float(probability),
                    }
                )
        fold_count += 1
    metrics = _binary_metrics(truths, probabilities)
    metrics["threshold_calibration"] = _threshold_sweep(
        truths,
        probabilities,
    )
    metrics["fold_count"] = fold_count
    metrics["misclassifications"] = errors
    return metrics


def _write_leakage_classifier(
    positive: list[Sample],
    negative: list[Sample],
    output: Path,
    *,
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    samples = positive + negative
    features = np.stack(
        [au_summary(sample.intensity, au_ids=DEFAULT_AU_IDS) for sample in samples]
    ).astype(np.float64)
    labels = np.asarray(
        [0] * len(positive) + [1] * len(negative),
        dtype=np.float64,
    )
    weights, intercept, mean, scale = _fit_binary_logistic(features, labels)
    model = {
        "schema_version": AU_CLASSIFIER_SCHEMA,
        "au_ids": list(DEFAULT_AU_IDS),
        "supported_au_ids": list(DEFAULT_AU_IDS),
        "missing_au_ids": [],
        "feature_type": "intensity",
        "summary_layout": {
            "blocks": ["median", "mad", "active_ratio"],
            "coactivation_pairs": [
                list(pair) for pair in _summary_pairs(DEFAULT_AU_IDS)
            ],
        },
        "feature_mean": [float(value) for value in mean],
        "feature_scale": [float(value) for value in scale],
        "weights": [float(value) for value in weights],
        "intercept": float(intercept),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "decision_threshold": float(decision_threshold),
        "training_source": "csv_only_grouped_validation_loader",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(model, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(output),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "decision_threshold": float(decision_threshold),
    }


def _diagnose(
    emotion: dict[str, Any],
    leakage: dict[str, Any],
    target_meta: dict[str, Any],
    negative_meta: dict[str, Any],
) -> dict[str, Any]:
    macro_auc = emotion.get("auc", {}).get("macro_ovr_auc")
    group_errors = emotion.get("group_error_rate", {})
    high_error_groups = [
        group
        for group, rate in sorted(
            group_errors.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if float(rate) >= 0.75
    ]
    threshold = leakage.get("threshold_calibration", {})
    return {
        "emotion_separation": (
            "poor"
            if macro_auc is not None and float(macro_auc) < 0.65
            else "requires_review"
        ),
        "emotion_separation_note": (
            "Macro one-vs-rest AUC is close to random; adding more samples "
            "without temporal features or cross-session normalization is "
            "unlikely to fix the dominant confusions."
        ),
        "dominant_emotion_confusions": emotion.get("top_confusions", [])[:8],
        "high_error_source_groups": high_error_groups,
        "likely_emotion_failure_modes": [
            "Frame-pooled AU distributions overlap across emotions.",
            "Presence activation is high and similar across classes, so it "
            "adds little class discrimination despite the 45% fusion weight.",
            "The merged anger class contains both FenNu and ShengQi source "
            "groups; public anger can remain one label, but internal style "
            "subclusters may be needed.",
            "There is no explicit neutral/non-emotion class in the rebuilt "
            "emotion profile, so weak or neutral clips can be forced into "
            "fear, anger, or disgust.",
            "High group error rates indicate capture-session or recording "
            "domain shift.",
        ],
        "leakage_threshold_note": {
            "default_0_5": {
                "recall": leakage.get("recall"),
                "specificity": leakage.get("specificity"),
                "f1": leakage.get("f1"),
            },
            "recommended_f1_threshold": threshold.get("best_f1"),
            "high_recall_threshold": threshold.get("best_balanced_accuracy"),
            "deployment_choice": (
                "Use the F1 threshold for the default review decision; "
                "use the high-recall threshold only for an explicit "
                "conservative screening mode."
            ),
        },
        "data_requirements": {
            "accepted_generated_samples_detected": 0,
            "negative_actor_groups": negative_meta.get("group_counts", {}),
            "recommended_additions": [
                "Known-good generated Wang Xing videos, labeled as "
                "target-compatible and kept separate from real samples.",
                "Known-bad generated videos covering identity drift, wrong "
                "expression, temporal collapse, mouth-only artifacts, and "
                "face replacement failures.",
                "Real neutral, speech, transition, and low-intensity clips "
                "to prevent forced emotion classification.",
                "Real Wang Xing clips from new recording sessions, lighting, "
                "camera distance, head pose, occlusion, and compression.",
                "More cross-identity negatives than two RAVDESS actors, "
                "including synthetic identities and difficult near-target "
                "facial motion.",
            ],
        },
        "data_quality": {
            "target_sample_count": target_meta.get("sample_count"),
            "target_skipped_count": target_meta.get("skipped_count"),
            "negative_sample_count": negative_meta.get("sample_count"),
            "negative_skipped_count": negative_meta.get("skipped_count"),
        },
    }


def main() -> int:
    args = _parse_args()
    if args.max_points < 8:
        raise ValueError("--max-points must be at least 8.")
    if args.max_errors < 0:
        raise ValueError("--max-errors cannot be negative.")

    au_root = (PROJECT_ROOT / args.au_root).resolve()
    negative_root = (PROJECT_ROOT / args.negative_root).resolve()
    target_samples, target_meta = _load_target_samples(
        au_root,
        max_points=args.max_points,
    )
    negative_samples, negative_meta = _load_negative_samples(
        negative_root,
        max_points=args.max_points,
    )
    if not target_samples:
        raise SystemExit("No target AU CSV samples were loaded.")
    leakage_validation = (
        _validate_leakage(
            target_samples,
            negative_samples,
            max_errors=args.max_errors,
        )
        if negative_samples
        else {
            "status": "unavailable",
            "reason": "No negative AU CSV samples were loaded.",
        }
    )
    classifier_output = (PROJECT_ROOT / args.leakage_output).resolve()
    classifier_training = (
        _write_leakage_classifier(
            target_samples,
            negative_samples,
            classifier_output,
            decision_threshold=float(
                leakage_validation.get("threshold_calibration", {})
                .get("recommended_threshold", 0.5)
            ),
        )
        if negative_samples
        else {
            "status": "unavailable",
            "reason": "No negative AU CSV samples were loaded.",
        }
    )
    emotion_validation = _validate_emotion(
        target_samples,
        max_errors=args.max_errors,
    )

    report = {
        "schema_version": "au_validation_v1",
        "evaluation_contract": {
            "feature_source": "AU CSV only",
            "reference_image_used": False,
            "reference_video_used": False,
            "ground_truth_video_used": False,
            "split_unit": "source subdirectory, never individual frames",
            "personal_score_fusion": (
                "0.55 * intensity + 0.45 * presence when Presence is available"
            ),
            "facial_dynamics_validation": (
                "Not precision/recall validated: no accepted_generated "
                "labeled samples are currently available."
            ),
        },
        "mapping": FULL_DATASET_CLASS_PREFIXES,
        "target_data": target_meta,
        "negative_data": negative_meta,
        "classifier_training": classifier_training,
        "emotion_classification": emotion_validation,
        "identity_deviation_classifier": leakage_validation,
        "data_gaps": {
            "accepted_generated_samples_detected": 0,
            "note": (
                "Current target root is labeled CL_* data. Known-good generated "
                "samples should be added as a separate source group before "
                "using them as positive target-compatible evidence."
            ),
        },
    }
    report["diagnostics"] = _diagnose(
        emotion_validation,
        leakage_validation,
        target_meta,
        negative_meta,
    )
    output = (PROJECT_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "target_samples": len(target_samples),
                "negative_samples": len(negative_samples),
                "emotion_accuracy": report["emotion_classification"]["accuracy"],
                "emotion_macro_f1": report["emotion_classification"]["macro"]["f1"],
                "leakage_accuracy_at_0_5": report["identity_deviation_classifier"].get(
                    "accuracy"
                ),
                "leakage_recall_at_0_5": report["identity_deviation_classifier"].get(
                    "recall"
                ),
                "leakage_recommended_threshold": report[
                    "identity_deviation_classifier"
                ]
                .get("threshold_calibration", {})
                .get("recommended_threshold"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
