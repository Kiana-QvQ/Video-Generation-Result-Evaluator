"""Real-manifold face detector for the isolated XiaoYue experiment.

This is a small-data alternative to binary deep classification. It fits a
robust normal range from accepted real face sequences, calibrates an anomaly
threshold using leave-one-out real scores and the six AI training scores, and
stores the result in a PyTorch checkpoint for the PT path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluator.modules.core.paths import project_path

from .xiaoyue_face_features import build_feature_table, sequence_summary

MODEL_TYPE = "xiaoyue_face_real_manifold_v2"
MOUTH_WEIGHT = 2.0
TARGET_REAL_RECALL = 0.90
SCORE_TEMPERATURE = 0.05
RANK_PROBABILITY_TEMPERATURE = 0.05
RANK_SCORE_SCALE = 100.0


def _train_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = manifest.get("pairs") or {}
    train = pairs.get("train") or {}
    return [*list(train.get("real") or []), *list(train.get("fake") or [])]


def _test_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = manifest.get("pairs") or {}
    test = pairs.get("test") or {}
    return [*list(test.get("real") or []), *list(test.get("fake") or [])]


def _bank_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return list(manifest.get("real_manifold_bank") or [])


def _vectors(
    items: list[dict[str, Any]],
    table: dict[str, Any],
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for item in items:
        video = str(project_path(str(item["video"])).resolve())
        record = table["features"].get(video)
        if record is None:
            raise ValueError(f"Missing face feature record: {video}")
        vectors.append(sequence_summary(record["face"], record["mouth"]))
    if not vectors:
        raise ValueError("No face vectors are available.")
    return np.stack(vectors).astype(np.float32)


def _weights(size: int) -> np.ndarray:
    weights = np.ones(size, dtype=np.float32)
    face_size = 63 * 3
    if size > face_size:
        weights[face_size:] = MOUTH_WEIGHT
    return weights


def _ranking_weights(size: int) -> np.ndarray:
    """Weight face dynamics above appearance for the displayed realness rank."""
    weights = np.zeros(size, dtype=np.float32)
    face_geometry_au_end = 36 * 3
    mouth_geometry_au_start = 63 * 3
    mouth_geometry_au_end = 83 * 3
    face_crop_start = 36 * 3
    face_crop_end = 61 * 3
    mouth_crop_start = 83 * 3
    mouth_crop_end = 88 * 3
    weights[:face_geometry_au_end] = 1.0
    weights[face_crop_start:face_crop_end] = 0.15
    weights[mouth_geometry_au_start:mouth_geometry_au_end] = MOUTH_WEIGHT
    weights[mouth_crop_start:mouth_crop_end] = 0.30
    return weights


def _distance(
    vector: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
) -> float:
    normalized = (vector - center) / scale
    return float(
        np.sqrt(
            np.sum(weights * np.square(normalized))
            / max(float(np.sum(weights)), 1e-6)
        )
    )


def _choose_threshold(
    real_loo: np.ndarray,
    ai_train: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    values = np.unique(np.concatenate([real_loo, ai_train]))
    boundaries = [float(values[0] - 1e-5)]
    boundaries.extend(
        float((left + right) * 0.5)
        for left, right in zip(values[:-1], values[1:])
    )
    boundaries.append(float(values[-1] + 1e-5))
    candidates: list[dict[str, float]] = []
    for threshold in boundaries:
        real_recall = float(np.mean(real_loo < threshold))
        ai_recall = float(np.mean(ai_train >= threshold))
        candidates.append(
            {
                "threshold": threshold,
                "real_recall": real_recall,
                "generated_recall": ai_recall,
                "balanced_accuracy": 0.5 * (real_recall + ai_recall),
            }
        )
    eligible = [
        row
        for row in candidates
        if row["real_recall"] >= TARGET_REAL_RECALL
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["generated_recall"],
                row["balanced_accuracy"],
                -row["threshold"],
            ),
        )
    else:
        selected = max(
            candidates,
            key=lambda row: (
                row["balanced_accuracy"],
                row["real_recall"],
                row["generated_recall"],
            ),
        )
    return float(selected["threshold"]), {
        "target_real_recall": TARGET_REAL_RECALL,
        "selected": selected,
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "all_candidates": candidates,
    }


def _probability(anomaly: float, threshold: float) -> float:
    value = (float(anomaly) - float(threshold)) / SCORE_TEMPERATURE
    value = float(np.clip(value, -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-value)))


def _realness_rank_score(anomaly: float) -> float:
    """Return a monotonic face-naturalness score for display/ranking only."""
    return float(RANK_SCORE_SCALE / (1.0 + max(float(anomaly), 0.0)))


def _component_weights(
    weights: np.ndarray,
    *,
    start: int,
    stop: int,
) -> np.ndarray:
    component = np.zeros_like(weights, dtype=np.float32)
    component[start:stop] = weights[start:stop]
    return component


def _ranking_real_probability(
    anomaly: float,
    threshold: float,
) -> float:
    value = (float(threshold) - float(anomaly)) / RANK_PROBABILITY_TEMPERATURE
    value = float(np.clip(value, -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-value)))


def fit_face_manifold(
    *,
    manifest: dict[str, Any],
    cache_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    bank = _bank_items(manifest)
    ai_train = [
        item
        for item in _train_items(manifest)
        if int(item.get("label_generated", 0)) == 1
    ]
    if len(bank) < 20 or len(ai_train) != 6:
        raise ValueError(
            f"Expected at least 20 real bank items and 6 AI items, got "
            f"{len(bank)} and {len(ai_train)}."
        )
    fit_manifest = {
        "pairs": {
            "train": {"real": bank, "fake": ai_train},
            "test": {"real": [], "fake": []},
        }
    }
    table = build_feature_table(fit_manifest, cache_path=cache_path)
    real_matrix = _vectors(bank, table)
    ai_matrix = _vectors(ai_train, table)
    center = np.median(real_matrix, axis=0).astype(np.float32)
    # Use the same train-only standard-deviation scale as the validation
    # protocol. Mixing IQR here with a standard-deviation calibration changes
    # the anomaly units and can turn a valid real test clip into a false AI.
    scale = np.maximum(
        np.std(real_matrix, axis=0),
        0.05,
    ).astype(np.float32)
    weights = _weights(real_matrix.shape[1])
    ranking_weights = _ranking_weights(real_matrix.shape[1])
    loo_scores: list[float] = []
    ranking_loo_scores: list[float] = []
    for index, vector in enumerate(real_matrix):
        loo_center = np.median(
            np.delete(real_matrix, index, axis=0),
            axis=0,
        )
        loo_scores.append(_distance(vector, loo_center, scale, weights))
        ranking_loo_scores.append(
            _distance(vector, loo_center, scale, ranking_weights)
        )
    ai_scores = np.asarray(
        [_distance(vector, center, scale, weights) for vector in ai_matrix],
        dtype=np.float32,
    )
    ranking_ai_scores = np.asarray(
        [
            _distance(vector, center, scale, ranking_weights)
            for vector in ai_matrix
        ],
        dtype=np.float32,
    )
    threshold, threshold_report = _choose_threshold(
        np.asarray(loo_scores, dtype=np.float32),
        ai_scores,
    )
    ranking_threshold, ranking_threshold_report = _choose_threshold(
        np.asarray(ranking_loo_scores, dtype=np.float32),
        ranking_ai_scores,
    )
    payload = {
        "schema_version": "xiaoyue_face_real_manifold_v2_profile",
        "subject": "xiaoyue",
        "model_type": MODEL_TYPE,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "ranking_weights": ranking_weights.tolist(),
        "threshold": threshold,
        "ranking_threshold": ranking_threshold,
        "score_temperature": SCORE_TEMPERATURE,
        "ranking_probability_temperature": RANK_PROBABILITY_TEMPERATURE,
        "mouth_weight": MOUTH_WEIGHT,
        "training_counts": {
            "real_manifold_bank": len(bank),
            "ai_train": len(ai_train),
        },
        "calibration": {
            "real_leave_one_out": {
                "p50": float(np.quantile(loo_scores, 0.50)),
                "p90": float(np.quantile(loo_scores, 0.90)),
                "p95": float(np.quantile(loo_scores, 0.95)),
                "max": float(np.max(loo_scores)),
            },
            "ai_train_scores": ai_scores.tolist(),
            "ranking_ai_train_scores": ranking_ai_scores.tolist(),
            "ranking_real_leave_one_out": {
                "p50": float(np.quantile(ranking_loo_scores, 0.50)),
                "p90": float(np.quantile(ranking_loo_scores, 0.90)),
                "p95": float(np.quantile(ranking_loo_scores, 0.95)),
                "max": float(np.max(ranking_loo_scores)),
            },
            "ranking_threshold_selection": ranking_threshold_report,
            "threshold_selection": threshold_report,
        },
        "feature_policy": {
            "full_frame_rgb_used": False,
            "full_frame_hsv_used": False,
            "background_used": False,
            "absolute_brightness_used": False,
            "mouth_priority": True,
            "ranking_score_semantics": (
                "Higher means more face-natural in the calibrated ranking "
                "axis; it is not a calibrated real-capture probability."
            ),
            "description": (
                "Robust real face manifold over pose-normalized AU/Face "
                "Mesh and brightness-centered local mouth/face features."
            ),
        },
        "training_videos": [
            str(project_path(str(item["video"])).resolve())
            for item in bank
        ],
        "test_training_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def score_face_manifold(
    *,
    manifest: dict[str, Any],
    profile: dict[str, Any],
    cache_path: Path,
) -> dict[str, Any]:
    items = _test_items(manifest)
    table = build_feature_table(
        {
            "pairs": {
                "train": {"real": [], "fake": []},
                "test": {
                    "real": [
                        item
                        for item in items
                        if int(item.get("label_generated", 0)) == 0
                    ],
                    "fake": [
                        item
                        for item in items
                        if int(item.get("label_generated", 0)) == 1
                    ],
                },
            }
        },
        cache_path=cache_path,
    )
    center = np.asarray(profile["center"], dtype=np.float32)
    scale = np.asarray(profile["scale"], dtype=np.float32)
    weights = np.asarray(profile["weights"], dtype=np.float32)
    ranking_weights = np.asarray(
        profile.get("ranking_weights", profile["weights"]),
        dtype=np.float32,
    )
    face_weights = _component_weights(
        ranking_weights,
        start=0,
        stop=min(189, len(ranking_weights)),
    )
    mouth_weights = _component_weights(
        ranking_weights,
        start=min(189, len(ranking_weights)),
        stop=min(270, len(ranking_weights)),
    )
    threshold = float(profile["threshold"])
    ranking_threshold = float(
        profile.get("ranking_threshold", threshold)
    )
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    predictions: list[int] = []
    for item in items:
        video = str(project_path(str(item["video"])).resolve())
        record = table["features"][video]
        vector = sequence_summary(record["face"], record["mouth"])
        anomaly = _distance(vector, center, scale, weights)
        ranking_anomaly = _distance(
            vector,
            center,
            scale,
            ranking_weights,
        )
        face_anomaly = _distance(
            vector,
            center,
            scale,
            face_weights,
        )
        mouth_anomaly = _distance(
            vector,
            center,
            scale,
            mouth_weights,
        )
        probability = _probability(anomaly, threshold)
        ranking_real_probability = _ranking_real_probability(
            ranking_anomaly,
            ranking_threshold,
        )
        realness_score = _realness_rank_score(ranking_anomaly)
        face_naturalness_score = _realness_rank_score(face_anomaly)
        mouth_geometry_score = _realness_rank_score(mouth_anomaly)
        mouth_summary = np.asarray(
            record.get("mouth_summary", np.zeros(8)),
            dtype=np.float32,
        )
        mouth_residual_naturalness = (
            float(np.clip(mouth_summary[1], 0.0, 1.0))
            if mouth_summary.size > 1
            else 0.5
        )
        mouth_flicker_naturalness = (
            float(np.clip(mouth_summary[6], 0.0, 1.0))
            if mouth_summary.size > 6
            else 0.5
        )
        mouth_continuity = (
            float(np.clip(mouth_summary[5], 0.0, 1.0))
            if mouth_summary.size > 5
            else 0.5
        )
        mouth_naturalness_score = float(
            0.55 * mouth_geometry_score
            + 0.30 * mouth_residual_naturalness * 100.0
            + 0.15 * mouth_flicker_naturalness * 100.0
        )
        display_score = ranking_real_probability * 100.0
        label = int(item.get("label_generated", 0))
        prediction = int(anomaly >= threshold)
        labels.append(label)
        predictions.append(prediction)
        rows.append(
            {
                "sample_id": item.get("sample_id"),
                "video": video,
                "label_generated": label,
                "prediction": "generated" if prediction else "real",
                "generated_probability": probability,
                "real_probability": 1.0 - probability,
                "classification_generated_probability": probability,
                "classification_real_probability": 1.0 - probability,
                "ranking_real_probability": ranking_real_probability,
                "ranking_generated_probability": 1.0 - ranking_real_probability,
                "display_real_probability": ranking_real_probability,
                "real_manifold_anomaly": anomaly,
                "ranking_anomaly": ranking_anomaly,
                "face_anomaly": face_anomaly,
                "mouth_anomaly": mouth_anomaly,
                "face_naturalness_score_0_100": face_naturalness_score,
                "mouth_naturalness_score_0_100": mouth_naturalness_score,
                "mouth_residual_naturalness_0_1": mouth_residual_naturalness,
                "mouth_flicker_naturalness_0_1": mouth_flicker_naturalness,
                "mouth_continuity_0_1": mouth_continuity,
                "realness_score_0_100": realness_score,
                "display_score_0_100": display_score,
                "threshold": threshold,
                "ranking_threshold": ranking_threshold,
                "geometry_valid_ratio": record["geometry_valid_ratio"],
                "mouth_quality_ratio": record["mouth_quality_ratio"],
                "mouth_priority": True,
            }
        )
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(predictions, dtype=np.int64)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return {
        "schema_version": "xiaoyue_face_real_manifold_v2_evaluation",
        "subject": "xiaoyue",
        "model_type": MODEL_TYPE,
        "headline": {
            "generated_recall": tp / (tp + fn) if tp + fn else None,
            "overall_accuracy": (tp + tn) / len(y) if len(y) else None,
            "generated_precision": tp / (tp + fp) if tp + fp else None,
            "real_recall": tn / (tn + fp) if tn + fp else None,
            "coverage": 1.0,
        },
        "confusion": {
            "tp_generated": tp,
            "tn_real": tn,
            "fp_real_as_generated": fp,
            "fn_generated_as_real": fn,
        },
        "rows": rows,
        "feature_policy": profile.get("feature_policy", {}),
        "training_allowed": False,
    }


def save_pt_checkpoint(profile: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": MODEL_TYPE,
            "profile": profile,
            "feature_policy": profile.get("feature_policy", {}),
        },
        output_path,
    )
