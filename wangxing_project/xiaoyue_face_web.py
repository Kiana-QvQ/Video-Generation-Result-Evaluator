"""Face-only web score profile for the isolated XiaoYue experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluator.modules.core.paths import project_path

from .xiaoyue_face_features import (
    build_feature_table,
    sequence_summary,
)

PROFILE_TYPE = "xiaoyue_face_mouth_web_profile_v1"


def _items_from_train(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = manifest.get("pairs") or {}
    train = pairs.get("train") or {}
    return [
        *list(train.get("real") or []),
        *list(train.get("fake") or []),
    ]


def _items_from_test(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if "pairs" in manifest:
        pairs = manifest.get("pairs") or {}
        test = pairs.get("test") or {}
        return [
            *list(test.get("real") or []),
            *list(test.get("fake") or []),
        ]
    return [
        *list(manifest.get("real") or []),
        *list(manifest.get("fake") or manifest.get("seedance") or []),
    ]


def _summary_weight_vector(size: int) -> np.ndarray:
    # sequence_summary stores face summary first, then mouth summary.
    face_size = 63 * 3
    weights = np.ones(size, dtype=np.float32)
    if size > face_size:
        weights[face_size:] = 2.0
    return weights


def fit_face_web_profile(
    *,
    manifest: dict[str, Any],
    cache_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    train_items = _items_from_train(manifest)
    if not train_items:
        raise ValueError("No training items found for XiaoYue face web profile.")
    train_manifest = {
        "pairs": {
            "train": {
                "real": [item for item in train_items if int(item.get("label_generated", 0)) == 0],
                "fake": [item for item in train_items if int(item.get("label_generated", 0)) == 1],
            },
            "test": {"real": [], "fake": []},
        }
    }
    table = build_feature_table(train_manifest, cache_path=cache_path)
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    for item in train_items:
        video = str(project_path(str(item["video"])).resolve())
        record = table["features"][video]
        vectors.append(sequence_summary(record["face"], record["mouth"]))
        labels.append(int(item.get("label_generated", 0)))
    matrix = np.stack(vectors).astype(np.float32)
    labels_array = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels_array)) < 2:
        raise ValueError("Face web profile needs both real and AI training items.")
    global_median = np.median(matrix, axis=0)
    global_scale = np.maximum(
        np.quantile(matrix, 0.75, axis=0)
        - np.quantile(matrix, 0.25, axis=0),
        0.05,
    ).astype(np.float32)
    weights = _summary_weight_vector(matrix.shape[1])
    locations: dict[str, list[float]] = {}
    for label, name in ((0, "real"), (1, "generated")):
        locations[name] = np.median(matrix[labels_array == label], axis=0).tolist()
    payload = {
        "schema_version": PROFILE_TYPE,
        "subject": "xiaoyue",
        "feature_policy": {
            "full_frame_rgb_used": False,
            "full_frame_hsv_used": False,
            "background_used": False,
            "absolute_brightness_used": False,
            "mouth_weight": 2.0,
            "description": (
                "Pose-normalized facial geometry/AU trajectories and "
                "brightness-centered local face crops; mouth branch is weighted."
            ),
        },
        "feature_dim": int(matrix.shape[1]),
        "weights": weights.tolist(),
        "global_median": global_median.tolist(),
        "global_scale": global_scale.tolist(),
        "locations": locations,
        "training_counts": {
            "real": int((labels_array == 0).sum()),
            "generated": int((labels_array == 1).sum()),
        },
        "training_videos": [
            str(project_path(str(item["video"])).resolve())
            for item in train_items
        ],
        "training_allowed": True,
        "test_training_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _score_vector(vector: np.ndarray, profile: dict[str, Any]) -> dict[str, float]:
    weights = np.asarray(profile["weights"], dtype=np.float32)
    scale = np.asarray(profile["global_scale"], dtype=np.float32)
    real = np.asarray(profile["locations"]["real"], dtype=np.float32)
    generated = np.asarray(profile["locations"]["generated"], dtype=np.float32)
    normalized = (vector - np.asarray(profile["global_median"], dtype=np.float32)) / scale
    def distance(location: np.ndarray) -> float:
        return float(
            np.sqrt(
                np.sum(weights * np.square((vector - location) / scale))
                / max(float(np.sum(weights)), 1e-6)
            )
        )
    real_distance = distance(real)
    generated_distance = distance(generated)
    real_similarity = float(np.exp(-real_distance))
    generated_similarity = float(np.exp(-generated_distance))
    total = real_similarity + generated_similarity
    return {
        "real_distance": real_distance,
        "generated_distance": generated_distance,
        "real_probability": real_similarity / total if total else 0.5,
        "generated_probability": generated_similarity / total if total else 0.5,
        "normalized_feature_l2": float(np.sqrt(np.mean(np.square(normalized)))),
    }


def evaluate_face_web(
    *,
    manifest: dict[str, Any],
    profile: dict[str, Any],
    cache_path: Path,
) -> dict[str, Any]:
    items = _items_from_test(manifest)
    if not items:
        raise ValueError("No test items found for XiaoYue face web evaluation.")
    test_real = [
        item for item in items if int(item.get("label_generated", 0)) == 0
    ]
    test_fake = [
        item for item in items if int(item.get("label_generated", 0)) == 1
    ]
    table = build_feature_table(
        {
            "pairs": {
                "train": {"real": [], "fake": []},
                "test": {"real": test_real, "fake": test_fake},
            }
        },
        cache_path=cache_path,
    )
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    predictions: list[int] = []
    for item in items:
        video = str(project_path(str(item["video"])).resolve())
        record = table["features"][video]
        vector = sequence_summary(record["face"], record["mouth"])
        score = _score_vector(vector, profile)
        prediction = int(score["generated_probability"] >= 0.5)
        label = int(item.get("label_generated", 0))
        labels.append(label)
        predictions.append(prediction)
        rows.append(
            {
                "sample_id": item.get("sample_id"),
                "video": video,
                "label_generated": label,
                "prediction": "generated" if prediction else "real",
                **score,
                "geometry_valid_ratio": record["geometry_valid_ratio"],
                "mouth_quality_ratio": record["mouth_quality_ratio"],
                "mouth_priority": True,
            }
        )
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    tp = int(((labels_array == 1) & (predictions_array == 1)).sum())
    tn = int(((labels_array == 0) & (predictions_array == 0)).sum())
    fp = int(((labels_array == 0) & (predictions_array == 1)).sum())
    fn = int(((labels_array == 1) & (predictions_array == 0)).sum())
    return {
        "schema_version": "xiaoyue_face_mouth_web_v1_evaluation",
        "subject": "xiaoyue",
        "headline": {
            "generated_recall": tp / (tp + fn) if tp + fn else None,
            "overall_accuracy": (tp + tn) / len(labels) if labels else None,
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
        "feature_policy": profile.get("feature_policy", {}),
        "rows": rows,
    }
