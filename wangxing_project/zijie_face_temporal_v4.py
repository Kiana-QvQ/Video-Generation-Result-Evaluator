"""Offline ZiJie V4 classifier with local face forensic evidence.

V4 keeps V3 untouched and adds brightness-normalized spatial detail and
translation-compensated residuals from local facial regions only. Model and
threshold selection use grouped training folds; the fixed holdout is evaluation
only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluator.modules.core.paths import project_path
from evaluator.vedio_pred.real_video_detector import _file_signature
from wangxing_project.face_crop_temporal import (
    FORENSIC_FEATURE_DIM,
    extract_face_crop_forensic_features,
)
from wangxing_project.zijie_face_temporal_v2 import (
    FLOW_FEATURE_DIM,
    WINDOW_DURATION_SECONDS,
    WINDOW_FRAMES,
    WINDOW_FPS,
    WINDOW_START_SECONDS,
    _choose_probability_threshold,
    _cross_validated_probabilities,
    _fit_classifier,
    _flow_cache_path,
    _flow_matrix,
    _matrix,
    _probabilities,
    _test_items,
    _train_items,
    _video_window_starts,
    temporal_descriptor,
)
from wangxing_project.xiaoyue_face_features import (
    build_feature_table,
    extract_face_sequences,
)

MODEL_TYPE = "zijie_face_mouth_temporal_v4"
TARGET_REAL_RECALL = 0.90
REGULARIZATION_CANDIDATES = (0.01, 0.02, 0.05, 0.10, 0.20)
BASE_DIM = 32
FLOW_START = BASE_DIM
FORENSIC_START = BASE_DIM + FLOW_FEATURE_DIM
TOTAL_DIM = FORENSIC_START + FORENSIC_FEATURE_DIM

# Region order: left eye, right eye, mouth, brow, lower face; six values each.
MOUTH_FORENSIC = tuple(
    [*range(FORENSIC_START + 12, FORENSIC_START + 18),
     *range(FORENSIC_START + 24, FORENSIC_START + 30)]
)
MOUTH_FLOW = tuple(
    [*range(FLOW_START + 8, FLOW_START + 12),
     *range(FLOW_START + 16, FLOW_START + 20)]
)
FEATURE_VARIANTS: dict[str, tuple[int, ...]] = {
    "temporal_flow_baseline": tuple(range(FORENSIC_START)),
    "local_forensic_only": tuple(range(FORENSIC_START, TOTAL_DIM)),
    "mouth_local_forensic": MOUTH_FORENSIC,
    "geometry_au_plus_forensic": tuple(
        [*range(0, 12), *range(16, 28), *range(FORENSIC_START, TOTAL_DIM)]
    ),
    "mouth_priority": tuple(
        [*range(16, 32), *MOUTH_FLOW, *MOUTH_FORENSIC]
    ),
    "temporal_plus_forensic": tuple(
        [*range(0, BASE_DIM), *range(FORENSIC_START, TOTAL_DIM)]
    ),
    "all_local_evidence": tuple(range(TOTAL_DIM)),
}


def _forensic_cache_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.stem}_forensic_v1.npz")


def _forensic_matrix(
    items: list[dict[str, Any]],
    *,
    cache_path: Path,
) -> np.ndarray:
    paths = [str(project_path(str(item["video"])).resolve()) for item in items]
    au_paths = [str(project_path(str(item["au"])).resolve()) for item in items]
    signatures = [
        f"{_file_signature(Path(video))}|{_file_signature(Path(au))}"
        for video, au in zip(paths, au_paths)
    ]
    config = {
        "version": "zijie_local_forensic_v1",
        "frames": WINDOW_FRAMES,
        "start_seconds": WINDOW_START_SECONDS,
        "duration_seconds": WINDOW_DURATION_SECONDS,
    }
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    if cache_path.is_file():
        try:
            with np.load(str(cache_path), allow_pickle=False) as payload:
                if (
                    str(payload["config_json"].item()) == config_json
                    and payload["paths"].tolist() == paths
                    and payload["signatures"].tolist() == signatures
                ):
                    return payload["features"].astype(np.float32)
        except (OSError, KeyError, ValueError):
            pass
    values: list[np.ndarray] = []
    for index, item in enumerate(items, start=1):
        values.append(
            extract_face_crop_forensic_features(
                video_path=project_path(str(item["video"])).resolve(),
                au_path=project_path(str(item["au"])).resolve(),
                max_frames=WINDOW_FRAMES,
                window_start_seconds=WINDOW_START_SECONDS,
                window_duration_seconds=WINDOW_DURATION_SECONDS,
            )
        )
        if index == 1 or index % 10 == 0 or index == len(items):
            print(f"[local forensic] {index}/{len(items)}", flush=True)
    matrix = np.stack(values).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(cache_path),
        paths=np.asarray(paths),
        signatures=np.asarray(signatures),
        config_json=np.asarray(config_json),
        features=matrix,
    )
    return matrix


def _training_matrix(
    items: list[dict[str, Any]],
    *,
    cache_path: Path,
) -> np.ndarray:
    table = build_feature_table(
        {"pairs": {"train": {"real": items, "fake": []}}},
        cache_path=cache_path,
        max_frames=WINDOW_FRAMES,
        window_start_seconds=WINDOW_START_SECONDS,
        window_duration_seconds=WINDOW_DURATION_SECONDS,
    )
    base = _matrix(items, table)
    flow = _flow_matrix(items, cache_path=_flow_cache_path(cache_path))
    forensic = _forensic_matrix(
        items,
        cache_path=_forensic_cache_path(cache_path),
    )
    return np.concatenate([base, flow, forensic], axis=1).astype(np.float32)


def _separation(real: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    real_p90 = float(np.quantile(real, 0.90))
    generated_p10 = float(np.quantile(generated, 0.10))
    return {
        "real_p90": real_p90,
        "generated_p10": generated_p10,
        "p10_generated_minus_p90_real": generated_p10 - real_p90,
        "maximum_real": float(np.max(real)),
        "minimum_generated": float(np.min(generated)),
        "minimum_generated_minus_maximum_real": float(
            np.min(generated) - np.max(real)
        ),
    }


def _select_model(
    real_x: np.ndarray,
    generated_x: np.ndarray,
    real_items: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for name, indexes in FEATURE_VARIANTS.items():
        columns = np.asarray(indexes, dtype=np.int64)
        for regularization in REGULARIZATION_CANDIDATES:
            real_oof, generated_oof = _cross_validated_probabilities(
                real_x[:, columns],
                generated_x[:, columns],
                real_items,
                regularization=regularization,
            )
            threshold, threshold_report = _choose_probability_threshold(
                real_oof,
                generated_oof,
            )
            selected_threshold = threshold_report["selected"]
            gap = _separation(real_oof, generated_oof)
            candidates.append(
                {
                    "name": name,
                    "feature_indexes": list(indexes),
                    "regularization": float(regularization),
                    "threshold_generated": threshold,
                    "real_oof": real_oof.astype(float).tolist(),
                    "generated_oof": generated_oof.astype(float).tolist(),
                    "real_recall": float(selected_threshold["real_recall"]),
                    "generated_recall": float(
                        selected_threshold["generated_recall"]
                    ),
                    "balanced_accuracy": float(
                        selected_threshold["balanced_accuracy"]
                    ),
                    "separation": gap,
                    "threshold_selection": threshold_report,
                }
            )
    baseline_candidates = [
        row for row in candidates if row["name"] == "temporal_flow_baseline"
    ]
    forensic_candidates = [
        row for row in candidates if row["name"] == "local_forensic_only"
    ]
    fusion_candidates: list[dict[str, Any]] = []
    for baseline in baseline_candidates:
        for forensic in forensic_candidates:
            for baseline_weight in (0.40, 0.50, 0.60, 0.75):
                forensic_weight = 1.0 - baseline_weight
                real_oof = (
                    baseline_weight * np.asarray(baseline["real_oof"])
                    + forensic_weight * np.asarray(forensic["real_oof"])
                )
                generated_oof = (
                    baseline_weight * np.asarray(baseline["generated_oof"])
                    + forensic_weight * np.asarray(forensic["generated_oof"])
                )
                threshold, threshold_report = _choose_probability_threshold(
                    real_oof,
                    generated_oof,
                )
                threshold_result = threshold_report["selected"]
                fusion_candidates.append(
                    {
                        "name": "temporal_flow_and_local_forensic_fusion",
                        "experts": [
                            {
                                "name": baseline["name"],
                                "feature_indexes": baseline["feature_indexes"],
                                "regularization": baseline["regularization"],
                                "weight": baseline_weight,
                            },
                            {
                                "name": forensic["name"],
                                "feature_indexes": forensic["feature_indexes"],
                                "regularization": forensic["regularization"],
                                "weight": forensic_weight,
                            },
                        ],
                        "threshold_generated": threshold,
                        "real_oof": real_oof.astype(float).tolist(),
                        "generated_oof": generated_oof.astype(float).tolist(),
                        "real_recall": float(threshold_result["real_recall"]),
                        "generated_recall": float(
                            threshold_result["generated_recall"]
                        ),
                        "balanced_accuracy": float(
                            threshold_result["balanced_accuracy"]
                        ),
                        "separation": _separation(real_oof, generated_oof),
                        "threshold_selection": threshold_report,
                    }
                )
    eligible = [
        row
        for row in fusion_candidates
        if row["real_recall"] >= TARGET_REAL_RECALL
        and row["generated_recall"] >= 1.0
    ]
    selected = max(
        eligible or fusion_candidates,
        key=lambda row: (
            row["generated_recall"],
            row["real_recall"],
            row["separation"]["p10_generated_minus_p90_real"],
            row["balanced_accuracy"],
        ),
    )
    return selected, {
        "selected": selected,
        "candidate_count": len(candidates),
        "fusion_candidate_count": len(fusion_candidates),
        "eligible_count": len(eligible),
        "candidates": candidates,
        "fusion_candidates": fusion_candidates,
    }


def _calibration_scale(selected: dict[str, Any]) -> float:
    robust_gap = float(
        selected["separation"]["p10_generated_minus_p90_real"]
    )
    return float(np.clip(robust_gap * 0.5, 0.04, 0.12))


def fit_zijie_face_temporal_v4(
    *,
    manifest: dict[str, Any],
    cache_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    train = (manifest.get("pairs") or {}).get("train") or {}
    real_items = list(train.get("real") or [])
    generated_items = list(train.get("fake") or [])
    if len(real_items) < 36 or len(generated_items) < 4:
        raise ValueError("ZiJie V4 requires at least 36 real and 4 unique AI videos.")
    items = [*real_items, *generated_items]
    matrix = _training_matrix(items, cache_path=cache_path)
    real_x = matrix[: len(real_items)]
    generated_x = matrix[len(real_items) :]
    selected, report = _select_model(real_x, generated_x, real_items)
    experts: list[dict[str, Any]] = []
    for expert_spec in selected["experts"]:
        columns = np.asarray(expert_spec["feature_indexes"], dtype=np.int64)
        scaler, classifier = _fit_classifier(
            real_x[:, columns],
            generated_x[:, columns],
            regularization=float(expert_spec["regularization"]),
        )
        experts.append(
            {
                "name": expert_spec["name"],
                "weight": float(expert_spec["weight"]),
                "feature_indexes": expert_spec["feature_indexes"],
                "regularization": expert_spec["regularization"],
                "scaler_mean": scaler.mean_.astype(float).tolist(),
                "scaler_scale": scaler.scale_.astype(float).tolist(),
                "coefficient": classifier.coef_[0].astype(float).tolist(),
                "intercept": float(classifier.intercept_[0]),
            }
        )
    profile = {
        "schema_version": "zijie_face_mouth_temporal_v4_profile",
        "subject": "zijie",
        "model_type": MODEL_TYPE,
        "window": {
            "start_seconds": WINDOW_START_SECONDS,
            "duration_seconds": WINDOW_DURATION_SECONDS,
            "sample_fps": WINDOW_FPS,
            "frames": WINDOW_FRAMES,
        },
        "video_aggregation": {
            "method": "median_generated_probability",
            "positions": ["start", "middle", "end"],
        },
        "feature_variant": selected["name"],
        "experts": experts,
        "threshold_generated": selected["threshold_generated"],
        "probability_margin_scale": _calibration_scale(selected),
        "training_counts": {
            "real": len(real_items),
            "generated_unique": len(generated_items),
        },
        "calibration": report,
        "feature_policy": {
            "background_used": False,
            "full_frame_rgb_used": False,
            "absolute_brightness_used": False,
            "face_mesh_used": True,
            "au_trajectory_used": True,
            "local_face_optical_flow_used": True,
            "local_normalized_texture_used": True,
            "motion_compensated_residual_used": True,
            "mouth_priority": True,
            "test_holdout_used_for_selection": False,
        },
        "test_training_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return profile


def _window_vector(item: dict[str, Any], start_seconds: float) -> np.ndarray:
    video = project_path(str(item["video"])).resolve()
    au = project_path(str(item["au"])).resolve()
    record = extract_face_sequences(
        video,
        au,
        max_frames=WINDOW_FRAMES,
        window_start_seconds=start_seconds,
        window_duration_seconds=WINDOW_DURATION_SECONDS,
    )
    from wangxing_project.face_crop_temporal import extract_face_crop_flow_features

    flow = extract_face_crop_flow_features(
        video,
        au,
        max_frames=WINDOW_FRAMES,
        window_start_seconds=start_seconds,
        window_duration_seconds=WINDOW_DURATION_SECONDS,
    )
    forensic = extract_face_crop_forensic_features(
        video,
        au,
        max_frames=WINDOW_FRAMES,
        window_start_seconds=start_seconds,
        window_duration_seconds=WINDOW_DURATION_SECONDS,
    )
    return np.concatenate([temporal_descriptor(record), flow, forensic])


def _raw_probability(vector: np.ndarray, profile: dict[str, Any]) -> float:
    probability = 0.0
    for expert in profile["experts"]:
        columns = np.asarray(expert["feature_indexes"], dtype=np.int64)
        mean = np.asarray(expert["scaler_mean"], dtype=np.float64)
        scale = np.maximum(
            np.asarray(expert["scaler_scale"], dtype=np.float64),
            1e-8,
        )
        coefficient = np.asarray(expert["coefficient"], dtype=np.float64)
        logit = float(np.dot((vector[columns] - mean) / scale, coefficient))
        logit += float(expert["intercept"])
        expert_probability = 1.0 / (
            1.0 + math.exp(-float(np.clip(logit, -30.0, 30.0)))
        )
        probability += float(expert["weight"]) * expert_probability
    return float(np.clip(probability, 0.0, 1.0))


def _margin_probability(raw: float, profile: dict[str, Any]) -> float:
    threshold = float(profile["threshold_generated"])
    scale = max(float(profile["probability_margin_scale"]), 1e-6)
    value = float(np.clip((raw - threshold) / scale, -30.0, 30.0))
    return float(1.0 / (1.0 + math.exp(-value)))


def _window_vectors_for_items(
    items: list[dict[str, Any]],
    *,
    cache_path: Path,
) -> tuple[list[list[float]], list[np.ndarray]]:
    paths = [str(project_path(str(item["video"])).resolve()) for item in items]
    au_paths = [str(project_path(str(item["au"])).resolve()) for item in items]
    signatures = [
        f"{_file_signature(Path(video))}|{_file_signature(Path(au))}"
        for video, au in zip(paths, au_paths)
    ]
    starts = [_video_window_starts(path) for path in paths]
    config_json = json.dumps(
        {
            "version": "zijie_v4_window_vectors_v1",
            "duration_seconds": WINDOW_DURATION_SECONDS,
            "frames": WINDOW_FRAMES,
            "starts": starts,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    if cache_path.is_file():
        try:
            with np.load(str(cache_path), allow_pickle=False) as payload:
                if (
                    str(payload["config_json"].item()) == config_json
                    and payload["paths"].tolist() == paths
                    and payload["signatures"].tolist() == signatures
                ):
                    counts = payload["counts"].astype(int).tolist()
                    flat = payload["vectors"].astype(np.float32)
                    vectors: list[np.ndarray] = []
                    offset = 0
                    for count in counts:
                        vectors.append(flat[offset : offset + count])
                        offset += count
                    print(f"[ZiJie V4 holdout] cache hit: {cache_path}", flush=True)
                    return starts, vectors
        except (OSError, KeyError, ValueError):
            pass
    vectors = []
    for index, (item, item_starts) in enumerate(zip(items, starts), start=1):
        vectors.append(
            np.stack([_window_vector(item, start) for start in item_starts])
            .astype(np.float32)
        )
        print(
            f"[ZiJie V4 holdout] {index}/{len(items)} "
            f"windows={len(item_starts)}",
            flush=True,
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(cache_path),
        paths=np.asarray(paths),
        signatures=np.asarray(signatures),
        config_json=np.asarray(config_json),
        counts=np.asarray([len(value) for value in vectors], dtype=np.int32),
        vectors=np.concatenate(vectors, axis=0),
    )
    return starts, vectors


def score_zijie_face_temporal_v4(
    *,
    manifest: dict[str, Any],
    profile: dict[str, Any],
    cache_path: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    predictions: list[int] = []
    threshold = float(profile["threshold_generated"])
    items = _test_items(manifest)
    starts_by_item, vectors_by_item = _window_vectors_for_items(
        items,
        cache_path=cache_path,
    )
    for item, starts, vectors in zip(items, starts_by_item, vectors_by_item):
        video = project_path(str(item["video"])).resolve()
        window_probabilities = [
            _raw_probability(vector, profile) for vector in vectors
        ]
        raw = float(np.median(window_probabilities))
        calibrated = _margin_probability(raw, profile)
        prediction = int(raw >= threshold)
        label = int(item.get("label_generated", 0))
        labels.append(label)
        predictions.append(prediction)
        rows.append(
            {
                "sample_id": item.get("sample_id"),
                "video": str(video),
                "label_generated": label,
                "prediction": "generated" if prediction else "real",
                "generated_probability": calibrated,
                "real_probability": 1.0 - calibrated,
                "raw_generated_probability": raw,
                "threshold_generated": threshold,
                "window_starts_seconds": starts,
                "window_raw_generated_probabilities": window_probabilities,
                "score_margin_from_threshold": raw - threshold,
            }
        )
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(predictions, dtype=np.int64)
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    real_raw = np.asarray(
        [row["raw_generated_probability"] for row in rows if not row["label_generated"]]
    )
    generated_raw = np.asarray(
        [row["raw_generated_probability"] for row in rows if row["label_generated"]]
    )
    return {
        "schema_version": "zijie_face_mouth_temporal_v4_evaluation",
        "subject": "zijie",
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
        "holdout_separation": (
            _separation(real_raw, generated_raw)
            if real_raw.size and generated_raw.size
            else None
        ),
        "rows": rows,
        "feature_policy": profile["feature_policy"],
        "training_allowed": False,
    }


def save_pt_checkpoint(profile: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": MODEL_TYPE,
            "profile": profile,
            "feature_policy": profile["feature_policy"],
        },
        str(output_path),
    )
