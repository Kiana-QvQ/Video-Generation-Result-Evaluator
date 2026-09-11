"""ZiJie V5 offline binary detector and AI-only quality ranker."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from evaluator.vedio_pred.real_video_detector import _file_signature
from wangxing_project.face_crop_temporal import extract_face_chroma_features

from wangxing_project.zijie_face_temporal_v4 import (
    _margin_probability,
    _raw_probability,
    _separation,
    _window_vectors_for_items,
    fit_zijie_face_temporal_v4,
)
from wangxing_project.zijie_face_temporal_v2 import temporal_descriptor

MODEL_TYPE = "zijie_face_mouth_temporal_v5"
RANK_FEATURE_VARIANTS = {
    "expert_means": (0, 3),
    "expert_distribution": (0, 1, 2, 3, 4, 5),
    "expert_consistency": tuple(range(10)),
    "reference_distance": tuple(range(10, 16)),
    "reference_mouth_priority": (11, 12, 13, 14, 15),
    "expert_and_reference": tuple(range(16)),
}
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)

# The 82-D V5 vector is partitioned so the two evidence scores do not reuse
# the final binary probability. Face uses upper-face dynamics; mouth uses
# mouth/lower-face geometry, AU, flow and local residual evidence.
FACE_NATURALNESS_INDEXES = tuple(
    [*range(0, 16), *range(32, 40), *range(44, 48),
     *range(52, 64), *range(70, 76)]
)
MOUTH_NATURALNESS_INDEXES = tuple(
    [*range(16, 32), *range(40, 44), *range(48, 52),
     *range(64, 70), *range(76, 82)]
)


def _cache_rows(path: Path, value_key: str) -> dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as payload:
        paths = [str(Path(value).resolve()) for value in payload["paths"].tolist()]
        values = payload[value_key]
    return {path: np.asarray(value, dtype=np.float32) for path, value in zip(paths, values)}


def _evidence_block_profile(
    vectors: np.ndarray,
    indexes: tuple[int, ...],
) -> dict[str, Any]:
    columns = np.asarray(indexes, dtype=np.int64)
    block = vectors[:, columns].astype(np.float64)
    center = np.median(block, axis=0)
    mad = np.median(np.abs(block - center), axis=0) * 1.4826
    std = np.std(block, axis=0)
    scale = np.maximum(np.where(mad > 1e-7, mad, std), 1e-6)
    distances = np.sqrt(np.mean(np.square((block - center) / scale), axis=1))
    median = float(np.quantile(distances, 0.50))
    p90 = float(np.quantile(distances, 0.90))
    temperature = max((p90 - median) / math.log(4.0), p90 * 0.05, 1e-3)
    return {
        "feature_indexes": list(indexes),
        "real_center": center.astype(float).tolist(),
        "real_scale": scale.astype(float).tolist(),
        "distance_median": median,
        "distance_p90": p90,
        "temperature": temperature,
        "mapping": "sigmoid_p50_80_p90_50",
    }


def fit_naturalness_evidence_profile(
    *,
    real_items: list[dict[str, Any]],
    cache_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    base_path = cache_root / "train_features.npz"
    flow_path = cache_root / "train_features_flow.npz"
    forensic_path = cache_root / "train_features_forensic_v1.npz"
    with np.load(str(base_path), allow_pickle=False) as payload:
        base_paths = [
            str(Path(value).resolve()) for value in payload["paths"].tolist()
        ]
        base = {
            path: temporal_descriptor({
                "face": payload["face"][index],
                "mouth": payload["mouth"][index],
            })
            for index, path in enumerate(base_paths)
        }
    flow = _cache_rows(flow_path, "flow")
    forensic = _cache_rows(forensic_path, "features")
    vectors = []
    for item in real_items:
        path = str(Path(str(item["video"])).resolve())
        if path not in base or path not in flow or path not in forensic:
            continue
        vectors.append(np.concatenate([base[path], flow[path], forensic[path]]))
    if len(vectors) < 20:
        raise ValueError(
            f"Naturalness evidence needs at least 20 real vectors; got {len(vectors)}."
        )
    matrix = np.stack(vectors).astype(np.float32)
    profile = {
        "schema_version": "zijie_v5_naturalness_evidence_v1",
        "subject": "zijie",
        "real_training_count": len(vectors),
        "face": _evidence_block_profile(matrix, FACE_NATURALNESS_INDEXES),
        "mouth": _evidence_block_profile(matrix, MOUTH_NATURALNESS_INDEXES),
        "classification_effect": "none",
        "coverage_effect": "none",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return profile


def naturalness_evidence_scores(
    vectors: np.ndarray,
    evidence_profile: dict[str, Any],
) -> dict[str, float]:
    values = np.asarray(vectors, dtype=np.float64)
    result: dict[str, float] = {}
    for name in ("face", "mouth"):
        spec = evidence_profile[name]
        indexes = np.asarray(spec["feature_indexes"], dtype=np.int64)
        center = np.asarray(spec["real_center"], dtype=np.float64)
        scale = np.maximum(np.asarray(spec["real_scale"], dtype=np.float64), 1e-8)
        window_distances = np.sqrt(
            np.mean(np.square((values[:, indexes] - center) / scale), axis=1)
        )
        distance = float(np.median(window_distances))
        p90 = float(spec["distance_p90"])
        temperature = max(float(spec["temperature"]), 1e-6)
        score = 100.0 / (
            1.0 + math.exp(float(np.clip((distance - p90) / temperature, -30, 30)))
        )
        result[f"{name}_naturalness_score_0_100"] = float(score)
        result[f"{name}_distance"] = distance
    return result


def au_landmark_coverage(au_path: str | Path) -> dict[str, float | int]:
    frame_indexes: list[int] = []
    valid_face: set[int] = set()
    valid_mouth: set[int] = set()
    face_fields = ("lm_mp_33_x", "lm_mp_33_y", "lm_mp_263_x", "lm_mp_263_y")
    mouth_fields = (
        "lm_mp_61_x", "lm_mp_61_y", "lm_mp_291_x", "lm_mp_291_y",
        "lm_mp_13_x", "lm_mp_13_y", "lm_mp_14_x", "lm_mp_14_y",
    )

    def finite_fields(row: dict[str, str], fields: tuple[str, ...]) -> bool:
        try:
            return all(math.isfinite(float(row.get(field, ""))) for field in fields)
        except (TypeError, ValueError):
            return False

    with Path(au_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle)):
            try:
                frame = int(float(row.get("frame_idx", row_number)))
            except (TypeError, ValueError):
                frame = row_number
            frame_indexes.append(frame)
            if finite_fields(row, face_fields):
                valid_face.add(frame)
            if finite_fields(row, mouth_fields):
                valid_mouth.add(frame)
    if not frame_indexes:
        return {
            "sampled_frame_span": 0,
            "face_valid_frames": 0,
            "mouth_valid_frames": 0,
            "face_coverage_0_1": 0.0,
            "mouth_coverage_0_1": 0.0,
        }
    span = max(frame_indexes) - min(frame_indexes) + 1
    return {
        "sampled_frame_span": span,
        "face_valid_frames": len(valid_face),
        "mouth_valid_frames": len(valid_mouth),
        "face_coverage_0_1": min(1.0, len(valid_face) / max(span, 1)),
        "mouth_coverage_0_1": min(1.0, len(valid_mouth) / max(span, 1)),
    }


def _chroma_matrix(items: list[dict[str, Any]], cache_path: Path) -> np.ndarray:
    paths = [str(Path(str(item["video"])).resolve()) for item in items]
    au_paths = [str(Path(str(item["au"])).resolve()) for item in items]
    signatures = [
        f"{_file_signature(Path(video))}|{_file_signature(Path(au))}"
        for video, au in zip(paths, au_paths)
    ]
    if cache_path.is_file():
        try:
            with np.load(str(cache_path), allow_pickle=False) as payload:
                if (
                    payload["paths"].tolist() == paths
                    and payload["signatures"].tolist() == signatures
                ):
                    return payload["features"].astype(np.float32)
        except (OSError, KeyError, ValueError):
            pass
    values = []
    for index, item in enumerate(items, start=1):
        values.append(
            extract_face_chroma_features(item["video"], item["au"])
        )
        if index == 1 or index % 10 == 0 or index == len(items):
            print(f"[ZiJie chroma] {index}/{len(items)}", flush=True)
    matrix = np.stack(values).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(cache_path),
        paths=np.asarray(paths),
        signatures=np.asarray(signatures),
        features=matrix,
    )
    return matrix


def fit_zijie_face_temporal_v5(
    *,
    manifest: dict[str, Any],
    cache_path: Path,
    output_path: Path,
    base_profile: dict[str, Any],
) -> dict[str, Any]:
    train = (manifest.get("pairs") or {}).get("train") or {}
    ranking_fit = list((manifest.get("ranking") or {}).get("fit") or [])
    ltx_manifest = {
        "pairs": {
            "train": {
                "real": list(train.get("real") or []),
                "fake": ranking_fit,
            }
        }
    }
    ltx_profile_path = output_path.with_name("zijie_ltx_domain_expert.json")
    ltx_profile = fit_zijie_face_temporal_v4(
        manifest=ltx_manifest,
        cache_path=cache_path,
        output_path=ltx_profile_path,
    )
    real_items = list(train.get("real") or [])
    real_chroma = _chroma_matrix(
        real_items,
        cache_path.with_name(f"{cache_path.stem}_real_chroma.npz"),
    )
    saturation_floor = float(
        max(0.05, np.quantile(real_chroma[:, 0], 0.01) - 0.03)
    )
    profile = {
        "schema_version": "zijie_face_mouth_temporal_v5_profile",
        "subject": "zijie",
        "model_type": MODEL_TYPE,
        "cascade": {
            "decision": "base_v4_or_ltx_domain_expert",
            "base_frozen": True,
            "description": (
                "The accepted V4 H3 detector is frozen. The LTX expert may "
                "add an AI decision but cannot retrain or shift the V4 head."
            ),
        },
        "base_profile": base_profile,
        "ltx_profile": ltx_profile,
        "face_chroma_gate": {
            "enabled": True,
            "saturation_floor": saturation_floor,
            "margin_scale": 0.03,
            "fit_source": "one_percentile_of_training_real_minus_0.03",
            "background_used": False,
        },
        "data_protocol": {
            "fixed_h3_final_preserved": True,
            "ltx_checkpoint_fit_count": len(ranking_fit),
            "ltx_checkpoint_holdout_count": len(
                ((manifest.get("ranking") or {}).get("holdout") or [])
            ),
        },
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
        },
        "test_training_allowed": False,
    }
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return profile


def score_zijie_face_temporal_v5(
    *, manifest: dict[str, Any], profile: dict[str, Any], cache_path: Path
) -> dict[str, Any]:
    items = [
        *list((((manifest.get("pairs") or {}).get("test") or {}).get("real") or [])),
        *list((((manifest.get("pairs") or {}).get("test") or {}).get("fake") or [])),
    ]
    starts_by_item, vectors_by_item = _window_vectors_for_items(
        items,
        cache_path=cache_path,
    )
    chroma = _chroma_matrix(
        items,
        cache_path.with_name(f"{cache_path.stem}_chroma.npz"),
    )
    base_profile = profile["base_profile"]
    ltx_profile = profile["ltx_profile"]
    base_threshold = float(base_profile["threshold_generated"])
    ltx_threshold = float(ltx_profile["threshold_generated"])
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    predictions: list[int] = []
    chroma_gate = profile["face_chroma_gate"]
    saturation_floor = float(chroma_gate["saturation_floor"])
    chroma_scale = max(float(chroma_gate["margin_scale"]), 1e-6)
    for item, starts, vectors, chroma_values in zip(
        items,
        starts_by_item,
        vectors_by_item,
        chroma,
    ):
        base_windows = [_raw_probability(vector, base_profile) for vector in vectors]
        ltx_windows = [_raw_probability(vector, ltx_profile) for vector in vectors]
        base_raw = float(np.median(base_windows))
        ltx_raw = float(np.median(ltx_windows))
        decision_margin = max(
            base_raw - base_threshold,
            ltx_raw - ltx_threshold,
            saturation_floor - float(chroma_values[0]),
        )
        prediction = int(decision_margin >= 0.0)
        base_calibrated = _margin_probability(base_raw, base_profile)
        ltx_calibrated = _margin_probability(ltx_raw, ltx_profile)
        generated_probability = max(base_calibrated, ltx_calibrated)
        chroma_probability = float(
            1.0
            / (
                1.0
                + np.exp(
                    -np.clip(
                        (saturation_floor - float(chroma_values[0]))
                        / chroma_scale,
                        -30.0,
                        30.0,
                    )
                )
            )
        )
        generated_probability = max(generated_probability, chroma_probability)
        label = int(item.get("label_generated", 0))
        labels.append(label)
        predictions.append(prediction)
        rows.append(
            {
                "sample_id": item.get("sample_id"),
                "video": item.get("video"),
                "label_generated": label,
                "prediction": "generated" if prediction else "real",
                "generated_probability": generated_probability,
                "real_probability": 1.0 - generated_probability,
                "decision_margin": decision_margin,
                "base_v4": {
                    "raw_generated_probability": base_raw,
                    "threshold_generated": base_threshold,
                    "window_probabilities": base_windows,
                },
                "ltx_expert": {
                    "raw_generated_probability": ltx_raw,
                    "threshold_generated": ltx_threshold,
                    "window_probabilities": ltx_windows,
                },
                "face_chroma_gate": {
                    "mean_saturation": float(chroma_values[0]),
                    "p10_saturation": float(chroma_values[1]),
                    "mean_chroma": float(chroma_values[2]),
                    "saturation_floor": saturation_floor,
                    "triggered": bool(float(chroma_values[0]) < saturation_floor),
                },
                "window_starts_seconds": starts,
            }
        )
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(predictions, dtype=np.int64)
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    real_margin = np.asarray(
        [row["decision_margin"] for row in rows if not row["label_generated"]]
    )
    generated_margin = np.asarray(
        [row["decision_margin"] for row in rows if row["label_generated"]]
    )
    return {
        "schema_version": "zijie_face_mouth_temporal_v5_evaluation",
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
            _separation(real_margin, generated_margin)
            if real_margin.size and generated_margin.size
            else None
        ),
        "rows": rows,
        "feature_policy": profile["feature_policy"],
        "training_allowed": False,
    }


def _expert_probability(
    vector: np.ndarray,
    expert: dict[str, Any],
) -> float:
    columns = np.asarray(expert["feature_indexes"], dtype=np.int64)
    mean = np.asarray(expert["scaler_mean"], dtype=np.float64)
    scale = np.maximum(np.asarray(expert["scaler_scale"], dtype=np.float64), 1e-8)
    coefficient = np.asarray(expert["coefficient"], dtype=np.float64)
    logit = float(np.dot((vector[columns] - mean) / scale, coefficient))
    logit += float(expert["intercept"])
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0))))


def _rank_features(
    vectors: np.ndarray,
    profile: dict[str, Any],
    reference_vectors: np.ndarray | None = None,
) -> np.ndarray:
    experts = list(profile["experts"])
    probabilities = [
        np.asarray(
            [_expert_probability(vector, expert) for vector in vectors],
            dtype=np.float64,
        )
        for expert in experts
    ]
    combined = np.asarray(
        [_raw_probability(vector, profile) for vector in vectors],
        dtype=np.float64,
    )
    values: list[float] = []
    for probability in probabilities:
        values.extend(
            [
                float(np.mean(probability)),
                float(np.std(probability)),
                float(np.max(probability) - np.min(probability)),
            ]
        )
    values.extend(
        [
            float(np.mean(combined)),
            float(np.std(combined)),
            float(np.max(combined) - np.min(combined)),
            float(np.mean(np.abs(probabilities[0] - probabilities[1]))),
        ]
    )
    if reference_vectors is not None:
        count = min(len(vectors), len(reference_vectors))
        scale = np.ones(vectors.shape[1], dtype=np.float64)
        for expert in experts:
            indexes = np.asarray(expert["feature_indexes"], dtype=np.int64)
            scale[indexes] = np.maximum(
                np.asarray(expert["scaler_scale"], dtype=np.float64),
                1e-8,
            )
        normalized_delta = (
            vectors[:count].astype(np.float64)
            - reference_vectors[:count].astype(np.float64)
        ) / scale

        def block_distance(start: int, stop: int) -> np.ndarray:
            return np.sqrt(
                np.mean(np.square(normalized_delta[:, start:stop]), axis=1)
            )

        blocks = [
            block_distance(0, 16),
            block_distance(16, 32),
            block_distance(32, 52),
            block_distance(52, 82),
        ]
        combined_distance = np.mean(np.stack(blocks, axis=1), axis=1)
        values.extend(
            [
                *[float(np.mean(block)) for block in blocks],
                float(np.std(combined_distance)),
                float(np.max(combined_distance)),
            ]
        )
    return np.asarray(values, dtype=np.float32)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(expected: np.ndarray, predicted: np.ndarray) -> float:
    if len(expected) < 2:
        return 0.0
    left = _rankdata(expected)
    right = _rankdata(predicted)
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _pairwise_rate(expected: np.ndarray, predicted: np.ndarray) -> float:
    correct = 0
    total = 0
    for left in range(len(expected)):
        for right in range(left + 1, len(expected)):
            expected_delta = float(expected[left] - expected[right])
            if abs(expected_delta) < 1e-8:
                continue
            predicted_delta = float(predicted[left] - predicted[right])
            correct += int(expected_delta * predicted_delta > 0.0)
            total += 1
    return correct / total if total else 0.0


def _rank_metrics(expected: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "spearman": _spearman(expected, predicted),
        "pairwise_ordering_rate": _pairwise_rate(expected, predicted),
        "mae": float(np.mean(np.abs(expected - predicted))),
        "score_range": float(np.max(predicted) - np.min(predicted)),
    }


def _fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    indexes: tuple[int, ...],
    alpha: float,
) -> tuple[StandardScaler, Ridge]:
    columns = np.asarray(indexes, dtype=np.int64)
    scaler = StandardScaler().fit(x[:, columns])
    model = Ridge(alpha=alpha).fit(scaler.transform(x[:, columns]), y)
    return scaler, model


def predict_quality_score(
    vectors: np.ndarray,
    binary_profile: dict[str, Any],
    rank_policy: dict[str, Any],
    reference_vectors: np.ndarray | None = None,
) -> float:
    features = _rank_features(vectors, binary_profile, reference_vectors)
    indexes = np.asarray(rank_policy["feature_indexes"], dtype=np.int64)
    mean = np.asarray(rank_policy["scaler_mean"], dtype=np.float64)
    scale = np.maximum(
        np.asarray(rank_policy["scaler_scale"], dtype=np.float64),
        1e-8,
    )
    coefficient = np.asarray(rank_policy["coefficient"], dtype=np.float64)
    value = float(
        np.dot((features[indexes] - mean) / scale, coefficient)
        + float(rank_policy["intercept"])
    )
    low, high = [float(item) for item in rank_policy["score_clip"]]
    return float(np.clip(value, low, high))


def fit_quality_ranker(
    *,
    manifest: dict[str, Any],
    binary_profile: dict[str, Any],
    cache_path: Path,
    output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ranking = manifest.get("ranking") or {}
    fit_items = list(ranking.get("fit") or [])
    holdout_items = list(ranking.get("holdout") or [])
    all_items = [*fit_items, *holdout_items]
    reference = ranking.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("ZiJie quality ranking requires an explicit GT reference.")
    _, all_vectors = _window_vectors_for_items(
        [*all_items, reference],
        cache_path=cache_path,
    )
    vectors = all_vectors[:-1]
    reference_vectors = all_vectors[-1]
    x = np.stack(
        [
            _rank_features(value, binary_profile, reference_vectors)
            for value in vectors
        ]
    )
    y = np.asarray(
        [float(item["quality_target_0_100"]) for item in all_items],
        dtype=np.float64,
    )
    fit_count = len(fit_items)
    fit_x, holdout_x = x[:fit_count], x[fit_count:]
    fit_y, holdout_y = y[:fit_count], y[fit_count:]
    candidates: list[dict[str, Any]] = []
    for name, indexes in RANK_FEATURE_VARIANTS.items():
        for alpha in RIDGE_ALPHAS:
            oof = np.zeros(fit_count, dtype=np.float64)
            for index in range(fit_count):
                keep = np.arange(fit_count) != index
                scaler, model = _fit_ridge(
                    fit_x[keep], fit_y[keep], indexes, alpha
                )
                columns = np.asarray(indexes, dtype=np.int64)
                oof[index] = model.predict(
                    scaler.transform(fit_x[index : index + 1, columns])
                )[0]
            metrics = _rank_metrics(fit_y, oof)
            candidates.append(
                {
                    "name": name,
                    "feature_indexes": list(indexes),
                    "alpha": alpha,
                    "oof_predictions": oof.astype(float).tolist(),
                    "metrics": metrics,
                }
            )
    selected = max(
        candidates,
        key=lambda row: (
            row["metrics"]["pairwise_ordering_rate"],
            row["metrics"]["spearman"],
            -row["metrics"]["mae"],
            -len(row["feature_indexes"]),
        ),
    )
    indexes = tuple(int(value) for value in selected["feature_indexes"])
    requires_gt_reference = any(index >= 10 for index in indexes)
    scaler, model = _fit_ridge(
        fit_x,
        fit_y,
        indexes,
        float(selected["alpha"]),
    )
    columns = np.asarray(indexes, dtype=np.int64)
    holdout_prediction = np.clip(
        model.predict(scaler.transform(holdout_x[:, columns])),
        5.0,
        80.0,
    )
    holdout_metrics = _rank_metrics(holdout_y, holdout_prediction)
    usable = bool(
        holdout_metrics["pairwise_ordering_rate"] >= 5.0 / 6.0
        and holdout_metrics["spearman"] >= 0.60
        and holdout_metrics["score_range"] >= 3.0
    )
    policy = {
        "schema_version": "zijie_ai_quality_rank_v1",
        "subject": "zijie",
        "development_only": True,
        "usable_for_offline_ranking": usable,
        "runtime_uses_checkpoint_step": False,
        "requires_gt_reference": requires_gt_reference,
        "reference_sample_id": (
            reference["sample_id"] if requires_gt_reference else None
        ),
        "feature_variant": selected["name"],
        "feature_indexes": list(indexes),
        "alpha": selected["alpha"],
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "coefficient": model.coef_.astype(float).tolist(),
        "intercept": float(model.intercept_),
        "score_clip": [5.0, 80.0],
        "fit_selection": {
            "selected": selected,
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
        "holdout_metrics": holdout_metrics,
    }
    rows = []
    for item, expected, predicted in zip(
        fit_items,
        fit_y,
        np.asarray(selected["oof_predictions"], dtype=np.float64),
    ):
        rows.append(
            {
                "sample_id": item["sample_id"],
                "checkpoint_step": item["checkpoint_step"],
                "split": "fit_oof",
                "expected_quality": float(expected),
                "predicted_quality": float(np.clip(predicted, 5.0, 80.0)),
            }
        )
    for item, expected, predicted in zip(
        holdout_items,
        holdout_y,
        holdout_prediction,
    ):
        rows.append(
            {
                "sample_id": item["sample_id"],
                "checkpoint_step": item["checkpoint_step"],
                "split": "holdout",
                "expected_quality": float(expected),
                "predicted_quality": float(predicted),
            }
        )
    rows.sort(key=lambda row: int(row["checkpoint_step"]))
    vector_by_id = {
        str(item["sample_id"]): value for item, value in zip(all_items, vectors)
    }
    for row in rows:
        item_vectors = vector_by_id[str(row["sample_id"])]
        binary_ai = float(
            np.median(
                [_raw_probability(vector, binary_profile) for vector in item_vectors]
            )
        )
        row["binary_generated_probability"] = binary_ai
        row["binary_real_direction_probability"] = 1.0 - binary_ai
    report = {
        "schema_version": "zijie_ai_quality_rank_evaluation_v1",
        "fit_count": fit_count,
        "holdout_count": len(holdout_items),
        "fit_oof_metrics": selected["metrics"],
        "holdout_metrics": holdout_metrics,
        "usable_for_offline_ranking": usable,
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return policy, report


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
