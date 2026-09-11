"""ZiJie V2 fixed-window face-and-mouth temporal classifier.

V1 compared full clips of very different durations.  V2 always compares the
first 1.5 seconds on the same 8 FPS / 13-frame time axis, then trains a small
supervised head on pose-normalized Face Mesh, AU and local face-crop dynamics.
It intentionally excludes global RGB, background and absolute brightness.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evaluator.modules.core.paths import project_path
from evaluator.vedio_pred.real_video_detector import _file_signature

from .face_crop_temporal import (
    FLOW_FEATURE_DIM,
    extract_face_crop_flow_features,
)
from .xiaoyue_face_features import (
    build_feature_table,
    extract_face_sequences,
)

MODEL_TYPE = "zijie_face_mouth_temporal_v3"
WINDOW_START_SECONDS = 0.0
WINDOW_DURATION_SECONDS = 1.5
WINDOW_FPS = 8
WINDOW_FRAMES = 13
VIDEO_WINDOW_AGGREGATION = "median_generated_probability"
TARGET_REAL_RECALL = 0.90
RANDOM_STATE = 42

FEATURE_VARIANTS: dict[str, tuple[int, ...]] = {
    # All fixed-window face, AU, mouth and local face-crop temporal features.
    "all_temporal": tuple(range(32)),
    # Remove crop statistics so compression or facial rendering texture cannot
    # dominate the decision; retain Face Mesh and AU temporal evidence.
    "geometry_au_only": tuple([*range(0, 12), *range(16, 28)]),
    # AU trajectories are the least sensitive to crop appearance.
    "au_only": tuple([*range(4, 12), *range(20, 28)]),
    # Keep only variation, velocity and jerk instead of pose/expression means.
    "dynamics_only": tuple(
        index
        for block_start in range(0, 32, 4)
        for index in range(block_start + 1, block_start + 4)
    ),
    # Dynamics from geometry and AU only, without any crop residual signal.
    "geometry_au_dynamics": tuple(
        index
        for block_start in (*range(0, 12, 4), *range(16, 28, 4))
        for index in range(block_start + 1, block_start + 4)
    ),
    # The 20 local optical-flow values are face-anchored and contain no
    # background or global image appearance.
    "flow_consistency_only": tuple(range(32, 32 + FLOW_FEATURE_DIM)),
    "geometry_au_plus_flow": tuple(
        [*range(0, 12), *range(16, 28), *range(32, 32 + FLOW_FEATURE_DIM)]
    ),
    "all_temporal_plus_flow": tuple(range(32 + FLOW_FEATURE_DIM)),
}
REGULARIZATION_CANDIDATES = (0.02, 0.05, 0.10, 0.20)


def _train_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    train = (manifest.get("pairs") or {}).get("train") or {}
    return [*list(train.get("real") or []), *list(train.get("fake") or [])]


def _test_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    test = (manifest.get("pairs") or {}).get("test") or {}
    return [*list(test.get("real") or []), *list(test.get("fake") or [])]


def _finite(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def _block_descriptor(sequence: np.ndarray, start: int, stop: int) -> list[float]:
    """Summarize one semantic feature block on the common time axis."""
    block = np.asarray(sequence[:, start:stop], dtype=np.float64)
    if block.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    velocity = np.abs(np.diff(block, axis=0))
    jerk = np.abs(np.diff(block, n=2, axis=0))
    return [
        _finite(float(np.mean(np.std(block, axis=0)))),
        _finite(float(np.mean(velocity))) if velocity.size else 0.0,
        _finite(float(np.quantile(velocity, 0.90))) if velocity.size else 0.0,
        _finite(float(np.quantile(jerk, 0.90))) if jerk.size else 0.0,
    ]


def temporal_descriptor(record: dict[str, Any]) -> np.ndarray:
    """Return a compact, duration-independent face/mouth dynamics vector."""
    face = np.asarray(record["face"], dtype=np.float32)
    mouth = np.asarray(record["mouth"], dtype=np.float32)
    values: list[float] = []
    # Face: geometry, AU intensity, AU presence, five local crop descriptors.
    for start, stop in ((0, 12), (12, 24), (24, 36), (36, 61)):
        values.extend(_block_descriptor(face, start, stop))
    # Mouth: geometry, mouth AUs, presence and the mouth-local crop descriptor.
    for start, stop in ((0, 6), (6, 14), (14, 20), (20, 25)):
        values.extend(_block_descriptor(mouth, start, stop))
    return np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _matrix(
    items: list[dict[str, Any]],
    table: dict[str, Any],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for item in items:
        video = str(project_path(str(item["video"])).resolve())
        rows.append(temporal_descriptor(table["features"][video]))
    if not rows:
        raise ValueError("No temporal feature rows are available.")
    return np.stack(rows).astype(np.float32)


def _flow_matrix(
    items: list[dict[str, Any]],
    *,
    cache_path: Path,
) -> np.ndarray:
    """Cache fixed-window local-flow features independently of base features."""
    paths = [
        str(project_path(str(item["video"])).resolve()) for item in items
    ]
    signatures = [
        _file_signature(project_path(str(item["video"])).resolve())
        for item in items
    ]
    config = {
        "version": "zijie_local_flow_v1",
        "frames": WINDOW_FRAMES,
        "start_seconds": WINDOW_START_SECONDS,
        "duration_seconds": WINDOW_DURATION_SECONDS,
    }
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    if cache_path.is_file():
        try:
            with np.load(str(cache_path), allow_pickle=False) as payload:
                cached_paths = [str(value) for value in payload["paths"].tolist()]
                cached_signatures = [
                    str(value) for value in payload["signatures"].tolist()
                ]
                if (
                    str(payload["config_json"].item()) == config_json
                    and cached_paths == paths
                    and cached_signatures == signatures
                ):
                    return payload["flow"].astype(np.float32)
        except (OSError, KeyError, ValueError):
            pass
    values: list[np.ndarray] = []
    for index, item in enumerate(items, start=1):
        values.append(
            extract_face_crop_flow_features(
                video_path=project_path(str(item["video"])).resolve(),
                au_path=project_path(str(item["au"])).resolve(),
                max_frames=WINDOW_FRAMES,
                window_start_seconds=WINDOW_START_SECONDS,
                window_duration_seconds=WINDOW_DURATION_SECONDS,
            )
        )
        if index % 5 == 0 or index == len(items):
            print(f"[face flow] {index}/{len(items)}", flush=True)
    matrix = np.stack(values).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(cache_path),
        paths=np.asarray(paths),
        signatures=np.asarray(signatures),
        config_json=np.asarray(config_json),
        flow=matrix,
    )
    return matrix


def _flow_cache_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.stem}_flow.npz")


def _video_window_starts(video_path: str | Path) -> list[float]:
    """Return start/middle/end windows without adding duration as a feature."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = max(float(capture.get(cv2.CAP_PROP_FPS) or 0.0), 1.0)
    count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0), 1)
    capture.release()
    available = max(0.0, (count - 1) / fps - WINDOW_DURATION_SECONDS)
    return sorted(
        {
            round(value, 6)
            for value in (0.0, available * 0.5, available)
        }
    )


def _window_vector(
    item: dict[str, Any],
    *,
    start_seconds: float,
) -> np.ndarray:
    video = project_path(str(item["video"])).resolve()
    au = project_path(str(item["au"])).resolve()
    record = extract_face_sequences(
        video,
        au,
        max_frames=WINDOW_FRAMES,
        window_start_seconds=start_seconds,
        window_duration_seconds=WINDOW_DURATION_SECONDS,
    )
    flow = extract_face_crop_flow_features(
        video,
        au,
        max_frames=WINDOW_FRAMES,
        window_start_seconds=start_seconds,
        window_duration_seconds=WINDOW_DURATION_SECONDS,
    )
    return np.concatenate([temporal_descriptor(record), flow], axis=0)


def _fit_classifier(
    real_x: np.ndarray,
    fake_x: np.ndarray,
    *,
    regularization: float,
) -> tuple[StandardScaler, LogisticRegression]:
    x = np.concatenate([real_x, fake_x], axis=0)
    y = np.asarray([0] * len(real_x) + [1] * len(fake_x), dtype=np.int64)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x)
    classifier = LogisticRegression(
        C=float(regularization),
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )
    classifier.fit(scaled, y)
    return scaler, classifier


def _probabilities(
    x: np.ndarray,
    *,
    scaler: StandardScaler,
    classifier: LogisticRegression,
) -> np.ndarray:
    return classifier.predict_proba(scaler.transform(x))[:, 1].astype(np.float64)


def _choose_probability_threshold(
    real_probability: np.ndarray,
    fake_probability: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    values = np.unique(np.concatenate([real_probability, fake_probability]))
    boundaries = [float(values[0] - 1e-8)]
    boundaries.extend(
        float((left + right) * 0.5)
        for left, right in zip(values[:-1], values[1:])
    )
    boundaries.append(float(values[-1] + 1e-8))
    candidates: list[dict[str, float]] = []
    for threshold in boundaries:
        real_recall = float(np.mean(real_probability < threshold))
        generated_recall = float(np.mean(fake_probability >= threshold))
        candidates.append(
            {
                "threshold_generated": threshold,
                "real_recall": real_recall,
                "generated_recall": generated_recall,
                "balanced_accuracy": 0.5 * (real_recall + generated_recall),
            }
        )
    eligible = [
        row for row in candidates if row["real_recall"] >= TARGET_REAL_RECALL
    ]
    selected = max(
        eligible or candidates,
        key=lambda row: (
            row["generated_recall"],
            row["balanced_accuracy"],
            -row["threshold_generated"],
        ),
    )
    return float(selected["threshold_generated"]), {
        "target_real_recall": TARGET_REAL_RECALL,
        "selected": selected,
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
    }


def _cross_validated_probabilities(
    real_x: np.ndarray,
    fake_x: np.ndarray,
    real_items: list[dict[str, Any]],
    *,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Produce train-only calibration scores without reusing a real capture."""
    groups = np.asarray(
        [str(item.get("group_id") or item.get("capture_id")) for item in real_items]
    )
    real_oof = np.zeros(len(real_x), dtype=np.float64)
    for group in sorted(set(groups.tolist())):
        test_mask = groups == group
        scaler, classifier = _fit_classifier(
            real_x[~test_mask],
            fake_x,
            regularization=regularization,
        )
        real_oof[test_mask] = _probabilities(
            real_x[test_mask],
            scaler=scaler,
            classifier=classifier,
        )
    fake_oof = np.zeros(len(fake_x), dtype=np.float64)
    for index in range(len(fake_x)):
        keep = np.arange(len(fake_x)) != index
        scaler, classifier = _fit_classifier(
            real_x,
            fake_x[keep],
            regularization=regularization,
        )
        fake_oof[index] = _probabilities(
            fake_x[index : index + 1],
            scaler=scaler,
            classifier=classifier,
        )[0]
    return real_oof, fake_oof


def _select_model_variant(
    real_x: np.ndarray,
    fake_x: np.ndarray,
    real_items: list[dict[str, Any]],
) -> tuple[str, tuple[int, ...], float, float, dict[str, Any]]:
    """Choose a constrained head from group/AI leave-one-out calibration."""
    candidates: list[dict[str, Any]] = []
    for name, indexes in FEATURE_VARIANTS.items():
        columns = np.asarray(indexes, dtype=np.int64)
        for regularization in REGULARIZATION_CANDIDATES:
            real_oof, fake_oof = _cross_validated_probabilities(
                real_x[:, columns],
                fake_x[:, columns],
                real_items,
                regularization=regularization,
            )
            threshold, report = _choose_probability_threshold(real_oof, fake_oof)
            selected = report["selected"]
            candidates.append(
                {
                    "name": name,
                    "feature_indexes": list(indexes),
                    "regularization": float(regularization),
                    "threshold_generated": threshold,
                    "real_oof": real_oof.astype(float).tolist(),
                    "generated_oof": fake_oof.astype(float).tolist(),
                    "selection": report,
                    "real_recall": float(selected["real_recall"]),
                    "generated_recall": float(selected["generated_recall"]),
                    "balanced_accuracy": float(selected["balanced_accuracy"]),
                }
            )
    eligible = [
        row
        for row in candidates
        if row["real_recall"] >= TARGET_REAL_RECALL
    ]
    selected = max(
        eligible or candidates,
        key=lambda row: (
            row["generated_recall"],
            row["balanced_accuracy"],
            -row["regularization"],
            -len(row["feature_indexes"]),
        ),
    )
    return (
        str(selected["name"]),
        tuple(int(value) for value in selected["feature_indexes"]),
        float(selected["regularization"]),
        float(selected["threshold_generated"]),
        {
            "selected": selected,
            "candidate_count": len(candidates),
            "eligible_count": len(eligible),
            "candidates": candidates,
        },
    )


def fit_zijie_face_temporal_v2(
    *,
    manifest: dict[str, Any],
    cache_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    train = (manifest.get("pairs") or {}).get("train") or {}
    real_items = list(train.get("real") or [])
    fake_items = list(train.get("fake") or [])
    if len(real_items) < 36 or len(fake_items) < 4:
        raise ValueError(
            "ZiJie V2 requires at least 36 real and 4 unique generated "
            "training videos."
        )
    table = build_feature_table(
        {"pairs": {"train": {"real": real_items, "fake": fake_items}}},
        cache_path=cache_path,
        max_frames=WINDOW_FRAMES,
        window_start_seconds=WINDOW_START_SECONDS,
        window_duration_seconds=WINDOW_DURATION_SECONDS,
    )
    real_x = _matrix(real_items, table)
    fake_x = _matrix(fake_items, table)
    flow = _flow_matrix(
        [*real_items, *fake_items],
        cache_path=_flow_cache_path(cache_path),
    )
    real_x = np.concatenate([real_x, flow[: len(real_x)]], axis=1)
    fake_x = np.concatenate([fake_x, flow[len(real_x) :]], axis=1)
    (
        variant_name,
        feature_indexes,
        regularization,
        threshold,
        model_selection,
    ) = _select_model_variant(
        real_x,
        fake_x,
        real_items,
    )
    columns = np.asarray(feature_indexes, dtype=np.int64)
    real_oof = np.asarray(
        model_selection["selected"]["real_oof"],
        dtype=np.float64,
    )
    fake_oof = np.asarray(
        model_selection["selected"]["generated_oof"],
        dtype=np.float64,
    )
    scaler, classifier = _fit_classifier(
        real_x[:, columns],
        fake_x[:, columns],
        regularization=regularization,
    )
    payload = {
        "schema_version": "zijie_face_mouth_temporal_v2_profile",
        "subject": "zijie",
        "model_type": MODEL_TYPE,
        "window": {
            "start_seconds": WINDOW_START_SECONDS,
            "duration_seconds": WINDOW_DURATION_SECONDS,
            "sample_fps": WINDOW_FPS,
            "frames": WINDOW_FRAMES,
        },
        "video_aggregation": {
            "method": VIDEO_WINDOW_AGGREGATION,
            "positions": ["start", "middle", "end"],
            "description": (
                "Each video is scored on start/middle/end fixed windows; the "
                "median generated probability is used so one atypical real "
                "moment does not determine the video-level decision."
            ),
        },
        "feature_variant": variant_name,
        "feature_indexes": list(feature_indexes),
        "regularization": regularization,
        "base_feature_dim": 32,
        "flow_feature_dim": FLOW_FEATURE_DIM,
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "coefficient": classifier.coef_[0].astype(float).tolist(),
        "intercept": float(classifier.intercept_[0]),
        "threshold_generated": threshold,
        "training_counts": {
            "real": len(real_items),
            "generated_unique": len(fake_items),
        },
        "calibration": {
            "real_group_oof_probability": real_oof.astype(float).tolist(),
            "generated_leave_one_out_probability": fake_oof.astype(float).tolist(),
            "model_selection": model_selection,
        },
        "feature_policy": {
            "full_frame_rgb_used": False,
            "full_frame_hsv_used": False,
            "background_used": False,
            "absolute_brightness_used": False,
            "face_mesh_used": True,
            "au_trajectory_used": True,
            "local_face_crop_temporal_used": True,
            "local_face_optical_flow_used": True,
            "mouth_priority": True,
            "duration_feature_used": False,
            "fixed_time_window_used": True,
            "description": (
                "Fixed 1.5-second face-and-mouth temporal classification over "
                "pose-normalized Face Mesh, AU and local brightness-centered "
                "crops."
            ),
        },
        "test_training_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _profile_probability(vector: np.ndarray, profile: dict[str, Any]) -> float:
    mean = np.asarray(profile["scaler_mean"], dtype=np.float64)
    scale = np.maximum(np.asarray(profile["scaler_scale"], dtype=np.float64), 1e-8)
    coefficient = np.asarray(profile["coefficient"], dtype=np.float64)
    logit = float(np.dot((vector.astype(np.float64) - mean) / scale, coefficient))
    logit += float(profile["intercept"])
    logit = float(np.clip(logit, -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-logit)))


def score_zijie_face_temporal_v2(
    *,
    manifest: dict[str, Any],
    profile: dict[str, Any],
    cache_path: Path,
) -> dict[str, Any]:
    items = _test_items(manifest)
    del cache_path  # Multi-window extraction is intentionally per-video.
    threshold = float(profile["threshold_generated"])
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    predictions: list[int] = []
    for item in items:
        video = str(project_path(str(item["video"])).resolve())
        starts = _video_window_starts(video)
        probabilities = [
            _profile_probability(
                _window_vector(item, start_seconds=start)[
                    np.asarray(profile["feature_indexes"], dtype=np.int64)
                ],
                profile,
            )
            for start in starts
        ]
        p_generated = float(np.median(probabilities))
        prediction = int(p_generated >= threshold)
        label = int(item.get("label_generated", 0))
        labels.append(label)
        predictions.append(prediction)
        rows.append(
            {
                "sample_id": item.get("sample_id"),
                "video": video,
                "label_generated": label,
                "prediction": "generated" if prediction else "real",
                "generated_probability": p_generated,
                "real_probability": 1.0 - p_generated,
                "threshold_generated": threshold,
                "window_starts_seconds": starts,
                "window_generated_probabilities": probabilities,
                "video_aggregation": VIDEO_WINDOW_AGGREGATION,
                "temporal_window_seconds": float(
                    (profile.get("window") or {}).get(
                        "duration_seconds",
                        WINDOW_DURATION_SECONDS,
                    )
                ),
                "temporal_window_frames": int(
                    (profile.get("window") or {}).get("frames", WINDOW_FRAMES)
                ),
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
        "schema_version": "zijie_face_mouth_temporal_v3_evaluation",
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
        output_path,
    )
