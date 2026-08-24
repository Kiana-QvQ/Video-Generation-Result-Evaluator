"""Small Wang Xing V5 DriveHead trained on facial motion sequences.

The head is intentionally linear and shallow.  It consumes AU/landmark
transition features and optionally reuses an existing MediaPipe Blendshape
cache.  It never updates or thresholds the frozen V3 classifier.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evaluator.modules.core.paths import project_path
from wangxing_project.blendshape_temporal import (
    BLENDSHAPE_FEATURE_DIM,
    blendshape_temporal_vector,
)
from wangxing_project.temporal_expression import (
    TRANSITION_FEATURE_NAMES,
    extract_transition_features,
)

DRIVE_SCHEMA = "wangxing_v5_drive_head_v1"
BLENDSHAPE_FEATURE_NAMES = tuple(
    f"blendshape_{index:03d}" for index in range(BLENDSHAPE_FEATURE_DIM)
)
DRIVE_FEATURE_NAMES = tuple(TRANSITION_FEATURE_NAMES)


def build_drive_feature_names(include_blendshape: bool = False) -> tuple[str, ...]:
    if not include_blendshape:
        return DRIVE_FEATURE_NAMES
    return (
        DRIVE_FEATURE_NAMES
        + BLENDSHAPE_FEATURE_NAMES
        + ("blendshape_missing_mask",)
    )


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _path_key(path: str | Path) -> str:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = project_path(value)
    try:
        return str(value.resolve()).casefold()
    except OSError:
        return str(value).casefold()


def _cache_key(video_path: Path) -> str:
    signature = f"{_path_key(video_path)}|{video_path.stat().st_size}"
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:24]


def _load_npz_lookup(
    cache_path: Path | None,
    *,
    expected_dim: int,
) -> dict[str, np.ndarray]:
    if cache_path is None or not cache_path.is_file():
        return {}
    try:
        payload = np.load(cache_path, allow_pickle=True)
        paths = payload["paths"]
        features = np.asarray(payload["features"], dtype=np.float32)
    except (OSError, KeyError, ValueError):
        return {}
    if features.ndim != 2 or features.shape[1] != expected_dim:
        return {}
    lookup: dict[str, np.ndarray] = {}
    for raw_path, vector in zip(paths.tolist(), features):
        raw_text = str(raw_path)
        # Older V4 transition caches key a video/AU pair with a pipe, while
        # newer caches key only the video.  Accept both formats so V5 can
        # reuse the cache without silently recomputing every training clip.
        keys = [raw_text]
        if "|" in raw_text:
            keys.append(raw_text.split("|", 1)[0])
        normalized = np.nan_to_num(
            np.asarray(vector, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        for key in keys:
            lookup[_path_key(key)] = normalized
    return lookup


def _read_feature_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_feature_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def extract_drive_feature_vector(
    *,
    video_path: str | Path,
    au_path: str | Path,
    cache_dir: str | Path | None = None,
    transition_cache: str | Path | None = None,
    blendshape_cache: str | Path | None = None,
    include_blendshape: bool = False,
    compute_blendshape: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    video = Path(video_path).expanduser().resolve()
    au = Path(au_path).expanduser().resolve()
    transition_lookup = _load_npz_lookup(
        Path(transition_cache).expanduser().resolve()
        if transition_cache
        else None,
        expected_dim=len(TRANSITION_FEATURE_NAMES),
    )
    blendshape_lookup = _load_npz_lookup(
        Path(blendshape_cache).expanduser().resolve()
        if blendshape_cache
        else None,
        expected_dim=BLENDSHAPE_FEATURE_DIM,
    )
    cache_path = (
        Path(cache_dir).expanduser().resolve() / "drive_features.json"
        if cache_dir
        else None
    )
    cache = _read_feature_cache(cache_path) if cache_path else {}
    key = _path_key(video)
    feature_names = build_drive_feature_names(include_blendshape)
    cached = cache.get(key)
    if isinstance(cached, dict):
        vector = np.asarray(cached.get("vector", []), dtype=np.float32)
        if vector.shape == (len(feature_names),):
            return vector, {
                "status": str(cached.get("status", "cache")),
                "feature_cache": str(cache_path),
                "transition_source": cached.get("transition_source"),
                "blendshape_source": cached.get("blendshape_source"),
            }

    transition_source = "computed"
    try:
        transition = transition_lookup.get(key)
        if transition is None:
            transition_values = extract_transition_features(
                video_path=video,
                au_path=au,
            )
            transition = np.asarray(
                [
                    _finite(transition_values.get(name))
                    for name in TRANSITION_FEATURE_NAMES
                ],
                dtype=np.float32,
            )
        else:
            transition_source = "npz_cache"
    except Exception as exc:  # noqa: BLE001 - auxiliary branch must degrade
        transition = np.zeros(len(TRANSITION_FEATURE_NAMES), dtype=np.float32)
        transition_source = f"error:{type(exc).__name__}"

    blendshape = blendshape_lookup.get(key) if include_blendshape else None
    blendshape_source = "missing"
    if blendshape is not None:
        blendshape_source = "npz_cache"
    elif compute_blendshape:
        try:
            blendshape = np.asarray(
                blendshape_temporal_vector(video),
                dtype=np.float32,
            )
            blendshape_source = "computed"
        except Exception as exc:  # noqa: BLE001 - optional feature
            blendshape = None
            blendshape_source = f"error:{type(exc).__name__}"
    if blendshape is None or blendshape.shape != (BLENDSHAPE_FEATURE_DIM,):
        blendshape = np.zeros(BLENDSHAPE_FEATURE_DIM, dtype=np.float32)
        blendshape_missing = 1.0
    else:
        blendshape_missing = 0.0

    vector_parts = [np.asarray(transition, dtype=np.float32)]
    if include_blendshape:
        vector_parts.extend(
            [
                np.asarray(blendshape, dtype=np.float32),
                np.asarray([blendshape_missing], dtype=np.float32),
            ]
        )
    vector = np.nan_to_num(
        np.concatenate(vector_parts),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    status = "ok" if video.is_file() and au.is_file() else "missing_inputs"
    details = {
        "status": status,
        "transition_source": transition_source,
        "blendshape_source": blendshape_source,
        "blendshape_missing_mask": blendshape_missing
        if include_blendshape
        else None,
    }
    if cache_path:
        cache[key] = {
            "vector": vector.astype(float).tolist(),
            **details,
        }
        _write_feature_cache(cache_path, cache)
    return vector, details


def _iter_training_rows(manifest: dict[str, Any]) -> Iterable[dict[str, Any]]:
    pairs = manifest.get("pairs") or {}
    train = pairs.get("train") or {}
    for label_name, label in (("real", 0), ("fake", 1)):
        for item in train.get(label_name, []) or []:
            if isinstance(item, str):
                yield {
                    "video": item,
                    "label_generated": label,
                    "group_id": Path(item).parent.as_posix(),
                }
            elif isinstance(item, dict):
                row = dict(item)
                row.setdefault("label_generated", label)
                row.setdefault(
                    "group_id",
                    row.get("source_video")
                    or Path(str(row.get("video", ""))).parent.as_posix(),
                )
                yield row


def train_drive_head(
    *,
    manifest_path: str | Path,
    cache_dir: str | Path,
    model_path: str | Path,
    transition_cache: str | Path | None = None,
    blendshape_cache: str | Path | None = None,
    include_blendshape: bool = False,
    compute_blendshape: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    rows = list(_iter_training_rows(manifest))
    usable: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    for index, row in enumerate(rows, start=1):
        video = project_path(str(row.get("video", "")))
        au_value = row.get("au")
        au = project_path(str(au_value)) if au_value else None
        if au is None or not video.is_file() or not au.is_file():
            continue
        if "data\\test" in str(video).casefold() or "data/test" in str(video).casefold():
            continue
        vector, details = extract_drive_feature_vector(
            video_path=video,
            au_path=au,
            cache_dir=cache_dir,
            transition_cache=transition_cache,
            blendshape_cache=blendshape_cache,
            include_blendshape=include_blendshape,
            compute_blendshape=compute_blendshape,
        )
        usable.append(
            {
                "video": str(video),
                "au": str(au),
                "label_generated": int(row.get("label_generated", 0)),
                "group_id": str(row.get("group_id") or video.parent),
                "details": details,
            }
        )
        features.append(vector)
        labels.append(int(row.get("label_generated", 0)))
        groups.append(str(row.get("group_id") or video.parent))
        if index % 25 == 0:
            print(f"[v5 drive features] {index}/{len(rows)}", flush=True)

    if len(set(labels)) < 2 or len(features) < 8:
        raise RuntimeError(
            f"Insufficient V5 DriveHead rows: count={len(features)}, "
            f"labels={sorted(set(labels))}"
        )
    matrix = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    group_array = np.asarray(groups)
    splitter = GroupShuffleSplit(
        n_splits=20,
        test_size=0.20,
        random_state=seed,
    )
    fit_idx = val_idx = None
    for candidate_fit, candidate_val in splitter.split(matrix, y, group_array):
        if len(np.unique(y[candidate_fit])) == 2 and len(
            np.unique(y[candidate_val])
        ) == 2:
            fit_idx, val_idx = candidate_fit, candidate_val
            break
    if fit_idx is None or val_idx is None:
        raise RuntimeError("Could not create a grouped DriveHead validation split.")

    scaler = StandardScaler()
    x_fit = scaler.fit_transform(matrix[fit_idx])
    x_val = scaler.transform(matrix[val_idx])
    model = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    model.fit(x_fit, y[fit_idx])
    val_probability = model.predict_proba(x_val)[:, 1]
    best = None
    for step in range(20, 81):
        threshold = step / 100.0
        predicted = (val_probability >= threshold).astype(np.int32)
        tp = int(((y[val_idx] == 1) & (predicted == 1)).sum())
        tn = int(((y[val_idx] == 0) & (predicted == 0)).sum())
        fp = int(((y[val_idx] == 0) & (predicted == 1)).sum())
        fn = int(((y[val_idx] == 1) & (predicted == 0)).sum())
        ai_recall = tp / (tp + fn) if tp + fn else 0.0
        real_recall = tn / (tn + fp) if tn + fp else 0.0
        accuracy = (tp + tn) / max(len(val_idx), 1)
        candidate = (min(ai_recall, real_recall), accuracy, threshold)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    payload = {
        "schema_version": DRIVE_SCHEMA,
        "model_type": DRIVE_SCHEMA,
        "feature_names": list(build_drive_feature_names(include_blendshape)),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "coef": model.coef_.reshape(-1).astype(float).tolist(),
        "intercept": float(model.intercept_[0]),
        "threshold_generated": float(best[2]),
        "seed": int(seed),
        "train_count": int(len(fit_idx)),
        "validation_count": int(len(val_idx)),
        "train_group_count": int(len(set(group_array[fit_idx]))),
        "validation_group_count": int(len(set(group_array[val_idx]))),
        "train_label_counts": {
            "real": int((y[fit_idx] == 0).sum()),
            "generated": int((y[fit_idx] == 1).sum()),
        },
        "validation_metrics": {
            "min_class_recall": float(best[0]),
            "accuracy": float(best[1]),
        },
        "manifest": str(manifest_file),
        "test_sets_excluded": True,
        "include_blendshape": bool(include_blendshape),
        "rows": usable,
    }
    output = Path(model_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def load_drive_head(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    model_path = Path(path).expanduser()
    if not model_path.is_file():
        return None
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def predict_drive_head(
    *,
    vector: np.ndarray,
    model: dict[str, Any] | None,
) -> tuple[float | None, dict[str, Any]]:
    if not model:
        return None, {"status": "unavailable"}
    names = tuple(model.get("feature_names") or ())
    if len(names) != len(vector):
        return None, {"status": "feature_schema_mismatch"}
    mean = np.asarray(model.get("scaler_mean", []), dtype=np.float64)
    scale = np.maximum(
        np.asarray(model.get("scaler_scale", []), dtype=np.float64),
        1e-8,
    )
    coef = np.asarray(model.get("coef", []), dtype=np.float64)
    if mean.shape != vector.shape or scale.shape != vector.shape or coef.shape != vector.shape:
        return None, {"status": "model_shape_mismatch"}
    scaled = (np.asarray(vector, dtype=np.float64) - mean) / scale
    logit = float(np.dot(scaled, coef) + float(model.get("intercept", 0.0)))
    probability_generated = 1.0 / (
        1.0 + math.exp(-max(-40.0, min(40.0, logit)))
    )
    return float(1.0 - probability_generated), {
        "status": "ok",
        "p_drive_real": float(1.0 - probability_generated),
        "p_drive_generated": float(probability_generated),
    }
