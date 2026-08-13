"""Wang Xing dual-scale video .pt detector (24f@1k + 8f@2k).

Builds a holdout-aligned train/test split, extracts two normalized feature
views, concatenates them, and trains one supervised classifier checkpoint.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .real_video_detector import (
    FEATURE_VERSION,
    _classification_metrics,
    _fit_temperature,
    _predict_classifier_logits,
    _set_seed,
    _standardize,
    discover_videos,
    extract_video_feature,
    load_or_extract_features,
)

DUAL_MODEL_TYPE = "wangxing_dual_scale_real_fake_v1"
SCALE_A = {"name": "f24_s1024", "num_frames": 24, "frame_size": 1024}
SCALE_B = {"name": "f8_s2048", "num_frames": 8, "frame_size": 2048}


def _resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.expanduser().resolve()


def _video_key(path: Path) -> str:
    return str(path.resolve()).casefold()


def evenly_sample_real_videos(
    real_root: Path,
    *,
    target_count: int = 120,
    exclude: set[str] | None = None,
    seed: int = 42,
) -> list[Path]:
    """Average-select real clips across emotion/source folders."""
    exclude = exclude or set()
    all_videos = [
        path
        for path in discover_videos(real_root)
        if _video_key(path) not in exclude
    ]
    by_folder: dict[str, list[Path]] = defaultdict(list)
    for path in all_videos:
        by_folder[str(path.parent.resolve())].append(path)
    folders = sorted(by_folder)
    if not folders:
        return []
    rng = random.Random(seed)
    for folder in folders:
        by_folder[folder] = sorted(by_folder[folder], key=lambda p: p.name.lower())
        rng.shuffle(by_folder[folder])

    selected: list[Path] = []
    cursor = 0
    while len(selected) < target_count and any(by_folder.values()):
        folder = folders[cursor % len(folders)]
        bucket = by_folder[folder]
        if bucket:
            selected.append(bucket.pop(0))
        cursor += 1
        if cursor > target_count * max(len(folders), 1) * 4:
            break
    return selected[:target_count]


def build_wangxing_split_manifest(
    *,
    project_root: Path,
    real_root: Path,
    fake_root: Path,
    holdout_manifest: Path,
    real_train_count: int = 120,
    seed: int = 42,
) -> dict[str, Any]:
    """Train = 120 evenly sampled non-holdout reals + non-holdout fakes.

    Test = forensics holdout videos (50 real + 50 generated when present).
    """
    holdout = json.loads(holdout_manifest.read_text(encoding="utf-8-sig"))
    holdout_real = [
        _resolve(item["video"], project_root)
        for item in holdout.get("real", [])
        if isinstance(item, dict) and item.get("video")
    ]
    holdout_fake = [
        _resolve(item["video"], project_root)
        for item in holdout.get("seedance", [])
        if isinstance(item, dict) and item.get("video")
    ]
    holdout_real = [path for path in holdout_real if path.is_file()]
    holdout_fake = [path for path in holdout_fake if path.is_file()]
    exclude = {_video_key(path) for path in holdout_real + holdout_fake}

    train_real = evenly_sample_real_videos(
        real_root,
        target_count=real_train_count,
        exclude=exclude,
        seed=seed,
    )
    all_fake = discover_videos(fake_root)
    train_fake = [path for path in all_fake if _video_key(path) not in exclude]

    manifest = {
        "schema_version": "wangxing_dual_pt_split_v1",
        "seed": seed,
        "scales": [SCALE_A, SCALE_B],
        "protocol": {
            "real_train_sampling": "even_by_parent_folder",
            "real_train_count_target": real_train_count,
            "test_source": str(holdout_manifest),
            "holdout_excluded_from_train": True,
            "normalization": "train-only mean/std per concatenated dual feature",
        },
        "counts": {
            "train_real": len(train_real),
            "train_fake": len(train_fake),
            "test_real": len(holdout_real),
            "test_fake": len(holdout_fake),
        },
        "train": {
            "real": [str(path) for path in train_real],
            "fake": [str(path) for path in train_fake],
        },
        "test": {
            "real": [str(path) for path in holdout_real],
            "fake": [str(path) for path in holdout_fake],
        },
    }
    return manifest


class DualScaleClassifier(nn.Module):
    """MLP over concatenated dual-scale normalized video features."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        hidden = max(128, min(512, int(input_dim) // 4))
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(max(hidden // 2, 1)),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).squeeze(1)


def _feature_dim(num_frames: int) -> int:
    from .real_video_detector import FRAME_FEATURE_DIM, TEMPORAL_FEATURE_DIM

    return int(num_frames) * FRAME_FEATURE_DIM + max(int(num_frames) - 1, 1) * TEMPORAL_FEATURE_DIM


def extract_dual_feature(
    video_path: Path,
    *,
    scale_a: dict[str, Any] = SCALE_A,
    scale_b: dict[str, Any] = SCALE_B,
) -> np.ndarray:
    feat_a = extract_video_feature(
        video_path,
        num_frames=int(scale_a["num_frames"]),
        frame_size=int(scale_a["frame_size"]),
    )
    feat_b = extract_video_feature(
        video_path,
        num_frames=int(scale_b["num_frames"]),
        frame_size=int(scale_b["frame_size"]),
    )
    return np.concatenate([feat_a, feat_b], axis=0).astype(np.float32)


def build_dual_feature_table(
    paths: list[Path],
    *,
    cache_path: Path,
    scale_a: dict[str, Any] = SCALE_A,
    scale_b: dict[str, Any] = SCALE_B,
) -> tuple[np.ndarray, list[Path], list[dict[str, str]]]:
    """Extract or load dual-scale concatenated features for paths."""
    cache_path = Path(cache_path)
    if cache_path.is_file():
        payload = np.load(str(cache_path), allow_pickle=True)
        cached_paths = [str(Path(item).resolve()) for item in payload["paths"].tolist()]
        wanted = [str(path.resolve()) for path in paths]
        if (
            list(payload.get("scale_a_name", [scale_a["name"]]))[0] == scale_a["name"]
            and list(payload.get("scale_b_name", [scale_b["name"]]))[0] == scale_b["name"]
            and cached_paths == wanted
        ):
            features = np.asarray(payload["features"], dtype=np.float32)
            valid = [Path(item) for item in cached_paths]
            return features, valid, []

    # Build each scale via existing cache helper, then concat in path order.
    cache_a = cache_path.with_name(cache_path.stem + f"__{scale_a['name']}.npz")
    cache_b = cache_path.with_name(cache_path.stem + f"__{scale_b['name']}.npz")
    feats_a, valid_a, err_a = load_or_extract_features(
        paths,
        cache_path=cache_a,
        num_frames=int(scale_a["num_frames"]),
        frame_size=int(scale_a["frame_size"]),
    )
    feats_b, valid_b, err_b = load_or_extract_features(
        paths,
        cache_path=cache_b,
        num_frames=int(scale_b["num_frames"]),
        frame_size=int(scale_b["frame_size"]),
    )
    map_a = {str(path.resolve()): feat for path, feat in zip(valid_a, feats_a)}
    map_b = {str(path.resolve()): feat for path, feat in zip(valid_b, feats_b)}
    features: list[np.ndarray] = []
    valid: list[Path] = []
    errors: list[str] = [str(item) for item in err_a] + [str(item) for item in err_b]
    for path in paths:
        key = str(path.resolve())
        left = map_a.get(key)
        right = map_b.get(key)
        if left is None or right is None:
            errors.append(f"{key}: missing_one_scale")
            continue
        if left.shape[0] != _feature_dim(int(scale_a["num_frames"])):
            errors.append(f"{key}: scale_a_dim_mismatch")
            continue
        if right.shape[0] != _feature_dim(int(scale_b["num_frames"])):
            errors.append(f"{key}: scale_b_dim_mismatch")
            continue
        features.append(np.concatenate([left, right], axis=0).astype(np.float32))
        valid.append(path)

    matrix = (
        np.stack(features).astype(np.float32)
        if features
        else np.zeros((0, _feature_dim(24) + _feature_dim(8)), dtype=np.float32)
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(cache_path),
        features=matrix,
        paths=np.asarray([str(path.resolve()) for path in valid], dtype=object),
        scale_a_name=np.asarray([scale_a["name"]]),
        scale_b_name=np.asarray([scale_b["name"]]),
        feature_version=np.asarray([FEATURE_VERSION]),
    )
    return matrix, valid, errors


def _collect_matrix(
    paths: list[Path],
    label: int,
    feature_map: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    missing: list[str] = []
    for path in paths:
        key = str(path.resolve())
        value = feature_map.get(key)
        if value is None:
            missing.append(key)
            continue
        rows.append(value)
        labels.append(label)
    if not rows:
        return (
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            missing,
        )
    return (
        np.stack(rows).astype(np.float32),
        np.asarray(labels, dtype=np.float32),
        missing,
    )


def train_wangxing_dual_pt(
    *,
    manifest: dict[str, Any],
    cache_dir: Path,
    model_path: Path,
    epochs: int = 80,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    seed: int = 42,
) -> dict[str, Any]:
    _set_seed(seed)
    train_real = [Path(item) for item in manifest["train"]["real"]]
    train_fake = [Path(item) for item in manifest["train"]["fake"]]
    test_real = [Path(item) for item in manifest["test"]["real"]]
    test_fake = [Path(item) for item in manifest["test"]["fake"]]
    all_paths = train_real + train_fake + test_real + test_fake

    dual_cache = Path(cache_dir) / "wangxing_dual_f24s1024_f8s2048.npz"
    features, valid, errors = build_dual_feature_table(
        all_paths,
        cache_path=dual_cache,
    )
    feature_map = {
        str(path.resolve()): features[index]
        for index, path in enumerate(valid)
    }

    x_train_real, y_train_real, miss_tr = _collect_matrix(train_real, 0, feature_map)
    x_train_fake, y_train_fake, miss_tf = _collect_matrix(train_fake, 1, feature_map)
    x_test_real, y_test_real, miss_er = _collect_matrix(test_real, 0, feature_map)
    x_test_fake, y_test_fake, miss_ef = _collect_matrix(test_fake, 1, feature_map)

    x_train = np.concatenate([x_train_real, x_train_fake], axis=0)
    y_train = np.concatenate([y_train_real, y_train_fake], axis=0)
    x_test = np.concatenate([x_test_real, x_test_fake], axis=0)
    y_test = np.concatenate([y_test_real, y_test_fake], axis=0)
    if len(x_train) < 8 or len(np.unique(y_train)) < 2:
        raise RuntimeError("训练集不足：需要同时包含真/假样本")
    if len(x_test) < 4 or len(np.unique(y_test)) < 2:
        raise RuntimeError("测试集不足：需要同时包含真/假样本")

    train_norm, mean, scale = _standardize(x_train, x_train)
    train_norm = np.clip(train_norm, -8.0, 8.0).astype(np.float32)
    test_norm = np.clip((x_test - mean) / scale, -8.0, 8.0).astype(np.float32)

    # Hold out 15% of train as validation for early selection / temperature.
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_train))
    rng.shuffle(indices)
    val_count = max(4, int(round(0.15 * len(indices))))
    # Keep both classes in val when possible.
    val_idx = indices[:val_count]
    fit_idx = indices[val_count:]
    if len(np.unique(y_train[fit_idx])) < 2 or len(np.unique(y_train[val_idx])) < 2:
        fit_idx = indices
        val_idx = indices[: max(4, len(indices) // 5)]

    x_fit, y_fit = train_norm[fit_idx], y_train[fit_idx]
    x_val, y_val = train_norm[val_idx], y_train[val_idx]

    model = DualScaleClassifier(input_dim=int(train_norm.shape[1]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=2e-4,
    )
    class_counts = np.bincount(y_fit.astype(np.int64), minlength=2)
    sample_weights = np.asarray(
        [1.0 / max(float(class_counts[int(label)]), 1.0) for label in y_fit],
        dtype=np.float64,
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit)),
        batch_size=max(1, int(batch_size)),
        sampler=WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        ),
    )
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_key = (-1.0, float("inf"))
    history: list[dict[str, Any]] = []

    for epoch in range(max(1, int(epochs))):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad()
                continue
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        val_logits = _predict_classifier_logits(model, x_val)
        val_prob = 1.0 / (1.0 + np.exp(-val_logits))
        val_metrics = _classification_metrics(y_val.astype(np.int64), val_prob)
        val_loss = float(
            criterion(
                torch.from_numpy(val_logits),
                torch.from_numpy(y_val),
            ).item()
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else math.inf,
                "validation_loss": val_loss,
                **val_metrics,
            }
        )
        key = (val_metrics["balanced_accuracy"], -val_loss)
        if key > best_key:
            best_key = key
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    val_logits = _predict_classifier_logits(model, x_val)
    temperature = _fit_temperature(val_logits, y_val)
    test_logits = _predict_classifier_logits(model, test_norm)
    test_prob_gen = 1.0 / (1.0 + np.exp(-test_logits / max(temperature, 1e-6)))
    # Align with forensics headline: generated recall + overall accuracy.
    pred_gen = (test_prob_gen >= 0.5).astype(np.int64)
    y_true = y_test.astype(np.int64)
    tp = int(((y_true == 1) & (pred_gen == 1)).sum())
    tn = int(((y_true == 0) & (pred_gen == 0)).sum())
    fp = int(((y_true == 0) & (pred_gen == 1)).sum())
    fn = int(((y_true == 1) & (pred_gen == 0)).sum())
    headline = {
        "generated_recall": tp / (tp + fn) if tp + fn else None,
        "overall_accuracy": (tp + tn) / len(y_true) if len(y_true) else None,
        "generated_precision": tp / (tp + fp) if tp + fp else None,
        "real_recall": tn / (tn + fp) if tn + fp else None,
        "coverage": 1.0,
    }
    test_metrics = _classification_metrics(y_true, test_prob_gen)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": DUAL_MODEL_TYPE,
        "feature_version": FEATURE_VERSION,
        "config": {
            "scales": [SCALE_A, SCALE_B],
            "input_dim": int(train_norm.shape[1]),
            "threshold_generated": 0.5,
            "probability_target": "generated",
        },
        "model_state": model.state_dict(),
        "feature_mean": mean.astype(np.float32),
        "feature_scale": scale.astype(np.float32),
        "temperature": float(temperature),
        "dataset": {
            "train_real": len(y_train_real),
            "train_fake": len(y_train_fake),
            "test_real": len(y_test_real),
            "test_fake": len(y_test_fake),
            "missing": {
                "train_real": miss_tr,
                "train_fake": miss_tf,
                "test_real": miss_er,
                "test_fake": miss_ef,
            },
            "extract_errors_preview": errors[:20],
        },
        "train_val_metrics_tail": history[-10:],
        "test_headline": headline,
        "test_metrics": test_metrics,
        "confusion": {
            "tp_generated": tp,
            "tn_real": tn,
            "fp_real_as_generated": fp,
            "fn_generated_as_real": fn,
        },
    }
    torch.save(checkpoint, model_path)
    return {
        "model_path": str(model_path),
        "cache_path": str(dual_cache),
        "headline": headline,
        "confusion": checkpoint["confusion"],
        "counts": checkpoint["dataset"],
        "temperature": float(temperature),
        "epochs_completed": len(history),
        "best_validation_balanced_accuracy": float(best_key[0]),
    }


def predict_wangxing_dual_pt(
    video_path: str | Path,
    model_path: str | Path,
) -> dict[str, Any]:
    video_path = Path(video_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    checkpoint = torch.load(str(model_path), map_location="cpu")
    if checkpoint.get("model_type") != DUAL_MODEL_TYPE:
        raise ValueError(f"Unsupported model_type: {checkpoint.get('model_type')}")
    scales = checkpoint["config"]["scales"]
    feature = extract_dual_feature(
        video_path,
        scale_a=scales[0],
        scale_b=scales[1],
    )
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
    normalized = np.clip((feature - mean) / np.maximum(scale, 1e-4), -8.0, 8.0)
    model = DualScaleClassifier(input_dim=int(checkpoint["config"]["input_dim"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    logit = float(
        _predict_classifier_logits(model, normalized[None, :])[0]
    )
    temperature = float(checkpoint.get("temperature", 1.0))
    p_gen = float(1.0 / (1.0 + math.exp(-logit / max(temperature, 1e-6))))
    p_gen = min(0.98, max(0.02, p_gen))
    decision = "generated" if p_gen >= 0.5 else "real"
    return {
        "prediction": decision,
        "generated_probability": round(p_gen, 4),
        "real_probability": round(1.0 - p_gen, 4),
        "logit": logit,
        "temperature": temperature,
        "model_path": str(model_path),
        "video_path": str(video_path),
    }
