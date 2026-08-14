"""Single-scale video frame-count ablation (8 / 24 / 32) for Wang Xing holdout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from evaluator.vedio_pred.real_video_detector import (
    _classification_metrics,
    _fit_temperature,
    _predict_classifier_logits,
    _set_seed,
    _standardize,
    load_or_extract_features,
)
from evaluator.vedio_pred.wangxing_dual_pt import DualScaleClassifier


def _collect(
    paths: list[Path],
    label: int,
    feature_map: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for path in paths:
        value = feature_map.get(str(path.resolve()))
        if value is None:
            continue
        rows.append(value)
        labels.append(label)
    if not rows:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(rows).astype(np.float32), np.asarray(labels, dtype=np.float32)


def train_and_eval_single_scale(
    *,
    manifest: dict[str, Any],
    num_frames: int,
    frame_size: int,
    cache_dir: Path,
    model_dir: Path,
    epochs: int = 80,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    seed: int = 42,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Train one single-scale video classifier and score holdout."""
    model_dir = Path(model_dir)
    model_path = model_dir / f"wangxing_single_f{num_frames}_s{frame_size}.pt"
    cache_path = Path(cache_dir) / f"wangxing_single_f{num_frames}_s{frame_size}.npz"
    if skip_existing and model_path.is_file():
        try:
            ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(model_path), map_location="cpu")
        headline = ckpt.get("test_headline") or {}
        confusion = ckpt.get("confusion") or {}
        if headline.get("generated_recall") is not None:
            print(
                f"\n=== skip existing frames={num_frames} size={frame_size} "
                f"gen_recall={headline.get('generated_recall')} "
                f"acc={headline.get('overall_accuracy')} ===",
                flush=True,
            )
            return {
                "num_frames": int(num_frames),
                "frame_size": int(frame_size),
                "model_path": str(model_path),
                "cache_path": str(cache_path),
                "headline": headline,
                "confusion": confusion,
                "counts": {},
                "temperature": float(ckpt.get("temperature", 1.0)),
                "skipped_existing": True,
            }

    _set_seed(seed + int(num_frames) * 1000 + int(frame_size))
    train_real = [Path(p) for p in manifest["train"]["real"]]
    train_fake = [Path(p) for p in manifest["train"]["fake"]]
    test_real = [Path(p) for p in manifest["test"]["real"]]
    test_fake = [Path(p) for p in manifest["test"]["fake"]]
    all_paths = train_real + train_fake + test_real + test_fake

    print(
        f"\n=== frames={num_frames} size={frame_size} "
        f"extract/cache {cache_path.name} ===",
        flush=True,
    )
    features, valid, errors = load_or_extract_features(
        all_paths,
        cache_path=cache_path,
        num_frames=int(num_frames),
        frame_size=int(frame_size),
    )
    feature_map = {
        str(path.resolve()): features[index]
        for index, path in enumerate(valid)
    }

    x_tr, y_tr = _collect(train_real, 0, feature_map)
    x_tf, y_tf = _collect(train_fake, 1, feature_map)
    x_er, y_er = _collect(test_real, 0, feature_map)
    x_ef, y_ef = _collect(test_fake, 1, feature_map)
    x_train = np.concatenate([x_tr, x_tf], axis=0)
    y_train = np.concatenate([y_tr, y_tf], axis=0)
    x_test = np.concatenate([x_er, x_ef], axis=0)
    y_test = np.concatenate([y_er, y_ef], axis=0)
    if len(x_train) < 8 or len(np.unique(y_train)) < 2:
        raise RuntimeError(f"train too small for frames={num_frames}")
    if len(x_test) < 4 or len(np.unique(y_test)) < 2:
        raise RuntimeError(f"test too small for frames={num_frames}")

    train_norm, mean, scale = _standardize(x_train, x_train)
    train_norm = np.clip(train_norm, -8.0, 8.0).astype(np.float32)
    test_norm = np.clip((x_test - mean) / scale, -8.0, 8.0).astype(np.float32)

    rng = np.random.default_rng(seed + int(num_frames) * 1000 + int(frame_size))
    indices = np.arange(len(y_train))
    rng.shuffle(indices)
    val_count = max(4, int(round(0.15 * len(indices))))
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
    for epoch in range(max(1, int(epochs))):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad):
                optimizer.zero_grad()
                continue
            optimizer.step()
        val_logits = _predict_classifier_logits(model, x_val)
        val_prob = 1.0 / (1.0 + np.exp(-val_logits))
        val_metrics = _classification_metrics(y_val.astype(np.int64), val_prob)
        val_loss = float(
            criterion(
                torch.from_numpy(val_logits),
                torch.from_numpy(y_val),
            ).item()
        )
        key = (val_metrics["balanced_accuracy"], -val_loss)
        if key > best_key:
            best_key = key
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"  epoch {epoch + 1}/{epochs} "
                f"val_bal_acc={val_metrics['balanced_accuracy']:.3f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    val_logits = _predict_classifier_logits(model, x_val)
    temperature = _fit_temperature(val_logits, y_val)
    test_logits = _predict_classifier_logits(model, test_norm)
    test_prob_gen = 1.0 / (1.0 + np.exp(-test_logits / max(temperature, 1e-6)))
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
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "wangxing_single_scale_real_fake_v1",
            "config": {
                "num_frames": int(num_frames),
                "frame_size": int(frame_size),
                "input_dim": int(train_norm.shape[1]),
            },
            "model_state": model.state_dict(),
            "feature_mean": mean.astype(np.float32),
            "feature_scale": scale.astype(np.float32),
            "temperature": float(temperature),
            "test_headline": headline,
            "confusion": {
                "tp_generated": tp,
                "tn_real": tn,
                "fp_real_as_generated": fp,
                "fn_generated_as_real": fn,
            },
        },
        model_path,
    )
    print(
        f"  headline frames={num_frames}: "
        f"gen_recall={headline['generated_recall']} "
        f"acc={headline['overall_accuracy']}",
        flush=True,
    )
    return {
        "num_frames": int(num_frames),
        "frame_size": int(frame_size),
        "model_path": str(model_path),
        "cache_path": str(cache_path),
        "headline": headline,
        "confusion": {
            "tp_generated": tp,
            "tn_real": tn,
            "fp_real_as_generated": fp,
            "fn_generated_as_real": fn,
        },
        "counts": {
            "train_real": int(len(y_tr)),
            "train_fake": int(len(y_tf)),
            "test_real": int(len(y_er)),
            "test_fake": int(len(y_ef)),
            "extract_errors": len(errors),
        },
        "temperature": float(temperature),
    }
