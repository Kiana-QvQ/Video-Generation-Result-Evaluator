"""Wang Xing multi-scale video .pt: concat 8/24/32 x 1k/2k into one classifier."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from evaluator.vedio_pred.real_video_detector import (
    FEATURE_VERSION,
    _classification_metrics,
    _fit_temperature,
    _predict_classifier_logits,
    _set_seed,
    _standardize,
    extract_video_feature,
    load_or_extract_features,
)
from evaluator.vedio_pred.wangxing_dual_pt import DualScaleClassifier, _feature_dim

MULTI_MODEL_TYPE = "wangxing_multi_scale_real_fake_v1"

DEFAULT_SCALES: list[dict[str, Any]] = [
    {"name": "f8_s1024", "num_frames": 8, "frame_size": 1024},
    {"name": "f24_s1024", "num_frames": 24, "frame_size": 1024},
    {"name": "f32_s1024", "num_frames": 32, "frame_size": 1024},
    {"name": "f8_s2048", "num_frames": 8, "frame_size": 2048},
    {"name": "f24_s2048", "num_frames": 24, "frame_size": 2048},
    {"name": "f32_s2048", "num_frames": 32, "frame_size": 2048},
]


def _scale_cache_name(scale: dict[str, Any]) -> str:
    return f"wangxing_single_f{int(scale['num_frames'])}_s{int(scale['frame_size'])}.npz"


def _expected_dim(scales: list[dict[str, Any]]) -> int:
    return int(sum(_feature_dim(int(scale["num_frames"])) for scale in scales))


def extract_multi_feature(
    video_path: Path,
    *,
    scales: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    scales = list(scales or DEFAULT_SCALES)
    parts = [
        extract_video_feature(
            video_path,
            num_frames=int(scale["num_frames"]),
            frame_size=int(scale["frame_size"]),
        )
        for scale in scales
    ]
    return np.concatenate(parts, axis=0).astype(np.float32)


def build_multi_feature_table(
    paths: list[Path],
    *,
    cache_dir: Path,
    concat_cache_path: Path,
    scales: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, list[Path], list[str]]:
    """Load/extract each scale (reuse single-scale caches) then concat."""
    scales = list(scales or DEFAULT_SCALES)
    concat_cache_path = Path(concat_cache_path)
    cache_dir = Path(cache_dir)
    scale_names = [str(scale["name"]) for scale in scales]

    if concat_cache_path.is_file():
        payload = np.load(str(concat_cache_path), allow_pickle=True)
        cached_paths = [str(Path(item).resolve()) for item in payload["paths"].tolist()]
        wanted = [str(path.resolve()) for path in paths]
        cached_names = [str(item) for item in payload.get("scale_names", []).tolist()]
        if cached_paths == wanted and cached_names == scale_names:
            return (
                np.asarray(payload["features"], dtype=np.float32),
                [Path(item) for item in cached_paths],
                [],
            )

    scale_maps: list[dict[str, np.ndarray]] = []
    errors: list[str] = []
    for scale in scales:
        single_cache = cache_dir / _scale_cache_name(scale)
        print(
            f"[multi] scale={scale['name']} cache={single_cache.name}",
            flush=True,
        )
        feats, valid, err = load_or_extract_features(
            paths,
            cache_path=single_cache,
            num_frames=int(scale["num_frames"]),
            frame_size=int(scale["frame_size"]),
        )
        errors.extend(str(item) for item in err)
        scale_maps.append(
            {str(path.resolve()): feat for path, feat in zip(valid, feats)}
        )

    features: list[np.ndarray] = []
    valid_paths: list[Path] = []
    expected = [_feature_dim(int(scale["num_frames"])) for scale in scales]
    for path in paths:
        key = str(path.resolve())
        parts: list[np.ndarray] = []
        ok = True
        for index, scale_map in enumerate(scale_maps):
            part = scale_map.get(key)
            if part is None:
                errors.append(f"{key}: missing_scale_{scales[index]['name']}")
                ok = False
                break
            if part.shape[0] != expected[index]:
                errors.append(f"{key}: dim_mismatch_{scales[index]['name']}")
                ok = False
                break
            parts.append(part)
        if not ok:
            continue
        features.append(np.concatenate(parts, axis=0).astype(np.float32))
        valid_paths.append(path)

    matrix = (
        np.stack(features).astype(np.float32)
        if features
        else np.zeros((0, _expected_dim(scales)), dtype=np.float32)
    )
    concat_cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(concat_cache_path),
        features=matrix,
        paths=np.asarray([str(path.resolve()) for path in valid_paths], dtype=object),
        scale_names=np.asarray(scale_names, dtype=object),
        feature_version=np.asarray([FEATURE_VERSION]),
    )
    return matrix, valid_paths, errors


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


def train_wangxing_multi_scale_pt(
    *,
    manifest: dict[str, Any],
    cache_dir: Path,
    model_path: Path,
    scales: list[dict[str, Any]] | None = None,
    epochs: int = 80,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    seed: int = 42,
) -> dict[str, Any]:
    scales = list(scales or DEFAULT_SCALES)
    _set_seed(seed)
    train_real = [Path(item) for item in manifest["train"]["real"]]
    train_fake = [Path(item) for item in manifest["train"]["fake"]]
    test_real = [Path(item) for item in manifest["test"]["real"]]
    test_fake = [Path(item) for item in manifest["test"]["fake"]]
    all_paths = train_real + train_fake + test_real + test_fake

    cache_dir = Path(cache_dir)
    concat_cache = cache_dir / "wangxing_multi_f8_24_32_s1024_2048.npz"
    features, valid, errors = build_multi_feature_table(
        all_paths,
        cache_dir=cache_dir,
        concat_cache_path=concat_cache,
        scales=scales,
    )
    feature_map = {
        str(path.resolve()): features[index]
        for index, path in enumerate(valid)
    }

    x_tr, y_tr, miss_tr = _collect_matrix(train_real, 0, feature_map)
    x_tf, y_tf, miss_tf = _collect_matrix(train_fake, 1, feature_map)
    x_er, y_er, miss_er = _collect_matrix(test_real, 0, feature_map)
    x_ef, y_ef, miss_ef = _collect_matrix(test_fake, 1, feature_map)
    x_train = np.concatenate([x_tr, x_tf], axis=0)
    y_train = np.concatenate([y_tr, y_tf], axis=0)
    x_test = np.concatenate([x_er, x_ef], axis=0)
    y_test = np.concatenate([y_er, y_ef], axis=0)
    if len(x_train) < 8 or len(np.unique(y_train)) < 2:
        raise RuntimeError("训练集不足：需要同时包含真/假样本")
    if len(x_test) < 4 or len(np.unique(y_test)) < 2:
        raise RuntimeError("测试集不足：需要同时包含真/假样本")

    train_norm, mean, scale = _standardize(x_train, x_train)
    train_norm = np.clip(train_norm, -8.0, 8.0).astype(np.float32)
    test_norm = np.clip((x_test - mean) / scale, -8.0, 8.0).astype(np.float32)

    rng = np.random.default_rng(seed)
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
    criterion = torch.nn.BCEWithLogitsLoss()
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

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": MULTI_MODEL_TYPE,
        "feature_version": FEATURE_VERSION,
        "config": {
            "scales": scales,
            "input_dim": int(train_norm.shape[1]),
            "threshold_generated": 0.5,
            "probability_target": "generated",
        },
        "model_state": model.state_dict(),
        "feature_mean": mean.astype(np.float32),
        "feature_scale": scale.astype(np.float32),
        "temperature": float(temperature),
        "dataset": {
            "train_real": len(y_tr),
            "train_fake": len(y_tf),
            "test_real": len(y_er),
            "test_fake": len(y_ef),
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
        "confusion": {
            "tp_generated": tp,
            "tn_real": tn,
            "fp_real_as_generated": fp,
            "fn_generated_as_real": fn,
        },
    }
    torch.save(checkpoint, model_path)
    print(
        f"headline multi-scale: gen_recall={headline['generated_recall']} "
        f"acc={headline['overall_accuracy']}",
        flush=True,
    )
    return {
        "model_path": str(model_path),
        "cache_path": str(concat_cache),
        "scales": scales,
        "headline": headline,
        "confusion": checkpoint["confusion"],
        "counts": checkpoint["dataset"],
        "temperature": float(temperature),
        "epochs_completed": len(history),
        "best_validation_balanced_accuracy": float(best_key[0]),
        "input_dim": int(train_norm.shape[1]),
    }


def predict_wangxing_multi_scale_pt(
    video_path: str | Path,
    model_path: str | Path,
) -> dict[str, Any]:
    video_path = Path(video_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    try:
        checkpoint = torch.load(
            str(model_path),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(str(model_path), map_location="cpu")
    if checkpoint.get("model_type") != MULTI_MODEL_TYPE:
        raise ValueError(f"Unsupported model_type: {checkpoint.get('model_type')}")
    scales = checkpoint["config"]["scales"]
    feature = extract_multi_feature(video_path, scales=scales)
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
    normalized = np.clip((feature - mean) / np.maximum(scale, 1e-4), -8.0, 8.0)
    model = DualScaleClassifier(input_dim=int(checkpoint["config"]["input_dim"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    logit = float(_predict_classifier_logits(model, normalized[None, :])[0])
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
        "scales": scales,
    }
