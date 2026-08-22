"""Expression-only PT v4.1 model.

This candidate intentionally excludes full-frame RGB, source-domain profile
distances, MUSIQ, and texture/frequency features from the primary classifier.
It trains only on real versus Seedance labels and uses AU/landmark sequences
plus Blendshape motion descriptors.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from evaluator.modules.core.paths import project_path
from evaluator.vedio_pred.real_video_detector import (
    _classification_metrics,
    _fit_temperature,
    _set_seed,
)
from wangxing_project.blendshape_temporal import (
    BLENDSHAPE_FEATURE_DIM,
    blendshape_temporal_vector,
)
from wangxing_project.expression_sequence import (
    SEQUENCE_FRAME_DIM,
    SEQUENCE_MAX_FRAMES,
    SEQUENCE_SUMMARY_DIM,
    extract_expression_sequence_features,
)
from wangxing_project.joint_au_pt import (
    is_forbidden_train_video,
    resolve_torch_device,
)
from wangxing_project.joint_au_pt_v3 import (
    _group_split,
)

V41_MODEL_TYPE = "wangxing_expression_authenticity_v41"
SEQUENCE_HIDDEN = 64
SUMMARY_HIDDEN = 32
BLENDSHAPE_HIDDEN = 48
FUSION_HIDDEN = 96


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.5
    for threshold in np.linspace(0.10, 0.90, 161):
        predictions = probabilities >= threshold
        tp = int(((labels == 1) & predictions).sum())
        tn = int(((labels == 0) & ~predictions).sum())
        fp = int(((labels == 0) & predictions).sum())
        fn = int(((labels == 1) & ~predictions).sum())
        generated_recall = tp / (tp + fn) if tp + fn else 0.0
        real_recall = tn / (tn + fp) if tn + fp else 0.0
        accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
        key = (
            min(generated_recall, real_recall),
            accuracy,
            -abs(float(threshold) - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _headline_with_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    total_count: int,
    threshold: float,
) -> tuple[dict[str, float | None], dict[str, int]]:
    labels = labels.astype(np.int64)
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return (
        {
            "generated_recall": tp / (tp + fn) if tp + fn else None,
            "overall_accuracy": (
                (tp + tn) / len(labels) if len(labels) else None
            ),
            "generated_precision": tp / (tp + fp) if tp + fp else None,
            "real_recall": tn / (tn + fp) if tn + fp else None,
            "coverage": len(labels) / total_count if total_count else 0.0,
            "decision_threshold": float(threshold),
        },
        {
            "tp_generated": tp,
            "tn_real": tn,
            "fp_real_as_generated": fp,
            "fn_generated_as_real": fn,
        },
    )


def _focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 1.5,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    probability = torch.sigmoid(logits)
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    return (((1.0 - pt) ** gamma) * bce).mean()


class ExpressionSequenceClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequence_encoder = nn.GRU(
            input_size=SEQUENCE_FRAME_DIM,
            hidden_size=SEQUENCE_HIDDEN,
            batch_first=True,
            bidirectional=True,
        )
        self.sequence_attention = nn.Linear(SEQUENCE_HIDDEN * 2, 1)
        self.summary_encoder = nn.Sequential(
            nn.Linear(SEQUENCE_SUMMARY_DIM, SUMMARY_HIDDEN),
            nn.LayerNorm(SUMMARY_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.blendshape_encoder = nn.Sequential(
            nn.Linear(BLENDSHAPE_FEATURE_DIM, BLENDSHAPE_HIDDEN),
            nn.LayerNorm(BLENDSHAPE_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.fusion = nn.Sequential(
            nn.Linear(SEQUENCE_HIDDEN * 2 + SUMMARY_HIDDEN + BLENDSHAPE_HIDDEN, FUSION_HIDDEN),
            nn.LayerNorm(FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(FUSION_HIDDEN, 1),
        )
        self.sequence_head = nn.Linear(SEQUENCE_HIDDEN * 2, 1)
        self.blendshape_head = nn.Linear(BLENDSHAPE_HIDDEN, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        summary: torch.Tensor,
        blendshape: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, _ = self.sequence_encoder(sequence)
        attention = torch.softmax(self.sequence_attention(encoded), dim=1)
        sequence_embedding = torch.sum(encoded * attention, dim=1)
        summary_embedding = self.summary_encoder(summary)
        blendshape_embedding = self.blendshape_encoder(blendshape)
        joint = self.fusion(
            torch.cat(
                [sequence_embedding, summary_embedding, blendshape_embedding],
                dim=1,
            )
        ).squeeze(1)
        if not return_aux:
            return joint
        return (
            joint,
            self.sequence_head(sequence_embedding).squeeze(1),
            self.blendshape_head(blendshape_embedding).squeeze(1),
        )


def _standardize(
    values: np.ndarray,
    fit_idx: np.ndarray,
    *,
    sequence: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    fit_values = values[fit_idx]
    if sequence:
        mean = fit_values.mean(axis=(0, 1))
        scale = fit_values.std(axis=(0, 1))
    else:
        mean = fit_values.mean(axis=0)
        scale = fit_values.std(axis=0)
    return (
        mean.astype(np.float32),
        np.maximum(scale, 1e-4).astype(np.float32),
    )


def _normalize(
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return np.clip(
        (values - mean) / np.maximum(scale, 1e-4),
        -8.0,
        8.0,
    ).astype(np.float32)


def _prepare_expression_data(manifest: dict[str, Any]) -> dict[str, Any]:
    if "pairs" not in manifest:
        raise ValueError("v4.1 manifest must contain explicit pairs.")

    def collect(
        pairs: list[dict[str, Any]],
        label: int,
        *,
        filter_forbidden: bool,
    ) -> dict[str, Any]:
        videos: list[str] = []
        aus: list[str] = []
        labels: list[int] = []
        groups: list[str] = []
        base_labels: list[int] = []
        missing: list[str] = []
        for item in pairs:
            video = project_path(str(item["video"]))
            au = project_path(str(item["au"]))
            if filter_forbidden and is_forbidden_train_video(video):
                raise ValueError(f"Forbidden Change clip in training: {video}")
            if not video.is_file() or not au.is_file():
                missing.append(str(video))
                continue
            videos.append(str(video.resolve()))
            aus.append(str(au.resolve()))
            labels.append(label)
            groups.append(str(item.get("group_id") or video.resolve()))
            base_labels.append(int(item.get("base_label", label)))
        return {
            "videos": videos,
            "aus": aus,
            "labels": np.asarray(labels, dtype=np.int64),
            "groups": groups,
            "base_labels": np.asarray(base_labels, dtype=np.int64),
            "missing": missing,
        }

    train_real = collect(
        list(manifest["pairs"]["train"]["real"]),
        0,
        filter_forbidden=True,
    )
    train_fake = collect(
        list(manifest["pairs"]["train"]["fake"]),
        1,
        filter_forbidden=True,
    )
    test_real = collect(
        list(manifest["pairs"]["test"]["real"]),
        0,
        filter_forbidden=False,
    )
    test_fake = collect(
        list(manifest["pairs"]["test"]["fake"]),
        1,
        filter_forbidden=False,
    )
    return {
        "train_videos": train_real["videos"] + train_fake["videos"],
        "train_aus": train_real["aus"] + train_fake["aus"],
        "train_labels": np.concatenate(
            [train_real["labels"], train_fake["labels"]]
        ),
        "train_groups": train_real["groups"] + train_fake["groups"],
        "train_base_labels": np.concatenate(
            [train_real["base_labels"], train_fake["base_labels"]]
        ),
        "test_videos": test_real["videos"] + test_fake["videos"],
        "test_aus": test_real["aus"] + test_fake["aus"],
        "test_labels": np.concatenate(
            [test_real["labels"], test_fake["labels"]]
        ),
        "counts": {
            "train_real": len(train_real["labels"]),
            "train_fake": len(train_fake["labels"]),
            "test_real": len(test_real["labels"]),
            "test_fake": len(test_fake["labels"]),
            "missing_train": train_real["missing"] + train_fake["missing"],
            "missing_test": test_real["missing"] + test_fake["missing"],
        },
        "test_total": len(test_real["labels"]) + len(test_fake["labels"]),
    }


def _sequence_matrix(
    au_paths: list[str],
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = cache_dir / "wangxing_v41_expression_sequence.npz"
    wanted = np.asarray(au_paths, dtype=object)
    cached: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if cache_path.is_file():
        try:
            with np.load(str(cache_path), allow_pickle=True) as payload:
                paths = payload["paths"].astype(object).tolist()
                sequences = payload["sequences"].astype(np.float32)
                summaries = payload["summaries"].astype(np.float32)
                if (
                    sequences.ndim == 3
                    and summaries.ndim == 2
                    and len(paths) == len(sequences) == len(summaries)
                    and sequences.shape[1:] == (
                        SEQUENCE_MAX_FRAMES,
                        SEQUENCE_FRAME_DIM,
                    )
                    and summaries.shape[1] == SEQUENCE_SUMMARY_DIM
                ):
                    cached = {
                        str(path): (sequences[index], summaries[index])
                        for index, path in enumerate(paths)
                    }
        except (KeyError, OSError, ValueError):
            cached = {}

    sequences: list[np.ndarray] = []
    summaries: list[np.ndarray] = []
    dirty = False
    for index, au in enumerate(wanted, start=1):
        key = str(au)
        if key in cached:
            sequence, summary = cached[key]
        else:
            sequence, summary = extract_expression_sequence_features(
                key,
                max_frames=SEQUENCE_MAX_FRAMES,
            )
            cached[key] = (
                np.asarray(sequence, dtype=np.float32),
                np.asarray(summary, dtype=np.float32),
            )
            dirty = True
        sequences.append(cached[key][0])
        summaries.append(cached[key][1])
        if index == 1 or index % 25 == 0 or index == len(wanted):
            print(
                f"[expression sequence] {index}/{len(wanted)}"
                f"{' cached' if key in cached and not dirty else ''}",
                flush=True,
            )
            if dirty:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    str(cache_path),
                    paths=np.asarray(list(cached), dtype=object),
                    sequences=np.stack(
                        [item[0] for item in cached.values()]
                    ).astype(np.float32),
                    summaries=np.stack(
                        [item[1] for item in cached.values()]
                    ).astype(np.float32),
                )
                dirty = False
    return (
        np.stack(sequences).astype(np.float32),
        np.stack(summaries).astype(np.float32),
    )


def _blendshape_matrix(video_paths: list[str], cache_dir: Path) -> np.ndarray:
    cache_path = cache_dir / "wangxing_v41_blendshape.npz"
    wanted = np.asarray(video_paths, dtype=object)
    cached: dict[str, np.ndarray] = {}
    if cache_path.is_file():
        try:
            with np.load(str(cache_path), allow_pickle=True) as payload:
                paths = payload["paths"].astype(object).tolist()
                features = payload["features"].astype(np.float32)
                if (
                    features.ndim == 2
                    and features.shape[1] == BLENDSHAPE_FEATURE_DIM
                    and len(paths) == len(features)
                ):
                    cached = {
                        str(path): features[index]
                        for index, path in enumerate(paths)
                    }
        except (KeyError, OSError, ValueError):
            cached = {}
    values: list[np.ndarray] = []
    dirty = False
    for index, video in enumerate(wanted, start=1):
        key = str(video)
        if key not in cached:
            cached[key] = np.asarray(
                blendshape_temporal_vector(key),
                dtype=np.float32,
            )
            dirty = True
        values.append(cached[key])
        if index == 1 or index % 10 == 0 or index == len(wanted):
            print(
                f"[blendshape v4.1] {index}/{len(wanted)}",
                flush=True,
            )
            if dirty:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    str(cache_path),
                    paths=np.asarray(list(cached), dtype=object),
                    features=np.stack(list(cached.values())).astype(np.float32),
                )
                dirty = False
    return np.stack(values).astype(np.float32)


def _make_tensors(
    sequence: np.ndarray,
    summary: np.ndarray,
    blendshape: np.ndarray,
    device: torch.device,
) -> list[torch.Tensor]:
    return [
        torch.from_numpy(sequence).to(device),
        torch.from_numpy(summary).to(device),
        torch.from_numpy(blendshape).to(device),
    ]


def train_wangxing_v41(
    *,
    manifest: dict[str, Any],
    cache_dir: Path,
    model_path: Path,
    epochs: int = 80,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    seed: int = 42,
    device: str = "cuda",
    model_type: str = V41_MODEL_TYPE,
) -> dict[str, Any]:
    torch_device = resolve_torch_device(device)
    _set_seed(seed)
    prepared = _prepare_expression_data(manifest)
    fit_idx, val_idx = _group_split(
        prepared["train_labels"],
        prepared["train_groups"],
        prepared["train_base_labels"],
        seed=seed,
    )
    train_sequence, train_summary = _sequence_matrix(
        prepared["train_aus"],
        cache_dir,
    )
    test_sequence, test_summary = _sequence_matrix(
        prepared["test_aus"],
        cache_dir / "test",
    )
    train_blendshape = _blendshape_matrix(
        prepared["train_videos"],
        cache_dir,
    )
    test_blendshape = _blendshape_matrix(
        prepared["test_videos"],
        cache_dir / "test",
    )
    sequence_mean, sequence_scale = _standardize(
        train_sequence,
        fit_idx,
        sequence=True,
    )
    summary_mean, summary_scale = _standardize(train_summary, fit_idx)
    blendshape_mean, blendshape_scale = _standardize(
        train_blendshape,
        fit_idx,
    )
    train_sequence = _normalize(
        train_sequence,
        sequence_mean,
        sequence_scale,
    )
    test_sequence = _normalize(test_sequence, sequence_mean, sequence_scale)
    train_summary = _normalize(train_summary, summary_mean, summary_scale)
    test_summary = _normalize(test_summary, summary_mean, summary_scale)
    train_blendshape = _normalize(
        train_blendshape,
        blendshape_mean,
        blendshape_scale,
    )
    test_blendshape = _normalize(
        test_blendshape,
        blendshape_mean,
        blendshape_scale,
    )

    x_sequence = train_sequence[fit_idx]
    x_summary = train_summary[fit_idx]
    x_blendshape = train_blendshape[fit_idx]
    y_fit = prepared["train_labels"][fit_idx].astype(np.float32)
    tensors = [
        torch.from_numpy(x_sequence),
        torch.from_numpy(x_summary),
        torch.from_numpy(x_blendshape),
        torch.from_numpy(y_fit),
    ]
    counts = np.bincount(y_fit.astype(np.int64), minlength=2)
    sample_weights = np.asarray(
        [1.0 / max(float(counts[int(value)]), 1.0) for value in y_fit],
        dtype=np.float64,
    )
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=max(1, int(batch_size)),
        sampler=WeightedRandomSampler(
            torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        ),
        pin_memory=torch_device.type == "cuda",
    )
    model = ExpressionSequenceClassifier().to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=2e-4,
    )
    criterion = nn.BCEWithLogitsLoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_key = (-1.0, float("inf"))
    history: list[dict[str, Any]] = []

    for epoch in range(max(1, int(epochs))):
        model.train()
        losses: list[float] = []
        for batch_sequence, batch_summary, batch_blendshape, target in loader:
            batch_sequence = batch_sequence.to(torch_device)
            batch_summary = batch_summary.to(torch_device)
            batch_blendshape = batch_blendshape.to(torch_device)
            target = target.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            joint, sequence_logit, blendshape_logit = model(
                batch_sequence,
                batch_summary,
                batch_blendshape,
                return_aux=True,
            )
            augmented_sequence = batch_sequence + (
                0.01 * torch.randn_like(batch_sequence)
            )
            frame_keep = (
                torch.rand(
                    augmented_sequence.shape[0],
                    augmented_sequence.shape[1],
                    1,
                    device=torch_device,
                )
                > 0.04
            ).to(augmented_sequence.dtype)
            augmented_sequence = augmented_sequence * frame_keep
            augmented_joint = model(
                augmented_sequence,
                batch_summary,
                batch_blendshape,
            )
            loss = (
                0.75 * _focal_bce_with_logits(joint, target)
                + 0.10 * criterion(sequence_logit, target)
                + 0.10 * criterion(blendshape_logit, target)
                + 0.05
                * F.mse_loss(
                    torch.sigmoid(augmented_joint),
                    torch.sigmoid(joint.detach()),
                )
            )
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_logits = model(
                *_make_tensors(
                    train_sequence[val_idx],
                    train_summary[val_idx],
                    train_blendshape[val_idx],
                    torch_device,
                )
            ).detach().cpu().numpy()
        val_prob = _sigmoid(val_logits)
        metrics = _classification_metrics(
            prepared["train_labels"][val_idx],
            val_prob,
        )
        val_loss = float(
            criterion(
                torch.from_numpy(val_logits),
                torch.from_numpy(
                    prepared["train_labels"][val_idx].astype(np.float32)
                ),
            )
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else math.inf,
                "validation_loss": val_loss,
                **metrics,
            }
        )
        key = (metrics["balanced_accuracy"], -val_loss)
        if key > best_key:
            best_key = key
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"[epoch {epoch + 1}/{epochs}] "
                f"train_loss={history[-1]['train_loss']:.4f} "
                f"val_bacc={metrics['balanced_accuracy']:.4f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(
            *_make_tensors(
                train_sequence[val_idx],
                train_summary[val_idx],
                train_blendshape[val_idx],
                torch_device,
            )
        ).detach().cpu().numpy()
        test_logits = model(
            *_make_tensors(
                test_sequence,
                test_summary,
                test_blendshape,
                torch_device,
            )
        ).detach().cpu().numpy()
    temperature = _fit_temperature(
        val_logits,
        prepared["train_labels"][val_idx],
    )
    val_prob = _sigmoid(val_logits / max(temperature, 1e-6))
    decision_threshold = _select_threshold(
        prepared["train_labels"][val_idx],
        val_prob,
    )
    test_prob = _sigmoid(test_logits / max(temperature, 1e-6))
    headline, confusion = _headline_with_threshold(
        prepared["test_labels"],
        test_prob,
        total_count=prepared["test_total"],
        threshold=decision_threshold,
    )
    checkpoint = {
        "model_type": model_type,
        "model_state": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "sequence_mean": sequence_mean,
        "sequence_scale": sequence_scale,
        "summary_mean": summary_mean,
        "summary_scale": summary_scale,
        "blendshape_mean": blendshape_mean,
        "blendshape_scale": blendshape_scale,
        "temperature": float(temperature),
        "decision_threshold": float(decision_threshold),
        "config": {
            "sequence_frame_dim": SEQUENCE_FRAME_DIM,
            "sequence_summary_dim": SEQUENCE_SUMMARY_DIM,
            "sequence_max_frames": SEQUENCE_MAX_FRAMES,
            "blendshape_dim": BLENDSHAPE_FEATURE_DIM,
            "sequence_hidden": SEQUENCE_HIDDEN,
            "fusion_mode": (
                "profile_independent_au_landmark_sequence"
                "_plus_blendshape_temporal_gru"
            ),
            "primary_signal": (
                "AU intensity/presence + normalized landmark trajectory "
                "+ expression dynamics + Blendshape statistics"
            ),
        },
        "dataset": prepared["counts"],
        "validation": {
            "fit_count": int(len(fit_idx)),
            "validation_count": int(len(val_idx)),
            "normalization_fit": "fit_groups_only",
        },
        "test_headline": headline,
        "confusion": confusion,
        "history_tail": history[-10:],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)
    return {
        "model_path": str(model_path),
        "headline": headline,
        "confusion": confusion,
        "counts": prepared["counts"],
        "temperature": float(temperature),
        "device": str(torch_device),
    }


def _load_model(
    path: Path,
    *,
    expected_model_type: str = V41_MODEL_TYPE,
) -> tuple[ExpressionSequenceClassifier, dict[str, Any]]:
    checkpoint = torch.load(str(path), map_location="cpu")
    if checkpoint.get("model_type") != expected_model_type:
        raise ValueError(
            f"Unsupported expression model: {checkpoint.get('model_type')}"
        )
    model = ExpressionSequenceClassifier()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def predict_wangxing_v41(
    *,
    video_path: Path,
    au_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    model, checkpoint = _load_model(model_path)
    return _predict_loaded_v41(
        model=model,
        checkpoint=checkpoint,
        video_path=video_path,
        au_path=au_path,
        model_path=model_path,
    )


def _predict_loaded_v41(
    *,
    model: ExpressionSequenceClassifier,
    checkpoint: dict[str, Any],
    video_path: Path,
    au_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    sequence, summary = extract_expression_sequence_features(
        au_path,
        max_frames=SEQUENCE_MAX_FRAMES,
    )
    blendshape = blendshape_temporal_vector(video_path)
    sequence = _normalize(
        sequence[None, ...],
        checkpoint["sequence_mean"],
        checkpoint["sequence_scale"],
    )
    summary = _normalize(
        summary[None, ...],
        checkpoint["summary_mean"],
        checkpoint["summary_scale"],
    )
    blendshape = _normalize(
        np.asarray(blendshape, dtype=np.float32)[None, ...],
        checkpoint["blendshape_mean"],
        checkpoint["blendshape_scale"],
    )
    with torch.no_grad():
        logit = float(
            model(
                torch.from_numpy(sequence),
                torch.from_numpy(summary),
                torch.from_numpy(blendshape),
            ).item()
        )
    probability = float(
        _sigmoid(
            np.asarray(
                [logit / max(float(checkpoint["temperature"]), 1e-6)]
            )
        )[0]
    )
    decision_threshold = float(
        checkpoint.get("decision_threshold", 0.5)
    )
    return {
        "prediction": (
            "generated"
            if probability >= decision_threshold
            else "real"
        ),
        "generated_probability": probability,
        "real_probability": 1.0 - probability,
        "decision_threshold": decision_threshold,
        "model_path": str(model_path),
        "fusion_mode": checkpoint["config"]["fusion_mode"],
    }


def evaluate_holdout_v41(
    *,
    holdout_manifest: Path,
    model_path: Path,
) -> dict[str, Any]:
    model, checkpoint = _load_model(model_path)
    holdout = json.loads(
        Path(holdout_manifest).read_text(encoding="utf-8-sig")
    )
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    probabilities: list[float] = []
    samples = [
        (0, "real", item) for item in holdout.get("real", [])
    ] + [
        (1, "generated", item)
        for item in holdout.get("seedance", [])
    ]
    for index, (label, source_label, item) in enumerate(samples, start=1):
        video = project_path(str(item["video"]))
        au = project_path(str(item["au"]))
        if not video.is_file() or not au.is_file():
            rows.append(
                {
                    "index": index,
                    "source_label": source_label,
                    "label_generated": label,
                    "status": "missing_inputs",
                    "video": str(video),
                    "au": str(au),
                }
            )
            continue
        result = _predict_loaded_v41(
            model=model,
            checkpoint=checkpoint,
            video_path=video,
            au_path=au,
            model_path=model_path,
        )
        labels.append(label)
        probabilities.append(float(result["generated_probability"]))
        rows.append(
            {
                "index": index,
                "source_label": source_label,
                "label_generated": label,
                "status": "ok",
                "video": str(video),
                "au": str(au),
                **result,
            }
        )
        if index == 1 or index % 10 == 0 or index == len(samples):
            print(
                f"[v4.1 evaluate] {index}/{len(samples)}",
                flush=True,
            )
    headline, confusion = _headline_with_threshold(
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float32),
        total_count=len(samples),
        threshold=float(checkpoint.get("decision_threshold", 0.5)),
    )
    return {
        "schema_version": "wangxing_expression_authenticity_v41_metrics_v1",
        "model_path": str(model_path),
        "holdout_manifest": str(holdout_manifest),
        "headline": headline,
        "confusion": confusion,
        "rows": rows,
    }
