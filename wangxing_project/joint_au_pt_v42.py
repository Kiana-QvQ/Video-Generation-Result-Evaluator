"""PT v4.2 expression model with AU relations and local temporal attention."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from evaluator.modules.core.paths import project_path
from wangxing_project.blendshape_temporal import (
    BLENDSHAPE_FEATURE_DIM,
    blendshape_temporal_vector,
)
from wangxing_project.expression_sequence import (
    PADDING_MASK_INDEX,
    SEQUENCE_FRAME_DIM_WITH_PADDING,
    SEQUENCE_MAX_FRAMES,
    SEQUENCE_SUMMARY_DIM,
    extract_expression_sequence_features,
)
from wangxing_project.joint_au_pt_v41 import (
    _focal_bce_with_logits,
    _group_split,
    _headline_with_threshold,
    _normalize,
    _prepare_expression_data,
    _select_threshold,
    _standardize,
)
from wangxing_project.joint_au_pt import resolve_torch_device
from evaluator.vedio_pred.real_video_detector import (
    _classification_metrics,
    _fit_temperature,
    _set_seed,
)

V42_MODEL_TYPE = "wangxing_expression_authenticity_v42"
SEQUENCE_FRAME_DIM = SEQUENCE_FRAME_DIM_WITH_PADDING
AU_COUNT = 12
AU_FEATURE_DIM = AU_COUNT * 2
LANDMARK_START = AU_FEATURE_DIM
LANDMARK_DIM = 17 * 2
AU_REL_HIDDEN = 32
TEMPORAL_HIDDEN = 64
LOCAL_HIDDEN = 48
SUMMARY_HIDDEN = 32
BLENDSHAPE_HIDDEN = 48
FUSION_HIDDEN = 96


class ExpressionRelationTemporalClassifier(nn.Module):
    """Small multimodal temporal head without RGB or texture inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.au_token_projection = nn.Linear(2, AU_REL_HIDDEN)
        self.au_relation = nn.MultiheadAttention(
            AU_REL_HIDDEN,
            num_heads=4,
            batch_first=True,
            dropout=0.10,
        )
        self.frame_projection = nn.Linear(
            SEQUENCE_FRAME_DIM + AU_REL_HIDDEN,
            TEMPORAL_HIDDEN * 2,
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                TEMPORAL_HIDDEN * 2,
                TEMPORAL_HIDDEN * 2,
                kernel_size=3,
                padding=1,
                groups=4,
            ),
            nn.GELU(),
            nn.Conv1d(
                TEMPORAL_HIDDEN * 2,
                TEMPORAL_HIDDEN * 2,
                kernel_size=1,
            ),
            nn.GroupNorm(4, TEMPORAL_HIDDEN * 2),
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=TEMPORAL_HIDDEN * 2,
            nhead=4,
            dim_feedforward=TEMPORAL_HIDDEN * 4,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=2,
        )
        self.temporal_attention = nn.Linear(TEMPORAL_HIDDEN * 2, 1)
        self.local_projection = nn.Sequential(
            nn.Linear(LANDMARK_DIM, LOCAL_HIDDEN),
            nn.LayerNorm(LOCAL_HIDDEN),
            nn.GELU(),
        )
        self.local_gru = nn.GRU(
            LOCAL_HIDDEN,
            LOCAL_HIDDEN // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.local_attention = nn.Linear(LOCAL_HIDDEN, 1)
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
            nn.Linear(
                TEMPORAL_HIDDEN * 2
                + LOCAL_HIDDEN
                + SUMMARY_HIDDEN
                + BLENDSHAPE_HIDDEN,
                FUSION_HIDDEN,
            ),
            nn.LayerNorm(FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(FUSION_HIDDEN, 1),
        )
        self.temporal_head = nn.Linear(TEMPORAL_HIDDEN * 2, 1)
        self.local_head = nn.Linear(LOCAL_HIDDEN, 1)
        self.blendshape_head = nn.Linear(BLENDSHAPE_HIDDEN, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        summary: torch.Tensor,
        blendshape: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size, frame_count, _ = sequence.shape
        au_pairs = sequence[:, :, :AU_FEATURE_DIM].reshape(
            batch_size * frame_count,
            AU_COUNT,
            2,
        )
        au_tokens = self.au_token_projection(au_pairs)
        au_tokens, _ = self.au_relation(au_tokens, au_tokens, au_tokens)
        au_embedding = au_tokens.mean(dim=1).reshape(
            batch_size,
            frame_count,
            AU_REL_HIDDEN,
        )
        frame_tokens = self.frame_projection(
            torch.cat([sequence, au_embedding], dim=-1)
        )
        padding_mask = sequence[:, :, PADDING_MASK_INDEX] > 0.5
        temporal = self.temporal_conv(
            frame_tokens.transpose(1, 2)
        ).transpose(1, 2)
        temporal = self.temporal_transformer(
            temporal,
            src_key_padding_mask=padding_mask,
        )
        temporal_logits = self.temporal_attention(temporal).squeeze(-1)
        temporal_logits = temporal_logits.masked_fill(
            padding_mask,
            -1e4,
        )
        temporal_weights = torch.softmax(temporal_logits, dim=1).unsqueeze(-1)
        temporal_embedding = torch.sum(
            temporal * temporal_weights,
            dim=1,
        )
        local = self.local_projection(
            sequence[:, :, LANDMARK_START : LANDMARK_START + LANDMARK_DIM]
        )
        local, _ = self.local_gru(local)
        local_logits = self.local_attention(local).squeeze(-1)
        local_logits = local_logits.masked_fill(padding_mask, -1e4)
        local_weights = torch.softmax(local_logits, dim=1).unsqueeze(-1)
        local_embedding = torch.sum(local * local_weights, dim=1)
        summary_embedding = self.summary_encoder(summary)
        blendshape_embedding = self.blendshape_encoder(blendshape)
        joint = self.fusion(
            torch.cat(
                [
                    temporal_embedding,
                    local_embedding,
                    summary_embedding,
                    blendshape_embedding,
                ],
                dim=1,
            )
        ).squeeze(1)
        if not return_aux:
            return joint
        return (
            joint,
            self.temporal_head(temporal_embedding).squeeze(1),
            self.local_head(local_embedding).squeeze(1),
            self.blendshape_head(blendshape_embedding).squeeze(1),
        )


def _make_tensors(
    sequence: np.ndarray,
    summary: np.ndarray,
    blendshape: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(sequence).to(device),
        torch.from_numpy(summary).to(device),
        torch.from_numpy(blendshape).to(device),
    )


def _sequence_matrix_v42(
    au_paths: list[str],
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = cache_dir / "wangxing_v42_expression_sequence.npz"
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
                    and sequences.shape[1:] == (
                        SEQUENCE_MAX_FRAMES,
                        SEQUENCE_FRAME_DIM,
                    )
                    and summaries.ndim == 2
                    and summaries.shape[1] == SEQUENCE_SUMMARY_DIM
                    and len(paths) == len(sequences) == len(summaries)
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
        if key not in cached:
            try:
                cached[key] = extract_expression_sequence_features(
                    key,
                    max_frames=SEQUENCE_MAX_FRAMES,
                    include_padding_mask=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"v4.2 expression sequence extraction failed for {key}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            dirty = True
        sequences.append(np.asarray(cached[key][0], dtype=np.float32))
        summaries.append(np.asarray(cached[key][1], dtype=np.float32))
        if index == 1 or index % 25 == 0 or index == len(wanted):
            print(f"[v4.2 sequence] {index}/{len(wanted)}", flush=True)
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
    return np.stack(sequences), np.stack(summaries)


def _blendshape_matrix_v42(
    video_paths: list[str],
    cache_dir: Path,
) -> np.ndarray:
    cache_path = cache_dir / "wangxing_v42_blendshape.npz"
    legacy_cache_path = cache_dir / "wangxing_v41_blendshape.npz"
    wanted = np.asarray(video_paths, dtype=object)
    cached: dict[str, np.ndarray] = {}
    source_cache_path = (
        cache_path if cache_path.is_file() else legacy_cache_path
    )
    if source_cache_path.is_file():
        try:
            with np.load(str(source_cache_path), allow_pickle=True) as payload:
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
            try:
                value = np.asarray(
                    blendshape_temporal_vector(key),
                    dtype=np.float32,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"v4.2 Blendshape extraction failed for {key}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if value.shape != (BLENDSHAPE_FEATURE_DIM,):
                raise ValueError(
                    f"v4.2 Blendshape shape mismatch for {key}: {value.shape}"
                )
            cached[key] = value
            dirty = True
        values.append(cached[key])
        if index == 1 or index % 10 == 0 or index == len(wanted):
            print(f"[v4.2 blendshape] {index}/{len(wanted)}", flush=True)
            if dirty:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    str(cache_path),
                    paths=np.asarray(list(cached), dtype=object),
                    features=np.stack(list(cached.values())).astype(np.float32),
                )
                dirty = False
    return np.stack(values).astype(np.float32)


def _train_model(
    *,
    prepared: dict[str, Any],
    cache_dir: Path,
    model_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    torch_device = resolve_torch_device(device)
    _set_seed(seed)
    fit_idx, val_idx = _group_split(
        prepared["train_labels"],
        prepared["train_groups"],
        prepared["train_base_labels"],
        seed=seed,
    )
    sequence_train_raw, summary_train = _sequence_matrix_v42(
        prepared["train_aus"],
        cache_dir,
    )
    sequence_test_raw, summary_test = _sequence_matrix_v42(
        prepared["test_aus"],
        cache_dir / "test",
    )
    blendshape_train = _blendshape_matrix_v42(
        prepared["train_videos"],
        cache_dir,
    )
    blendshape_test = _blendshape_matrix_v42(
        prepared["test_videos"],
        cache_dir / "test",
    )
    sequence_mean, sequence_scale = _standardize(
        sequence_train_raw,
        fit_idx,
        sequence=True,
    )
    summary_mean, summary_scale = _standardize(summary_train, fit_idx)
    blendshape_mean, blendshape_scale = _standardize(
        blendshape_train,
        fit_idx,
    )
    sequence_train = _normalize(
        sequence_train_raw,
        sequence_mean,
        sequence_scale,
    )
    sequence_test = _normalize(
        sequence_test_raw,
        sequence_mean,
        sequence_scale,
    )
    sequence_train[:, :, PADDING_MASK_INDEX] = sequence_train_raw[
        :, :, PADDING_MASK_INDEX
    ]
    sequence_test[:, :, PADDING_MASK_INDEX] = sequence_test_raw[
        :, :, PADDING_MASK_INDEX
    ]
    summary_train = _normalize(summary_train, summary_mean, summary_scale)
    summary_test = _normalize(summary_test, summary_mean, summary_scale)
    blendshape_train = _normalize(
        blendshape_train,
        blendshape_mean,
        blendshape_scale,
    )
    blendshape_test = _normalize(
        blendshape_test,
        blendshape_mean,
        blendshape_scale,
    )
    labels = prepared["train_labels"]
    y_fit = labels[fit_idx].astype(np.float32)
    class_counts = np.bincount(y_fit.astype(np.int64), minlength=2)
    weights = np.asarray(
        [
            1.0 / max(float(class_counts[int(value)]), 1.0)
            for value in y_fit
        ],
        dtype=np.float64,
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(sequence_train[fit_idx]),
            torch.from_numpy(summary_train[fit_idx]),
            torch.from_numpy(blendshape_train[fit_idx]),
            torch.from_numpy(y_fit),
        ),
        batch_size=max(1, int(batch_size)),
        sampler=torch.utils.data.WeightedRandomSampler(
            torch.from_numpy(weights),
            num_samples=len(weights),
            replacement=True,
        ),
        pin_memory=torch_device.type == "cuda",
    )
    model = ExpressionRelationTemporalClassifier().to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=3e-4,
    )
    criterion = nn.BCEWithLogitsLoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_key = (-1.0, float("inf"))
    history: list[dict[str, Any]] = []
    for epoch in range(max(1, int(epochs))):
        model.train()
        losses: list[float] = []
        for seq, summary, blendshape, target in loader:
            seq = seq.to(torch_device)
            summary = summary.to(torch_device)
            blendshape = blendshape.to(torch_device)
            target = target.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            joint, temporal, local, bs = model(
                seq,
                summary,
                blendshape,
                return_aux=True,
            )
            perturbed = seq.clone()
            perturbed[:, :, :PADDING_MASK_INDEX] = (
                perturbed[:, :, :PADDING_MASK_INDEX]
                + 0.01
                * torch.randn_like(
                    perturbed[:, :, :PADDING_MASK_INDEX]
                )
            )
            keep = (
                torch.rand(
                    perturbed.shape[0],
                    perturbed.shape[1],
                    1,
                    device=torch_device,
                )
                > 0.04
            ).to(perturbed.dtype)
            perturbed[:, :, :PADDING_MASK_INDEX] = (
                perturbed[:, :, :PADDING_MASK_INDEX] * keep
            )
            augmented_joint = model(
                perturbed,
                summary,
                blendshape,
            )
            loss = (
                0.70 * _focal_bce_with_logits(joint, target)
                + 0.10 * criterion(temporal, target)
                + 0.10 * criterion(local, target)
                + 0.05 * criterion(bs, target)
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
                    sequence_train[val_idx],
                    summary_train[val_idx],
                    blendshape_train[val_idx],
                    torch_device,
                )
            ).detach().cpu().numpy()
        val_prob = 1.0 / (1.0 + np.exp(-val_logits))
        metrics = _classification_metrics(
            labels[val_idx],
            val_prob,
        )
        val_loss = float(
            criterion(
                torch.from_numpy(val_logits),
                torch.from_numpy(labels[val_idx].astype(np.float32)),
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
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"[v4.2 epoch {epoch + 1}/{epochs}] "
                f"loss={history[-1]['train_loss']:.4f} "
                f"val_bacc={metrics['balanced_accuracy']:.4f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(
            *_make_tensors(
                sequence_train[val_idx],
                summary_train[val_idx],
                blendshape_train[val_idx],
                torch_device,
            )
        ).detach().cpu().numpy()
        test_logits = model(
            *_make_tensors(
                sequence_test,
                summary_test,
                blendshape_test,
                torch_device,
            )
        ).detach().cpu().numpy()
    temperature = _fit_temperature(val_logits, labels[val_idx])
    val_prob = 1.0 / (
        1.0 + np.exp(-val_logits / max(temperature, 1e-6))
    )
    threshold = _select_threshold(labels[val_idx], val_prob)
    test_prob = 1.0 / (
        1.0 + np.exp(-test_logits / max(temperature, 1e-6))
    )
    headline, confusion = _headline_with_threshold(
        prepared["test_labels"],
        test_prob,
        total_count=prepared["test_total"],
        threshold=threshold,
    )
    checkpoint = {
        "model_type": V42_MODEL_TYPE,
        "model_state": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "sequence_mean": sequence_mean,
        "sequence_scale": sequence_scale,
        "summary_mean": summary_mean,
        "summary_scale": summary_scale,
        "blendshape_mean": blendshape_mean,
        "blendshape_scale": blendshape_scale,
        "temperature": float(temperature),
        "decision_threshold": float(threshold),
        "config": {
            "sequence_frame_dim": SEQUENCE_FRAME_DIM,
            "sequence_summary_dim": SEQUENCE_SUMMARY_DIM,
            "sequence_max_frames": SEQUENCE_MAX_FRAMES,
            "blendshape_dim": BLENDSHAPE_FEATURE_DIM,
            "au_relation_hidden": AU_REL_HIDDEN,
            "temporal_hidden": TEMPORAL_HIDDEN,
            "local_hidden": LOCAL_HIDDEN,
            "fusion_mode": (
                "au_relation_attention_plus_temporal_conv_transformer"
                "_plus_local_landmark_gru_plus_blendshape"
            ),
        },
        "dataset": prepared["counts"],
        "validation": {
            "fit_count": int(len(fit_idx)),
            "validation_count": int(len(val_idx)),
            "normalization_fit": "fit_groups_only",
            "threshold_selection": "validation_min_class_recall_then_accuracy",
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
        "decision_threshold": float(threshold),
        "device": str(torch_device),
    }


def train_wangxing_v42(
    *,
    manifest: dict[str, Any],
    cache_dir: Path,
    model_path: Path,
    epochs: int = 80,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    seed: int = 42,
    device: str = "cuda",
) -> dict[str, Any]:
    prepared = _prepare_expression_data(manifest)
    return _train_model(
        prepared=prepared,
        cache_dir=cache_dir,
        model_path=model_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
    )


def _load_model(
    path: Path,
) -> tuple[ExpressionRelationTemporalClassifier, dict[str, Any]]:
    checkpoint = torch.load(str(path), map_location="cpu")
    if checkpoint.get("model_type") != V42_MODEL_TYPE:
        raise ValueError(
            f"Unsupported v4.2 model: {checkpoint.get('model_type')}"
        )
    model = ExpressionRelationTemporalClassifier()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _predict_loaded(
    model: ExpressionRelationTemporalClassifier,
    checkpoint: dict[str, Any],
    video_path: Path,
    au_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    sequence_raw, summary = extract_expression_sequence_features(
        au_path,
        max_frames=SEQUENCE_MAX_FRAMES,
        include_padding_mask=True,
    )
    blendshape = blendshape_temporal_vector(video_path)
    sequence = _normalize(
        sequence_raw[None, ...],
        checkpoint["sequence_mean"],
        checkpoint["sequence_scale"],
    )
    sequence[:, :, PADDING_MASK_INDEX] = sequence_raw[
        None, :, PADDING_MASK_INDEX
    ]
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
        1.0
        / (
            1.0
            + math.exp(
                -logit / max(float(checkpoint["temperature"]), 1e-6)
            )
        )
    )
    threshold = float(checkpoint.get("decision_threshold", 0.5))
    return {
        "prediction": (
            "generated" if probability >= threshold else "real"
        ),
        "generated_probability": probability,
        "real_probability": 1.0 - probability,
        "decision_threshold": threshold,
        "model_path": str(model_path),
        "fusion_mode": checkpoint["config"]["fusion_mode"],
    }


def predict_wangxing_v42(
    *,
    video_path: Path,
    au_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    model, checkpoint = _load_model(model_path)
    return _predict_loaded(
        model,
        checkpoint,
        video_path,
        au_path,
        model_path,
    )


def evaluate_holdout_v42(
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
        result = _predict_loaded(
            model,
            checkpoint,
            video,
            au,
            model_path,
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
            print(f"[v4.2 evaluate] {index}/{len(samples)}", flush=True)
    headline, confusion = _headline_with_threshold(
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float32),
        total_count=len(samples),
        threshold=float(checkpoint.get("decision_threshold", 0.5)),
    )
    return {
        "schema_version": "wangxing_expression_authenticity_v42_metrics_v1",
        "model_path": str(model_path),
        "holdout_manifest": str(holdout_manifest),
        "headline": headline,
        "confusion": confusion,
        "rows": rows,
    }
