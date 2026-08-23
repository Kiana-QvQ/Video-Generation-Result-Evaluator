"""PT v4.3: expression head plus low-weight face-crop temporal auxiliary head."""

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
from evaluator.vedio_pred.real_video_detector import (
    _classification_metrics,
    _fit_temperature,
    _set_seed,
)
from wangxing_project.face_crop_temporal import (
    CROP_FRAME_DIM,
    CROP_MAX_FRAMES,
    CROP_SUMMARY_DIM,
    extract_face_crop_temporal_features,
)
from wangxing_project.joint_au_pt import resolve_torch_device
from wangxing_project.joint_au_pt_v41 import (
    _blendshape_matrix,
    _focal_bce_with_logits,
    _group_split,
    _headline_with_threshold,
    _normalize,
    _prepare_expression_data,
    _select_threshold,
    _standardize,
)
from wangxing_project.joint_au_pt_v42 import (
    ExpressionRelationTemporalClassifier,
    _blendshape_matrix_v42,
    _sequence_matrix_v42,
)

V43_MODEL_TYPE = "wangxing_expression_authenticity_v43"
CROP_HIDDEN = 32
CROP_FUSION_HIDDEN = 64


class ExpressionFaceCropClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.expression_model = ExpressionRelationTemporalClassifier()
        self.crop_gru = nn.GRU(
            CROP_FRAME_DIM,
            CROP_HIDDEN,
            batch_first=True,
            bidirectional=True,
        )
        self.crop_attention = nn.Linear(CROP_HIDDEN * 2, 1)
        self.crop_summary = nn.Sequential(
            nn.Linear(CROP_SUMMARY_DIM, 16),
            nn.LayerNorm(16),
            nn.GELU(),
        )
        self.crop_head = nn.Linear(CROP_HIDDEN * 2, 1)
        self.fusion = nn.Sequential(
            nn.Linear(1 + CROP_HIDDEN * 2 + 16, CROP_FUSION_HIDDEN),
            nn.LayerNorm(CROP_FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(CROP_FUSION_HIDDEN, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        summary: torch.Tensor,
        blendshape: torch.Tensor,
        crop_sequence: torch.Tensor,
        crop_summary: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expression_logit = self.expression_model(
            sequence,
            summary,
            blendshape,
        )
        crop_encoded, _ = self.crop_gru(crop_sequence)
        crop_weights = torch.softmax(
            self.crop_attention(crop_encoded),
            dim=1,
        )
        crop_embedding = torch.sum(crop_encoded * crop_weights, dim=1)
        crop_summary_embedding = self.crop_summary(crop_summary)
        joint = self.fusion(
            torch.cat(
                [
                    expression_logit.unsqueeze(1),
                    crop_embedding,
                    crop_summary_embedding,
                ],
                dim=1,
            )
        ).squeeze(1)
        if not return_aux:
            return joint
        return (
            joint,
            expression_logit,
            self.crop_head(crop_embedding).squeeze(1),
        )


def _crop_matrix(
    video_paths: list[str],
    au_paths: list[str],
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = cache_dir / "wangxing_v43_face_crop.npz"
    wanted = np.asarray(
        [f"{video}|{au}" for video, au in zip(video_paths, au_paths)],
        dtype=object,
    )
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
                        CROP_MAX_FRAMES,
                        CROP_FRAME_DIM,
                    )
                    and summaries.shape[1] == CROP_SUMMARY_DIM
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
    for index, (video, au) in enumerate(
        zip(video_paths, au_paths),
        start=1,
    ):
        key = f"{video}|{au}"
        if key not in cached:
            try:
                cached[key] = extract_face_crop_temporal_features(
                    video,
                    au,
                    max_frames=CROP_MAX_FRAMES,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"v4.3 face-crop extraction failed for {video}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            dirty = True
        sequences.append(np.asarray(cached[key][0], dtype=np.float32))
        summaries.append(np.asarray(cached[key][1], dtype=np.float32))
        if index == 1 or index % 10 == 0 or index == len(wanted):
            print(f"[v4.3 face-crop] {index}/{len(wanted)}", flush=True)
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


def _make_tensors(
    sequence: np.ndarray,
    summary: np.ndarray,
    blendshape: np.ndarray,
    crop_sequence: np.ndarray,
    crop_summary: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.from_numpy(value).to(device)
        for value in (
            sequence,
            summary,
            blendshape,
            crop_sequence,
            crop_summary,
        )
    )


def _train_v43(
    *,
    prepared: dict[str, Any],
    cache_dir: Path,
    model_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    model_factory: type[nn.Module] = ExpressionFaceCropClassifier,
    model_type: str = V43_MODEL_TYPE,
    fusion_mode: str = (
        "expression_relation_temporal_head_plus_low_weight"
        "_face_crop_temporal_auxiliary"
    ),
) -> dict[str, Any]:
    torch_device = resolve_torch_device(device)
    _set_seed(seed)
    fit_idx, val_idx = _group_split(
        prepared["train_labels"],
        prepared["train_groups"],
        prepared["train_base_labels"],
        seed=seed,
    )
    sequence_train, summary_train = _sequence_matrix_v42(
        prepared["train_aus"],
        cache_dir,
    )
    sequence_test, summary_test = _sequence_matrix_v42(
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
    crop_train, crop_summary_train = _crop_matrix(
        prepared["train_videos"],
        prepared["train_aus"],
        cache_dir,
    )
    crop_test, crop_summary_test = _crop_matrix(
        prepared["test_videos"],
        prepared["test_aus"],
        cache_dir / "test",
    )
    stats = {}
    for name, values in (
        ("sequence", sequence_train),
        ("summary", summary_train),
        ("blendshape", blendshape_train),
        ("crop", crop_train),
        ("crop_summary", crop_summary_train),
    ):
        stats[f"{name}_mean"], stats[f"{name}_scale"] = _standardize(
            values,
            fit_idx,
            sequence=values.ndim == 3,
        )
    sequence_train = _normalize(
        sequence_train,
        stats["sequence_mean"],
        stats["sequence_scale"],
    )
    sequence_test = _normalize(
        sequence_test,
        stats["sequence_mean"],
        stats["sequence_scale"],
    )
    summary_train = _normalize(
        summary_train,
        stats["summary_mean"],
        stats["summary_scale"],
    )
    summary_test = _normalize(
        summary_test,
        stats["summary_mean"],
        stats["summary_scale"],
    )
    blendshape_train = _normalize(
        blendshape_train,
        stats["blendshape_mean"],
        stats["blendshape_scale"],
    )
    blendshape_test = _normalize(
        blendshape_test,
        stats["blendshape_mean"],
        stats["blendshape_scale"],
    )
    crop_train = _normalize(
        crop_train,
        stats["crop_mean"],
        stats["crop_scale"],
    )
    crop_test = _normalize(
        crop_test,
        stats["crop_mean"],
        stats["crop_scale"],
    )
    crop_summary_train = _normalize(
        crop_summary_train,
        stats["crop_summary_mean"],
        stats["crop_summary_scale"],
    )
    crop_summary_test = _normalize(
        crop_summary_test,
        stats["crop_summary_mean"],
        stats["crop_summary_scale"],
    )
    labels = prepared["train_labels"]
    y_fit = labels[fit_idx].astype(np.float32)
    counts = np.bincount(y_fit.astype(np.int64), minlength=2)
    sampler_weights = torch.from_numpy(
        np.asarray(
            [
                1.0 / max(float(counts[int(value)]), 1.0)
                for value in y_fit
            ],
            dtype=np.float64,
        )
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(sequence_train[fit_idx]),
            torch.from_numpy(summary_train[fit_idx]),
            torch.from_numpy(blendshape_train[fit_idx]),
            torch.from_numpy(crop_train[fit_idx]),
            torch.from_numpy(crop_summary_train[fit_idx]),
            torch.from_numpy(y_fit),
        ),
        batch_size=max(1, int(batch_size)),
        sampler=torch.utils.data.WeightedRandomSampler(
            sampler_weights,
            num_samples=len(sampler_weights),
            replacement=True,
        ),
        pin_memory=torch_device.type == "cuda",
    )
    model = model_factory().to(torch_device)
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
        for batch in loader:
            seq, summary, blendshape, crop, crop_summary, target = (
                item.to(torch_device) for item in batch
            )
            optimizer.zero_grad(set_to_none=True)
            joint, expression, crop_logit = model(
                seq,
                summary,
                blendshape,
                crop,
                crop_summary,
                return_aux=True,
            )
            crop_augmented = crop + 0.01 * torch.randn_like(crop)
            augmented_joint = model(
                seq,
                summary,
                blendshape,
                crop_augmented,
                crop_summary,
            )
            loss = (
                0.65 * _focal_bce_with_logits(joint, target)
                + 0.15 * criterion(expression, target)
                + 0.10 * criterion(crop_logit, target)
                + 0.10
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
                    crop_train[val_idx],
                    crop_summary_train[val_idx],
                    torch_device,
                )
            ).detach().cpu().numpy()
        val_prob = 1.0 / (1.0 + np.exp(-val_logits))
        metrics = _classification_metrics(labels[val_idx], val_prob)
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
                f"[v4.3 epoch {epoch + 1}/{epochs}] "
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
                crop_train[val_idx],
                crop_summary_train[val_idx],
                torch_device,
            )
        ).detach().cpu().numpy()
        test_logits = model(
            *_make_tensors(
                sequence_test,
                summary_test,
                blendshape_test,
                crop_test,
                crop_summary_test,
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
        "model_type": model_type,
        "model_state": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        **stats,
        "temperature": float(temperature),
        "decision_threshold": float(threshold),
        "config": {
            "crop_frame_dim": CROP_FRAME_DIM,
            "crop_summary_dim": CROP_SUMMARY_DIM,
            "crop_max_frames": CROP_MAX_FRAMES,
            "crop_hidden": CROP_HIDDEN,
            "fusion_mode": fusion_mode,
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


def train_wangxing_v43(
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
    return _train_v43(
        prepared=_prepare_expression_data(manifest),
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
    *,
    model_factory: type[nn.Module] = ExpressionFaceCropClassifier,
    expected_model_type: str = V43_MODEL_TYPE,
) -> tuple[ExpressionFaceCropClassifier, dict[str, Any]]:
    checkpoint = torch.load(str(path), map_location="cpu")
    if checkpoint.get("model_type") != expected_model_type:
        raise ValueError(
            f"Unsupported face-crop model: {checkpoint.get('model_type')}"
        )
    model = model_factory()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _predict_loaded(
    model: ExpressionFaceCropClassifier,
    checkpoint: dict[str, Any],
    video: Path,
    au: Path,
    model_path: Path,
) -> dict[str, Any]:
    from wangxing_project.joint_au_pt_v42 import (
        _blendshape_matrix_v42,
        _sequence_matrix_v42,
    )
    sequence, summary = _sequence_matrix_v42([str(au)], Path(
        str(model_path) + ".predict_cache"
    ))
    blendshape = _blendshape_matrix_v42([str(video)], Path(
        str(model_path) + ".predict_cache"
    ))
    crop, crop_summary = _crop_matrix(
        [str(video)],
        [str(au)],
        Path(str(model_path) + ".predict_cache"),
    )
    sequence = _normalize(
        sequence,
        checkpoint["sequence_mean"],
        checkpoint["sequence_scale"],
    )
    summary = _normalize(
        summary,
        checkpoint["summary_mean"],
        checkpoint["summary_scale"],
    )
    blendshape = _normalize(
        blendshape,
        checkpoint["blendshape_mean"],
        checkpoint["blendshape_scale"],
    )
    crop = _normalize(crop, checkpoint["crop_mean"], checkpoint["crop_scale"])
    crop_summary = _normalize(
        crop_summary,
        checkpoint["crop_summary_mean"],
        checkpoint["crop_summary_scale"],
    )
    with torch.no_grad():
        logit = float(
            model(
                torch.from_numpy(sequence),
                torch.from_numpy(summary),
                torch.from_numpy(blendshape),
                torch.from_numpy(crop),
                torch.from_numpy(crop_summary),
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
        "prediction": "generated" if probability >= threshold else "real",
        "generated_probability": probability,
        "real_probability": 1.0 - probability,
        "decision_threshold": threshold,
        "model_path": str(model_path),
        "fusion_mode": checkpoint["config"]["fusion_mode"],
    }


def predict_wangxing_v43(
    *,
    video_path: Path,
    au_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    model, checkpoint = _load_model(model_path)
    return _predict_loaded(model, checkpoint, video_path, au_path, model_path)


def evaluate_holdout_v43(
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
        result = _predict_loaded(model, checkpoint, video, au, model_path)
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
            print(f"[v4.3 evaluate] {index}/{len(samples)}", flush=True)
    headline, confusion = _headline_with_threshold(
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float32),
        total_count=len(samples),
        threshold=float(checkpoint.get("decision_threshold", 0.5)),
    )
    return {
        "schema_version": "wangxing_expression_authenticity_v43_metrics_v1",
        "model_path": str(model_path),
        "holdout_manifest": str(holdout_manifest),
        "headline": headline,
        "confusion": confusion,
        "rows": rows,
    }
