"""Face-geometry-aware v4 PT model.

This keeps the v3 temporal video/AU model as a base and adds a separate
quality-aware landmark/pose branch. It is intentionally parallel to v3.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from evaluator.modules.core.paths import project_path
from evaluator.vedio_pred.real_video_detector import (
    _classification_metrics,
    _fit_temperature,
    _set_seed,
)
from wangxing_project.face_geometry import (
    FACE_GEOMETRY_DIM,
    extract_face_geometry_features,
)
from wangxing_project.temporal_expression import (
    TRANSITION_FEATURE_NAMES,
    transition_feature_vector,
)
from wangxing_project.joint_au_pt import resolve_torch_device
from wangxing_project.joint_au_pt import resolve_au_csv_for_video
from wangxing_project.joint_au_pt_v3 import (
    DEFAULT_AUX_LOSS_WEIGHTS,
    TemporalAUVideoClassifier,
    _fit_sequence_stats,
    _group_split,
    _headline,
    _normalize_features,
    _prepare_data,
)

V4_MODEL_TYPE = "wangxing_face_geometry_v4"
FACE_HIDDEN = 32
TRANSITION_DIM = len(TRANSITION_FEATURE_NAMES)
TRANSITION_HIDDEN = 64


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


class FaceGeometryAwareClassifier(nn.Module):
    def __init__(self, *, modality_dropout: float = 0.10) -> None:
        super().__init__()
        self.video_model = TemporalAUVideoClassifier(
            modality_dropout=modality_dropout
        )
        self.face_encoder = nn.Sequential(
            nn.Linear(FACE_GEOMETRY_DIM, FACE_HIDDEN),
            nn.LayerNorm(FACE_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.transition_encoder = nn.Sequential(
            nn.Linear(TRANSITION_DIM, TRANSITION_HIDDEN),
            nn.LayerNorm(TRANSITION_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.fusion = nn.Sequential(
            nn.Linear(1 + FACE_HIDDEN + TRANSITION_HIDDEN, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, 1),
        )
        self.face_head = nn.Linear(FACE_HIDDEN, 1)

    def forward(
        self,
        frame_a: torch.Tensor,
        temporal_a: torch.Tensor,
        frame_b: torch.Tensor,
        temporal_b: torch.Tensor,
        au: torch.Tensor,
        face_geometry: torch.Tensor,
        transition_features: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_joint, _, _ = self.video_model(
            frame_a,
            temporal_a,
            frame_b,
            temporal_b,
            au,
            return_aux=True,
        )
        face = self.face_encoder(face_geometry)
        transition = self.transition_encoder(transition_features)
        joint = self.fusion(
            torch.cat(
                [base_joint.unsqueeze(1), face, transition],
                dim=1,
            )
        ).squeeze(1)
        if not return_aux:
            return joint
        return joint, base_joint, self.face_head(face).squeeze(1)


def _standardize_face(
    values: np.ndarray,
    fit_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = values[fit_idx].mean(axis=0).astype(np.float32)
    scale = values[fit_idx].std(axis=0).astype(np.float32)
    scale = np.maximum(scale, 1e-4)
    return mean, scale


def _normalize_face(
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return np.clip((values - mean) / scale, -8.0, 8.0).astype(np.float32)


def _face_matrix(paths: list[str]) -> np.ndarray:
    return np.stack(
        [extract_face_geometry_features(path) for path in paths]
    ).astype(np.float32)


def _transition_matrix(
    video_paths: list[str],
    au_paths: list[str],
    cache_dir: Path,
) -> np.ndarray:
    cache_path = cache_dir / "wangxing_v4_transition.npz"
    wanted = np.asarray(
        [f"{video}|{au}" for video, au in zip(video_paths, au_paths)],
        dtype=object,
    )
    if cache_path.is_file():
        try:
            with np.load(str(cache_path), allow_pickle=True) as payload:
                cached_paths = payload["paths"].astype(object)
                if np.array_equal(cached_paths, wanted):
                    return payload["features"].astype(np.float32)
        except (KeyError, OSError, ValueError):
            pass
    values = np.stack(
        [
            transition_feature_vector(video_path=video, au_path=au)
            for video, au in zip(video_paths, au_paths)
        ]
    ).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(cache_path),
        paths=wanted,
        features=values,
    )
    return values


def train_wangxing_v4(
    *,
    manifest: dict[str, Any],
    cache_dir: Path,
    model_path: Path,
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
    epochs: int = 80,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    seed: int = 42,
    device: str = "cuda",
    modality_dropout: float = 0.10,
) -> dict[str, Any]:
    torch_device = resolve_torch_device(device)
    _set_seed(seed)
    prepared = _prepare_data(
        manifest=manifest,
        cache_dir=cache_dir,
        source_profile=source_profile,
        forensics_profiles=forensics_profiles,
    )
    fit_idx, val_idx = _group_split(
        prepared["train_labels"],
        prepared["train_groups"],
        prepared["train_base_labels"],
        seed=seed,
    )
    stats = _fit_sequence_stats(prepared["train_features"], fit_idx)
    train_features = _normalize_features(prepared["train_features"], stats)
    test_features = _normalize_features(prepared["test_features"], stats)
    face_train_raw = _face_matrix(prepared["train_au_paths"])
    face_test_raw = _face_matrix(prepared["test_au_paths"])
    transition_train_raw = _transition_matrix(
        prepared["train_video_paths"],
        prepared["train_au_paths"],
        Path(cache_dir),
    )
    transition_test_raw = _transition_matrix(
        prepared["test_video_paths"],
        prepared["test_au_paths"],
        Path(cache_dir) / "test",
    )
    face_mean, face_scale = _standardize_face(face_train_raw, fit_idx)
    face_train = _normalize_face(face_train_raw, face_mean, face_scale)
    face_test = _normalize_face(face_test_raw, face_mean, face_scale)
    transition_mean, transition_scale = _standardize_face(
        transition_train_raw,
        fit_idx,
    )
    transition_train = _normalize_face(
        transition_train_raw,
        transition_mean,
        transition_scale,
    )
    transition_test = _normalize_face(
        transition_test_raw,
        transition_mean,
        transition_scale,
    )

    labels = prepared["train_labels"]
    x_fit = {name: value[fit_idx] for name, value in train_features.items()}
    x_val = {name: value[val_idx] for name, value in train_features.items()}
    face_fit = face_train[fit_idx]
    face_val = face_train[val_idx]
    transition_fit = transition_train[fit_idx]
    transition_val = transition_train[val_idx]
    tensors = [
        torch.from_numpy(np.ascontiguousarray(x_fit[name]))
        for name in ("frame_a", "temporal_a", "frame_b", "temporal_b", "au")
    ]
    tensors.append(torch.from_numpy(np.ascontiguousarray(face_fit)))
    tensors.append(
        torch.from_numpy(np.ascontiguousarray(transition_fit))
    )
    y_fit = labels[fit_idx].astype(np.float32)
    tensors.append(torch.from_numpy(y_fit))
    counts = np.bincount(y_fit.astype(np.int64), minlength=2)
    weights = np.asarray(
        [1.0 / max(float(counts[int(label)]), 1.0) for label in y_fit],
        dtype=np.float64,
    )
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=max(1, int(batch_size)),
        sampler=WeightedRandomSampler(
            torch.from_numpy(weights),
            num_samples=len(weights),
            replacement=True,
        ),
        pin_memory=torch_device.type == "cuda",
    )
    model = FaceGeometryAwareClassifier(
        modality_dropout=modality_dropout
    ).to(torch_device)
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
        for batch in loader:
            tensors_on_device = [
                tensor.to(torch_device, non_blocking=torch_device.type == "cuda")
                for tensor in batch
            ]
            *inputs, target = tensors_on_device
            optimizer.zero_grad(set_to_none=True)
            joint, base_logit, face_logit = model(
                *inputs,
                return_aux=True,
            )
            loss = (
                DEFAULT_AUX_LOSS_WEIGHTS["joint"] * criterion(joint, target)
                + DEFAULT_AUX_LOSS_WEIGHTS["video"] * criterion(base_logit, target)
                + DEFAULT_AUX_LOSS_WEIGHTS["au"] * criterion(face_logit, target)
            )
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        val_tensors = [
            torch.from_numpy(np.ascontiguousarray(x_val[name])).to(torch_device)
            for name in ("frame_a", "temporal_a", "frame_b", "temporal_b", "au")
        ]
        val_tensors.append(torch.from_numpy(face_val).to(torch_device))
        val_tensors.append(
            torch.from_numpy(transition_val).to(torch_device)
        )
        with torch.no_grad():
            val_logits = model(*val_tensors).detach().cpu().numpy()
        val_prob = _sigmoid(val_logits)
        metrics = _classification_metrics(
            prepared["train_labels"][val_idx],
            val_prob,
        )
        val_loss = float(
            criterion(
                torch.from_numpy(val_logits),
                torch.from_numpy(prepared["train_labels"][val_idx].astype(np.float32)),
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

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    val_tensors = [
        torch.from_numpy(np.ascontiguousarray(x_val[name])).to(torch_device)
        for name in ("frame_a", "temporal_a", "frame_b", "temporal_b", "au")
    ]
    val_tensors.append(torch.from_numpy(face_val).to(torch_device))
    val_tensors.append(
        torch.from_numpy(transition_val).to(torch_device)
    )
    test_tensors = [
        torch.from_numpy(np.ascontiguousarray(test_features[name])).to(torch_device)
        for name in ("frame_a", "temporal_a", "frame_b", "temporal_b", "au")
    ]
    test_tensors.append(torch.from_numpy(face_test).to(torch_device))
    test_tensors.append(
        torch.from_numpy(transition_test).to(torch_device)
    )
    with torch.no_grad():
        val_logits = model(*val_tensors).detach().cpu().numpy()
        test_logits = model(*test_tensors).detach().cpu().numpy()
    temperature = _fit_temperature(
        val_logits,
        prepared["train_labels"][val_idx],
    )
    test_prob = _sigmoid(test_logits / max(temperature, 1e-6))
    headline, confusion = _headline(
        prepared["test_labels"],
        test_prob,
        total_count=prepared["test_total"],
    )

    checkpoint = {
        "model_type": V4_MODEL_TYPE,
        "model_state": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "base_stats": {
            name: value.astype(np.float32)
            for name, value in stats.items()
        },
        "face_mean": face_mean,
        "face_scale": face_scale,
        "transition_mean": transition_mean,
        "transition_scale": transition_scale,
        "temperature": float(temperature),
        "config": {
            "face_geometry_dim": FACE_GEOMETRY_DIM,
            "face_hidden": FACE_HIDDEN,
            "transition_dim": TRANSITION_DIM,
            "transition_hidden": TRANSITION_HIDDEN,
            "modality_dropout": float(modality_dropout),
            "fusion_mode": (
                "v3_temporal_au_plus_face_geometry_plus_transition_windows"
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


def _load_model(path: Path) -> tuple[FaceGeometryAwareClassifier, dict[str, Any]]:
    checkpoint = torch.load(str(path), map_location="cpu")
    if checkpoint.get("model_type") != V4_MODEL_TYPE:
        raise ValueError(f"Unsupported v4 model: {checkpoint.get('model_type')}")
    model = FaceGeometryAwareClassifier(
        modality_dropout=float(
            checkpoint.get("config", {}).get("modality_dropout", 0.10)
        )
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def predict_wangxing_v4(
    *,
    video_path: Path,
    au_path: Path,
    model_path: Path,
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
) -> dict[str, Any]:
    from wangxing_project.joint_au_pt_v3 import (
        _extract_sequence,
    )
    model, checkpoint = _load_model(model_path)
    from evaluator.vedio_pred.wangxing_dual_pt import SCALE_A, SCALE_B
    frame_a, temporal_a = _extract_sequence(
        video_path,
        num_frames=int(SCALE_A["num_frames"]),
        frame_size=int(SCALE_A["frame_size"]),
    )
    frame_b, temporal_b = _extract_sequence(
        video_path,
        num_frames=int(SCALE_B["num_frames"]),
        frame_size=int(SCALE_B["frame_size"]),
    )
    from evaluator.modules.forensics.learned_fusion_head import (
        extract_fusion_features,
    )
    au_values, _ = extract_fusion_features(
        source_profile=source_profile,
        forensics_profiles=forensics_profiles,
        au_path=au_path,
    )
    base = {
        "frame_a": frame_a[None, ...],
        "temporal_a": temporal_a[None, ...],
        "frame_b": frame_b[None, ...],
        "temporal_b": temporal_b[None, ...],
        "au": np.asarray(au_values, dtype=np.float32)[None, ...],
    }
    stats = checkpoint["base_stats"]
    normalized = {
        name: np.clip(
            (value - stats[f"{name}_mean"])
            / np.maximum(stats[f"{name}_scale"], 1e-4),
            -8.0,
            8.0,
        ).astype(np.float32)
        for name, value in base.items()
    }
    face = extract_face_geometry_features(au_path)[None, :]
    face = np.clip(
        (face - checkpoint["face_mean"])
        / np.maximum(checkpoint["face_scale"], 1e-4),
        -8.0,
        8.0,
    ).astype(np.float32)
    transition = transition_feature_vector(
        video_path=video_path,
        au_path=au_path,
    )[None, :]
    transition = np.clip(
        (transition - checkpoint["transition_mean"])
        / np.maximum(checkpoint["transition_scale"], 1e-4),
        -8.0,
        8.0,
    ).astype(np.float32)
    tensors = [
        torch.from_numpy(normalized[name])
        for name in ("frame_a", "temporal_a", "frame_b", "temporal_b", "au")
    ]
    tensors.append(torch.from_numpy(face))
    tensors.append(torch.from_numpy(transition))
    with torch.no_grad():
        logit = float(model(*tensors).item())
    probability = float(
        1.0
        / (
            1.0
            + math.exp(
                -logit / max(float(checkpoint.get("temperature", 1.0)), 1e-6)
            )
        )
    )
    return {
        "prediction": "generated" if probability >= 0.5 else "real",
        "generated_probability": probability,
        "real_probability": 1.0 - probability,
        "model_path": str(model_path),
        "fusion_mode": (
            "v3_temporal_au_plus_face_geometry_plus_transition_windows"
        ),
    }


def evaluate_holdout_v4(
    *,
    holdout_manifest: Path,
    model_path: Path,
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
) -> dict[str, Any]:
    holdout = json.loads(
        Path(holdout_manifest).read_text(encoding="utf-8-sig")
    )
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    probabilities: list[float] = []
    samples: list[tuple[int, str, dict[str, Any]]] = []
    samples.extend((0, "real", item) for item in holdout.get("real", []))
    samples.extend((1, "generated", item) for item in holdout.get("seedance", []))
    for index, (label, source_label, item) in enumerate(samples, start=1):
        video = project_path(str(item["video"]))
        au = resolve_au_csv_for_video(video, au_hint=item.get("au"))
        if au is None or not video.is_file():
            rows.append(
                {
                    "index": index,
                    "source_label": source_label,
                    "label_generated": label,
                    "status": "missing_inputs",
                    "video": str(video),
                    "au": None if au is None else str(au),
                }
            )
            continue
        result = predict_wangxing_v4(
            video_path=video,
            au_path=au,
            model_path=model_path,
            source_profile=source_profile,
            forensics_profiles=forensics_profiles,
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
    headline, confusion = _headline(
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float32),
        total_count=len(samples),
    )
    return {
        "schema_version": "wangxing_face_geometry_v4_holdout_metrics_v1",
        "model_path": str(model_path),
        "holdout_manifest": str(holdout_manifest),
        "headline": headline,
        "confusion": confusion,
        "rows": rows,
    }
