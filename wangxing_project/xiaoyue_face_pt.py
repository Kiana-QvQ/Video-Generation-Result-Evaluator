"""Face-only PT classifier for the isolated XiaoYue experiment.

The model has no full-frame RGB/HSV input. Its main branches consume
pose-normalized facial geometry/AU trajectories and a separate mouth
trajectory. Local crop features are grayscale, brightness-centered, and
anchored to landmarks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from evaluator.modules.core.paths import project_path
from evaluator.vedio_pred.real_video_detector import _set_seed

from .xiaoyue_face_features import (
    FACE_SEQUENCE_DIM,
    MOUTH_SEQUENCE_DIM,
    build_feature_table,
)

MODEL_TYPE = "xiaoyue_face_mouth_temporal_v1"
HIDDEN_DIM = 48
# The bidirectional GRU uses HIDDEN_DIM // 2 per direction, so its
# concatenated output is HIDDEN_DIM rather than HIDDEN_DIM * 2.
EMBED_DIM = HIDDEN_DIM


def _device(value: str) -> torch.device:
    requested = str(value or "cuda").strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return resolved


def _headline(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict[str, float | None], dict[str, int]]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    predictions = (probabilities >= 0.5).astype(np.int64)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return (
        {
            "generated_recall": tp / (tp + fn) if tp + fn else None,
            "overall_accuracy": (tp + tn) / len(labels) if len(labels) else None,
            "generated_precision": tp / (tp + fp) if tp + fp else None,
            "real_recall": tn / (tn + fp) if tn + fp else None,
            "coverage": 1.0,
        },
        {
            "tp_generated": tp,
            "tn_real": tn,
            "fp_real_as_generated": fp,
            "fn_generated_as_real": fn,
        },
    )


class _TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.convolution = nn.Sequential(
            nn.Conv1d(input_dim, HIDDEN_DIM, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.gru = nn.GRU(
            input_size=HIDDEN_DIM,
            hidden_size=HIDDEN_DIM // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(EMBED_DIM, 1)
        self.output = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
            nn.GELU(),
            nn.Dropout(0.10),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence = self.input_norm(sequence)
        convolved = self.convolution(sequence.transpose(1, 2)).transpose(1, 2)
        encoded, _ = self.gru(convolved)
        weights = torch.softmax(self.attention(encoded), dim=1)
        pooled = torch.sum(encoded * weights, dim=1)
        return self.output(pooled)


class XiaoYueFaceMouthClassifier(nn.Module):
    """Two-branch face classifier with an explicitly weighted mouth head."""

    def __init__(self) -> None:
        super().__init__()
        self.face_encoder = _TemporalEncoder(FACE_SEQUENCE_DIM)
        self.mouth_encoder = _TemporalEncoder(MOUTH_SEQUENCE_DIM)
        self.face_head = nn.Linear(EMBED_DIM, 1)
        self.mouth_head = nn.Linear(EMBED_DIM, 1)
        self.fusion = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        face: torch.Tensor,
        mouth: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        face_embedding = self.face_encoder(face)
        mouth_embedding = self.mouth_encoder(mouth)
        face_logit = self.face_head(face_embedding).squeeze(1)
        mouth_logit = self.mouth_head(mouth_embedding).squeeze(1)
        fusion_logit = self.fusion(
            torch.cat([face_embedding, mouth_embedding], dim=1)
        ).squeeze(1)
        # Keep mouth as a first-class signal instead of letting a global
        # fusion layer silently ignore it on the tiny subject-specific set.
        joint = (
            0.45 * fusion_logit
            + 0.20 * face_logit
            + 0.35 * mouth_logit
        )
        if not return_aux:
            return joint
        return joint, face_logit, mouth_logit


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _split_groups(
    labels: np.ndarray,
    groups: Sequence[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    validation: set[str] = set()
    for label in (0, 1):
        candidates = sorted(
            {
                str(group)
                for group, current in zip(groups, labels)
                if int(current) == label
            }
        )
        if len(candidates) < 2:
            raise ValueError(
                "Face-only training needs at least two groups per class."
            )
        validation.add(str(rng.choice(candidates)))
    val_idx = np.asarray(
        [index for index, group in enumerate(groups) if str(group) in validation],
        dtype=np.int64,
    )
    fit_idx = np.asarray(
        [index for index, group in enumerate(groups) if str(group) not in validation],
        dtype=np.int64,
    )
    return fit_idx, val_idx


def _fit_stats(
    face: np.ndarray,
    mouth: np.ndarray,
    fit_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "face_mean": np.mean(face[fit_idx], axis=(0, 1)).astype(np.float32),
        "face_scale": np.maximum(
            np.std(face[fit_idx], axis=(0, 1)),
            1e-3,
        ).astype(np.float32),
        "mouth_mean": np.mean(mouth[fit_idx], axis=(0, 1)).astype(np.float32),
        "mouth_scale": np.maximum(
            np.std(mouth[fit_idx], axis=(0, 1)),
            1e-3,
        ).astype(np.float32),
    }


def _normalize(
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return np.clip(
        (values - mean.reshape(1, 1, -1)) / scale.reshape(1, 1, -1),
        -8.0,
        8.0,
    ).astype(np.float32)


def _sample_payload(
    manifest: dict[str, Any],
    table: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if "pairs" not in manifest:
        raise ValueError("Face-only training manifest must contain pairs.")
    train = [
        *list(manifest["pairs"].get("train", {}).get("real") or []),
        *list(manifest["pairs"].get("train", {}).get("fake") or []),
    ]
    test = [
        *list(manifest["pairs"].get("test", {}).get("real") or []),
        *list(manifest["pairs"].get("test", {}).get("fake") or []),
    ]
    available = table["features"]
    for item in [*train, *test]:
        video = str(project_path(str(item["video"])).resolve())
        if video not in available:
            raise ValueError(f"Missing face features for manifest video: {video}")
    return train, test


def _arrays(
    items: Sequence[dict[str, Any]],
    table: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    face: list[np.ndarray] = []
    mouth: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    paths: list[str] = []
    for item in items:
        video = str(project_path(str(item["video"])).resolve())
        record = table["features"][video]
        face.append(record["face"])
        mouth.append(record["mouth"])
        labels.append(int(item.get("label_generated", 0)))
        groups.append(str(item.get("group_id") or video))
        paths.append(video)
    return (
        np.stack(face).astype(np.float32),
        np.stack(mouth).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        groups,
        paths,
    )


def _predict(
    model: XiaoYueFaceMouthClassifier,
    face: np.ndarray,
    mouth: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        joint, face_logit, mouth_logit = model(
            torch.from_numpy(face).to(device),
            torch.from_numpy(mouth).to(device),
            return_aux=True,
        )
    logits = torch.stack([joint, face_logit, mouth_logit], dim=1)
    probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    return probabilities[:, 0], probabilities[:, 1], probabilities[:, 2]


def train_xiaoyue_face(
    *,
    manifest: dict[str, Any],
    cache_path: Path,
    model_path: Path,
    metrics_path: Path,
    epochs: int = 80,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    seed: int = 42,
    device: str = "cuda",
) -> dict[str, Any]:
    torch_device = _device(device)
    _set_seed(seed)
    table = build_feature_table(manifest, cache_path=cache_path)
    train_items, test_items = _sample_payload(manifest, table)
    face, mouth, labels, groups, train_paths = _arrays(train_items, table)
    test_face, test_mouth, test_labels, _, test_paths = _arrays(
        test_items,
        table,
    )
    fit_idx, val_idx = _split_groups(labels, groups, seed)
    stats = _fit_stats(face, mouth, fit_idx)
    normalized_face = _normalize(face, stats["face_mean"], stats["face_scale"])
    normalized_mouth = _normalize(
        mouth,
        stats["mouth_mean"],
        stats["mouth_scale"],
    )
    normalized_test_face = _normalize(
        test_face,
        stats["face_mean"],
        stats["face_scale"],
    )
    normalized_test_mouth = _normalize(
        test_mouth,
        stats["mouth_mean"],
        stats["mouth_scale"],
    )
    model = XiaoYueFaceMouthClassifier().to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=1e-3,
    )
    class_counts = np.bincount(labels[fit_idx], minlength=2)
    sample_weights = np.asarray(
        [1.0 / max(float(class_counts[label]), 1.0) for label in labels[fit_idx]],
        dtype=np.float64,
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized_face[fit_idx]),
            torch.from_numpy(normalized_mouth[fit_idx]),
            torch.from_numpy(labels[fit_idx].astype(np.float32)),
        ),
        batch_size=max(1, int(batch_size)),
        sampler=WeightedRandomSampler(
            torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        ),
    )
    criterion = nn.BCEWithLogitsLoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_key = (-1.0, float("inf"))
    history: list[dict[str, Any]] = []
    val_face = normalized_face[val_idx]
    val_mouth = normalized_mouth[val_idx]
    val_labels = labels[val_idx]
    for epoch in range(max(1, int(epochs))):
        model.train()
        losses: list[float] = []
        for batch_face, batch_mouth, target in loader:
            batch_face = batch_face.to(torch_device)
            batch_mouth = batch_mouth.to(torch_device)
            target = target.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            joint, face_logit, mouth_logit = model(
                batch_face,
                batch_mouth,
                return_aux=True,
            )
            loss = (
                0.70 * criterion(joint, target)
                + 0.10 * criterion(face_logit, target)
                + 0.20 * criterion(mouth_logit, target)
            )
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        val_joint, _, _ = _predict(
            model,
            val_face,
            val_mouth,
            torch_device,
        )
        val_headline, _ = _headline(val_labels, val_joint)
        val_loss = float(
            criterion(
                torch.from_numpy(
                    np.log(np.clip(val_joint, 1e-6, 1.0 - 1e-6))
                    - np.log(np.clip(1.0 - val_joint, 1e-6, 1.0))
                ),
                torch.from_numpy(val_labels.astype(np.float32)),
            )
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else None,
                "validation_loss": val_loss,
                "validation": val_headline,
            }
        )
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"[face PT] epoch {epoch + 1}/{epochs} "
                f"train_loss={history[-1]['train_loss']} "
                f"val_acc={val_headline['overall_accuracy']}",
                flush=True,
            )
        key = (
            float(val_headline["overall_accuracy"] or 0.0),
            val_loss,
        )
        if key[0] > best_key[0] or (
            key[0] == best_key[0] and key[1] < best_key[1]
        ):
            best_key = key
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    test_joint, test_face_probability, test_mouth_probability = _predict(
        model,
        normalized_test_face,
        normalized_test_mouth,
        torch_device,
    )
    headline, confusion = _headline(test_labels, test_joint)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": MODEL_TYPE,
            "config": {
                "face_sequence_dim": FACE_SEQUENCE_DIM,
                "mouth_sequence_dim": MOUTH_SEQUENCE_DIM,
                "hidden_dim": HIDDEN_DIM,
                "mouth_joint_weight": 0.35,
                "full_frame_features": False,
            },
            "model_state": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            "stats": stats,
            "feature_paths": train_paths,
        },
        model_path,
    )
    payload = {
        "schema_version": "xiaoyue_face_mouth_temporal_v1_metrics",
        "subject": "xiaoyue",
        "model_path": str(model_path.resolve()),
        "headline": headline,
        "confusion": confusion,
        "counts": {
            "train_real": int((labels == 0).sum()),
            "train_ai": int((labels == 1).sum()),
            "test_real": int((test_labels == 0).sum()),
            "test_ai": int((test_labels == 1).sum()),
        },
        "validation": {
            "fit_count": int(len(fit_idx)),
            "validation_count": int(len(val_idx)),
            "fit_groups": sorted(
                {groups[index] for index in fit_idx.tolist()}
            ),
            "validation_groups": sorted(
                {groups[index] for index in val_idx.tolist()}
            ),
        },
        "architecture": (
            "pose-normalized AU/Face Mesh temporal branch + dedicated "
            "mouth geometry/AU/local-crop branch; no full-frame appearance"
        ),
        "test_rows": [
            {
                "video": path,
                "label_generated": int(label),
                "prediction": "generated" if probability >= 0.5 else "real",
                "generated_probability": float(probability),
                "face_branch_generated_probability": float(face_probability),
                "mouth_branch_generated_probability": float(mouth_probability),
            }
            for path, label, probability, face_probability, mouth_probability in zip(
                test_paths,
                test_labels,
                test_joint,
                test_face_probability,
                test_mouth_probability,
            )
        ],
        "feature_policy": {
            "background_used": False,
            "absolute_brightness_used": False,
            "full_frame_rgb_used": False,
            "mouth_priority": True,
        },
        "history": history,
        "device": str(torch_device),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def evaluate_xiaoyue_face(
    *,
    manifest: dict[str, Any],
    cache_path: Path,
    model_path: Path,
    output_path: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = XiaoYueFaceMouthClassifier()
    model.load_state_dict(checkpoint["model_state"])
    torch_device = _device(device)
    model.to(torch_device)
    table = build_feature_table(manifest, cache_path=cache_path)
    items = [
        *list(manifest.get("real") or []),
        *list(manifest.get("fake") or manifest.get("seedance") or []),
    ]
    face, mouth, labels, _, paths = _arrays(items, table)
    stats = checkpoint["stats"]
    face = _normalize(face, stats["face_mean"], stats["face_scale"])
    mouth = _normalize(mouth, stats["mouth_mean"], stats["mouth_scale"])
    joint, face_probability, mouth_probability = _predict(
        model,
        face,
        mouth,
        torch_device,
    )
    headline, confusion = _headline(labels, joint)
    payload = {
        "schema_version": "xiaoyue_face_mouth_temporal_v1_evaluation",
        "subject": "xiaoyue",
        "model_path": str(model_path.resolve()),
        "headline": headline,
        "confusion": confusion,
        "rows": [
            {
                "video": path,
                "label_generated": int(label),
                "prediction": "generated" if probability >= 0.5 else "real",
                "generated_probability": float(probability),
                "face_branch_generated_probability": float(face_score),
                "mouth_branch_generated_probability": float(mouth_score),
            }
            for path, label, probability, face_score, mouth_score in zip(
                paths,
                labels,
                joint,
                face_probability,
                mouth_probability,
            )
        ],
        "feature_policy": {
            "background_used": False,
            "absolute_brightness_used": False,
            "full_frame_rgb_used": False,
            "mouth_priority": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
