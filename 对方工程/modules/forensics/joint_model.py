"""Shared-input, multi-head model for Wang Xing video forensics.

The model intentionally consumes pre-extracted frame features. Feature
extraction is kept outside this module so training cannot accidentally
re-read the large AU/video corpus while another profile build is running.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

MODALITIES = ("visual", "facial", "texture", "audio")
HEADS = (
    "identity",
    "expression",
    "expression_support",
    "quality",
    "artifact",
)


def _as_tensor(value: Any, *, device: torch.device | None = None) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _masked_mean(values: Tensor, frame_mask: Tensor) -> Tensor:
    weights = frame_mask.to(dtype=values.dtype).unsqueeze(-1)
    total = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / total


class JointForensicsModel(nn.Module):
    """A small multi-task temporal model over pre-extracted frame features.

    The four input modalities can be supplied independently. Missing
    modalities are represented by zeros plus an explicit presence channel, so
    missing audio or unavailable Face Mesh data are not silently treated as
    measured values.
    """

    def __init__(
        self,
        *,
        visual_dim: int = 0,
        facial_dim: int = 0,
        texture_dim: int = 0,
        audio_dim: int = 0,
        expression_classes: int = 7,
        hidden_dim: int = 128,
        layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.1,
        max_frames: int = 256,
    ) -> None:
        super().__init__()
        self.feature_dims = {
            "visual": int(visual_dim),
            "facial": int(facial_dim),
            "texture": int(texture_dim),
            "audio": int(audio_dim),
        }
        if sum(self.feature_dims.values()) <= 0:
            raise ValueError("At least one input feature dimension is required.")
        if expression_classes < 2:
            raise ValueError("At least two expression classes are required.")
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads.")

        input_dim = sum(self.feature_dims.values()) + len(MODALITIES)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_frames, hidden_dim)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        self.max_frames = int(max_frames)
        self.expression_classes = int(expression_classes)
        self.heads = nn.ModuleDict(
            {
                "identity": nn.Linear(hidden_dim, 1),
                "expression": nn.Linear(hidden_dim, expression_classes),
                "expression_support": nn.Linear(hidden_dim, 1),
                "quality": nn.Linear(hidden_dim, 1),
                "artifact": nn.Linear(hidden_dim, 1),
            }
        )

    def _reference_shape(
        self,
        modalities: Mapping[str, Tensor | np.ndarray | None],
    ) -> tuple[int, int, torch.device]:
        for name in MODALITIES:
            value = modalities.get(name)
            if value is None:
                continue
            tensor = _as_tensor(value)
            if tensor.ndim != 3:
                raise ValueError(
                    f"{name} features must have shape [batch, frames, features]."
                )
            return tensor.shape[0], tensor.shape[1], tensor.device
        raise ValueError("At least one modality must be supplied.")

    def _modality_tensor(
        self,
        name: str,
        value: Tensor | np.ndarray | None,
        presence_value: Tensor | np.ndarray | None,
        *,
        batch_size: int,
        frame_count: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        width = self.feature_dims[name]
        if width <= 0:
            return (
                torch.zeros(
                    batch_size,
                    frame_count,
                    0,
                    dtype=torch.float32,
                    device=device,
                ),
                torch.zeros(
                    batch_size,
                    frame_count,
                    1,
                    dtype=torch.float32,
                    device=device,
                ),
            )
        if value is None:
            tensor = torch.zeros(
                batch_size,
                frame_count,
                width,
                dtype=torch.float32,
                device=device,
            )
        else:
            tensor = _as_tensor(value, device=device)
            if tensor.ndim != 3 or tensor.shape != (
                batch_size,
                frame_count,
                width,
            ):
                raise ValueError(
                    f"{name} features must have shape "
                    f"[{batch_size}, {frame_count}, {width}]."
                )
        if presence_value is None:
            present = torch.zeros(
                batch_size,
                frame_count,
                1,
                dtype=torch.float32,
                device=device,
            ) if value is None else torch.ones(
                batch_size,
                frame_count,
                1,
                dtype=torch.float32,
                device=device,
            )
        else:
            present_values = _as_tensor(
                presence_value,
                device=device,
            )
            if present_values.shape != (batch_size, frame_count):
                raise ValueError(
                    f"{name} presence must have shape "
                    f"[{batch_size}, {frame_count}]."
                )
            present = present_values.unsqueeze(-1).clamp(0.0, 1.0)
        return tensor, present

    def forward(
        self,
        modalities: Mapping[str, Tensor | np.ndarray | None],
        *,
        frame_mask: Tensor | np.ndarray | None = None,
        modality_presence: Mapping[
            str,
            Tensor | np.ndarray | None,
        ] | None = None,
    ) -> dict[str, Tensor]:
        batch_size, frame_count, device = self._reference_shape(modalities)
        if frame_count > self.max_frames:
            raise ValueError(
                f"Received {frame_count} frames but max_frames is "
                f"{self.max_frames}."
            )
        if frame_count <= 0:
            raise ValueError("At least one frame is required.")
        pieces: list[Tensor] = []
        presence: list[Tensor] = []
        for name in MODALITIES:
            features, available = self._modality_tensor(
                name,
                modalities.get(name),
                (
                    modality_presence.get(name)
                    if modality_presence is not None
                    else None
                ),
                batch_size=batch_size,
                frame_count=frame_count,
                device=device,
            )
            pieces.append(features)
            presence.append(available)
        inputs = torch.cat([*pieces, *presence], dim=-1)
        encoded = self.input_projection(inputs)
        encoded = encoded + self.position_embedding[:, :frame_count]

        if frame_mask is None:
            mask = torch.ones(
                batch_size,
                frame_count,
                dtype=torch.bool,
                device=device,
            )
        else:
            mask = _as_tensor(frame_mask, device=device).bool()
            if mask.shape != (batch_size, frame_count):
                raise ValueError(
                    "frame_mask must have shape [batch, frames]."
                )
        if not torch.all(mask.any(dim=1)):
            raise ValueError(
                "Every sample must contain at least one valid frame."
            )
        encoded = self.temporal_encoder(
            encoded,
            src_key_padding_mask=~mask,
        )
        pooled = _masked_mean(encoded, mask)
        return {
            "identity_logit": self.heads["identity"](pooled).squeeze(-1),
            "expression_logits": self.heads["expression"](pooled),
            "expression_support_logit": self.heads[
                "expression_support"
            ](pooled).squeeze(-1),
            "quality_logit": self.heads["quality"](pooled).squeeze(-1),
            "artifact_logit": self.heads["artifact"](pooled).squeeze(-1),
        }

    @staticmethod
    def probabilities(outputs: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Convert logits to user-facing probabilities."""
        return {
            "identity_0_1": torch.sigmoid(outputs["identity_logit"]),
            "expression_distribution": torch.softmax(
                outputs["expression_logits"],
                dim=-1,
            ),
            "expression_support_0_1": torch.sigmoid(
                outputs["expression_support_logit"]
            ),
            "quality_0_1": torch.sigmoid(outputs["quality_logit"]),
            "artifact_0_1": torch.sigmoid(outputs["artifact_logit"]),
        }


def multitask_loss(
    outputs: Mapping[str, Tensor],
    labels: Mapping[str, Tensor | np.ndarray | None],
    *,
    task_weights: Mapping[str, float] | None = None,
) -> dict[str, Tensor]:
    """Compute losses only for labels that are actually present.

    Binary labels use NaN as a missing-value marker. Expression class labels
    use -1 as a missing-value marker. This lets source, expression, and
    quality annotations arrive at different times without inventing labels.
    """
    weights = {
        "identity": 1.0,
        "expression": 1.0,
        "expression_support": 1.0,
        "quality": 1.0,
        "artifact": 1.0,
    }
    if task_weights:
        weights.update(
            {key: float(value) for key, value in task_weights.items()}
        )

    zero = next(iter(outputs.values())).sum() * 0.0
    losses: dict[str, Tensor] = {}

    def binary_loss(
        task: str,
        output_key: str,
        label_key: str,
    ) -> None:
        value = labels.get(label_key)
        if value is None:
            losses[task] = zero
            return
        target = _as_tensor(value, device=outputs[output_key].device)
        valid = torch.isfinite(target)
        if not torch.any(valid):
            losses[task] = zero
            return
        raw = nn.functional.binary_cross_entropy_with_logits(
            outputs[output_key][valid],
            target[valid],
            reduction="mean",
        )
        losses[task] = raw * weights[task]

    binary_loss("identity", "identity_logit", "identity")
    binary_loss(
        "expression_support",
        "expression_support_logit",
        "expression_support",
    )
    binary_loss("quality", "quality_logit", "quality")
    binary_loss("artifact", "artifact_logit", "artifact")

    expression = labels.get("expression")
    if expression is None:
        losses["expression"] = zero
    else:
        target = _as_tensor(
            expression,
            device=outputs["expression_logits"].device,
        ).long()
        valid = target >= 0
        losses["expression"] = (
            nn.functional.cross_entropy(
                outputs["expression_logits"][valid],
                target[valid],
            )
            * weights["expression"]
            if torch.any(valid)
            else zero
        )
    losses["total"] = sum(losses.values(), zero)
    return losses


def load_feature_npz(
    path: str | Path,
) -> tuple[dict[str, np.ndarray | None], np.ndarray | None]:
    """Load the stable feature-file contract used by the training script."""
    with np.load(path, allow_pickle=False) as payload:
        modalities: dict[str, np.ndarray | None] = {}
        for name in MODALITIES:
            modalities[name] = (
                np.asarray(payload[name], dtype=np.float32)
                if name in payload
                else None
            )
        frame_mask = (
            np.asarray(payload["frame_mask"], dtype=np.bool_)
            if "frame_mask" in payload
            else None
        )
    return modalities, frame_mask
