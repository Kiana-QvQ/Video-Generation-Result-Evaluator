"""Offline torch.hub entrypoints for the DINO ViT-B/16 checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vision_transformer import vit_base


def _load_checkpoint(model: torch.nn.Module, checkpoint: Any) -> torch.nn.Module:
    if not checkpoint:
        return model
    checkpoint_path = Path(str(checkpoint)).expanduser()
    if not checkpoint_path.exists():
        return model
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        model.load_state_dict(state, strict=False)
    return model


def dino_vitb16(pretrained: bool = False, **kwargs: Any) -> torch.nn.Module:
    """Build DINO's ViT-B/16 shape without an online download."""
    checkpoint = kwargs.pop("checkpoint", None)
    checkpoint = kwargs.pop("checkpoint_path", checkpoint)
    checkpoint = kwargs.pop("path", checkpoint)
    kwargs.pop("source", None)
    model = vit_base(patch_size=16, num_classes=0, **kwargs)
    return _load_checkpoint(model, checkpoint) if pretrained else model
