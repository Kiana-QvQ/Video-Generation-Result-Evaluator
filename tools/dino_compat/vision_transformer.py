"""Compatibility constructors for VBench's DINO ViT-B/16 loader."""

from __future__ import annotations

from typing import Any

from timm.models.vision_transformer import VisionTransformer


def vit_base(
    patch_size: int = 16,
    num_classes: int = 0,
    **kwargs: Any,
) -> VisionTransformer:
    """Return a ViT with the parameter names used by DINO's checkpoint."""
    kwargs.pop("drop_path_rate", None)
    return VisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        num_classes=num_classes,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        **kwargs,
    )


def dino_vitb16(**kwargs: Any) -> VisionTransformer:
    return vit_base(patch_size=16, **kwargs)
