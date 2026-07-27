from __future__ import annotations

import os
import gc
from pathlib import Path
from typing import Any

import numpy as np

from .runtime import MODEL_CACHE_DIR


VICLIP_CHECKPOINT = MODEL_CACHE_DIR / "viclip" / "ViClip-InternVid-10M-FLT.pth"
VICLIP_FRAMES = 8
VICLIP_IMAGE_SIZE = 224
_MODEL_CACHE: dict[tuple[str, bool], Any] = {}


def clear_viclip_cache() -> None:
    """Release ViCLIP before another GPU-backed judge is used."""
    models = list(_MODEL_CACHE.values())
    _MODEL_CACHE.clear()
    for model in models:
        del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def viclip_enabled(device: str) -> bool:
    flag = os.environ.get("EVALUATOR_VICLIP_ENABLED", "auto").lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if not VICLIP_CHECKPOINT.exists():
        return False
    if flag in {"1", "true", "yes", "on"}:
        return True
    if device == "cuda":
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False
    return False


def _resolved_device(device: str) -> str:
    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
    return "cpu"


def _load_checkpoint() -> dict[str, Any]:
    import torch

    try:
        loaded = torch.load(
            str(VICLIP_CHECKPOINT),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        loaded = torch.load(
            str(VICLIP_CHECKPOINT),
            map_location="cpu",
            weights_only=True,
        )
    if isinstance(loaded, dict) and isinstance(loaded.get("model"), dict):
        return loaded["model"]
    if not isinstance(loaded, dict):
        raise ValueError("ViCLIP checkpoint does not contain a state dictionary.")
    return loaded


def _build_model(device: str, need_text: bool) -> Any:
    import torch
    from clip.model import LayerNorm, QuickGELU, Transformer
    from torch import nn

    class VisionEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = 1024
            self.conv1 = nn.Conv3d(
                3,
                width,
                (1, 14, 14),
                (1, 14, 14),
                (0, 0, 0),
                bias=False,
            )
            scale = width**-0.5
            self.class_embedding = nn.Parameter(scale * torch.randn(width))
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(257, width)
            )
            self.temporal_positional_embedding = nn.Parameter(
                torch.zeros(1, VICLIP_FRAMES, width)
            )
            self.ln_pre = LayerNorm(width)
            self.transformer = Transformer(width, 24, 16)
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(torch.empty(width, 768))

        def forward(self, video: torch.Tensor) -> torch.Tensor:
            batch, _, frame_count, _, _ = video.shape
            x = self.conv1(video)
            _, _, _, height, width = x.shape
            x = x.permute(0, 2, 3, 4, 1).reshape(
                batch * frame_count,
                height * width,
                -1,
            )
            class_tokens = self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0],
                1,
                x.shape[-1],
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([class_tokens, x], dim=1)
            x = x + self.positional_embedding.to(x.dtype)

            cls_tokens = x[:batch, :1, :]
            x = x[:, 1:]
            x = x.reshape(batch, frame_count, height * width, -1)
            x = x.permute(0, 2, 1, 3).reshape(
                batch * height * width,
                frame_count,
                -1,
            )
            if frame_count == 1:
                x = x + self.temporal_positional_embedding.mean(1)
            else:
                temporal = self.temporal_positional_embedding
                if frame_count != VICLIP_FRAMES:
                    temporal = torch.nn.functional.interpolate(
                        temporal.permute(0, 2, 1),
                        size=frame_count,
                        mode="linear",
                        align_corners=False,
                    ).permute(0, 2, 1)
                x = x + temporal.to(x.dtype)
            x = x.reshape(batch, height * width * frame_count, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = self.ln_post(x[0])
            return x @ self.proj

    class TextEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = 768
            self.transformer = Transformer(width, 12, 12)
            self.token_embedding = nn.Embedding(49408, width)
            self.positional_embedding = nn.Parameter(
                torch.empty(32, width)
            )
            self.ln_final = LayerNorm(width)
            self.text_projection = nn.Parameter(torch.empty(width, 768))

        def forward(self, text: torch.Tensor) -> torch.Tensor:
            x = self.token_embedding(text).to(self.positional_embedding.dtype)
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)]
            return x @ self.text_projection

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision_encoder = VisionEncoder()
            if need_text:
                self.text_encoder = TextEncoder()

        def encode_video(self, video: torch.Tensor) -> torch.Tensor:
            return self.vision_encoder(video)

        def encode_text(self, text: torch.Tensor) -> torch.Tensor:
            if not hasattr(self, "text_encoder"):
                raise RuntimeError("This ViCLIP instance was loaded without text.")
            return self.text_encoder(text)

    model = Model()
    checkpoint = _load_checkpoint()
    vision_state = {
        key.removeprefix("vision_encoder."): value
        for key, value in checkpoint.items()
        if key.startswith("vision_encoder.")
    }
    missing, unexpected = model.vision_encoder.load_state_dict(
        vision_state,
        strict=False,
    )
    if missing or unexpected:
        raise RuntimeError(
            f"ViCLIP vision checkpoint mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    if need_text:
        text_state = {
            key.removeprefix("text_encoder."): value
            for key, value in checkpoint.items()
            if key.startswith("text_encoder.")
        }
        missing, unexpected = model.text_encoder.load_state_dict(  # type: ignore[attr-defined]
            text_state,
            strict=False,
        )
        if missing or unexpected:
            raise RuntimeError(
                f"ViCLIP text checkpoint mismatch: missing={missing}, "
                f"unexpected={unexpected}"
            )
    del checkpoint

    resolved = _resolved_device(device)
    dtype = torch.float16 if resolved == "cuda" else torch.float32
    model = model.to(device=resolved, dtype=dtype).eval()
    return model


def _get_model(device: str, need_text: bool) -> Any:
    resolved = _resolved_device(device)
    if not need_text and (resolved, True) in _MODEL_CACHE:
        return _MODEL_CACHE[(resolved, True)]
    key = (resolved, need_text)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _build_model(resolved, need_text)
    return _MODEL_CACHE[key]


def _preprocess_frames(frames: list[np.ndarray]) -> Any:
    import torch
    from PIL import Image
    from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

    preprocess = Compose(
        [
            Resize(VICLIP_IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(VICLIP_IMAGE_SIZE),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    selected = _select_frames(frames)
    return torch.stack([preprocess(Image.fromarray(frame)) for frame in selected])


def _select_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    if not frames:
        raise ValueError("ViCLIP requires at least one video frame.")
    indices = np.rint(
        np.linspace(0, len(frames) - 1, VICLIP_FRAMES)
    ).astype(np.int64)
    return [frames[int(index)] for index in indices]


def _normalize(value: Any) -> Any:
    import torch

    return value / (value.norm(dim=-1, keepdim=True) + 1e-6)


def video_similarity(
    result_frames: list[np.ndarray],
    reference_frames: list[np.ndarray],
    device: str,
    need_text: bool = False,
) -> dict[str, Any]:
    import torch

    model = _get_model(device, need_text=need_text)
    resolved = _resolved_device(device)
    dtype = torch.float16 if resolved == "cuda" else torch.float32
    result = _preprocess_frames(result_frames).unsqueeze(0).permute(
        0,
        2,
        1,
        3,
        4,
    ).to(
        device=resolved,
        dtype=dtype,
    )
    reference = _preprocess_frames(reference_frames).unsqueeze(0).permute(
        0,
        2,
        1,
        3,
        4,
    ).to(
        device=resolved,
        dtype=dtype,
    )
    with torch.no_grad():
        result_embedding = _normalize(model.encode_video(result))
        reference_embedding = _normalize(model.encode_video(reference))
        score = float((result_embedding @ reference_embedding.T).item())
    score = float(max(0.0, min(1.0, (score + 1.0) / 2.0)))
    return {
        "score_0_1": score,
        "raw_cosine": score * 2.0 - 1.0,
        "device": resolved,
        "frames": VICLIP_FRAMES,
    }


def text_similarity(
    video_frames: list[np.ndarray],
    prompt: str,
    device: str,
) -> dict[str, Any]:
    import torch
    import clip

    model = _get_model(device, need_text=True)
    resolved = _resolved_device(device)
    dtype = torch.float16 if resolved == "cuda" else torch.float32
    video = _preprocess_frames(video_frames).unsqueeze(0).permute(
        0,
        2,
        1,
        3,
        4,
    ).to(
        device=resolved,
        dtype=dtype,
    )
    tokens = clip.tokenize(
        [prompt],
        context_length=32,
        truncate=True,
    ).to(resolved)
    with torch.no_grad():
        video_embedding = _normalize(model.encode_video(video))
        text_embedding = _normalize(model.encode_text(tokens))
        raw_cosine = float((video_embedding @ text_embedding.T).item())
    score = float(max(0.0, min(1.0, (raw_cosine + 1.0) / 2.0)))
    return {
        "score_0_1": score,
        "raw_cosine": raw_cosine,
        "device": resolved,
        "frames": VICLIP_FRAMES,
    }
