from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HardwarePolicy:
    tier: str
    requested_device: str
    resolved_device: str
    cuda_available: bool
    vram_gb: float | None
    serial_models: bool
    judge_model: str
    viclip_enabled_by_default: bool
    viclip_frames: int
    etva_frames: int
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload


def _env_vram() -> float | None:
    value = os.environ.get("EVALUATOR_GPU_MEMORY_GB", "").strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _cuda_info() -> tuple[bool, float | None]:
    forced_vram = _env_vram()
    try:
        import torch

        available = bool(torch.cuda.is_available())
        if not available:
            return False, forced_vram
        actual_vram = float(
            torch.cuda.get_device_properties(0).total_memory
            / (1024**3)
        )
        return True, forced_vram or actual_vram
    except Exception:
        return False, forced_vram


def resolve_policy(requested_device: str = "auto") -> HardwarePolicy:
    requested = requested_device.lower()
    cuda_available, vram_gb = _cuda_info()
    resolved = "cuda" if requested == "cuda" and cuda_available else "cpu"
    if requested == "auto" and cuda_available:
        resolved = "cuda"

    if not cuda_available or resolved == "cpu":
        tier = "cpu"
        judge_model = "qwen2_vl_2b_awq"
        viclip_default = False
        notes = (
            "CUDA is unavailable or CPU was requested.",
            "Use explicit CUDA for ViCLIP and VLM acceleration.",
        )
    elif vram_gb is not None and vram_gb < 10:
        tier = "compact_8gb"
        judge_model = "qwen2_vl_2b_awq"
        viclip_default = True
        notes = (
            "Keep all heavyweight models serial.",
            "Use Qwen2-VL-2B AWQ as the 8GB ETVA judge.",
            "Do not auto-start VideoScore2 on this tier.",
        )
    elif vram_gb is not None and vram_gb < 20:
        tier = "balanced_12gb"
        judge_model = "qwen2_5_vl_3b_awq"
        viclip_default = True
        notes = (
            "Keep the VLM judge and ViCLIP mutually exclusive.",
            "Qwen2.5-VL-3B AWQ is the preferred judge upgrade.",
        )
    else:
        tier = "full_24gb"
        judge_model = "videoscore2_bf16"
        viclip_default = True
        notes = (
            "VideoScore2 is an opt-in capability and still requires a verified backend.",
            "Never keep VideoScore2 and another VLM resident together.",
        )

    return HardwarePolicy(
        tier=tier,
        requested_device=requested,
        resolved_device=resolved,
        cuda_available=cuda_available,
        vram_gb=round(vram_gb, 2) if vram_gb is not None else None,
        serial_models=True,
        judge_model=judge_model,
        viclip_enabled_by_default=viclip_default,
        viclip_frames=8,
        etva_frames=4 if tier in {"cpu", "compact_8gb"} else 8,
        notes=notes,
    )
