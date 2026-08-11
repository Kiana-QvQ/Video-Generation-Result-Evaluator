from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import resolve_profile
from .runtime import PROJECT_ROOT


DEFAULT_JUDGE_ID = "qwen2_vl_2b_awq"
PROFILE_PATH = (
    resolve_profile(
        "model_profile.json",
        "config/model_profile.json",
    )
    or (PROJECT_ROOT / "config" / "model_profile.json")
)

MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "qwen2_vl_2b_awq": {
        "label": "Qwen2-VL-2B-Instruct-AWQ",
        "role": "default_etva_judge",
        "disk_gb": 2.74,
        "minimum_vram_gb": 8,
        "status": "recommended",
        "reason": "Already cached and safe for serial inference on an 8GB GPU.",
    },
    "qwen2_5_vl_3b_awq": {
        "label": "Qwen2.5-VL-3B-Instruct-AWQ",
        "role": "quality_upgrade",
        "disk_gb": 3.42,
        "minimum_vram_gb": 12,
        "status": "optional",
        "reason": "Better VLM capacity, but it is not the 8GB default.",
    },
    "videoscore2_bf16": {
        "label": "TIGER-Lab/VideoScore2 BF16",
        "role": "large_quality_upgrade",
        "disk_gb": 16.6,
        "minimum_vram_gb": 24,
        "status": "optional",
        "reason": "Use only on a larger GPU or after independently validating a quantized build.",
    },
}


def get_recommended_model(vram_gb: float | None = None) -> dict[str, Any]:
    """Return the safest useful judge for the requested hardware budget."""
    configured = load_profile()
    configured_modules = {
        str(item.get("id")): item
        for item in configured.get("modules", [])
        if isinstance(item, dict) and item.get("id")
    }
    configured_modules.setdefault(
        "qwen2_vl_2b_awq",
        configured_modules.get("vlm_judge", {}),
    )
    configured_modules.setdefault(
        "videoscore2_bf16",
        configured_modules.get("videoscore2", {}),
    )
    candidates = [
        profile_id
        for profile_id in MODEL_PROFILES
        if profile_id in configured_modules
    ]
    if not candidates:
        candidates = list(MODEL_PROFILES)

    def minimum_vram(profile_id: str) -> float:
        configured_value = configured_modules.get(profile_id, {}).get(
            "minimum_vram_gb"
        )
        if configured_value is not None:
            try:
                return float(configured_value)
            except (TypeError, ValueError):
                pass
        return float(MODEL_PROFILES[profile_id]["minimum_vram_gb"])

    if vram_gb is None:
        selected_id = min(candidates, key=minimum_vram)
    else:
        fitting = [
            profile_id
            for profile_id in candidates
            if minimum_vram(profile_id) <= vram_gb
        ]
        selected_id = max(
            fitting or candidates,
            key=minimum_vram,
        )
    if vram_gb is not None and minimum_vram(selected_id) > vram_gb:
        selected_id = min(candidates, key=minimum_vram)

    selected = dict(MODEL_PROFILES[selected_id])
    selected.update(
        {
            key: value
            for key, value in configured_modules.get(selected_id, {}).items()
            if key in {"label", "minimum_vram_gb", "disk_budget_gb", "backend"}
        }
    )
    selected["id"] = selected_id
    selected["configured_profile"] = configured.get("name", "fallback")
    selected["observed_vram_gb"] = vram_gb
    selected["selection_policy"] = (
        "8GB: Qwen2-VL-2B AWQ; 12GB+: Qwen2.5-VL-3B AWQ; "
        "24GB+: VideoScore2 BF16."
    )
    return selected


def load_profile() -> dict[str, Any]:
    """Load the checked-in compact profile without making it a hard dependency."""
    try:
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"name": "balanced_8gb", "default_judge": DEFAULT_JUDGE_ID}
