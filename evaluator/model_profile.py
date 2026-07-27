from __future__ import annotations

import json
from typing import Any

from .runtime import PROJECT_ROOT


DEFAULT_JUDGE_ID = "qwen2_vl_2b_awq"
PROFILE_PATH = PROJECT_ROOT / "model_profile_compact_9p6g.json"

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
    if vram_gb is not None and vram_gb >= 24:
        selected_id = "videoscore2_bf16"
    elif vram_gb is not None and vram_gb >= 12:
        selected_id = "qwen2_5_vl_3b_awq"
    else:
        selected_id = DEFAULT_JUDGE_ID
    selected = dict(MODEL_PROFILES[selected_id])
    selected["id"] = selected_id
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
