"""Reserved model slots for this project (no download; deploy later).

Peer ``evaluator/checkpoints`` is only a placeholder folder on our side.
Frame-Audit / VLM judges use ``model_cache/vlm_judge/`` when present.
Wang Xing video ``.pt`` lives under ``outputs/vedio_pred/models/`` during
project validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Display / config registry only — paths may be missing until server deploy.
MODEL_SLOTS: dict[str, dict[str, Any]] = {
    "vlm_judge_qwen2_vl_2b_awq": {
        "family": "vlm_judge",
        "label": "Qwen2-VL-2B-Instruct-AWQ",
        "path": PROJECT_ROOT / "model_cache" / "vlm_judge" / "Qwen2-VL-2B-Instruct-AWQ",
        "enabled_when_present": True,
        "note": "Project Frame-Audit judge slot; skip if VRAM insufficient.",
    },
    "vlm_judge_qwen25_vl_3b_awq": {
        "family": "vlm_judge",
        "label": "Qwen2.5-VL-3B-Instruct-AWQ",
        "path": PROJECT_ROOT
        / "model_cache"
        / "vlm_judge"
        / "Qwen2.5-VL-3B-Instruct-AWQ",
        "enabled_when_present": True,
        "note": "Optional upgrade slot; not required for Wang Xing hard detect.",
    },
    "wangxing_dual_scale_pt": {
        "family": "wangxing_video_pt",
        "label": "WangXing dual-scale real/fake .pt",
        "path": PROJECT_ROOT
        / "outputs"
        / "vedio_pred"
        / "models"
        / "wangxing_dual_scale_classifier.pt",
        "enabled_when_present": True,
        "note": "Project-validated Wang Xing authenticity branch.",
    },
}


def list_model_slots() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot_id, meta in MODEL_SLOTS.items():
        path = Path(meta["path"])
        present = path.is_file() or path.is_dir()
        rows.append(
            {
                "id": slot_id,
                "family": meta["family"],
                "label": meta["label"],
                "path": str(path),
                "present": present,
                "selectable": bool(meta.get("enabled_when_present")) and present,
                "note": meta.get("note", ""),
            }
        )
    return rows


def resolve_wangxing_pt_path(override: str | Path | None = None) -> Path | None:
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path if path.is_file() else None
    default = Path(MODEL_SLOTS["wangxing_dual_scale_pt"]["path"])
    return default if default.is_file() else None
