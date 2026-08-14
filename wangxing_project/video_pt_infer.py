"""Project-side inference for Wang Xing dual-scale .pt (no peer host edits)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluator.vedio_pred.wangxing_dual_pt import (
    DUAL_MODEL_TYPE,
    DualScaleClassifier,
    extract_dual_feature,
)
from evaluator.vedio_pred.real_video_detector import _predict_classifier_logits
from wangxing_project.multi_scale_pt import (
    MULTI_MODEL_TYPE,
    predict_wangxing_multi_scale_pt,
)


def predict_dual_pt(video_path: str | Path, model_path: str | Path) -> dict[str, Any]:
    video_path = Path(video_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")

    try:
        checkpoint = torch.load(
            str(model_path),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(str(model_path), map_location="cpu")

    model_type = checkpoint.get("model_type")
    if model_type == MULTI_MODEL_TYPE:
        return predict_wangxing_multi_scale_pt(video_path, model_path)
    if model_type != DUAL_MODEL_TYPE:
        raise ValueError(f"Unsupported model_type: {model_type}")
    scales = checkpoint["config"]["scales"]
    feature = extract_dual_feature(
        video_path,
        scale_a=scales[0],
        scale_b=scales[1],
    )
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
    normalized = np.clip((feature - mean) / np.maximum(scale, 1e-4), -8.0, 8.0)
    model = DualScaleClassifier(input_dim=int(checkpoint["config"]["input_dim"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    logit = float(_predict_classifier_logits(model, normalized[None, :])[0])
    temperature = float(checkpoint.get("temperature", 1.0))
    p_gen = float(1.0 / (1.0 + math.exp(-logit / max(temperature, 1e-6))))
    p_gen = min(0.98, max(0.02, p_gen))
    decision = "generated" if p_gen >= 0.5 else "real"
    return {
        "prediction": decision,
        "generated_probability": round(p_gen, 4),
        "real_probability": round(1.0 - p_gen, 4),
        "logit": logit,
        "temperature": temperature,
        "model_path": str(model_path),
        "video_path": str(video_path),
    }
