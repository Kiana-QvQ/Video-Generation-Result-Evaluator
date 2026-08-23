"""PT v4.4 with explicit 85% expression / 15% face-crop fusion."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evaluator.modules.core.paths import project_path
from wangxing_project.joint_au_pt_v43 import (
    ExpressionFaceCropClassifier,
    V43_MODEL_TYPE,
    _load_model,
    _predict_loaded,
    _train_v43,
)
from wangxing_project.joint_au_pt_v41 import _prepare_expression_data

V44_MODEL_TYPE = "wangxing_expression_authenticity_v44"
EXPRESSION_WEIGHT = 0.85
CROP_WEIGHT = 0.15


def _logit_probability(probability: torch.Tensor) -> torch.Tensor:
    return torch.logit(torch.clamp(probability, 1e-5, 1.0 - 1e-5))


class MonotonicExpressionFaceCropClassifier(ExpressionFaceCropClassifier):
    """Keep the crop branch bounded instead of letting fusion reweight it."""

    def forward(
        self,
        sequence: torch.Tensor,
        summary: torch.Tensor,
        blendshape: torch.Tensor,
        crop_sequence: torch.Tensor,
        crop_summary: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, expression_logit, crop_logit = super().forward(
            sequence,
            summary,
            blendshape,
            crop_sequence,
            crop_summary,
            return_aux=True,
        )
        final_probability = (
            EXPRESSION_WEIGHT * torch.sigmoid(expression_logit)
            + CROP_WEIGHT * torch.sigmoid(crop_logit)
        )
        final_logit = _logit_probability(final_probability)
        if not return_aux:
            return final_logit
        return final_logit, expression_logit, crop_logit


def train_wangxing_v44(
    *,
    manifest: dict,
    cache_dir: Path,
    model_path: Path,
    epochs: int = 80,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    prepared = _prepare_expression_data(manifest)
    return _train_v43(
        prepared=prepared,
        cache_dir=cache_dir,
        model_path=model_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
        model_factory=MonotonicExpressionFaceCropClassifier,
        model_type=V44_MODEL_TYPE,
        fusion_mode=(
            "explicit_monotonic_85_percent_expression"
            "_plus_15_percent_face_crop"
        ),
    )


def _load_model_v44(
    path: Path,
) -> tuple[MonotonicExpressionFaceCropClassifier, dict]:
    model, checkpoint = _load_model(
        path,
        model_factory=MonotonicExpressionFaceCropClassifier,
        expected_model_type=V44_MODEL_TYPE,
    )
    return model, checkpoint


def predict_wangxing_v44(
    *,
    video_path: Path,
    au_path: Path,
    model_path: Path,
) -> dict:
    model, checkpoint = _load_model_v44(model_path)
    return _predict_loaded(
        model,
        checkpoint,
        video_path,
        au_path,
        model_path,
    )


def evaluate_holdout_v44(
    *,
    holdout_manifest: Path,
    model_path: Path,
) -> dict:
    model, checkpoint = _load_model_v44(model_path)
    holdout = json.loads(
        Path(holdout_manifest).read_text(encoding="utf-8-sig")
    )
    rows = []
    labels: list[int] = []
    probabilities: list[float] = []
    samples = [
        (0, "real", item) for item in holdout.get("real", [])
    ] + [
        (1, "generated", item)
        for item in holdout.get("seedance", [])
    ]
    for index, (label, source_label, item) in enumerate(samples, start=1):
        video = project_path(str(item["video"]))
        au = project_path(str(item["au"]))
        if not video.is_file() or not au.is_file():
            rows.append(
                {
                    "index": index,
                    "source_label": source_label,
                    "label_generated": label,
                    "status": "missing_inputs",
                    "video": str(video),
                    "au": str(au),
                }
            )
            continue
        result = _predict_loaded(
            model,
            checkpoint,
            video,
            au,
            model_path,
        )
        labels.append(label)
        probabilities.append(float(result["generated_probability"]))
        rows.append(
            {
                "index": index,
                "source_label": source_label,
                "label_generated": label,
                "status": "ok",
                "video": str(video),
                "au": str(au),
                **result,
            }
        )
        if index == 1 or index % 10 == 0 or index == len(samples):
            print(f"[v4.4 evaluate] {index}/{len(samples)}", flush=True)
    from wangxing_project.joint_au_pt_v41 import _headline_with_threshold

    headline, confusion = _headline_with_threshold(
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float32),
        total_count=len(samples),
        threshold=float(checkpoint.get("decision_threshold", 0.5)),
    )
    return {
        "schema_version": "wangxing_expression_authenticity_v44_metrics_v1",
        "model_path": str(model_path),
        "holdout_manifest": str(holdout_manifest),
        "headline": headline,
        "confusion": confusion,
        "rows": rows,
    }
