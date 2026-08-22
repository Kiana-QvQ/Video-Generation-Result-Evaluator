"""PT v4.2 candidate: v4.1 expression sequence model with validation threshold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluator.modules.core.paths import project_path
from wangxing_project.joint_au_pt_v41 import (
    V41_MODEL_TYPE,
    _headline_with_threshold,
    _load_model,
    _predict_loaded_v41,
    evaluate_holdout_v41,
    train_wangxing_v41,
)

V42_MODEL_TYPE = "wangxing_expression_authenticity_v42"


def train_wangxing_v42(
    *,
    manifest: dict[str, Any],
    cache_dir: Path,
    model_path: Path,
    epochs: int = 80,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    seed: int = 42,
    device: str = "cuda",
) -> dict[str, Any]:
    return train_wangxing_v41(
        manifest=manifest,
        cache_dir=cache_dir,
        model_path=model_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
        model_type=V42_MODEL_TYPE,
    )


def predict_wangxing_v42(
    *,
    video_path: Path,
    au_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    model, checkpoint = _load_model(
        model_path,
        expected_model_type=V42_MODEL_TYPE,
    )
    return _predict_loaded_v41(
        model=model,
        checkpoint=checkpoint,
        video_path=video_path,
        au_path=au_path,
        model_path=model_path,
    )


def evaluate_holdout_v42(
    *,
    holdout_manifest: Path,
    model_path: Path,
) -> dict[str, Any]:
    holdout = json.loads(
        Path(holdout_manifest).read_text(encoding="utf-8-sig")
    )
    model, checkpoint = _load_model(
        model_path,
        expected_model_type=V42_MODEL_TYPE,
    )
    rows: list[dict[str, Any]] = []
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
        result = _predict_loaded_v41(
            model=model,
            checkpoint=checkpoint,
            video_path=video,
            au_path=au,
            model_path=model_path,
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
            print(f"[v4.2 evaluate] {index}/{len(samples)}", flush=True)
    headline, confusion = _headline_with_threshold(
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float32),
        total_count=len(samples),
        threshold=float(checkpoint.get("decision_threshold", 0.5)),
    )
    return {
        "schema_version": "wangxing_expression_authenticity_v42_metrics_v1",
        "model_path": str(model_path),
        "holdout_manifest": str(holdout_manifest),
        "headline": headline,
        "confusion": confusion,
        "rows": rows,
    }
