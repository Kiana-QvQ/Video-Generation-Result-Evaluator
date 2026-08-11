from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.forensics.joint_model import (  # noqa: E402
    MODALITIES,
    JointForensicsModel,
    load_feature_npz,
)
from evaluator.modules.core.paths import project_path  # noqa: E402


def _device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available.")
        return torch.device("cuda")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one pre-extracted NPZ feature file with the isolated "
            "joint forensics model."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument(
        "--output",
        default="outputs/forensics/joint_forensics_report.json",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()

    device = _device(args.device)
    checkpoint_path = project_path(args.checkpoint)
    feature_path = project_path(args.features)
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    dims = checkpoint["feature_dims"]
    config = checkpoint.get("model_config", {})
    expression_classes = list(checkpoint.get("expression_classes", []))
    if len(expression_classes) < 2:
        raise SystemExit(
            "Checkpoint must contain at least two expression classes."
        )
    model = JointForensicsModel(
        visual_dim=int(dims.get("visual", 0)),
        facial_dim=int(dims.get("facial", 0)),
        texture_dim=int(dims.get("texture", 0)),
        audio_dim=int(dims.get("audio", 0)),
        expression_classes=len(expression_classes),
        hidden_dim=int(config.get("hidden_dim", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    modalities, frame_mask = load_feature_npz(feature_path)
    reference = next(
        (
            array
            for array in modalities.values()
            if array is not None
        ),
        None,
    )
    if reference is None:
        raise SystemExit("The NPZ feature file contains no modalities.")
    frame_count = reference.shape[0]
    batched_modalities: dict[str, Any] = {}
    presence: dict[str, Any] = {}
    for name in MODALITIES:
        array = modalities.get(name)
        if array is None:
            batched_modalities[name] = None
            presence[name] = None
            continue
        tensor = torch.as_tensor(array, dtype=torch.float32)
        batched_modalities[name] = tensor.unsqueeze(0).to(device)
        presence[name] = torch.ones(
            1,
            frame_count,
            dtype=torch.float32,
            device=device,
        )
    if frame_mask is None:
        mask = torch.ones(
            1,
            frame_count,
            dtype=torch.bool,
            device=device,
        )
    else:
        mask = torch.as_tensor(
            frame_mask,
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0)
    with torch.no_grad():
        outputs = model(
            batched_modalities,
            frame_mask=mask,
            modality_presence=presence,
        )
        probabilities = model.probabilities(outputs)
    expression_distribution = probabilities["expression_distribution"][0]
    top_expression = (
        expression_classes[int(torch.argmax(expression_distribution))]
        if expression_classes
        else None
    )
    report = {
        "schema_version": "joint_forensics_report_v1",
        "checkpoint": str(checkpoint_path),
        "features": str(feature_path),
        "outputs": {
            "identity_0_1": float(probabilities["identity_0_1"][0]),
            "expression_support_0_1": float(
                probabilities["expression_support_0_1"][0]
            ),
            "quality_0_1": float(probabilities["quality_0_1"][0]),
            "artifact_0_1": float(probabilities["artifact_0_1"][0]),
            "expression_distribution": {
                name: float(expression_distribution[index])
                for index, name in enumerate(expression_classes)
            },
            "top_expression": top_expression,
        },
        "disabled_tasks": checkpoint.get("disabled_tasks", []),
        "interpretation": (
            "Independent evidence heads; do not treat artifact probability "
            "as a quality penalty."
        ),
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["outputs"], ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
