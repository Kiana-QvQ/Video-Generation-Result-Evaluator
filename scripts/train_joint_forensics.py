from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.forensics.joint_model import (
    MODALITIES,
    JointForensicsModel,
    load_feature_npz,
    multitask_loss,
)
from evaluator.modules.core.paths import project_path


def _device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available.")
        return torch.device("cuda")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _feature_path(
    record: dict[str, Any],
    *,
    feature_root: Path | None,
) -> Path | None:
    explicit = record.get("feature_path")
    if isinstance(explicit, str) and explicit.strip():
        return project_path(explicit)
    if feature_root is None:
        return None
    video_path = record.get("video_path")
    if not isinstance(video_path, str) or not video_path.strip():
        return None
    return feature_root / Path(video_path).with_suffix(".npz")


def _records_with_features(
    payload: dict[str, Any],
    *,
    feature_root: Path | None,
    split: str,
    max_records: int,
) -> list[dict[str, Any]]:
    records = [
        record
        for record in payload.get("records", [])
        if isinstance(record, dict)
        and record.get("split") == split
        and _feature_path(record, feature_root=feature_root) is not None
        and _feature_path(record, feature_root=feature_root).is_file()
    ]
    if max_records > 0:
        records = records[:max_records]
    if not records:
        raise SystemExit(
            "No feature files were found. Populate feature_path in the "
            "manifest or pass --feature-root."
        )
    return records


class JointFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        feature_root: Path | None,
        expression_classes: list[str],
    ) -> None:
        self.records = records
        self.feature_root = feature_root
        self.expression_index = {
            name: index for index, name in enumerate(expression_classes)
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = _feature_path(record, feature_root=self.feature_root)
        if path is None:
            raise ValueError(f"Missing feature path for record {index}.")
        modalities, frame_mask = load_feature_npz(path)
        labels: dict[str, float | int] = {
            "identity": float("nan"),
            "expression_support": float("nan"),
            "quality": float("nan"),
            "artifact": float("nan"),
            "expression": -1,
        }
        if record.get("identity_label") is not None:
            labels["identity"] = float(record["identity_label"])
        if record.get("expression_support_label") is not None:
            labels["expression_support"] = float(
                record["expression_support_label"]
            )
        if record.get("quality_label") is not None:
            labels["quality"] = float(record["quality_label"])
        if record.get("_artifact_label_disabled"):
            labels["artifact"] = float("nan")
        elif record.get("artifact_label") is not None:
            labels["artifact"] = float(record["artifact_label"])
        elif record.get("source_label") is not None:
            # This is source classification, not a universal artifact label.
            labels["artifact"] = float(record["source_label"])
        expression = record.get("expression_class")
        if expression in self.expression_index:
            labels["expression"] = self.expression_index[expression]
        return {
            "modalities": modalities,
            "frame_mask": frame_mask,
            "labels": labels,
        }


def _collate(
    items: list[dict[str, Any]],
    *,
    feature_dims: dict[str, int],
) -> dict[str, Any]:
    frame_count = max(
        max(
            (
                array.shape[0]
                for array in item["modalities"].values()
                if array is not None
            ),
            default=1,
        )
        for item in items
    )
    batch_size = len(items)
    modalities: dict[str, Tensor] = {}
    presence: dict[str, Tensor] = {}
    for name in MODALITIES:
        width = feature_dims[name]
        values = torch.zeros(batch_size, frame_count, width)
        available = torch.zeros(batch_size, frame_count)
        for row, item in enumerate(items):
            array = item["modalities"].get(name)
            if array is None:
                continue
            tensor = torch.as_tensor(array, dtype=torch.float32)
            length = min(frame_count, tensor.shape[0])
            if width and tensor.shape[1] != width:
                raise ValueError(
                    f"{name} feature width changed within the dataset."
                )
            if width:
                values[row, :length] = tensor[:length]
                available[row, :length] = 1.0
        modalities[name] = values
        presence[name] = available
    frame_mask = torch.zeros(batch_size, frame_count, dtype=torch.bool)
    for row, item in enumerate(items):
        supplied = item["frame_mask"]
        if supplied is None:
            length = max(
                (
                    array.shape[0]
                    for array in item["modalities"].values()
                    if array is not None
                ),
                default=1,
            )
            frame_mask[row, :length] = True
        else:
            mask = torch.as_tensor(supplied, dtype=torch.bool)
            frame_mask[row, : min(frame_count, mask.shape[0])] = mask[
                :frame_count
            ]
    labels: dict[str, Tensor] = {}
    for key in ("identity", "expression_support", "quality", "artifact"):
        labels[key] = torch.as_tensor(
            [item["labels"][key] for item in items],
            dtype=torch.float32,
        )
    labels["expression"] = torch.as_tensor(
        [item["labels"]["expression"] for item in items],
        dtype=torch.long,
    )
    return {
        "modalities": modalities,
        "modality_presence": presence,
        "frame_mask": frame_mask,
        "labels": labels,
    }


def _infer_dims(
    records: Iterable[dict[str, Any]],
    *,
    feature_root: Path | None,
) -> dict[str, int]:
    dims = {name: 0 for name in MODALITIES}
    for record in records:
        path = _feature_path(record, feature_root=feature_root)
        if path is None:
            continue
        modalities, _ = load_feature_npz(path)
        for name, array in modalities.items():
            if array is not None:
                if np.asarray(array).ndim != 2:
                    raise ValueError(
                        f"{name} features must have shape [frames, features]."
                    )
                dims[name] = max(dims[name], int(array.shape[1]))
    if sum(dims.values()) <= 0:
        raise SystemExit("Feature files contain no usable modalities.")
    return dims


def _disable_single_class_tasks(records: list[dict[str, Any]]) -> list[str]:
    def effective_label(record: dict[str, Any], key: str) -> Any:
        if key == "artifact_label":
            explicit = record.get("artifact_label")
            return (
                explicit
                if explicit is not None
                else record.get("source_label")
            )
        return record.get(key)

    disabled: list[str] = []
    for key in (
        "identity_label",
        "expression_support_label",
        "quality_label",
        "artifact_label",
    ):
        values = {
            value
            for record in records
            if (value := effective_label(record, key)) is not None
        }
        if len(values) >= 2:
            continue
        if not values:
            disabled.append(key)
            continue
        disabled.append(key)
        for record in records:
            if key == "artifact_label":
                record["_artifact_label_disabled"] = True
            else:
                record[key] = None
    return disabled


def _train_epoch(
    model: JointForensicsModel,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    batches = 0
    for batch in loader:
        modalities = {
            name: value.to(device)
            for name, value in batch["modalities"].items()
        }
        presence = {
            name: value.to(device)
            for name, value in batch["modality_presence"].items()
        }
        outputs = model(
            modalities,
            frame_mask=batch["frame_mask"].to(device),
            modality_presence=presence,
        )
        labels = {
            name: value.to(device)
            for name, value in batch["labels"].items()
        }
        losses = multitask_loss(outputs, labels)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        total += float(losses["total"].detach().cpu())
        batches += 1
    return total / max(batches, 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the small shared-input multi-head forensics model from "
            "pre-extracted NPZ features. This command never extracts CSV or "
            "video features itself."
        )
    )
    parser.add_argument(
        "--manifest",
        default="outputs/forensics/joint_forensics_manifest.json",
    )
    parser.add_argument("--feature-root")
    parser.add_argument(
        "--output",
        default="outputs/forensics/joint_forensics_model.pt",
    )
    parser.add_argument(
        "--split",
        default="profile_train",
        help="Only records with this manifest split are eligible for training.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    manifest_path = project_path(args.manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    expression_classes = list(payload.get("expression_classes", []))
    if len(expression_classes) < 2:
        raise SystemExit("Manifest must declare at least two expression classes.")
    feature_root = (
        project_path(args.feature_root) if args.feature_root else None
    )
    records = _records_with_features(
        payload,
        feature_root=feature_root,
        split=args.split,
        max_records=args.max_records,
    )
    disabled_tasks = _disable_single_class_tasks(records)
    if disabled_tasks:
        print(
            "WARNING: disabled single-class or unlabeled tasks: "
            + ", ".join(disabled_tasks)
        )
    dims = _infer_dims(records, feature_root=feature_root)
    dataset = JointFeatureDataset(
        records,
        feature_root=feature_root,
        expression_classes=expression_classes,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda items: _collate(items, feature_dims=dims),
    )
    model = JointForensicsModel(
        visual_dim=dims["visual"],
        facial_dim=dims["facial"],
        texture_dim=dims["texture"],
        audio_dim=dims["audio"],
        expression_classes=len(expression_classes),
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        loss = _train_epoch(model, loader, optimizer, device)
        print(f"epoch={epoch} loss={loss:.6f} device={device}")

    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "joint_forensics_model_v1",
            "feature_dims": dims,
            "expression_classes": expression_classes,
            "model_config": {
                "hidden_dim": args.hidden_dim,
                "expression_classes": len(expression_classes),
            },
            "trained_split": args.split,
            "record_count": len(records),
            "disabled_tasks": disabled_tasks,
            "state_dict": model.state_dict(),
        },
        output,
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
