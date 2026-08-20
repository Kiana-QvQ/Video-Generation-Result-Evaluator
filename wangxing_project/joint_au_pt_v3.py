"""Train-only domain-generalization temporal AU + video detector.

v3 keeps frame-level video tokens instead of flattening the 24/8 frame views:
- per-frame handcrafted features: 96 dimensions;
- adjacent-frame temporal features: 6 dimensions;
- shared bidirectional GRU + attention pooling for each scale;
- AU-conditioned video gate and joint/video/AU auxiliary heads.

The module is parallel to the v1/v2 models and never reads Change clips unless
the caller explicitly passes a manifest containing them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from evaluator.modules.core.paths import project_path
from evaluator.modules.forensics.learned_fusion_head import (
    extract_fusion_features,
)
from evaluator.vedio_pred.real_video_detector import (
    FEATURE_VERSION,
    _classification_metrics,
    _file_signature,
    _fit_temperature,
    _frame_feature,
    _read_sampled_frames,
    _set_seed,
    _standardize,
)
from evaluator.vedio_pred.wangxing_dual_pt import (
    SCALE_A,
    SCALE_B,
)
from wangxing_project.joint_au_pt import (
    AU_DIM,
    _extract_au_matrix,
    is_forbidden_train_video,
    resolve_au_csv_for_video,
    resolve_torch_device,
)

V3_MODEL_TYPE = "wangxing_temporal_au_video_v3"
FRAME_DIM = 96
TEMPORAL_DIM = 6
TEMPORAL_HIDDEN = 128
VIDEO_DIM = TEMPORAL_HIDDEN * 2
AU_HIDDEN = 64
FUSION_HIDDEN = 128
FUSION_BOTTLENECK = 64
DEFAULT_MODALITY_DROPOUT = 0.10
DEFAULT_AUX_LOSS_WEIGHTS = {
    "joint": 0.80,
    "video": 0.10,
    "au": 0.10,
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _extract_sequence(
    video_path: Path,
    *,
    num_frames: int,
    frame_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    frames = _read_sampled_frames(
        video_path=video_path,
        num_frames=num_frames,
        frame_size=frame_size,
    )
    frame_features = np.stack(
        [_frame_feature(frame) for frame in frames],
        axis=0,
    ).astype(np.float32)
    temporal_features: list[np.ndarray] = []
    previous_gray: np.ndarray | None = None
    for frame in frames:
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            / 255.0
        )
        if previous_gray is not None:
            difference = np.abs(gray - previous_gray)
            temporal_features.append(
                np.asarray(
                    [
                        float(np.mean(difference)),
                        float(np.std(difference)),
                        float(np.percentile(difference, 95)),
                        float(np.mean(difference > 0.12)),
                        float(np.mean(np.abs(np.diff(difference, axis=0)))),
                        float(np.mean(np.abs(np.diff(difference, axis=1)))),
                    ],
                    dtype=np.float32,
                )
            )
        previous_gray = gray
    temporal = np.stack(temporal_features, axis=0).astype(np.float32)
    return frame_features, temporal


def _sequence_signature(path: Path) -> str:
    return _file_signature(path)


def build_sequence_table(
    paths: list[Path],
    *,
    cache_path: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], list[Path], list[str]]:
    """Extract or load both fixed-length temporal feature views."""
    cache_path = Path(cache_path)
    wanted = [str(path.resolve()) for path in paths]
    signatures = [_sequence_signature(path) for path in paths]
    if cache_path.is_file():
        try:
            with np.load(str(cache_path), allow_pickle=False) as payload:
                cached_paths = [str(value) for value in payload["paths"].tolist()]
                cached_signatures = [
                    str(value)
                    for value in payload["signatures"].tolist()
                ]
                if cached_paths == wanted and cached_signatures == signatures:
                    rows = {
                        path: {
                            "frame_a": payload["frame_a"][index].astype(
                                np.float32
                            ),
                            "temporal_a": payload["temporal_a"][index].astype(
                                np.float32
                            ),
                            "frame_b": payload["frame_b"][index].astype(
                                np.float32
                            ),
                            "temporal_b": payload["temporal_b"][index].astype(
                                np.float32
                            ),
                        }
                        for index, path in enumerate(cached_paths)
                    }
                    return rows, [Path(path) for path in cached_paths], []
        except (KeyError, OSError, ValueError):
            pass

    rows: dict[str, dict[str, np.ndarray]] = {}
    errors: list[str] = []
    ordered_valid: list[Path] = []
    for index, path in enumerate(paths, start=1):
        try:
            frame_a, temporal_a = _extract_sequence(
                path,
                num_frames=int(SCALE_A["num_frames"]),
                frame_size=int(SCALE_A["frame_size"]),
            )
            frame_b, temporal_b = _extract_sequence(
                path,
                num_frames=int(SCALE_B["num_frames"]),
                frame_size=int(SCALE_B["frame_size"]),
            )
            key = str(path.resolve())
            rows[key] = {
                "frame_a": frame_a,
                "temporal_a": temporal_a,
                "frame_b": frame_b,
                "temporal_b": temporal_b,
            }
            ordered_valid.append(path)
        except Exception as exc:  # noqa: BLE001 - retain batch progress
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
        if index % 10 == 0 or index == len(paths):
            print(f"[v3 feature] {index}/{len(paths)}", flush=True)

    if rows:
        keys = [str(path.resolve()) for path in ordered_valid]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_path),
            paths=np.asarray(keys),
            signatures=np.asarray(
                [_sequence_signature(path) for path in ordered_valid]
            ),
            frame_a=np.stack([rows[key]["frame_a"] for key in keys]),
            temporal_a=np.stack([rows[key]["temporal_a"] for key in keys]),
            frame_b=np.stack([rows[key]["frame_b"] for key in keys]),
            temporal_b=np.stack([rows[key]["temporal_b"] for key in keys]),
            feature_version=np.asarray([FEATURE_VERSION]),
        )
    return rows, ordered_valid, errors


def _group_split(
    labels: np.ndarray,
    group_ids: list[str],
    base_labels: np.ndarray,
    *,
    seed: int,
    validation_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold complete source groups out of validation."""
    labels = np.asarray(labels, dtype=np.int64)
    base_labels = np.asarray(base_labels, dtype=np.int64)
    if not (
        len(labels) == len(group_ids) == len(base_labels)
        and len(np.unique(labels)) >= 2
    ):
        raise ValueError("Invalid v3 group split inputs.")

    group_to_indices: dict[str, list[int]] = {}
    group_to_base: dict[str, int] = {}
    for index, group_id in enumerate(group_ids):
        group_to_indices.setdefault(str(group_id), []).append(index)
        group_to_base.setdefault(str(group_id), int(base_labels[index]))
    groups_by_label: dict[int, list[str]] = {0: [], 1: []}
    for group_id, label in group_to_base.items():
        groups_by_label.setdefault(label, []).append(group_id)

    rng = np.random.default_rng(seed)
    validation_groups: set[str] = set()
    for label in (0, 1):
        groups = sorted(groups_by_label.get(label, []))
        if len(groups) < 2:
            raise ValueError(
                "Need at least two source groups per base class for validation."
            )
        count = max(1, int(round(len(groups) * validation_ratio)))
        count = min(count, len(groups) - 1)
        selected = rng.choice(groups, size=count, replace=False)
        validation_groups.update(str(value) for value in selected)

    val_idx = np.asarray(
        [
            index
            for index, group_id in enumerate(group_ids)
            if str(group_id) in validation_groups
        ],
        dtype=np.int64,
    )
    fit_idx = np.asarray(
        [
            index
            for index, group_id in enumerate(group_ids)
            if str(group_id) not in validation_groups
        ],
        dtype=np.int64,
    )
    if len(np.unique(labels[fit_idx])) < 2 or len(np.unique(labels[val_idx])) < 2:
        raise ValueError("Group split lost one of the target classes.")
    return fit_idx, val_idx


class TemporalBranch(nn.Module):
    """Shared temporal adapter for one frame/temporal feature scale."""

    def __init__(
        self,
        *,
        max_frames: int,
        hidden: int = TEMPORAL_HIDDEN,
    ) -> None:
        super().__init__()
        self.frame_projection = nn.Linear(FRAME_DIM, hidden)
        self.temporal_projection = nn.Linear(TEMPORAL_DIM, hidden)
        self.position = nn.Parameter(torch.zeros(1, max_frames, hidden))
        self.scale_embedding = nn.Embedding(2, hidden)
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(hidden, 1)
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.10),
        )

    def forward(
        self,
        frame_features: torch.Tensor,
        temporal_features: torch.Tensor,
        *,
        scale_id: int,
    ) -> torch.Tensor:
        batch, frame_count, _ = frame_features.shape
        padded_temporal = torch.zeros(
            batch,
            frame_count,
            TEMPORAL_DIM,
            dtype=frame_features.dtype,
            device=frame_features.device,
        )
        padded_temporal[:, 1:, :] = temporal_features
        scale_token = self.scale_embedding.weight[int(scale_id)].view(
            1,
            1,
            -1,
        )
        tokens = (
            self.frame_projection(frame_features)
            + self.temporal_projection(padded_temporal)
            + self.position[:, :frame_count, :]
            + scale_token
        )
        encoded, _ = self.gru(tokens)
        weights = torch.softmax(self.attention(encoded), dim=1)
        pooled = torch.sum(encoded * weights, dim=1)
        return self.output(pooled)


class TemporalAUVideoClassifier(nn.Module):
    """Temporal dual-scale video encoder with AU-conditioned fusion."""

    def __init__(
        self,
        *,
        au_dim: int = AU_DIM,
        modality_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.au_dim = int(au_dim)
        self.modality_dropout = float(modality_dropout)
        self.temporal_branch = TemporalBranch(max_frames=24)
        self.video_projection = nn.Sequential(
            nn.Linear(TEMPORAL_HIDDEN * 2, VIDEO_DIM),
            nn.LayerNorm(VIDEO_DIM),
            nn.GELU(),
        )
        self.au_encoder = nn.Sequential(
            nn.Linear(self.au_dim, AU_HIDDEN),
            nn.LayerNorm(AU_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.au_gate = nn.Sequential(
            nn.Linear(AU_HIDDEN, VIDEO_DIM),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(VIDEO_DIM + AU_HIDDEN, FUSION_HIDDEN),
            nn.LayerNorm(FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(FUSION_HIDDEN, FUSION_BOTTLENECK),
            nn.LayerNorm(FUSION_BOTTLENECK),
            nn.GELU(),
            nn.Linear(FUSION_BOTTLENECK, 1),
        )
        self.video_head = nn.Linear(VIDEO_DIM, 1)
        self.au_head = nn.Linear(AU_HIDDEN, 1)

    def _drop_modalities(
        self,
        video: torch.Tensor,
        au: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0:
            return video, au
        probability = min(max(self.modality_dropout, 0.0), 0.5)
        drop_video = (
            torch.rand(video.shape[0], 1, device=video.device)
            < probability
        )
        drop_au = (
            torch.rand(au.shape[0], 1, device=au.device)
            < probability
        )
        drop_au = drop_au & ~drop_video
        return (
            video.masked_fill(drop_video, 0.0),
            au.masked_fill(drop_au, 0.0),
        )

    def forward(
        self,
        frame_a: torch.Tensor,
        temporal_a: torch.Tensor,
        frame_b: torch.Tensor,
        temporal_b: torch.Tensor,
        au: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled_a = self.temporal_branch(
            frame_a,
            temporal_a,
            scale_id=0,
        )
        pooled_b = self.temporal_branch(
            frame_b,
            temporal_b,
            scale_id=1,
        )
        video = self.video_projection(torch.cat([pooled_a, pooled_b], dim=1))
        au_embedding = self.au_encoder(au)
        video, au_embedding = self._drop_modalities(video, au_embedding)
        gated_video = video * (0.5 + self.au_gate(au_embedding))
        joint = self.fusion(
            torch.cat([gated_video, au_embedding], dim=1)
        ).squeeze(1)
        if not return_aux:
            return joint
        return (
            joint,
            self.video_head(video).squeeze(1),
            self.au_head(au_embedding).squeeze(1),
        )


def _prepare_data(
    *,
    manifest: dict[str, Any],
    cache_dir: Path,
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
) -> dict[str, Any]:
    if "pairs" not in manifest:
        raise ValueError("v3 manifest must contain explicit pairs.")

    def _filter_train(
        pairs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        kept = [
            item
            for item in pairs
            if not is_forbidden_train_video(item["video"])
        ]
        if len(kept) != len(pairs):
            raise ValueError("Change clip was found in v3 training pairs.")
        return kept

    train_real = _filter_train(list(manifest["pairs"]["train"]["real"]))
    train_fake = _filter_train(list(manifest["pairs"]["train"]["fake"]))
    test_real = list(manifest["pairs"]["test"]["real"])
    test_fake = list(manifest["pairs"]["test"]["fake"])
    all_pairs = train_real + train_fake + test_real + test_fake
    all_videos = [project_path(str(item["video"])) for item in all_pairs]
    sequence_map, valid_videos, video_errors = build_sequence_table(
        all_videos,
        cache_path=Path(cache_dir) / "wangxing_v3_sequences.npz",
    )
    au_map, au_errors = _extract_au_matrix(
        all_pairs,
        source_profile=source_profile,
        forensics_profiles=forensics_profiles,
        cache_path=Path(cache_dir) / "wangxing_v3_au25.npz",
    )

    def _collect(
        pairs: list[dict[str, Any]],
        label: int,
    ) -> dict[str, Any]:
        rows: list[dict[str, np.ndarray]] = []
        labels: list[int] = []
        groups: list[str] = []
        base_labels: list[int] = []
        video_paths: list[str] = []
        au_paths: list[str] = []
        missing: list[str] = []
        for item in pairs:
            video = project_path(str(item["video"]))
            au_path = project_path(str(item["au"]))
            video_key = str(video.resolve())
            au_key = str(au_path.resolve())
            sequence = sequence_map.get(video_key)
            au_vector = au_map.get(au_key)
            if sequence is None or au_vector is None:
                missing.append(video_key)
                continue
            rows.append(
                {
                    **sequence,
                    "au": np.asarray(au_vector, dtype=np.float32),
                }
            )
            labels.append(int(label))
            groups.append(
                str(item.get("group_id") or video_key)
            )
            base_labels.append(int(item.get("base_label", label)))
            video_paths.append(str(video.resolve()))
            au_paths.append(str(au_path.resolve()))
        return {
            "rows": rows,
            "labels": np.asarray(labels, dtype=np.int64),
            "groups": groups,
            "base_labels": np.asarray(base_labels, dtype=np.int64),
            "video_paths": video_paths,
            "au_paths": au_paths,
            "missing": missing,
        }

    train = _collect(train_real, 0)
    fake = _collect(train_fake, 1)
    test_real_data = _collect(test_real, 0)
    test_fake_data = _collect(test_fake, 1)

    def _stack(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        if not items:
            raise RuntimeError("No valid v3 feature rows.")
        return {
            "frame_a": np.stack([item["frame_a"] for item in items]),
            "temporal_a": np.stack([item["temporal_a"] for item in items]),
            "frame_b": np.stack([item["frame_b"] for item in items]),
            "temporal_b": np.stack([item["temporal_b"] for item in items]),
            "au": np.stack([item["au"] for item in items]),
        }

    return {
        "train_features": _stack(train["rows"] + fake["rows"]),
        "train_labels": np.concatenate([train["labels"], fake["labels"]]),
        "train_groups": train["groups"] + fake["groups"],
        "train_base_labels": np.concatenate(
            [train["base_labels"], fake["base_labels"]]
        ),
        "train_video_paths": train["video_paths"] + fake["video_paths"],
        "train_au_paths": train["au_paths"] + fake["au_paths"],
        "test_features": _stack(
            test_real_data["rows"] + test_fake_data["rows"]
        ),
        "test_labels": np.concatenate(
            [test_real_data["labels"], test_fake_data["labels"]]
        ),
        "test_video_paths": (
            test_real_data["video_paths"] + test_fake_data["video_paths"]
        ),
        "test_au_paths": (
            test_real_data["au_paths"] + test_fake_data["au_paths"]
        ),
        "counts": {
            "train_real": len(train["labels"]),
            "train_fake": len(fake["labels"]),
            "test_real": len(test_real_data["labels"]),
            "test_fake": len(test_fake_data["labels"]),
            "missing": {
                "train_real": train["missing"],
                "train_fake": fake["missing"],
                "test_real": test_real_data["missing"],
                "test_fake": test_fake_data["missing"],
            },
            "video_extract_errors_preview": video_errors[:20],
            "au_extract_errors_preview": au_errors[:20],
        },
        "test_total": len(test_real) + len(test_fake),
    }


def _fit_sequence_stats(
    features: dict[str, np.ndarray],
    fit_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    stats: dict[str, np.ndarray] = {}
    for name in ("frame_a", "frame_b"):
        values = features[name][fit_idx].reshape(-1, FRAME_DIM)
        _, mean, scale = _standardize(values, values)
        stats[f"{name}_mean"] = mean
        stats[f"{name}_scale"] = scale
    for name in ("temporal_a", "temporal_b"):
        values = features[name][fit_idx].reshape(-1, TEMPORAL_DIM)
        _, mean, scale = _standardize(values, values)
        stats[f"{name}_mean"] = mean
        stats[f"{name}_scale"] = scale
    values = features["au"][fit_idx]
    _, mean, scale = _standardize(values, values)
    stats["au_mean"] = mean
    stats["au_scale"] = scale
    return stats


def _normalize_features(
    features: dict[str, np.ndarray],
    stats: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for name in ("frame_a", "frame_b", "temporal_a", "temporal_b", "au"):
        normalized[name] = np.clip(
            (
                features[name]
                - stats[f"{name}_mean"]
            )
            / np.maximum(stats[f"{name}_scale"], 1e-4),
            -8.0,
            8.0,
        ).astype(np.float32)
    return normalized


def _headline(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    total_count: int,
) -> tuple[dict[str, float | None], dict[str, int]]:
    labels = labels.astype(np.int64)
    predictions = (probabilities >= 0.5).astype(np.int64)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return (
        {
            "generated_recall": tp / (tp + fn) if tp + fn else None,
            "overall_accuracy": (
                (tp + tn) / len(labels) if len(labels) else None
            ),
            "generated_precision": tp / (tp + fp) if tp + fp else None,
            "real_recall": tn / (tn + fp) if tn + fp else None,
            "coverage": len(labels) / total_count if total_count else 0.0,
        },
        {
            "tp_generated": tp,
            "tn_real": tn,
            "fp_real_as_generated": fp,
            "fn_generated_as_real": fn,
        },
    )


def train_wangxing_v3(
    *,
    manifest: dict[str, Any],
    cache_dir: Path,
    model_path: Path,
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
    epochs: int = 80,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    seed: int = 42,
    device: str | torch.device | None = "cuda",
    modality_dropout: float = 0.10,
) -> dict[str, Any]:
    torch_device = resolve_torch_device(
        str(device) if device is not None else "cuda"
    )
    _set_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(
        f"Temporal v3 training device={torch_device} "
        f"({torch.cuda.get_device_name(torch_device) if torch_device.type == 'cuda' else 'cpu'}). "
        "Feature extraction stays on CPU.",
        flush=True,
    )

    prepared = _prepare_data(
        manifest=manifest,
        cache_dir=cache_dir,
        source_profile=source_profile,
        forensics_profiles=forensics_profiles,
    )
    fit_idx, val_idx = _group_split(
        prepared["train_labels"],
        prepared["train_groups"],
        prepared["train_base_labels"],
        seed=seed,
    )
    stats = _fit_sequence_stats(prepared["train_features"], fit_idx)
    train_features = _normalize_features(prepared["train_features"], stats)
    test_features = _normalize_features(prepared["test_features"], stats)
    labels = prepared["train_labels"]
    x_fit = {name: value[fit_idx] for name, value in train_features.items()}
    x_val = {name: value[val_idx] for name, value in train_features.items()}
    x_test = test_features
    y_fit = labels[fit_idx].astype(np.float32)
    y_val = labels[val_idx].astype(np.float32)
    y_test = prepared["test_labels"].astype(np.int64)

    model = TemporalAUVideoClassifier(
        modality_dropout=modality_dropout,
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=2e-4,
    )
    class_counts = np.bincount(y_fit.astype(np.int64), minlength=2)
    sample_weights = np.asarray(
        [
            1.0 / max(float(class_counts[int(label)]), 1.0)
            for label in y_fit
        ],
        dtype=np.float64,
    )
    tensors = [
        torch.from_numpy(np.ascontiguousarray(x_fit[name]))
        for name in ("frame_a", "temporal_a", "frame_b", "temporal_b", "au")
    ]
    tensors.append(torch.from_numpy(np.ascontiguousarray(y_fit)))
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=max(1, int(batch_size)),
        sampler=WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        ),
        pin_memory=torch_device.type == "cuda",
    )
    criterion = nn.BCEWithLogitsLoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_key = (-1.0, float("inf"))
    history: list[dict[str, Any]] = []
    aux_weights = DEFAULT_AUX_LOSS_WEIGHTS

    for epoch in range(max(1, int(epochs))):
        model.train()
        losses: list[float] = []
        for batch in loader:
            batch = [
                tensor.to(
                    torch_device,
                    non_blocking=torch_device.type == "cuda",
                )
                for tensor in batch
            ]
            *inputs, target = batch
            optimizer.zero_grad(set_to_none=True)
            joint, video_logit, au_logit = model(
                *inputs,
                return_aux=True,
            )
            loss = (
                aux_weights["joint"] * criterion(joint, target)
                + aux_weights["video"] * criterion(video_logit, target)
                + aux_weights["au"] * criterion(au_logit, target)
            )
            if not torch.isfinite(loss):
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_tensors = [
            torch.from_numpy(np.ascontiguousarray(x_val[name])).to(
                torch_device
            )
            for name in ("frame_a", "temporal_a", "frame_b", "temporal_b", "au")
        ]
        with torch.no_grad():
            val_logits = model(*val_tensors).detach().cpu().numpy()
        val_prob = 1.0 / (1.0 + np.exp(-val_logits))
        val_metrics = _classification_metrics(
            y_val.astype(np.int64),
            val_prob,
        )
        val_loss = float(
            criterion(
                torch.from_numpy(val_logits),
                torch.from_numpy(y_val),
            ).item()
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else math.inf,
                "validation_loss": val_loss,
                **val_metrics,
            }
        )
        if epoch == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == int(epochs):
            print(
                f"epoch {epoch + 1}/{epochs} "
                f"train_loss={history[-1]['train_loss']:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_bacc={val_metrics['balanced_accuracy']:.4f}",
                flush=True,
            )
        key = (val_metrics["balanced_accuracy"], -val_loss)
        if key > best_key:
            best_key = key
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    val_tensors = [
        torch.from_numpy(np.ascontiguousarray(x_val[name])).to(torch_device)
        for name in ("frame_a", "temporal_a", "frame_b", "temporal_b", "au")
    ]
    test_tensors = [
        torch.from_numpy(np.ascontiguousarray(x_test[name])).to(torch_device)
        for name in ("frame_a", "temporal_a", "frame_b", "temporal_b", "au")
    ]
    with torch.no_grad():
        val_logits = model(*val_tensors).detach().cpu().numpy()
        test_logits = model(*test_tensors).detach().cpu().numpy()
    temperature = _fit_temperature(val_logits, y_val)
    test_prob = 1.0 / (
        1.0 + np.exp(-test_logits / max(temperature, 1e-6))
    )
    headline, confusion = _headline(
        y_test,
        test_prob,
        total_count=prepared["test_total"],
    )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": V3_MODEL_TYPE,
        "feature_version": FEATURE_VERSION,
        "config": {
            "scales": [SCALE_A, SCALE_B],
            "frame_dim": FRAME_DIM,
            "temporal_dim": TEMPORAL_DIM,
            "temporal_hidden": TEMPORAL_HIDDEN,
            "video_dim": VIDEO_DIM,
            "au_dim": AU_DIM,
            "au_hidden": AU_HIDDEN,
            "fusion_hidden": FUSION_HIDDEN,
            "fusion_bottleneck": FUSION_BOTTLENECK,
            "modality_dropout": float(modality_dropout),
            "fusion_mode": "temporal_adapter_au_conditioned_gate",
            "auxiliary_heads": True,
        },
        "model_state": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "stats": {
            name: value.astype(np.float32)
            for name, value in stats.items()
        },
        "temperature": float(temperature),
        "device_used": str(torch_device),
        "aux_loss_weights": dict(aux_weights),
        "dataset": prepared["counts"],
        "validation": {
            "fit_count": int(len(fit_idx)),
            "validation_count": int(len(val_idx)),
            "fit_group_count": len(
                set(prepared["train_groups"][int(index)] for index in fit_idx)
            ),
            "validation_group_count": len(
                set(prepared["train_groups"][int(index)] for index in val_idx)
            ),
            "normalization_fit": "fit_groups_only",
        },
        "train_val_metrics_tail": history[-10:],
        "test_headline": headline,
        "confusion": confusion,
    }
    torch.save(checkpoint, model_path)
    return {
        "model_path": str(model_path),
        "headline": headline,
        "confusion": confusion,
        "counts": prepared["counts"],
        "validation": checkpoint["validation"],
        "temperature": float(temperature),
        "device": str(torch_device),
    }


def _model_from_checkpoint(
    checkpoint: dict[str, Any],
) -> TemporalAUVideoClassifier:
    config = checkpoint.get("config") or {}
    model = TemporalAUVideoClassifier(
        au_dim=int(config.get("au_dim", AU_DIM)),
        modality_dropout=float(config.get("modality_dropout", 0.10)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def predict_wangxing_v3(
    *,
    video_path: str | Path,
    au_path: str | Path,
    model_path: str | Path,
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = torch.load(
        str(Path(model_path).expanduser().resolve()),
        map_location="cpu",
    )
    if checkpoint.get("model_type") != V3_MODEL_TYPE:
        raise ValueError(f"Unsupported model_type: {checkpoint.get('model_type')}")
    video = Path(video_path).expanduser().resolve()
    au = Path(au_path).expanduser().resolve()
    frame_a, temporal_a = _extract_sequence(
        video,
        num_frames=int(SCALE_A["num_frames"]),
        frame_size=int(SCALE_A["frame_size"]),
    )
    frame_b, temporal_b = _extract_sequence(
        video,
        num_frames=int(SCALE_B["num_frames"]),
        frame_size=int(SCALE_B["frame_size"]),
    )
    au_vector, au_details = extract_fusion_features(
        au_path=au,
        wangxing_source_profile=source_profile,
        forensics_profiles=forensics_profiles,
    )
    raw = {
        "frame_a": frame_a[None, ...],
        "temporal_a": temporal_a[None, ...],
        "frame_b": frame_b[None, ...],
        "temporal_b": temporal_b[None, ...],
        "au": np.asarray(au_vector, dtype=np.float32)[None, ...],
    }
    stats = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in checkpoint["stats"].items()
    }
    normalized = _normalize_features(raw, stats)
    model = _model_from_checkpoint(checkpoint)
    tensors = [
        torch.from_numpy(normalized[name])
        for name in ("frame_a", "temporal_a", "frame_b", "temporal_b", "au")
    ]
    with torch.no_grad():
        logit = float(model(*tensors)[0].item())
    temperature = float(checkpoint.get("temperature", 1.0))
    probability = float(
        1.0
        / (1.0 + math.exp(-logit / max(temperature, 1e-6)))
    )
    return {
        "prediction": "generated" if probability >= 0.5 else "real",
        "generated_probability": probability,
        "real_probability": 1.0 - probability,
        "logit": logit,
        "temperature": temperature,
        "model_path": str(Path(model_path).resolve()),
        "video_path": str(video),
        "au_path": str(au),
        "fusion_mode": "temporal_adapter_au_conditioned_gate",
        "au_quality_min": float(au_details.get("quality_min", 0.5)),
    }


def evaluate_holdout_v3(
    *,
    holdout_manifest: Path,
    model_path: Path,
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
) -> dict[str, Any]:
    holdout = _load_json(holdout_manifest)
    samples: list[tuple[int, str, dict[str, Any]]] = []
    for item in holdout.get("real", []):
        samples.append((0, "real", item))
    for item in holdout.get("seedance", []):
        samples.append((1, "generated", item))
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    probabilities: list[float] = []
    for index, (label, source_label, item) in enumerate(samples, start=1):
        video = project_path(str(item["video"]))
        au = resolve_au_csv_for_video(video, au_hint=item.get("au"))
        if au is None or not video.is_file():
            rows.append(
                {
                    "index": index,
                    "source_label": source_label,
                    "label_generated": label,
                    "status": "missing_inputs",
                    "video": str(video),
                    "au": None if au is None else str(au),
                }
            )
            continue
        scored = predict_wangxing_v3(
            video_path=video,
            au_path=au,
            model_path=model_path,
            source_profile=source_profile,
            forensics_profiles=forensics_profiles,
        )
        labels.append(label)
        probabilities.append(float(scored["generated_probability"]))
        rows.append(
            {
                "index": index,
                "source_label": source_label,
                "label_generated": label,
                "status": "ok",
                "video": str(video),
                "au": str(au),
                **scored,
            }
        )
        print(
            f"[{index}/{len(samples)}] {source_label} "
            f"pred={scored['prediction']} "
            f"p_gen={scored['generated_probability']:.4f}",
            flush=True,
        )
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float32)
    headline, confusion = _headline(
        y,
        p,
        total_count=len(samples),
    )
    return {
        "schema_version": "wangxing_temporal_au_video_v3_holdout_metrics_v1",
        "model_path": str(model_path),
        "holdout_manifest": str(holdout_manifest),
        "headline": headline,
        "confusion": confusion,
        "rows": rows,
    }
