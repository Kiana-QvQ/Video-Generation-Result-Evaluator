"""Two-branch AU + dual-scale video fusion model.

This is a parallel successor to ``joint_au_pt.py``:
- video and AU features use separate encoders;
- AU features condition the video branch through a lightweight gate;
- joint, video-only, and AU-only heads are trained together;
- the existing v1 joint model and default dual model are untouched.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from evaluator.modules.core.paths import PROJECT_ROOT, project_path
from evaluator.vedio_pred.real_video_detector import (
    FEATURE_VERSION,
    _classification_metrics,
    _fit_temperature,
    _set_seed,
    _standardize,
)
from evaluator.vedio_pred.wangxing_dual_pt import (
    SCALE_A,
    SCALE_B,
    build_dual_feature_table,
)
from wangxing_project.joint_au_pt import (
    AU_DIM,
    _collect_joint_matrix,
    _extract_au_matrix,
    _predict_logits,
    _split_fit_validation,
    attach_au_pairs,
    is_forbidden_train_video,
    predict_wangxing_joint_au_pt,
    resolve_au_csv_for_video,
    resolve_torch_device,
)

JOINT_V2_MODEL_TYPE = "wangxing_joint_au_dual_pt_v2"
VIDEO_DIM = 3252
VIDEO_HIDDEN = 256
AU_HIDDEN = 64
FUSION_HIDDEN = 128
FUSION_BOTTLENECK = 64
DEFAULT_MODALITY_DROPOUT = 0.10
DEFAULT_AUX_LOSS_WEIGHTS = {
    "joint": 0.80,
    "video": 0.10,
    "au": 0.10,
}


class JointAUVideoClassifier(nn.Module):
    """AU-conditioned two-branch classifier with auxiliary modality heads."""

    def __init__(
        self,
        *,
        video_dim: int = VIDEO_DIM,
        au_dim: int = AU_DIM,
        video_hidden: int = VIDEO_HIDDEN,
        au_hidden: int = AU_HIDDEN,
        fusion_hidden: int = FUSION_HIDDEN,
        fusion_bottleneck: int = FUSION_BOTTLENECK,
        modality_dropout: float = DEFAULT_MODALITY_DROPOUT,
    ) -> None:
        super().__init__()
        self.video_dim = int(video_dim)
        self.au_dim = int(au_dim)
        self.video_hidden = int(video_hidden)
        self.au_hidden = int(au_hidden)
        self.modality_dropout = float(modality_dropout)

        self.video_encoder = nn.Sequential(
            nn.Linear(self.video_dim, self.video_hidden),
            nn.LayerNorm(self.video_hidden),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.au_encoder = nn.Sequential(
            nn.Linear(self.au_dim, self.au_hidden),
            nn.LayerNorm(self.au_hidden),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.au_gate = nn.Sequential(
            nn.Linear(self.au_hidden, self.video_hidden),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.video_hidden + self.au_hidden, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(fusion_hidden, fusion_bottleneck),
            nn.LayerNorm(fusion_bottleneck),
            nn.GELU(),
            nn.Linear(fusion_bottleneck, 1),
        )
        self.video_head = nn.Linear(self.video_hidden, 1)
        self.au_head = nn.Linear(self.au_hidden, 1)

    def _apply_modality_dropout(
        self,
        video: torch.Tensor,
        au: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0.0:
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
        # Do not remove both modalities from the same sample.
        drop_au = drop_au & ~drop_video
        return (
            video.masked_fill(drop_video, 0.0),
            au.masked_fill(drop_au, 0.0),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video_input = inputs[:, : self.video_dim]
        au_input = inputs[:, self.video_dim : self.video_dim + self.au_dim]
        video = self.video_encoder(video_input)
        au = self.au_encoder(au_input)
        video, au = self._apply_modality_dropout(video, au)

        # Keep a residual 0.5 scale so an uncertain AU gate cannot erase video.
        gated_video = video * (0.5 + self.au_gate(au))
        joint_logit = self.fusion(
            torch.cat([gated_video, au], dim=1)
        ).squeeze(1)
        if not return_aux:
            return joint_logit
        video_logit = self.video_head(video).squeeze(1)
        au_logit = self.au_head(au).squeeze(1)
        return joint_logit, video_logit, au_logit


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _predict_joint_logits(
    model: JointAUVideoClassifier,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    array = np.asarray(features, dtype=np.float32)
    with torch.no_grad():
        tensor = torch.from_numpy(array).to(device)
        logits = model(tensor)
    return logits.detach().cpu().numpy().astype(np.float32)


def _model_from_checkpoint(
    checkpoint: dict[str, Any],
) -> JointAUVideoClassifier:
    config = checkpoint.get("config") or {}
    model = JointAUVideoClassifier(
        video_dim=int(config.get("video_dim", VIDEO_DIM)),
        au_dim=int(config.get("au_dim", AU_DIM)),
        video_hidden=int(config.get("video_hidden", VIDEO_HIDDEN)),
        au_hidden=int(config.get("au_hidden", AU_HIDDEN)),
        fusion_hidden=int(config.get("fusion_hidden", FUSION_HIDDEN)),
        fusion_bottleneck=int(
            config.get("fusion_bottleneck", FUSION_BOTTLENECK)
        ),
        modality_dropout=float(
            config.get("modality_dropout", DEFAULT_MODALITY_DROPOUT)
        ),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model


def _prepare_data(
    *,
    manifest: dict[str, Any],
    cache_dir: Path,
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
) -> dict[str, Any]:
    if "pairs" not in manifest:
        raise ValueError("Manifest is missing pairs; attach AU pairs first.")

    def _filter_train(
        pairs: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], list[str]]:
        kept: list[dict[str, str]] = []
        dropped: list[str] = []
        for item in pairs:
            if is_forbidden_train_video(item["video"]):
                dropped.append(item["video"])
            else:
                kept.append(item)
        return kept, dropped

    train_real, dropped_real = _filter_train(
        list(manifest["pairs"]["train"]["real"])
    )
    train_fake, dropped_fake = _filter_train(
        list(manifest["pairs"]["train"]["fake"])
    )
    test_real = list(manifest["pairs"]["test"]["real"])
    test_fake = list(manifest["pairs"]["test"]["fake"])
    all_pairs = train_real + train_fake + test_real + test_fake
    all_videos = [Path(item["video"]) for item in all_pairs]

    video_matrix, valid_videos, video_errors = build_dual_feature_table(
        all_videos,
        cache_path=Path(cache_dir) / "wangxing_dual_f24s1024_f8s2048.npz",
    )
    video_map = {
        str(path.resolve()): video_matrix[index]
        for index, path in enumerate(valid_videos)
    }
    au_map, au_errors = _extract_au_matrix(
        all_pairs,
        source_profile=source_profile,
        forensics_profiles=forensics_profiles,
        cache_path=Path(cache_dir) / "wangxing_joint_au25.npz",
    )

    x_tr, y_tr, miss_tr, keys_tr = _collect_joint_matrix(
        train_real,
        0,
        video_features=video_map,
        au_features=au_map,
    )
    x_tf, y_tf, miss_tf, keys_tf = _collect_joint_matrix(
        train_fake,
        1,
        video_features=video_map,
        au_features=au_map,
    )
    x_er, y_er, miss_er, _ = _collect_joint_matrix(
        test_real,
        0,
        video_features=video_map,
        au_features=au_map,
    )
    x_ef, y_ef, miss_ef, _ = _collect_joint_matrix(
        test_fake,
        1,
        video_features=video_map,
        au_features=au_map,
    )
    x_train = np.concatenate([x_tr, x_tf], axis=0)
    y_train = np.concatenate([y_tr, y_tf], axis=0)
    x_test = np.concatenate([x_er, x_ef], axis=0)
    y_test = np.concatenate([y_er, y_ef], axis=0)
    if x_train.ndim != 2 or x_train.shape[1] != VIDEO_DIM + AU_DIM:
        raise RuntimeError(
            f"Expected joint train dim {VIDEO_DIM + AU_DIM}, "
            f"got {x_train.shape}"
        )
    if x_test.ndim != 2 or x_test.shape[1] != VIDEO_DIM + AU_DIM:
        raise RuntimeError(
            f"Expected joint test dim {VIDEO_DIM + AU_DIM}, "
            f"got {x_test.shape}"
        )
    if len(y_train) < 8 or len(np.unique(y_train)) < 2:
        raise RuntimeError("Joint v2 training data needs both classes.")
    if len(y_test) < 4 or len(np.unique(y_test)) < 2:
        raise RuntimeError("Joint v2 test data needs both classes.")

    return {
        "x_train": x_train,
        "y_train": y_train,
        "train_keys": keys_tr + keys_tf,
        "x_test": x_test,
        "y_test": y_test,
        "test_total": len(test_real) + len(test_fake),
        "counts": {
            "train_real": len(y_tr),
            "train_fake": len(y_tf),
            "test_real": len(y_er),
            "test_fake": len(y_ef),
            "dropped_forbidden_train": dropped_real + dropped_fake,
            "missing": {
                "train_real": miss_tr,
                "train_fake": miss_tf,
                "test_real": miss_er,
                "test_fake": miss_ef,
            },
            "video_extract_errors_preview": video_errors[:20],
            "au_extract_errors_preview": au_errors[:20],
        },
    }


def _headline(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    total_count: int,
) -> tuple[dict[str, float | None], dict[str, int]]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    labels = labels.astype(np.int64)
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


def train_wangxing_joint_au_pt_v2(
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
    modality_dropout: float = DEFAULT_MODALITY_DROPOUT,
) -> dict[str, Any]:
    torch_device = resolve_torch_device(
        str(device) if device is not None else "cuda"
    )
    _set_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device_name = (
        torch.cuda.get_device_name(torch_device)
        if torch_device.type == "cuda"
        else "cpu"
    )
    print(
        f"Joint v2 training device={torch_device} ({device_name}). "
        "Video/AU feature extraction stays on CPU.",
        flush=True,
    )

    prepared = _prepare_data(
        manifest=manifest,
        cache_dir=cache_dir,
        source_profile=source_profile,
        forensics_profiles=forensics_profiles,
    )
    x_train = prepared["x_train"]
    y_train = prepared["y_train"]
    x_test = prepared["x_test"]
    y_test = prepared["y_test"]
    fit_idx, val_idx = _split_fit_validation(
        y_train,
        prepared["train_keys"],
        seed=seed,
    )
    x_fit, mean, scale = _standardize(
        x_train[fit_idx],
        x_train[fit_idx],
    )
    x_fit = np.clip(x_fit, -8.0, 8.0).astype(np.float32)
    x_val = np.clip(
        (x_train[val_idx] - mean) / scale,
        -8.0,
        8.0,
    ).astype(np.float32)
    x_test_norm = np.clip(
        (x_test - mean) / scale,
        -8.0,
        8.0,
    ).astype(np.float32)
    y_fit = y_train[fit_idx].astype(np.float32)
    y_val = y_train[val_idx].astype(np.float32)

    model = JointAUVideoClassifier(
        video_dim=VIDEO_DIM,
        au_dim=AU_DIM,
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
    pin_memory = torch_device.type == "cuda"
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.ascontiguousarray(x_fit)),
            torch.from_numpy(np.ascontiguousarray(y_fit)),
        ),
        batch_size=max(1, int(batch_size)),
        sampler=WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        ),
        pin_memory=pin_memory,
    )
    criterion = nn.BCEWithLogitsLoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_key = (-1.0, float("inf"))
    history: list[dict[str, Any]] = []
    aux_weights = DEFAULT_AUX_LOSS_WEIGHTS

    for epoch in range(max(1, int(epochs))):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(torch_device, non_blocking=pin_memory)
            batch_y = batch_y.to(torch_device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            joint_logit, video_logit, au_logit = model(
                batch_x,
                return_aux=True,
            )
            loss = (
                aux_weights["joint"] * criterion(joint_logit, batch_y)
                + aux_weights["video"] * criterion(video_logit, batch_y)
                + aux_weights["au"] * criterion(au_logit, batch_y)
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

        val_logits = _predict_joint_logits(model, x_val, torch_device)
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
    val_logits = _predict_joint_logits(model, x_val, torch_device)
    temperature = _fit_temperature(val_logits, y_val)
    test_logits = _predict_joint_logits(model, x_test_norm, torch_device)
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
        "model_type": JOINT_V2_MODEL_TYPE,
        "feature_version": FEATURE_VERSION,
        "config": {
            "video_dim": VIDEO_DIM,
            "au_dim": AU_DIM,
            "video_hidden": VIDEO_HIDDEN,
            "au_hidden": AU_HIDDEN,
            "fusion_hidden": FUSION_HIDDEN,
            "fusion_bottleneck": FUSION_BOTTLENECK,
            "modality_dropout": float(modality_dropout),
            "scales": [SCALE_A, SCALE_B],
            "input_dim": VIDEO_DIM + AU_DIM,
            "fusion_mode": "two_branch_au_conditioned_gate",
            "auxiliary_heads": True,
        },
        "model_state": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "feature_mean": mean.astype(np.float32),
        "feature_scale": scale.astype(np.float32),
        "temperature": float(temperature),
        "device_used": str(torch_device),
        "aux_loss_weights": dict(aux_weights),
        "dataset": prepared["counts"],
        "validation": {
            "fit_count": int(len(fit_idx)),
            "validation_count": int(len(val_idx)),
            "normalization_fit": "fit_subset_only",
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
        "input_dim": VIDEO_DIM + AU_DIM,
        "video_dim": VIDEO_DIM,
        "au_dim": AU_DIM,
        "device": str(torch_device),
    }


def predict_wangxing_joint_au_pt_v2(
    *,
    video_path: str | Path,
    au_path: str | Path,
    model_path: str | Path,
    source_profile: dict[str, Any],
    forensics_profiles: dict[str, Any],
) -> dict[str, Any]:
    video_path = Path(video_path).expanduser().resolve()
    au_path = Path(au_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    checkpoint = torch.load(str(model_path), map_location="cpu")
    if checkpoint.get("model_type") != JOINT_V2_MODEL_TYPE:
        raise ValueError(f"Unsupported model_type: {checkpoint.get('model_type')}")

    scales = checkpoint["config"]["scales"]
    from evaluator.vedio_pred.wangxing_dual_pt import extract_dual_feature

    video_vec = extract_dual_feature(
        video_path,
        scale_a=scales[0],
        scale_b=scales[1],
    )
    from evaluator.modules.forensics.learned_fusion_head import (
        extract_fusion_features,
    )

    au_vec, au_dict = extract_fusion_features(
        au_path=au_path,
        wangxing_source_profile=source_profile,
        forensics_profiles=forensics_profiles,
    )
    feature = np.concatenate(
        [video_vec.astype(np.float32), np.asarray(au_vec, dtype=np.float32)],
        axis=0,
    )
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
    if feature.shape != mean.shape:
        raise ValueError(
            f"Feature dim mismatch: got {feature.shape}, expected {mean.shape}"
        )
    normalized = np.clip(
        (feature - mean) / np.maximum(scale, 1e-4),
        -8.0,
        8.0,
    ).astype(np.float32)
    model = _model_from_checkpoint(checkpoint)
    model.eval()
    logit = float(model(torch.from_numpy(normalized[None, :]))[0].item())
    temperature = float(checkpoint.get("temperature", 1.0))
    p_gen = float(
        1.0
        / (1.0 + math.exp(-logit / max(temperature, 1e-6)))
    )
    decision = "generated" if p_gen >= 0.5 else "real"
    return {
        "prediction": decision,
        "generated_probability": p_gen,
        "real_probability": 1.0 - p_gen,
        "logit": logit,
        "temperature": temperature,
        "model_path": str(model_path),
        "video_path": str(video_path),
        "au_path": str(au_path),
        "fusion_mode": "two_branch_au_conditioned_gate",
        "au_quality_min": float(au_dict.get("quality_min", 0.5)),
    }


def evaluate_holdout_joint_au_pt_v2(
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
        if not isinstance(item, dict) or not item.get("video"):
            continue
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
        scored = predict_wangxing_joint_au_pt_v2(
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
        "schema_version": "wangxing_joint_au_pt_v2_holdout_metrics_v1",
        "model_path": str(model_path),
        "holdout_manifest": str(holdout_manifest),
        "headline": headline,
        "confusion": confusion,
        "rows": rows,
    }
