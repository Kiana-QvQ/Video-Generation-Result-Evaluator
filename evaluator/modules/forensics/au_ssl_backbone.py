"""Lightweight self-supervised AU temporal backbone (TCAE / AU-vMAE style).

Trains on unlabeled AU trajectories with:
1. Reconstruction loss (TCAE-like temporal autoencoder)
2. One-step future prediction
3. Random / tube frame masking (VideoMAE-style)

No manual AU intensity labels are required. Weights are optional; when absent,
``extract_backbone_features`` falls back to a training-free proxy score.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

AU_SSL_BACKBONE_SCHEMA = "au_ssl_backbone_tcae_v1"
DEFAULT_SEQ_LEN = 32
DEFAULT_LATENT_DIM = 32
DEFAULT_HIDDEN = 64


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def default_backbone_path() -> Path:
    override = os.environ.get("EVALUATOR_AU_SSL_BACKBONE")
    if override:
        return Path(override).expanduser()
    try:
        from ..core.paths import MODULES_ROOT

        return MODULES_ROOT / "assets" / "models" / "au_ssl_tcae.pt"
    except Exception:
        return Path("au_ssl_tcae.pt")


def _window_matrix(matrix: np.ndarray, seq_len: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.shape[0] >= seq_len:
        start = max(0, (matrix.shape[0] - seq_len) // 2)
        return matrix[start : start + seq_len]
    pad = seq_len - matrix.shape[0]
    return np.pad(matrix, ((0, pad), (0, 0)), mode="edge")


def _numpy_proxy_score(matrix: np.ndarray) -> dict[str, float]:
    """Training-free stand-in when no backbone weights are available."""
    window = _window_matrix(matrix, DEFAULT_SEQ_LEN)
    if window.shape[0] < 4:
        return {
            "ssl_backbone_recon_error": 0.0,
            "ssl_backbone_pred_error": 0.0,
            "ssl_backbone_score_0_1": 0.5,
        }
    recon = window.copy()
    recon[1:-1] = 0.5 * (window[:-2] + window[2:])
    recon_err = float(np.mean(np.abs(window - recon)))
    pred = window[:-1] + np.median(np.diff(window, axis=0), axis=0, keepdims=True)
    pred_err = float(np.mean(np.abs(window[1:] - pred)))
    score = _clamp(1.0 - 2.5 * recon_err - 2.0 * pred_err)
    return {
        "ssl_backbone_recon_error": recon_err,
        "ssl_backbone_pred_error": pred_err,
        "ssl_backbone_score_0_1": score,
    }


class TemporalAUAutoencoder:
    """1D temporal conv autoencoder + future head over AU channels."""

    def __init__(
        self,
        n_channels: int,
        *,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden: int = DEFAULT_HIDDEN,
    ) -> None:
        import torch
        from torch import nn

        self.n_channels = int(n_channels)
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)

        class _Net(nn.Module):
            def __init__(self_inner) -> None:
                super().__init__()
                self_inner.encoder = nn.Sequential(
                    nn.Conv1d(n_channels, hidden, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Conv1d(hidden, latent_dim, kernel_size=3, padding=1),
                )
                self_inner.decoder = nn.Sequential(
                    nn.Conv1d(latent_dim, hidden, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Conv1d(hidden, n_channels, kernel_size=5, padding=2),
                    nn.Sigmoid(),
                )
                self_inner.predict = nn.Sequential(
                    nn.Conv1d(latent_dim, hidden, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(hidden, n_channels, kernel_size=3, padding=1),
                    nn.Sigmoid(),
                )

            def forward(self_inner, x, mask=None):
                # x: (B, T, C) -> (B, C, T)
                xt = x.transpose(1, 2)
                if mask is not None:
                    mt = mask.transpose(1, 2)
                    xt = xt * mt
                z = self_inner.encoder(xt)
                recon = self_inner.decoder(z).transpose(1, 2)
                pred = self_inner.predict(z).transpose(1, 2)
                return recon, pred, z.transpose(1, 2)

        self.net = _Net()
        self._torch = torch

    def to(self, device: str | Any) -> TemporalAUAutoencoder:
        self.net.to(device)
        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AU_SSL_BACKBONE_SCHEMA,
            "n_channels": self.n_channels,
            "latent_dim": self.latent_dim,
            "hidden": self.hidden,
            "state_dict": self.net.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.net.load_state_dict(payload["state_dict"])


def _make_masks(
    batch: Any,
    *,
    mask_ratio: float,
    tube_ratio: float,
    rng: np.random.Generator,
) -> Any:
    import torch

    bsz, seq_len, channels = batch.shape
    mask = torch.ones((bsz, seq_len, channels), dtype=batch.dtype)
    for index in range(bsz):
        # Random frame drops.
        n_drop = max(1, int(round(seq_len * mask_ratio)))
        drop_idx = rng.choice(np.arange(1, seq_len - 1), size=min(n_drop, seq_len - 2), replace=False)
        mask[index, drop_idx, :] = 0.0
        # Contiguous tube mask.
        tube_len = max(2, int(round(seq_len * tube_ratio)))
        start = int(rng.integers(1, max(2, seq_len - tube_len)))
        mask[index, start : start + tube_len, :] = 0.0
    return mask.to(batch.device)


def train_au_ssl_backbone(
    sequences: Sequence[np.ndarray],
    *,
    seq_len: int = DEFAULT_SEQ_LEN,
    latent_dim: int = DEFAULT_LATENT_DIM,
    hidden: int = DEFAULT_HIDDEN,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    mask_ratio: float = 0.15,
    tube_ratio: float = 0.15,
    device: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Train TCAE-style backbone on unlabeled AU windows."""
    import torch
    from torch import nn

    if not sequences:
        raise ValueError("Need at least one AU sequence to train the backbone.")
    channels = int(max(seq.shape[1] if seq.ndim == 2 else 1 for seq in sequences))
    windows: list[np.ndarray] = []
    for seq in sequences:
        matrix = np.asarray(seq, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.shape[1] < channels:
            matrix = np.pad(
                matrix,
                ((0, 0), (0, channels - matrix.shape[1])),
                mode="constant",
            )
        elif matrix.shape[1] > channels:
            matrix = matrix[:, :channels]
        if matrix.shape[0] < 8:
            continue
        # Multiple sliding windows per clip.
        step = max(4, seq_len // 2)
        for start in range(0, max(1, matrix.shape[0] - seq_len + 1), step):
            windows.append(_window_matrix(matrix[start : start + seq_len], seq_len))
        if matrix.shape[0] < seq_len:
            windows.append(_window_matrix(matrix, seq_len))
    if len(windows) < 4:
        raise ValueError("Not enough AU windows for SSL backbone training.")

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalAUAutoencoder(
        channels,
        latent_dim=latent_dim,
        hidden=hidden,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    rng = np.random.default_rng(seed)
    data = np.stack(windows, axis=0)
    history: list[float] = []

    model.net.train()
    for _epoch in range(int(epochs)):
        order = rng.permutation(len(data))
        epoch_losses: list[float] = []
        for begin in range(0, len(order), batch_size):
            indexes = order[begin : begin + batch_size]
            batch_np = data[indexes]
            batch = torch.from_numpy(batch_np).to(resolved_device)
            mask = _make_masks(
                batch,
                mask_ratio=mask_ratio,
                tube_ratio=tube_ratio,
                rng=rng,
            )
            recon, pred, _ = model.net(batch, mask=mask)
            recon_loss = loss_fn(recon, batch)
            pred_loss = loss_fn(pred[:, :-1, :], batch[:, 1:, :])
            loss = recon_loss + 0.5 * pred_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        history.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

    payload = model.state_dict()
    payload.update(
        {
            "seq_len": int(seq_len),
            "train_window_count": len(windows),
            "train_sequence_count": len(sequences),
            "epochs": int(epochs),
            "loss_history": history,
            "final_loss": history[-1] if history else None,
            "manual_labels_required": False,
            "device_trained": resolved_device,
        }
    )
    return payload


def save_backbone(payload: dict[str, Any], path: str | Path | None = None) -> Path:
    import torch

    target = Path(path) if path is not None else default_backbone_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def load_backbone(
    path: str | Path | None = None,
    *,
    map_location: str | None = None,
) -> dict[str, Any] | None:
    import torch

    target = Path(path) if path is not None else default_backbone_path()
    if not target.is_file():
        return None
    device = map_location or "cpu"
    try:
        return torch.load(target, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(target, map_location=device)


def extract_backbone_features(
    au_matrix: np.ndarray,
    *,
    weights: dict[str, Any] | None = None,
    weights_path: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run trained backbone (or numpy proxy) and emit scalar features."""
    matrix = np.asarray(au_matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    payload = weights
    if payload is None:
        payload = load_backbone(weights_path)
    if payload is None:
        features = _numpy_proxy_score(matrix)
        return {
            "schema_version": AU_SSL_BACKBONE_SCHEMA,
            "status": "proxy_untrained",
            "backend": "numpy_proxy",
            "features": features,
            "manual_labels_required": False,
            "note": (
                "No trained AU SSL backbone found; using training-free proxy. "
                "Run scripts/train_au_ssl_backbone.py on data/au."
            ),
        }

    import torch

    n_channels = int(payload.get("n_channels", matrix.shape[1]))
    seq_len = int(payload.get("seq_len", DEFAULT_SEQ_LEN))
    model = TemporalAUAutoencoder(
        n_channels,
        latent_dim=int(payload.get("latent_dim", DEFAULT_LATENT_DIM)),
        hidden=int(payload.get("hidden", DEFAULT_HIDDEN)),
    )
    model.load_state_dict(payload)
    resolved = device or "cpu"
    model.to(resolved)
    model.net.eval()

    window = _window_matrix(matrix, seq_len)
    if window.shape[1] < n_channels:
        window = np.pad(
            window,
            ((0, 0), (0, n_channels - window.shape[1])),
            mode="constant",
        )
    elif window.shape[1] > n_channels:
        window = window[:, :n_channels]
    tensor = torch.from_numpy(window[None, ...]).to(resolved)
    with torch.no_grad():
        recon, pred, latent = model.net(tensor)
    recon_err = float(torch.mean(torch.abs(recon - tensor)).item())
    pred_err = float(
        torch.mean(torch.abs(pred[:, :-1, :] - tensor[:, 1:, :])).item()
    )
    latent_energy = float(torch.mean(torch.abs(latent)).item())
    latent_std = float(torch.std(latent).item())
    score = _clamp(1.0 - 2.2 * recon_err - 1.8 * pred_err)
    return {
        "schema_version": AU_SSL_BACKBONE_SCHEMA,
        "status": "available",
        "backend": "torch_tcae",
        "features": {
            "ssl_backbone_recon_error": recon_err,
            "ssl_backbone_pred_error": pred_err,
            "ssl_backbone_latent_energy": latent_energy,
            "ssl_backbone_latent_std": latent_std,
            "ssl_backbone_score_0_1": score,
        },
        "manual_labels_required": False,
        "note": (
            "Trained TCAE / masked-frame AU backbone. No manual AU labels used."
        ),
    }


def merge_backbone_into_ssl(
    ssl_result: dict[str, Any],
    backbone_result: dict[str, Any],
) -> dict[str, Any]:
    """Blend backbone score into the existing SSL feature bundle."""
    features = dict(ssl_result.get("features", {}))
    backbone_features = dict(backbone_result.get("features", {}))
    features.update(backbone_features)
    base = float(features.get("ssl_au_score_0_1", 0.5))
    backbone = float(backbone_features.get("ssl_backbone_score_0_1", 0.5))
    if not math.isfinite(base):
        base = 0.5
    if not math.isfinite(backbone):
        backbone = 0.5
    weight = 0.45 if backbone_result.get("status") == "available" else 0.20
    features["ssl_au_score_0_1"] = _clamp((1.0 - weight) * base + weight * backbone)
    enriched = dict(ssl_result)
    enriched["features"] = features
    enriched["ssl_backbone"] = {
        "status": backbone_result.get("status"),
        "backend": backbone_result.get("backend"),
        "note": backbone_result.get("note"),
    }
    return enriched
