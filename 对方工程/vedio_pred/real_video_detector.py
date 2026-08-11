from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".flv",
}
FEATURE_VERSION = "real-video-feature-v1"
FRAME_FEATURE_DIM = 96
TEMPORAL_FEATURE_DIM = 6
PROBABILITY_FLOOR = 0.02
PROBABILITY_CEILING = 0.98
MIN_CALIBRATION_TEMPERATURE = 1.25


def discover_videos(data_dir: str | Path) -> List[Path]:
    """Recursively discover videos while keeping all dataset files in video/."""

    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"真实视频数据目录不存在: {root}")
    videos = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos, key=lambda path: str(path).lower())


def _sample_indices(frame_count: int, num_frames: int) -> List[int]:
    if frame_count <= 0:
        return []
    count = max(2, int(num_frames))
    indices = np.linspace(
        0,
        max(frame_count - 1, 0),
        num=count,
        dtype=np.int64,
    )
    return [int(index) for index in indices]


def _read_sampled_frames(
        video_path: str | Path,
        num_frames: int,
        frame_size: int,
) -> List[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _sample_indices(frame_count, num_frames)
    frames: List[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        frame = cv2.resize(
            frame,
            (int(frame_size), int(frame_size)),
            interpolation=cv2.INTER_AREA,
        )
        frames.append(frame)
    capture.release()

    if not frames:
        raise RuntimeError(f"视频未读取到有效帧: {video_path}")

    # Some codecs return fewer frames after random seeking. Repeat nearby
    # samples so every video still produces the same-size feature vector.
    target_count = max(2, int(num_frames))
    if len(frames) < target_count:
        source = list(frames)
        frames = [
            source[
                min(
                    int(round(index * (len(source) - 1) / max(target_count - 1, 1))),
                    len(source) - 1,
                )
            ]
            for index in range(target_count)
        ]
    elif len(frames) > target_count:
        frames = frames[:target_count]
    return frames


def _frame_feature(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    bgr = frame.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] /= 180.0
    hsv[:, :, 1:] /= 255.0

    gray_stats = np.asarray(
        [
            float(np.mean(gray)),
            float(np.std(gray)),
            float(np.percentile(gray, 5)),
            float(np.percentile(gray, 50)),
            float(np.percentile(gray, 95)),
        ],
        dtype=np.float32,
    )
    color_stats = np.concatenate(
        [
            np.mean(bgr, axis=(0, 1)),
            np.std(bgr, axis=(0, 1)),
            np.mean(hsv, axis=(0, 1)),
        ]
    ).astype(np.float32)
    histogram = np.histogram(
        gray,
        bins=16,
        range=(0.0, 1.0),
    )[0].astype(np.float32)
    histogram /= max(float(histogram.sum()), 1.0)

    edge_map = cv2.Canny(
        (gray * 255.0).astype(np.uint8),
        threshold1=50,
        threshold2=150,
    )
    edge_ratio = float(np.mean(edge_map > 0))
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    texture_signal = min(math.log1p(max(laplacian_variance, 0.0)) / 8.0, 1.0)
    texture_stats = np.asarray(
        [edge_ratio, texture_signal],
        dtype=np.float32,
    )

    low_frequency = cv2.resize(
        gray,
        (8, 8),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32).reshape(-1)

    return np.concatenate(
        [
            gray_stats,
            color_stats,
            histogram,
            texture_stats,
            low_frequency,
        ]
    ).astype(np.float32)


def extract_video_feature(
        video_path: str | Path,
        num_frames: int = 12,
        frame_size: int = 64,
) -> np.ndarray:
    """Extract a compact fixed-length spatiotemporal descriptor."""

    frames = _read_sampled_frames(
        video_path=video_path,
        num_frames=num_frames,
        frame_size=frame_size,
    )
    frame_features = [
        _frame_feature(frame)
        for frame in frames
    ]
    temporal_features: List[np.ndarray] = []
    previous_gray: Optional[np.ndarray] = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
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

    feature = np.concatenate(
        [
            np.concatenate(frame_features),
            np.concatenate(temporal_features),
        ]
    ).astype(np.float32)
    return np.nan_to_num(feature, nan=0.0, posinf=1.0, neginf=0.0)


def _file_signature(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"


def load_or_extract_features(
        videos: Sequence[Path],
        cache_path: str | Path,
        num_frames: int,
        frame_size: int,
) -> Tuple[np.ndarray, List[Path], List[str]]:
    """Reuse a cache only when the exact video files are unchanged."""

    cache_path = Path(cache_path)
    signatures = [_file_signature(path) for path in videos]
    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            cached_paths = [str(value) for value in cached["paths"].tolist()]
            cached_signatures = [
                str(value) for value in cached["signatures"].tolist()
            ]
            if (
                    cached_paths == [str(path) for path in videos]
                    and cached_signatures == signatures
            ):
                return (
                    cached["features"].astype(np.float32),
                    list(videos),
                    [],
                )
        except Exception:
            pass

    features: List[np.ndarray] = []
    valid_paths: List[Path] = []
    errors: List[str] = []
    for index, path in enumerate(videos, start=1):
        try:
            features.append(
                extract_video_feature(
                    path,
                    num_frames=num_frames,
                    frame_size=frame_size,
                )
            )
            valid_paths.append(path)
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
        if index % 10 == 0 or index == len(videos):
            print(f"[feature] {index}/{len(videos)}")

    if not features:
        raise RuntimeError("没有视频能够提取有效特征")
    feature_array = np.stack(features).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=feature_array,
        paths=np.asarray([str(path) for path in valid_paths]),
        signatures=np.asarray(
            [_file_signature(path) for path in valid_paths]
        ),
        feature_version=np.asarray([FEATURE_VERSION]),
        num_frames=np.asarray([num_frames]),
        frame_size=np.asarray([frame_size]),
    )
    return feature_array, valid_paths, errors


class RealVideoAutoencoder(nn.Module):
    """Small one-class neural network trained only on real videos."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            latent_dim: int = 32,
    ) -> None:
        super().__init__()
        hidden_dim = max(32, min(int(hidden_dim), 512))
        latent_dim = max(8, min(int(latent_dim), hidden_dim // 2))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


def _standardize(
        train_features: np.ndarray,
        features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0).astype(np.float32)
    scale = train_features.std(axis=0).astype(np.float32)
    scale = np.maximum(scale, 1e-4)
    normalized = (features - mean) / scale
    return normalized.astype(np.float32), mean, scale


def _reconstruction_errors(
        model: RealVideoAutoencoder,
        features: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(features)
        outputs = model(inputs)
        errors = torch.mean((outputs - inputs) ** 2, dim=1)
    return errors.cpu().numpy().astype(np.float32)


def _center_distances(features: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(features), axis=1)).astype(np.float32)


def _nearest_distances(
        features: np.ndarray,
        references: np.ndarray,
) -> np.ndarray:
    values = []
    for feature in features:
        distances = np.sqrt(
            np.mean(np.square(references - feature[None, :]), axis=1)
        )
        values.append(float(np.min(distances)))
    return np.asarray(values, dtype=np.float32)


def _threshold(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    quantile = float(np.percentile(values, 95.0))
    margin = max(
        float(np.std(values) * 2.0),
        abs(quantile) * 0.25,
        1e-4,
    )
    return {
        "threshold": quantile,
        "margin": margin,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_real_video_detector(
        data_dir: str | Path,
        model_path: str | Path,
        cache_path: Optional[str | Path] = None,
        num_frames: int = 12,
        frame_size: int = 64,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        epochs: int = 80,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_ratio: float = 0.20,
        max_videos: Optional[int] = None,
        seed: int = 42,
) -> Dict[str, Any]:
    """Train the one-class autoencoder using real videos only."""

    _set_seed(seed)
    videos = discover_videos(data_dir)
    if max_videos is not None:
        videos = videos[:max(1, int(max_videos))]
    if len(videos) < 4:
        raise ValueError(
            f"至少需要 4 个真实视频才能划分训练/验证集，当前只有 {len(videos)} 个"
        )

    data_dir = Path(data_dir).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    if cache_path is None:
        cache_path = data_dir.parent / "cache" / (
            f"real_features_f{num_frames}_s{frame_size}.npz"
        )
    features, valid_paths, errors = load_or_extract_features(
        videos=videos,
        cache_path=cache_path,
        num_frames=num_frames,
        frame_size=frame_size,
    )
    if len(valid_paths) < 4:
        raise ValueError(
            f"有效视频少于 4 个，无法训练；有效数量: {len(valid_paths)}"
        )

    order = np.arange(len(valid_paths))
    np.random.shuffle(order)
    features = features[order]
    valid_paths = [valid_paths[index] for index in order]
    validation_count = max(
        1,
        min(
            len(valid_paths) - 1,
            int(round(len(valid_paths) * float(validation_ratio))),
        ),
    )
    validation_features = features[:validation_count]
    train_features = features[validation_count:]
    train_paths = valid_paths[validation_count:]
    validation_paths = valid_paths[:validation_count]

    train_normalized, mean, scale = _standardize(
        train_features,
        train_features,
    )
    validation_normalized = (
        (validation_features - mean) / scale
    ).astype(np.float32)
    model = RealVideoAutoencoder(
        input_dim=train_normalized.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=1e-4,
    )
    train_tensor = torch.from_numpy(train_normalized)
    best_state = None
    best_validation_loss = float("inf")
    stale_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(max(1, int(epochs))):
        model.train()
        permutation = torch.randperm(train_tensor.shape[0])
        train_losses: List[float] = []
        for start in range(0, train_tensor.shape[0], max(1, int(batch_size))):
            batch_indices = permutation[start:start + max(1, int(batch_size))]
            batch = train_tensor[batch_indices]
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = F.mse_loss(reconstruction, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                F.mse_loss(
                    model(torch.from_numpy(validation_normalized)),
                    torch.from_numpy(validation_normalized),
                ).cpu().item()
            )
        train_loss = float(np.mean(train_losses))
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 15:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    train_errors = _reconstruction_errors(model, train_normalized)
    validation_errors = _reconstruction_errors(model, validation_normalized)
    train_center = _center_distances(train_normalized)
    validation_center = _center_distances(validation_normalized)
    validation_nearest = _nearest_distances(
        validation_normalized,
        train_normalized,
    )

    checkpoint = {
        "format_version": 1,
        "feature_version": FEATURE_VERSION,
        "config": {
            "num_frames": int(num_frames),
            "frame_size": int(frame_size),
            "input_dim": int(train_normalized.shape[1]),
            "hidden_dim": int(hidden_dim),
            "latent_dim": int(latent_dim),
        },
        "model_state": model.state_dict(),
        "feature_mean": mean,
        "feature_scale": scale,
        "reference_features": train_normalized.astype(np.float32),
        "thresholds": {
            "reconstruction": _threshold(validation_errors),
            "center_distance": _threshold(validation_center),
            "nearest_real": _threshold(validation_nearest),
        },
        "dataset": {
            "data_dir": str(data_dir),
            "video_count": len(valid_paths),
            "train_count": len(train_paths),
            "validation_count": len(validation_paths),
            "train_paths": [str(path) for path in train_paths],
            "validation_paths": [str(path) for path in validation_paths],
            "skipped": errors,
        },
        "training": {
            "seed": int(seed),
            "epochs_requested": int(epochs),
            "epochs_completed": len(history),
            "best_validation_loss": float(best_validation_loss),
            "history_tail": history[-10:],
            "train_reconstruction_mean": float(np.mean(train_errors)),
            "validation_reconstruction_mean": float(
                np.mean(validation_errors)
            ),
            "validation_reconstruction_p95": float(
                np.percentile(validation_errors, 95.0)
            ),
            "validation_center_distance_p95": float(
                np.percentile(validation_center, 95.0)
            ),
            "validation_nearest_real_p95": float(
                np.percentile(validation_nearest, 95.0)
            ),
        },
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)
    return {
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "video_count": len(valid_paths),
        "train_count": len(train_paths),
        "validation_count": len(validation_paths),
        "skipped_count": len(errors),
        "input_dim": int(train_normalized.shape[1]),
        "epochs_completed": len(history),
        "best_validation_loss": float(best_validation_loss),
        "cache_path": str(cache_path),
    }


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _smooth_probability(raw_probability: float) -> float:
    """Keep probabilities informative without ever reporting certainty."""

    raw_probability = max(0.0, min(1.0, float(raw_probability)))
    probability = (
        PROBABILITY_FLOOR
        + raw_probability
        * (PROBABILITY_CEILING - PROBABILITY_FLOOR)
    )
    return round(probability, 4)


def _anomaly_component(
        value: float,
        threshold: Dict[str, float],
) -> float:
    return (
        float(value) - float(threshold["threshold"])
    ) / max(float(threshold["margin"]), 1e-6)


def predict_video(
        video_path: str | Path,
        model_path: str | Path,
) -> Dict[str, Any]:
    """Predict whether a video is outside the learned real-video distribution."""

    video_path = Path(video_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"待检测视频不存在: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"真实视频检测模型不存在: {model_path}")

    checkpoint = torch.load(str(model_path), map_location="cpu")
    config = checkpoint["config"]
    feature = extract_video_feature(
        video_path,
        num_frames=int(config["num_frames"]),
        frame_size=int(config["frame_size"]),
    )
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
    normalized = ((feature - mean) / np.maximum(scale, 1e-4)).astype(np.float32)
    model = RealVideoAutoencoder(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    reconstruction_error = float(
        _reconstruction_errors(model, normalized[None, :])[0]
    )
    center_distance = float(_center_distances(normalized[None, :])[0])
    reference_features = np.asarray(
        checkpoint["reference_features"],
        dtype=np.float32,
    )
    nearest_real_distance = float(
        _nearest_distances(normalized[None, :], reference_features)[0]
    )
    thresholds = checkpoint["thresholds"]
    reconstruction_component = _anomaly_component(
        reconstruction_error,
        thresholds["reconstruction"],
    )
    center_component = _anomaly_component(
        center_distance,
        thresholds["center_distance"],
    )
    nearest_component = _anomaly_component(
        nearest_real_distance,
        thresholds["nearest_real"],
    )
    anomaly_logit = (
        0.60 * reconstruction_component
        + 0.20 * center_component
        + 0.20 * nearest_component
    )
    generated_probability = _smooth_probability(_sigmoid(anomaly_logit))
    real_probability = round(1.0 - generated_probability, 4)
    if generated_probability >= 0.65:
        prediction = "generated"
        label = "更可能是生成视频"
    elif real_probability >= 0.65:
        prediction = "real"
        label = "更可能是真实视频"
    else:
        prediction = "uncertain"
        label = "暂无法区分"

    def real_signal(component: float) -> str:
        if component > 0.5:
            return "偏向生成"
        if component < -0.5:
            return "偏向真实"
        return "信号中性"

    evidence = [
        {
            "指标": "神经网络重建误差",
            "指标得分": round(
                max(0.0, min(100.0, (1.0 - reconstruction_component) * 50.0 + 50.0)),
                2,
            ),
            "方向": real_signal(reconstruction_component),
            "说明": (
                f"当前值 {reconstruction_error:.6f}，真实验证集阈值 "
                f"{thresholds['reconstruction']['threshold']:.6f}"
            ),
        },
        {
            "指标": "到真实样本中心距离",
            "指标得分": round(
                max(0.0, min(100.0, (1.0 - center_component) * 50.0 + 50.0)),
                2,
            ),
            "方向": real_signal(center_component),
            "说明": (
                f"当前值 {center_distance:.6f}，真实验证集阈值 "
                f"{thresholds['center_distance']['threshold']:.6f}"
            ),
        },
        {
            "指标": "最近真实视频距离",
            "指标得分": round(
                max(0.0, min(100.0, (1.0 - nearest_component) * 50.0 + 50.0)),
                2,
            ),
            "方向": real_signal(nearest_component),
            "说明": (
                f"当前值 {nearest_real_distance:.6f}，真实验证集阈值 "
                f"{thresholds['nearest_real']['threshold']:.6f}"
            ),
        },
    ]
    return {
        "预测": prediction,
        "标签": label,
        "生成概率": generated_probability,
        "真实概率": real_probability,
        "证据强度": round(
            min(1.0, 0.45 + abs(generated_probability - 0.5) * 1.1),
            4,
        ),
        "结论": (
            f"神经网络判定：当前视频{label}，生成概率 "
            f"{generated_probability * 100.0:.1f}%，真实概率 "
            f"{real_probability * 100.0:.1f}%。"
        ),
        "证据": evidence,
        "方法": (
            "真实视频 One-Class Autoencoder + 真实样本中心/最近邻距离"
        ),
        "说明": (
            "模型只使用真实视频训练，概率表示该视频偏离真实视频分布的程度；"
            "要得到经过校准的绝对真假概率，还需要加入已标注的生成视频进行监督校准。"
        ),
        "模型路径": str(model_path),
        "真实数据目录": checkpoint["dataset"]["data_dir"],
        "真实训练视频数": checkpoint["dataset"]["train_count"],
        "真实验证视频数": checkpoint["dataset"]["validation_count"],
        "神经网络重建误差": reconstruction_error,
        "真实样本中心距离": center_distance,
        "最近真实视频距离": nearest_real_distance,
        "模型版本": checkpoint.get("feature_version", FEATURE_VERSION),
    }


def _group_key(path: Path, label: int) -> str:
    """Group by source folder/file stem so near-duplicates stay together."""

    if label == 0:
        return f"real::{path.parent.resolve()}"
    stem = path.stem.replace(" (1)", "")
    return f"fake::{stem}"


def _split_paths_by_group(
        paths: Sequence[Path],
        label: int,
        validation_ratio: float,
        seed: int,
) -> Tuple[List[Path], List[Path]]:
    grouped: Dict[str, List[Path]] = {}
    for path in paths:
        grouped.setdefault(_group_key(path, label), []).append(path)
    groups = list(grouped.values())
    random.Random(seed + label).shuffle(groups)
    validation_count = max(
        1,
        min(
            len(groups) - 1,
            int(round(len(groups) * float(validation_ratio))),
        ),
    )
    validation = [
        path
        for group in groups[:validation_count]
        for path in group
    ]
    train = [
        path
        for group in groups[validation_count:]
        for path in group
    ]
    return train, validation


def _limit_paths_by_group(
        paths: Sequence[Path],
        label: int,
        limit: Optional[int],
) -> List[Path]:
    if limit is None or int(limit) >= len(paths):
        return list(paths)
    grouped: Dict[str, List[Path]] = {}
    for path in paths:
        grouped.setdefault(_group_key(path, label), []).append(path)
    groups = list(grouped.values())
    selected: List[Path] = []
    cursor = 0
    while len(selected) < int(limit) and groups:
        group = groups[cursor % len(groups)]
        item_index = len(selected) // len(groups)
        if item_index < len(group):
            selected.append(group[item_index])
        cursor += 1
        if cursor > int(limit) * max(len(groups), 1) * 2:
            break
    return selected[:int(limit)]


class RealFakeVideoNetwork(nn.Module):
    """Spatial-frame encoder plus temporal GRU binary classifier."""

    def __init__(
            self,
            num_frames: int,
            frame_feature_dim: int = FRAME_FEATURE_DIM,
            temporal_feature_dim: int = TEMPORAL_FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.num_frames = int(num_frames)
        self.frame_feature_dim = int(frame_feature_dim)
        self.temporal_feature_dim = int(temporal_feature_dim)
        self.frame_encoder = nn.Sequential(
            nn.Linear(self.frame_feature_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.temporal_encoder = nn.Sequential(
            nn.Linear(self.temporal_feature_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.frame_gru = nn.GRU(
            input_size=96,
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_gru = nn.GRU(
            input_size=32,
            hidden_size=32,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        frame_end = self.num_frames * self.frame_feature_dim
        frame_values = inputs[:, :frame_end]
        temporal_values = inputs[:, frame_end:]
        frames = frame_values.view(
            -1,
            self.num_frames,
            self.frame_feature_dim,
        )
        temporal = temporal_values.view(
            -1,
            max(self.num_frames - 1, 1),
            self.temporal_feature_dim,
        )
        frame_encoded = self.frame_encoder(frames)
        temporal_encoded = self.temporal_encoder(temporal)
        frame_sequence, _ = self.frame_gru(frame_encoded)
        temporal_sequence, _ = self.temporal_gru(temporal_encoded)
        frame_pool = torch.cat(
            [
                frame_sequence.mean(dim=1),
                frame_sequence.amax(dim=1),
            ],
            dim=1,
        )
        temporal_pool = torch.cat(
            [
                temporal_sequence.mean(dim=1),
                temporal_sequence.amax(dim=1),
            ],
            dim=1,
        )
        return self.classifier(
            torch.cat([frame_pool, temporal_pool], dim=1)
        ).squeeze(1)


def _classification_metrics(
        labels: np.ndarray,
        probabilities: np.ndarray,
) -> Dict[str, float]:
    labels = labels.astype(np.int64)
    probabilities = probabilities.astype(np.float32)
    predictions = (probabilities >= 0.5).astype(np.int64)
    true_positive = float(np.sum((predictions == 1) & (labels == 1)))
    true_negative = float(np.sum((predictions == 0) & (labels == 0)))
    false_positive = float(np.sum((predictions == 1) & (labels == 0)))
    false_negative = float(np.sum((predictions == 0) & (labels == 1)))
    positive_count = max(true_positive + false_negative, 1.0)
    negative_count = max(true_negative + false_positive, 1.0)
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / positive_count
    specificity = true_negative / negative_count
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    result = {
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1": float(f1),
        "false_positive_rate_real": float(false_positive / negative_count),
    }
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(labels)) == 2:
            result["roc_auc"] = float(roc_auc_score(labels, probabilities))
    except Exception:
        pass
    return result


def _fit_temperature(
        logits: np.ndarray,
        labels: np.ndarray,
) -> float:
    """Fit a stable scalar temperature on held-out logits."""

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if not np.all(np.isfinite(logits)):
        return 1.0
    temperatures = np.exp(
        np.linspace(
            np.log(MIN_CALIBRATION_TEMPERATURE),
            np.log(10.0),
            96,
        )
    )
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in temperatures:
        scaled = np.clip(logits / temperature, -60.0, 60.0)
        probabilities = 1.0 / (1.0 + np.exp(-scaled))
        loss = -np.mean(
            labels * np.log(np.maximum(probabilities, 1e-7))
            + (1.0 - labels) * np.log(
                np.maximum(1.0 - probabilities, 1e-7)
            )
        )
        if np.isfinite(loss) and loss < best_loss:
            best_loss = float(loss)
            best_temperature = float(temperature)
    return best_temperature


def _predict_classifier_logits(
        model: RealFakeVideoNetwork,
        features: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features))
    return logits.cpu().numpy().astype(np.float32)


def train_real_fake_detector(
        real_dir: str | Path,
        fake_dir: str | Path,
        model_path: str | Path,
        real_cache_path: str | Path,
        fake_cache_path: str | Path,
        num_frames: int = 8,
        frame_size: int = 48,
        epochs: int = 60,
        batch_size: int = 32,
        learning_rate: float = 3e-4,
        validation_ratio: float = 0.20,
        hidden_seed: int = 42,
        max_real_videos: Optional[int] = None,
        max_fake_videos: Optional[int] = None,
) -> Dict[str, Any]:
    """Train a supervised real/fake classifier with group-aware validation."""

    _set_seed(hidden_seed)
    real_paths = discover_videos(real_dir)
    fake_paths = discover_videos(fake_dir)
    real_paths = _limit_paths_by_group(
        real_paths,
        label=0,
        limit=max_real_videos,
    )
    fake_paths = _limit_paths_by_group(
        fake_paths,
        label=1,
        limit=max_fake_videos,
    )
    if len(real_paths) < 4 or len(fake_paths) < 4:
        raise ValueError(
            f"真实/假视频都至少需要 4 个，当前真实={len(real_paths)}，假={len(fake_paths)}"
        )

    real_train, real_validation = _split_paths_by_group(
        real_paths,
        label=0,
        validation_ratio=validation_ratio,
        seed=hidden_seed,
    )
    fake_train, fake_validation = _split_paths_by_group(
        fake_paths,
        label=1,
        validation_ratio=validation_ratio,
        seed=hidden_seed,
    )
    real_features, real_valid, real_errors = load_or_extract_features(
        real_paths,
        cache_path=real_cache_path,
        num_frames=num_frames,
        frame_size=frame_size,
    )
    fake_features, fake_valid, fake_errors = load_or_extract_features(
        fake_paths,
        cache_path=fake_cache_path,
        num_frames=num_frames,
        frame_size=frame_size,
    )
    real_map = {
        str(path.resolve()): feature
        for path, feature in zip(real_valid, real_features)
    }
    fake_map = {
        str(path.resolve()): feature
        for path, feature in zip(fake_valid, fake_features)
    }

    def collect(
            paths: Sequence[Path],
            label: int,
    ) -> Tuple[List[np.ndarray], List[int], List[str]]:
        values: List[np.ndarray] = []
        labels: List[int] = []
        missing: List[str] = []
        source = real_map if label == 0 else fake_map
        for path in paths:
            value = source.get(str(path.resolve()))
            if value is None:
                missing.append(str(path))
                continue
            values.append(value)
            labels.append(label)
        return values, labels, missing

    train_real_features, train_real_labels, missing_real_train = collect(
        real_train,
        0,
    )
    train_fake_features, train_fake_labels, missing_fake_train = collect(
        fake_train,
        1,
    )
    validation_real_features, validation_real_labels, missing_real_val = collect(
        real_validation,
        0,
    )
    validation_fake_features, validation_fake_labels, missing_fake_val = collect(
        fake_validation,
        1,
    )
    train_values = train_real_features + train_fake_features
    train_labels = np.asarray(
        train_real_labels + train_fake_labels,
        dtype=np.float32,
    )
    validation_values = validation_real_features + validation_fake_features
    validation_labels = np.asarray(
        validation_real_labels + validation_fake_labels,
        dtype=np.float32,
    )
    if len(np.unique(train_labels)) < 2 or len(np.unique(validation_labels)) < 2:
        raise RuntimeError("训练/验证集必须同时包含真实和假视频")

    train_array = np.stack(train_values).astype(np.float32)
    validation_array = np.stack(validation_values).astype(np.float32)
    train_normalized, mean, scale = _standardize(
        train_array,
        train_array,
    )
    train_normalized = np.clip(
        train_normalized,
        -8.0,
        8.0,
    ).astype(np.float32)
    validation_normalized = (
        (validation_array - mean) / scale
    )
    validation_normalized = np.clip(
        validation_normalized,
        -8.0,
        8.0,
    ).astype(np.float32)
    expected_dim = (
        num_frames * FRAME_FEATURE_DIM
        + max(num_frames - 1, 1) * TEMPORAL_FEATURE_DIM
    )
    if train_normalized.shape[1] != expected_dim:
        raise RuntimeError(
            f"特征维度不匹配: actual={train_normalized.shape[1]}, expected={expected_dim}"
        )

    model = RealFakeVideoNetwork(
        num_frames=num_frames,
        frame_feature_dim=FRAME_FEATURE_DIM,
        temporal_feature_dim=TEMPORAL_FEATURE_DIM,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=2e-4,
    )
    train_tensor = torch.from_numpy(train_normalized)
    label_tensor = torch.from_numpy(train_labels)
    class_counts = np.bincount(train_labels.astype(np.int64), minlength=2)
    sample_weights = np.asarray(
        [
            1.0 / max(float(class_counts[int(label)]), 1.0)
            for label in train_labels
        ],
        dtype=np.float64,
    )
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    loader = DataLoader(
        TensorDataset(train_tensor, label_tensor),
        batch_size=max(1, int(batch_size)),
        sampler=sampler,
    )
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_key = (-1.0, float("inf"))
    history: List[Dict[str, Any]] = []

    for epoch in range(max(1, int(epochs))):
        model.train()
        train_losses: List[float] = []
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            if not torch.isfinite(gradient_norm):
                optimizer.zero_grad()
                continue
            optimizer.step()
            if not all(
                    torch.isfinite(parameter).all().item()
                    for parameter in model.parameters()
            ):
                optimizer.zero_grad()
                continue
            train_losses.append(float(loss.detach().cpu().item()))

        validation_logits = _predict_classifier_logits(
            model,
            validation_normalized,
        )
        validation_probabilities = 1.0 / (
            1.0 + np.exp(-validation_logits)
        )
        validation_metrics = _classification_metrics(
            validation_labels.astype(np.int64),
            validation_probabilities,
        )
        validation_loss = float(
            criterion(
                torch.from_numpy(validation_logits),
                torch.from_numpy(validation_labels),
            ).item()
        )
        train_loss = (
            float(np.mean(train_losses))
            if train_losses
            else float("inf")
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                **validation_metrics,
            }
        )
        selection_key = (
            validation_metrics["balanced_accuracy"],
            -validation_loss,
        )
        if selection_key > best_key:
            best_key = selection_key
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    validation_logits = _predict_classifier_logits(
        model,
        validation_normalized,
    )
    temperature = _fit_temperature(
        validation_logits,
        validation_labels,
    )
    calibrated_probabilities = 1.0 / (
        1.0 + np.exp(-validation_logits / temperature)
    )
    validation_metrics = _classification_metrics(
        validation_labels.astype(np.int64),
        calibrated_probabilities,
    )

    model_path = Path(model_path).expanduser().resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": "supervised_real_fake_v1",
        "feature_version": FEATURE_VERSION,
        "config": {
            "num_frames": int(num_frames),
            "frame_size": int(frame_size),
            "input_dim": int(train_normalized.shape[1]),
            "frame_feature_dim": FRAME_FEATURE_DIM,
            "temporal_feature_dim": TEMPORAL_FEATURE_DIM,
        },
        "model_state": model.state_dict(),
        "feature_mean": mean,
        "feature_scale": scale,
        "temperature": float(temperature),
        "dataset": {
            "real_dir": str(Path(real_dir).expanduser().resolve()),
            "fake_dir": str(Path(fake_dir).expanduser().resolve()),
            "real_count": len(real_valid),
            "fake_count": len(fake_valid),
            "train_real_count": len(train_real_features),
            "train_fake_count": len(train_fake_features),
            "validation_real_count": len(validation_real_features),
            "validation_fake_count": len(validation_fake_features),
            "skipped": (
                real_errors
                + fake_errors
                + missing_real_train
                + missing_fake_train
                + missing_real_val
                + missing_fake_val
            ),
        },
        "training": {
            "epochs": int(epochs),
            "history_tail": history[-10:],
            "validation_metrics": validation_metrics,
            "temperature": float(temperature),
        },
    }
    torch.save(checkpoint, model_path)
    return {
        "model_path": str(model_path),
        "real_count": len(real_valid),
        "fake_count": len(fake_valid),
        "train_real_count": len(train_real_features),
        "train_fake_count": len(train_fake_features),
        "validation_real_count": len(validation_real_features),
        "validation_fake_count": len(validation_fake_features),
        "temperature": float(temperature),
        "validation_metrics": validation_metrics,
        "epochs": int(epochs),
    }


def predict_supervised_video(
        video_path: str | Path,
        model_path: str | Path,
) -> Dict[str, Any]:
    """Predict with the supervised real/fake temporal network."""

    video_path = Path(video_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    checkpoint = torch.load(str(model_path), map_location="cpu")
    config = checkpoint["config"]
    feature = extract_video_feature(
        video_path,
        num_frames=int(config["num_frames"]),
        frame_size=int(config["frame_size"]),
    )
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
    normalized = ((feature - mean) / np.maximum(scale, 1e-4)).astype(np.float32)
    model = RealFakeVideoNetwork(
        num_frames=int(config["num_frames"]),
        frame_feature_dim=int(config["frame_feature_dim"]),
        temporal_feature_dim=int(config["temporal_feature_dim"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    logit = float(_predict_classifier_logits(model, normalized[None, :])[0])
    temperature = max(
        float(checkpoint.get("temperature", 1.0)),
        MIN_CALIBRATION_TEMPERATURE,
    )
    fake_probability = _smooth_probability(
        _sigmoid(logit / temperature)
    )
    real_probability = round(1.0 - fake_probability, 4)
    if fake_probability >= 0.65:
        prediction = "generated"
        label = "更可能是生成视频"
    elif real_probability >= 0.65:
        prediction = "real"
        label = "更可能是真实视频"
    else:
        prediction = "uncertain"
        label = "暂无法区分"
    return {
        "预测": prediction,
        "标签": label,
        "生成概率": fake_probability,
        "真实概率": real_probability,
        "证据强度": round(0.45 + abs(fake_probability - 0.5) * 1.1, 4),
        "结论": (
            f"监督式神经网络判定：当前视频{label}，生成概率 "
            f"{fake_probability * 100.0:.1f}%，真实概率 "
            f"{real_probability * 100.0:.1f}%。"
        ),
        "证据": [
            {
                "指标": "真实/生成监督分类器",
                "指标得分": round(real_probability * 100.0, 2),
                "方向": "偏向生成" if fake_probability >= 0.5 else "偏向真实",
                "说明": f"校准温度={temperature:.4f}，分类 logit={logit:.4f}",
            }
        ],
        "方法": "空间帧编码器 + 双向 GRU 时序编码 + 温度校准监督分类",
        "说明": (
            "模型使用真实视频和 WangXing_Seedance 假视频监督训练；"
            "概率已在分组留出验证集上做温度校准。"
        ),
        "模型路径": str(model_path),
        "真实数据目录": checkpoint["dataset"]["real_dir"],
        "假数据目录": checkpoint["dataset"]["fake_dir"],
        "真实训练视频数": checkpoint["dataset"]["train_real_count"],
        "假训练视频数": checkpoint["dataset"]["train_fake_count"],
        "真实验证视频数": checkpoint["dataset"]["validation_real_count"],
        "假验证视频数": checkpoint["dataset"]["validation_fake_count"],
        "验证指标": checkpoint["training"]["validation_metrics"],
        "模型版本": checkpoint.get("feature_version", FEATURE_VERSION),
    }


def predict_any_video(
        video_path: str | Path,
        model_path: str | Path,
) -> Dict[str, Any]:
    """Dispatch between the new supervised model and the legacy one-class model."""

    checkpoint = torch.load(str(Path(model_path)), map_location="cpu")
    if checkpoint.get("model_type") == "supervised_real_fake_v1":
        return predict_supervised_video(video_path, model_path)
    return predict_video(video_path, model_path)
