from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .real_video_detector import train_real_video_detector
except ImportError:
    from real_video_detector import train_real_video_detector


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="只使用真实视频训练轻量 One-Class 真伪检测网络"
    )
    parser.add_argument(
        "--data-dir",
        default=str(base_dir / "MD_CL"),
        help="真实视频目录，默认 video/MD_CL",
    )
    parser.add_argument(
        "--model-path",
        default=str(base_dir / "models" / "real_video_autoencoder.pt"),
        help="模型输出路径，默认保存在 video/models/",
    )
    parser.add_argument(
        "--cache-path",
        default=str(base_dir / "cache" / "real_features.npz"),
        help="特征缓存路径，默认保存在 video/cache/",
    )
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--frame-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="调试时限制视频数量；正式训练不要设置",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = train_real_video_detector(
        data_dir=args.data_dir,
        model_path=args.model_path,
        cache_path=args.cache_path,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_ratio=args.validation_ratio,
        max_videos=args.max_videos,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

