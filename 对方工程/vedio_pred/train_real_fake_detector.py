from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .real_video_detector import train_real_fake_detector
except ImportError:
    from real_video_detector import train_real_fake_detector


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="使用真实视频和假视频训练监督式真假鉴别网络"
    )
    parser.add_argument(
        "--real-dir",
        default=str(base_dir / "MD_CL"),
        help="真实视频目录，默认 video_pred/MD_CL",
    )
    parser.add_argument(
        "--fake-dir",
        default=str(base_dir / "WangXing_Seedance"),
        help="假视频目录，默认 video_pred/WangXing_Seedance",
    )
    parser.add_argument(
        "--model-path",
        default=str(base_dir / "models" / "real_fake_video_classifier.pt"),
        help="输出模型路径",
    )
    parser.add_argument(
        "--real-cache",
        default=str(base_dir / "cache" / "real_features_f8_s48.npz"),
        help="真实视频特征缓存",
    )
    parser.add_argument(
        "--fake-cache",
        default=str(base_dir / "cache" / "fake_features_f8_s48.npz"),
        help="假视频特征缓存",
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-real-videos", type=int, default=None)
    parser.add_argument("--max-fake-videos", type=int, default=None)
    args = parser.parse_args()

    result = train_real_fake_detector(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        model_path=args.model_path,
        real_cache_path=args.real_cache,
        fake_cache_path=args.fake_cache,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_ratio=args.validation_ratio,
        hidden_seed=args.seed,
        max_real_videos=args.max_real_videos,
        max_fake_videos=args.max_fake_videos,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
