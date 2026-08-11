from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .real_video_detector import predict_any_video
except ImportError:
    from real_video_detector import predict_any_video


def predict_real_video(
        video_path: str | Path,
        model_path: str | Path | None = None,
) -> dict:
    """Predict the uploaded video with the trained real-video detector."""

    base_dir = Path(__file__).resolve().parent
    resolved_model_path = (
        Path(model_path).expanduser()
        if model_path is not None
        else base_dir / "models" / "real_fake_video_classifier.pt"
    )
    if not resolved_model_path.is_absolute():
        cwd_candidate = (Path.cwd() / resolved_model_path).resolve()
        resolved_model_path = (
            cwd_candidate
            if cwd_candidate.exists()
            else (base_dir / resolved_model_path).resolve()
        )
    return predict_any_video(
        video_path=video_path,
        model_path=resolved_model_path,
    )


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="使用真实视频 One-Class 模型判断输入视频真伪倾向"
    )
    parser.add_argument("--video", required=True, help="待检测视频路径")
    parser.add_argument(
        "--model-path",
        default=str(base_dir / "models" / "real_fake_video_classifier.pt"),
        help="训练好的模型路径",
    )
    args = parser.parse_args()
    result = predict_real_video(
        video_path=args.video,
        model_path=args.model_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
