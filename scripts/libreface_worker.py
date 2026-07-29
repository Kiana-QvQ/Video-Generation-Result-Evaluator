from __future__ import annotations

import argparse
import os
from pathlib import Path


def _normalise_frame_paths(libreface_module: object) -> None:
    original = libreface_module.get_frames_from_video_ffmpeg

    def fixed_frame_loader(*args: object, **kwargs: object):
        frames = original(*args, **kwargs)
        if "path_to_frame" in frames:
            frames = frames.copy()
            frames["path_to_frame"] = frames["path_to_frame"].map(
                lambda value: str(value).replace("\\", "/")
            )
        return frames

    libreface_module.get_frames_from_video_ffmpeg = fixed_frame_loader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--temp", required=True)
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", args.temp)
    cache_root = weights_dir / "cache"
    huggingface_root = weights_dir / "huggingface"
    cache_root.mkdir(parents=True, exist_ok=True)
    huggingface_root.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(weights_dir)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ["HF_HOME"] = str(huggingface_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(
        huggingface_root / "hub"
    )
    os.environ["TRANSFORMERS_CACHE"] = str(
        huggingface_root / "transformers"
    )
    # LibreFace's bundled gdown resolves its cache at import time and does
    # not create the parent directory before saving cookies on Windows.
    gdown_home = weights_dir / "gdown_home"
    (gdown_home / ".cache" / "gdown").mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(gdown_home)
    os.environ["USERPROFILE"] = str(gdown_home)
    import libreface

    _normalise_frame_paths(libreface)
    result = libreface.get_facial_attributes(
        args.input_path,
        output_save_path=args.output_path,
        model_choice="joint_au_detection_intensity_estimator",
        temp_dir=args.temp,
        device=args.device,
        batch_size=max(1, int(args.batch_size)),
        num_workers=max(0, int(args.num_workers)),
        weights_download_dir=args.weights_dir,
    )
    if result is False or not Path(args.output_path).is_file():
        raise RuntimeError(
            "LibreFace returned without writing an AU CSV."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
