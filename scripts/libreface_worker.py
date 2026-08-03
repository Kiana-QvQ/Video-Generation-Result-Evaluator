from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _normalise_frame_paths(libreface_module: object) -> None:
    original = libreface_module.get_frames_from_video_ffmpeg

    def fixed_frame_loader(*args: object, **kwargs: object):
        frames = original(*args, **kwargs)
        if "path_to_frame" in frames:
            frames = frames.copy()
            frames["path_to_frame"] = frames["path_to_frame"].map(
                lambda value: str(value).replace("\\", "/")
            )
        time_field = next(
            (
                name
                for name in frames.columns
                if str(name).casefold() == "frame_time_in_ms"
            ),
            None,
        )
        if time_field is not None:
            try:
                values = frames[time_field].astype(float)
                if (
                    not values.empty
                    and float(values.max())
                    < max(100.0, len(values) * 2.0)
                ):
                    frames = frames.copy()
                    frames[time_field] = values * 1000.0
            except (TypeError, ValueError):
                pass
        return frames

    libreface_module.get_frames_from_video_ffmpeg = fixed_frame_loader


class _InsightFaceCropFallback:
    """Detect difficult poses and create a square crop for LibreFace."""

    def __init__(self, *, model_root: Path, temp_dir: Path) -> None:
        from insightface.app import FaceAnalysis

        model_dir = model_root / "models" / "buffalo_l"
        detector_weights = model_dir / "det_10g.onnx"
        if not detector_weights.is_file():
            raise FileNotFoundError(
                f"InsightFace detector weights were not found: {detector_weights}"
            )

        self._temp_dir = temp_dir / "insightface_fallback"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._detect_interval = 6
        self._last_bbox: tuple[float, float, float, float] | None = None
        self._last_score = float("nan")
        self._app = FaceAnalysis(
            name="buffalo_l",
            root=str(model_root),
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(
            ctx_id=-1,
            det_thresh=0.35,
            det_size=(320, 320),
        )

    def align(
        self,
        image_path: str,
        *,
        frame_index: int,
    ) -> tuple[str, dict[str, float | str], dict[str, float]] | None:
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            return None

        if (
            self._last_bbox is None
            or frame_index % self._detect_interval == 0
        ):
            faces = self._app.get(image, max_num=1)
            if not faces:
                self._last_bbox = None
                return None
            face = faces[0]
            score = float(face.det_score)
            if score < 0.35:
                self._last_bbox = None
                return None
            self._last_bbox = tuple(
                float(value) for value in face.bbox.tolist()
            )
            self._last_score = score
        else:
            score = self._last_score
        if self._last_bbox is None:
            return None

        height, width = image.shape[:2]
        left, top, right, bottom = self._last_bbox
        box_width = max(right - left, 1.0)
        box_height = max(bottom - top, 1.0)
        # Side-profile boxes often include the neck. A square crop keeps the
        # visible face large enough for LibreFace without inventing landmarks.
        side = int(round(max(box_width * 2.4, box_height * 0.92)))
        side = max(128, min(side, max(width, height) * 2))
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        x0 = int(round(center_x - side / 2.0))
        y0 = int(round(center_y - side / 2.0))
        x1 = x0 + side
        y1 = y0 + side

        pad = max(4, int(round(side * 0.08)))
        padded = cv2.copyMakeBorder(
            image,
            max(0, -y0 - pad),
            max(0, y1 + pad - height),
            max(0, -x0 - pad),
            max(0, x1 + pad - width),
            cv2.BORDER_REPLICATE,
        )
        offset_x = max(0, -x0 - pad)
        offset_y = max(0, -y0 - pad)
        crop = padded[
            offset_y : offset_y + side + 2 * pad,
            offset_x : offset_x + side + 2 * pad,
        ]
        crop = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)

        output_path = self._temp_dir / f"fallback_{frame_index:09d}.png"
        if not cv2.imwrite(str(output_path), crop):
            return None

        metadata: dict[str, float | str] = {
            "pitch": float("nan"),
            "yaw": float("nan"),
            "roll": float("nan"),
            "face_alignment_method": "insightface_bbox",
            "face_detection_score": score,
            "face_bbox_xmin": max(0.0, min(float(width), left)),
            "face_bbox_ymin": max(0.0, min(float(height), top)),
            "face_bbox_xmax": max(0.0, min(float(width), right)),
            "face_bbox_ymax": max(0.0, min(float(height), bottom)),
        }
        return str(output_path), metadata, {}


def _get_aligned_video_frames_with_fallback(
    libreface_module: object,
    frames_df: object,
    *,
    temp_dir: str,
    face_fallback: str,
    face_fallback_first: bool,
) -> tuple[list[str], list[dict[str, object]], list[dict[str, float]]]:
    """Keep MediaPipe as the primary aligner and recover dropped frames."""
    import pandas as pd
    from tqdm import tqdm

    get_aligned_image = libreface_module.get_aligned_image
    fallback: _InsightFaceCropFallback | None = None
    if face_fallback == "insightface":
        try:
            fallback = _InsightFaceCropFallback(
                model_root=PROJECT_ROOT / "model_cache" / "insightface",
                temp_dir=Path(temp_dir),
            )
        except Exception as exc:
            print(
                f"InsightFace fallback unavailable: {type(exc).__name__}: {exc}"
            )

    aligned_paths: list[str] = []
    headpose_list: list[dict[str, object]] = []
    landmark_list: list[dict[str, float]] = []
    indexes_to_drop: list[int] = []
    for index, row in tqdm(
        frames_df.iterrows(),
        desc="Aligning face for video frames...",
    ):
        image_path = str(row["path_to_frame"])
        if face_fallback_first and fallback is not None:
            recovered = fallback.align(
                image_path,
                frame_index=int(row["frame_idx"]),
            )
            if recovered is not None:
                aligned_image_path, headpose, landmarks = recovered
                aligned_paths.append(aligned_image_path)
                headpose_list.append(headpose)
                landmark_list.append(landmarks)
                continue
        try:
            aligned_image_path, headpose, landmarks = get_aligned_image(
                image_path,
                temp_dir=os.path.join(temp_dir, "mediapipe"),
            )
            headpose = {
                **headpose,
                "face_alignment_method": "mediapipe",
                "face_detection_score": float("nan"),
            }
            aligned_paths.append(aligned_image_path)
            headpose_list.append(headpose)
            landmark_list.append(landmarks)
            continue
        except Exception as exc:
            if fallback is None:
                indexes_to_drop.append(index)
                continue
            recovered = fallback.align(
                image_path,
                frame_index=int(row["frame_idx"]),
            )
            if recovered is None:
                print(
                    f"Frame {row['frame_idx']} could not be aligned: "
                    f"{type(exc).__name__}: {exc}"
                )
                indexes_to_drop.append(index)
                continue
            aligned_image_path, headpose, landmarks = recovered
            aligned_paths.append(aligned_image_path)
            headpose_list.append(headpose)
            landmark_list.append(landmarks)

    if indexes_to_drop:
        frames_df.drop(index=indexes_to_drop, inplace=True)
    frames_df.reset_index(drop=True, inplace=True)
    if indexes_to_drop:
        print(
            f"Dropped {len(indexes_to_drop)} frames because no face was "
            "detected by MediaPipe or InsightFace."
        )
    if frames_df.empty:
        print("No face detected in the provided video")
    if len(frames_df) != len(aligned_paths):
        raise RuntimeError(
            "Aligned frame metadata is out of sync with the input frame table."
        )
    if not isinstance(frames_df, pd.DataFrame):
        raise TypeError("LibreFace frame table is not a pandas DataFrame.")
    return aligned_paths, headpose_list, landmark_list


def _get_au_attributes_video(
    libreface_module: object,
    video_path: str,
    *,
    temp_dir: str,
    weights_dir: str,
    device: str,
    batch_size: int,
    num_workers: int,
) -> object:
    """Run only the AU models required by this extraction pipeline."""
    import pandas as pd

    frames_df = libreface_module.get_frames_from_video_ffmpeg(
        video_path,
        temp_dir=temp_dir,
    )
    video_name = Path(video_path).stem
    aligned_paths, headpose_list, landmark_list = (
        libreface_module.get_aligned_video_frames(
            frames_df,
            temp_dir=os.path.join(temp_dir, video_name),
        )
    )
    if frames_df.empty:
        raise RuntimeError("No face detected in the video.")

    frames_df = frames_df.drop("path_to_frame", axis=1)
    frames_df["headpose"] = headpose_list
    frames_df = frames_df.join(
        pd.json_normalize(frames_df["headpose"])
    ).drop("headpose", axis="columns")
    if any(landmark_list):
        frames_df["landmarks_3d"] = landmark_list
        frames_df = frames_df.join(
            pd.json_normalize(frames_df["landmarks_3d"])
        ).drop("landmarks_3d", axis="columns")

    detected_aus, au_intensities = (
        libreface_module.get_au_intensities_and_detect_aus_video(
            aligned_paths,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            weights_download_dir=weights_dir,
        )
    )
    return frames_df.join(detected_aus).join(au_intensities)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--temp", required=True)
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--face-fallback",
        choices=("none", "insightface"),
        default="insightface",
    )
    parser.add_argument(
        "--face-fallback-first",
        action="store_true",
        help="Try the local fallback before MediaPipe for difficult-pose retries.",
    )
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
    if args.face_fallback != "none":
        libreface.get_aligned_video_frames = (
            lambda frames_df, temp_dir=args.temp: (
                _get_aligned_video_frames_with_fallback(
                    libreface,
                    frames_df,
                    temp_dir=temp_dir,
                    face_fallback=args.face_fallback,
                    face_fallback_first=args.face_fallback_first,
                )
            )
        )
    result = _get_au_attributes_video(
        libreface,
        args.input_path,
        temp_dir=args.temp,
        weights_dir=args.weights_dir,
        device=args.device,
        batch_size=max(1, int(args.batch_size)),
        num_workers=max(0, int(args.num_workers)),
    )
    result.to_csv(args.output_path, index=False)
    if not Path(args.output_path).is_file():
        raise RuntimeError(
            "AU extraction completed without writing an AU CSV."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
