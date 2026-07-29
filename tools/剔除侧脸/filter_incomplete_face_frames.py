"""Standalone ArcFace frame filter.

The script has no dependency on the project evaluator. It uses InsightFace
FaceAnalysis (detector, five facial keypoints, and ArcFace recognition model)
to remove frames that do not contain sufficiently complete facial evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class FilterConfig:
    model_name: str = "buffalo_l"
    model_root: Path = Path("models")
    device: str = "auto"
    det_size: int = 640
    min_detection_score: float = 0.60
    min_keypoint_coverage: float = 1.00
    min_face_area_ratio: float = 0.01
    min_border_margin_ratio: float = 0.01
    min_symmetry: float = 0.75
    min_eye_separation_ratio: float = 0.30
    min_quality: float = 0.60
    min_sharpness: float = 0.0


@dataclass
class FrameAudit:
    frame_index: int
    timestamp_seconds: float
    keep: bool
    quality_score: float
    reasons: list[str]
    face_count: int
    face_bbox: list[int] | None
    detection_score: float
    keypoint_coverage: float
    face_area_ratio: float
    border_margin_ratio: float
    symmetry_score: float
    eye_separation_ratio: float
    arcface_embedding_valid: bool
    sharpness: float


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _ratio_score(value: float, minimum: float) -> float:
    if minimum <= 0.0:
        return 1.0
    return _clamp(value / (minimum * 2.0))


def _sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _bbox_values(
    bbox: Any,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float] | None:
    values = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if len(values) < 4:
        return None
    x0, y0, x1, y1 = [float(value) for value in values[:4]]
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        return None
    return x0, y0, x1, y1


def _public_bbox(
    bbox: tuple[float, float, float, float] | None,
    frame_width: int,
    frame_height: int,
) -> list[int] | None:
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    left = max(0, min(frame_width - 1, int(round(x0))))
    top = max(0, min(frame_height - 1, int(round(y0))))
    right = max(left + 1, min(frame_width, int(round(x1))))
    bottom = max(top + 1, min(frame_height, int(round(y1))))
    return [left, top, right - left, bottom - top]


def _keypoint_coverage(
    keypoints: Any,
    frame_width: int,
    frame_height: int,
) -> tuple[float, np.ndarray | None]:
    if keypoints is None:
        return 0.0, None
    points = np.asarray(keypoints, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 5 or points.shape[1] < 2:
        return 0.0, None
    points = points[:5, :2]
    finite = np.isfinite(points).all(axis=1)
    inside = (
        finite
        & (points[:, 0] >= 0.0)
        & (points[:, 0] < float(frame_width))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < float(frame_height))
    )
    return float(np.mean(inside)), points


def _symmetry_score(keypoints: np.ndarray | None) -> float:
    if keypoints is None or len(keypoints) < 5:
        return 0.0
    # InsightFace five-point order: left eye, right eye, nose, left mouth,
    # right mouth. A strong profile usually compresses one side.
    nose = keypoints[2]
    left = float(
        np.mean(
            [
                np.linalg.norm(nose - keypoints[0]),
                np.linalg.norm(nose - keypoints[3]),
            ]
        )
    )
    right = float(
        np.mean(
            [
                np.linalg.norm(nose - keypoints[1]),
                np.linalg.norm(nose - keypoints[4]),
            ]
        )
    )
    if left <= 1e-6 or right <= 1e-6:
        return 0.0
    return _clamp(min(left, right) / max(left, right))


def _eye_separation_ratio(
    keypoints: np.ndarray | None,
    bbox: tuple[float, float, float, float] | None,
) -> float:
    if keypoints is None or len(keypoints) < 2 or bbox is None:
        return 0.0
    face_width = max(float(bbox[2] - bbox[0]), 1.0)
    eye_distance = float(np.linalg.norm(keypoints[0] - keypoints[1]))
    return _clamp(eye_distance / face_width)


def _resolve_device(device: str) -> tuple[str, list[str]]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is not installed. Install "
            "tools/face_filter_requirements.txt."
        ) from exc

    available = set(ort.get_available_providers())
    cuda_available = "CUDAExecutionProvider" in available
    normalized = device.lower()
    if normalized == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested but ONNX Runtime has no "
            "CUDAExecutionProvider. Use --device cpu or install "
            "onnxruntime-gpu."
        )
    use_cuda = normalized == "cuda" or (
        normalized == "auto" and cuda_available
    )
    if use_cuda:
        return "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return "cpu", ["CPUExecutionProvider"]


class ArcFaceCompletenessDetector:
    backend = "insightface_arcface"

    def __init__(self, config: FilterConfig) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "insightface is not installed. Install "
                "tools/face_filter_requirements.txt."
            ) from exc

        self.device, providers = _resolve_device(config.device)
        self.model_name = config.model_name
        self.model_root = Path(config.model_root).expanduser().resolve()
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.app = FaceAnalysis(
            name=config.model_name,
            root=str(self.model_root),
            providers=providers,
        )
        self.app.prepare(
            ctx_id=0 if self.device == "cuda" else -1,
            det_size=(int(config.det_size), int(config.det_size)),
        )
        self.providers = providers

    def close(self) -> None:
        return None

    def evaluate(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_seconds: float,
        config: FilterConfig,
    ) -> FrameAudit:
        height, width = frame.shape[:2]
        sharpness = _sharpness(frame)
        faces = self.app.get(frame)
        if not faces:
            return FrameAudit(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                keep=False,
                quality_score=0.0,
                reasons=["no_face"],
                face_count=0,
                face_bbox=None,
                detection_score=0.0,
                keypoint_coverage=0.0,
                face_area_ratio=0.0,
                border_margin_ratio=0.0,
                symmetry_score=0.0,
                eye_separation_ratio=0.0,
                arcface_embedding_valid=False,
                sharpness=sharpness,
            )

        face = max(
            faces,
            key=lambda item: max(
                0.0,
                float(item.bbox[2] - item.bbox[0]),
            )
            * max(
                0.0,
                float(item.bbox[3] - item.bbox[1]),
            ),
        )
        bbox = _bbox_values(face.bbox, width, height)
        public_bbox = _public_bbox(bbox, width, height)
        if bbox is None:
            face_area_ratio = 0.0
            border_margin_ratio = 0.0
        else:
            x0, y0, x1, y1 = bbox
            box_width = max(0.0, x1 - x0)
            box_height = max(0.0, y1 - y0)
            face_area_ratio = box_width * box_height / float(width * height)
            border_pixels = min(x0, y0, width - x1, height - y1)
            border_margin_ratio = max(
                0.0,
                border_pixels / max(box_width, box_height, 1.0),
            )

        keypoint_coverage, keypoints = _keypoint_coverage(
            getattr(face, "kps", None),
            width,
            height,
        )
        symmetry = _symmetry_score(keypoints)
        eye_separation_ratio = _eye_separation_ratio(keypoints, bbox)
        detection_score = _clamp(
            _safe_float(getattr(face, "det_score", 0.0))
        )
        embedding = getattr(face, "embedding", None)
        embedding_valid = bool(
            embedding is not None
            and np.linalg.norm(np.asarray(embedding, dtype=np.float32)) > 1e-6
        )
        quality_score = float(
            0.30 * detection_score
            + 0.20 * keypoint_coverage
            + 0.15 * _ratio_score(
                face_area_ratio,
                config.min_face_area_ratio,
            )
            + 0.15 * _ratio_score(
                border_margin_ratio,
                config.min_border_margin_ratio,
            )
            + 0.20 * symmetry
        )

        reasons: list[str] = []
        if detection_score < config.min_detection_score:
            reasons.append("low_detection_score")
        if keypoint_coverage < config.min_keypoint_coverage:
            reasons.append("incomplete_keypoints")
        if face_area_ratio < config.min_face_area_ratio:
            reasons.append("face_too_small")
        if border_margin_ratio < config.min_border_margin_ratio:
            reasons.append("face_touches_border")
        if (
            symmetry < config.min_symmetry
            or eye_separation_ratio < config.min_eye_separation_ratio
        ):
            reasons.append("face_too_profile")
        if not embedding_valid:
            reasons.append("no_arcface_embedding")
        if config.min_sharpness > 0.0 and sharpness < config.min_sharpness:
            reasons.append("too_blurry")
        if quality_score < config.min_quality:
            reasons.append("quality_below_threshold")

        return FrameAudit(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            keep=not reasons,
            quality_score=quality_score,
            reasons=reasons,
            face_count=len(faces),
            face_bbox=public_bbox,
            detection_score=detection_score,
            keypoint_coverage=keypoint_coverage,
            face_area_ratio=face_area_ratio,
            border_margin_ratio=border_margin_ratio,
            symmetry_score=symmetry,
            eye_separation_ratio=eye_separation_ratio,
            arcface_embedding_valid=embedding_valid,
            sharpness=sharpness,
        )


def _validate_config(config: FilterConfig) -> None:
    if config.det_size <= 0:
        raise ValueError("det_size must be positive.")
    if not 0.0 <= config.min_detection_score <= 1.0:
        raise ValueError("min_detection_score must be between 0 and 1.")
    if not 0.0 <= config.min_keypoint_coverage <= 1.0:
        raise ValueError("min_keypoint_coverage must be between 0 and 1.")
    if not 0.0 <= config.min_face_area_ratio <= 1.0:
        raise ValueError("min_face_area_ratio must be between 0 and 1.")
    if config.min_border_margin_ratio < 0.0:
        raise ValueError("min_border_margin_ratio cannot be negative.")
    if not 0.0 <= config.min_symmetry <= 1.0:
        raise ValueError("min_symmetry must be between 0 and 1.")
    if not 0.0 <= config.min_eye_separation_ratio <= 1.0:
        raise ValueError("min_eye_separation_ratio must be between 0 and 1.")
    if not 0.0 <= config.min_quality <= 1.0:
        raise ValueError("min_quality must be between 0 and 1.")
    if config.min_sharpness < 0.0:
        raise ValueError("min_sharpness cannot be negative.")


def filter_video(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    config: FilterConfig,
) -> dict[str, Any]:
    _validate_config(config)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output video must be different from the input video.")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open input video: {input_path}")

    fps = _safe_float(capture.get(cv2.CAP_PROP_FPS), default=30.0)
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Input video has invalid dimensions.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0.0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Unable to open output video: {output_path}")

    detector = ArcFaceCompletenessDetector(config)
    records: list[dict[str, Any]] = []
    kept_count = 0
    dropped_count = 0
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            timestamp = frame_index / max(fps, 1e-6)
            audit = detector.evaluate(
                frame,
                frame_index,
                timestamp,
                config,
            )
            records.append(asdict(audit))
            if audit.keep:
                writer.write(frame)
                kept_count += 1
            else:
                dropped_count += 1
            frame_index += 1
    finally:
        detector.close()
        capture.release()
        writer.release()

    if kept_count == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            "No complete-face frames were kept; no filtered video was written."
        )

    reason_counts: Counter[str] = Counter()
    for record in records:
        reason_counts.update(record["reasons"])

    report = {
        "schema_version": "face_filter_arcface_v1",
        "backend": detector.backend,
        "model": {
            "name": config.model_name,
            "root": str(config.model_root),
            "device": detector.device,
            "providers": detector.providers,
        },
        "input": str(input_path),
        "output": str(output_path),
        "video": {
            "fps": fps,
            "frame_count_reported": frame_count,
            "frame_count_read": frame_index,
            "width": width,
            "height": height,
        },
        "filter_config": {
            **asdict(config),
            "model_root": str(config.model_root),
        },
        "summary": {
            "kept_frames": kept_count,
            "dropped_frames": dropped_count,
            "kept_ratio": kept_count / max(frame_index, 1),
            "dropped_ratio": dropped_count / max(frame_index, 1),
            "reason_counts": dict(reason_counts),
            "audio_preserved": False,
        },
        "frames": records,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove incomplete-face frames with standalone ArcFace/"
            "InsightFace."
        )
    )
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Filtered video path.")
    parser.add_argument(
        "--report",
        help="JSON report path. Defaults to <output>.face_filter.json.",
    )
    parser.add_argument(
        "--model-name",
        default="buffalo_l",
        help="InsightFace model pack. buffalo_l is the accuracy-first default.",
    )
    parser.add_argument(
        "--model-root",
        default="models",
        help="Directory used for InsightFace model files.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=640,
        help="Square detector input size.",
    )
    parser.add_argument(
        "--min-detection-score",
        type=float,
        default=0.60,
        help="Minimum ArcFace/InsightFace detector confidence.",
    )
    parser.add_argument(
        "--min-keypoint-coverage",
        type=float,
        default=1.00,
        help="Required ratio of the five keypoints inside the frame.",
    )
    parser.add_argument(
        "--min-face-area-ratio",
        type=float,
        default=0.01,
        help="Minimum face bbox area divided by image area.",
    )
    parser.add_argument(
        "--min-border-margin-ratio",
        type=float,
        default=0.01,
        help="Minimum face-to-border distance relative to face size.",
    )
    parser.add_argument(
        "--min-symmetry",
        type=float,
        default=0.75,
        help="Minimum five-point left/right symmetry; rejects strong profiles.",
    )
    parser.add_argument(
        "--min-eye-separation-ratio",
        type=float,
        default=0.30,
        help="Minimum distance between the two eyes divided by face width.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.60,
        help="Minimum combined completeness quality.",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=0.0,
        help="Optional Laplacian sharpness gate. Zero disables it.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    output_path = Path(args.output)
    config = FilterConfig(
        model_name=args.model_name,
        model_root=Path(args.model_root),
        device=args.device,
        det_size=args.det_size,
        min_detection_score=args.min_detection_score,
        min_keypoint_coverage=args.min_keypoint_coverage,
        min_face_area_ratio=args.min_face_area_ratio,
        min_border_margin_ratio=args.min_border_margin_ratio,
        min_symmetry=args.min_symmetry,
        min_eye_separation_ratio=args.min_eye_separation_ratio,
        min_quality=args.min_quality,
        min_sharpness=args.min_sharpness,
    )
    report_path = (
        Path(args.report)
        if args.report
        else output_path.with_suffix(".face_filter.json")
    )
    report = filter_video(
        Path(args.input),
        output_path,
        report_path,
        config=config,
    )
    summary = report["summary"]
    print(
        f"Filtered video written to {output_path}\n"
        f"Kept {summary['kept_frames']} / "
        f"{report['video']['frame_count_read']} frames "
        f"({summary['kept_ratio']:.1%}).\n"
        f"Report written to {report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
