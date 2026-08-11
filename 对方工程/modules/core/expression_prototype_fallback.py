"""Expression-directory fallback when AU CSV is unavailable.

Collaborator host apps keep compressed expression prototypes under
``Expression/`` next to ``modules/``. Without an AU CSV the Wang Xing AU
path cannot score; this module matches generated frames against those
prototypes and returns UI-compatible radar fields.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .paths import PACKAGE_ROOT, WORKSPACE_ROOT

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAME_PATTERN = re.compile(
    r"^(?P<action>.+?)_(?P<view>Front|Left|Right)(?:\.[^.]+)?$",
    re.IGNORECASE,
)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def resolve_expression_dir(
    explicit: str | Path | None = None,
) -> Path | None:
    """Find the collaborator ``Expression`` asset directory."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for root in (PACKAGE_ROOT, WORKSPACE_ROOT, Path.cwd()):
        candidates.append(root / "Expression")
        candidates.append(root / "BiaoQing")
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_dir() and any(
            item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
            for item in path.iterdir()
        ):
            return path.resolve()
    return None


def _parse_name(path: Path) -> tuple[str, str]:
    match = NAME_PATTERN.match(path.stem)
    if match is None:
        return path.stem, "Unknown"
    return match.group("action"), match.group("view").capitalize()


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    # OpenCV loads BGR; accept either by leaving caller responsible.
    return np.ascontiguousarray(image)


def _read_bgr(path: Path) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None
    # cv2.imread fails on non-ASCII paths on some Windows builds.
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return image


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    try:
        import cv2

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        return image[:, :, ::-1]


def _center_square_crop(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = min(height, width)
    y0 = max(0, (height - side) // 2)
    x0 = max(0, (width - side) // 2)
    return image[y0 : y0 + side, x0 : x0 + side]


def _haar_cascade_path() -> Path | None:
    """Resolve Haar XML, copying to ASCII temp when the venv path is non-ASCII."""
    try:
        import cv2
    except ImportError:
        return None
    source = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not source.is_file():
        return None
    if not any(ord(char) > 127 for char in str(source)):
        return source
    import tempfile

    cache = Path(tempfile.gettempdir()) / "video_evaluator_opencv"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / source.name
    if not target.is_file() or target.stat().st_size != source.stat().st_size:
        target.write_bytes(source.read_bytes())
    return target


def _face_crop_bgr(image: np.ndarray) -> np.ndarray:
    """Crop a face ROI; fall back to center crop if Haar is unavailable."""
    try:
        import cv2
    except ImportError:
        return image
    try:
        cascade_path = _haar_cascade_path()
        if cascade_path is None:
            return _center_square_crop(image)
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            return _center_square_crop(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(48, 48))
        if faces is None or len(faces) == 0:
            return _center_square_crop(image)
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        pad = int(0.18 * max(w, h))
        y0 = max(0, y - pad)
        x0 = max(0, x - pad)
        y1 = min(image.shape[0], y + h + pad)
        x1 = min(image.shape[1], x + w + pad)
        return image[y0:y1, x0:x1]
    except Exception:
        return _center_square_crop(image)


def _texture_vector(face_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import cv2
    except ImportError:
        flat = np.asarray(face_bgr, dtype=np.float32).reshape(-1)
        return flat[:64], flat[:64], np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    face = cv2.resize(face_bgr, (96, 96), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    detail = gray - blur
    structure = cv2.resize(blur, (8, 8)).reshape(-1)
    detail_hist = cv2.resize(np.abs(detail), (8, 8)).reshape(-1)
    regions = []
    for y0, y1, x0, x1 in (
        (8, 40, 12, 44),
        (8, 40, 52, 84),
        (40, 68, 24, 72),
        (60, 88, 20, 76),
    ):
        patch = detail[y0:y1, x0:x1]
        regions.append(float(np.mean(np.abs(patch))))
    return (
        structure.astype(np.float32),
        detail_hist.astype(np.float32),
        np.asarray(regions, dtype=np.float32),
    )


def _geometry_from_landmarks(landmarks: np.ndarray) -> np.ndarray | None:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 468:
        return None
    pairs = (
        (33, 263),
        (61, 291),
        (13, 14),
        (70, 300),
        (107, 336),
        (1, 152),
        (78, 308),
    )
    eye_width = float(np.linalg.norm(points[263, :2] - points[33, :2]))
    scale = max(eye_width, 1e-6)
    values: list[float] = []
    for left, right in pairs:
        values.append(float(np.linalg.norm(points[left, :2] - points[right, :2]) / scale))
    mouth = points[[61, 291, 13, 14, 78, 308], :2]
    values.extend(mouth.reshape(-1).tolist())
    brow = points[[70, 105, 107, 336, 334, 300], :2]
    values.extend(((brow - points[1, :2]) / scale).reshape(-1).tolist())
    return np.asarray(values, dtype=np.float32)


def _gaze_from_landmarks(landmarks: np.ndarray) -> np.ndarray | None:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 478:
        if points.ndim == 2 and points.shape[0] >= 468:
            left = 0.5 * (points[33, :2] + points[133, :2])
            right = 0.5 * (points[362, :2] + points[263, :2])
            return np.asarray(
                [
                    float((left[0] + right[0]) * 0.5 - points[1, 0]),
                    float((left[1] + right[1]) * 0.5 - points[1, 1]),
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
        return None
    left_iris = points[list(range(468, 473)), :2].mean(axis=0)
    right_iris = points[list(range(473, 478)), :2].mean(axis=0)
    left_eye = 0.5 * (points[33, :2] + points[133, :2])
    right_eye = 0.5 * (points[362, :2] + points[263, :2])
    left_w = max(float(np.linalg.norm(points[33, :2] - points[133, :2])), 1e-6)
    right_w = max(float(np.linalg.norm(points[362, :2] - points[263, :2])), 1e-6)
    return np.asarray(
        [
            float((left_iris[0] - left_eye[0]) / left_w),
            float((left_iris[1] - left_eye[1]) / left_w),
            float((right_iris[0] - right_eye[0]) / right_w),
            float((right_iris[1] - right_eye[1]) / right_w),
        ],
        dtype=np.float32,
    )


def _estimate_view(landmarks: np.ndarray | None) -> str:
    if landmarks is None or len(landmarks) < 264:
        return "Unknown"
    points = np.asarray(landmarks, dtype=np.float32)
    inter_eye = float(points[263, 0] - points[33, 0])
    if abs(inter_eye) < 1e-6:
        return "Unknown"
    eye_mid = float((points[33, 0] + points[263, 0]) / 2.0)
    nose_offset = float((points[1, 0] - eye_mid) / abs(inter_eye))
    if nose_offset < -0.08:
        return "Left"
    if nose_offset > 0.08:
        return "Right"
    return "Front"


def _cosine_to_score(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    left = np.asarray(a, dtype=np.float32).reshape(-1)
    right = np.asarray(b, dtype=np.float32).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        return None
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < 1e-8:
        return None
    return _clamp((float(np.dot(left, right) / denom) + 1.0) / 2.0)


def _feature_similarity(
    a: np.ndarray | None,
    b: np.ndarray | None,
    scale: float,
) -> float | None:
    if a is None or b is None:
        return None
    left = np.asarray(a, dtype=np.float32).reshape(-1)
    right = np.asarray(b, dtype=np.float32).reshape(-1)
    if left.shape != right.shape:
        return None
    error = float(np.mean(np.abs(left - right) / max(scale, 1e-3)))
    return _clamp(math.exp(-error))


@dataclass
class _Prototype:
    path: str
    action: str
    view: str
    geometry: np.ndarray | None
    gaze: np.ndarray | None
    structure: np.ndarray
    detail: np.ndarray
    region_energy: np.ndarray


@dataclass
class _FrameFeat:
    geometry: np.ndarray | None
    gaze: np.ndarray | None
    structure: np.ndarray
    detail: np.ndarray
    region_energy: np.ndarray
    view: str


def _extract_frame_features(
    frame_bgr: np.ndarray,
    normalizer: Any | None,
    *,
    timestamp_ms: int,
) -> _FrameFeat | None:
    face = _face_crop_bgr(frame_bgr)
    structure, detail, region_energy = _texture_vector(face)
    landmarks = None
    if normalizer is not None and getattr(normalizer, "available", False):
        rgb = _bgr_to_rgb(frame_bgr)
        result = normalizer.process_frame(rgb, timestamp_ms=timestamp_ms)
        if result is not None and result.landmarks_pose_normalized.size:
            landmarks = result.landmarks_pose_normalized
    return _FrameFeat(
        geometry=_geometry_from_landmarks(landmarks) if landmarks is not None else None,
        gaze=_gaze_from_landmarks(landmarks) if landmarks is not None else None,
        structure=structure,
        detail=detail,
        region_energy=region_energy,
        view=_estimate_view(landmarks),
    )


def _load_prototypes(
    expression_dir: Path,
    normalizer: Any | None,
    *,
    max_refs: int = 120,
) -> list[_Prototype]:
    paths = sorted(
        [
            path
            for path in expression_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    )[:max_refs]
    prototypes: list[_Prototype] = []
    for index, path in enumerate(paths):
        image = _read_bgr(path)
        if image is None:
            continue
        features = _extract_frame_features(
            image,
            normalizer,
            timestamp_ms=index * 33,
        )
        if features is None:
            continue
        action, view = _parse_name(path)
        prototypes.append(
            _Prototype(
                path=str(path),
                action=action,
                view=view if view != "Unknown" else features.view,
                geometry=features.geometry,
                gaze=features.gaze,
                structure=features.structure,
                detail=features.detail,
                region_energy=features.region_energy,
            )
        )
    return prototypes


def _match_frame(
    frame: _FrameFeat,
    prototypes: Sequence[_Prototype],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for reference in prototypes:
        structure_score = _cosine_to_score(frame.structure, reference.structure) or 0.0
        detail_score = _cosine_to_score(frame.detail, reference.detail) or 0.0
        region_score = _cosine_to_score(
            frame.region_energy,
            reference.region_energy,
        ) or 0.0
        wrinkle_score = _clamp(0.65 * detail_score + 0.35 * region_score)
        texture_score = _clamp(0.55 * structure_score + 0.45 * detail_score)
        geometry_score = _feature_similarity(frame.geometry, reference.geometry, 0.08)
        gaze_score = _feature_similarity(frame.gaze, reference.gaze, 0.12)
        parts: list[tuple[float, float]] = [
            (texture_score, 0.24),
            (wrinkle_score, 0.22),
        ]
        if geometry_score is not None:
            parts.append((geometry_score, 0.36))
        if gaze_score is not None:
            parts.append((gaze_score, 0.18))
        total_weight = sum(weight for _, weight in parts)
        score = _clamp(
            sum(value * weight for value, weight in parts) / max(total_weight, 1e-6)
        )
        if frame.view != "Unknown" and reference.view != "Unknown":
            score = _clamp(score * (0.96 + 0.04 * float(frame.view == reference.view)))
        candidate = {
            "action": reference.action,
            "view": reference.view,
            "path": reference.path,
            "score": score,
            "geometry_score": geometry_score,
            "gaze_score": gaze_score,
            "texture_score": texture_score,
            "wrinkle_score": wrinkle_score,
        }
        if best is None or score > float(best["score"]):
            best = candidate
    return best


def _motion_score(records: Sequence[dict[str, Any]]) -> float:
    if len(records) < 2:
        return 0.55
    geometry_series = [
        np.asarray(item["geometry"], dtype=np.float32).reshape(-1)
        for item in records
        if item.get("geometry") is not None
    ]
    wrinkle_series = [
        float(item["match"].get("wrinkle_score") or 0.0)
        for item in records
    ]
    if len(geometry_series) >= 2:
        deltas = [
            float(np.linalg.norm(geometry_series[index] - geometry_series[index - 1]))
            for index in range(1, len(geometry_series))
        ]
        geometry_stability = _clamp(math.exp(-float(np.mean(deltas)) / 0.35))
    else:
        geometry_stability = 0.55
    if len(wrinkle_series) >= 2:
        wrinkle_deltas = [
            abs(wrinkle_series[index] - wrinkle_series[index - 1])
            for index in range(1, len(wrinkle_series))
        ]
        wrinkle_stability = _clamp(math.exp(-float(np.mean(wrinkle_deltas)) / 0.20))
    else:
        wrinkle_stability = 0.55
    return _clamp(0.55 * geometry_stability + 0.45 * wrinkle_stability)


def score_expression_prototypes(
    frames: Sequence[Any],
    *,
    expression_dir: str | Path | None = None,
    max_frames: int = 16,
    sample_fps: float = 8.0,
) -> dict[str, Any]:
    """Score generated frames against Expression prototypes without AU CSV."""
    del sample_fps  # reserved for API parity with runtime callers
    directory = resolve_expression_dir(expression_dir)
    if directory is None:
        return {
            "score": None,
            "status": "unavailable",
            "details": {
                "warning": "Expression 参考目录不存在或为空",
                "method": "expression_prototype_fallback",
            },
        }
    if not frames:
        return {
            "score": None,
            "status": "unavailable",
            "details": {
                "warning": "没有可用于 Expression 匹配的生成视频帧",
                "expression_dir": str(directory),
                "method": "expression_prototype_fallback",
            },
        }

    limit = max(2, int(max_frames))
    selected = list(frames)
    if len(selected) > limit:
        indexes = [
            int(round(index * (len(selected) - 1) / (limit - 1)))
            for index in range(limit)
        ]
        selected = [selected[index] for index in indexes]

    normalizer = None
    try:
        from .face_landmarker import FacePoseNormalizer

        try:
            candidate = FacePoseNormalizer(download_model=False)
        except Exception:
            candidate = None
        if candidate is not None and getattr(candidate, "available", False):
            normalizer = candidate
        elif candidate is not None:
            candidate.close()
    except Exception:
        normalizer = None

    try:
        prototypes = _load_prototypes(directory, normalizer)
        if not prototypes:
            return {
                "score": None,
                "status": "unavailable",
                "details": {
                    "warning": "Expression 参考图无法提取可比较特征",
                    "expression_dir": str(directory),
                    "method": "expression_prototype_fallback",
                },
            }

        records: list[dict[str, Any]] = []
        for index, frame in enumerate(selected):
            array = np.asarray(frame)
            if array.ndim != 3:
                continue
            # Host ``main.py`` stores OpenCV BGR frames. Keep BGR for OpenCV
            # helpers; convert to RGB only when feeding MediaPipe.
            frame_bgr = np.ascontiguousarray(array[:, :, :3])
            features = _extract_frame_features(
                frame_bgr,
                normalizer,
                timestamp_ms=index * 33,
            )
            if features is None:
                continue
            match = _match_frame(features, prototypes)
            if match is None:
                continue
            records.append(
                {
                    "geometry": features.geometry,
                    "match": match,
                }
            )

        if not records:
            return {
                "score": None,
                "status": "unavailable",
                "details": {
                    "warning": "没有生成视频帧能够匹配 Expression 参考表情",
                    "expression_dir": str(directory),
                    "reference_count": len(prototypes),
                    "method": "expression_prototype_fallback",
                },
            }

        frame_scores = [float(item["match"]["score"]) for item in records]
        frame_match_score = float(np.mean(frame_scores))
        geometry_values = [
            float(item["match"]["geometry_score"])
            for item in records
            if item["match"].get("geometry_score") is not None
        ]
        gaze_values = [
            float(item["match"]["gaze_score"])
            for item in records
            if item["match"].get("gaze_score") is not None
        ]
        wrinkle_values = [
            float(item["match"]["wrinkle_score"])
            for item in records
            if item["match"].get("wrinkle_score") is not None
        ]
        geometry_score = (
            float(np.mean(geometry_values)) if geometry_values else frame_match_score
        )
        gaze_score = float(np.mean(gaze_values)) if gaze_values else frame_match_score
        wrinkle_score = (
            float(np.mean(wrinkle_values)) if wrinkle_values else frame_match_score
        )
        motion_score = _motion_score(records)
        coverage = len(records) / max(len(selected), 1)
        consistency = _clamp(math.exp(-float(np.std(frame_scores)) / 0.15))
        score = _clamp(
            0.50 * frame_match_score
            + 0.15 * consistency
            + 0.20 * motion_score
            + 0.15 * geometry_score
        )
        score = _clamp(0.90 * score + 0.10 * coverage)

        return {
            "score": score,
            "status": "partial",
            "details": {
                "placeholder": False,
                "method": "expression_prototype_fallback",
                "geometry_method": "expression_prototype_match",
                "gaze_method": "expression_prototype_match",
                "expression_dir": str(directory),
                "reference_count": len(prototypes),
                "valid_face_frames": len(records),
                "total_sampled_frames": len(selected),
                "face_coverage": coverage,
                "frame_match_score": frame_match_score,
                "geometry_score": geometry_score,
                "gaze_score": gaze_score,
                "wrinkle_score": wrinkle_score,
                "texture_score": wrinkle_score,
                "motion_score": motion_score,
                "match_consistency_score": consistency,
                "profile_compatibility_0_1": frame_match_score,
                "muscle_action_evidence_0_1": geometry_score,
                "action_coherence_0_1": motion_score,
                "active_ratio_0_1": coverage,
                "landmark_coverage_0_1": coverage,
                "warning": (
                    "未提供 AU CSV，已回退到 Expression 动作原型匹配；"
                    "完整王兴 AU 专项仍建议提供旁路 AU 表。"
                ),
                "reference_used": True,
                "reference_note": (
                    "Scored against local Expression prototypes because AU CSV "
                    "was unavailable."
                ),
            },
        }
    finally:
        if normalizer is not None:
            normalizer.close()
