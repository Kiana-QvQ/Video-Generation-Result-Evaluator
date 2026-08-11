"""MediaPipe Face Landmarker pose normalization helpers.

Uses Tasks FaceLandmarker when a ``.task`` model is available; otherwise falls
back to classic Face Mesh landmarks. Either path produces pose-canonical 3D
landmarks so AU / motion features are less sensitive to head yaw / pitch / roll.

Blendshapes and facial transformation matrices are consumed when the Tasks API
returns them. Iris landmarks are included when refine/iris outputs exist.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

FACE_LANDMARKER_SCHEMA = "face_landmarker_pose_v1"
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# MediaPipe Face Mesh / Face Landmarker anchors for similarity alignment.
# Left/right eye outer corners, nose tip, chin.
POSE_ANCHORS = (33, 263, 1, 152)
IRIS_LEFT = (468, 469, 470, 471, 472)
IRIS_RIGHT = (473, 474, 475, 476, 477)
# Canonical 2D/3D target in a unit face frame after pose normalization.
CANONICAL_ANCHORS = np.asarray(
    [
        [-0.35, 0.05, 0.0],
        [0.35, 0.05, 0.0],
        [0.0, 0.0, 0.12],
        [0.0, 0.55, -0.02],
    ],
    dtype=np.float64,
)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def default_model_path() -> Path:
    """Resolve FaceLandmarker model: package asset → env → temp cache."""
    override = os.environ.get("EVALUATOR_FACE_LANDMARKER_MODEL")
    if override:
        return Path(override).expanduser()
    try:
        from .paths import MODULES_ROOT

        bundled = MODULES_ROOT / "assets" / "models" / "face_landmarker.task"
        if bundled.is_file() and bundled.stat().st_size > 1024:
            return bundled
        # Prefer writing into the shipped assets tree when writable.
        return bundled
    except Exception:
        pass
    cache_root = Path(
        os.environ.get(
            "EVALUATOR_MODEL_CACHE",
            Path(tempfile.gettempdir()) / "video_evaluator_models",
        )
    )
    return cache_root / "face_landmarker.task"


def ensure_face_landmarker_model(
    path: str | Path | None = None,
    *,
    download: bool = True,
) -> Path | None:
    """Return a local model path, optionally downloading the official task."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        primary = default_model_path()
        candidates.append(primary)
        # If package assets path is not writable / missing, fall back to temp.
        temp_fallback = (
            Path(tempfile.gettempdir())
            / "video_evaluator_models"
            / "face_landmarker.task"
        )
        if primary.resolve() != temp_fallback.resolve():
            candidates.append(temp_fallback)

    for target in candidates:
        if target.is_file() and target.stat().st_size > 1024:
            return target

    if not download:
        return None

    for target in candidates:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".task.partial")
            urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, tmp)
            if tmp.stat().st_size < 1024:
                tmp.unlink(missing_ok=True)
                continue
            tmp.replace(target)
            return target
        except Exception:
            try:
                tmp.unlink(missing_ok=True)  # type: ignore[name-defined]
            except Exception:
                pass
            continue
    return None


def estimate_similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Return a 4x4 similarity transform mapping source -> target."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[0] < 3:
        raise ValueError("Need matching Nx3 point sets with N>=3.")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_scale = float(np.linalg.norm(source_centered))
    target_scale = float(np.linalg.norm(target_centered))
    if source_scale < 1e-8 or target_scale < 1e-8:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = target_mean - source_mean
        return matrix
    source_norm = source_centered / source_scale
    target_norm = target_centered / target_scale
    covariance = source_norm.T @ target_norm
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    scale = target_scale / source_scale
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = target_mean - scale * rotation @ source_mean
    return matrix


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([points, ones], axis=1)
    transformed = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    return transformed[:, :3].astype(np.float32)


def normalize_landmarks_pose(
    landmarks_xyz: np.ndarray,
    *,
    transform_matrix: np.ndarray | None = None,
    anchors: Sequence[int] = POSE_ANCHORS,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose-normalize 3D landmarks into a canonical face frame.

    Prefer an official facial transformation matrix when provided by
    FaceLandmarker; otherwise estimate a similarity transform from anchors.
    """
    points = np.asarray(landmarks_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("landmarks_xyz must be shaped (N, 3+).")
    xyz = points[:, :3]
    if transform_matrix is not None:
        matrix = np.asarray(transform_matrix, dtype=np.float64)
        if matrix.shape == (4, 4):
            return apply_transform(xyz, matrix), matrix
        if matrix.shape == (3, 4):
            padded = np.eye(4, dtype=np.float64)
            padded[:3, :] = matrix
            return apply_transform(xyz, padded), padded
    source_anchors = []
    for index in anchors:
        if index >= len(xyz):
            raise ValueError(f"Missing pose anchor landmark {index}.")
        source_anchors.append(xyz[index])
    matrix = estimate_similarity_transform(
        np.asarray(source_anchors, dtype=np.float64),
        CANONICAL_ANCHORS,
    )
    return apply_transform(xyz, matrix), matrix


def iris_relative_features(
    landmarks_xyz: np.ndarray,
) -> dict[str, float]:
    """Cheap iris relative offsets after pose normalization."""
    points = np.asarray(landmarks_xyz, dtype=np.float32)
    features: dict[str, float] = {}
    for name, indices, eye_outer, eye_inner in (
        ("left", IRIS_LEFT, 33, 133),
        ("right", IRIS_RIGHT, 263, 362),
    ):
        if max(indices) >= len(points) or max(eye_outer, eye_inner) >= len(points):
            features[f"iris_{name}_available"] = 0.0
            continue
        iris = points[list(indices)].mean(axis=0)
        eye_center = 0.5 * (points[eye_outer] + points[eye_inner])
        eye_width = float(np.linalg.norm(points[eye_outer] - points[eye_inner]))
        offset = (iris - eye_center) / max(eye_width, 1e-6)
        features[f"iris_{name}_available"] = 1.0
        features[f"iris_{name}_offset_x"] = float(offset[0])
        features[f"iris_{name}_offset_y"] = float(offset[1])
        features[f"iris_{name}_offset_z"] = float(offset[2])
    return features


@dataclass
class FaceLandmarkerFrame:
    landmarks_xyz: np.ndarray
    landmarks_pose_normalized: np.ndarray
    transform_matrix: np.ndarray
    blendshapes: dict[str, float] = field(default_factory=dict)
    backend: str = "unknown"
    face_score: float = 0.0


@dataclass
class FaceLandmarkerSequence:
    frames: list[FaceLandmarkerFrame]
    backend: str
    schema_version: str = FACE_LANDMARKER_SCHEMA

    @property
    def valid_frame_ratio(self) -> float:
        if not self.frames:
            return 0.0
        return float(
            sum(frame.landmarks_xyz.size > 0 for frame in self.frames)
            / len(self.frames)
        )


class FacePoseNormalizer:
    """Track faces and emit pose-normalized landmarks / blendshapes."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        download_model: bool = False,
        prefer_tasks: bool = True,
    ) -> None:
        self.backend = "unavailable"
        self.note = "MediaPipe unavailable."
        self._landmarker = None
        self._mesh = None
        self._timestamp_ms = 0
        if prefer_tasks:
            resolved = ensure_face_landmarker_model(
                model_path,
                download=download_model,
            )
            if resolved is not None:
                self._init_tasks(resolved)
        if self._landmarker is None:
            self._init_face_mesh()

    def _prepare_mediapipe_ascii_modules(self) -> None:
        import mediapipe as mp
        from mediapipe.python import solution_base

        package_root = Path(mp.__file__).resolve().parent
        if not any(ord(char) > 127 for char in str(package_root)):
            return
        ascii_root = (
            Path(tempfile.gettempdir())
            / "video_evaluator_mediapipe"
            / "mediapipe"
        )
        if any(ord(char) > 127 for char in str(ascii_root)):
            raise OSError(
                "The system temp path also contains non-ASCII characters."
            )
        ascii_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            package_root / "modules",
            ascii_root / "modules",
            dirs_exist_ok=True,
        )
        solution_base.__file__ = str(ascii_root / "python" / "solution_base.py")

    def _init_tasks(self, model_path: Path) -> None:
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options as mp_base

            options = vision.FaceLandmarkerOptions(
                base_options=mp_base.BaseOptions(
                    model_asset_path=str(model_path),
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            self.backend = "mediapipe_face_landmarker"
            self.note = "Using MediaPipe Tasks FaceLandmarker."
        except Exception as exc:
            self._landmarker = None
            self.note = f"FaceLandmarker init failed: {exc}"

    def _init_face_mesh(self) -> None:
        try:
            import mediapipe as mp

            self._prepare_mediapipe_ascii_modules()
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.backend = "mediapipe_face_mesh"
            self.note = (
                "Using MediaPipe Face Mesh with estimated pose normalization "
                "(FaceLandmarker model unavailable)."
            )
        except Exception as exc:
            self._mesh = None
            self.backend = "unavailable"
            self.note = f"MediaPipe Face Mesh init failed: {exc}"

    @property
    def available(self) -> bool:
        return self._landmarker is not None or self._mesh is not None

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        if self._mesh is not None:
            self._mesh.close()
            self._mesh = None

    def __enter__(self) -> FacePoseNormalizer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def process_frame(
        self,
        frame_rgb: np.ndarray,
        *,
        timestamp_ms: int | None = None,
    ) -> FaceLandmarkerFrame | None:
        if self._landmarker is not None:
            return self._process_tasks(frame_rgb, timestamp_ms=timestamp_ms)
        if self._mesh is not None:
            return self._process_mesh(frame_rgb)
        return None

    def _process_tasks(
        self,
        frame_rgb: np.ndarray,
        *,
        timestamp_ms: int | None,
    ) -> FaceLandmarkerFrame | None:
        import mediapipe as mp

        if timestamp_ms is None:
            self._timestamp_ms += 33
            timestamp_ms = self._timestamp_ms
        else:
            self._timestamp_ms = int(timestamp_ms)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        if not result.face_landmarks:
            return None
        landmarks = np.asarray(
            [[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]],
            dtype=np.float32,
        )
        matrix = None
        if result.facial_transformation_matrixes:
            matrix = np.asarray(
                result.facial_transformation_matrixes[0],
                dtype=np.float64,
            )
        normalized, matrix = normalize_landmarks_pose(
            landmarks,
            transform_matrix=matrix,
        )
        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                blendshapes[str(category.category_name)] = float(category.score)
        return FaceLandmarkerFrame(
            landmarks_xyz=landmarks,
            landmarks_pose_normalized=normalized,
            transform_matrix=matrix,
            blendshapes=blendshapes,
            backend=self.backend,
            face_score=1.0,
        )

    def _process_mesh(self, frame_rgb: np.ndarray) -> FaceLandmarkerFrame | None:
        result = self._mesh.process(frame_rgb)
        if not result.multi_face_landmarks:
            return None
        landmarks = np.asarray(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in result.multi_face_landmarks[0].landmark
            ],
            dtype=np.float32,
        )
        normalized, matrix = normalize_landmarks_pose(landmarks)
        return FaceLandmarkerFrame(
            landmarks_xyz=landmarks,
            landmarks_pose_normalized=normalized,
            transform_matrix=matrix,
            blendshapes={},
            backend=self.backend,
            face_score=1.0,
        )

    def process_frames(
        self,
        frames_rgb: Sequence[np.ndarray],
        *,
        sample_fps: float = 8.0,
    ) -> FaceLandmarkerSequence:
        outputs: list[FaceLandmarkerFrame] = []
        step_ms = int(round(1000.0 / max(float(sample_fps), 0.1)))
        for index, frame in enumerate(frames_rgb):
            result = self.process_frame(frame, timestamp_ms=index * step_ms)
            if result is None:
                outputs.append(
                    FaceLandmarkerFrame(
                        landmarks_xyz=np.zeros((0, 3), dtype=np.float32),
                        landmarks_pose_normalized=np.zeros(
                            (0, 3),
                            dtype=np.float32,
                        ),
                        transform_matrix=np.eye(4, dtype=np.float64),
                        backend=self.backend,
                        face_score=0.0,
                    )
                )
            else:
                outputs.append(result)
        return FaceLandmarkerSequence(frames=outputs, backend=self.backend)


def summarize_pose_sequence(
    sequence: FaceLandmarkerSequence,
) -> dict[str, Any]:
    """Aggregate pose-normalized landmark / blendshape temporal features."""
    valid = [
        frame
        for frame in sequence.frames
        if frame.landmarks_pose_normalized.size > 0
    ]
    if not valid:
        return {
            "schema_version": FACE_LANDMARKER_SCHEMA,
            "backend": sequence.backend,
            "status": "unavailable",
            "landmark_valid_frame_ratio": 0.0,
            "features": {},
        }
    group_specs = {
        "brow_left": (70, 105, 107),
        "brow_right": (300, 334, 336),
        "eye_left": (33, 133, 145, 159),
        "eye_right": (263, 362, 374, 386),
        "mouth": (13, 14, 61, 291),
        "jaw": (172, 397, 152),
    }
    features: dict[str, float] = {
        "landmark_valid_frame_ratio": sequence.valid_frame_ratio,
    }
    for group_name, indexes in group_specs.items():
        centroids = []
        for frame in valid:
            points = frame.landmarks_pose_normalized
            if max(indexes) >= len(points):
                continue
            centroids.append(points[list(indexes)].mean(axis=0))
        if len(centroids) < 2:
            continue
        matrix = np.stack(centroids)
        motion = np.linalg.norm(np.diff(matrix, axis=0), axis=1)
        features[f"pose_norm_{group_name}_motion_mean"] = float(np.mean(motion))
        features[f"pose_norm_{group_name}_motion_p95"] = float(
            np.quantile(motion, 0.95)
        )
        features[f"pose_norm_{group_name}_std"] = float(np.mean(np.std(matrix, axis=0)))

    iris_rows = [iris_relative_features(frame.landmarks_pose_normalized) for frame in valid]
    for key in (
        "iris_left_offset_x",
        "iris_left_offset_y",
        "iris_right_offset_x",
        "iris_right_offset_y",
    ):
        values = [
            float(row[key])
            for row in iris_rows
            if key in row and math.isfinite(float(row[key]))
        ]
        if values:
            features[f"{key}_mean"] = float(np.mean(values))
            features[f"{key}_std"] = float(np.std(values))

    blend_names = sorted(
        {
            name
            for frame in valid
            for name in frame.blendshapes
        }
    )
    for name in blend_names:
        series = np.asarray(
            [float(frame.blendshapes.get(name, 0.0)) for frame in valid],
            dtype=np.float32,
        )
        features[f"blendshape_{name}_mean"] = float(np.mean(series))
        features[f"blendshape_{name}_std"] = float(np.std(series))
        if len(series) > 1:
            features[f"blendshape_{name}_velocity_p95"] = float(
                np.quantile(np.abs(np.diff(series)), 0.95)
            )

    # Head-pose stability from transform translations / rotations.
    translations = np.stack(
        [frame.transform_matrix[:3, 3] for frame in valid],
        axis=0,
    )
    features["head_translation_std"] = float(np.mean(np.std(translations, axis=0)))
    rotations = []
    for frame in valid:
        rotation = frame.transform_matrix[:3, :3]
        # Approximate yaw/pitch/roll magnitude via Frobenius distance to I.
        rotations.append(float(np.linalg.norm(rotation - np.eye(3))))
    features["head_rotation_delta_mean"] = float(np.mean(rotations))
    features["pose_normalization_confidence_0_1"] = _clamp(
        0.55
        + 0.45 * sequence.valid_frame_ratio
        - 0.20 * features.get("head_translation_std", 0.0)
    )
    return {
        "schema_version": FACE_LANDMARKER_SCHEMA,
        "backend": sequence.backend,
        "status": "available",
        "landmark_valid_frame_ratio": sequence.valid_frame_ratio,
        "blendshape_count": len(blend_names),
        "features": features,
        "note": (
            "Pose-normalized Face Landmarker / Face Mesh features. "
            "They reduce head-pose leakage into muscle-motion evidence."
        ),
    }


def normalize_csv_landmark_frame(
    points_xy: dict[int, np.ndarray],
    *,
    points_z: dict[int, float] | None = None,
) -> dict[int, np.ndarray]:
    """Pose-normalize sparse CSV landmarks (2D or 2.5D) into canonical xy."""
    if not all(index in points_xy for index in (33, 263)):
        return {}
    xyz: dict[int, np.ndarray] = {}
    for index, point in points_xy.items():
        z = 0.0 if points_z is None else float(points_z.get(index, 0.0))
        xyz[index] = np.asarray([point[0], point[1], z], dtype=np.float32)
    # Synthesize nose tip when landmark 1 is absent.
    if 1 not in xyz and 10 in points_xy and 152 in points_xy:
        xyz[1] = np.asarray(
            [
                0.5 * (points_xy[10][0] + points_xy[152][0]),
                0.5 * (points_xy[10][1] + points_xy[152][1]),
                0.05,
            ],
            dtype=np.float32,
        )
    if any(index not in xyz for index in POSE_ANCHORS):
        return {}
    source_anchors = np.stack([xyz[index] for index in POSE_ANCHORS], axis=0)
    matrix = estimate_similarity_transform(
        source_anchors.astype(np.float64),
        CANONICAL_ANCHORS,
    )
    indexes = sorted(xyz)
    stacked = np.stack([xyz[index] for index in indexes], axis=0)
    normalized = apply_transform(stacked, matrix)
    return {
        index: normalized[row_index, :2]
        for row_index, index in enumerate(indexes)
    }
