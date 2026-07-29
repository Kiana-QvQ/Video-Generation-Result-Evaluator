from __future__ import annotations

import importlib.util
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .video_metrics import (
    _aligned_sample_indices,
    _align_ground_truth_frame,
    _read_frames,
    _resize_like,
    DEFAULT_SAMPLE_FPS,
    SEMANTIC_WINDOW_FRAMES,
    SEMANTIC_WINDOW_OVERLAP,
    SEMANTIC_WINDOW_SECONDS,
    sample_aligned_video_windows,
    sample_video_frames,
    sample_video_windows,
    VideoInfo,
    evaluate_full_reference,
    probe_video,
)
from .runtime import MODEL_CACHE_DIR, OUTPUT_DIR, prepare_pyiqa_checkpoint
from .model_profile import get_recommended_model
from .etva_judge import evaluate_etva_judge, etva_service_available
from .hardware_policy import resolve_policy
from .vbench_runner import run_vbench
from .viclip_backend import (
    VICLIP_CHECKPOINT,
    clear_viclip_cache,
    text_similarity as viclip_text_similarity,
    video_similarity as viclip_video_similarity,
    viclip_enabled,
)


WEIGHTS = {
    "identity": 35,
    "texture": 15,
    "expression": 15,
    "temporal": 25,
    "aesthetics": 10,
}


def _model_status(
    name: str,
    purpose: str,
    ready: bool,
    note: str,
    *,
    optional: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "purpose": purpose,
        "status": "ready" if ready else ("optional" if optional else "unavailable"),
        "ready": ready,
        "note": note,
    }


def get_model_inventory() -> list[dict[str, Any]]:
    """Return lightweight model readiness data for the web UI."""
    cache = MODEL_CACHE_DIR
    arcface_ready = (
        importlib.util.find_spec("insightface") is not None
        and (cache / "insightface" / "models" / "buffalo_l" / "w600k_r50.onnx").exists()
    )
    clip_ready = (
        importlib.util.find_spec("clip") is not None
        and (cache / "clip" / "ViT-B-32.pt").exists()
    )
    lpips_ready = (
        importlib.util.find_spec("lpips") is not None
        and (cache / "checkpoints" / "alexnet-owt-7be5be79.pth").exists()
    )
    pyiqa_ready = importlib.util.find_spec("pyiqa") is not None
    maniqa_ready = pyiqa_ready and prepare_pyiqa_checkpoint("maniqa-pipal") is not None
    musiq_ready = pyiqa_ready and prepare_pyiqa_checkpoint("musiq") is not None
    viclip_ready = (
        cache / "viclip" / "ViClip-InternVid-10M-FLT.pth"
    ).exists()
    qwen_vlm_ready = (
        importlib.util.find_spec("transformers") is not None
        and (
            cache
            / "vlm_judge"
            / "Qwen2-VL-2B-Instruct-AWQ"
            / "model.safetensors"
        ).exists()
    )
    onnx_cuda_ready = False
    try:
        import onnxruntime as ort

        onnx_cuda_ready = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        pass
    mediapipe_ready = False
    mediapipe_note = "未安装，将回退到人脸框抖动代理。"
    try:
        import mediapipe as mp

        mediapipe_ready = hasattr(mp, "solutions") and hasattr(
            mp.solutions,
            "face_mesh",
        )
        if mediapipe_ready:
            mediapipe_note = "Face Mesh 可用。"
        else:
            mediapipe_note = (
                "MediaPipe 已安装，但当前版本没有 mp.solutions.face_mesh，"
                "将回退到人脸框抖动代理。"
            )
    except Exception as exc:
        mediapipe_note = f"MediaPipe 加载失败，将回退到人脸框代理：{exc}"
    try:
        from .vbench_runner import discover_vbench

        vbench_ready = bool(discover_vbench().get("available"))
    except Exception:
        vbench_ready = False

    qwen_service_active = etva_service_available()
    qwen_model = _model_status(
        "ETVA VLM Judge (Qwen2-VL-2B AWQ)",
        "细粒度视频问答式评委",
        qwen_vlm_ready,
        (
            "Qwen2-VL-2B AWQ 已缓存，且 Judge HTTP 服务已连接。"
            if qwen_vlm_ready and qwen_service_active
            else "Qwen2-VL-2B AWQ 已下载，但 Judge HTTP 服务未连接。"
            if qwen_vlm_ready
            else "未检测到精简 VLM Judge 权重。"
        ),
        optional=True,
    )
    qwen_model["service_active"] = qwen_service_active

    return [
        _model_status(
            "ViCLIP",
            "Prompt-视频语义相似度",
            viclip_ready,
            (
                "ViCLIP 权重已缓存。"
                if viclip_ready
                else "未检测到 ViCLIP 权重，当前使用 CLIP ViT-B/32 基线。"
            ),
            optional=True,
        ),
        qwen_model,
        _model_status(
            "ArcFace",
            "角色 / 身份一致性",
            arcface_ready,
            (
                "InsightFace buffalo_l 已缓存，ONNX Runtime CUDA provider 可用。"
                if onnx_cuda_ready
                else "InsightFace buffalo_l 已缓存，但 ONNX Runtime 没有 CUDA provider，ArcFace 将使用 CPU。"
            )
            if arcface_ready
            else "未检测到 ArcFace 权重，将回退到人脸特征代理。",
        ),
        _model_status(
            "CLIP ViT-B/32",
            "Prompt-视频语义对齐",
            clip_ready,
            "本地 CLIP 权重已缓存。" if clip_ready else "未检测到本地 CLIP 权重，Prompt 对齐不可用。",
        ),
        _model_status(
            "LPIPS (Alex)",
            "有 GT 的感知距离",
            lpips_ready,
            "LPIPS 权重已缓存。" if lpips_ready else "首次使用可能需要下载 LPIPS 权重。",
        ),
        _model_status(
            "MANIQA",
            "无 GT 的感知质量",
            maniqa_ready,
            "MANIQA 权重已缓存。" if maniqa_ready else "未检测到 MANIQA 权重。",
            optional=True,
        ),
        _model_status(
            "MUSIQ",
            "无 GT 的感知质量",
            musiq_ready,
            "MUSIQ 权重已缓存。" if musiq_ready else "未检测到 MUSIQ 权重。",
            optional=True,
        ),
        _model_status(
            "MediaPipe Face Mesh",
            "表情关键点 / 时间稳定性",
            mediapipe_ready,
            mediapipe_note,
        ),
        _model_status(
            "VBench",
            "额外多维视频基准",
            vbench_ready,
            "检测到 VBench 执行入口。" if vbench_ready else "未配置 VBench，保留为可选后端。",
            optional=True,
        ),
        _model_status(
            "VideoScore2",
            "整体偏好与细粒度视频语义",
            False,
            "首版不伪造在线或未配置模型结果，后续通过独立适配器接入。",
            optional=True,
        ),
    ]


def get_model_recommendation() -> dict[str, Any]:
    """Expose the 8GB-first model policy without loading a model."""
    return get_recommended_model()


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _safe_mean(values: Iterable[float | int | None]) -> float | None:
    valid: list[float] = []
    for value in values:
        if value is None:
            continue
        parsed = float(value)
        if math.isfinite(parsed):
            valid.append(parsed)
    return float(np.mean(valid)) if valid else None


def _lower_tail(values: Iterable[float], fraction: float = 0.1) -> list[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return []
    count = max(1, int(math.ceil(len(ordered) * fraction)))
    return ordered[:count]


def _robust_average(
    values: Iterable[float | int | None],
    tail_weight: float = 0.2,
) -> tuple[float | None, float | None]:
    valid = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not valid:
        return None, None
    mean = float(np.mean(valid))
    tail = _safe_mean(_lower_tail(valid))
    if tail is None:
        return mean, None
    weight = _clamp(float(tail_weight))
    return _clamp((1.0 - weight) * mean + weight * tail), tail


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else np.zeros_like(vector)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = _normalize(left)
    right = _normalize(right)
    value = float(np.dot(left, right))
    return _clamp((value + 1.0) / 2.0)


def _sample_video(path: str | Path, max_frames: int) -> tuple[dict[str, Any], np.ndarray, list[np.ndarray]]:
    info, indices, _, frames = sample_video_frames(path, max_frames)
    return info, indices, frames


def _sample_aligned_videos(
    result_path: str | Path,
    reference_path: str | Path,
    max_frames: int,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
]:
    result_info = probe_video(result_path).to_dict()
    reference_info = probe_video(reference_path).to_dict()
    _, result_indices, reference_indices, _ = _aligned_sample_indices(
        VideoInfo(**result_info),
        VideoInfo(**reference_info),
        max_frames,
    )
    result_frames = _read_frames(result_info["path"], result_indices)
    reference_frames = _read_frames(reference_info["path"], reference_indices)
    reference_frames = [
        _align_ground_truth_frame(reference_frame, result_frame)
        for result_frame, reference_frame in zip(result_frames, reference_frames)
    ]
    return result_info, result_indices, result_frames, reference_frames


def _read_reference_image(
    path: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
) -> list[np.ndarray]:
    if not path:
        return []
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    frames: list[np.ndarray] = []
    for image_path in paths:
        try:
            encoded = np.fromfile(str(image_path), dtype=np.uint8)
        except (OSError, ValueError):
            continue
        image = (
            cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if encoded.size
            else None
        )
        if image is not None:
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return frames


class _FaceDetector:
    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        classifier_path = cascade_path
        # OpenCV on Windows may fail to open model paths containing Chinese
        # characters. Copy the bundled cascade to an ASCII temp path first.
        if cascade_path.exists() and any(ord(char) > 127 for char in str(cascade_path)):
            try:
                temp_dir = Path(tempfile.gettempdir()) / "video_evaluator_models"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / cascade_path.name
                if not temp_path.exists() or temp_path.stat().st_size != cascade_path.stat().st_size:
                    shutil.copyfile(cascade_path, temp_path)
                classifier_path = temp_path
            except OSError:
                pass
        self.classifier = cv2.CascadeClassifier(str(classifier_path))
        self.available = not self.classifier.empty()

    def detect(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        if not self.available:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )
        if len(faces) == 0:
            return None
        x, y, width, height = max(faces, key=lambda item: int(item[2]) * int(item[3]))
        return int(x), int(y), int(width), int(height)


def _crop_face(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    margin: float = 0.18,
) -> np.ndarray:
    if bbox is None:
        return frame
    x, y, width, height = bbox
    pad_x = int(width * margin)
    pad_y = int(height * margin)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(frame.shape[1], x + width + pad_x)
    y1 = min(frame.shape[0], y + height + pad_y)
    return frame[y0:y1, x0:x1]


class _IdentityBackend:
    """Use InsightFace when installed, otherwise an explicit face proxy."""

    def __init__(self, detector: _FaceDetector, device: str = "auto") -> None:
        self.detector = detector
        self.insight_app: Any | None = None
        self.device = "cpu"
        self.backend = "face_crop_proxy"
        self.note = (
            "InsightFace 未安装，身份一致性使用 OpenCV 人脸框和人脸裁剪特征代理；"
            "安装可选人脸依赖后会自动切换为 ArcFace。"
        )
        if importlib.util.find_spec("insightface") is not None:
            try:
                from insightface.app import FaceAnalysis

                face_device = os.environ.get(
                    "EVALUATOR_FACE_DEVICE",
                    device,
                ).lower()
                if face_device == "auto":
                    try:
                        import torch

                        face_device = (
                            "cuda" if torch.cuda.is_available() else "cpu"
                        )
                    except ImportError:
                        face_device = "cpu"
                cuda_provider_available = False
                if face_device == "cuda":
                    try:
                        import onnxruntime as ort

                        cuda_provider_available = (
                            "CUDAExecutionProvider"
                            in ort.get_available_providers()
                        )
                    except Exception:
                        cuda_provider_available = False
                    if not cuda_provider_available:
                        face_device = "cpu"
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if face_device == "cuda"
                    else ["CPUExecutionProvider"]
                )
                self.insight_app = FaceAnalysis(
                    name="buffalo_l",
                    root=str(MODEL_CACHE_DIR / "insightface"),
                    providers=providers,
                )
                self.insight_app.prepare(
                    ctx_id=0 if face_device == "cuda" else -1,
                    det_size=(640, 640),
                )
                self.device = face_device
                self.backend = "arcface"
                if cuda_provider_available:
                    self.note = "使用 InsightFace ArcFace 人脸嵌入（CUDA）。"
                elif face_device == "cpu":
                    self.note = (
                        "使用 InsightFace ArcFace 人脸嵌入（CPU）；"
                        "ONNX Runtime CUDA provider 不可用。"
                    )
            except Exception as exc:
                self.note = f"InsightFace 初始化失败，回退到人脸特征代理：{exc}"

    def _proxy_embedding(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
    ) -> np.ndarray | None:
        if bbox is None:
            return None
        crop = _crop_face(frame, bbox)
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        histograms = [
            cv2.calcHist([gray], [0], None, [16], [0, 1]).flatten(),
            cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten(),
            cv2.calcHist([hsv], [1], None, [12], [0, 256]).flatten(),
        ]
        gray = (gray - float(gray.mean())) / (float(gray.std()) + 1e-6)
        descriptor = np.concatenate([gray.flatten(), *histograms])
        return _normalize(descriptor)

    def embedding(
        self,
        frame: np.ndarray,
    ) -> tuple[
        np.ndarray | None,
        tuple[int, int, int, int] | None,
        str,
    ]:
        if self.insight_app is not None:
            try:
                faces = self.insight_app.get(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if faces:
                    face = max(faces, key=lambda item: float(item.bbox[2] - item.bbox[0]) * float(item.bbox[3] - item.bbox[1]))
                    bbox = tuple(int(round(value)) for value in face.bbox)
                    return _normalize(np.asarray(face.embedding)), bbox, "arcface"  # type: ignore[arg-type]
            except Exception:
                pass
            # Keep ArcFace and the pixel proxy separate. Mixing their vector
            # dimensions makes a partial run impossible to compare safely.
            return None, None, "arcface"
        bbox = self.detector.detect(frame)
        return self._proxy_embedding(frame, bbox), bbox, "face_crop_proxy"


class _LandmarkTracker:
    def __init__(self) -> None:
        self.mesh: Any | None = None
        self.backend = "face_box_proxy"
        self.note = "MediaPipe 未安装，关键点指标使用人脸框代理。"
        if importlib.util.find_spec("mediapipe") is not None:
            try:
                import mediapipe as mp
                from mediapipe.python import solution_base

                # MediaPipe's native resource loader can fail when the
                # project path contains Chinese characters on Windows.
                package_root = Path(mp.__file__).resolve().parent
                if any(ord(char) > 127 for char in str(package_root)):
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
                    solution_base.__file__ = str(
                        ascii_root / "python" / "solution_base.py"
                    )

                self.mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.backend = "mediapipe_face_mesh"
                self.note = "使用 MediaPipe Face Mesh 关键点。"
            except Exception as exc:
                self.note = f"MediaPipe 初始化失败，使用人脸框代理：{exc}"

    @property
    def available(self) -> bool:
        return self.mesh is not None

    def extract(self, frame: np.ndarray) -> np.ndarray | None:
        if self.mesh is None:
            return None
        try:
            result = self.mesh.process(frame)
            if not result.multi_face_landmarks:
                return None
            return np.asarray(
                [
                    [landmark.x, landmark.y, landmark.z]
                    for landmark in result.multi_face_landmarks[0].landmark
                ],
                dtype=np.float32,
            )
        except Exception:
            return None


def _reference_frames(
    reference_image: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    reference_video: str | Path | None,
    ground_truth: str | Path | None,
    max_frames: int,
) -> tuple[list[np.ndarray], str]:
    image_frames = _read_reference_image(reference_image)
    if image_frames:
        return image_frames, "reference_image"
    if reference_video:
        try:
            _, _, frames = _sample_video(reference_video, max_frames)
            return frames, "reference_video"
        except Exception:
            pass
    if ground_truth:
        try:
            _, _, frames = _sample_video(ground_truth, max_frames)
            return frames, "gt_video"
        except Exception:
            pass
    return [], "none"


def _identity_reference_frames(
    reference_image: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    reference_video: str | Path | None,
    ground_truth: str | Path | None,
    max_frames: int,
) -> tuple[list[np.ndarray], str]:
    """Combine every available identity source instead of choosing one."""
    frames: list[np.ndarray] = []
    sources: list[str] = []

    image_frames = _read_reference_image(reference_image)
    if image_frames:
        frames.extend(image_frames)
        sources.append("reference_image")

    for path, source in (
        (reference_video, "reference_video"),
        (ground_truth, "gt_video"),
    ):
        if not path:
            continue
        try:
            _, _, video_frames = _sample_video(path, max_frames)
        except Exception:
            continue
        if video_frames:
            frames.extend(video_frames)
            sources.append(source)

    return frames, "+".join(sources) if sources else "none"


def evaluate_identity(
    result_path: str | Path,
    reference_image: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    reference_video: str | Path | None,
    ground_truth: str | Path | None,
    max_frames: int,
    device: str = "auto",
) -> dict[str, Any]:
    detector = _FaceDetector()
    backend = _IdentityBackend(detector, device=device)
    reference_frames, reference_source = _identity_reference_frames(
        reference_image,
        reference_video,
        ground_truth,
        max_frames,
    )
    if not reference_frames:
        return {
            "status": "unavailable",
            "backend": backend.backend,
            "reference_source": reference_source,
            "reason": "没有参考图、参考视频或 GT，无法建立身份基准。",
            "note": backend.note,
            "metrics": {},
            "frame_records": [],
            "warnings": [],
        }

    reference_embeddings: list[np.ndarray] = []
    reference_backends: set[str] = set()
    for frame in reference_frames:
        embedding, _, embedding_backend = backend.embedding(frame)
        if embedding is not None:
            reference_embeddings.append(embedding)
            reference_backends.add(embedding_backend)
    if not reference_embeddings:
        return {
            "status": "unavailable",
            "backend": backend.backend,
            "reference_source": reference_source,
            "reason": "参考素材中没有检测到可用人脸。",
            "note": backend.note,
            "metrics": {},
            "frame_records": [],
            "warnings": [],
        }

    reference_embedding = _normalize(np.mean(reference_embeddings, axis=0))
    result_info, indices, frames = _sample_video(result_path, max_frames)
    similarities: list[float] = []
    records: list[dict[str, Any]] = []
    frame_backends: set[str] = set(reference_backends)
    for index, frame in zip(indices, frames):
        embedding, bbox, embedding_backend = backend.embedding(frame)
        frame_backends.add(embedding_backend)
        similarity = _cosine_similarity(reference_embedding, embedding) if embedding is not None else None
        if similarity is not None:
            similarities.append(similarity)
        records.append(
            {
                "sample_index": len(records),
                "result_frame": int(index),
                "timestamp_seconds": (
                    round(int(index) / float(result_info["fps"]), 4)
                    if result_info["fps"] > 0
                    else None
                ),
                "face_found": embedding is not None,
                "identity_backend": embedding_backend,
                "identity_similarity": round(similarity, 6) if similarity is not None else None,
                "face_bbox": list(bbox) if bbox is not None else None,
            }
        )

    if not similarities:
        return {
            "status": "unavailable",
            "backend": backend.backend,
            "reference_source": reference_source,
            "reason": "结果视频中没有检测到可用人脸。",
            "note": backend.note,
            "metrics": {},
            "frame_records": records,
            "warnings": [],
        }

    scoring_values = [
        record["identity_similarity"]
        if record["identity_similarity"] is not None
        else 0.0
        for record in records
    ]
    # "Tail 10%" means the lower tail, not the last timestamps in the clip.
    tail_values = _lower_tail(scoring_values)
    valid_frame_ratio = float(len(similarities) / max(len(records), 1))
    robust_score, _ = _robust_average(scoring_values)
    return {
        "status": (
            "available"
            if frame_backends == {"arcface"} and valid_frame_ratio == 1.0
            else "partial"
        ),
        "backend": (
            "arcface"
            if frame_backends == {"arcface"}
            else "+".join(sorted(frame_backends))
        ),
        "reference_source": reference_source,
        "reason": None,
        "note": backend.note,
        "metrics": {
            "mean_similarity": _safe_mean(scoring_values),
            "tail_10pct_similarity": _safe_mean(tail_values),
            "variance": float(np.var(scoring_values)),
            "minimum_similarity": float(np.min(scoring_values)),
            "valid_frame_ratio": valid_frame_ratio,
            "score_0_1": robust_score,
        },
        "frame_records": records,
        "warnings": [],
    }


def _high_frequency_energy(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
) -> float:
    roi = _crop_face(frame, bbox, margin=0.05)
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    high_pass = gray - blur
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    skin_mask = cv2.inRange(
        hsv,
        np.array([0, 20, 35], dtype=np.uint8),
        np.array([35, 230, 255], dtype=np.uint8),
    ) > 0
    if int(np.count_nonzero(skin_mask)) >= 64:
        high_values = np.abs(high_pass[skin_mask])
        base_values = np.abs(gray[skin_mask])
    else:
        high_values = np.abs(high_pass).reshape(-1)
        base_values = np.abs(gray).reshape(-1)
    return float(np.mean(high_values) / (np.mean(base_values) + 1e-6))


def _optional_iqa(
    frames: list[np.ndarray],
    metric_name: str,
    device: str,
) -> tuple[float | None, str]:
    if os.environ.get("EVALUATOR_DISABLE_OPTIONAL_IQA", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return None, f"{metric_name} 已按环境变量关闭"
    if importlib.util.find_spec("pyiqa") is None:
        return None, "pyiqa 未安装"
    try:
        import torch
        import pyiqa

        checkpoint = prepare_pyiqa_checkpoint(metric_name)
        if checkpoint is None:
            return None, f"{metric_name} 权重未完整缓存，已跳过自动下载"
        requested_device = os.environ.get("EVALUATOR_IQA_DEVICE", "").lower()
        if requested_device not in {"cpu", "cuda"}:
            requested_device = device
        resolved_device = (
            "cuda"
            if requested_device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )
        if metric_name == "maniqa-pipal":
            metric = _create_offline_maniqa_metric(
                pyiqa,
                resolved_device,
            )
        else:
            metric = pyiqa.create_metric(metric_name, device=resolved_device)
        values: list[float] = []
        with torch.no_grad():
            for frame in frames:
                tensor = (
                    torch.from_numpy(frame)
                    .permute(2, 0, 1)
                    .float()
                    .div(255.0)
                    .unsqueeze(0)
                    .to(resolved_device)
                )
                value = float(metric(tensor).detach().cpu().reshape(-1)[0])
                if value > 1.0:
                    value /= 100.0
                values.append(value)
        return _safe_mean(values), f"pyiqa/{metric_name}"
    except Exception as exc:
        return None, f"{metric_name} 不可用：{exc}"


def _create_offline_maniqa_metric(pyiqa: Any, device: str) -> Any:
    """Avoid timm's redundant ImageNet download; PIPAL contains the ViT weights."""
    import timm

    original_create_model = timm.create_model

    def create_model_without_download(
        model_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if model_name == "vit_base_patch8_224":
            kwargs["pretrained"] = False
        return original_create_model(model_name, *args, **kwargs)

    timm.create_model = create_model_without_download
    try:
        return pyiqa.create_metric("maniqa-pipal", device=device)
    finally:
        timm.create_model = original_create_model


def evaluate_texture(
    result_path: str | Path,
    ground_truth: str | Path | None,
    reference_image: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    reference_video: str | Path | None,
    max_frames: int,
    calculate_lpips: bool,
    device: str,
) -> dict[str, Any]:
    detector = _FaceDetector()
    result_info, result_indices, result_frames = _sample_video(result_path, max_frames)
    result_boxes = [detector.detect(frame) for frame in result_frames]
    warnings: list[str] = []
    ground_truth_provided = bool(ground_truth)
    ground_truth_fallback_reason: str | None = None

    full_reference: dict[str, Any] | None = None
    if ground_truth:
        try:
            full_reference = evaluate_full_reference(
                result_path=result_path,
                ground_truth_path=ground_truth,
                max_frames=max_frames,
                calculate_lpips=calculate_lpips,
                device=device,
            )
        except (FileNotFoundError, ValueError) as exc:
            warnings.append(
                f"GT 全参考指标不可用，已降级到无 GT 纹理评估：{exc}"
            )
            ground_truth_fallback_reason = str(exc)
            ground_truth = None

    if ground_truth and full_reference is not None:
        result_eval_indices = [
            int(record["result_frame"]) for record in full_reference["records"]
        ]
        gt_eval_indices = [
            int(record["gt_frame"]) for record in full_reference["records"]
        ]
        result_eval_frames = _read_frames(result_path, result_eval_indices)
        gt_eval_frames = _read_frames(ground_truth, gt_eval_indices)
        result_eval_boxes = [detector.detect(frame) for frame in result_eval_frames]
        gt_eval_boxes = [detector.detect(frame) for frame in gt_eval_frames]
        ratio_values: list[float | None] = []
        for result_frame, gt_frame, result_bbox, gt_bbox in zip(
            result_eval_frames,
            gt_eval_frames,
            result_eval_boxes,
            gt_eval_boxes,
        ):
            gt_frame = _align_ground_truth_frame(gt_frame, result_frame)
            result_value = _high_frequency_energy(result_frame, result_bbox)
            gt_value = _high_frequency_energy(gt_frame, gt_bbox)
            ratio_values.append(
                result_value / (gt_value + 1e-6) if gt_value > 1e-6 else None
            )
        valid_ratios = [
            value for value in ratio_values if value is not None
        ]
        records = [
            {
                "sample_index": int(record["sample_index"]),
                "result_frame": int(record["result_frame"]),
                "gt_frame": int(record["gt_frame"]),
                "timestamp_seconds": record.get("timestamp_seconds"),
                "psnr_db": record["psnr_db"],
                "ssim": record["ssim"],
                "lpips": record["lpips"],
                "high_frequency_ratio": (
                    round(ratio_values[i], 6)
                    if ratio_values[i] is not None
                    else None
                ),
            }
            for i, record in enumerate(full_reference["records"])
        ]
        warnings.extend(full_reference["warnings"])
        lpips_value = full_reference["metrics"]["lpips"]
        if not calculate_lpips:
            warnings.append("LPIPS 被用户关闭，本次第 2 类只能算 PSNR/SSIM。")
        psnr_value = full_reference["metrics"]["psnr_db"]
        if psnr_value is not None and math.isinf(float(psnr_value)):
            psnr_score = 1.0
        elif psnr_value is not None and math.isfinite(float(psnr_value)):
            psnr_score = 1.0 - math.exp(
                -max(float(psnr_value) - 20.0, 0.0) / 20.0
            )
        else:
            psnr_score = None
        ssim_value = full_reference["metrics"]["ssim"]
        texture_scores = [
            _clamp(float(ssim_value)) if ssim_value is not None else None,
            psnr_score,
            (
                math.exp(-float(lpips_value) / 0.2)
                if lpips_value is not None
                else None
            ),
        ]
        valid_texture_scores = [
            float(value) for value in texture_scores if value is not None
        ]
        return {
            "status": "available" if lpips_value is not None else "partial",
            "mode": "full_reference",
            "ground_truth_status": "used",
            "ground_truth_provided": True,
            "ground_truth_usable": True,
            "ground_truth_fallback_reason": None,
            "ground_truth_alignment": full_reference.get("alignment"),
            "backend": "PSNR/SSIM/LPIPS (GT) + high_frequency_proxy",
            "metrics": {
                "psnr_db": full_reference["metrics"]["psnr_db"],
                "ssim": full_reference["metrics"]["ssim"],
                "lpips": full_reference["metrics"]["lpips"],
                "high_frequency_retention_ratio": _safe_mean(valid_ratios),
                "score_0_1": (
                    _safe_mean(valid_texture_scores)
                    if valid_texture_scores
                    else None
                ),
            },
            "reference_source": "gt_video",
            "note": (
                "第 2 类使用逐帧对应 GT 计算 PSNR、SSIM、LPIPS；"
                "高频能比仅作为低优先级辅助项。"
            ),
            "warnings": warnings,
            "frame_records": records,
        }

    reference_frames, reference_source = _reference_frames(
        reference_image,
        reference_video,
        None,
        max_frames,
    )
    if reference_video and reference_source != "reference_image":
        try:
            (
                result_info,
                result_indices,
                result_frames,
                reference_frames,
            ) = _sample_aligned_videos(
                result_path,
                reference_video,
                max_frames,
            )
            result_boxes = [detector.detect(frame) for frame in result_frames]
        except Exception as exc:
            warnings.append(f"参考视频高频细节对齐失败：{exc}")
    reference_boxes = [detector.detect(frame) for frame in reference_frames]
    result_hf = [
        _high_frequency_energy(frame, bbox)
        for frame, bbox in zip(result_frames, result_boxes)
    ]
    reference_hf = [
        _high_frequency_energy(frame, bbox)
        for frame, bbox in zip(reference_frames, reference_boxes)
    ]
    maniqa, maniqa_backend = _optional_iqa(result_frames, "maniqa-pipal", device)
    musiq, musiq_backend = _optional_iqa(result_frames, "musiq", device)
    warnings.extend([maniqa_backend, musiq_backend])
    retention = None
    if reference_hf:
        if reference_source == "reference_image":
            reference_value = reference_hf[0]
            retention_values = [
                value / (reference_value + 1e-6)
                for value in result_hf
            ]
        else:
            retention_values = [
                value / (reference_hf[i] + 1e-6)
                for i, value in enumerate(result_hf[: len(reference_hf)])
            ]
        retention = _safe_mean(retention_values)
    high_frequency_score = (
        math.exp(-abs(math.log(max(float(retention), 1e-6))))
        if retention is not None
        else None
    )
    texture_scores = [
        _clamp(float(value))
        for value in (maniqa, musiq, high_frequency_score)
        if value is not None and math.isfinite(float(value))
    ]
    records = [
        {
            "sample_index": i,
            "result_frame": int(index),
            "timestamp_seconds": (
                round(int(index) / float(result_info["fps"]), 4)
                if result_info["fps"] > 0
                else None
            ),
            "high_frequency_ratio": round(value, 6),
        }
        for i, (index, value) in enumerate(zip(result_indices, result_hf))
    ]
    return {
        "status": "partial",
        "mode": "no_gt",
        "ground_truth_status": (
            "uploaded_but_unusable" if ground_truth_provided else "not_uploaded"
        ),
        "ground_truth_provided": ground_truth_provided,
        "ground_truth_usable": False,
        "ground_truth_fallback_reason": ground_truth_fallback_reason,
        "backend": f"MANIQA={maniqa_backend}; MUSIQ={musiq_backend}; high_frequency_proxy",
        "metrics": {
            "psnr_db": None,
            "ssim": None,
            "lpips": None,
            "maniqa": maniqa,
            "musiq": musiq,
            "high_frequency_ratio": _safe_mean(result_hf),
            "high_frequency_retention_ratio": retention,
            "score_0_1": _safe_mean(texture_scores),
        },
        "reference_source": reference_source,
        "note": (
            "无 GT 时第 2 类不计算 PSNR、SSIM、LPIPS，"
            "改用 MANIQA/MUSIQ（可选）和自定义高频能比。"
        ),
        "warnings": warnings,
        "frame_records": records,
    }


def evaluate_text_alignment(
    result_path: str | Path,
    prompt_text: str | None,
    max_frames: int,
    device: str,
    use_viclip: bool = True,
) -> dict[str, Any]:
    prompt = (prompt_text or "").strip()
    if not prompt:
        return {
            "status": "unavailable",
            "backend": "none",
            "reason": "没有提供文本 Prompt，无法计算文本-视频语义对齐。",
            "note": "第 3 类可以改用参考视频或人工评分。",
            "metrics": {},
            "frame_records": [],
            "warnings": [],
        }

    if use_viclip and viclip_enabled(device):
        try:
            info, windows = sample_video_windows(result_path, max_frames)
            window_scores: list[float] = []
            raw_scores: list[float] = []
            window_records: list[dict[str, Any]] = []
            for window in windows:
                viclip_result = viclip_text_similarity(
                    window["frames"],
                    prompt,
                    device,
                )
                window_scores.append(float(viclip_result["score_0_1"]))
                raw_scores.append(float(viclip_result["raw_cosine"]))
                window_records.append(
                    {
                        "sample_index": int(window["window_index"]),
                        "window_index": int(window["window_index"]),
                        "window_start_seconds": window["start_seconds"],
                        "window_end_seconds": window["end_seconds"],
                        "result_frame": int(window["indices"][0]),
                        "timestamp_seconds": window["start_seconds"],
                        "text_video_similarity": float(
                            viclip_result["score_0_1"]
                        ),
                    }
                )
            score, tail_score = _robust_average(window_scores)
            if score is None:
                raise ValueError("ViCLIP produced no usable window scores.")
            return {
                "status": "available",
                "backend": "viclip_internvid_10m_flt",
                "prompt": prompt,
                "reason": None,
                "note": (
                    "ViCLIP video-text alignment is active. "
                    "The local OpenAI CLIP fallback is not used for this result."
                ),
                "metrics": {
                    "score_0_1": score,
                    "raw_cosine_mean": _safe_mean(raw_scores),
                    "window_mean_score": _safe_mean(window_scores),
                    "window_lower_10pct_score": tail_score,
                    "window_count": len(window_scores),
                    "valid_frame_ratio": 1.0,
                    "device": viclip_result["device"],
                    "model_checkpoint": str(VICLIP_CHECKPOINT),
                },
                "frame_records": window_records,
                "warnings": [],
            }
        except Exception as exc:
            viclip_error = str(exc)
        else:
            viclip_error = None
    else:
        viclip_error = None

    weight_path = MODEL_CACHE_DIR / "clip" / "ViT-B-32.pt"
    if importlib.util.find_spec("clip") is None:
        return {
            "status": "unavailable",
            "backend": "openai_clip_framewise",
            "reason": "OpenAI CLIP Python 包未安装。",
            "note": "请安装可选依赖或改用人工评分。",
            "metrics": {},
            "frame_records": [],
            "warnings": (
                [f"ViCLIP unavailable: {viclip_error}"]
                if viclip_error
                else []
            ),
        }
    if not weight_path.exists():
        return {
            "status": "unavailable",
            "backend": "openai_clip_framewise",
            "reason": f"CLIP 权重不存在：{weight_path}",
            "note": "请先运行 scripts/download-optional-assets.ps1 下载 ViT-B/32。",
            "metrics": {},
            "frame_records": [],
            "warnings": (
                [f"ViCLIP unavailable: {viclip_error}"]
                if viclip_error
                else []
            ),
        }

    try:
        import clip
        import torch
        from PIL import Image

        requested_device = os.environ.get(
            "EVALUATOR_SEMANTIC_DEVICE",
            "",
        ).lower()
        if requested_device not in {"cpu", "cuda"}:
            requested_device = device
        resolved_device = (
            "cuda"
            if requested_device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )
        model, preprocess = clip.load(
            str(weight_path),
            device=resolved_device,
            jit=False,
        )
        model.eval()
        with torch.no_grad():
            text_features = model.encode_text(
                clip.tokenize([prompt]).to(resolved_device)
            )
            text_features = text_features / (
                text_features.norm(dim=-1, keepdim=True) + 1e-6
            )

            info, indices, frames = _sample_video(result_path, max_frames)
            similarities: list[float] = []
            batch_size = 8
            for start in range(0, len(frames), batch_size):
                image_batch = torch.stack(
                    [
                        preprocess(Image.fromarray(frame))
                        for frame in frames[start : start + batch_size]
                    ]
                ).to(resolved_device)
                image_features = model.encode_image(image_batch)
                image_features = image_features / (
                    image_features.norm(dim=-1, keepdim=True) + 1e-6
                )
                batch_scores = (
                    image_features @ text_features.T
                ).flatten().detach().cpu()
                similarities.extend(float(value) for value in batch_scores)

        score_values = [_clamp((value + 1.0) / 2.0) for value in similarities]
        return {
            "status": "partial",
            "backend": "openai_clip_framewise",
            "prompt": prompt,
            "reason": None,
            "note": (
                "这是逐帧图像 CLIP 文本对齐基线，不是 VideoCLIP/ViCLIP；"
                "结果用于语义条件参考，不能替代 GT 全参考指标。"
            ),
            "metrics": {
                "score_0_1": _safe_mean(score_values),
                "raw_cosine_mean": _safe_mean(similarities),
                "valid_frame_ratio": 1.0,
                "device": resolved_device,
            },
            "frame_records": [
                {
                    "sample_index": i,
                    "result_frame": int(index),
                    "timestamp_seconds": (
                        round(int(index) / float(info["fps"]), 4)
                        if info["fps"] > 0
                        else None
                    ),
                    "text_video_similarity": round(score_values[i], 6),
                }
                for i, index in enumerate(indices)
            ],
            "warnings": [],
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "backend": "openai_clip_framewise",
            "reason": f"CLIP 文本-视频评估失败：{exc}",
            "note": "请回退到参考视频代理或人工评分。",
            "metrics": {},
            "frame_records": [],
            "warnings": [str(exc)],
        }


EXPRESSION_LANDMARK_INDICES = np.array(
    [13, 14, 61, 291, 159, 145, 386, 374, 70, 300],
    dtype=np.int64,
)


def _expression_descriptors(
    frames: list[np.ndarray],
    detector: _FaceDetector,
    landmark_tracker: _LandmarkTracker,
) -> tuple[np.ndarray, str]:
    if landmark_tracker.available and frames:
        landmark_descriptors: list[np.ndarray] = []
        for frame in frames:
            landmarks = landmark_tracker.extract(frame)
            if (
                landmarks is None
                or landmarks.ndim != 2
                or landmarks.shape[0] <= int(EXPRESSION_LANDMARK_INDICES.max())
            ):
                landmark_descriptors = []
                break
            # Mouth, eyes, brows, and jaw carry most expression information.
            selected = landmarks[EXPRESSION_LANDMARK_INDICES, :2]
            center = selected.mean(axis=0, keepdims=True)
            scale = float(np.linalg.norm(selected.max(axis=0) - selected.min(axis=0)))
            landmark_descriptors.append(
                ((selected - center) / (scale + 1e-6)).flatten()
            )
        if len(landmark_descriptors) == len(frames):
            return np.stack(landmark_descriptors), "mediapipe_face_mesh"

    # A sequence must use one descriptor shape. If Face Mesh misses even one
    # frame, use the proxy consistently instead of mixing 20-D landmarks with
    # 576-D image descriptors.
    descriptors: list[np.ndarray] = []
    face_count = 0
    for frame in frames:
        bbox = detector.detect(frame)
        if bbox is not None:
            face_count += 1
        crop = _crop_face(frame, bbox)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
        gray = (gray - float(gray.mean())) / (float(gray.std()) + 1e-6)
        descriptors.append(gray.flatten())
    if not descriptors:
        return np.empty((0, 576), dtype=np.float32), "unavailable"
    backend = "face_crop_motion_proxy" if face_count else "full_frame_motion_proxy"
    return np.stack(descriptors), backend


def _motion_signature(
    frames: list[np.ndarray],
    detector: _FaceDetector,
    landmark_tracker: _LandmarkTracker,
) -> tuple[np.ndarray, np.ndarray, str]:
    descriptors, backend = _expression_descriptors(frames, detector, landmark_tracker)
    if len(descriptors) < 2:
        return np.empty((0, descriptors.shape[1] if descriptors.ndim == 2 else 576)), np.empty(0), backend
    deltas = np.diff(descriptors, axis=0)
    magnitudes = np.linalg.norm(deltas, axis=1)
    normalized = deltas / (magnitudes[:, None] + 1e-6)
    return normalized, magnitudes, backend


def _motion_direction_similarity(
    result_vector: np.ndarray,
    result_magnitude: float,
    reference_vector: np.ndarray,
    reference_magnitude: float,
) -> float:
    result_is_static = float(result_magnitude) <= 1e-6
    reference_is_static = float(reference_magnitude) <= 1e-6
    if result_is_static and reference_is_static:
        return 1.0
    if result_is_static or reference_is_static:
        return 0.0
    cosine = float(np.dot(result_vector, reference_vector))
    return _clamp((cosine + 1.0) / 2.0)


def evaluate_expression(
    result_path: str | Path,
    ground_truth: str | Path | None,
    reference_video: str | Path | None,
    manual_score: float | None,
    max_frames: int,
    device: str = "auto",
    use_viclip: bool = True,
    need_viclip_text: bool = False,
) -> dict[str, Any]:
    detector = _FaceDetector()
    landmark_tracker = _LandmarkTracker()
    result_info, result_indices, result_frames = _sample_video(result_path, max_frames)
    reference_path = reference_video or ground_truth
    if not reference_path:
        score = _clamp(float(manual_score or 0.0) / 5.0)
        return {
            "status": "manual" if manual_score is not None else "unavailable",
            "mode": "manual",
            "backend": "manual_1_to_5",
            "score_0_1": score if manual_score is not None else None,
            "manual_score": manual_score,
            "videoclip": None,
            "viclip": None,
            "reference_source": "none",
            "note": "没有参考表情视频，按截图要求使用人工 1~5 分。",
            "warnings": [],
            "frame_records": [],
        }

    try:
        result_info, result_indices, result_frames, reference_frames = (
            _sample_aligned_videos(
                result_path,
                reference_path,
                max_frames,
            )
        )
    except Exception as exc:
        return {
            "status": "manual" if manual_score is not None else "unavailable",
            "mode": "manual",
            "backend": "manual_1_to_5",
            "score_0_1": _clamp(float(manual_score or 0.0) / 5.0) if manual_score is not None else None,
            "manual_score": manual_score,
            "videoclip": None,
            "viclip": None,
            "reference_source": str(reference_path),
            "note": "参考表情视频读取失败，回退人工评分。",
            "warnings": [str(exc)],
            "frame_records": [],
        }

    count = min(len(result_frames), len(reference_frames))
    viclip_result: dict[str, Any] | None = None
    viclip_warning: str | None = None
    viclip_window_records: list[dict[str, Any]] = []
    if use_viclip and viclip_enabled(device):
        try:
            _, _, semantic_windows = sample_aligned_video_windows(
                result_path,
                reference_path,
                max_frames,
            )
            window_scores: list[float] = []
            raw_scores: list[float] = []
            window_device = device
            for window in semantic_windows:
                window_result = viclip_video_similarity(
                    window["result_frames"],
                    window["reference_frames"],
                    device,
                    need_text=need_viclip_text,
                )
                window_device = str(window_result["device"])
                window_scores.append(float(window_result["score_0_1"]))
                raw_scores.append(float(window_result["raw_cosine"]))
                viclip_window_records.append(
                    {
                        "window_index": int(window["window_index"]),
                        "window_start_seconds": window["start_seconds"],
                        "window_end_seconds": window["end_seconds"],
                        "score_0_1": float(window_result["score_0_1"]),
                    }
                )
            score, tail_score = _robust_average(window_scores)
            if score is None:
                raise ValueError("ViCLIP produced no usable window scores.")
            viclip_result = {
                "score_0_1": score,
                "raw_cosine": _safe_mean(raw_scores),
                "window_mean_score": _safe_mean(window_scores),
                "window_lower_10pct_score": tail_score,
                "window_count": len(window_scores),
                "device": window_device,
                "frames": 8,
            }
        except Exception as exc:
            viclip_warning = f"ViCLIP unavailable: {exc}"

    result_motion, result_magnitude, result_backend = _motion_signature(
        result_frames[:count],
        detector,
        landmark_tracker,
    )
    reference_motion, reference_magnitude, reference_backend = _motion_signature(
        reference_frames[:count],
        detector,
        landmark_tracker,
    )
    if (
        len(result_motion) > 0
        and len(reference_motion) > 0
        and (
            result_motion.shape[1] != reference_motion.shape[1]
            or result_backend != reference_backend
        )
    ):
        # Compare both clips with the same fallback when Face Mesh coverage
        # differs between the generated and reference videos.
        landmark_tracker.mesh = None
        result_motion, result_magnitude, result_backend = _motion_signature(
            result_frames[:count],
            detector,
            landmark_tracker,
        )
        reference_motion, reference_magnitude, reference_backend = _motion_signature(
            reference_frames[:count],
            detector,
            landmark_tracker,
        )
    if len(result_motion) == 0 or len(reference_motion) == 0:
        if viclip_result is not None:
            return {
                "status": "available",
                "mode": "reference_viclip",
                "backend": "viclip_internvid_10m_flt",
                "score_0_1": viclip_result["score_0_1"],
                "manual_score": manual_score,
                "videoclip": None,
                "viclip": viclip_result,
                "reference_source": (
                    "reference_video" if reference_video else "gt_video"
                ),
                "note": "ViCLIP video-video similarity is active.",
                "warnings": [],
                "frame_records": [],
                "window_records": viclip_window_records,
            }
        score = _clamp(float(manual_score or 0.0) / 5.0)
        return {
            "status": "manual" if manual_score is not None else "unavailable",
            "mode": "manual",
            "backend": "manual_1_to_5",
            "score_0_1": score if manual_score is not None else None,
            "manual_score": manual_score,
            "videoclip": None,
            "viclip": None,
            "reference_source": "reference_video",
            "note": "未提取到足够运动特征，回退人工评分。",
            "warnings": ["参考视频或结果视频帧数不足，无法计算表情运动相似度。"],
            "frame_records": [],
        }

    pair_count = min(len(result_motion), len(reference_motion))
    cosine_values = [
        _motion_direction_similarity(
            result_motion[i],
            result_magnitude[i],
            reference_motion[i],
            reference_magnitude[i],
        )
        for i in range(pair_count)
    ]
    intensity_values = [
        _clamp(
            1.0
            - abs(
                math.log(
                    (float(result_magnitude[i]) + 1e-6)
                    / (float(reference_magnitude[i]) + 1e-6)
                )
            )
            / 4.0
        )
        for i in range(pair_count)
    ]
    proxy_score = 0.7 * float(np.mean(cosine_values)) + 0.3 * float(np.mean(intensity_values))
    score = (
        0.7 * float(viclip_result["score_0_1"]) + 0.3 * proxy_score
        if viclip_result is not None
        else proxy_score
    )
    backend = f"{result_backend} vs {reference_backend}"
    if viclip_result is not None:
        backend = f"viclip_internvid_10m_flt + {backend}"
    note = (
        "ViCLIP video similarity and face/画面运动轨迹 proxy are active."
        if viclip_result is not None
        else "VideoCLIP/ViCLIP 未安装，当前使用人脸/画面运动轨迹代理。"
    )
    return {
        "status": "available" if viclip_result is not None else "partial",
        "mode": (
            "reference_viclip"
            if viclip_result is not None
            else "reference_video_proxy"
        ),
        "backend": backend,
        "score_0_1": _clamp(score),
        "manual_score": manual_score,
        "videoclip": None,
        "viclip": viclip_result,
        "reference_source": "reference_video" if reference_video else "gt_video",
        "note": note,
        "warnings": [viclip_warning] if viclip_warning else [],
        "metrics": {
            "viclip_video_similarity": (
                viclip_result["score_0_1"] if viclip_result is not None else None
            ),
            "viclip_window_count": (
                viclip_result.get("window_count")
                if viclip_result is not None
                else 0
            ),
            "motion_proxy_score": _clamp(proxy_score),
        },
        "frame_records": [
            {
                "sample_index": i + 1,
                "transition_index": i,
                "from_frame": int(result_indices[i]),
                "to_frame": int(result_indices[i + 1]),
                "result_frame": int(result_indices[i + 1]),
                "timestamp_seconds": (
                    round(int(result_indices[i + 1]) / float(result_info["fps"]), 4)
                    if result_info["fps"] > 0
                    else None
                ),
                "expression_motion_similarity": round(
                    0.7 * cosine_values[i] + 0.3 * intensity_values[i],
                    6,
                ),
            }
            for i in range(pair_count)
        ],
        "window_records": viclip_window_records,
    }


def _flow(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        gray_a,
        gray_b,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def _warp_error(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    # Sample the previous frame for each target pixel using backward flow.
    backward_flow = _flow(frame_b, frame_a)
    height, width = frame_a.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (grid_x + backward_flow[:, :, 0]).astype(np.float32)
    map_y = (grid_y + backward_flow[:, :, 1]).astype(np.float32)
    warped = cv2.remap(frame_a, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return float(np.mean(np.abs(warped.astype(np.float32) - frame_b.astype(np.float32))) / 255.0)


def _flow_endpoint_error(frame_a: np.ndarray, frame_b: np.ndarray, reference_a: np.ndarray, reference_b: np.ndarray) -> float:
    reference_a = _align_ground_truth_frame(reference_a, frame_a)
    reference_b = _align_ground_truth_frame(reference_b, frame_b)
    generated_flow = _flow(frame_a, frame_b)
    reference_flow = _flow(reference_a, reference_b)
    difference = generated_flow - reference_flow
    return float(np.mean(np.linalg.norm(difference, axis=2)) / max(frame_a.shape[:2]))


TEMPORAL_LANDMARK_INDICES = np.array(
    [
        1,
        13,
        14,
        33,
        61,
        70,
        105,
        133,
        145,
        159,
        263,
        291,
        300,
        334,
        362,
        374,
        386,
    ],
    dtype=np.int64,
)


def _normalize_landmarks(landmarks: np.ndarray) -> np.ndarray | None:
    if (
        landmarks.ndim != 2
        or landmarks.shape[0] <= int(TEMPORAL_LANDMARK_INDICES.max())
    ):
        return None
    points = landmarks[TEMPORAL_LANDMARK_INDICES, :2].astype(np.float32)
    points -= points.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(np.square(points), axis=1))))
    if not math.isfinite(scale) or scale <= 1e-6:
        return None
    return points / scale


def _align_landmarks(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    covariance = points.T @ reference
    left, _, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return points @ rotation


def _face_box_jitter(
    frames: list[np.ndarray],
    detector: _FaceDetector,
    landmark_tracker: _LandmarkTracker,
) -> tuple[float | None, str]:
    if landmark_tracker.available:
        landmark_sequences: list[np.ndarray] = []
        for frame in frames:
            landmarks = landmark_tracker.extract(frame)
            if landmarks is None:
                landmark_sequences = []
                break
            normalized = _normalize_landmarks(landmarks)
            if normalized is None:
                landmark_sequences = []
                break
            landmark_sequences.append(normalized)
        if len(landmark_sequences) == len(frames) and len(landmark_sequences) >= 2:
            reference = landmark_sequences[0]
            aligned = np.stack(
                [_align_landmarks(points, reference) for points in landmark_sequences]
            )
            stacked = aligned
            if len(stacked) >= 3:
                displacement = np.diff(stacked, n=2, axis=0)
            else:
                displacement = np.diff(stacked, axis=0)
            return float(np.mean(np.linalg.norm(displacement, axis=2))), "mediapipe_landmark_jitter"

    normalized_boxes: list[np.ndarray] = []
    for frame in frames:
        bbox = detector.detect(frame)
        if bbox is None:
            continue
        x, y, width, height = bbox
        frame_height, frame_width = frame.shape[:2]
        normalized_boxes.append(
            np.array(
                [
                    (x + width / 2) / max(frame_width, 1),
                    (y + height / 2) / max(frame_height, 1),
                    width / max(frame_width, 1),
                    height / max(frame_height, 1),
                ],
                dtype=np.float32,
            )
        )
    if len(normalized_boxes) < 2:
        return None, "unavailable"
    differences = np.diff(np.stack(normalized_boxes), axis=0)
    return float(np.mean(np.linalg.norm(differences, axis=1))), "face_box_jitter_proxy"


def _iter_video_frames(
    path: str | Path,
) -> Iterable[tuple[dict[str, Any], int, np.ndarray]]:
    info = probe_video(path).to_dict()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    try:
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            yield (
                info,
                frame_index,
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            frame_index += 1
    finally:
        capture.release()


def _normalized_face_box(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> np.ndarray | None:
    if bbox is None:
        return None
    x, y, width, height = bbox
    frame_height, frame_width = frame.shape[:2]
    return np.array(
        [
            (x + width / 2) / max(frame_width, 1),
            (y + height / 2) / max(frame_height, 1),
            width / max(frame_width, 1),
            height / max(frame_height, 1),
        ],
        dtype=np.float32,
    )


def _jitter_from_landmark_sequences(
    sequences: list[np.ndarray | None],
) -> float | None:
    values: list[float] = []
    valid_indices = [
        index
        for index, sequence in enumerate(sequences)
        if sequence is not None
    ]
    if len(valid_indices) < 2:
        return None

    run_start = valid_indices[0]
    previous = run_start
    for current in valid_indices[1:] + [None]:
        if current is None or current != previous + 1:
            run = [
                sequence
                for sequence in sequences[run_start:previous + 1]
                if sequence is not None
            ]
            if len(run) >= 2:
                reference = run[0]
                aligned = np.stack(
                    [
                        _align_landmarks(points, reference)
                        for points in run
                    ]
                )
                displacement = (
                    np.diff(aligned, n=2, axis=0)
                    if len(aligned) >= 3
                    else np.diff(aligned, axis=0)
                )
                values.extend(
                    np.linalg.norm(displacement, axis=2).reshape(-1).tolist()
                )
            if current is None:
                break
            run_start = current
        previous = current
    return float(np.mean(values)) if values else None


def _jitter_from_box_sequences(
    sequences: list[np.ndarray | None],
) -> float | None:
    values: list[float] = []
    for previous, current in zip(sequences, sequences[1:]):
        if previous is None or current is None:
            continue
        values.append(float(np.linalg.norm(current - previous)))
    return float(np.mean(values)) if values else None


def _full_temporal_scan(
    result_path: str | Path,
    detector: _FaceDetector,
    landmark_tracker: _LandmarkTracker,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    list[float],
    float | None,
    str,
]:
    result_info: dict[str, Any] | None = None
    result_indices: list[int] = []
    self_errors: list[float] = []
    landmark_sequences: list[np.ndarray | None] = []
    box_sequences: list[np.ndarray | None] = []
    previous_frame: np.ndarray | None = None

    for info, frame_index, frame in _iter_video_frames(result_path):
        result_info = info
        result_indices.append(frame_index)
        if previous_frame is not None:
            self_errors.append(_warp_error(previous_frame, frame))
        previous_frame = frame

        landmarks = landmark_tracker.extract(frame)
        normalized_landmarks = (
            _normalize_landmarks(landmarks)
            if landmarks is not None
            else None
        )
        landmark_sequences.append(normalized_landmarks)
        box_sequences.append(
            _normalized_face_box(
                frame,
                detector.detect(frame) if normalized_landmarks is None else None,
            )
        )

    if result_info is None:
        raise ValueError("The video contains no readable frames.")

    landmark_jitter = _jitter_from_landmark_sequences(landmark_sequences)
    if landmark_jitter is not None:
        return (
            result_info,
            np.asarray(result_indices, dtype=np.int64),
            self_errors,
            landmark_jitter,
            "mediapipe_landmark_jitter",
        )
    box_jitter = _jitter_from_box_sequences(box_sequences)
    if box_jitter is not None:
        return (
            result_info,
            np.asarray(result_indices, dtype=np.int64),
            self_errors,
            box_jitter,
            "face_box_jitter_proxy",
        )
    return (
        result_info,
        np.asarray(result_indices, dtype=np.int64),
        self_errors,
        None,
        "unavailable",
    )


def evaluate_temporal(
    result_path: str | Path,
    ground_truth: str | Path | None,
    reference_video: str | Path | None,
    identity: dict[str, Any],
    max_frames: int,
) -> dict[str, Any]:
    detector = _FaceDetector()
    landmark_tracker = _LandmarkTracker()
    (
        result_info,
        result_indices,
        self_errors,
        jitter,
        jitter_backend,
    ) = _full_temporal_scan(
        result_path,
        detector,
        landmark_tracker,
    )
    reference_path = reference_video or ground_truth
    reference_result_frames: list[np.ndarray] = []
    reference_frames: list[np.ndarray] = []
    reference_result_indices = np.asarray([], dtype=np.int64)
    reference_error: str | None = None
    if reference_path:
        try:
            (
                _,
                reference_result_indices,
                reference_result_frames,
                reference_frames,
            ) = (
                _sample_aligned_videos(
                    result_path,
                    reference_path,
                    max_frames,
                )
            )
        except Exception as exc:
            reference_error = f"参考光流计算失败：{exc}"
    if len(result_indices) < 2 or len(self_errors) < 1:
        return {
            "status": "unavailable",
            "mode": "self_warping",
            "backend": "opencv_farneback",
            "metrics": {},
            "note": "结果视频至少需要两帧。",
            "warnings": [],
            "frame_records": [],
        }

    reference_epe: list[float] = []
    reference_epe_by_result_frame: dict[int, float] = {}
    if reference_frames:
        try:
            for i in range(len(reference_frames) - 1):
                value = _flow_endpoint_error(
                    reference_result_frames[i],
                    reference_result_frames[i + 1],
                    reference_frames[i],
                    reference_frames[i + 1],
                )
                reference_epe.append(value)
                if i + 1 < len(reference_result_indices):
                    reference_epe_by_result_frame[
                        int(reference_result_indices[i + 1])
                    ] = value
            if len(reference_epe) == 0 and reference_error is None:
                reference_error = "参考视频不足两帧，无法计算参考光流端点误差。"
        except Exception as exc:
            reference_epe = []
            reference_error = f"参考光流端点误差计算失败：{exc}"

    identity_variance = identity.get("metrics", {}).get("variance")
    score_parts = [1.0 - _clamp(float(np.mean(self_errors)))]
    if reference_epe:
        score_parts.append(1.0 - _clamp(float(np.mean(reference_epe))))
    if jitter is not None:
        score_parts.append(1.0 - _clamp(jitter * 10.0))
    if identity_variance is not None:
        score_parts.append(1.0 - _clamp(float(identity_variance) * 10.0))
    warnings: list[str] = []
    if jitter is None:
        warnings.append("未检测到足够人脸，landmark jitter 不可用。")
    elif jitter_backend != "mediapipe_landmark_jitter":
        warnings.append("未使用 MediaPipe Face Mesh，当前 jitter 是人脸框运动代理。")
    if reference_error:
        warnings.append(reference_error)
    return {
        "status": (
            "available"
            if jitter_backend == "mediapipe_landmark_jitter"
            else "partial"
        ),
        "mode": "reference_flow" if reference_epe else "self_warping",
        "backend": f"Farneback optical flow + {jitter_backend}",
        "metrics": {
            "generated_warping_error": _safe_mean(self_errors),
            "reference_flow_endpoint_error": _safe_mean(reference_epe),
            "landmark_jitter": jitter,
            "identity_similarity_variance": identity_variance,
            "generated_frame_count": int(len(result_indices)),
            "temporal_scan": "full_sequential_result_video",
            "stability_score_0_1": float(np.mean(score_parts)),
        },
        "note": (
            "有参考视频/GT 时比较结果与参考光流端点误差；"
            "否则计算结果视频自身的运动对齐 warping error。"
        ),
        "warnings": warnings,
        "frame_records": [
            {
                "sample_index": i + 1,
                "transition_index": i,
                "from_frame": int(result_indices[i]),
                "to_frame": int(result_indices[i + 1]),
                "result_frame": int(result_indices[i + 1]),
                "timestamp_seconds": (
                    round(int(result_indices[i + 1]) / float(result_info["fps"]), 4)
                    if result_info["fps"] > 0
                    else None
                ),
                "warping_error": round(self_errors[i], 6),
                "reference_flow_endpoint_error": (
                    round(
                        reference_epe_by_result_frame[
                            int(result_indices[i + 1])
                        ],
                        6,
                    )
                    if int(result_indices[i + 1])
                    in reference_epe_by_result_frame
                    else None
                ),
            }
            for i in range(len(self_errors))
        ],
    }


def _technical_aesthetic_score(frames: list[np.ndarray]) -> float | None:
    values: list[float] = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        brightness = float(gray.mean())
        exposure = 1.0 - _clamp(abs(brightness - 0.5) * 2.0)
        clipping = float(np.mean((gray < 0.02) | (gray > 0.98)))
        clipping_score = 1.0 - _clamp(clipping * 8.0)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        sharpness_score = 1.0 - math.exp(-sharpness * 8.0)
        colorfulness = float(np.std(frame.astype(np.float32), axis=(0, 1)).mean() / 80.0)
        values.append(float(np.mean([exposure, clipping_score, sharpness_score, _clamp(colorfulness)])))
    return _safe_mean(values)


def evaluate_aesthetics(
    result_path: str | Path,
    manual_score: float | None,
    max_frames: int,
    output_root: str | Path | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    _, _, frames = _sample_video(result_path, max_frames)
    technical_score = _technical_aesthetic_score(frames)
    if manual_score is not None:
        return {
        "status": "manual" if manual_score is not None else "unavailable",
        "mode": "manual",
        "backend": "manual_1_to_5 + technical_proxy",
        "score_0_1": _clamp(float(manual_score) / 5.0),
        "metrics": {
            "manual_score_1_to_5": manual_score,
            "technical_proxy_0_to_1": technical_score,
            "manual_score_0_to_1": (
                _clamp(float(manual_score) / 5.0) if manual_score is not None else None
            ),
        },
        "note": "截图注明暂无成熟自动方法，因此人工评分为主，技术质量仅作辅助。",
        "warnings": [],
        "frame_records": [],
    }

    metrics = {
        "manual_score_1_to_5": None,
        "technical_proxy_0_to_1": technical_score,
        "manual_score_0_to_1": None,
        "vbench_aesthetic_quality_raw": None,
        "vbench_aesthetic_quality_0_to_1": None,
    }
    if device == "cpu":
        return {
            "status": "unavailable",
            "mode": "vbench",
            "backend": "vbench_aesthetic_quality",
            "score_0_1": None,
            "metrics": metrics,
            "note": "VBench aesthetic_quality requires CUDA; no manual score was provided.",
            "warnings": [
                "VBench aesthetic_quality was skipped because the effective device is CPU."
            ],
            "frame_records": [],
        }
    warnings: list[str] = []
    vbench_result: dict[str, Any] = {}
    try:
        vbench_result = run_vbench(
            result_path,
            ["aesthetic_quality"],
            output_root or OUTPUT_DIR,
        )
    except Exception as exc:
        warnings.append(
            f"VBench aesthetic_quality failed: {type(exc).__name__}: {exc}"
        )

    records = vbench_result.get("records", [])
    record = next(
        (
            item
            for item in records
            if item.get("dimension") == "aesthetic_quality"
        ),
        {},
    )
    raw_score = record.get("score")
    normalized_score: float | None = None
    if raw_score is not None and vbench_result.get("status") == "completed":
        try:
            parsed_score = float(raw_score)
        except (TypeError, ValueError):
            parsed_score = None
        if parsed_score is not None and math.isfinite(parsed_score):
            if 0.0 <= parsed_score <= 1.0:
                normalized_score = parsed_score
                metrics["vbench_aesthetic_quality_raw"] = parsed_score
                metrics["vbench_aesthetic_quality_0_to_1"] = parsed_score
            else:
                warnings.append(
                    "VBench aesthetic_quality returned a score outside the expected 0-1 range."
                )
    elif raw_score is not None:
        warnings.append(
            "VBench returned a score, but the evaluation process did not complete successfully."
        )

    if normalized_score is not None:
        return {
            "status": "available",
            "mode": "vbench",
            "backend": "vbench_aesthetic_quality",
            "score_0_1": normalized_score,
            "metrics": metrics,
            "vbench": {
                "status": vbench_result.get("status"),
                "backend": vbench_result.get("backend"),
                "output_dir": vbench_result.get("output_dir"),
                "record": record,
            },
            "note": (
                "No manual aesthetic score was provided; "
                "VBench aesthetic_quality is the primary score."
            ),
            "warnings": warnings,
            "frame_records": [],
        }

    failure_detail = (
        vbench_result.get("installation")
        or vbench_result.get("stderr")
        or vbench_result.get("status")
        or "No VBench aesthetic_quality score was returned."
    )
    warnings.append(f"VBench aesthetic_quality unavailable: {failure_detail}")
    return {
        "status": "unavailable",
        "mode": "vbench",
        "backend": "vbench_aesthetic_quality",
        "score_0_1": None,
        "metrics": metrics,
        "vbench": {
            "status": vbench_result.get("status", "unavailable"),
            "backend": vbench_result.get("backend"),
            "output_dir": vbench_result.get("output_dir"),
            "record": record,
        },
        "note": (
            "No manual aesthetic score was provided, but VBench "
            "aesthetic_quality did not return a usable score."
        ),
        "warnings": warnings,
        "frame_records": [],
    }


def _format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "不可用"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def evaluate_all(
    result_path: str | Path,
    ground_truth: str | Path | None,
    reference_image: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    reference_video: str | Path | None,
    prompt_text: str | None = None,
    max_frames: int = 64,
    calculate_lpips: bool = True,
    device: str = "auto",
    manual_expression_score: float | None = None,
    manual_aesthetic_score: float | None = None,
    vbench_output_root: str | Path | None = None,
) -> dict[str, Any]:
    policy = resolve_policy(device)
    effective_device = policy.resolved_device
    qwen_service_active = etva_service_available()
    allow_copresent = os.environ.get(
        "EVALUATOR_ALLOW_COPRESENT_MODELS",
        "0",
    ).lower() in {"1", "true", "yes", "on"}
    use_viclip = not qwen_service_active or allow_copresent
    identity = evaluate_identity(
        result_path,
        reference_image,
        reference_video,
        ground_truth,
        max_frames,
        device=effective_device,
    )
    texture = evaluate_texture(
        result_path,
        ground_truth,
        reference_image,
        reference_video,
        max_frames,
        calculate_lpips,
        effective_device,
    )
    aesthetics = evaluate_aesthetics(
        result_path,
        manual_aesthetic_score,
        max_frames,
        output_root=vbench_output_root,
        device=effective_device,
    )
    expression = evaluate_expression(
        result_path,
        ground_truth,
        reference_video,
        manual_expression_score,
        max_frames,
        device=effective_device,
        use_viclip=use_viclip,
        need_viclip_text=bool(prompt_text and prompt_text.strip()),
    )
    text_alignment = evaluate_text_alignment(
        result_path,
        prompt_text,
        max_frames,
        effective_device,
        use_viclip=use_viclip,
    )
    text_score = (
        text_alignment.get("metrics", {}).get("score_0_1")
        if text_alignment["status"] != "unavailable"
        else None
    )
    if text_score is not None:
        expression.setdefault("metrics", {})["text_video_alignment"] = text_score
        expression.setdefault("frame_records", []).extend(
            text_alignment.get("frame_records", [])
        )
    expression_style_score = expression.get("score_0_1")
    # Do not keep local ViCLIP resident while the external VLM judge runs.
    clear_viclip_cache()
    etva_judge = evaluate_etva_judge(
        result_path=result_path,
        prompt_text=prompt_text,
        reference_path=reference_video or ground_truth,
        max_frames=max_frames,
        window_frames=policy.etva_frames,
        service_available=qwen_service_active,
    )
    etva_score = (
        etva_judge.get("score_0_1")
        if etva_judge["status"] == "available"
        else None
    )
    if etva_score is not None:
        expression.setdefault("metrics", {})["etva_judge_score_0_1"] = etva_score
        expression.setdefault("metrics", {})["etva_window_count"] = etva_judge.get(
            "metrics",
            {},
        ).get("window_count")
    elif etva_judge.get("warnings"):
        expression.setdefault("warnings", []).extend(etva_judge["warnings"])

    semantic_score = (
        float(etva_score)
        if etva_score is not None
        else (float(text_score) if text_score is not None else None)
    )
    semantic_backend = (
        etva_judge.get("backend")
        if etva_score is not None
        else text_alignment.get("backend")
        if text_score is not None
        else None
    )
    if semantic_score is not None and semantic_backend:
        if expression_style_score is not None:
            expression["score_0_1"] = _clamp(
                0.6 * float(expression_style_score)
                + 0.4 * semantic_score
            )
        else:
            expression["score_0_1"] = _clamp(semantic_score)
        expression["backend"] = (
            f'{expression.get("backend", "manual")} + {semantic_backend}'
        )
        if etva_score is not None:
            expression["status"] = (
                "available"
                if "viclip_internvid_10m_flt" in expression.get("backend", "")
                else "partial"
            )
            expression["note"] = (
                f'{expression.get("note", "")} ETVA Qwen judge is active.'
            ).strip()
        else:
            expression["status"] = (
                "available"
                if expression.get("status") == "available"
                and text_alignment["backend"] == "viclip_internvid_10m_flt"
                else "partial"
            )
            expression["note"] = (
                f'{expression.get("note", "")} '
                "已加入文本-视频语义对齐分数。"
            ).strip()

    temporal = evaluate_temporal(
        result_path,
        ground_truth,
        reference_video,
        identity,
        max_frames,
    )
    categories = {
        "identity": identity,
        "texture": texture,
        "expression": expression,
        "temporal": temporal,
        "aesthetics": aesthetics,
    }
    aesthetics_score = aesthetics.get("score_0_1")
    if aesthetics_score is None:
        aesthetics_score = aesthetics.get("metrics", {}).get(
            "manual_score_0_to_1"
        )
    category_scores = {
        "identity": identity.get("metrics", {}).get("score_0_1"),
        "texture": texture.get("metrics", {}).get("score_0_1"),
        "expression": expression.get("score_0_1"),
        "temporal": temporal.get("metrics", {}).get("stability_score_0_1"),
        "aesthetics": aesthetics_score,
    }
    for category_name, score in category_scores.items():
        # Keep one canonical category score for API consumers and the UI.
        categories[category_name]["score_0_1"] = score
    summary = [
        {
            "类别": "1. 角色一致性",
            "权重": "35%",
            "状态": identity["status"],
            "标准化分数": _format_metric(
                identity.get("metrics", {}).get("score_0_1"),
                4,
            ),
            "核心结果": (
                f'平均 { _format_metric(identity.get("metrics", {}).get("mean_similarity"), 4) }；'
                f'尾部10% { _format_metric(identity.get("metrics", {}).get("tail_10pct_similarity"), 4) }；'
                f'方差 { _format_metric(identity.get("metrics", {}).get("variance"), 6) }'
            ),
            "后端": identity.get("backend"),
        },
        {
            "类别": "2. 质感和细节",
            "权重": "15%",
            "状态": texture["status"],
            "标准化分数": _format_metric(
                texture["metrics"].get("score_0_1"),
                4,
            ),
            "核心结果": (
                f'PSNR {_format_metric(texture["metrics"].get("psnr_db"))}；'
                f'SSIM {_format_metric(texture["metrics"].get("ssim"), 6)}；'
                f'LPIPS {_format_metric(texture["metrics"].get("lpips"), 6)}'
            )
            if texture["mode"] == "full_reference"
            else (
                f'MANIQA {_format_metric(texture["metrics"].get("maniqa"), 6)}；'
                f'MUSIQ {_format_metric(texture["metrics"].get("musiq"), 6)}；'
                f'高频比 {_format_metric(texture["metrics"].get("high_frequency_ratio"), 6)}'
            ),
            "后端": texture.get("backend"),
        },
        {
            "类别": "3. 表情准确",
            "权重": "15%",
            "状态": expression["status"],
            "标准化分数": _format_metric(expression.get("score_0_1"), 4),
            "核心结果": (
                f'综合得分 {_format_metric(expression.get("score_0_1"), 4)} / 1.0；'
                f'文本-视频 {_format_metric(expression.get("metrics", {}).get("text_video_alignment"), 4)}'
            ),
            "后端": expression.get("backend"),
        },
        {
            "类别": "4. 时间稳定性",
            "权重": "25%",
            "状态": temporal["status"],
            "标准化分数": _format_metric(
                temporal.get("metrics", {}).get("stability_score_0_1"),
                4,
            ),
            "核心结果": (
                f'稳定性 {_format_metric(temporal.get("metrics", {}).get("stability_score_0_1"), 4)}；'
                f'warping {_format_metric(temporal.get("metrics", {}).get("generated_warping_error"), 6)}；'
                f'jitter {_format_metric(temporal.get("metrics", {}).get("landmark_jitter"), 6)}'
            ),
            "后端": temporal.get("backend"),
        },
        {
            "类别": "5. 美学质量",
            "权重": "10%",
            "状态": aesthetics["status"],
            "标准化分数": _format_metric(
                aesthetics.get("score_0_1"),
                4,
            ),
            "核心结果": (
                f'人工 {_format_metric(aesthetics.get("metrics", {}).get("manual_score_1_to_5"), 2)} / 5；'
                f'技术代理 {_format_metric(aesthetics.get("metrics", {}).get("technical_proxy_0_to_1"), 4)}'
            ),
            "后端": aesthetics.get("backend"),
        },
    ]

    frame_records: dict[int, dict[str, Any]] = {}

    def merge_records(records: list[dict[str, Any]]) -> None:
        for record in records:
            key = int(record.get("result_frame", record.get("sample_index", len(frame_records))))
            frame_records.setdefault(
                key,
                {
                    "sample_index": record.get("sample_index", len(frame_records)),
                    "result_frame": key,
                },
            )
            frame_records[key].update(record)

    merge_records(texture.get("frame_records", []))
    merge_records(identity.get("frame_records", []))
    merge_records(expression.get("frame_records", []))
    merge_records(temporal.get("frame_records", []))
    for record in frame_records.values():
        if "timestamp_seconds" not in record:
            record["timestamp_seconds"] = None
    frame_table = sorted(frame_records.values(), key=lambda record: record["result_frame"])

    warnings: list[str] = []
    category_labels = {
        "identity": "角色一致性",
        "texture": "质感和细节",
        "expression": "表情准确",
        "temporal": "时间稳定性",
        "aesthetics": "美学",
    }
    for category_name, category in categories.items():
        warnings.extend(category.get("warnings", []))
        if category["status"] in {"partial", "unavailable"}:
            reason = category.get("reason")
            note = category.get("note")
            if reason:
                warnings.append(f"{category_labels[category_name]}：{reason}")
            if note:
                warnings.append(f"{category_labels[category_name]}后端说明：{note}")
    score_entries = [
        (WEIGHTS[category_name], category_scores[category_name])
        for category_name in WEIGHTS
    ]
    valid_score_entries = [
        (weight, float(score))
        for weight, score in score_entries
        if score is not None and math.isfinite(float(score))
    ]
    score_weight = sum(weight for weight, _ in valid_score_entries)
    weighted_score = (
        sum(weight * score for weight, score in valid_score_entries) / score_weight
        if score_weight
        else None
    )
    all_scores_available = len(valid_score_entries) == len(score_entries)
    all_backends_complete = all(
        category["status"] in {"available", "manual"}
        for category in categories.values()
    )
    report_status = (
        "complete"
        if all_scores_available and all_backends_complete
        else "partial"
    )
    evaluation_mode = (
        "full_reference"
        if texture.get("mode") == "full_reference"
        else (
            "reference_material"
            if reference_image or reference_video
            else (
                "prompt_only"
                if (prompt_text or "").strip()
                else "result_only"
            )
        )
    )
    return {
        "status": report_status,
        "coverage": f"{len(valid_score_entries)}/5",
        "evaluation_mode": evaluation_mode,
        "prompt_text": prompt_text,
        "sampling_policy": {
            "regular_sample_fps": DEFAULT_SAMPLE_FPS,
            "max_frames": int(max_frames),
            "temporal_scan": "full_sequential_result_video",
            "semantic_window_frames": SEMANTIC_WINDOW_FRAMES,
            "semantic_window_seconds": SEMANTIC_WINDOW_SECONDS,
            "semantic_window_overlap": SEMANTIC_WINDOW_OVERLAP,
            "semantic_window_aggregation": (
                "0.8 * window_mean + 0.2 * lower_10pct_window_mean"
            ),
        },
        "weighted_score_0_1": weighted_score,
        "weighted_score_0_100": weighted_score * 100 if weighted_score is not None else None,
        "weighted_score_weight_coverage": score_weight,
        "weighted_score_status": (
            "complete" if all_scores_available else "partial"
        ),
        "weights": WEIGHTS,
        "evaluation_plan": {
            "full_reference_metrics": (
                "仅当上传逐帧对应的 GT 视频时，"
                "第 2 类计算 PSNR/SSIM/LPIPS。"
            ),
            "identity_reference": "参考图 > 参考视频 > GT 视频。",
            "expression_reference": (
                "参考视频/GT 计算表情与动作风格分；Prompt 或 ETVA 计算语义分；"
                "缺少风格参考时使用人工 1~5 分，二者同时存在时按 60% 风格 + 40% 语义合成。"
            ),
            "condition_alignment": "Prompt + 生成视频使用逐帧 CLIP 基线；不替代 VideoCLIP/ViCLIP。",
            "temporal_reference": "参考视频 > GT 视频 > 结果视频自对齐。",
            "aesthetic_reference": "人工 1~5 分为主，技术代理为辅助。",
        },
        "input_contract": {
            "result_video": "必填，待评估的生成结果视频。",
            "ground_truth_video": (
                "可选，但必须与结果在内容和时间上逐帧对应；"
                "仅此输入启用第 2 类 PSNR/SSIM/LPIPS。"
            ),
            "reference_image": (
                "可选，用于角色外观和身份一致性，不作为 PSNR/SSIM/LPIPS 的 GT。"
            ),
            "reference_video": (
                "可选，用于表情、动作和时间稳定性，不作为 PSNR/SSIM/LPIPS 的 GT。"
            ),
            "prompt_text": "可选，用于文本-视频语义对齐。",
        },
        "summary": summary,
        "categories": categories,
        "hardware_policy": policy.to_dict(),
        "qwen_service_active": qwen_service_active,
        "condition_alignment": text_alignment,
        "etva_judge": etva_judge,
        "frame_records": frame_table,
        "warnings": warnings,
    }
