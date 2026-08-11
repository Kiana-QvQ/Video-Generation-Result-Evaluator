#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频生成模型结果评估脚本

设计目标：
1. 生成视频是必填输入；
2. 人物图像和参考视频是可选输入；
3. 评分结构对应需求截图：
   - 角色一致性：35%
   - 质感和细节：15%
   - 表情/文本准确性：15%
   - 时间稳定性：25%
   - 美学：10%
4. 使用 Qwen3-VL-4B-Instruct 模型自动评估 QA 问题，替代 CLIP
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from detail_expression_metrics import (
    compute_detail_metric as compute_detail_metric_extension,
    compute_face_expression_metric as compute_face_expression_metric_extension,
)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    import torch
    from PIL import Image
    PIL_AVAILABLE = True
    QWEN_AVAILABLE = True
except ImportError as e:
    QWEN_AVAILABLE = False
    Qwen3VLForConditionalGeneration = None
    AutoProcessor = None
    torch = None
    PIL_AVAILABLE = False

LOGGER = logging.getLogger("video_generation_evaluator")
PROJECT_DIR = Path(__file__).resolve().parent

# Model initialization is much more expensive than one evaluation. Reuse
# heavyweight models across web requests while serializing GPU generation.
_QWEN_MODEL_CACHE: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
_QWEN_MODEL_CACHE_LOCK = threading.Lock()
_QWEN_INFERENCE_LOCK = threading.Lock()
_FACE_ANALYZER_CACHE: Dict[Tuple[str, str], "FaceAnalyzer"] = {}
_FACE_ANALYZER_CACHE_LOCK = threading.Lock()
_EXPRESSION_REFERENCE_CACHE: Dict[
    str,
    Tuple[Tuple[Tuple[str, int, int], ...], List["ExpressionReference"]],
] = {}
_EXPRESSION_REFERENCE_CACHE_LOCK = threading.Lock()


def resolve_project_path(path: Optional[str]) -> Optional[Path]:
    """Resolve relative paths from the evaluator project directory."""

    if path is None:
        return None
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = PROJECT_DIR / value
    return value.resolve()

WEIGHTS = {
    # The five visible quality dimensions retain the original total weights.
    "identity": 0.35,
    "detail": 0.15,
    "expression_text": 0.15,
    "face_expression": 0.15,
    "temporal": 0.25,
    "aesthetic": 0.10,
}


@dataclass
class VideoSamples:
    """视频采样结果。帧默认使用 BGR 排列，便于直接交给 OpenCV。"""

    path: str
    frames: List[Any]
    indices: List[int]
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.fps <= 0 or self.frame_count <= 0:
            return None
        return float(self.frame_count / self.fps)


@dataclass
class FaceObservation:
    """单帧人脸检测结果。"""

    bbox: Optional[Tuple[float, float, float, float]] = None
    embedding: Optional[Any] = None
    detector: str = "none"
    confidence: Optional[float] = None


@dataclass
class ExpressionReference:
    """One action/view prototype from the internal Expression directory."""

    action: str
    view: str
    path: str
    face: FaceObservation
    landmarks: Optional[Any]
    geometry: Optional[Any]
    gaze: Optional[Any]
    structure: Any
    detail: Any
    region_energy: Any


@dataclass
class MetricResult:
    """一个顶层评估项的结果。"""

    name: str
    score: Optional[float]
    weight: float
    status: str
    details: Dict[str, Any] = field(default_factory=dict)


def require_basic_dependencies() -> None:
    """在真正读取视频前检查基础依赖。"""

    missing = []
    if np is None:
        missing.append("numpy")
    if cv2 is None:
        missing.append("opencv-python")
    if missing:
        raise RuntimeError(
            "缺少基础依赖："
            + ", ".join(missing)
            + "。请先运行：pip install "
            + " ".join(missing)
        )


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """把数值限制在指定区间。"""

    return float(max(low, min(high, value)))


def safe_mean(values: Sequence[float]) -> Optional[float]:
    """计算非空平均值。"""

    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def safe_std(values: Sequence[float]) -> Optional[float]:
    """计算非空标准差。"""

    if not values:
        return None
    return float(np.std(np.asarray(values, dtype=np.float32)))


def normalize_score_1_to_5(value: Optional[float]) -> Optional[float]:
    """将人工 1~5 分映射到 0~1。"""

    if value is None:
        return None
    return clamp(float(value) / 5.0)


def cosine_similarity(a: Any, b: Any) -> Optional[float]:
    """计算两个向量的余弦相似度。"""

    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < 1e-8 or b_norm < 1e-8:
        return None
    return float(np.dot(a, b) / (a_norm * b_norm))


def similarity_to_face_score(
        similarity: float,
        low: float = 0.25,
        high: float = 0.65,
) -> float:
    """
    将 ArcFace 余弦相似度做一个可解释的阈值标定。

    low/high 不是通用真值，而是工程默认值。不同 ArcFace 模型、数据集和
    摄像机角度会导致相似度分布不同，建议在自己的验证集上重新标定。
    """

    if high <= low:
        raise ValueError("face-sim-high 必须大于 face-sim-low")
    return clamp((float(similarity) - low) / (high - low))


def robust_scale(
        values: Sequence[float],
        low_percentile: float = 10.0,
        high_percentile: float = 90.0,
) -> Optional[float]:
    """用分位数把一组质量数值映射到 0~1，减少异常帧的影响。"""

    if not values:
        return None
    arr = np.asarray(values, dtype=np.float32)
    low = float(np.percentile(arr, low_percentile))
    high = float(np.percentile(arr, high_percentile))
    if high - low < 1e-8:
        return 0.5
    return clamp((float(np.median(arr)) - low) / (high - low))


def read_video_samples(path: str, max_frames: int) -> VideoSamples:
    """均匀读取视频帧，避免将长视频全部加载到内存。"""

    require_basic_dependencies()
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在：{video_path}")
    if max_frames < 2:
        raise ValueError("max_frames 至少需要为 2")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames: List[Any] = []
    indices: List[int] = []

    if frame_count > 0:
        sample_count = min(max_frames, frame_count)
        target_indices = np.linspace(
            0, frame_count - 1, sample_count, dtype=np.int64
        )
        target_position = 0
        current_index = 0
        target_list = target_indices.tolist()
        while target_position < len(target_list):
            # Sequential grab avoids a costly key-frame seek for every sample.
            ok = capture.grab()
            if not ok:
                break
            if current_index == int(target_list[target_position]):
                ok, frame = capture.retrieve()
                if ok and frame is not None:
                    frames.append(frame)
                    indices.append(current_index)
                target_position += 1
            current_index += 1
    else:
        # 某些编码格式拿不到总帧数，只能顺序读取后均匀保留。
        all_frames: List[Any] = []
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            all_frames.append(frame)
        frame_count = len(all_frames)
        if frame_count:
            selected = np.linspace(
                0, frame_count - 1, min(max_frames, frame_count), dtype=np.int64
            )
            for index in selected.tolist():
                frames.append(all_frames[int(index)])
                indices.append(int(index))

    capture.release()

    if not frames:
        raise RuntimeError(f"视频未读取到有效帧：{video_path}")
    if width <= 0 or height <= 0:
        height, width = frames[0].shape[:2]

    return VideoSamples(
        path=str(video_path),
        frames=frames,
        indices=indices,
        fps=fps,
        frame_count=frame_count or len(frames),
        width=width,
        height=height,
    )


def read_image(path: str) -> Any:
    """读取一张 BGR 图像。"""

    require_basic_dependencies()
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"人物图像不存在：{image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图像：{image_path}")
    return image


def resolve_person_image_paths(inputs: Optional[Sequence[str]]) -> List[Path]:
    """
    展开人物参考图输入。

    --person-image 可以重复传入多个文件，也可以传入一个目录或通配符：
        --person-image view_front.jpg --person-image view_side.jpg
        --person-image ./person_views/
        --person-image "./person_views/*.png"
    """

    if not inputs:
        return []

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }
    resolved: List[Path] = []
    seen = set()

    for raw_input in inputs:
        raw_path = Path(raw_input)
        has_wildcard = any(char in raw_input for char in "*?[")

        if has_wildcard:
            candidates = [
                Path(path)
                for path in sorted(glob.glob(raw_input))
            ]
            if not candidates:
                raise FileNotFoundError(
                    f"人物参考图通配符没有匹配到文件：{raw_input}"
                )
        elif raw_path.is_dir():
            candidates = sorted(
                (
                    path
                    for path in raw_path.iterdir()
                    if path.is_file()
                       and path.suffix.lower() in image_extensions
                ),
                key=lambda path: path.name.lower(),
            )
            if not candidates:
                raise FileNotFoundError(
                    f"人物参考图目录中没有支持的图片：{raw_path}"
                )
        else:
            candidates = [raw_path]

        for candidate in candidates:
            if not candidate.exists():
                raise FileNotFoundError(f"人物图像不存在：{candidate}")
            if not candidate.is_file():
                raise ValueError(f"人物参考图不是文件：{candidate}")
            if candidate.suffix.lower() not in image_extensions:
                raise ValueError(
                    f"不支持的人物参考图格式：{candidate.suffix}，"
                    f"支持 {sorted(image_extensions)}"
                )
            absolute_path = candidate.resolve()
            path_key = str(absolute_path).lower()
            if path_key not in seen:
                seen.add(path_key)
                resolved.append(absolute_path)

    if not resolved:
        raise ValueError("没有解析出任何人物参考图")
    return resolved


def read_images(paths: Sequence[Path]) -> List[Any]:
    """批量读取人物参考图。"""

    return [read_image(str(path)) for path in paths]


def bbox_area(bbox: Optional[Tuple[float, float, float, float]]) -> float:
    """计算人脸框面积。"""

    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_to_tuple(bbox: Any) -> Tuple[float, float, float, float]:
    """将不同检测器的 bbox 转成统一格式。"""

    values = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]
    return tuple(float(v) for v in values)  # type: ignore[return-value]


def expand_bbox(
        bbox: Tuple[float, float, float, float],
        width: int,
        height: int,
        margin: float = 0.20,
) -> Tuple[int, int, int, int]:
    """给人脸框增加边界，截取包含头发和下巴的局部区域。"""

    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    x1 = int(max(0, math.floor(x1 - margin * w)))
    y1 = int(max(0, math.floor(y1 - margin * h)))
    x2 = int(min(width, math.ceil(x2 + margin * w)))
    y2 = int(min(height, math.ceil(y2 + margin * h)))
    return x1, y1, x2, y2


class FaceAnalyzer:
    """
    人脸检测和 ArcFace 特征提取。

    首选 InsightFace buffalo_l。若不可用，则回退到 OpenCV Haar 人脸检测。
    回退模式没有真正的人脸身份 embedding，只能提供"人脸出现率/框稳定性"
    代理分。
    """

    def __init__(self, device: str = "auto", model_name: str = "buffalo_l"):
        require_basic_dependencies()
        self.device = device
        self.model_name = model_name
        self.insightface_app = None
        self.haar = None
        self.detector_name = "none"
        self._try_init_insightface()
        if self.insightface_app is None:
            self._try_init_haar()

    def _try_init_insightface(self) -> None:
        try:
            from insightface.app import FaceAnalysis

            providers = None
            ctx_id = -1
            if self.device in {"auto", "cuda", "gpu"}:
                providers = [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]
                ctx_id = 0
            elif self.device == "cpu":
                providers = ["CPUExecutionProvider"]
                ctx_id = -1

            kwargs: Dict[str, Any] = {"name": self.model_name}
            if providers is not None:
                kwargs["providers"] = providers
            self.insightface_app = FaceAnalysis(**kwargs)
            self.insightface_app.prepare(
                ctx_id=ctx_id,
                det_size=(640, 640),
            )
            self.detector_name = f"insightface:{self.model_name}"
            LOGGER.info("已启用 %s", self.detector_name)
        except Exception as exc:
            LOGGER.warning(
                "InsightFace 不可用，将回退到 OpenCV Haar。原因：%s", exc
            )
            self.insightface_app = None

    def _try_init_haar(self) -> None:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            classifier = cv2.CascadeClassifier(cascade_path)
            if not classifier.empty():
                self.haar = classifier
                self.detector_name = "opencv-haar-proxy"
                LOGGER.info("已启用 %s", self.detector_name)
        except Exception as exc:
            LOGGER.warning("OpenCV Haar 也不可用：%s", exc)

    def analyze(self, frame: Any) -> FaceObservation:
        """选择面积最大的人脸作为当前帧主人物。"""

        if self.insightface_app is not None:
            try:
                faces = self.insightface_app.get(frame)
                if faces:
                    face = max(
                        faces,
                        key=lambda item: bbox_area(bbox_to_tuple(item.bbox)),
                    )
                    bbox = bbox_to_tuple(face.bbox)
                    embedding = getattr(face, "normed_embedding", None)
                    if embedding is None:
                        embedding = getattr(face, "embedding", None)
                    confidence = getattr(face, "det_score", None)
                    return FaceObservation(
                        bbox=bbox,
                        embedding=embedding,
                        detector=self.detector_name,
                        confidence=(
                            float(confidence) if confidence is not None else None
                        ),
                    )
            except Exception as exc:
                LOGGER.debug("InsightFace 单帧推理失败：%s", exc)

        if self.haar is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self.haar.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(24, 24),
            )
            if len(detections):
                x, y, w, h = max(
                    detections,
                    key=lambda item: int(item[2]) * int(item[3]),
                )
                return FaceObservation(
                    bbox=(float(x), float(y), float(x + w), float(y + h)),
                    detector=self.detector_name,
                )

        return FaceObservation(detector=self.detector_name)

    def analyze_frames(self, frames: Sequence[Any]) -> List[FaceObservation]:
        """批量提取人脸信息。"""

        return [self.analyze(frame) for frame in frames]


def get_cached_face_analyzer(device: str, model_name: str) -> FaceAnalyzer:
    """Reuse InsightFace/Haar initialization across evaluations."""

    key = (device, model_name)
    with _FACE_ANALYZER_CACHE_LOCK:
        analyzer = _FACE_ANALYZER_CACHE.get(key)
        if analyzer is None:
            analyzer = FaceAnalyzer(device=device, model_name=model_name)
            _FACE_ANALYZER_CACHE[key] = analyzer
        else:
            LOGGER.info("Reuse cached face analyzer: %s", analyzer.detector_name)
        return analyzer


class LandmarkAnalyzer:
    """
    MediaPipe Face Mesh 关键点提取器。

    关键点用于表情序列和时间稳定性，不直接把头部平移、缩放误判成表情变化。
    """

    def __init__(self):
        require_basic_dependencies()
        self.face_mesh = None
        self.backend = "none"
        try:
            import mediapipe as mp

            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.backend = "mediapipe-face-mesh"
            LOGGER.info("已启用 %s", self.backend)
        except Exception as exc:
            LOGGER.warning("MediaPipe 不可用，表情关键点项将按需降级：%s", exc)

    def extract(self, frame: Any) -> Optional[Any]:
        """提取一帧的人脸关键点，返回归一化坐标。"""

        if self.face_mesh is None:
            return None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb)
            if not result.multi_face_landmarks:
                return None
            points = result.multi_face_landmarks[0].landmark
            return np.asarray(
                [[point.x, point.y, point.z] for point in points],
                dtype=np.float32,
            )
        except Exception as exc:
            LOGGER.debug("MediaPipe 单帧推理失败：%s", exc)
            return None

    def extract_frames(self, frames: Sequence[Any]) -> List[Optional[Any]]:
        """批量提取关键点。"""

        return [self.extract(frame) for frame in frames]

    def close(self) -> None:
        """释放 MediaPipe 资源。"""

        if self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except Exception:
                pass


class QwenVLScorer:
    """
    使用 Qwen3-VL-4B-Instruct 模型评估视频帧与问题的匹配度。
    替代 CLIP 功能。
    """

    def __init__(
            self,
            model_path: str,
            device: str = "auto",
            max_frames: int = 8,
            qa_json_path: Optional[str] = None,
    ):
        if not QWEN_AVAILABLE:
            raise RuntimeError(
                "Qwen3-VL 依赖不可用。请安装：pip install transformers torch pillow\n"
                "注意：需要使用 transformers >= 4.45.0 版本"
            )

        self.model_path = resolve_project_path(model_path)
        if self.model_path is None:
            raise ValueError("Qwen 模型路径不能为空")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Qwen 模型路径不存在：{model_path}")

        self.max_frames = max_frames
        self.scoring_instruction = (
            "分数为 0、0.5 或 1，其中 0=不符合，0.5=部分符合，1=完全符合。"
        )

        # 确定设备
        if device in {"auto", "cuda", "gpu"}:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        LOGGER.info(
            "Qwen 模型加载中，设备: %s, 路径: %s",
            self.device,
            self.model_path,
        )

        # 加载模型和处理器 - 使用与 test.py 相同的方式
        cache_key = (str(self.model_path), self.device)
        with _QWEN_MODEL_CACHE_LOCK:
            cached_bundle = _QWEN_MODEL_CACHE.get(cache_key)
        try:
            if cached_bundle is None:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    str(self.model_path),
                    dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                )
                self.processor = AutoProcessor.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=True,
                )
                self.model.eval()
                if self.device == "cuda":
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
                with _QWEN_MODEL_CACHE_LOCK:
                    _QWEN_MODEL_CACHE[cache_key] = (self.model, self.processor)
        except Exception as e:
            LOGGER.error(f"模型加载失败: {e}")
            raise
        if cached_bundle is not None:
            self.model, self.processor = cached_bundle
            LOGGER.info("Reuse cached Qwen model: %s", self.model_path)

        # 加载 QA 问题
        self.qa_questions = self._load_qa_questions(qa_json_path)

        LOGGER.info("Qwen 模型加载完成，共有 %d 个 QA 问题", len(self.qa_questions))

    def _load_qa_questions(self, qa_json_path: Optional[str]) -> List[Dict[str, Any]]:
        """加载 QA 问题列表，支持从 JSON 文件读取。"""
        if qa_json_path is None:
            return self._default_qa_questions()

        qa_path = resolve_project_path(qa_json_path)
        if qa_path is None:
            return self._default_qa_questions()
        if not qa_path.exists():
            raise FileNotFoundError(f"QA JSON 文件不存在：{qa_path}")

        with qa_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "questions" in data:
            questions = data["questions"]
            self.scoring_instruction = data.get(
                "scoring_instruction",
                self.scoring_instruction,
            )
        elif isinstance(data, list):
            questions = data
        else:
            raise ValueError("QA JSON 格式错误，应为列表或包含 'questions' 字段的对象")

        # 确保每个问题都有 question 字段
        for q in questions:
            if "question" not in q:
                raise ValueError("每个 QA 问题必须包含 'question' 字段")

        return questions

    def _default_qa_questions(self) -> List[Dict[str, Any]]:
        """默认的 QA 问题列表。"""
        return [
            {"question": "人物外貌（五官、脸型）是否与参考图像一致？"},
            {"question": "服装、发型是否与参考图像一致且稳定？"},
            {"question": "面部是否清晰、稳定，没有变形或模糊？"},
            {"question": "肢体动作是否流畅、自然、不僵硬？"},
            {"question": "面部表情变化是否自然？"},
            {"question": "人物呼吸和身体起伏是否自然？"},
            {"question": "人物是否稳定处于画面中心位置？"},
            {"question": "人物画面占比是否合适且稳定？"},
            {"question": "视频画面是否流畅，具有电影质感？"},
            {"question": "视频是否存在模糊、重影或闪烁问题？"},
        ]

    def _frame_to_pil(self, frame: Any) -> "Image.Image":
        """将 OpenCV BGR 帧转换为 PIL Image。"""
        from PIL import Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    def _prepare_messages(
            self,
            frames: List[Any],
            question: str,
            person_images: Optional[List[Any]] = None,
            reference_frames: Optional[List[Any]] = None,
            prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        准备 Qwen 对话消息。

        生成视频和参考视频必须分组标注，否则模型容易把参考帧当成
        生成结果的一部分，导致比较方向不稳定。
        """
        content = []

        def append_sampled_frames(
                label: str,
                source_frames: Optional[List[Any]],
                limit: int,
        ) -> None:
            if not source_frames:
                return
            content.append({"type": "text", "text": label})
            sample_indices = np.linspace(
                0,
                len(source_frames) - 1,
                min(limit, len(source_frames)),
                dtype=np.int64
            ).tolist()
            for idx in sample_indices:
                pil_img = self._frame_to_pil(source_frames[idx])
                if pil_img.size[0] > 448 or pil_img.size[1] > 448:
                    pil_img.thumbnail((448, 448))
                content.append({
                    "type": "image",
                    "image": pil_img,
                })

        # 每段视频最多取 8 帧，保留时间顺序，同时兼顾准确率和输入规模。
        append_sampled_frames(
            "以下是生成视频的时间顺序采样帧：",
            frames,
            min(self.max_frames, 8),
        )
        append_sampled_frames(
            "以下是参考视频的时间顺序采样帧：",
            reference_frames,
            min(self.max_frames, 8),
        )

        if person_images:
            content.append({"type": "text", "text": "以下是人物参考图片："})
            for img in person_images[:2]:  # 最多使用 2 张参考图
                pil_img = self._frame_to_pil(img)
                if pil_img.size[0] > 448 or pil_img.size[1] > 448:
                    pil_img.thumbnail((448, 448))
                content.append({
                    "type": "image",
                    "image": pil_img,
                })

        prompt_text = (prompt or "").strip() or "未提供生成提示词。"
        content.append({
            "type": "text",
            "text": (
                "请严格区分生成视频、参考视频和人物参考图片。"
                "身份、服装和配饰问题优先与人物参考图片比较，忽略姿态、镜头、背景和表情差异；"
                "表情和动作问题优先比较生成视频与参考视频的时间变化；"
                "清晰度和连续性问题只评价生成视频本身。"
                "请结合全部时间顺序帧进行判断，不要只依据单帧。\n\n"
                f"目标生成提示词：{prompt_text}\n\n"
                f"问题：{question}\n\n"
                f"评分标准：{self.scoring_instruction}\n"
                "请只输出 JSON，例如 {\"score\": 0.5}，不要输出解释文字。"
            ),
        })

        return [{"role": "user", "content": content}]

    def _prepare_visual_content(
            self,
            frames: Sequence[Any],
            person_images: Optional[Sequence[Any]] = None,
            reference_frames: Optional[Sequence[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert shared visual inputs once for multi-question QA."""

        content: List[Dict[str, Any]] = []

        def append_sampled_frames(
                label: str,
                source_frames: Optional[Sequence[Any]],
                limit: int,
        ) -> None:
            if not source_frames:
                return
            content.append({"type": "text", "text": label})
            sample_indices = np.linspace(
                0,
                len(source_frames) - 1,
                min(limit, len(source_frames)),
                dtype=np.int64,
            ).tolist()
            for idx in sample_indices:
                pil_img = self._frame_to_pil(source_frames[int(idx)])
                if pil_img.size[0] > 448 or pil_img.size[1] > 448:
                    pil_img.thumbnail((448, 448))
                content.append({"type": "image", "image": pil_img})

        append_sampled_frames(
            "Generated video frames in chronological order:",
            frames,
            min(self.max_frames, 6),
        )
        append_sampled_frames(
            "Reference video frames in chronological order:",
            reference_frames,
            min(self.max_frames, 6),
        )

        if person_images:
            content.append({"type": "text", "text": "Person reference images:"})
            for image in list(person_images)[:2]:
                pil_img = self._frame_to_pil(image)
                if pil_img.size[0] > 448 or pil_img.size[1] > 448:
                    pil_img.thumbnail((448, 448))
                content.append({"type": "image", "image": pil_img})
        return content

    def _prepare_batch_messages(
            self,
            visual_content: Sequence[Dict[str, Any]],
            questions: Sequence[str],
            prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build one request that scores all QA questions."""

        prompt_text = (prompt or "").strip() or "No generation prompt was provided."
        question_text = "\n".join(
            f"{index + 1}. {question}"
            for index, question in enumerate(questions)
        )
        content = list(visual_content)
        content.append({
            "type": "text",
            "text": (
                "Evaluate every question using the full chronological video. "
                "Return only a JSON array with one object per question, in the "
                "same order, using scores 0, 0.5, or 1. "
                "Do not include explanations.\n"
                f"Generation prompt: {prompt_text}\n"
                f"Scoring rule: {self.scoring_instruction}\n"
                f"Questions:\n{question_text}\n"
                'Output example: [{"score": 0.5}, {"score": 1}]'
            ),
        })
        return [{"role": "user", "content": content}]

    def _generate_response(
            self,
            messages: List[Dict[str, Any]],
            max_new_tokens: int,
    ) -> str:
        """Run one serialized generation against the shared model."""

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with _QWEN_INFERENCE_LOCK:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _extract_batch_scores(
            self,
            response: str,
            expected_count: int,
    ) -> Optional[List[float]]:
        """Parse a strict JSON score list returned by the batch request."""

        import re

        candidates = re.findall(r"(\[[\s\S]*\]|\{[\s\S]*\})", response)
        for candidate in reversed(candidates):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if isinstance(payload, dict):
                values = (
                    payload.get("scores")
                    or payload.get("results")
                    or payload.get("questions")
                )
            else:
                values = payload
            if not isinstance(values, list) or len(values) != expected_count:
                continue

            scores: List[float] = []
            valid = True
            for value in values:
                raw_score = value.get("score") if isinstance(value, dict) else value
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    valid = False
                    break
                if score not in {0.0, 0.25, 0.5, 0.75, 1.0}:
                    valid = False
                    break
                scores.append(score)
            if valid:
                return scores
        return None

    def _extract_score(self, response: str) -> Optional[float]:
        """从模型响应中提取分数。"""
        import re
        # 小数必须放在整数前，否则 \b 会把 “0.5” 先匹配成 “0”。
        match = re.search(
            r'(?<![\d.])(0\.25|0\.5(?:0+)?|0\.75|0\.0+|1\.0+|0|1)(?![\d.])',
            response.strip(),
        )
        if match:
            val = float(match.group(1))
            if val == 0.0:
                return 0.0
            if val == 1.0:
                return 1.0
            return val
        return None

    def evaluate_single_question(
            self,
            frames: List[Any],
            question: str,
            person_images: Optional[List[Any]] = None,
            reference_frames: Optional[List[Any]] = None,
            prompt: Optional[str] = None,
    ) -> Tuple[Optional[float], str]:
        """
        评估单个问题，使用与 test.py 相同的方式调用模型。
        """
        messages = self._prepare_messages(
            frames,
            question,
            person_images,
            reference_frames,
            prompt,
        )

        try:
            response = self._generate_response(messages, max_new_tokens=64)
            score = self._extract_score(response)
            return score, response

        except Exception as exc:
            LOGGER.error("Qwen 推理失败：%s", exc)
            import traceback
            LOGGER.error(traceback.format_exc())
            return None, f"ERROR: {exc}"

    def evaluate_all_questions(
            self,
            frames: List[Any],
            person_images: Optional[List[Any]] = None,
            reference_frames: Optional[List[Any]] = None,
            prompt: Optional[str] = None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        评估所有问题，返回平均分数和详细结果。
        """
        results = []
        valid_scores: List[Tuple[float, float]] = []

        questions = [q.get("question", "") for q in self.qa_questions]
        batch_scores: Optional[List[float]] = None
        batch_response = ""
        # Keep questions independent: this is more reliable for Chinese
        # identity, clothing, expression, and temporal comparisons.
        use_batch_qa = False
        if use_batch_qa and questions:
            try:
                visual_content = self._prepare_visual_content(
                    frames,
                    person_images=person_images,
                    reference_frames=reference_frames,
                )
                batch_messages = self._prepare_batch_messages(
                    visual_content,
                    questions,
                    prompt=prompt,
                )
                batch_response = self._generate_response(
                    batch_messages,
                    max_new_tokens=96,
                )
                batch_scores = self._extract_batch_scores(
                    batch_response,
                    expected_count=len(questions),
                )
                if (
                        batch_scores is not None
                        and len(batch_scores) > 1
                        and len(set(batch_scores)) == 1
                ):
                    LOGGER.warning(
                        "Batch QA returned identical scores for every question; "
                        "falling back to independent question inference"
                    )
                    batch_scores = None
                if batch_scores is None:
                    LOGGER.warning(
                        "Batch QA output could not be parsed; falling back to per-question inference"
                    )
            except Exception as exc:
                LOGGER.warning(
                    "Batch QA inference failed; falling back to per-question inference: %s",
                    exc,
                )

        if batch_scores is not None:
            for q, score in zip(self.qa_questions, batch_scores):
                try:
                    question_weight = max(float(q.get("weight", 1.0) or 1.0), 0.0)
                except (TypeError, ValueError):
                    question_weight = 1.0
                results.append({
                    "id": q.get("id"),
                    "category": q.get("category"),
                    "question": q.get("question", ""),
                    "weight": question_weight,
                    "score": score,
                    "raw_response": batch_response,
                })
                if question_weight > 0:
                    valid_scores.append((score, question_weight))

            total_weight = sum(weight for _, weight in valid_scores)
            avg_score = (
                sum(score * weight for score, weight in valid_scores) / total_weight
                if total_weight > 0
                else 0.0
            )
            return avg_score, results

        for i, q in enumerate(self.qa_questions):
            question = q.get("question", "")
            try:
                question_weight = max(float(q.get("weight", 1.0) or 1.0), 0.0)
            except (TypeError, ValueError):
                question_weight = 1.0
            LOGGER.info("评估问题 %d/%d: %s", i + 1, len(self.qa_questions), question[:50])
            score, response = self.evaluate_single_question(
                frames,
                question,
                person_images,
                reference_frames,
                prompt,
            )

            result = {
                "id": q.get("id"),
                "category": q.get("category"),
                "question": question,
                "weight": question_weight,
                "score": score,
                "raw_response": response,
            }
            results.append(result)

            if score is not None and question_weight > 0:
                valid_scores.append((score, question_weight))

        # 按用户在 QA 中设置的权重计算平均分，避免问题数量改变后指标失真。
        total_weight = sum(weight for _, weight in valid_scores)
        avg_score = (
            sum(score * weight for score, weight in valid_scores) / total_weight
            if total_weight > 0
            else 0.0
        )

        return avg_score, results

    def save_qa_results(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """保存 QA 评估结果到 JSON 文件。"""
        output = {
            "questions": results,
            "scoring_instruction": self.scoring_instruction,
            "source": "Qwen3-VL-4B-Instruct 自动评估",
        }

        resolved_output_path = resolve_project_path(output_path)
        if resolved_output_path is None:
            raise ValueError("Qwen 输出路径不能为空")
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        LOGGER.info("QA 结果已保存到: %s", resolved_output_path)

    @staticmethod
    def load_qa_results(qwen_output_path: str) -> Tuple[float, List[Dict[str, Any]]]:
        """
        从已有的 Qwen 输出文件加载评估结果。

        Args:
            qwen_output_path: Qwen 评估结果 JSON 文件路径

        Returns:
            (平均分数, 问题列表)
        """
        qwen_path = resolve_project_path(qwen_output_path)
        if qwen_path is None:
            raise ValueError("Qwen 评估结果路径不能为空")
        if not qwen_path.exists():
            raise FileNotFoundError(f"Qwen 评估结果文件不存在：{qwen_path}")

        with qwen_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data.get("questions", [])
        if not questions:
            raise ValueError("Qwen 评估结果中没有 'questions' 字段或为空")

        # 提取所有有效分数
        scores = []
        for q in questions:
            score = q.get("score")
            if score is not None:
                # 确保分数在 0-1 范围内
                if isinstance(score, (int, float)):
                    if score > 1.0:
                        score = score / 100.0 if score <= 100 else score / 5.0
                    scores.append(float(score))

        avg_score = float(np.mean(scores)) if scores else 0.0
        LOGGER.info("从 %s 加载了 %d 个问题，平均分: %.4f", qwen_output_path, len(questions), avg_score)

        return avg_score, questions


def load_external_qa_score(path: Optional[str]) -> Optional[Tuple[float, Dict[str, Any]]]:
    """
    读取外部 Video-LLM/人工原子问题评分。

    支持：
    1. [{"question": "...", "score": 1}, ...]
    2. {"questions": [{"question": "...", "score": 0.5}, ...]}
    3. {"score": 0.8}
    """

    if not path:
        return None
    qa_path = resolve_project_path(path)
    if qa_path is None:
        return None
    if not qa_path.exists():
        raise FileNotFoundError(f"qa-json 不存在：{qa_path}")
    with qa_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "score" in payload:
        raw_score = float(payload["score"])
        score = raw_score / 5.0 if raw_score > 1.0 else raw_score
        return clamp(score), {"source": "external_qa_summary"}

    if isinstance(payload, dict):
        items = payload.get("questions", payload.get("items", []))
    else:
        items = payload
    if not isinstance(items, list):
        raise ValueError("qa-json 应为列表，或包含 questions/items 列表的对象")

    scores = []
    for item in items:
        if not isinstance(item, dict) or "score" not in item:
            continue
        raw_score = item["score"]
        if raw_score is None:
            continue
        raw_score = float(raw_score)
        scores.append(raw_score / 5.0 if raw_score > 1.0 else raw_score)
    if not scores:
        return None
    return clamp(float(np.mean(scores))), {
        "source": "external_qa_items",
        "question_count": len(scores),
    }


def face_centers_and_scales(
        observations: Sequence[FaceObservation],
        width: int,
        height: int,
) -> Tuple[List[Any], List[float]]:
    """提取归一化人脸中心和面积比例。"""

    centers: List[Any] = []
    scales: List[float] = []
    for observation in observations:
        if observation.bbox is None:
            continue
        x1, y1, x2, y2 = observation.bbox
        centers.append(
            np.asarray(
                [((x1 + x2) / 2) / max(width, 1), ((y1 + y2) / 2) / max(height, 1)],
                dtype=np.float32,
            )
        )
        scales.append(
            bbox_area(observation.bbox) / max(float(width * height), 1.0)
        )
    return centers, scales


def compute_identity_metric(
        gen_samples: VideoSamples,
        gen_faces: Sequence[FaceObservation],
        person_images: Optional[Sequence[Any]],
        person_image_paths: Optional[Sequence[str]],
        face_analyzer: FaceAnalyzer,
        face_sim_low: float,
        face_sim_high: float,
) -> MetricResult:
    """
    角色一致性 35%：
    - 有一张或多张人物图像：比较生成视频每帧与所有参考视图的 ArcFace
      相似度，并取当前帧最匹配的参考视图，适合正面/侧面/背面等三视图；
    - 无人物图像：比较每帧与视频内稳健中心模板的相似度；
    - 没有 ArcFace：退化为人脸出现率和人脸框稳定性。
    """

    valid_embeddings = [
        observation.embedding
        for observation in gen_faces
        if observation.embedding is not None
    ]

    reference_embeddings: List[Any] = []
    reference_source = "video_robust_template"
    valid_reference_indices: List[int] = []
    if person_images:
        for index, image in enumerate(person_images):
            reference_observation = face_analyzer.analyze(image)
            if reference_observation.embedding is not None:
                reference_embeddings.append(reference_observation.embedding)
                valid_reference_indices.append(index)
        if reference_embeddings:
            reference_source = "person_images_multi_view"

    if not reference_embeddings and valid_embeddings:
        vectors = np.asarray(valid_embeddings, dtype=np.float32)
        reference_embedding = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(reference_embedding))
        if norm > 1e-8:
            reference_embedding = reference_embedding / norm
            reference_embeddings = [reference_embedding]

    if reference_embeddings and valid_embeddings:
        similarities: List[float] = []
        best_view_indices: List[int] = []
        for observation in gen_faces:
            view_similarities = [
                cosine_similarity(observation.embedding, reference_embedding)
                for reference_embedding in reference_embeddings
            ]
            valid_view_similarities = [
                (index, value)
                for index, value in enumerate(view_similarities)
                if value is not None
            ]
            if valid_view_similarities:
                best_index, best_similarity = max(
                    valid_view_similarities,
                    key=lambda item: item[1],
                )
                # reference_embeddings 只包含检测成功的参考图，因此把索引
                # 映射回原始参考图序号，便于在报告中定位具体视角。
                if reference_source == "person_images_multi_view":
                    best_view_indices.append(valid_reference_indices[best_index])
                else:
                    best_view_indices.append(best_index)
                similarities.append(float(best_similarity))

        if similarities:
            scores = [
                similarity_to_face_score(
                    value,
                    low=face_sim_low,
                    high=face_sim_high,
                )
                for value in similarities
            ]
            tail_count = max(1, int(math.ceil(len(scores) * 0.10)))
            tail_scores = scores[-tail_count:]
            mean_score = float(np.mean(scores))
            tail_score = float(np.mean(tail_scores))
            similarity_std = float(np.std(similarities))
            stability_score = math.exp(-similarity_std / 0.12)
            coverage = len(similarities) / max(len(gen_faces), 1)
            final_score = clamp(
                0.55 * mean_score
                + 0.30 * tail_score
                + 0.15 * stability_score
            )
            final_score = clamp(0.90 * final_score + 0.10 * coverage)
            return MetricResult(
                name="角色一致性",
                score=final_score,
                weight=WEIGHTS["identity"],
                status="ok",
                details={
                    "method": "ArcFace cosine similarity",
                    "reference_source": reference_source,
                    "reference_image_count": len(person_images or []),
                    "valid_reference_images": len(valid_reference_indices),
                    "reference_image_paths": list(person_image_paths or []),
                    "multi_view_aggregation": (
                        "每个生成帧与所有参考视图比较，取最高相似度"
                        if reference_source == "person_images_multi_view"
                        else "视频内有效人脸 embedding 的归一化均值"
                    ),
                    "detector": face_analyzer.detector_name,
                    "valid_face_frames": len(similarities),
                    "total_sampled_frames": len(gen_faces),
                    "face_coverage": coverage,
                    "mean_similarity": safe_mean(similarities),
                    "tail_10_percent_similarity": safe_mean(tail_scores),
                    "similarity_std": similarity_std,
                    "best_reference_view_histogram": {
                        str(index): best_view_indices.count(index)
                        for index in sorted(set(best_view_indices))
                    },
                    "face_sim_low": face_sim_low,
                    "face_sim_high": face_sim_high,
                },
            )

    centers, scales = face_centers_and_scales(
        gen_faces,
        gen_samples.width,
        gen_samples.height,
    )
    detected_count = len(centers)
    coverage = detected_count / max(len(gen_faces), 1)
    if detected_count >= 2:
        center_array = np.asarray(centers, dtype=np.float32)
        scale_array = np.asarray(scales, dtype=np.float32)
        center_jitter = float(np.mean(np.std(center_array, axis=0)))
        scale_jitter = float(np.std(scale_array) / max(np.mean(scale_array), 1e-6))
        stability_score = clamp(
            0.5 * math.exp(-center_jitter / 0.08)
            + 0.5 * math.exp(-scale_jitter / 0.35)
        )
        proxy_score = clamp(0.70 * coverage + 0.30 * stability_score)
        return MetricResult(
            name="角色一致性",
            score=proxy_score,
            weight=WEIGHTS["identity"],
            status="proxy",
            details={
                "method": "人脸出现率 + 人脸框稳定性",
                "warning": "未启用 InsightFace/ArcFace，不能视为真正身份相似度",
                "detector": face_analyzer.detector_name,
                "reference_image_count": len(person_images or []),
                "reference_image_paths": list(person_image_paths or []),
                "face_coverage": coverage,
                "center_jitter": center_jitter,
                "scale_jitter": scale_jitter,
            },
        )

    return MetricResult(
        name="角色一致性",
        score=None,
        weight=WEIGHTS["identity"],
        status="unavailable",
        details={
            "method": "ArcFace",
            "warning": "没有检测到可用人脸，无法计算角色一致性",
            "detector": face_analyzer.detector_name,
            "reference_image_count": len(person_images or []),
            "reference_image_paths": list(person_image_paths or []),
        },
    )


def frame_sharpness(frame: Any) -> float:
    """在统一尺寸上用 Laplacian 方差估计清晰度。"""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def high_frequency_ratio(frame: Any) -> float:
    """
    估计高频细节占比。

    这里不是严格的"皮肤高频保留率"，但在没有 GT 时可以作为工程代理。
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    high_frequency_energy = float(np.mean(np.abs(laplacian)))
    total_energy = float(np.mean(np.abs(gray - np.mean(gray)))) + 1e-6
    return high_frequency_energy / total_energy


def crop_face_or_frame(
        frame: Any,
        observation: FaceObservation,
) -> Any:
    """优先裁剪人脸区域，没有人脸时返回整帧。"""

    if observation.bbox is None:
        return frame
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = expand_bbox(observation.bbox, width, height)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else frame


def resize_for_metric(image: Any, size: Tuple[int, int] = (224, 224)) -> Any:
    """统一感知指标输入大小。"""

    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def relative_metric_match(value_a: float, value_b: float) -> float:
    """比较两个质量统计量的相对量级，不要求两帧像素级对齐。"""

    value_a = max(float(value_a), 1e-6)
    value_b = max(float(value_b), 1e-6)
    log_ratio = abs(math.log(value_a / value_b))
    return clamp(1.0 - log_ratio / math.log(4.0))


def simple_ssim_score(image_a: Any, image_b: Any) -> float:
    """不依赖额外包的简化 SSIM，用作 LPIPS 不可用时的参考代理。"""

    a = cv2.cvtColor(resize_for_metric(image_a), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(resize_for_metric(image_b), cv2.COLOR_BGR2GRAY)
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (
            (mu_a * mu_a + mu_b * mu_b + c1)
            * (sigma_a + sigma_b + c2)
    )
    return clamp(float(np.mean(numerator / (denominator + 1e-8))))


class LPIPSScorer:
    """可选 LPIPS 封装，模型只在用户显式指定 --use-lpips 时加载。"""

    def __init__(self, device: str = "auto"):
        try:
            import lpips
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "启用 LPIPS 需要安装 torch、torchvision 和 lpips"
            ) from exc

        if device in {"auto", "cuda", "gpu"}:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            torch_device = "cpu"
        self.torch = torch
        self.device = torch.device(torch_device)
        self.model = lpips.LPIPS(net="alex").to(self.device)
        self.model.eval()

    def score(self, image_a: Any, image_b: Any) -> float:
        """LPIPS 距离转为相似度分数。"""

        a = cv2.cvtColor(resize_for_metric(image_a), cv2.COLOR_BGR2RGB)
        b = cv2.cvtColor(resize_for_metric(image_b), cv2.COLOR_BGR2RGB)
        tensor_a = self.torch.from_numpy(a).permute(2, 0, 1).float() / 127.5 - 1.0
        tensor_b = self.torch.from_numpy(b).permute(2, 0, 1).float() / 127.5 - 1.0
        with self.torch.no_grad():
            distance = float(
                self.model(
                    tensor_a.unsqueeze(0).to(self.device),
                    tensor_b.unsqueeze(0).to(self.device),
                )
                    .item()
            )
        # LPIPS 距离越小越好，工程上用指数映射到 0~1。
        return clamp(math.exp(-3.0 * max(distance, 0.0)))


def compute_detail_metric(
        gen_samples: VideoSamples,
        gen_faces: Sequence[FaceObservation],
        ref_samples: Optional[VideoSamples],
        ref_faces: Optional[Sequence[FaceObservation]],
        use_lpips: bool,
        device: str,
) -> MetricResult:
    """
    质感和细节 15%。

    这是视频质量指标，不应把两段不同镜头的像素差异误判成细节损失。
    因此参考视频只用于比较清晰度/高频统计分布；LPIPS 仅作为可选辅助证据。
    """

    gen_crops = [
        crop_face_or_frame(frame, observation)
        for frame, observation in zip(gen_samples.frames, gen_faces)
    ]
    if not gen_crops:
        return MetricResult(
            name="质感和细节",
            score=None,
            weight=WEIGHTS["detail"],
            status="unavailable",
            details={"warning": "没有可用生成视频帧"},
        )

    gen_sharpness = [frame_sharpness(crop) for crop in gen_crops]
    gen_hf = [high_frequency_ratio(crop) for crop in gen_crops]

    sharpness_score = clamp(
        1.0 - math.exp(-math.log1p(float(np.median(gen_sharpness))) / 5.5)
    )
    hf_score = clamp(
        1.0 - math.exp(-float(np.median(gen_hf)) / 0.08)
    )
    generated_quality_score = clamp(
        0.60 * sharpness_score + 0.40 * hf_score
    )

    if ref_samples is not None:
        if ref_faces is None:
            ref_faces = [FaceObservation() for _ in ref_samples.frames]
        ref_crops = [
            crop_face_or_frame(frame, observation)
            for frame, observation in zip(ref_samples.frames, ref_faces)
        ]
        pair_count = min(len(gen_crops), len(ref_crops))
        if pair_count:
            ref_sharpness = [
                frame_sharpness(crop)
                for crop in ref_crops
            ]
            ref_hf = [
                high_frequency_ratio(crop)
                for crop in ref_crops
            ]
            sharpness_match = relative_metric_match(
                float(np.median(gen_sharpness)),
                float(np.median(ref_sharpness)),
            )
            hf_match = relative_metric_match(
                float(np.median(gen_hf)),
                float(np.median(ref_hf)),
            )
            reference_quality_score = clamp(
                0.60 * sharpness_match + 0.40 * hf_match
            )

            lpips_scorer = None
            lpips_scores: List[float] = []
            if use_lpips:
                try:
                    lpips_scorer = LPIPSScorer(device=device)
                except Exception as exc:
                    LOGGER.warning("LPIPS 初始化失败，将忽略 LPIPS 辅助项：%s", exc)

            for pair_index in range(pair_count):
                gen_index = int(
                    round(
                        pair_index
                        * max(len(gen_crops) - 1, 0)
                        / max(pair_count - 1, 1)
                    )
                )
                ref_index = int(
                    round(
                        pair_index
                        * max(len(ref_crops) - 1, 0)
                        / max(pair_count - 1, 1)
                    )
                )
                if lpips_scorer is not None:
                    lpips_scores.append(
                        lpips_scorer.score(
                            gen_crops[gen_index],
                            ref_crops[ref_index],
                        )
                    )

            lpips_score = (
                float(np.mean(lpips_scores))
                if lpips_scores
                else None
            )
            if lpips_score is not None:
                reference_quality_score = clamp(
                    0.80 * reference_quality_score + 0.20 * lpips_score
                )

            final_score = clamp(
                0.70 * generated_quality_score
                + 0.30 * reference_quality_score
            )
            return MetricResult(
                name="质感和细节",
                score=final_score,
                weight=WEIGHTS["detail"],
                status="ok" if lpips_score is not None else "proxy",
                details={
                    "method": (
                        "生成质量 + 参考视频清晰度/高频分布"
                        + (" + LPIPS 辅助" if lpips_score is not None else "")
                    ),
                    "paired_frames": pair_count,
                    "generated_quality_score": generated_quality_score,
                    "reference_quality_score": reference_quality_score,
                    "sharpness_match": sharpness_match,
                    "high_frequency_match": hf_match,
                    "lpips_similarity": lpips_score,
                    "generated_median_sharpness": float(np.median(gen_sharpness)),
                    "generated_median_high_frequency_ratio": float(np.median(gen_hf)),
                    "reference_median_sharpness": float(np.median(ref_sharpness)),
                    "reference_median_high_frequency_ratio": float(np.median(ref_hf)),
                    "warning": "参考视频不再进行未对齐的像素级 SSIM，避免把镜头差异误判为细节损失。",
                },
            )

    return MetricResult(
        name="质感和细节",
        score=generated_quality_score,
        weight=WEIGHTS["detail"],
        status="proxy",
        details={
            "method": "Laplacian 清晰度 + 高频能量代理",
            "generated_median_sharpness": float(np.median(gen_sharpness)),
            "generated_median_high_frequency_ratio": float(np.median(gen_hf)),
            "sharpness_proxy_score": sharpness_score,
            "high_frequency_proxy_score": hf_score,
            "warning": "没有参考视频时，本项是无参考质量代理，不代表真实感的全部维度。",
        },
    )


def euclidean_distance(a: Any, b: Any) -> float:
    """计算两个点或向量的欧氏距离。"""

    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def expression_features(landmarks: Optional[Any]) -> Optional[Any]:
    """
    从 MediaPipe 关键点提取与表情相关的归一化特征。

    这些索引对应眼睛、鼻子和嘴部区域。特征使用眼间距归一化，以减少
    人物远近变化和头部平移的影响。
    """

    if landmarks is None or len(landmarks) < 292:
        return None
    points = np.asarray(landmarks, dtype=np.float32)
    inter_eye = euclidean_distance(points[33, :2], points[263, :2])
    if inter_eye < 1e-6:
        return None

    left_eye_open = euclidean_distance(points[159, :2], points[145, :2])
    right_eye_open = euclidean_distance(points[386, :2], points[374, :2])
    mouth_open = euclidean_distance(points[13, :2], points[14, :2])
    mouth_width = euclidean_distance(points[61, :2], points[291, :2])
    nose_to_mouth = euclidean_distance(points[1, :2], points[13, :2])
    mouth_corner_height = float(
        abs(points[61, 1] + points[291, 1] - 2 * points[13, 1])
        / inter_eye
    )
    return np.asarray(
        [
            left_eye_open / inter_eye,
            right_eye_open / inter_eye,
            mouth_open / inter_eye,
            mouth_width / inter_eye,
            nose_to_mouth / inter_eye,
            mouth_corner_height,
        ],
        dtype=np.float32,
    )


def resample_sequence(sequence: Any, length: int) -> Any:
    """沿时间维线性重采样，允许生成视频和参考视频帧数不同。"""

    sequence = np.asarray(sequence, dtype=np.float32)
    if len(sequence) == length:
        return sequence
    if len(sequence) == 0:
        return np.empty((0,) + sequence.shape[1:], dtype=np.float32)
    old_x = np.linspace(0.0, 1.0, len(sequence))
    new_x = np.linspace(0.0, 1.0, length)
    if sequence.ndim == 1:
        return np.interp(new_x, old_x, sequence).astype(np.float32)
    output = np.empty((length,) + sequence.shape[1:], dtype=np.float32)
    for index in np.ndindex(sequence.shape[1:]):
        values = sequence[(slice(None),) + index]
        output[(slice(None),) + index] = np.interp(new_x, old_x, values)
    return output


def estimate_expression_from_landmarks(
        landmark_list: Sequence[Optional[Any]],
) -> Optional[Tuple[float, Dict[str, Any]]]:
    """仅根据生成视频关键点估计表情变化的自然度和稳定性。"""

    features = [
        feature
        for feature in (expression_features(points) for points in landmark_list)
        if feature is not None
    ]
    if len(features) < 3:
        return None

    array = np.asarray(features, dtype=np.float32)
    frame_change = np.abs(np.diff(array, axis=0))
    change_mean = float(np.mean(frame_change))
    change_std = float(np.std(frame_change))
    acceleration = (
        np.abs(np.diff(array, n=2, axis=0))
        if len(array) >= 3
        else np.empty((0, array.shape[1]), dtype=np.float32)
    )
    acceleration_mean = (
        float(np.mean(acceleration))
        if acceleration.size
        else 0.0
    )
    left_right_change = float(
        np.mean(np.abs(frame_change[:, 0] - frame_change[:, 1]))
    )
    expression_activity = clamp(change_mean / 0.025)
    expression_smoothness = clamp(math.exp(-acceleration_mean / 0.012))
    expression_balance = clamp(
        math.exp(-left_right_change / max(change_mean, 0.005))
    )
    final_score = clamp(
        0.45 * expression_activity
        + 0.35 * expression_smoothness
        + 0.20 * expression_balance
    )
    return final_score, {
        "method": "MediaPipe 关键点表情变化算法",
        "valid_frames": len(features),
        "activity_score": expression_activity,
        "smoothness_score": expression_smoothness,
        "symmetry_score": expression_balance,
        "frame_change_mean": change_mean,
        "acceleration_mean": acceleration_mean,
        "warning": "无参考视频时，该分数衡量表情变化的自然度和稳定性，不代表目标表情语义的绝对正确率",
    }


def estimate_expression_from_face_motion(
        frames: Sequence[Any],
        faces: Sequence[FaceObservation],
) -> Optional[Tuple[float, Dict[str, Any]]]:
    """没有 MediaPipe 时，用人脸区域帧间变化估计表情连续性。"""

    crops = [
        crop_face_or_frame(frame, observation)
        for frame, observation in zip(frames, faces)
        if observation.bbox is not None
    ]
    if len(crops) < 3:
        return None

    normalized = [
        cv2.cvtColor(
            cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        ).astype(np.float32)
        / 255.0
        for crop in crops
    ]
    frame_changes = np.asarray(
        [
            float(np.mean(np.abs(current - previous)))
            for previous, current in zip(normalized[:-1], normalized[1:])
        ],
        dtype=np.float32,
    )
    if len(frame_changes) < 2:
        return None

    change_mean = float(np.mean(frame_changes))
    change_std = float(np.std(frame_changes))
    activity_score = clamp(change_mean / 0.08)
    continuity_score = clamp(
        math.exp(-change_std / max(change_mean, 0.01))
    )
    final_score = clamp(
        0.65 * activity_score
        + 0.35 * continuity_score
    )
    return final_score, {
        "method": "OpenCV 人脸区域运动算法",
        "valid_frames": len(crops),
        "activity_score": activity_score,
        "continuity_score": continuity_score,
        "frame_change_mean": change_mean,
        "frame_change_std": change_std,
        "warning": "无面部关键点和参考视频时，该分数衡量人脸区域变化的连续性，不代表表情语义绝对正确率",
    }


EXPRESSION_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}
EXPRESSION_VIEW_NAMES = {"Front", "Left", "Right"}
EXPRESSION_TEXTURE_REGIONS = (
    ("forehead", (34, 16, 126, 52)),
    ("brow_eye", (24, 42, 136, 92)),
    ("cheek_left", (12, 78, 72, 128)),
    ("cheek_right", (88, 78, 148, 128)),
    ("mouth", (38, 96, 122, 148)),
    ("jaw", (24, 116, 136, 158)),
)


def parse_expression_reference_name(path: Path) -> Tuple[str, str]:
    """Parse ``<action>_<view>`` names used by the Expression assets."""

    parts = path.stem.split("_")
    if parts and parts[-1] in EXPRESSION_VIEW_NAMES:
        return "_".join(parts[:-1]) or path.stem, parts[-1]
    return path.stem, "Unknown"


def expression_geometry_features(landmarks: Optional[Any]) -> Optional[Any]:
    """Return normalized muscle/action geometry from a MediaPipe face mesh."""

    if landmarks is None or len(landmarks) < 292:
        return None
    points = np.asarray(landmarks, dtype=np.float32)
    inter_eye = euclidean_distance(points[33, :2], points[263, :2])
    if inter_eye < 1e-6:
        return None
    base = expression_features(points)
    if base is None:
        return None

    extra = np.asarray(
        [
            euclidean_distance(points[70, :2], points[159, :2]) / inter_eye,
            euclidean_distance(points[300, :2], points[386, :2]) / inter_eye,
            euclidean_distance(points[234, :2], points[454, :2]) / inter_eye,
            euclidean_distance(points[10, :2], points[152, :2]) / inter_eye,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, extra]).astype(np.float32)


def expression_gaze_features(landmarks: Optional[Any]) -> Optional[Any]:
    """Estimate normalized left/right iris position for eye-gaze scoring."""

    if landmarks is None or len(landmarks) < 478:
        return None
    points = np.asarray(landmarks, dtype=np.float32)
    values: List[float] = []
    for iris_indices, corner_indices, lid_indices in (
        ((468, 469, 470, 471, 472), (33, 133), (159, 145)),
        ((473, 474, 475, 476, 477), (362, 263), (386, 374)),
    ):
        iris = np.mean(points[list(iris_indices), :2], axis=0)
        corners = points[list(corner_indices), :2]
        lids = points[list(lid_indices), :2]
        eye_width = max(float(np.linalg.norm(corners[1] - corners[0])), 1e-6)
        eye_height = max(float(np.linalg.norm(lids[1] - lids[0])), 1e-6)
        x_min = float(np.min(corners[:, 0]))
        x_ratio = (float(iris[0]) - x_min) / eye_width
        eye_mid_y = float(np.mean(lids[:, 1]))
        y_ratio = (float(iris[1]) - eye_mid_y) / eye_height
        values.extend([x_ratio, y_ratio])
    return np.asarray(values, dtype=np.float32)


def canonical_face_image(
        frame: Any,
        observation: FaceObservation,
        landmarks: Optional[Any],
        size: Tuple[int, int] = (160, 160),
) -> Any:
    """Align a face by both eyes and the nose, preserving expression regions."""

    if landmarks is not None and len(landmarks) >= 292:
        points = np.asarray(landmarks, dtype=np.float32)
        height, width = frame.shape[:2]
        scale = np.asarray([width, height], dtype=np.float32)
        left_eye = np.mean(points[[33, 133, 159, 145], :2], axis=0) * scale
        right_eye = np.mean(points[[263, 362, 386, 374], :2], axis=0) * scale
        nose = points[1, :2] * scale
        if float(np.linalg.norm(right_eye - left_eye)) > 8.0:
            source = np.asarray([left_eye, right_eye, nose], dtype=np.float32)
            target = np.asarray(
                [[48.0, 58.0], [112.0, 58.0], [80.0, 84.0]],
                dtype=np.float32,
            )
            transform = cv2.getAffineTransform(source, target)
            return cv2.warpAffine(
                frame,
                transform,
                size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

    crop = crop_face_or_frame(frame, observation)
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def face_texture_features(image: Any) -> Tuple[Any, Any, Any]:
    """Extract illumination-normalized structure and wrinkle-region energy."""

    image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_centered = (gray - float(np.mean(gray))) / (
        float(np.std(gray)) + 1e-6
    )
    blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
    high_pass = gray - blurred
    high_pass = high_pass / (float(np.std(gray)) + 1e-6)

    structure = cv2.resize(
        gray_centered,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    ).reshape(-1).astype(np.float32)
    detail = cv2.resize(
        high_pass,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    ).reshape(-1).astype(np.float32)
    region_energy = np.asarray(
        [
            float(np.mean(np.abs(high_pass[y1:y2, x1:x2])))
            for _, (x1, y1, x2, y2) in EXPRESSION_TEXTURE_REGIONS
        ],
        dtype=np.float32,
    )
    return structure, detail, region_energy


def coarse_expression_geometry_features(image: Any) -> Any:
    """OpenCV-only fallback for coarse eye, cheek, mouth and jaw movement."""

    image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    boxes = (
        (28, 48, 76, 88),
        (84, 48, 132, 88),
        (36, 88, 76, 132),
        (84, 88, 124, 132),
        (42, 100, 118, 148),
        (24, 112, 136, 158),
    )
    values: List[float] = []
    for x1, y1, x2, y2 in boxes:
        region = gray[y1:y2, x1:x2]
        if region.size == 0:
            values.extend([0.0, 0.0, 0.0, 0.0])
            continue
        laplacian = cv2.Laplacian(region, cv2.CV_32F)
        horizontal_profile = np.mean(
            np.abs(np.diff(region, axis=1)),
            axis=1,
        )
        values.extend(
            [
                float(np.mean(region)),
                float(np.std(region)),
                float(np.mean(np.abs(laplacian))),
                float(np.mean(horizontal_profile)),
            ]
        )
    return np.asarray(values, dtype=np.float32)


def coarse_gaze_features(image: Any) -> Any:
    """Estimate eye direction from the darkest stable pixels in each eye."""

    image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    boxes = (
        (36, 56, 74, 82),
        (86, 56, 124, 82),
    )
    values: List[float] = []
    for x1, y1, x2, y2 in boxes:
        region = gray[y1:y2, x1:x2]
        if region.size == 0:
            values.extend([0.5, 0.0])
            continue
        darkness = np.max(region) - region
        threshold = float(np.percentile(darkness, 70.0))
        darkness = np.maximum(darkness - threshold, 0.0)
        total = float(np.sum(darkness))
        if total < 1e-6:
            values.extend([0.5, 0.0])
            continue
        yy, xx = np.indices(region.shape, dtype=np.float32)
        values.extend(
            [
                float(np.sum(xx * darkness) / total / max(region.shape[1] - 1, 1)),
                float(np.sum(yy * darkness) / total / max(region.shape[0] - 1, 1)),
            ]
        )
    return np.asarray(values, dtype=np.float32)


def expression_reference_signature(
        paths: Sequence[Path],
) -> Tuple[Tuple[str, int, int], ...]:
    return tuple(
        (
            str(path),
            int(path.stat().st_size),
            int(path.stat().st_mtime_ns),
        )
        for path in paths
    )


def load_expression_reference_descriptors(
        expression_dir: Optional[Path],
        face_analyzer: FaceAnalyzer,
        landmark_analyzer: LandmarkAnalyzer,
) -> List[ExpressionReference]:
    """Load and cache the internal action/view face prototypes."""

    if expression_dir is None:
        return []
    expression_dir = expression_dir.resolve()
    if not expression_dir.exists() or not expression_dir.is_dir():
        LOGGER.warning("Expression 参考目录不存在：%s", expression_dir)
        return []

    paths = sorted(
        (
            path for path in expression_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in EXPRESSION_IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )
    signature = expression_reference_signature(paths)
    cache_key = str(expression_dir)
    with _EXPRESSION_REFERENCE_CACHE_LOCK:
        cached = _EXPRESSION_REFERENCE_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]

    references: List[ExpressionReference] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            LOGGER.warning("无法读取 Expression 参考图：%s", path)
            continue
        action, view = parse_expression_reference_name(path)
        face = face_analyzer.analyze(image)
        landmarks = landmark_analyzer.extract(image)
        aligned = canonical_face_image(image, face, landmarks)
        structure, detail, region_energy = face_texture_features(aligned)
        geometry = expression_geometry_features(landmarks)
        if geometry is None:
            geometry = coarse_expression_geometry_features(aligned)
        gaze = expression_gaze_features(landmarks)
        if gaze is None:
            gaze = coarse_gaze_features(aligned)
        references.append(
            ExpressionReference(
                action=action,
                view=view,
                path=str(path),
                face=face,
                landmarks=landmarks,
                geometry=geometry,
                gaze=gaze,
                structure=structure,
                detail=detail,
                region_energy=region_energy,
            )
        )

    with _EXPRESSION_REFERENCE_CACHE_LOCK:
        _EXPRESSION_REFERENCE_CACHE[cache_key] = (signature, references)
    LOGGER.info(
        "Expression 参考表情已加载：%d/%d 张，目录=%s",
        len(references),
        len(paths),
        expression_dir,
    )
    return references


def vector_similarity(a: Optional[Any], b: Optional[Any]) -> Optional[float]:
    similarity = cosine_similarity(a, b)
    if similarity is None:
        return None
    return clamp((float(similarity) + 1.0) / 2.0)


def expression_feature_similarity(
        value_a: Optional[Any],
        value_b: Optional[Any],
        scale: Optional[Any],
) -> Optional[float]:
    if value_a is None or value_b is None:
        return None
    a = np.asarray(value_a, dtype=np.float32).reshape(-1)
    b = np.asarray(value_b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        return None
    if scale is None:
        scale = np.full_like(a, 0.08, dtype=np.float32)
    scale_array = np.maximum(np.asarray(scale, dtype=np.float32), 1e-3)
    error = float(np.mean(np.abs(a - b) / scale_array))
    return clamp(math.exp(-error))


def estimate_expression_view(landmarks: Optional[Any]) -> str:
    if landmarks is None or len(landmarks) < 292:
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


def expression_action_group(action: str) -> str:
    lowered = action.lower()
    if "look" in lowered or "eye" in lowered:
        return "eye_gaze"
    if "brow" in lowered or "sneer" in lowered:
        return "brow"
    if "mouth" in lowered or "lip" in lowered or "jaw" in lowered:
        return "mouth_jaw"
    if "cheek" in lowered:
        return "cheek"
    if "head" in lowered:
        return "head_pose"
    if "neutral" in lowered:
        return "neutral"
    return "other"


EXPRESSION_ACTION_METADATA: Dict[str, Dict[str, str]] = {
    "Brow_Down": {
        "label": "压眉",
        "description": "双眉向下、向内收紧",
        "focus": "眉间、额头和上眼睑",
    },
    "Brow_Raise": {
        "label": "抬眉",
        "description": "双眉上提",
        "focus": "额头纹、眉峰和上眼睑",
    },
    "Cheek_Blow": {
        "label": "鼓腮",
        "description": "双颊鼓起",
        "focus": "面颊体积、嘴角和脸颊纹理",
    },
    "Eye_CheekRaise_L": {
        "label": "左眼周/脸颊抬升",
        "description": "左侧眼周与脸颊向上收紧",
        "focus": "左眼下方、左脸颊和嘴角",
    },
    "Eye_CheekRaise_R": {
        "label": "右眼周/脸颊抬升",
        "description": "右侧眼周与脸颊向上收紧",
        "focus": "右眼下方、右脸颊和嘴角",
    },
    "Eye_FaceScrunch": {
        "label": "眯眼皱鼻",
        "description": "眼周收紧并带动鼻梁和面部挤压",
        "focus": "眼角、鼻梁和眉间",
    },
    "Eye_LookDown": {
        "label": "眼神向下",
        "description": "视线下移",
        "focus": "瞳孔位置、上/下眼睑和眼周",
    },
    "Eye_Pouch_Mouth_Smile": {
        "label": "卧蚕带动的微笑",
        "description": "眼下肌肉与嘴角共同形成微笑",
        "focus": "眼下卧蚕、脸颊、嘴角和法令纹",
    },
    "Eye_Squint": {
        "label": "眯眼",
        "description": "上下眼睑收紧",
        "focus": "眼睑、眼角和眼周细纹",
    },
    "EyeLookLeft": {
        "label": "眼神向左",
        "description": "视线向画面左侧移动",
        "focus": "瞳孔位置、眼白比例和眼睑",
    },
    "EyeLookRight": {
        "label": "眼神向右",
        "description": "视线向画面右侧移动",
        "focus": "瞳孔位置、眼白比例和眼睑",
    },
    "EyeLookUp": {
        "label": "眼神向上",
        "description": "视线上移",
        "focus": "瞳孔位置、上眼睑和额头",
    },
    "HeadDown": {
        "label": "低头",
        "description": "头部向下转动",
        "focus": "眼神、下颌轮廓和脸部透视",
    },
    "HeadLeft": {
        "label": "向左转头",
        "description": "头部向画面左侧转动",
        "focus": "左右眼比例、鼻梁和脸颊轮廓",
    },
    "HeadRight": {
        "label": "向右转头",
        "description": "头部向画面右侧转动",
        "focus": "左右眼比例、鼻梁和脸颊轮廓",
    },
    "HeadUp": {
        "label": "抬头",
        "description": "头部向上转动",
        "focus": "眼神、颈部衔接和下颌轮廓",
    },
    "JawBackward": {
        "label": "下颌后收",
        "description": "下巴向后收回",
        "focus": "下巴、嘴周和下颌线",
    },
    "JawForward": {
        "label": "下颌前伸",
        "description": "下巴向前推出",
        "focus": "下巴、嘴周和下颌线",
    },
    "JawLeft": {
        "label": "下颌左移",
        "description": "下巴向画面左侧移动",
        "focus": "嘴角、下巴和左右脸颊",
    },
    "JawRight": {
        "label": "下颌右移",
        "description": "下巴向画面右侧移动",
        "focus": "嘴角、下巴和左右脸颊",
    },
    "Jaw_ChinRaise": {
        "label": "抬下巴",
        "description": "下巴上抬并拉紧下颌",
        "focus": "下巴、颏部和下颌线",
    },
    "Jaw_OpenWrinkle": {
        "label": "张口带动下颌纹理",
        "description": "张口时下颌和嘴周纹理变化",
        "focus": "嘴唇、下巴和下颌皮肤纹理",
    },
    "Lip_UpperFunnel": {
        "label": "上唇收拢",
        "description": "上唇向前收成漏斗形",
        "focus": "上唇、人中和鼻下区域",
    },
    "Lips_Bite": {
        "label": "咬唇",
        "description": "嘴唇内收或被牙齿轻咬",
        "focus": "上下唇、嘴角和下巴",
    },
    "Mouth_CornerDepress": {
        "label": "嘴角下压",
        "description": "两侧嘴角向下牵拉",
        "focus": "嘴角、法令纹和下巴",
    },
    "Mouth_Dimple": {
        "label": "酒窝/嘴角收紧",
        "description": "嘴角内收形成局部凹陷",
        "focus": "嘴角、脸颊和酒窝区域",
    },
    "Mouth_Kiss": {
        "label": "嘟嘴亲吻",
        "description": "双唇前突收拢",
        "focus": "上下唇、嘴周和下巴",
    },
    "Mouth_Left": {
        "label": "嘴部左偏",
        "description": "嘴角向画面左侧偏移",
        "focus": "左右嘴角、脸颊和下巴",
    },
    "Mouth_LipRaise": {
        "label": "上唇抬起",
        "description": "上唇上提并露出鼻下区域变化",
        "focus": "上唇、鼻翼和法令纹",
    },
    "Mouth_LipsPress": {
        "label": "闭唇用力",
        "description": "双唇闭合并向内收紧",
        "focus": "唇线、嘴角和下巴",
    },
    "Mouth_LipsPress2": {
        "label": "加强闭唇",
        "description": "更明显的双唇收紧",
        "focus": "唇线、嘴角和下巴",
    },
    "Mouth_Press": {
        "label": "嘴部按压",
        "description": "嘴唇和嘴周肌肉向内压紧",
        "focus": "嘴唇、嘴角和鼻唇沟",
    },
    "Mouth_PurseWrinkle": {
        "label": "嘟嘴皱纹",
        "description": "嘴唇收拢时形成放射状嘴周纹理",
        "focus": "嘴周纹理、唇线和下巴",
    },
    "Mouth_Right": {
        "label": "嘴部右偏",
        "description": "嘴角向画面右侧偏移",
        "focus": "左右嘴角、脸颊和下巴",
    },
    "Mouth_Stretch": {
        "label": "咧嘴拉伸",
        "description": "嘴角向两侧拉开",
        "focus": "嘴角、脸颊和唇周纹理",
    },
    "MouthSqueeze": {
        "label": "抿嘴挤压",
        "description": "嘴唇和嘴角向中心挤压",
        "focus": "唇线、嘴角和下巴",
    },
    "Neutral": {
        "label": "中性表情",
        "description": "面部肌肉自然放松、双眼自然睁开",
        "focus": "五官比例、眼神稳定和皮肤基础纹理",
    },
    "Neutral_CloseEye": {
        "label": "中性闭眼",
        "description": "放松状态下自然闭眼",
        "focus": "上下眼睑贴合、眼角和眼周纹理",
    },
    "Sneer": {
        "label": "轻蔑皱鼻",
        "description": "单侧上唇、鼻翼和嘴角牵拉",
        "focus": "鼻翼、上唇、法令纹和单侧嘴角",
    },
}


def expression_action_metadata(action: str) -> Dict[str, str]:
    metadata = EXPRESSION_ACTION_METADATA.get(action)
    if metadata is not None:
        return metadata
    return {
        "label": action.replace("_", " "),
        "description": "Expression 参考表情原型",
        "focus": "面部肌肉、眼神和局部皮肤纹理",
    }


def face_action_diagnosis(
        action: str,
        geometry_score: Optional[float],
        gaze_score: Optional[float],
        texture_score: Optional[float],
        wrinkle_score: Optional[float],
) -> Dict[str, str]:
    """Translate the weakest action component into an actionable diagnosis."""

    metadata = expression_action_metadata(action)
    components = [
        ("肌肉几何", geometry_score),
        ("眼神", gaze_score),
        ("皮肤纹理", texture_score),
        ("皱纹细节", wrinkle_score),
    ]
    available = [
        (label, float(score))
        for label, score in components
        if score is not None
    ]
    if not available:
        return {
            "weak_dimension": "证据不足",
            "diagnosis": f"{metadata['focus']}缺少可比较的局部人脸证据。",
            "advice": f"补充清晰正脸帧，并复核{metadata['focus']}的稳定性。",
        }

    weak_dimension, _ = min(available, key=lambda item: item[1])
    if weak_dimension == "肌肉几何":
        diagnosis = f"{metadata['focus']}的牵拉幅度或相对位置与参考原型偏离。"
        advice = f"调整{metadata['focus']}的动作幅度和过渡，避免五官跟随表情发生漂移。"
    elif weak_dimension == "眼神":
        diagnosis = f"{metadata['focus']}中的视线方向或眼睑配合不够一致。"
        advice = f"固定瞳孔移动轨迹，并让眼睑开合与{metadata['label']}同步变化。"
    elif weak_dimension == "皮肤纹理":
        diagnosis = f"{metadata['focus']}的局部皮肤层次偏平或清晰度不足。"
        advice = f"减少{metadata['focus']}的过度磨皮和压缩模糊，保留肌肉牵拉后的明暗层次。"
    else:
        diagnosis = f"{metadata['focus']}的皱纹/高频细节没有稳定跟随肌肉动作。"
        advice = f"让{metadata['focus']}的细纹随{metadata['label']}连续变化，避免冻结纹理或帧间闪烁。"

    return {
        "weak_dimension": weak_dimension,
        "diagnosis": diagnosis,
        "advice": advice,
    }


def feature_scales_by_dimension(
        values: Sequence[Any],
        base_scale: float,
) -> Dict[int, Any]:
    """Build robust scales without stacking mixed-length feature vectors."""

    grouped: Dict[int, List[Any]] = {}
    for value in values:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        grouped.setdefault(len(array), []).append(array)
    return {
        dimension: np.std(np.stack(arrays, axis=0), axis=0) + base_scale
        for dimension, arrays in grouped.items()
    }


def match_expression_reference(
        structure: Any,
        detail: Any,
        region_energy: Any,
        geometry: Optional[Any],
        gaze: Optional[Any],
        view: str,
        references: Sequence[ExpressionReference],
        geometry_scale: Optional[Any],
        gaze_scale: Optional[Any],
) -> Optional[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for reference in references:
        structure_score = vector_similarity(structure, reference.structure)
        detail_score = vector_similarity(detail, reference.detail)
        region_scores = [
            relative_metric_match(float(value_a), float(value_b))
            for value_a, value_b in zip(
                np.asarray(region_energy).reshape(-1),
                np.asarray(reference.region_energy).reshape(-1),
            )
        ]
        wrinkle_score = clamp(
            0.65 * (detail_score if detail_score is not None else 0.0)
            + 0.35 * (safe_mean(region_scores) or 0.0)
        )
        texture_score = clamp(
            0.55 * (structure_score if structure_score is not None else 0.0)
            + 0.45 * (detail_score if detail_score is not None else 0.0)
        )
        reference_geometry_scale = geometry_scale
        if isinstance(geometry_scale, dict) and reference.geometry is not None:
            reference_geometry_scale = geometry_scale.get(
                len(np.asarray(reference.geometry).reshape(-1))
            )
        reference_gaze_scale = gaze_scale
        if isinstance(gaze_scale, dict) and reference.gaze is not None:
            reference_gaze_scale = gaze_scale.get(
                len(np.asarray(reference.gaze).reshape(-1))
            )
        geometry_score = expression_feature_similarity(
            geometry,
            reference.geometry,
            reference_geometry_scale,
        )
        gaze_score = expression_feature_similarity(
            gaze,
            reference.gaze,
            reference_gaze_scale,
        )

        weighted_parts: List[Tuple[float, float]] = [
            (texture_score, 0.24),
            (wrinkle_score, 0.22),
        ]
        if geometry_score is not None:
            weighted_parts.append((geometry_score, 0.36))
        if gaze_score is not None:
            weighted_parts.append((gaze_score, 0.18))
        total_weight = sum(weight for _, weight in weighted_parts)
        score = clamp(
            sum(value * weight for value, weight in weighted_parts)
            / max(total_weight, 1e-6)
        )
        if view != "Unknown" and reference.view != "Unknown":
            view_match = 1.0 if view == reference.view else 0.0
            score = clamp(score * (0.96 + 0.04 * view_match))
        matches.append(
            {
                "action": reference.action,
                "group": expression_action_group(reference.action),
                "view": reference.view,
                "path": reference.path,
                "score": score,
                "geometry_score": geometry_score,
                "gaze_score": gaze_score,
                "texture_score": texture_score,
                "wrinkle_score": wrinkle_score,
            }
        )

    if not matches:
        return None
    matches.sort(key=lambda item: float(item["score"]), reverse=True)
    return {
        "best": matches[0],
        "top_matches": matches[:3],
    }


def compute_face_motion_quality(
        records: Sequence[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """Check whether muscle shape and local wrinkle detail move coherently."""

    usable = [
        record for record in records
        if record.get("geometry") is not None
        and record.get("region_energy") is not None
    ]
    if len(usable) < 3:
        return 0.70, {
            "method": "expression geometry/high-frequency temporal proxy",
            "valid_frames": len(usable),
            "warning": "有效人脸帧不足，肌肉与皱纹时序分数按保守代理值计算",
        }

    geometry = np.asarray(
        [record["geometry"] for record in usable],
        dtype=np.float32,
    )
    region_energy = np.asarray(
        [record["region_energy"] for record in usable],
        dtype=np.float32,
    )
    geometry_step = np.linalg.norm(np.diff(geometry, axis=0), axis=1)
    detail_step = np.linalg.norm(np.diff(region_energy, axis=0), axis=1)

    geometry_acceleration = (
        float(np.mean(np.abs(np.diff(geometry_step))))
        if len(geometry_step) >= 2
        else 0.0
    )
    detail_acceleration = (
        float(np.mean(np.abs(np.diff(detail_step))))
        if len(detail_step) >= 2
        else 0.0
    )
    geometry_smoothness = clamp(
        math.exp(-geometry_acceleration / 0.035)
    )
    wrinkle_smoothness = clamp(
        math.exp(-detail_acceleration / 0.12)
    )

    geometry_activity = float(np.median(geometry_step))
    detail_activity = float(np.median(detail_step))
    if geometry_activity < 0.004 and detail_activity < 0.025:
        response_score = 1.0
    elif geometry_activity < 0.004:
        response_score = 0.35
    elif detail_activity < 0.008:
        response_score = 0.55
    else:
        log_ratio = np.log(
            (detail_step + 1e-4) / (geometry_step + 1e-4)
        )
        response_score = clamp(math.exp(-float(np.std(log_ratio)) / 1.2))

    gaze_smoothness = 1.0
    gaze_values = [
        record["gaze"] for record in usable
        if record.get("gaze") is not None
    ]
    if len(gaze_values) >= 3:
        gaze_step = np.linalg.norm(
            np.diff(np.asarray(gaze_values, dtype=np.float32), axis=0),
            axis=1,
        )
        gaze_acceleration = (
            float(np.mean(np.abs(np.diff(gaze_step))))
            if len(gaze_step) >= 2
            else 0.0
        )
        gaze_smoothness = clamp(
            math.exp(-gaze_acceleration / 0.08)
        )

    final_score = clamp(
        0.35 * geometry_smoothness
        + 0.30 * wrinkle_smoothness
        + 0.20 * response_score
        + 0.15 * gaze_smoothness
    )
    return final_score, {
        "method": "expression geometry/high-frequency temporal proxy",
        "valid_frames": len(usable),
        "geometry_smoothness": geometry_smoothness,
        "wrinkle_smoothness": wrinkle_smoothness,
        "muscle_wrinkle_response": response_score,
        "gaze_smoothness": gaze_smoothness,
        "geometry_activity": geometry_activity,
        "wrinkle_activity": detail_activity,
        "geometry_acceleration": geometry_acceleration,
        "wrinkle_acceleration": detail_acceleration,
    }


def compute_face_expression_metric(
        gen_samples: VideoSamples,
        gen_faces: Sequence[FaceObservation],
        gen_landmarks: Sequence[Optional[Any]],
        expression_references: Sequence[ExpressionReference],
) -> MetricResult:
    """Score generated faces against action-specific expression prototypes."""

    if not expression_references:
        return MetricResult(
            name="人脸表情与肌肉运动",
            score=None,
            weight=WEIGHTS["face_expression"],
            status="unavailable",
            details={
                "warning": "没有可用的 Expression 参考图",
            },
        )

    geometry_values = [
        reference.geometry
        for reference in expression_references
        if reference.geometry is not None
    ]
    gaze_values = [
        reference.gaze
        for reference in expression_references
        if reference.gaze is not None
    ]
    geometry_scale = (
        feature_scales_by_dimension(geometry_values, 0.04)
        if geometry_values
        else None
    )
    gaze_scale = (
        feature_scales_by_dimension(gaze_values, 0.08)
        if gaze_values
        else None
    )

    records: List[Dict[str, Any]] = []
    for frame_index, (frame, face, landmarks) in enumerate(
            zip(gen_samples.frames, gen_faces, gen_landmarks)
    ):
        aligned = canonical_face_image(frame, face, landmarks)
        structure, detail, region_energy = face_texture_features(aligned)
        geometry = expression_geometry_features(landmarks)
        if geometry is None:
            geometry = coarse_expression_geometry_features(aligned)
        gaze = expression_gaze_features(landmarks)
        if gaze is None:
            gaze = coarse_gaze_features(aligned)
        view = estimate_expression_view(landmarks)
        match = match_expression_reference(
            structure=structure,
            detail=detail,
            region_energy=region_energy,
            geometry=geometry,
            gaze=gaze,
            view=view,
            references=expression_references,
            geometry_scale=geometry_scale,
            gaze_scale=gaze_scale,
        )
        if match is None:
            continue
        records.append(
            {
                "frame_index": int(gen_samples.indices[frame_index]),
                "geometry": geometry,
                "gaze": gaze,
                "region_energy": region_energy,
                "match": match["best"],
                "top_matches": match["top_matches"],
            }
        )

    if not records:
        return MetricResult(
            name="人脸表情与肌肉运动",
            score=None,
            weight=WEIGHTS["face_expression"],
            status="unavailable",
            details={
                "reference_count": len(expression_references),
                "warning": "没有生成视频帧能够提取可比较的人脸表情特征",
            },
        )

    frame_scores = [float(record["match"]["score"]) for record in records]
    frame_match_score = float(np.mean(frame_scores))
    tail_count = max(1, int(math.ceil(len(frame_scores) * 0.20)))
    tail_score = float(np.mean(sorted(frame_scores)[:tail_count]))
    consistency_score = clamp(
        math.exp(-float(np.std(frame_scores)) / 0.15)
    )
    motion_score, motion_details = compute_face_motion_quality(records)
    coverage = len(records) / max(len(gen_samples.frames), 1)
    final_score = clamp(
        0.50 * frame_match_score
        + 0.15 * tail_score
        + 0.15 * consistency_score
        + 0.20 * motion_score
    )
    final_score = clamp(0.90 * final_score + 0.10 * coverage)

    action_metrics: Dict[str, Dict[str, List[float]]] = {}
    group_scores: Dict[str, List[float]] = {}
    for record in records:
        match = record["match"]
        action_bucket = action_metrics.setdefault(
            match["action"],
            {
                "score": [],
                "geometry_score": [],
                "gaze_score": [],
                "texture_score": [],
                "wrinkle_score": [],
            },
        )
        action_bucket["score"].append(float(match["score"]))
        for key in (
            "geometry_score",
            "gaze_score",
            "texture_score",
            "wrinkle_score",
        ):
            value = match.get(key)
            if value is not None:
                action_bucket[key].append(float(value))
        group_scores.setdefault(match["group"], []).append(
            float(match["score"])
        )
    action_summary = []
    for action, values in action_metrics.items():
        geometry_score = safe_mean(values["geometry_score"])
        gaze_score = safe_mean(values["gaze_score"])
        texture_score = safe_mean(values["texture_score"])
        wrinkle_score = safe_mean(values["wrinkle_score"])
        metadata = expression_action_metadata(action)
        diagnosis = face_action_diagnosis(
            action=action,
            geometry_score=geometry_score,
            gaze_score=gaze_score,
            texture_score=texture_score,
            wrinkle_score=wrinkle_score,
        )
        action_summary.append(
            {
                "action": action,
                "label": metadata["label"],
                "description": metadata["description"],
                "focus": metadata["focus"],
                "group": expression_action_group(action),
                "score": float(np.median(values["score"])),
                "frame_count": len(values["score"]),
                "geometry_score": geometry_score,
                "gaze_score": gaze_score,
                "texture_score": texture_score,
                "wrinkle_score": wrinkle_score,
                **diagnosis,
            }
        )
    action_summary.sort(key=lambda item: item["score"], reverse=True)
    group_summary = {
        group: float(np.median(values))
        for group, values in group_scores.items()
    }

    geometry_scores = [
        record["match"]["geometry_score"]
        for record in records
        if record["match"].get("geometry_score") is not None
    ]
    gaze_scores = [
        record["match"]["gaze_score"]
        for record in records
        if record["match"].get("gaze_score") is not None
    ]
    texture_scores = [
        float(record["match"]["texture_score"])
        for record in records
    ]
    wrinkle_scores = [
        float(record["match"]["wrinkle_score"])
        for record in records
    ]
    top_frame_matches = [
        {
            "frame_index": record["frame_index"],
            "action": record["match"]["action"],
            "view": record["match"]["view"],
            "score": float(record["match"]["score"]),
            "geometry_score": record["match"].get("geometry_score"),
            "gaze_score": record["match"].get("gaze_score"),
            "texture_score": float(record["match"]["texture_score"]),
            "wrinkle_score": float(record["match"]["wrinkle_score"]),
        }
        for record in sorted(
            records,
            key=lambda item: float(item["match"]["score"]),
            reverse=True,
        )[:8]
    ]
    mesh_geometry_available = any(
        reference.landmarks is not None and len(reference.landmarks) >= 292
        for reference in expression_references
    ) and any(
        landmarks is not None and len(landmarks) >= 292
        for landmarks in gen_landmarks
    )
    mesh_gaze_available = any(
        reference.landmarks is not None and len(reference.landmarks) >= 478
        for reference in expression_references
    ) and any(
        landmarks is not None and len(landmarks) >= 478
        for landmarks in gen_landmarks
    )
    ranked_low_reference_actions = sorted(
        action_summary,
        key=lambda item: item["score"],
    )
    low_reference_actions = ranked_low_reference_actions[:5]
    semantic_qa_suggestions = [
        (
            f"“{item['label']}”：{item['diagnosis']}"
            f" 建议：{item['advice']}"
        )
        for item in low_reference_actions[:3]
    ]
    details = {
        "score": final_score,
        "method": (
            "Expression 动作原型匹配：MediaPipe 肌肉几何 + 眼神虹膜位置 "
            "+ 局部皱纹/高频纹理"
        ),
        "expression_dir": str(
            Path(expression_references[0].path).resolve().parent
        ),
        "reference_count": len(expression_references),
        "valid_face_frames": len(records),
        "total_sampled_frames": len(gen_samples.frames),
        "face_coverage": coverage,
        "frame_match_score": frame_match_score,
        "tail_20_percent_score": tail_score,
        "match_consistency_score": consistency_score,
        "geometry_score": safe_mean(geometry_scores),
        "gaze_score": safe_mean(gaze_scores),
        "texture_score": safe_mean(texture_scores),
        "wrinkle_score": safe_mean(wrinkle_scores),
        "motion_score": motion_score,
        "motion_details": motion_details,
        "geometry_method": (
            "MediaPipe Face Mesh"
            if mesh_geometry_available
            else "OpenCV 分区形状/高频 proxy"
        ),
        "gaze_method": (
            "MediaPipe iris landmarks"
            if mesh_gaze_available
            else "OpenCV 眼区暗部质心 proxy"
        ),
        "action_scores": action_summary,
        "low_reference_actions": low_reference_actions,
        "action_match_explanation": (
            "这里展示 Expression 参考原型中匹配度相对较低的表情，"
            "用于定位需要检查的局部区域；它不是动作识别结果，"
            "也不能据此判断人物一定没有执行该动作。"
        ),
        "semantic_qa_suggestions": semantic_qa_suggestions,
        "group_scores": group_summary,
        "top_frame_matches": top_frame_matches,
        "evidence": [
            {
                "label": "表情动作原型匹配",
                "value": f"{frame_match_score * 100.0:.1f}%",
            },
            {
                "label": "眼神/虹膜位置匹配",
                "value": (
                    f"{float(np.mean(gaze_scores)) * 100.0:.1f}%"
                    if gaze_scores
                    else "不可用"
                ),
            },
            {
                "label": "肌肉几何与皱纹纹理匹配",
                "value": f"{float(np.mean(wrinkle_scores)) * 100.0:.1f}%",
            },
            {
                "label": "连续帧肌肉-皱纹同步",
                "value": f"{motion_score * 100.0:.1f}%",
            },
        ],
    }
    if not mesh_gaze_available:
        details["warning"] = (
            "当前环境未启用 MediaPipe/虹膜关键点，眼神和肌肉分数使用 OpenCV proxy"
        )
    return MetricResult(
        name="人脸表情与肌肉运动",
        score=final_score,
        weight=WEIGHTS["face_expression"],
        status="ok" if mesh_geometry_available else "proxy",
        details=details,
    )


def compute_expression_text_metric(
        gen_samples: VideoSamples,
        gen_faces: Sequence[FaceObservation],
        gen_landmarks: Sequence[Optional[Any]],
        ref_samples: Optional[VideoSamples],
        ref_landmarks: Optional[Sequence[Optional[Any]]],
        prompt: Optional[str],
        qa_score: Optional[Tuple[float, Dict[str, Any]]],
        qwen_qa_result: Optional[Tuple[float, List[Dict[str, Any]]]],
) -> MetricResult:
    """
    表情/文本准确性 15%。

    优先融合 Qwen3-VL 评估结果、外部视频模型问答、参考视频表情序列；
    缺少参考视频时，使用 MediaPipe 关键点算法评估表情变化的自然度。
    """

    details: Dict[str, Any] = {}
    sub_scores: List[Tuple[float, float, str]] = []

    # Qwen3-VL 评估结果
    if qwen_qa_result is not None:
        score, qa_details = qwen_qa_result
        sub_scores.append((score, 0.50, "Qwen3-VL 视频理解问答"))
        details["qwen_qa"] = {
            "score": score,
            "source": "Qwen3-VL-4B-Instruct 自动评估",
        }
        details["qa_scores"] = [
            {
                "id": item.get("id"),
                "category": item.get("category", "general"),
                "question": item.get("question", ""),
                "score": item.get("score"),
            }
            for item in qa_details
            if item.get("score") is not None
        ]

    # 外部 QA 结果
    if qa_score is not None:
        score, qa_details = qa_score
        sub_scores.append((score, 0.30, "外部视频模型问答"))
        details["external_qa"] = qa_details

    # 表情关键点算法
    expression_score: Optional[float] = None
    if ref_samples is not None and ref_landmarks is not None:
        gen_features = [
            feature
            for feature in (expression_features(points) for points in gen_landmarks)
            if feature is not None
        ]
        ref_features = [
            feature
            for feature in (expression_features(points) for points in ref_landmarks)
            if feature is not None
        ]
        if len(gen_features) >= 2 and len(ref_features) >= 2:
            target_length = max(2, min(len(gen_features), len(ref_features)))
            gen_array = resample_sequence(np.asarray(gen_features), target_length)
            ref_array = resample_sequence(np.asarray(ref_features), target_length)
            feature_scale = np.std(ref_array, axis=0) + 0.05
            normalized_mae = float(
                np.mean(np.abs(gen_array - ref_array) / feature_scale)
            )
            expression_score = clamp(math.exp(-normalized_mae))
            details["expression_method"] = "MediaPipe 参考视频表情特征对齐"
            details["expression_score"] = expression_score
            details["expression_normalized_mae"] = normalized_mae
            details["expression_valid_frames"] = {
                "generated": len(gen_features),
                "reference": len(ref_features),
            }
    else:
        expression_result = estimate_expression_from_landmarks(gen_landmarks)
        if expression_result is None:
            expression_result = estimate_expression_from_face_motion(
                gen_samples.frames,
                gen_faces,
            )
        if expression_result is not None:
            expression_score, expression_details = expression_result
            details["expression_method"] = expression_details["method"]
            details["expression_score"] = expression_score
            details["expression_algorithm"] = expression_details

    if expression_score is not None:
        sub_scores.append((expression_score, 0.20, "表情关键点算法"))

    if not sub_scores:
        return MetricResult(
            name="表情/文本准确性",
            score=None,
            weight=WEIGHTS["expression_text"],
            status="unavailable",
            details={
                "warning": (
                    "当前没有可用的 Qwen 评估、参考视频、关键点或外部视频模型结果，"
                    "无法生成表情/文本动态分数"
                ),
                **details,
            },
        )

    total_sub_weight = sum(weight for _, weight, _ in sub_scores)
    final_score = sum(score * weight for score, weight, _ in sub_scores)
    final_score /= total_sub_weight
    details["sub_scores"] = {
        source: score for score, _, source in sub_scores
    }
    return MetricResult(
        name="表情/文本准确性",
        score=clamp(final_score),
        weight=WEIGHTS["expression_text"],
        status="ok" if qwen_qa_result is not None or qa_score is not None else "proxy",
        details=details,
    )


def procrustes_align(points: Any, target: Any) -> Any:
    """
    用相似变换去除人脸平移、缩放和整体旋转。

    这样计算出来的残差更接近五官自身的抖动，而不是头部整体运动。
    """

    points = np.asarray(points, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if points.shape != target.shape or points.ndim != 2:
        return points

    points_center = points.mean(axis=0, keepdims=True)
    target_center = target.mean(axis=0, keepdims=True)
    points_zero = points - points_center
    target_zero = target - target_center
    points_norm = float(np.linalg.norm(points_zero))
    target_norm = float(np.linalg.norm(target_zero))
    if points_norm < 1e-8 or target_norm < 1e-8:
        return points

    points_zero = points_zero / points_norm
    target_zero = target_zero / target_norm
    covariance = points_zero.T @ target_zero
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = (points_zero @ rotation) * target_norm + target_center
    return aligned


def normalized_landmark_sequence(
        landmark_list: Sequence[Optional[Any]],
) -> List[Any]:
    """对每帧关键点做尺度归一化和相似变换对齐。"""

    valid = [points for points in landmark_list if points is not None]
    if not valid:
        return []
    reference = np.mean(np.asarray(valid, dtype=np.float32), axis=0)
    output = []
    for points in valid:
        points = np.asarray(points, dtype=np.float32)[:, :2]
        target = reference[:, :2]
        output.append(procrustes_align(points, target))
    return output


def compute_landmark_jitter(landmark_list: Sequence[Optional[Any]]) -> Optional[float]:
    """计算去除头部刚性运动后的关键点 jitter。"""

    sequence = normalized_landmark_sequence(landmark_list)
    if len(sequence) < 3:
        return None
    array = np.asarray(sequence, dtype=np.float32)
    velocity = np.diff(array, axis=0)
    acceleration = np.diff(velocity, axis=0)
    # 二阶差分对局部跳动更敏感，使用平均绝对值减少异常点影响。
    return float(np.mean(np.linalg.norm(acceleration, axis=-1)))


def calculate_optical_flow_warp_errors(frames: Sequence[Any]) -> List[float]:
    """计算相邻帧的光流反向 warp 误差。"""

    if len(frames) < 2:
        return []
    errors = []
    for previous, current in zip(frames[:-1], frames[1:]):
        previous_small = cv2.resize(previous, (256, 256))
        current_small = cv2.resize(current, (256, 256))
        previous_gray = cv2.cvtColor(previous_small, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        grid_x, grid_y = np.meshgrid(
            np.arange(256, dtype=np.float32),
            np.arange(256, dtype=np.float32),
        )
        map_x = grid_x - flow[..., 0]
        map_y = grid_y - flow[..., 1]
        warped = cv2.remap(
            previous_gray,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        error = np.mean(
            np.abs(warped.astype(np.float32) - current_gray.astype(np.float32))
        ) / 255.0
        errors.append(float(error))
    return errors


def calculate_flow_sequence(frames: Sequence[Any]) -> Optional[Any]:
    """提取低分辨率光流序列，供有参考视频时比较运动端点误差。"""

    if len(frames) < 2:
        return None
    flows = []
    for previous, current in zip(frames[:-1], frames[1:]):
        previous_small = cv2.resize(previous, (64, 64))
        current_small = cv2.resize(current, (64, 64))
        previous_gray = cv2.cvtColor(previous_small, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=11,
            iterations=2,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )
        flows.append(flow.reshape(-1, 2).mean(axis=0))
    return np.asarray(flows, dtype=np.float32)


def compute_temporal_metric(
        gen_samples: VideoSamples,
        gen_faces: Sequence[FaceObservation],
        gen_landmarks: Sequence[Optional[Any]],
        ref_samples: Optional[VideoSamples],
        ref_faces: Optional[Sequence[FaceObservation]],
        ref_landmarks: Optional[Sequence[Optional[Any]]],
        face_sim_low: float,
        face_sim_high: float,
) -> MetricResult:
    """
    时间稳定性 25%：
    - 身份相似度帧间波动；
    - 去除头部刚性运动后的关键点 jitter；
    - 相邻帧光流 warp error；
    - 有参考视频时额外比较生成/参考运动序列。
    """

    sub_scores: List[Tuple[float, str]] = []
    details: Dict[str, Any] = {}

    embeddings = [
        observation.embedding
        for observation in gen_faces
        if observation.embedding is not None
    ]
    if len(embeddings) >= 3:
        template = np.mean(np.asarray(embeddings), axis=0)
        template_norm = float(np.linalg.norm(template))
        if template_norm > 1e-8:
            template = template / template_norm
            similarities = [
                cosine_similarity(embedding, template)
                for embedding in embeddings
            ]
            similarities = [
                float(value) for value in similarities if value is not None
            ]
            if len(similarities) >= 3:
                sim_std = float(np.std(similarities))
                sim_diff = float(np.mean(np.abs(np.diff(similarities))))
                identity_temporal_score = clamp(
                    0.60 * math.exp(-sim_std / 0.10)
                    + 0.40 * math.exp(-sim_diff / 0.08)
                )
                sub_scores.append((identity_temporal_score, "ArcFace 波动"))
                details["identity_temporal_score"] = identity_temporal_score
                details["identity_similarity_std"] = sim_std
                details["identity_similarity_mean_abs_diff"] = sim_diff

    landmark_jitter = compute_landmark_jitter(gen_landmarks)
    if landmark_jitter is not None:
        landmark_score = clamp(math.exp(-landmark_jitter / 0.012))
        sub_scores.append((landmark_score, "关键点 jitter"))
        details["landmark_jitter"] = landmark_jitter
        details["landmark_jitter_score"] = landmark_score

    warp_errors = calculate_optical_flow_warp_errors(gen_samples.frames)
    warp_score: Optional[float] = None
    if warp_errors:
        warp_mean = float(np.mean(warp_errors))
        warp_score = clamp(math.exp(-4.0 * warp_mean))
        sub_scores.append((warp_score, "自视频光流 warp error"))
        details["warp_error_mean"] = warp_mean
        details["warp_score"] = warp_score

    if ref_samples is not None:
        gen_flow = calculate_flow_sequence(gen_samples.frames)
        ref_flow = calculate_flow_sequence(ref_samples.frames)
        if gen_flow is not None and ref_flow is not None:
            target_length = max(1, min(len(gen_flow), len(ref_flow)))
            gen_flow = resample_sequence(gen_flow, target_length)
            ref_flow = resample_sequence(ref_flow, target_length)
            flow_scale = float(np.std(ref_flow)) + 0.05
            flow_rmse = float(np.sqrt(np.mean((gen_flow - ref_flow) ** 2)))
            endpoint_score = clamp(math.exp(-flow_rmse / flow_scale))
            sub_scores.append((endpoint_score, "参考视频运动端点误差"))
            details["reference_flow_rmse"] = flow_rmse
            details["reference_flow_score"] = endpoint_score

        if ref_landmarks is not None and landmark_jitter is not None:
            ref_jitter = compute_landmark_jitter(ref_landmarks)
            if ref_jitter is not None:
                # 这里比较稳定性量级，而不是强行要求两段视频逐帧一致。
                jitter_ratio = abs(landmark_jitter - ref_jitter) / (
                        abs(ref_jitter) + 0.01
                )
                stability_match_score = clamp(math.exp(-jitter_ratio))
                sub_scores.append(
                    (stability_match_score, "参考视频关键点稳定性匹配")
                )
                details["reference_landmark_jitter"] = ref_jitter
                details["reference_landmark_stability_score"] = (
                    stability_match_score
                )

    if not sub_scores:
        return MetricResult(
            name="时间稳定性",
            score=None,
            weight=WEIGHTS["temporal"],
            status="unavailable",
            details={
                "warning": "没有足够帧数或可用特征，无法评估时间稳定性"
            },
        )

    final_score = float(np.mean([score for score, _ in sub_scores]))
    details["available_submetrics"] = [name for _, name in sub_scores]
    return MetricResult(
        name="时间稳定性",
        score=clamp(final_score),
        weight=WEIGHTS["temporal"],
        status="ok" if ref_samples is not None else "proxy",
        details=details,
    )


def compute_aesthetic_metric(
        samples: VideoSamples,
        faces: Sequence[FaceObservation],
) -> MetricResult:
    """
    美学 10%：使用多帧图像质量和构图算法生成分数。

    评分包含主体曝光、过曝/欠曝、对比度、清晰度、构图稳定性和画面
    时间一致性，不使用固定人工分数。
    """

    exposure_scores = []
    clipping_scores = []
    contrast_scores = []
    sharpness_values = []
    luma_values = []
    for frame in samples.frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mean_luma = float(np.mean(gray))
        luma_values.append(mean_luma)
        percentile_low, percentile_high = np.percentile(gray, [5.0, 95.0])
        # 以有效动态范围和主体平均亮度共同估计曝光，避免纯黑背景拉低分数。
        dynamic_range = clamp(float(percentile_high - percentile_low) / 0.75)
        exposure_center = 1.0 - min(abs(mean_luma - 0.45) / 0.45, 1.0)
        exposure_scores.append(clamp(0.55 * dynamic_range + 0.45 * exposure_center))
        clipping_ratio = float(
            np.mean((gray < 0.02) | (gray > 0.98))
        )
        clipping_scores.append(clamp(1.0 - clipping_ratio * 10.0))
        contrast_scores.append(clamp(float(np.std(gray)) / 0.25))
        sharpness_values.append(frame_sharpness(frame))

    sharpness_score = clamp(
        1.0 - math.exp(-math.log1p(float(np.median(sharpness_values))) / 5.5)
    )
    sharpness_median = max(float(np.median(sharpness_values)), 1e-6)
    sharpness_consistency_score = clamp(
        math.exp(
            -float(np.std(sharpness_values))
            / max(sharpness_median, 1.0)
        )
    )
    frame_luma_std = float(np.std(luma_values))
    luma_consistency_score = clamp(math.exp(-frame_luma_std / 0.08))
    temporal_consistency_score = clamp(
        0.55 * luma_consistency_score
        + 0.45 * sharpness_consistency_score
    )

    composition_score: Optional[float] = None
    centers, scales = face_centers_and_scales(
        faces,
        samples.width,
        samples.height,
    )
    if len(centers) >= 2:
        center_array = np.asarray(centers, dtype=np.float32)
        scale_array = np.asarray(scales, dtype=np.float32)
        center_jitter = float(np.mean(np.std(center_array, axis=0)))
        scale_jitter = float(
            np.std(scale_array) / max(float(np.mean(scale_array)), 1e-6)
        )
        composition_score = clamp(
            0.60 * math.exp(-center_jitter / 0.08)
            + 0.40 * math.exp(-scale_jitter / 0.35)
        )

    weighted_scores: List[Tuple[float, float]] = [
        (float(np.mean(exposure_scores)), 0.22),
        (float(np.mean(clipping_scores)), 0.18),
        (float(np.mean(contrast_scores)), 0.15),
        (sharpness_score, 0.22),
        (temporal_consistency_score, 0.10),
    ]
    if composition_score is not None:
        weighted_scores.append((composition_score, 0.13))
    else:
        # 没有人脸框时，将构图权重按比例分配给已有图像质量指标。
        extra_weight = 0.13 / len(weighted_scores)
        weighted_scores = [
            (score, weight + extra_weight)
            for score, weight in weighted_scores
        ]
    total_weight = sum(weight for _, weight in weighted_scores)
    final_score = clamp(
        sum(score * weight for score, weight in weighted_scores) / total_weight
    )
    return MetricResult(
        name="美学",
        score=final_score,
        weight=WEIGHTS["aesthetic"],
        status="proxy",
        details={
            "method": "多帧曝光/裁剪/对比度/清晰度/构图算法",
            "exposure_score": float(np.mean(exposure_scores)),
            "clipping_score": float(np.mean(clipping_scores)),
            "contrast_score": float(np.mean(contrast_scores)),
            "sharpness_score": sharpness_score,
            "temporal_consistency_score": temporal_consistency_score,
            "composition_score": composition_score,
            "frame_luma_std": frame_luma_std,
            "sharpness_consistency_score": sharpness_consistency_score,
            "face_coverage": len(centers) / max(len(faces), 1),
            "warning": (
                "该分数是可复现的图像质量与构图算法分，"
                "不等同于主观审美评审"
            ),
        },
    )


def metric_status_label(status: str) -> str:
    """将内部状态转换为中文状态。"""

    return {
        "ok": "已完成",
        "proxy": "质量估计",
        "manual": "人工评分",
        "unavailable": "证据不足",
    }.get(status, status)


def score_level(score: Optional[float]) -> str:
    """将 0~1 分数转换为中文等级。"""

    if score is None:
        return "无法判断"
    if score >= 0.90:
        return "优秀"
    if score >= 0.80:
        return "较好"
    if score >= 0.65:
        return "一般"
    if score >= 0.50:
        return "偏弱"
    return "较差"


def unique_texts(items: Sequence[str], limit: int = 4) -> List[str]:
    """保持顺序去重，避免多个指标输出相同的模板句。"""

    result: List[str] = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
        if len(result) >= limit:
            break
    return result


def build_metric_analysis(metric: MetricResult) -> Dict[str, Any]:
    """根据具体质量证据生成用户可执行、按指标区分的分析。"""

    score = metric.score
    details = metric.details
    if score is None:
        return {
            "结论": "当前证据不足，无法可靠判断本项。",
            "好在哪里": [],
            "不好在哪里": ["本项没有形成稳定、可复核的质量证据。"],
            "可能影响": ["本项结果不应单独作为视频质量结论。"],
            "后续改进方向": [
                "补充与本项对应的参考图像或参考视频，并确认主体在主要片段中持续可见。"
            ],
        }

    level = score_level(score)
    common_conclusion = f"本项得分为 {score * 100:.2f} 分，整体表现{level}。"
    good: List[str] = []
    bad: List[str] = []
    impact: List[str] = []
    improvements: List[str] = []

    if metric.name == "角色一致性":
        coverage = float(details.get("face_coverage", 0.0) or 0.0)
        similarity = float(details.get("mean_similarity", 0.0) or 0.0)
        similarity_std = float(details.get("similarity_std", 0.0) or 0.0)
        center_jitter = float(details.get("center_jitter", 0.0) or 0.0)
        scale_jitter = float(details.get("scale_jitter", 0.0) or 0.0)

        if coverage >= 0.95:
            good.append(f"主体在 {coverage * 100:.1f}% 的采样帧中保持可见。")
        if similarity >= 0.70:
            good.append("脸型和五官特征与参考主体的整体相似度较高。")
        if similarity_std <= 0.06:
            good.append("帧间身份特征变化较小，未见明显换脸式跳变。")
        if coverage < 0.95:
            bad.append(f"有 {(1.0 - coverage) * 100:.1f}% 的采样帧未稳定捕捉到主体。")
            improvements.append(
                "优先检查主体出框、侧脸、遮挡和快速转头片段，调整构图或减少这些片段中的身份丢失。"
            )
        if similarity < 0.70:
            bad.append("部分帧的脸型或五官比例偏离参考主体。")
            improvements.append(
                "针对偏离最大的片段固定眼睛、鼻梁、嘴角和下颌轮廓，避免表情变化带动五官位置漂移。"
            )
        if similarity_std > 0.06:
            bad.append("帧间身份特征波动较大，局部可能出现五官变形。")
            improvements.append(
                "抽查身份波动最大的开头、中段和结尾帧，分别修复脸部边缘、眼睛和嘴部的跳变。"
            )
        if center_jitter > 0.03:
            improvements.append("固定人物在画面中的中心位置，减少镜头漂移造成的主体比例变化。")
        if scale_jitter > 0.08:
            improvements.append("统一人物画面占比，避免前后景别变化让脸部大小产生突变。")
        if not improvements:
            improvements.append("保持当前人物占比和姿态范围，并继续抽查大幅转头与遮挡片段。")
        impact.append("身份漂移会直接破坏观众对人物连续性的判断。")

    elif metric.name == "质感和细节":
        sharpness = float(
            details.get(
                "sharpness_proxy_score",
                details.get("generated_quality_score", 0.0),
            )
            or 0.0
        )
        high_frequency = float(
            details.get(
                "high_frequency_proxy_score",
                details.get("high_frequency_match", 0.0),
            )
            or 0.0
        )
        reference_quality = details.get("reference_quality_score")

        if sharpness >= 0.70:
            good.append("人脸轮廓、发丝边缘和服装边界保持了较好的清晰度。")
        if high_frequency >= 0.70:
            good.append("局部纹理信息较充足，没有明显的整体糊化。")
        if reference_quality is not None and float(reference_quality) >= 0.75:
            good.append("生成视频的清晰度和纹理统计与参考视频接近。")
        if sharpness < 0.70:
            bad.append("边缘清晰度偏弱，近景中可能出现糊边或细节软化。")
            improvements.append(
                "优先恢复人脸五官、发丝边缘和服装褶皱的局部清晰度，避免对整幅画面统一锐化。"
            )
        if high_frequency < 0.70:
            bad.append("皮肤、头发或织物的高频纹理不足，画面可能偏平滑。")
            improvements.append(
                "减少过度磨皮和降噪，针对发丝、织物和皮肤纹理增加局部细节恢复。"
            )
        if reference_quality is not None and float(reference_quality) < 0.70:
            bad.append("生成视频与参考视频的清晰度或纹理层次差异较明显。")
            improvements.append(
                "统一生成与参考视频的分辨率、曝光和压缩条件，再比较相同主体区域的细节层次。"
            )
        if not improvements:
            improvements.append("保持当前细节处理，重点复核运动、暗部和人物快速转动时的纹理稳定性。")
        impact.append("细节软化会降低皮肤、头发和服装材质的真实感。")

    elif metric.name == "表情/文本准确性":
        expression_score = details.get("expression_score")
        qa_scores = details.get("qa_scores", [])
        category_scores: Dict[str, List[float]] = {}
        for item in qa_scores:
            item_score = item.get("score")
            if item_score is None:
                continue
            category_scores.setdefault(item.get("category", "general"), []).append(
                float(item_score)
            )
        category_mean = {
            category: float(np.mean(values))
            for category, values in category_scores.items()
            if values
        }
        text_score = category_mean.get("text")
        expression_qa_score = category_mean.get("expression")

        if text_score is not None and text_score >= 0.75:
            good.append("提示词相关问题的完成度较高，主体、动作和画面要求基本落地。")
        if expression_qa_score is not None and expression_qa_score >= 0.75:
            good.append("表情和动作相关问题得分较高，人物表演变化较自然。")
        if expression_score is not None and float(expression_score) >= 0.75:
            good.append("面部关键点变化与目标表演的时间趋势较一致。")
        if text_score is not None and text_score < 0.65:
            bad.append("提示词相关要求完成不充分，可能存在动作、镜头或风格遗漏。")
            improvements.append(
                "把提示词拆成主体、动作、镜头和风格四类可观察要求，逐项确认视频中是否真的出现。"
            )
        if expression_qa_score is not None and expression_qa_score < 0.65:
            bad.append("表情或动作变化与目标意图不完全一致。")
            improvements.append(
                "明确眼神方向、眉眼变化、嘴角动作和动作速度，减少一句话中多个表演要求互相冲突。"
            )
        if expression_score is not None and float(expression_score) < 0.70:
            bad.append("面部关键点在时间上的变化偏离目标表演。")
            improvements.append(
                "重点修复眼睛、眉毛、嘴角和呼吸起伏的时间曲线，避免只修单帧外观。"
            )
        if text_score is None and expression_score is None:
            bad.append("当前缺少足够的提示词语义或表情变化证据。")
            improvements.append(
                "补充明确的动作/表情目标，并确保参考视频覆盖完整的表演过程。"
            )
        if not improvements:
            improvements.append("保持当前表演节奏，继续验证大幅转头、遮挡和情绪变化片段。")
        impact.append("表情或提示词执行偏差会让人物行为与预期不一致。")

    elif metric.name == "时间稳定性":
        warp_score = float(details.get("warp_score", 0.0) or 0.0)
        identity_score = float(details.get("identity_temporal_score", 0.0) or 0.0)
        landmark_score = float(details.get("landmark_jitter_score", 0.0) or 0.0)
        if warp_score >= 0.80:
            good.append("相邻帧的整体运动衔接较顺，未见明显重影。")
        if identity_score >= 0.80:
            good.append("人物身份特征随时间变化平稳。")
        if landmark_score >= 0.80:
            good.append("面部关键点抖动较小，表情过渡较连续。")
        if warp_score and warp_score < 0.80:
            bad.append("相邻帧的整体对齐误差偏大，存在局部跳动或重影风险。")
            improvements.append("定位光流误差最高的时间片段，优先修复运动边缘、手部和脸部轮廓的跨帧错位。")
        if identity_score and identity_score < 0.80:
            bad.append("身份特征随时间波动，可能出现脸型或五官突然变化。")
            improvements.append("降低相邻帧之间的身份变化幅度，重点固定眼睛间距、鼻梁和下颌轮廓。")
        if landmark_score and landmark_score < 0.80:
            bad.append("面部关键点抖动偏大，表情过渡可能不连贯。")
            improvements.append("对眉眼、嘴角和下巴区域增加时序平滑，避免表情变化出现突然跳变。")
        if not improvements:
            improvements.append("保持当前运动节奏，并重点复核镜头切换、快速动作和暗部片段。")
        impact.append("时序不稳定会造成闪烁、重影和动作不连续，降低观看流畅度。")

    elif metric.name == "美学":
        exposure = float(details.get("exposure_score", 0.0) or 0.0)
        clipping = float(details.get("clipping_score", 0.0) or 0.0)
        contrast = float(details.get("contrast_score", 0.0) or 0.0)
        sharpness = float(details.get("sharpness_score", 0.0) or 0.0)
        composition = details.get("composition_score")
        temporal = float(details.get("temporal_consistency_score", 0.0) or 0.0)

        if exposure >= 0.75:
            good.append("主体曝光和明暗层次较自然。")
        if clipping >= 0.85:
            good.append("高光和暗部细节保留较完整。")
        if contrast >= 0.70:
            good.append("主体与背景具有较好的明暗区分度。")
        if composition is not None and float(composition) >= 0.75:
            good.append("主体位置和画面占比较稳定。")
        if exposure < 0.75:
            bad.append("曝光层次偏弱，主体可能过暗、过亮或缺少明暗过渡。")
            improvements.append("先调整主体曝光和背景亮度，再处理色彩，避免用整体提亮掩盖脸部细节。")
        if clipping < 0.85:
            bad.append("高光或暗部存在细节丢失。")
            improvements.append("压低过曝区域并抬升关键暗部，优先保护脸部、头发和服装纹理。")
        if contrast < 0.70:
            bad.append("主体与背景的对比度不足，画面容易发灰或主体不突出。")
            improvements.append("拉开主体与背景的亮度或色彩差异，同时避免黑背景被压成无层次纯黑。")
        if sharpness < 0.70:
            bad.append("画面边缘和镜头质感偏弱。")
            improvements.append("只增强主体边缘和关键材质，不要对噪声、压缩块或背景统一锐化。")
        if composition is not None and float(composition) < 0.75:
            bad.append("主体位置或画面占比在时间上变化较大。")
            improvements.append("固定主体中心、头顶留白和人物占比，减少镜头漂移带来的构图不稳。")
        if temporal < 0.80:
            bad.append("亮度或清晰度存在帧间变化，可能产生轻微闪烁。")
            improvements.append("统一相邻帧的曝光和锐度，重点检查灯光变化与背景区域。")
        if not improvements:
            improvements.append("保持当前构图和曝光风格，继续检查不同动作幅度下的主体突出度。")
        impact.append("美学问题主要影响第一眼观感、主体突出程度和镜头质感。")

    elif metric.name == "人脸表情与肌肉运动":
        geometry_score = float(details.get("geometry_score", 0.0) or 0.0)
        gaze_score = details.get("gaze_score")
        wrinkle_score = float(details.get("wrinkle_score", 0.0) or 0.0)
        motion_score = float(details.get("motion_score", 0.0) or 0.0)
        coverage = float(details.get("face_coverage", 0.0) or 0.0)
        low_reference_actions = details.get("low_reference_actions", [])

        if geometry_score >= 0.75:
            good.append("眉眼、嘴部和下颌的肌肉形变与 Expression 动作原型较一致。")
        if gaze_score is not None and float(gaze_score) >= 0.75:
            good.append("眼神/虹膜位置变化与参考表情较一致。")
        if wrinkle_score >= 0.75:
            good.append("眼周、脸颊、嘴周和下颌的局部皱纹/高频纹理保留较好。")
        if motion_score >= 0.75:
            good.append("连续帧中肌肉形变和皱纹细节运动较平滑，闪烁风险较低。")

        if coverage < 0.90:
            bad.append(f"仅有 {coverage * 100.0:.1f}% 的采样帧形成有效人脸专项匹配。")
            improvements.append("修复侧脸、快速运动和遮挡片段中的人脸检测与细节保持。")
        if geometry_score < 0.70:
            bad.append("肌肉形变与参考动作原型偏差较大，可能存在表情僵硬或五官漂移。")
            improvements.append("重点检查眉毛、眼睑、嘴角和下颌的动作幅度与过渡曲线。")
        if gaze_score is not None and float(gaze_score) < 0.70:
            bad.append("眼神方向或虹膜位置与参考动作不一致。")
            improvements.append("修复视线方向、瞳孔位置和眼睑开合的同步关系。")
        if wrinkle_score < 0.70:
            bad.append("肌肉形变对应的皱纹和局部皮肤高频细节不足或不稳定。")
            improvements.append("减少过度磨皮和局部闪烁，恢复眼周、鼻唇沟与嘴周纹理。")
        if motion_score < 0.70:
            bad.append("肌肉运动与局部皱纹变化不同步，可能出现冻结纹理或纹理闪烁。")
            improvements.append("让皱纹细节随眉眼、脸颊和嘴部形变连续变化，而不是逐帧重绘。")
        if low_reference_actions:
            action_labels = "、".join(
                item.get("label", item.get("action", "参考表情"))
                for item in low_reference_actions[:3]
            )
            bad.append(
                f"需要关注的参考表情包括：{action_labels}；"
                "这些项目的匹配度低于质量阈值，但不等同于动作缺失。"
            )
            action_improvements = []
            for item in low_reference_actions[:2]:
                action_improvements.append(
                    f"{item.get('label', '参考表情')}："
                    f"{item.get('advice', '复核局部表情细节。')}"
                )
            improvements = action_improvements + improvements
        impact.append("人脸表情和肌肉纹理失真会直接降低人物真实感、眼神可信度和表演连续性。")

    if not good:
        if score >= 0.65:
            good.append("各项子指标整体处于可接受范围，没有明显短板。")
        else:
            good.append("当前没有达到优势阈值的单项证据。")
    if not bad:
        bad.append("本项未发现明确的低分证据。")
    if not improvements:
        improvements.append("保持当前表现，继续抽查开头、中段和结尾的质量一致性。")

    return {
        "结论": common_conclusion,
        "好在哪里": unique_texts(good, 3),
        "不好在哪里": unique_texts(bad, 3),
        "可能影响": unique_texts(impact, 2),
        "后续改进方向": unique_texts(improvements, 3),
    }


def build_overall_analysis(
        metrics: Sequence[MetricResult],
        overall_score: Optional[float],
) -> Dict[str, Any]:
    """生成总分对应的中文总结。"""

    available = [metric for metric in metrics if metric.score is not None]
    if overall_score is None or not available:
        return {
            "结论": "没有足够的有效分数生成总评。",
            "主要优点": ["当前没有可验证的整体优势。"],
            "主要问题": ["有效评分覆盖不足。"],
            "可能影响": ["总分不能代表完整的视频质量。"],
            "后续改进方向": ["补充缺失指标对应的参考素材，并确认视频主体在主要片段中持续可见。"],
        }

    ranked = sorted(available, key=lambda metric: metric.score or 0.0)
    strongest = sorted(
        available,
        key=lambda metric: metric.score or 0.0,
        reverse=True,
    )
    analyses = {
        metric.name: build_metric_analysis(metric)
        for metric in available
    }
    strengths: List[str] = [
        f"{strongest[0].name}得分最高，为 {strongest[0].score * 100:.2f} 分。"
    ]
    problems: List[str] = [
        f"{ranked[0].name}得分最低，为 {ranked[0].score * 100:.2f} 分。"
    ]
    impacts: List[str] = []
    improvements: List[str] = []

    for metric in strongest[:3]:
        strengths.extend(analyses[metric.name]["好在哪里"][:1])
    for metric in ranked[:3]:
        problems.extend(analyses[metric.name]["不好在哪里"][:1])
        impacts.extend(analyses[metric.name]["可能影响"][:1])
    # 综合建议按每个指标各取一条，保证五项指标都有对应的可执行动作，
    # 同时避免最低分项的多条模板建议挤占全部位置。
    for metric in ranked:
        metric_improvements = analyses[metric.name]["后续改进方向"]
        if metric_improvements:
            improvements.append(
                f"{metric.name}：{metric_improvements[0]}"
            )

    return {
        "结论": f"综合得分为 {overall_score:.2f} 分，整体表现{score_level(overall_score / 100.0)}。",
        "主要优点": unique_texts(strengths, 4),
        "主要问题": unique_texts(problems, 5),
        "可能影响": unique_texts(impacts, 3),
        "后续改进方向": unique_texts(improvements, 5),
    }


def metric_to_json(metric: MetricResult) -> Dict[str, Any]:
    """只输出用户需要的分数和中文分析，不暴露底层中间量。"""

    return {
        "项目": metric.name,
        "分数": (
            round(metric.score * 100.0, 2)
            if metric.score is not None
            else None
        ),
        "权重": round(metric.weight * 100.0, 2),
        "状态": metric_status_label(metric.status),
        "分析": build_metric_analysis(metric),
    }


def aggregate_metrics(
        metrics: Sequence[MetricResult],
        renormalize_missing: bool = True,
) -> Dict[str, Any]:
    """按可用项聚合总分，并报告实际覆盖权重。"""

    available = [metric for metric in metrics if metric.score is not None]
    if not available:
        return {
            "overall_score_100": None,
            "available_weight": 0.0,
            "coverage_percent": 0.0,
            "warning": "没有任何可用指标",
        }

    numerator = sum(
        float(metric.score) * float(metric.weight)
        for metric in available
    )
    available_weight = sum(float(metric.weight) for metric in available)
    denominator = available_weight if renormalize_missing else 1.0
    overall = clamp(numerator / denominator) * 100.0
    return {
        "overall_score_100": overall,
        "available_weight": available_weight,
        "coverage_percent": available_weight * 100.0,
        "renormalize_missing": renormalize_missing,
        "unavailable_metrics": [
            metric.name for metric in metrics if metric.score is None
        ],
    }


def json_safe(value: Any) -> Any:
    """将 NumPy、Path 和 NaN 转成可序列化格式。"""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if np is not None and isinstance(value, np.generic):
        return json_safe(value.item())
    if np is not None and isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def save_json(payload: Dict[str, Any], path: str) -> None:
    """保存 UTF-8 中文 JSON 报告。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            json_safe(payload),
            handle,
            ensure_ascii=False,
            indent=2,
        )


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    """执行完整评估流程。"""

    require_basic_dependencies()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    gen_video_path = resolve_project_path(args.gen_video)
    ref_video_path = resolve_project_path(args.ref_video)
    qa_json_path = resolve_project_path(args.qa_json)
    expression_dir = resolve_project_path(
        getattr(args, "expression_dir", "Expression")
    )
    qwen_model_path = resolve_project_path(args.qwen_model_path)
    qwen_output_path = resolve_project_path(getattr(args, "qwen_output", None))
    output_path = resolve_project_path(args.output)

    person_image_inputs = args.person_image
    if isinstance(person_image_inputs, (str, Path)):
        person_image_inputs = [str(person_image_inputs)]
    person_image_inputs = [
        str(resolve_project_path(path))
        for path in (person_image_inputs or [])
    ]

    if gen_video_path is None or output_path is None:
        raise ValueError("生成视频路径和报告输出路径不能为空")

    # Keep all downstream components on the same absolute paths, regardless of
    # the directory from which main.py or server.py was launched.
    args.gen_video = str(gen_video_path)
    args.ref_video = str(ref_video_path) if ref_video_path is not None else None
    args.person_image = person_image_inputs
    args.qa_json = str(qa_json_path) if qa_json_path is not None else None
    args.expression_dir = (
        str(expression_dir) if expression_dir is not None else None
    )
    args.qwen_model_path = (
        str(qwen_model_path) if qwen_model_path is not None else None
    )
    args.qwen_output = (
        str(qwen_output_path) if qwen_output_path is not None else None
    )
    args.output = str(output_path)

    gen_samples = read_video_samples(args.gen_video, args.max_frames)
    ref_samples = (
        read_video_samples(args.ref_video, args.max_frames)
        if args.ref_video
        else None
    )
    person_image_paths = resolve_person_image_paths(args.person_image)
    person_images = read_images(person_image_paths)

    face_analyzer = get_cached_face_analyzer(
        device=args.device,
        model_name=args.face_model,
    )
    gen_faces = face_analyzer.analyze_frames(gen_samples.frames)
    ref_faces = (
        face_analyzer.analyze_frames(ref_samples.frames)
        if ref_samples is not None
        else None
    )

    landmark_analyzer = LandmarkAnalyzer()
    try:
        gen_landmarks = landmark_analyzer.extract_frames(gen_samples.frames)
        ref_landmarks = (
            landmark_analyzer.extract_frames(ref_samples.frames)
            if ref_samples is not None
            else None
        )
        expression_references = load_expression_reference_descriptors(
            expression_dir=expression_dir,
            face_analyzer=face_analyzer,
            landmark_analyzer=landmark_analyzer,
        )
    finally:
        landmark_analyzer.close()

    # 外部 QA 分数（如果有）
    qa_score = load_external_qa_score(args.qa_json)

    # Qwen3-VL 评估 - 始终重新运行模型，不加载已有结果
    qwen_qa_result = None
    qwen_error = None
    if args.use_qwen:
        try:
            if not QWEN_AVAILABLE:
                qwen_error = (
                    "Qwen3-VL 依赖不可用，请检查 transformers、torch 和 pillow。"
                )
                LOGGER.warning(qwen_error)
            elif not args.qwen_model_path:
                qwen_error = "未指定 Qwen 模型路径。"
                LOGGER.warning(qwen_error)
            else:
                LOGGER.info("开始运行 Qwen3-VL 模型评估...")
                qwen_scorer = QwenVLScorer(
                    model_path=args.qwen_model_path,
                    device=args.device,
                    max_frames=args.max_frames,
                    qa_json_path=args.qa_json,
                )
                qwen_score, qwen_results = qwen_scorer.evaluate_all_questions(
                    gen_samples.frames,
                    person_images=person_images,
                    reference_frames=(
                        ref_samples.frames
                        if ref_samples is not None
                        else None
                    ),
                    prompt=args.prompt,
                )
                qwen_qa_result = (qwen_score, qwen_results)

                # 保存 Qwen 评估结果
                if args.qwen_output:
                    qwen_scorer.save_qa_results(qwen_results, args.qwen_output)
                LOGGER.info("Qwen3-VL 评估完成，平均分: %.4f", qwen_score)

        except Exception as exc:
            qwen_error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("Qwen3-VL 评估失败: %s", qwen_error)
            import traceback
            LOGGER.debug(traceback.format_exc())

    identity_metric = compute_identity_metric(
        gen_samples=gen_samples,
        gen_faces=gen_faces,
        person_images=person_images,
        person_image_paths=[str(path) for path in person_image_paths],
        face_analyzer=face_analyzer,
        face_sim_low=args.face_sim_low,
        face_sim_high=args.face_sim_high,
    )
    detail_metric = compute_detail_metric_extension(
        generated_video=gen_samples,
        reference_video=ref_samples,
        reference_images=person_images,
        max_frames=args.max_frames,
    )
    expression_text_metric = compute_expression_text_metric(
        gen_samples=gen_samples,
        gen_faces=gen_faces,
        gen_landmarks=gen_landmarks,
        ref_samples=ref_samples,
        ref_landmarks=ref_landmarks,
        prompt=args.prompt,
        qa_score=qa_score,
        qwen_qa_result=qwen_qa_result,
    )
    face_expression_metric = compute_face_expression_metric_extension(
        generated_video=gen_samples,
        reference_video=ref_samples,
        reference_images=person_images,
        max_frames=args.max_frames,
    )

    temporal_metric = compute_temporal_metric(
        gen_samples=gen_samples,
        gen_faces=gen_faces,
        gen_landmarks=gen_landmarks,
        ref_samples=ref_samples,
        ref_faces=ref_faces,
        ref_landmarks=ref_landmarks,
        face_sim_low=args.face_sim_low,
        face_sim_high=args.face_sim_high,
    )
    aesthetic_metric = compute_aesthetic_metric(
        samples=gen_samples,
        faces=gen_faces,
    )

    metrics = [
        identity_metric,
        detail_metric,
        expression_text_metric,
        face_expression_metric,
        temporal_metric,
        aesthetic_metric,
    ]
    quality_metrics = [
        metric
        for metric in metrics
        if metric.name != "人脸表情与肌肉运动"
    ]
    aggregation = aggregate_metrics(
        quality_metrics,
        renormalize_missing=not args.no_renormalize_missing,
    )

    face_detector_label = (
        "OpenCV Haar 人脸检测（代理）"
        if face_analyzer.detector_name == "opencv-haar-proxy"
        else face_analyzer.detector_name
    )
    landmark_backend_label = (
        "未启用"
        if landmark_analyzer.backend == "none"
        else "MediaPipe 人脸关键点"
    )

    # 构建 Qwen 结果摘要
    qwen_summary = None
    if qwen_qa_result is not None:
        score, results = qwen_qa_result
        qwen_summary = {
            "average_score": score,
            "question_count": len(results),
            "results": results,
        }

    try:
        try:
            from video_pred.predict_real_video import predict_real_video
        except ModuleNotFoundError:
            # On-disk package is historically named ``vedio_pred``.
            from vedio_pred.predict_real_video import predict_real_video

        authenticity = predict_real_video(args.gen_video)
        authenticity["模型状态"] = "已启用"
        LOGGER.info(
            "真实/生成模型调用成功: %s",
            authenticity.get("模型路径"),
        )
    except Exception as exc:
        LOGGER.exception("真实/生成模型调用失败")
        authenticity = {
            "预测": "uncertain",
            "标签": "真伪模型不可用",
            "生成概率": None,
            "真实概率": None,
            "证据强度": 0.0,
            "结论": "当前视频无法完成真实/生成概率判断。",
            "证据": [],
            "方法": "video_pred/predict_real_video.py",
            "说明": f"{type(exc).__name__}: {exc}",
            "模型状态": "不可用",
        }
    authenticity["人脸表情与肌肉运动"] = metric_to_json(
        face_expression_metric
    )
    authenticity["人脸专项：表情、眼神与肌肉皱纹"] = json_safe(
        face_expression_metric.details
    )
    overall_analysis = build_overall_analysis(
        quality_metrics,
        aggregation["overall_score_100"],
    )
    overall_analysis["视频真伪判断"] = authenticity.get("结论")

    report = {
        "face_expression_evaluation": json_safe(
            face_expression_metric.details
        ),
        "视频真伪判断": authenticity,
        "版本": "3.0",
        "输入": {
            "生成视频": args.gen_video,
            "人物参考图": args.person_image,
            "人物参考图展开结果": [
                str(path) for path in person_image_paths
            ],
            "参考视频": args.ref_video,
            "提示词": args.prompt,
            "外部问答结果": args.qa_json,
            "Qwen模型路径": args.qwen_model_path,
        },
        "视频信息": {
            "生成视频": {
                "帧率": gen_samples.fps,
                "总帧数": gen_samples.frame_count,
                "采样帧数": len(gen_samples.frames),
                "宽度": gen_samples.width,
                "高度": gen_samples.height,
                "时长秒数": gen_samples.duration_seconds,
            },
            "参考视频": (
                {
                    "帧率": ref_samples.fps,
                    "总帧数": ref_samples.frame_count,
                    "采样帧数": len(ref_samples.frames),
                    "宽度": ref_samples.width,
                    "高度": ref_samples.height,
                    "时长秒数": ref_samples.duration_seconds,
                }
                if ref_samples is not None
                else None
            ),
        },
        "权重": {
            "角色一致性": WEIGHTS["identity"] * 100.0,
            "质感和细节": WEIGHTS["detail"] * 100.0,
            "表情/文本准确性": WEIGHTS["expression_text"] * 100.0,
            "时间稳定性": WEIGHTS["temporal"] * 100.0,
            "美学": WEIGHTS["aesthetic"] * 100.0,
        },
        "评分": [
            metric_to_json(metric)
            for metric in quality_metrics
        ],
        "总分": (
            round(aggregation["overall_score_100"], 2)
            if aggregation["overall_score_100"] is not None
            else None
        ),
        "有效评分覆盖率": round(aggregation["coverage_percent"], 2),
        "总分析": overall_analysis,
        "评估组成": {
            "视频真实/生成分析": authenticity,
            "质量评分项目": [
                metric_to_json(metric)
                for metric in quality_metrics
            ],
        },
        "Qwen评估结果": qwen_summary,
        "运行信息": {
            "人脸分析器": face_detector_label,
            "Qwen模型": (
                "已启用"
                if qwen_qa_result is not None
                else ("已关闭" if not args.use_qwen else "不可用")
            ),
            "Qwen结果文件": args.qwen_output,
            "Qwen错误": qwen_error,
            "LPIPS": (
                "已启用"
                if args.use_lpips and ref_samples is not None
                else "未启用"
            ),
            "关键点分析器": landmark_backend_label,
            "缺失评分项目": [
                metric.name
                for metric in quality_metrics
                if metric.score is None
            ],
            "真伪模型": authenticity.get("模型状态"),
        },
    }
    save_json(report, args.output)
    return report


def print_summary(report: Dict[str, Any]) -> None:
    """在终端打印简洁结果。"""

    print(f"总分: {report['总分']:.2f}/100"
          if report["总分"] is not None
          else "总分: N/A")
    print(
        f"可用权重: {report['有效评分覆盖率']:.1f}%"
        + (
            "（缺失项已重新归一化）"
            if report["有效评分覆盖率"] < 100.0
            else ""
        )
    )

    # 显示 Qwen 评估结果
    qwen_summary = report.get("Qwen评估结果")
    if qwen_summary:
        print(f"Qwen3-VL 评估平均分: {qwen_summary['average_score']:.4f}")

    for metric in report["评分"]:
        score = metric["分数"]
        score_text = f"{score:.2f}" if score is not None else "N/A"
        print(
            f"- {metric['项目']}: {score_text}/100"
            f" | 状态={metric['状态']}"
            f" | 权重={metric['权重']:.0f}%"
        )
    print("生成视频: " + str(report["输入"].get("生成视频")))


def main() -> int:
    """命令行入口。"""

    # Windows 终端可能默认使用本地代码页，主动切换可避免中文摘要乱码。
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = evaluate(args)
        print_summary(report)
        print(f"评估结果已保存到: {args.output}")
        return 0
    except Exception as exc:
        LOGGER.error("%s", exc)
        import traceback
        LOGGER.error(traceback.format_exc())
        return 1


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser(description="评估视频生成模型结果，输出 JSON 评分报告")

    # 输入文件
    parser.add_argument("--gen-video", default="input/generated.mp4", help="生成视频路径，必填")
    parser.add_argument("--person-image", default=["input/ref_image"], action="append", metavar="PATH",
                        help="用于多图ArcFace角色一致性")
    parser.add_argument("--ref-video", default=None, help="参考/GT 视频，可选；用于表情、细节和运动稳定性比较")
    parser.add_argument("--prompt",
                        default="人物外貌与服装严格参考@图片1 @图片2 @图片3 。保持五官清晰、面部稳定不变形、人体结构正常、服装发型全程一致。黑色背景。人物自然，轻微呼吸起伏，人物眼神变化遵循@视频1 中人物的眼神，人物面部肌肉运动遵循@视频1 中人物的面部运动变化。动作缓慢轻柔不僵硬。人物整体动作表演遵循@视频1 中人物的动态动作变化。人物表演严格参考使用@视频1 中的语音，保持人物处于画面中心位置。稳定保持@图片1 中人物在镜头中的画面占比不变。画面丝滑流畅。电影质感镜头，无模糊无重影无闪烁。",
                        help="文本 prompt，可选")
    parser.add_argument("--qa-json", default="vedio_qa.json",
                        help="QA 问题 JSON 文件，包含 questions 列表，每个问题包含 question 字段，score 可选")

    # Qwen 模型相关（替代 CLIP）
    parser.add_argument("--use-qwen", dest="use_qwen", default=True, help="启用 Qwen3-VL 模型评估，默认启用")
    parser.add_argument("--qwen-model-path", default="checkpoints/Qwen3-VL-8B-Instruct")
    parser.add_argument("--qwen-output", default="qwen_evaluation.json",
                        help="Qwen 评估结果输出路径，默认 qwen_evaluation.json。 如果文件已存在，将直接加载而不重新运行模型")

    # 其他模型选项
    parser.add_argument("--use-lpips", action="store_true", help="有参考视频时启用 LPIPS，需要 lpips/torch/torchvision")
    parser.add_argument("--face-model", default="buffalo_l", help="InsightFace 模型名，默认 buffalo_l")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "gpu"], default="cuda", help="推理设备，默认 auto")

    # 评估参数
    parser.add_argument("--max-frames", type=int, default=32, help="每个视频最多均匀采样帧数，默认 32")
    parser.add_argument("--face-sim-low", type=float, default=0.25, help="ArcFace 相似度标定下限，默认 0.25")
    parser.add_argument("--face-sim-high", type=float, default=0.65, help="ArcFace 相似度标定上限，默认 0.65")
    parser.add_argument("--no-renormalize-missing", action="store_true", help="缺失指标不重新归一化，缺失项按 0 分计入总分")

    # 输出
    parser.add_argument("--output", default="evaluation_result.json", help="输出 JSON 报告路径，默认 evaluation_result.json")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")

    return parser


if __name__ == "__main__":
    sys.exit(main())
