# -*- coding: utf-8 -*-
"""质感/细节与人脸表情/肌肉运动的独立评分入口。

当前版本的两个公开函数先返回固定占位结果，不调用 ``main.py`` 中的旧实现。
后续修改算法时，直接在本文件的两个公开函数中替换占位逻辑即可。

两项指标的原有计算原理：

1. ``质感和细节``：
   - 对人脸区域或整帧计算 Laplacian 方差，作为清晰度/边缘细节信号；
   - 计算高频能量与整体亮度变化的比例，作为纹理细节代理；
   - 有参考视频时比较生成视频与参考视频的统计量，必要时再融合 LPIPS。
2. ``人脸表情与肌肉运动``：
   - 将生成视频帧与 Expression 目录中的表情原型进行匹配；
   - 优先使用 MediaPipe Face Mesh 的肌肉几何和虹膜位置；
   - 缺少关键点时使用 OpenCV 分区特征作为代理；
   - 再根据连续帧中的几何变化、皱纹高频变化和眼神变化计算时序同步性。

``max_frames`` 用于限制本模块实际参与评分的最多帧数。正常情况下
``main.py`` 已经使用同一个参数完成视频采样，因此不会产生二次降采样。
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


LegacyMetricImplementation = Callable[..., Any]


def _validate_max_frames(max_frames: Optional[int]) -> Optional[int]:
    """校验最大采样帧数。

    ``None`` 表示不在本模块内再次限制帧数；整数表示最多保留的帧数。
    至少保留 2 帧，才能计算帧间清晰度变化、表情变化和时序连续性。
    """

    if max_frames is None:
        return None
    value = int(max_frames)
    if value < 2:
        raise ValueError("max_frames 至少需要为 2")
    return value


def _uniform_sample_indices(length: int, max_frames: Optional[int]) -> List[int]:
    """在完整时间轴上均匀选择帧索引。

    选择首帧和末帧，并将中间帧按等间隔分布，避免只取视频开头导致
    表情动作后半段或细节变化完全没有参与评分。
    """

    limit = _validate_max_frames(max_frames)
    if length <= 0:
        return []
    if limit is None or length <= limit:
        return list(range(length))
    if limit == 2:
        return [0, length - 1]
    return [
        int(round(index * (length - 1) / (limit - 1)))
        for index in range(limit)
    ]


def _select_sequence(
    values: Optional[Sequence[Any]],
    indices: Sequence[int],
) -> Optional[List[Any]]:
    """按照同一组帧索引同步选择检测结果或关键点序列。"""

    if values is None:
        return None
    return [values[index] for index in indices if index < len(values)]


def _select_video_samples(
    samples: Any,
    indices: Sequence[int],
) -> Any:
    """同步裁剪 VideoSamples 的 frames 和 indices。

    ``VideoSamples`` 是 ``main.py`` 中的 dataclass。保留原始
    ``frame_count``、``fps``、宽高等视频元数据，只替换实际参与指标计算的
    采样帧和原视频帧索引。
    """

    if samples is None or not indices:
        return samples
    if not is_dataclass(samples):
        # 非项目内 VideoSamples 对象无法安全复制，交给调用方原样处理。
        return samples
    return replace(
        samples,
        frames=[samples.frames[index] for index in indices],
        indices=[samples.indices[index] for index in indices],
    )


def _limit_video_inputs(
    samples: Any,
    observations: Sequence[Any],
    max_frames: Optional[int],
) -> Tuple[Any, Sequence[Any], List[int]]:
    """对视频帧和对应人脸结果执行一致的最大帧数限制。"""

    limit = _validate_max_frames(max_frames)
    frame_count = len(getattr(samples, "frames", []) or [])
    indices = _uniform_sample_indices(frame_count, limit)
    if len(indices) == frame_count:
        return samples, observations, indices
    return (
        _select_video_samples(samples, indices),
        _select_sequence(observations, indices) or [],
        indices,
    )


def _legacy_compute_detail_metric(
    *,
    gen_samples: Any,
    gen_faces: Sequence[Any],
    ref_samples: Optional[Any],
    ref_faces: Optional[Sequence[Any]],
    use_lpips: bool,
    device: str,
    legacy_implementation: LegacyMetricImplementation,
    max_frames: Optional[int] = None,
) -> Any:
    """计算“质感和细节”指标。

    参数含义：
    - ``gen_samples``：生成视频的采样结果，包含帧、原视频索引和视频元数据。
    - ``gen_faces``：生成视频每个采样帧的人脸检测结果。
    - ``ref_samples``：可选参考视频采样结果，用于清晰度和高频分布比较。
    - ``ref_faces``：参考视频对应的人脸检测结果。
    - ``use_lpips``：是否允许旧算法加载 LPIPS 作为辅助相似度证据。
    - ``device``：LPIPS 使用的设备，例如 ``cuda``、``cpu`` 或 ``auto``。
    - ``legacy_implementation``：当前阶段使用的 main.py 原有算法实现。
    - ``max_frames``：本模块最多使用的帧数；超过时沿时间轴均匀抽样。

    当前阶段的得分仍由 ``legacy_implementation`` 计算，因此不会改变原有
    清晰度、高频纹理、参考视频和 LPIPS 的融合方式。
    """

    gen_samples, gen_faces, _ = _limit_video_inputs(
        gen_samples,
        gen_faces,
        max_frames,
    )
    if ref_samples is not None:
        ref_indices = _uniform_sample_indices(
            len(getattr(ref_samples, "frames", []) or []),
            max_frames,
        )
        ref_samples = _select_video_samples(ref_samples, ref_indices)
        if ref_faces is not None:
            ref_faces = _select_sequence(ref_faces, ref_indices)

    return legacy_implementation(
        gen_samples=gen_samples,
        gen_faces=gen_faces,
        ref_samples=ref_samples,
        ref_faces=ref_faces,
        use_lpips=use_lpips,
        device=device,
    )


def _legacy_compute_face_expression_metric(
    *,
    gen_samples: Any,
    gen_faces: Sequence[Any],
    gen_landmarks: Sequence[Optional[Any]],
    expression_references: Sequence[Any],
    legacy_implementation: LegacyMetricImplementation,
    max_frames: Optional[int] = None,
) -> Any:
    """计算“人脸表情与肌肉运动”指标。

    参数含义：
    - ``gen_samples``：生成视频的采样结果。
    - ``gen_faces``：生成视频每个采样帧的人脸检测结果。
    - ``gen_landmarks``：每个采样帧的 MediaPipe 人脸关键点，可为空。
    - ``expression_references``：Expression 目录加载出的表情/视角原型。
    - ``legacy_implementation``：当前阶段使用的 main.py 原有算法实现。
    - ``max_frames``：本模块最多使用的帧数；会同步限制帧、人脸和关键点。

    当前阶段的原有算法先计算每帧与表情原型的匹配分数，再融合：
    - 表情动作原型匹配；
    - 最低分尾部帧表现；
    - 帧间匹配稳定性；
    - 肌肉几何、皱纹高频和眼神的时序同步；
    最后再加入有效人脸覆盖率，得到最终分数。
    """

    gen_samples, gen_faces, gen_indices = _limit_video_inputs(
        gen_samples,
        gen_faces,
        max_frames,
    )
    gen_landmarks = _select_sequence(gen_landmarks, gen_indices) or []

    return legacy_implementation(
        gen_samples=gen_samples,
        gen_faces=gen_faces,
        gen_landmarks=gen_landmarks,
        expression_references=expression_references,
    )


# ============================================================================
# 新版四参数接口
# ============================================================================
#
# 下面两个函数会覆盖上面的旧实现名称。旧实现保留并改为 _legacy_*，
# 方便后续对照算法，但 main.py 当前只调用这两个四参数占位函数。

DETAIL_PLACEHOLDER_SCORE = 0.75
FACE_EXPRESSION_PLACEHOLDER_SCORE = 0.72


@dataclass
class MetricResult:
    """与 main.py 兼容的指标结果对象。"""

    name: str
    score: Optional[float]
    weight: float
    status: str
    details: Dict[str, Any] = field(default_factory=dict)


def _video_frame_count(video: Any) -> int:
    """读取视频采样对象中的帧数；无法读取时返回 0。"""

    frames = getattr(video, "frames", None)
    if frames is None:
        return 0
    try:
        return len(frames)
    except TypeError:
        return 0


def _reference_image_count(images: Optional[Sequence[Any]]) -> int:
    """统计参考图片数量，兼容空列表、单张图片和图片序列。"""

    if images is None:
        return 0
    if isinstance(images, (str, bytes)):
        return 1
    try:
        return len(images)
    except TypeError:
        return 1


def compute_detail_metric(
    generated_video: Any,
    reference_video: Optional[Any],
    reference_images: Optional[Sequence[Any]],
    max_frames: Optional[int],
) -> MetricResult:
    """计算“质感和细节”，当前返回固定占位结果。

    参数含义：
    - ``generated_video``：生成视频采样对象，后续算法从这里读取视频帧。
    - ``reference_video``：可选参考视频采样对象，没有时传 ``None``。
    - ``reference_images``：可选参考图片序列，没有时传 ``None`` 或空列表。
    - ``max_frames``：最多参与计算的采样帧数，至少为 2。

    未来计算原理：
    先计算生成视频的人脸区域或整帧清晰度，再计算高频纹理比例；
    有参考视频或参考图片时，对齐主体区域后比较清晰度、纹理层次和细节保留。
    当前固定返回 75 分，保证 main.py 的调用、报告和界面渲染保持可用。
    """

    max_frames = _validate_max_frames(max_frames)
    generated_count = _video_frame_count(generated_video)
    reference_count = _video_frame_count(reference_video)
    image_count = _reference_image_count(reference_images)
    score = DETAIL_PLACEHOLDER_SCORE

    return MetricResult(
        name="质感和细节",
        score=score,
        weight=0.15,
        status="proxy",
        details={
            "placeholder": True,
            "method": "detail_expression_metrics 固定占位结果",
            "generated_quality_score": score,
            "sharpness_proxy_score": score,
            "high_frequency_proxy_score": score,
            "reference_quality_score": score if reference_video is not None else None,
            "paired_frames": (
                min(generated_count, reference_count)
                if reference_video is not None
                else 0
            ),
            "generated_frame_count": generated_count,
            "reference_frame_count": reference_count,
            "reference_image_count": image_count,
            "max_frames": max_frames,
            "warning": (
                "当前为固定占位分数，后续修改本模块函数后才执行真实"
                "质感和细节计算。"
            ),
        },
    )


def compute_face_expression_metric(
    generated_video: Any,
    reference_video: Optional[Any],
    reference_images: Optional[Sequence[Any]],
    max_frames: Optional[int],
) -> MetricResult:
    """计算“人脸表情与肌肉运动”，当前返回固定占位结果。

    参数含义：
    - ``generated_video``：生成视频采样对象，后续算法从这里读取连续帧。
    - ``reference_video``：可选表演参考视频采样对象，没有时传 ``None``。
    - ``reference_images``：可选人物参考图片序列，没有时传 ``None`` 或空列表。
    - ``max_frames``：最多参与计算的采样帧数，至少为 2。

    未来计算原理：
    将生成视频的人脸表情与参考视频或参考图片中的目标状态进行匹配，
    综合肌肉几何、眼神位置、局部皱纹纹理和连续帧同步性得到分数。
    当前固定返回 72 分，并保留页面需要的完整详情字段。
    """

    max_frames = _validate_max_frames(max_frames)
    generated_count = _video_frame_count(generated_video)
    reference_count = _video_frame_count(reference_video)
    image_count = _reference_image_count(reference_images)
    coverage = 1.0 if generated_count > 0 else 0.0
    score = FACE_EXPRESSION_PLACEHOLDER_SCORE

    return MetricResult(
        name="人脸表情与肌肉运动",
        score=score,
        weight=0.15,
        status="proxy",
        details={
            "placeholder": True,
            "score": score,
            "method": "detail_expression_metrics 固定占位结果",
            "reference_count": image_count,
            "valid_face_frames": generated_count,
            "total_sampled_frames": generated_count,
            "face_coverage": coverage,
            "frame_match_score": score,
            "tail_20_percent_score": score,
            "match_consistency_score": score,
            "geometry_score": score,
            "gaze_score": score,
            "texture_score": score,
            "wrinkle_score": score,
            "motion_score": score,
            "motion_details": {},
            "geometry_method": "占位结果",
            "gaze_method": "占位结果",
            "action_scores": [],
            "low_reference_actions": [],
            "group_scores": {},
            "top_frame_matches": [],
            "semantic_qa_suggestions": [],
            "generated_frame_count": generated_count,
            "reference_frame_count": reference_count,
            "reference_image_count": image_count,
            "max_frames": max_frames,
            "warning": (
                "当前为固定占位分数，后续修改本模块函数后才执行真实"
                "人脸表情和肌肉运动计算。"
            ),
            "evidence": [
                {"label": "表情动作原型匹配", "value": f"{score * 100.0:.1f}%"},
                {"label": "眼神/虹膜位置匹配", "value": f"{score * 100.0:.1f}%"},
                {"label": "肌肉几何与皱纹纹理匹配", "value": f"{score * 100.0:.1f}%"},
                {"label": "连续帧肌肉-皱纹同步", "value": f"{score * 100.0:.1f}%"},
            ],
        },
    )
