# -*- coding: utf-8 -*-
"""质感/细节与人脸表情/肌肉运动的独立评分入口。

公开接口与协作方约定的四参数签名保持一致。两个公开函数不调用旧版
``main.py`` 实现，而是调用 ``modules/core/detail_expression_runtime.py``
中的真实王兴专项评分，并读取包内 ``modules/assets/profiles``。

两项指标的计算原理：

1. ``质感和细节``：
   - 对人脸区域或整帧计算纹理、高频和光流时序特征；
   - 叠加无参考 VQA（内置 NR-VQA；可选 DOVER/FAST-VQA/RAPIQUE，不用 VMAF）；
   - 结合 forensics 画像中的真人域证据、时序稳定和细节清晰度；
   - 最终按王兴专项五维雷达口径得到综合分。
2. ``人脸表情与肌肉运动``：
   - 优先使用 AU 表格提取表情动作和连续帧运动特征；
   - 结合表情 profile 符合度、动态自然度和时序连贯性；
   - **无 AU CSV 时**：从生成视频自动合成 AU（MediaPipe blendshape/关键点），
     再对照王兴表情 profile（约 600+ 样本）评分；仅当合成失败才回退 Expression 图；
   - 完整旁路 AU 表仍是最高质量路径。

``max_frames`` 用于限制本模块实际参与评分的最多帧数。正常情况下
调用方已经使用同一个参数完成视频采样，传入带 ``frames`` 的对象时不会
重复读取视频；传入路径时则由本入口按相同上限采样。

仅有「视频地址 + 处理帧率」时可用::

    video = prepare_generated_video(r"a.mp4", sample_fps=8, max_frames=24, au_csv=r"a.csv")
    compute_detail_metric(video, None, None, 24)
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    # Supports importing the final project as ``Evaluator``.
    from .modules.core.detail_expression_runtime import (
        PreparedVideo,
        prepare_video_input,
        score_detail_quality,
        score_face_expression,
    )
except ImportError:
    # Supports direct host entrypoints: ``python app.py`` / ``python main.py``.
    from modules.core.detail_expression_runtime import (
        PreparedVideo,
        prepare_video_input,
        score_detail_quality,
        score_face_expression,
    )


LegacyMetricImplementation = Callable[..., Any]


def prepare_generated_video(
    path: str,
    *,
    sample_fps: float = 8.0,
    max_frames: int = 24,
    au_csv: str | None = None,
    expected_class: str | None = None,
) -> PreparedVideo:
    """把视频地址、处理帧率和可选 AU 路径整理成入口输入对象。

    ``max_frames`` 只控制预采样对象中的帧数；公开评分函数仍保持原有
    四参数调用方式。
    """

    payload: Dict[str, Any] = {
        "path": path,
        "sample_fps": sample_fps,
    }
    if au_csv:
        payload["au_csv"] = au_csv
    if expected_class:
        payload["expected_class"] = expected_class
    return prepare_video_input(
        payload,
        max_frames=max_frames,
        sample_fps=sample_fps,
    )


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

    当前公开四参数接口不再走此 legacy 钩子；保留供对照旧算法。
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

    当前公开四参数接口不再走此 legacy 钩子；保留供对照旧算法。
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
# 下面两个函数是协作方 / main.py 应调用的入口。旧实现保留为 _legacy_*，
# 方便对照；真实分数由 ``modules/core/detail_expression_runtime.py`` 计算。

DETAIL_PLACEHOLDER_SCORE = 0.75  # 历史占位常量，现仅作兼容保留，不参与打分
FACE_EXPRESSION_PLACEHOLDER_SCORE = 0.72


@dataclass
class MetricResult:
    """与 main.py 兼容的指标结果对象。

    ``score`` 保持 0 到 1 的内部比例；``score_0_100`` 供页面直接展示。
    """

    name: str
    score: Optional[float]
    weight: float
    status: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def score_0_100(self) -> Optional[float]:
        """百分制分数，便于页面直接展示。"""

        return None if self.score is None else float(self.score) * 100.0


def _video_frame_count(video: Any) -> int:
    """读取视频采样对象中的帧数；无法读取时返回 0。"""

    frames = getattr(video, "frames", None)
    if frames is None and isinstance(video, dict):
        frames = video.get("frames")
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


def _compat_detail_details(
    score: Optional[float],
    details: Dict[str, Any],
    *,
    reference_video: Optional[Any],
) -> Dict[str, Any]:
    """补齐原占位版 details 字段，避免对方 UI 缺 key。"""

    out = dict(details)
    out.setdefault("placeholder", False)
    out.setdefault("generated_quality_score", score)
    out.setdefault(
        "sharpness_proxy_score",
        out.get("detail_clarity_0_1", score),
    )
    out.setdefault(
        "high_frequency_proxy_score",
        out.get("texture_evidence_0_1", score),
    )
    out.setdefault(
        "reference_quality_score",
        None,
    )
    out.setdefault("reference_used", False)
    out.setdefault("generated_frame_count", out.get("generated_frame_count", 0))
    out.setdefault("reference_frame_count", out.get("reference_frame_count", 0))
    out.setdefault("reference_image_count", out.get("reference_image_count", 0))
    return out


def _compat_expression_details(
    score: Optional[float],
    details: Dict[str, Any],
) -> Dict[str, Any]:
    """补齐原占位版表情 details / evidence，避免对方 UI 缺 key。"""

    out = dict(details)
    out.setdefault("placeholder", False)
    out.setdefault("score", score)
    out.setdefault(
        "frame_match_score",
        out.get("profile_compatibility_0_1", score),
    )
    out.setdefault("tail_20_percent_score", score)
    out.setdefault(
        "match_consistency_score",
        out.get("action_coherence_0_1", score),
    )
    out.setdefault(
        "geometry_score",
        out.get("muscle_action_evidence_0_1", score),
    )
    out.setdefault("gaze_score", score)
    out.setdefault(
        "texture_score",
        out.get("muscle_action_evidence_0_1", score),
    )
    out.setdefault("wrinkle_score", out.get("texture_score", score))
    out.setdefault(
        "motion_score",
        out.get("action_coherence_0_1", score),
    )
    out.setdefault("motion_details", out.get("dimensions") or {})
    out.setdefault("geometry_method", "wangxing_specialization")
    out.setdefault("gaze_method", "wangxing_specialization")
    out.setdefault("action_scores", [])
    out.setdefault("low_reference_actions", [])
    out.setdefault("group_scores", {})
    out.setdefault("top_frame_matches", [])
    out.setdefault("semantic_qa_suggestions", [])
    if "evidence" not in out and score is not None:
        def _display_score(key: str) -> float:
            value = out.get(key)
            return float(score if value is None else value) * 100.0

        out["evidence"] = [
            {
                "label": "画像符合度",
                "value": f"{_display_score('profile_compatibility_0_1'):.1f}%",
            },
            {
                "label": "肌肉动作证据",
                "value": f"{_display_score('muscle_action_evidence_0_1'):.1f}%",
            },
            {
                "label": "动作连贯",
                "value": f"{_display_score('action_coherence_0_1'):.1f}%",
            },
            {
                "label": "关键点覆盖",
                "value": f"{_display_score('landmark_coverage_0_1'):.1f}%",
            },
        ]
    return out


def compute_detail_metric(
    generated_video: Any,
    reference_video: Optional[Any],
    reference_images: Optional[Sequence[Any]],
    max_frames: Optional[int],
) -> MetricResult:
    """计算“质感和细节”王兴专项综合分。

    参数含义：
    - ``generated_video``：生成视频路径、采样对象，或包含 ``path`` /
      ``sample_fps`` / ``au_csv`` 的字典。
    - ``reference_video``：保留的可选参考视频参数，没有时传 ``None``。
    - ``reference_images``：保留的可选参考图片序列，没有时传 ``None``。
    - ``max_frames``：最多参与计算的采样帧数，至少为 2。

    返回值：
    - ``MetricResult.score``：真实专项分数，范围为 0 到 1；
    - ``MetricResult.details``：纹理五维分项、采样信息、profile 和警告。

    当前专项按网页王兴质感雷达口径计算。参考视频和图片参数保留用于
    调用兼容及详情记录，不作为当前专项的帧级参考对齐输入。
    """

    max_frames = _validate_max_frames(max_frames)
    generated = prepare_video_input(generated_video, max_frames=max_frames)
    reference = (
        prepare_video_input(reference_video, max_frames=max_frames)
        if reference_video is not None
        else None
    )
    payload = score_detail_quality(
        generated=generated,
        reference=reference,
        reference_images=reference_images,
        max_frames=max_frames,
    )
    score = payload.get("score")
    return MetricResult(
        name="质感和细节",
        score=score,
        weight=0.15,
        status=str(payload.get("status") or "partial"),
        details=_compat_detail_details(
            score,
            dict(payload.get("details") or {}),
            reference_video=reference_video,
        ),
    )


def compute_face_expression_metric(
    generated_video: Any,
    reference_video: Optional[Any],
    reference_images: Optional[Sequence[Any]],
    max_frames: Optional[int],
) -> MetricResult:
    """计算“人脸表情与肌肉运动”王兴专项综合分。

    参数含义：
    - ``generated_video``：生成视频路径、采样对象，或包含 ``path`` /
      ``sample_fps`` / ``au_csv`` 的字典。
    - ``reference_video``：保留的可选表演参考视频参数，没有时传 ``None``。
    - ``reference_images``：保留的可选人物参考图片序列，没有时传 ``None``。
    - ``max_frames``：最多参与计算的采样帧数，至少为 2。

    有 AU CSV 时走完整专项五维；无 AU 时自动回退 ``Expression/``
    动作原型匹配并返回 ``partial``（含 UI 兼容的
    ``frame_match_score`` / ``geometry_score`` 等字段）。
    ``details.warning`` 会说明当前走了哪条路径。参考参数不参与当前专项的帧级对齐。
    """

    max_frames = _validate_max_frames(max_frames)
    generated = prepare_video_input(generated_video, max_frames=max_frames)
    reference = (
        prepare_video_input(reference_video, max_frames=max_frames)
        if reference_video is not None
        else None
    )
    payload = score_face_expression(
        generated=generated,
        reference=reference,
        reference_images=reference_images,
        max_frames=max_frames,
    )
    score = payload.get("score")
    return MetricResult(
        name="人脸表情与肌肉运动",
        score=score,
        weight=0.15,
        status=str(payload.get("status") or "partial"),
        details=_compat_expression_details(
            score,
            dict(payload.get("details") or {}),
        ),
    )
