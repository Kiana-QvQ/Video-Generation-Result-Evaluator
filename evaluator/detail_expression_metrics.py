# -*- coding: utf-8 -*-
"""质感/细节与人脸表情/肌肉运动的独立评分入口。

协作方或本仓都可通过本文件的两个公开函数拿到结果；实现已接到
``evaluator`` 包内的王兴质量旁路与 forensics 纹理/面部运动代码，并读取
``assets/profiles`` 中的画像。

公开接口保持四参数（与历史 main.py 约定兼容）：

1. ``generated_video``：生成视频路径、采样对象，或
   ``{"path": "...", "au_csv": "...", "sample_fps": 8}``
2. ``reference_video``：可选参考视频（同上）
3. ``reference_images``：可选参考图
4. ``max_frames``：最多参与计算的采样帧数（≥2）

若只有「视频地址 + 处理帧率」，可先调用 ::

    video = prepare_generated_video(r"candidate.mp4", sample_fps=8, max_frames=24)
    compute_detail_metric(video, None, None, 24)

``max_frames`` 用于限制本模块实际参与评分的最多帧数。从路径采样时还会读取
对象上的 ``sample_fps`` / ``fps``（默认 8）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .core.detail_expression_runtime import (
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
) -> PreparedVideo:
    """把视频地址（与可选 AU / 处理帧率）整理成公开函数可接收的输入。"""

    payload: Dict[str, Any] = {
        "path": path,
        "sample_fps": sample_fps,
    }
    if au_csv:
        payload["au_csv"] = au_csv
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
    """在完整时间轴上均匀选择帧索引。"""

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
    """同步裁剪 VideoSamples 的 frames 和 indices。"""

    if samples is None or not indices:
        return samples
    if not is_dataclass(samples):
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
    """计算“质感和细节”指标（旧版 legacy 钩子，保留对照）。"""

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
    """计算“人脸表情与肌肉运动”指标（旧版 legacy 钩子，保留对照）。"""

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


@dataclass
class MetricResult:
    """与协作方 / 历史 main.py 兼容的指标结果对象。"""

    name: str
    score: Optional[float]
    weight: float
    status: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def score_0_100(self) -> float | None:
        """Return the composite score in the web display scale."""
        return self.score * 100.0 if self.score is not None else None


def compute_detail_metric(
    generated_video: Any,
    reference_video: Optional[Any],
    reference_images: Optional[Sequence[Any]],
    max_frames: Optional[int],
) -> MetricResult:
    """计算“质感和细节”，调用本包 forensics / 王兴纹理实现。

    参数含义：
    - ``generated_video``：生成视频路径、采样对象，或含 ``path`` /
      ``sample_fps`` 的字典。
    - ``reference_video``：可选参考视频。
    - ``reference_images``：可选参考图片序列。
    - ``max_frames``：最多参与计算的采样帧数，至少为 2。
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
    return MetricResult(
        name="质感和细节",
        score=payload.get("score"),
        weight=0.15,
        status=str(payload.get("status") or "partial"),
        details=dict(payload.get("details") or {}),
    )


def compute_face_expression_metric(
    generated_video: Any,
    reference_video: Optional[Any],
    reference_images: Optional[Sequence[Any]],
    max_frames: Optional[int],
) -> MetricResult:
    """计算“人脸表情与肌肉运动”，调用本包王兴旁路 / forensics 实现。

    有 AU CSV 时（同名 csv、``au_csv`` 字段，或字典 ``au_csv``）会走完整
    表情肌肉链路；否则回退为视频人脸纹理/时序代理分。
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
    return MetricResult(
        name="人脸表情与肌肉运动",
        score=payload.get("score"),
        weight=0.15,
        status=str(payload.get("status") or "partial"),
        details=dict(payload.get("details") or {}),
    )
