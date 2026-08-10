"""Runtime that powers collaborator yellow-box entrypoints.

``detail_expression_metrics`` keeps a stable four-argument public API.
This module turns those inputs into calls against the packaged Wangxing /
forensics implementations and bundled profiles.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .paths import profile_path
from .video_metrics import DEFAULT_SAMPLE_FPS, probe_video, sample_video_frames


@dataclass
class PreparedVideo:
    """Normalized video input for yellow-box scoring."""

    path: str | None = None
    frames: list[Any] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    fps: float = DEFAULT_SAMPLE_FPS
    sample_fps: float = DEFAULT_SAMPLE_FPS
    frame_count: int = 0
    au_csv: str | None = None
    source: str = "unknown"


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _as_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    if isinstance(value, dict):
        for key in ("path", "video", "video_path", "generated_video"):
            if value.get(key):
                return Path(str(value[key]))
    for attribute in ("path", "video_path", "generated_video"):
        candidate = getattr(value, attribute, None)
        if candidate:
            return Path(str(candidate))
    return None


def _as_au_csv(value: Any, video_path: Path | None) -> str | None:
    if isinstance(value, dict):
        for key in ("au_csv", "au_path", "csv"):
            raw = value.get(key)
            if raw:
                path = Path(str(raw))
                if path.is_file():
                    return str(path.resolve())
    for attribute in ("au_csv", "au_path"):
        raw = getattr(value, attribute, None)
        if raw:
            path = Path(str(raw))
            if path.is_file():
                return str(path.resolve())
    if video_path is None:
        return None
    stem = video_path.with_suffix(".csv")
    if stem.is_file():
        return str(stem.resolve())
    sibling = video_path.parent / f"{video_path.stem}_au.csv"
    if sibling.is_file():
        return str(sibling.resolve())
    return None


def _sample_fps_of(value: Any, default: float = DEFAULT_SAMPLE_FPS) -> float:
    if isinstance(value, dict):
        for key in ("sample_fps", "fps", "processing_fps"):
            if value.get(key) is not None:
                try:
                    return max(0.1, float(value[key]))
                except (TypeError, ValueError):
                    pass
    for attribute in ("sample_fps", "fps", "processing_fps"):
        raw = getattr(value, attribute, None)
        if raw is not None:
            try:
                return max(0.1, float(raw))
            except (TypeError, ValueError):
                pass
    return float(default)


def prepare_video_input(
    video: Any,
    *,
    max_frames: Optional[int] = None,
    sample_fps: float | None = None,
) -> PreparedVideo:
    """Normalize a path / samples object into frames + metadata.

    Accepted ``video`` forms:
    - video path string / ``Path``
    - dict with ``path`` / ``video_path`` and optional ``au_csv`` / ``sample_fps``
    - object with ``.path`` / ``.frames`` / ``.au_csv`` / ``.sample_fps``
    """
    path = _as_path(video)
    fps = _sample_fps_of(video, sample_fps or DEFAULT_SAMPLE_FPS)
    if sample_fps is not None:
        fps = max(0.1, float(sample_fps))
    limit = 24 if max_frames is None else max(2, int(max_frames))
    au_csv = _as_au_csv(video, path if path and path.suffix else None)

    frames_attr = getattr(video, "frames", None)
    if frames_attr is None and isinstance(video, dict):
        frames_attr = video.get("frames")
    if frames_attr is not None:
        frames = list(frames_attr)
        indices_attr = getattr(video, "indices", None)
        if indices_attr is None and isinstance(video, dict):
            indices_attr = video.get("indices")
        indices = (
            [int(value) for value in indices_attr]
            if indices_attr is not None
            else list(range(len(frames)))
        )
        if len(frames) > limit:
            chosen = [
                int(round(index * (len(frames) - 1) / (limit - 1)))
                for index in range(limit)
            ]
            frames = [frames[index] for index in chosen]
            indices = [indices[index] for index in chosen if index < len(indices)]
        return PreparedVideo(
            path=str(path.resolve()) if path and path.is_file() else None,
            frames=frames,
            indices=indices,
            fps=fps,
            sample_fps=fps,
            frame_count=len(frames),
            au_csv=au_csv,
            source="preloaded_frames",
        )

    if path is None or not path.is_file():
        return PreparedVideo(
            path=str(path) if path is not None else None,
            sample_fps=fps,
            fps=fps,
            au_csv=au_csv,
            source="missing",
        )

    info, indices, _timestamps, frames = sample_video_frames(
        path,
        max_frames=limit,
        sample_fps=fps,
    )
    return PreparedVideo(
        path=str(path.resolve()),
        frames=list(frames),
        indices=[int(value) for value in indices],
        fps=float(info.get("fps") or fps),
        sample_fps=fps,
        frame_count=len(frames),
        au_csv=au_csv,
        source="sampled_from_path",
    )


def _load_forensics_profiles() -> dict[str, Any]:
    path = profile_path("forensics_profiles")
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def _load_expression_profile() -> tuple[dict[str, Any] | None, str | None]:
    path = profile_path("wangxing_expression_profile")
    if path is None or not path.is_file():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        return None, None
    return payload, str(path)


def _score_from_values(values: Sequence[float | None]) -> float | None:
    usable = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not usable:
        return None
    return _clamp(float(np.mean(usable)))


def score_detail_quality(
    *,
    generated: PreparedVideo,
    reference: PreparedVideo | None,
    reference_images: Optional[Sequence[Any]],
    max_frames: Optional[int],
) -> dict[str, Any]:
    """Run packaged texture / detail scoring."""
    from ..forensics.texture_detail import score_texture_detail
    from ..wangxing.wangxing_quality_supplement import _sample_face_texture

    limit = 24 if max_frames is None else max(2, int(max_frames))
    profiles = _load_forensics_profiles()
    texture_profile = profiles.get("texture_detail")

    texture_input: Any
    if generated.path and Path(generated.path).is_file():
        texture_input = generated.path
    elif generated.frames:
        texture_input = generated.frames
    else:
        return {
            "score": None,
            "status": "unavailable",
            "details": {
                "warning": "未提供可用的生成视频路径或帧序列。",
                "generated_frame_count": 0,
                "max_frames": max_frames,
            },
        }

    forensic = score_texture_detail(
        texture_input,
        texture_profile if isinstance(texture_profile, dict) else None,
        max_frames=limit,
        sample_fps=float(generated.sample_fps or DEFAULT_SAMPLE_FPS),
        detect_faces=True,
    )
    forensic_metrics = forensic.get("metrics") or {}
    local_texture = None
    if generated.path and Path(generated.path).is_file():
        local_texture = _sample_face_texture(
            generated.path,
            max_frames=min(limit, 24),
        )

    local_score = None
    edge_clarity = None
    if isinstance(local_texture, dict) and local_texture.get("status") == "ready":
        local_score = local_texture.get("score_0_1")
        edge_clarity = _clamp(
            0.55
            * _clamp(
                (
                    math.log1p(
                        float(local_texture.get("laplacian_variance_mean", 0.0))
                    )
                    - 3.0
                )
                / 4.0
            )
            + 0.45
            * _clamp(
                (float(local_texture.get("edge_density_mean", 0.0)) - 0.04)
                / 0.18
            )
        )

    ref_score = None
    if reference is not None and (
        (reference.path and Path(reference.path).is_file()) or reference.frames
    ):
        ref_input: Any = (
            reference.path
            if reference.path and Path(reference.path).is_file()
            else reference.frames
        )
        ref_result = score_texture_detail(
            ref_input,
            texture_profile if isinstance(texture_profile, dict) else None,
            max_frames=limit,
            sample_fps=float(reference.sample_fps or DEFAULT_SAMPLE_FPS),
        )
        ref_metrics = ref_result.get("metrics") or {}
        ref_score = ref_metrics.get("real_capture_likelihood_0_1")
        if ref_score is None:
            ref_score = ref_metrics.get("temporal_stability_proxy_0_1")

    score = _score_from_values(
        [
            forensic_metrics.get("real_capture_likelihood_0_1"),
            forensic_metrics.get("temporal_stability_proxy_0_1"),
            local_score,
            edge_clarity,
        ]
    )
    image_count = 0
    if reference_images is not None:
        if isinstance(reference_images, (str, bytes, Path)):
            image_count = 1
        else:
            try:
                image_count = len(reference_images)
            except TypeError:
                image_count = 1

    return {
        "score": score,
        "status": "available" if score is not None else "partial",
        "details": {
            "placeholder": False,
            "method": "forensics.texture_detail + wangxing face-crop texture",
            "generated_quality_score": score,
            "sharpness_proxy_score": edge_clarity if edge_clarity is not None else score,
            "high_frequency_proxy_score": local_score if local_score is not None else score,
            "reference_quality_score": ref_score,
            "paired_frames": (
                min(generated.frame_count, reference.frame_count)
                if reference is not None
                else 0
            ),
            "generated_frame_count": generated.frame_count,
            "reference_frame_count": (
                reference.frame_count if reference is not None else 0
            ),
            "reference_image_count": image_count,
            "max_frames": max_frames,
            "sample_fps": generated.sample_fps,
            "video_path": generated.path,
            "forensics": {
                "status": forensic.get("status"),
                "metrics": forensic_metrics,
            },
            "local_face_texture": (
                {
                    key: value
                    for key, value in local_texture.items()
                    if not str(key).startswith("per_frame_")
                }
                if isinstance(local_texture, dict)
                else None
            ),
        },
    }


def score_face_expression(
    *,
    generated: PreparedVideo,
    reference: PreparedVideo | None,
    reference_images: Optional[Sequence[Any]],
    max_frames: Optional[int],
) -> dict[str, Any]:
    """Run packaged facial-expression / muscle scoring."""
    from ..forensics.facial_motion import score_facial_motion
    from ..wangxing.wangxing_quality_supplement import (
        _sample_face_texture,
        evaluate_quality_supplement,
    )

    limit = 24 if max_frames is None else max(2, int(max_frames))
    profiles = _load_forensics_profiles()
    facial_profile = profiles.get("facial_motion")
    expression_profile, expression_profile_path = _load_expression_profile()

    supplement = None
    facial_forensics = None
    if generated.au_csv and Path(generated.au_csv).is_file():
        supplement = evaluate_quality_supplement(
            au_csv=generated.au_csv,
            video_path=generated.path,
            expression_profile=expression_profile,
            expression_profile_path=expression_profile_path,
            max_texture_frames=min(limit, 24),
        )
        facial_forensics = score_facial_motion(
            generated.au_csv,
            facial_profile if isinstance(facial_profile, dict) else {},
        )

    local_texture = None
    if generated.path and Path(generated.path).is_file():
        local_texture = _sample_face_texture(
            generated.path,
            max_frames=min(limit, 24),
        )
    elif generated.frames:
        from ..forensics.texture_detail import score_texture_detail

        frame_texture = score_texture_detail(
            generated.frames,
            None,
            max_frames=min(limit, len(generated.frames)),
            sample_fps=float(generated.sample_fps or 8.0),
            detect_faces=True,
        )
        metrics = frame_texture.get("metrics") or {}
        local_texture = {
            "status": "ready",
            "score_0_1": metrics.get("detail_quality_proxy_0_1")
            or metrics.get("temporal_stability_proxy_0_1"),
            "temporal_stability_0_1": metrics.get(
                "temporal_stability_proxy_0_1"
            ),
            "backend": "preloaded_frames_texture_proxy",
        }

    facial_block = (supplement or {}).get("facial_expression_muscle") or {}
    facial_metrics = facial_block.get("metrics") or {}
    forensic_metrics = (facial_forensics or {}).get("metrics") or {}

    score = _score_from_values(
        [
            facial_block.get("score_0_1"),
            forensic_metrics.get("real_capture_likelihood_0_1"),
            forensic_metrics.get("motion_coherence_0_1"),
            (local_texture or {}).get("score_0_1")
            if generated.au_csv is None
            else None,
            (local_texture or {}).get("temporal_stability_0_1")
            if generated.au_csv is None
            else None,
        ]
    )

    coverage = 1.0 if generated.frame_count > 0 else 0.0
    if facial_metrics:
        coverage = float(
            facial_metrics.get("face_coverage")
            or coverage
        )

    image_count = 0
    if reference_images is not None:
        if isinstance(reference_images, (str, bytes, Path)):
            image_count = 1
        else:
            try:
                image_count = len(reference_images)
            except TypeError:
                image_count = 1

    warning = None
    if generated.au_csv is None:
        warning = (
            "未提供 AU CSV（可传同名 .csv、video.au_csv，或 "
            "{'path': video, 'au_csv': csv}）。当前使用视频人脸纹理/"
            "时序稳定性作为表情肌肉代理分。"
        )

    evidence = []
    for key, label in (
        ("motion_prototype_match_0_1", "表情动作原型匹配"),
        ("eye_gaze_match_0_1", "眼神/虹膜位置匹配"),
        ("muscle_geometry_0_1", "肌肉几何"),
        ("wrinkle_high_frequency_0_1", "肌肉几何与皱纹纹理匹配"),
        ("muscle_wrinkle_sync_0_1", "连续帧肌肉-皱纹同步"),
    ):
        value = facial_metrics.get(key)
        if value is None:
            continue
        evidence.append(
            {"label": label, "value": f"{float(value) * 100.0:.1f}%"}
        )
    if not evidence and score is not None:
        evidence = [
            {
                "label": "视频人脸纹理/动态代理",
                "value": f"{float(score) * 100.0:.1f}%",
            }
        ]

    return {
        "score": score,
        "status": "available" if score is not None else "partial",
        "details": {
            "placeholder": False,
            "score": score,
            "method": (
                "wangxing_quality_supplement + forensics.facial_motion"
                if generated.au_csv
                else "video-only face texture / temporal proxy"
            ),
            "reference_count": image_count,
            "valid_face_frames": generated.frame_count,
            "total_sampled_frames": generated.frame_count,
            "face_coverage": coverage,
            "frame_match_score": facial_metrics.get(
                "motion_prototype_match_0_1", score
            ),
            "tail_20_percent_score": score,
            "match_consistency_score": forensic_metrics.get(
                "motion_coherence_0_1", score
            ),
            "geometry_score": facial_metrics.get("muscle_geometry_0_1", score),
            "gaze_score": facial_metrics.get("eye_gaze_match_0_1", score),
            "texture_score": facial_metrics.get(
                "wrinkle_high_frequency_0_1",
                (local_texture or {}).get("score_0_1", score),
            ),
            "wrinkle_score": facial_metrics.get(
                "wrinkle_high_frequency_0_1",
                (local_texture or {}).get("score_0_1", score),
            ),
            "motion_score": forensic_metrics.get(
                "motion_coherence_0_1", score
            ),
            "motion_details": forensic_metrics,
            "geometry_method": "AU + Face Mesh"
            if generated.au_csv
            else "video proxy",
            "gaze_method": "AU gaze columns"
            if generated.au_csv
            else "unavailable",
            "action_scores": [],
            "low_reference_actions": [],
            "group_scores": {},
            "top_frame_matches": [],
            "semantic_qa_suggestions": [],
            "generated_frame_count": generated.frame_count,
            "reference_frame_count": (
                reference.frame_count if reference is not None else 0
            ),
            "reference_image_count": image_count,
            "max_frames": max_frames,
            "sample_fps": generated.sample_fps,
            "video_path": generated.path,
            "au_csv": generated.au_csv,
            "expression_profile": expression_profile_path,
            "supplement": facial_block or None,
            "forensics": {
                "status": (facial_forensics or {}).get("status"),
                "metrics": forensic_metrics,
            },
            "warning": warning,
            "evidence": evidence,
        },
    }


def describe_probe(path: str | Path) -> dict[str, Any]:
    info = probe_video(path)
    return info.to_dict()
