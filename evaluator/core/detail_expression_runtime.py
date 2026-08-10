"""Runtime for the standalone Wang Xing specialization entrypoints.

The web UI renders two five-dimension Wang Xing radar composites. This module
reproduces those composites without importing the web application or the
ordinary five-category evaluator.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..forensics import analyze_forensics
from ..wangxing.wangxing_specialization import score_expression_profile
from .paths import profile_path
from .video_metrics import DEFAULT_SAMPLE_FPS, probe_video, sample_video_frames


@dataclass
class PreparedVideo:
    """Normalized video input for the standalone specialization API."""

    path: str | None = None
    frames: list[Any] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    fps: float = DEFAULT_SAMPLE_FPS
    sample_fps: float = DEFAULT_SAMPLE_FPS
    frame_count: int = 0
    au_csv: str | None = None
    expected_class: str | None = None
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
            if raw and Path(str(raw)).is_file():
                return str(Path(str(raw)).resolve())
    for attribute in ("au_csv", "au_path"):
        raw = getattr(value, attribute, None)
        if raw and Path(str(raw)).is_file():
            return str(Path(str(raw)).resolve())
    if video_path is None:
        return None
    for candidate in (
        video_path.with_suffix(".csv"),
        video_path.parent / f"{video_path.stem}_au.csv",
    ):
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _as_expected_class(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("expected_class", "wangxing_expected_class"):
            if value.get(key):
                return str(value[key])
    for attribute in ("expected_class", "wangxing_expected_class"):
        raw = getattr(value, attribute, None)
        if raw:
            return str(raw)
    return None


def _sample_fps_of(value: Any, default: float = DEFAULT_SAMPLE_FPS) -> float:
    keys = ("sample_fps", "fps", "processing_fps")
    values = (
        [value.get(key) for key in keys]
        if isinstance(value, dict)
        else [getattr(value, key, None) for key in keys]
    )
    for raw in values:
        if raw is None:
            continue
        try:
            return max(0.1, float(raw))
        except (TypeError, ValueError):
            continue
    return float(default)


def prepare_video_input(
    video: Any,
    *,
    max_frames: Optional[int] = None,
    sample_fps: float | None = None,
) -> PreparedVideo:
    """Normalize a path, mapping, or preloaded-frame object."""
    path = _as_path(video)
    fps = _sample_fps_of(video, sample_fps or DEFAULT_SAMPLE_FPS)
    if sample_fps is not None:
        fps = max(0.1, float(sample_fps))
    limit = 16 if max_frames is None else max(2, int(max_frames))
    au_csv = _as_au_csv(video, path if path and path.suffix else None)
    expected_class = _as_expected_class(video)

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
            selected = [
                int(round(index * (len(frames) - 1) / (limit - 1)))
                for index in range(limit)
            ]
            frames = [frames[index] for index in selected]
            indices = [
                indices[index] for index in selected if index < len(indices)
            ]
        return PreparedVideo(
            path=str(path.resolve()) if path and path.is_file() else None,
            frames=frames,
            indices=indices,
            fps=fps,
            sample_fps=fps,
            frame_count=len(frames),
            au_csv=au_csv,
            expected_class=expected_class,
            source="preloaded_frames",
        )

    if path is None or not path.is_file():
        return PreparedVideo(
            path=str(path) if path is not None else None,
            fps=fps,
            sample_fps=fps,
            au_csv=au_csv,
            expected_class=expected_class,
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
        expected_class=expected_class,
        source="sampled_from_path",
    )


def _load_json_profile(key: str) -> dict[str, Any]:
    path = profile_path(key)
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def _load_forensics_profiles() -> dict[str, Any]:
    return _load_json_profile("forensics_profiles")


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


def _first_score(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            return _clamp(numeric)
    return None


def _composite_details(
    dimensions: dict[str, float | None],
) -> tuple[float | None, dict[str, dict[str, float | None]]]:
    score = _score_from_values(list(dimensions.values()))
    return score, {
        key: {
            "score_0_1": value,
            "score_0_100": (
                float(value) * 100.0 if value is not None else None
            ),
        }
        for key, value in dimensions.items()
    }


def _forensics_input(video: PreparedVideo) -> Any | None:
    if video.path and Path(video.path).is_file():
        return video.path
    if video.frames:
        return video.frames
    return None


def _run_specialization_forensics(
    generated: PreparedVideo,
    profiles: dict[str, Any],
    *,
    max_frames: int,
) -> dict[str, Any]:
    facial_input = (
        generated.au_csv
        if generated.au_csv and Path(generated.au_csv).is_file()
        else None
    )
    texture_input = _forensics_input(generated)
    if facial_input is None and texture_input is None:
        return {}
    return analyze_forensics(
        facial_motion=facial_input,
        facial_motion_profile=profiles.get("facial_motion"),
        texture_detail=texture_input,
        texture_detail_profile=profiles.get("texture_detail"),
        authenticity_calibrator=profiles.get("authenticity_calibrator"),
        max_frames=max_frames,
        sample_fps=float(generated.sample_fps or DEFAULT_SAMPLE_FPS),
        detect_faces=True,
    )


def _load_expression_result(
    generated: PreparedVideo,
    profile: dict[str, Any] | None,
) -> dict[str, Any]:
    if (
        not profile
        or generated.au_csv is None
        or not Path(generated.au_csv).is_file()
    ):
        return {}
    return score_expression_profile(
        generated.au_csv,
        profile,
        expected_class=generated.expected_class,
    )


def _profile_version(profile: dict[str, Any] | None) -> str | None:
    if not isinstance(profile, dict):
        return None
    for key in ("schema_version", "profile_schema_version"):
        if profile.get(key) is not None:
            return str(profile[key])
    return None


def _reference_image_count(images: Optional[Sequence[Any]]) -> int:
    if images is None:
        return 0
    if isinstance(images, (str, bytes, Path)):
        return 1
    try:
        return len(images)
    except TypeError:
        return 1


def score_detail_quality(
    *,
    generated: PreparedVideo,
    reference: PreparedVideo | None,
    reference_images: Optional[Sequence[Any]],
    max_frames: Optional[int],
) -> dict[str, Any]:
    """Return the web-equivalent Wang Xing texture/detail composite."""
    limit = 16 if max_frames is None else max(2, int(max_frames))
    profiles = _load_forensics_profiles()
    forensics = _run_specialization_forensics(
        generated,
        profiles,
        max_frames=limit,
    )
    branches = forensics.get("branches", {})
    texture_result = branches.get("texture_detail") or {}
    texture_metrics = texture_result.get("metrics", {})
    if not isinstance(texture_metrics, dict):
        texture_metrics = {}
    fusion = forensics.get("fusion", {})
    if not isinstance(fusion, dict):
        fusion = {}

    dimensions = {
        "texture_evidence_0_1": _first_score(
            texture_metrics.get("raw_real_domain_evidence_0_1"),
        ),
        "temporal_stability_0_1": _first_score(
            texture_metrics.get("micro_temporal_naturalness_0_1"),
            texture_metrics.get("temporal_stability_proxy_0_1"),
        ),
        "detail_clarity_0_1": _first_score(
            texture_metrics.get("optical_flow_homogeneity_0_1"),
            (
                1.0 - float(texture_metrics["texture_flicker_0_1"])
                if texture_metrics.get("texture_flicker_0_1") is not None
                else None
            ),
        ),
        "real_domain_fit_0_1": _first_score(
            texture_metrics.get("real_domain_fit_0_1"),
        ),
        "fusion_evidence_0_1": _first_score(
            forensics.get("scores", {}).get(
                "raw_real_domain_evidence_0_1",
            ),
            fusion.get("raw_real_domain_evidence_0_1"),
        ),
    }
    score, dimension_details = _composite_details(dimensions)
    image_count = _reference_image_count(reference_images)
    has_input = bool(_forensics_input(generated))
    return {
        "score": score,
        "status": "available" if score is not None else "unavailable",
        "details": {
            "scope": "wangxing_specialization_texture",
            "placeholder": False,
            "method": "web_wangxing_texture_radar_composite",
            "composite_score_0_1": score,
            "composite_score_0_100": (
                score * 100.0 if score is not None else None
            ),
            "dimensions": dimension_details,
            "metric_labels": {
                "texture_evidence_0_1": "质感证据",
                "temporal_stability_0_1": "时序稳定",
                "detail_clarity_0_1": "细节清晰",
                "real_domain_fit_0_1": "真人域拟合",
                "fusion_evidence_0_1": "融合证据",
            },
            "texture_evidence_0_1": dimensions["texture_evidence_0_1"],
            "temporal_stability_0_1": dimensions[
                "temporal_stability_0_1"
            ],
            "detail_clarity_0_1": dimensions["detail_clarity_0_1"],
            "real_domain_fit_0_1": dimensions["real_domain_fit_0_1"],
            "fusion_evidence_0_1": dimensions["fusion_evidence_0_1"],
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
            "profile_schema_version": _profile_version(
                profiles.get("texture_detail"),
            ),
            "input_available": has_input,
            "forensics": forensics,
        },
    }


def score_face_expression(
    *,
    generated: PreparedVideo,
    reference: PreparedVideo | None,
    reference_images: Optional[Sequence[Any]],
    max_frames: Optional[int],
) -> dict[str, Any]:
    """Return the web-equivalent Wang Xing expression composite."""
    limit = 16 if max_frames is None else max(2, int(max_frames))
    profiles = _load_forensics_profiles()
    expression_profile, expression_profile_path = _load_expression_profile()
    expression_result = _load_expression_result(
        generated,
        expression_profile,
    )
    forensics = _run_specialization_forensics(
        generated,
        profiles,
        max_frames=limit,
    )
    facial_result = forensics.get("branches", {}).get("facial_motion") or {}
    facial_metrics = facial_result.get("metrics", {})
    if not isinstance(facial_metrics, dict):
        facial_metrics = {}
    event_statistics = expression_result.get("event_statistics") or {}

    dimensions = {
        "profile_compatibility_0_1": _first_score(
            expression_result.get("compatibility_0_1"),
        ),
        "muscle_action_evidence_0_1": _first_score(
            facial_metrics.get("raw_real_domain_evidence_0_1"),
            facial_metrics.get("training_free_motion_prior_0_1"),
            facial_metrics.get("motion_coherence_0_1"),
        ),
        "action_coherence_0_1": _first_score(
            facial_metrics.get("au_relation_consistency_0_1"),
            facial_metrics.get("motion_coherence_0_1"),
        ),
        "active_ratio_0_1": _first_score(
            facial_metrics.get("au_dynamics_naturalness_0_1"),
            event_statistics.get("active_ratio"),
        ),
        "landmark_coverage_0_1": _first_score(
            facial_metrics.get("landmark_valid_frame_ratio"),
            event_statistics.get("longest_event_ratio"),
        ),
    }
    score, dimension_details = _composite_details(dimensions)
    image_count = _reference_image_count(reference_images)
    warning = (
        "AU CSV is required for the complete Wang Xing expression and "
        "facial-motion dimensions."
        if generated.au_csv is None
        else None
    )
    return {
        "score": score,
        "status": "available" if score is not None else "partial",
        "details": {
            "scope": "wangxing_specialization_expression",
            "placeholder": False,
            "method": "web_wangxing_expression_radar_composite",
            "composite_score_0_1": score,
            "composite_score_0_100": (
                score * 100.0 if score is not None else None
            ),
            "dimensions": dimension_details,
            "metric_labels": {
                "profile_compatibility_0_1": "画像符合度",
                "muscle_action_evidence_0_1": "肌肉动作证据",
                "action_coherence_0_1": "动作连贯",
                "active_ratio_0_1": "活跃比例",
                "landmark_coverage_0_1": "关键点覆盖",
            },
            "profile_compatibility_0_1": dimensions[
                "profile_compatibility_0_1"
            ],
            "muscle_action_evidence_0_1": dimensions[
                "muscle_action_evidence_0_1"
            ],
            "action_coherence_0_1": dimensions["action_coherence_0_1"],
            "active_ratio_0_1": dimensions["active_ratio_0_1"],
            "landmark_coverage_0_1": dimensions["landmark_coverage_0_1"],
            "selected_profile": expression_result.get("selected_profile"),
            "selected_profile_display_name": expression_result.get(
                "selected_profile_display_name",
            ),
            "profile_result": expression_result,
            "reference_count": image_count,
            "valid_face_frames": generated.frame_count,
            "total_sampled_frames": generated.frame_count,
            "face_coverage": _first_score(
                facial_metrics.get("landmark_valid_frame_ratio"),
            ),
            "reference_frame_count": (
                reference.frame_count if reference is not None else 0
            ),
            "reference_image_count": image_count,
            "max_frames": max_frames,
            "sample_fps": generated.sample_fps,
            "video_path": generated.path,
            "au_csv": generated.au_csv,
            "expression_profile": expression_profile_path,
            "profile_schema_version": _profile_version(expression_profile),
            "forensics": forensics,
            "warning": warning,
        },
    }


def describe_probe(path: str | Path) -> dict[str, Any]:
    info = probe_video(path)
    return info.to_dict()
