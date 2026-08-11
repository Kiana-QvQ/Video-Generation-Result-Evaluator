"""Runtime for the standalone Wang Xing specialization entrypoints.

The web UI renders two five-dimension Wang Xing radar composites. This module
reproduces those composites without importing the web application or the
ordinary five-category evaluator.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
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


def _existing_path(
    raw: Any,
    *,
    relative_to: Path | None = None,
) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    candidates = [path]
    if not path.is_absolute() and relative_to is not None:
        candidates.insert(0, relative_to / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _as_au_csv(value: Any, video_path: Path | None) -> str | None:
    base_dir = video_path.parent if video_path is not None else None
    if isinstance(value, dict):
        for key in ("au_csv", "au_path", "csv"):
            path = _existing_path(value.get(key), relative_to=base_dir)
            if path is not None:
                return str(path)
    for attribute in ("au_csv", "au_path"):
        path = _existing_path(
            getattr(value, attribute, None),
            relative_to=base_dir,
        )
        if path is not None:
            return str(path)
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
        if indices_attr is None:
            indices = list(range(len(frames)))
        else:
            indices = [int(value) for value in indices_attr]
            if len(indices) < len(frames):
                indices.extend(range(len(indices), len(frames)))
            else:
                indices = indices[: len(frames)]
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


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def _profile_sample_count(profile: dict[str, Any] | None) -> int | None:
    if not isinstance(profile, dict):
        return None
    provenance = profile.get("provenance")
    if isinstance(provenance, dict):
        for key in ("sample_count", "real_sample_count"):
            value = provenance.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
    classes = profile.get("classes")
    if isinstance(classes, dict):
        total = 0
        found = False
        for item in classes.values():
            if isinstance(item, dict) and item.get("sample_count") is not None:
                found = True
                total += int(item.get("sample_count") or 0)
        if found:
            return total
    return None


def _ensure_au_csv(
    generated: PreparedVideo,
    *,
    max_frames: int,
) -> tuple[PreparedVideo, str | None]:
    """Attach a real AU CSV: side-car first, else synthesize from frames."""
    if generated.au_csv and Path(generated.au_csv).is_file():
        return generated, "sidecar"
    if not generated.frames:
        return generated, None
    from .au_from_video import synthesize_au_csv_from_frames

    synthesized = synthesize_au_csv_from_frames(
        generated.frames,
        indices=generated.indices,
        sample_fps=float(generated.sample_fps or DEFAULT_SAMPLE_FPS),
        video_path=generated.path,
        download_model=True,
    )
    if not synthesized:
        return generated, None
    return replace(generated, au_csv=synthesized), "synthesized_from_video"


def _score_source_profile(au_csv: str | None) -> dict[str, Any]:
    """Use bundled wangxing_source_profile when AU is available."""
    if not au_csv or not Path(au_csv).is_file():
        return {
            "status": "skipped",
            "reason": "au_csv_missing",
            "profile": "wangxing_source_profile",
        }
    path = profile_path("wangxing_source_profile")
    if path is None or not path.is_file():
        return {
            "status": "unavailable",
            "reason": "profile_missing",
            "profile": "wangxing_source_profile",
        }
    try:
        from ..wangxing.wangxing_specialization import score_source_profile

        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            return {"status": "unavailable", "reason": "invalid_profile"}
        result = score_source_profile(au_csv, payload)
        result["profile_path"] = str(path)
        result["profile_sample_hint"] = _profile_sample_count(payload)
        return result
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "profile": "wangxing_source_profile",
        }


def _score_identity_profile(
    video_path: str | None,
    *,
    max_frames: int,
) -> dict[str, Any]:
    """Use bundled wangxing_identity_profile when InsightFace is available."""
    if not video_path or not Path(video_path).is_file():
        return {
            "status": "skipped",
            "reason": "video_path_missing",
            "profile": "wangxing_identity_profile",
        }
    path = profile_path("wangxing_identity_profile")
    if path is None or not path.is_file():
        return {
            "status": "unavailable",
            "reason": "profile_missing",
            "profile": "wangxing_identity_profile",
        }
    try:
        from ..core.holistic_evaluator import _FaceDetector, _IdentityBackend
        from ..wangxing.wangxing_specialization import evaluate_identity_profile

        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            return {"status": "unavailable", "reason": "invalid_profile"}
        backend = _IdentityBackend(_FaceDetector(), device="cpu")
        result = evaluate_identity_profile(
            video_path,
            payload,
            backend,
            max_frames=max_frames,
        )
        result["profile_path"] = str(path)
        provenance = payload.get("provenance")
        if isinstance(provenance, dict) and provenance.get("sample_count") is not None:
            result["profile_sample_count"] = int(provenance["sample_count"])
        return result
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "profile": "wangxing_identity_profile",
        }


def _bundled_asset_status() -> dict[str, Any]:
    """Report which trained assets are present for collaborators."""
    status: dict[str, Any] = {}
    for key in (
        "wangxing_expression_profile",
        "wangxing_identity_profile",
        "wangxing_source_profile",
        "wangxing_au_profile",
        "original_emotion_au_profile",
        "forensics_profiles",
        "forensics_authenticity_calibrator",
        "holdout_split",
        "model_profile",
    ):
        path = profile_path(key)
        status[key] = {
            "present": bool(path and path.is_file()),
            "path": str(path) if path else None,
        }
    try:
        from .face_landmarker import default_model_path

        landmarker = default_model_path()
        status["face_landmarker_task"] = {
            "present": landmarker.is_file() and landmarker.stat().st_size > 1024,
            "path": str(landmarker),
        }
    except Exception as exc:
        status["face_landmarker_task"] = {
            "present": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return status


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
    generated, au_source = _ensure_au_csv(generated, max_frames=limit)
    forensics = _run_specialization_forensics(
        generated,
        profiles,
        max_frames=limit,
    )
    branches = _mapping(forensics.get("branches"))
    texture_result = _mapping(branches.get("texture_detail"))
    texture_metrics = _mapping(texture_result.get("metrics"))
    fusion = _mapping(forensics.get("fusion"))
    forensic_scores = _mapping(forensics.get("scores"))
    flicker = _first_score(texture_metrics.get("texture_flicker_0_1"))

    dimensions = {
        "texture_evidence_0_1": _first_score(
            texture_metrics.get("raw_real_domain_evidence_0_1"),
        ),
        "temporal_stability_0_1": _first_score(
            texture_metrics.get("micro_temporal_naturalness_0_1"),
            texture_metrics.get("temporal_stability_proxy_0_1"),
        ),
        "detail_clarity_0_1": _first_score(
            texture_metrics.get("nr_vqa_score_0_1"),
            texture_metrics.get("optical_flow_homogeneity_0_1"),
            1.0 - flicker if flicker is not None else None,
        ),
        "real_domain_fit_0_1": _first_score(
            texture_metrics.get("real_domain_fit_0_1"),
        ),
        "fusion_evidence_0_1": _first_score(
            forensic_scores.get("raw_real_domain_evidence_0_1"),
            fusion.get("raw_real_domain_evidence_0_1"),
        ),
    }
    score, dimension_details = _composite_details(dimensions)
    image_count = _reference_image_count(reference_images)
    has_input = bool(_forensics_input(generated))
    status = (
        "unavailable"
        if score is None
        else (
            "available"
            if all(value is not None for value in dimensions.values())
            else "partial"
        )
    )
    return {
        "score": score,
        "status": status,
        "details": {
            "scope": "wangxing_specialization_texture",
            "placeholder": False,
            "method": "web_wangxing_texture_radar_composite",
            "au_source": au_source,
            "bundled_assets": _bundled_asset_status(),
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
            "reference_used": False,
            "reference_note": (
                "Reference video/images are accepted for API compatibility; "
                "the web-equivalent texture radar scores the generated input."
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
    """Return the web-equivalent Wang Xing expression composite."""
    limit = 16 if max_frames is None else max(2, int(max_frames))
    profiles = _load_forensics_profiles()
    expression_profile, expression_profile_path = _load_expression_profile()
    generated, au_source = _ensure_au_csv(generated, max_frames=limit)
    expression_result = _load_expression_result(
        generated,
        expression_profile,
    )
    forensics = _run_specialization_forensics(
        generated,
        profiles,
        max_frames=limit,
    )
    branches = _mapping(forensics.get("branches"))
    facial_result = _mapping(branches.get("facial_motion"))
    facial_metrics = _mapping(facial_result.get("metrics"))
    event_statistics = _mapping(expression_result.get("event_statistics"))

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
            facial_metrics.get("ssl_temporal_consistency_0_1"),
            facial_metrics.get("au_relation_consistency_0_1"),
            facial_metrics.get("motion_coherence_0_1"),
        ),
        "active_ratio_0_1": _first_score(
            facial_metrics.get("au_dynamics_naturalness_0_1"),
            facial_metrics.get("ssl_au_score_0_1"),
            event_statistics.get("active_ratio"),
        ),
        "landmark_coverage_0_1": _first_score(
            facial_metrics.get("pose_normalized_frame_ratio"),
            facial_metrics.get("landmark_valid_frame_ratio"),
            event_statistics.get("longest_event_ratio"),
        ),
    }
    score, dimension_details = _composite_details(dimensions)
    image_count = _reference_image_count(reference_images)
    profile_samples = _profile_sample_count(expression_profile)
    source_result = _score_source_profile(generated.au_csv)
    identity_result = _score_identity_profile(
        generated.path,
        max_frames=limit,
    )
    bundled_assets = _bundled_asset_status()

    # Last resort only: Expression/*.jpg prototypes when AU synthesis failed.
    if score is None and generated.frames and au_source is None:
        from .expression_prototype_fallback import score_expression_prototypes

        fallback = score_expression_prototypes(
            generated.frames,
            max_frames=limit,
            sample_fps=float(generated.sample_fps or DEFAULT_SAMPLE_FPS),
        )
        fallback_score = fallback.get("score")
        if fallback_score is not None:
            fallback_details = _mapping(fallback.get("details"))
            return {
                "score": float(fallback_score),
                "status": str(fallback.get("status") or "partial"),
                "details": {
                    "scope": "expression_prototype_fallback",
                    "placeholder": False,
                    "method": "expression_prototype_fallback",
                    "composite_score_0_1": float(fallback_score),
                    "composite_score_0_100": float(fallback_score) * 100.0,
                    "dimensions": dimension_details,
                    "metric_labels": {
                        "profile_compatibility_0_1": "动作原型匹配",
                        "muscle_action_evidence_0_1": "肌肉几何",
                        "action_coherence_0_1": "肌肉-皱纹同步",
                        "active_ratio_0_1": "有效帧覆盖",
                        "landmark_coverage_0_1": "关键点覆盖",
                    },
                    "profile_compatibility_0_1": fallback_details.get(
                        "profile_compatibility_0_1"
                    ),
                    "muscle_action_evidence_0_1": fallback_details.get(
                        "muscle_action_evidence_0_1"
                    ),
                    "action_coherence_0_1": fallback_details.get(
                        "action_coherence_0_1"
                    ),
                    "active_ratio_0_1": fallback_details.get("active_ratio_0_1"),
                    "landmark_coverage_0_1": fallback_details.get(
                        "landmark_coverage_0_1"
                    ),
                    "frame_match_score": fallback_details.get("frame_match_score"),
                    "geometry_score": fallback_details.get("geometry_score"),
                    "gaze_score": fallback_details.get("gaze_score"),
                    "wrinkle_score": fallback_details.get("wrinkle_score"),
                    "texture_score": fallback_details.get("texture_score"),
                    "motion_score": fallback_details.get("motion_score"),
                    "geometry_method": fallback_details.get("geometry_method"),
                    "gaze_method": fallback_details.get("gaze_method"),
                    "expression_dir": fallback_details.get("expression_dir"),
                    "selected_profile": expression_result.get("selected_profile"),
                    "selected_profile_display_name": expression_result.get(
                        "selected_profile_display_name",
                    ),
                    "profile_result": expression_result,
                    "reference_count": fallback_details.get(
                        "reference_count",
                        image_count,
                    ),
                    "valid_face_frames": fallback_details.get(
                        "valid_face_frames",
                        0,
                    ),
                    "total_sampled_frames": generated.frame_count,
                    "face_coverage": fallback_details.get("face_coverage"),
                    "reference_frame_count": (
                        reference.frame_count if reference is not None else 0
                    ),
                    "reference_image_count": image_count,
                    "max_frames": max_frames,
                    "sample_fps": generated.sample_fps,
                    "video_path": generated.path,
                    "au_csv": generated.au_csv,
                    "expression_profile": expression_profile_path,
                    "profile_schema_version": _profile_version(
                        expression_profile
                    ),
                    "forensics": forensics,
                    "warning": fallback_details.get("warning"),
                    "reference_used": True,
                    "reference_note": fallback_details.get("reference_note"),
                },
            }

    missing_dimensions = [
        key for key, value in dimensions.items() if value is None
    ]
    warning: str | None = None
    if generated.au_csv is None:
        warning = (
            "AU CSV is required for the complete Wang Xing expression and "
            "facial-motion dimensions; Expression fallback also unavailable."
        )
    elif expression_profile is None:
        warning = "Wang Xing expression profile is unavailable."
    elif not expression_result:
        warning = "AU CSV could not be scored by the expression profile."
    elif missing_dimensions:
        warning = (
            "Some expression dimensions are unavailable: "
            + ", ".join(missing_dimensions)
        )
    if au_source == "synthesized_from_video" and score is not None:
        synth_note = (
            "未提供旁路 AU CSV，已从生成视频自动合成 AU，"
            f"并对照王兴表情 profile（{profile_samples or 'n/a'} 条样本）评分。"
        )
        warning = f"{synth_note} {warning}" if warning else synth_note
    status = (
        "partial"
        if score is None and generated.au_csv is None
        else (
            "unavailable"
            if score is None
            else (
                "available"
                if not missing_dimensions and au_source == "sidecar"
                else "partial"
            )
        )
    )
    profile_compat = dimensions["profile_compatibility_0_1"]
    muscle = dimensions["muscle_action_evidence_0_1"]
    coherence = dimensions["action_coherence_0_1"]
    return {
        "score": score,
        "status": status,
        "details": {
            "scope": "wangxing_specialization_expression",
            "placeholder": False,
            "method": "web_wangxing_expression_radar_composite",
            "geometry_method": "wangxing_specialization",
            "gaze_method": "wangxing_specialization",
            "au_source": au_source,
            "wangxing_source": source_result,
            "wangxing_identity": identity_result,
            "bundled_assets": bundled_assets,
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
            "profile_compatibility_0_1": profile_compat,
            "muscle_action_evidence_0_1": muscle,
            "action_coherence_0_1": coherence,
            "active_ratio_0_1": dimensions["active_ratio_0_1"],
            "landmark_coverage_0_1": dimensions["landmark_coverage_0_1"],
            # UI radar keys used by collaborator app.py
            "frame_match_score": profile_compat if profile_compat is not None else score,
            "geometry_score": muscle if muscle is not None else score,
            "gaze_score": profile_compat if profile_compat is not None else score,
            "wrinkle_score": muscle if muscle is not None else score,
            "texture_score": muscle if muscle is not None else score,
            "motion_score": coherence if coherence is not None else score,
            "selected_profile": expression_result.get("selected_profile"),
            "selected_profile_display_name": expression_result.get(
                "selected_profile_display_name",
            ),
            "profile_result": expression_result,
            "profile_sample_count": profile_samples,
            "reference_count": profile_samples if profile_samples is not None else image_count,
            "valid_face_frames": (
                round(
                    float(facial_metrics["landmark_valid_frame_ratio"])
                    * generated.frame_count
                )
                if facial_metrics.get("landmark_valid_frame_ratio") is not None
                else 0
            ),
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
            "reference_used": bool(profile_samples),
            "reference_note": (
                "Wang Xing expression profile scoring "
                f"(au_source={au_source or 'missing'})."
            ),
        },
    }


def describe_probe(path: str | Path) -> dict[str, Any]:
    info = probe_video(path)
    return info.to_dict()
