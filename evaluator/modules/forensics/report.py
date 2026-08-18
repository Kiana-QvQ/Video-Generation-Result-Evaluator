from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .facial_motion import score_facial_motion
from .seedance_authenticity import (
    fuse_authenticity_evidence,
    rank_window_evidence,
    summarize_window_evidence,
)
from .texture_detail import score_texture_detail

FORENSICS_REPORT_SCHEMA = "video_forensics_report_v1"


def _window_records(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    records = result.get("window_records")
    if isinstance(records, list):
        return records
    feature_record = result.get("feature_record", {})
    if isinstance(feature_record, dict) and isinstance(
        feature_record.get("window_records"),
        list,
    ):
        return feature_record["window_records"]
    return []


def analyze_forensics(
    *,
    facial_motion: dict[str, Any] | str | Path | None = None,
    facial_motion_profile: dict[str, Any] | None = None,
    texture_detail: dict[str, Any] | Sequence[np.ndarray] | str | Path | None = None,
    texture_detail_profile: dict[str, Any] | None = None,
    authenticity_calibrator: dict[str, Any] | None = None,
    face_boxes: Sequence[tuple[int, int, int, int] | None] | None = None,
    max_frames: int = 64,
    sample_fps: float = 8.0,
    detect_faces: bool = True,
    nr_vqa_backends: Sequence[str] | None = None,
    nr_vqa_ensemble: bool = False,
    device: str = "auto",
) -> dict[str, Any]:
    """Run either branch or both and keep their evidence separate."""
    facial_result = None
    if facial_motion is not None:
        facial_result = score_facial_motion(
            facial_motion,
            facial_motion_profile or {},
        )

    texture_result = None
    if texture_detail is not None:
        texture_result = score_texture_detail(
            texture_detail,
            texture_detail_profile,
            face_boxes=face_boxes,
            max_frames=max_frames,
            sample_fps=sample_fps,
            detect_faces=detect_faces,
            nr_vqa_backends=nr_vqa_backends,
            nr_vqa_ensemble=nr_vqa_ensemble,
            device=device,
        )

    authenticity = fuse_authenticity_evidence(
        facial_result,
        texture_result,
        calibrator=authenticity_calibrator,
    )
    raw_fused = authenticity.get("raw_real_domain_evidence_0_1")
    calibrated_probability = authenticity.get(
        "calibrated_real_probability_0_1"
    )
    facial_score = None
    if facial_result is not None:
        facial_metrics = facial_result.get("metrics", {})
        facial_score = facial_metrics.get(
            "real_capture_likelihood_0_1"
        )
        if facial_score is None:
            facial_score = facial_metrics.get("motion_coherence_0_1")
    texture_score = None
    if texture_result is not None:
        texture_metrics = texture_result.get("metrics", {})
        texture_score = texture_metrics.get(
            "real_capture_likelihood_0_1"
        )
        if texture_score is None:
            texture_score = texture_metrics.get(
                "temporal_stability_proxy_0_1"
            )
    calibrated = authenticity.get("status") == "calibrated"
    report_status = (
        "calibrated"
        if calibrated
        else (
            "profile_evidence_only"
            if authenticity.get("status") == "uncalibrated"
            else "features_only"
        )
    )
    return {
        "schema_version": FORENSICS_REPORT_SCHEMA,
        "status": report_status,
        "branches": {
            "facial_motion": facial_result,
            "texture_detail": texture_result,
        },
        "fusion": {
            "real_capture_likelihood_0_1": calibrated_probability,
            "raw_real_domain_evidence_0_1": raw_fused,
            "branch_count": len([item for item in (facial_result, texture_result) if item]),
            "profile_scored_branch_count": len(
                [
                    item
                    for item in (facial_result, texture_result)
                    if item
                    and item.get("metrics", {}).get(
                        "raw_real_domain_evidence_0_1",
                        item.get("metrics", {}).get(
                            "real_capture_likelihood_0_1"
                        ),
                    )
                    is not None
                ]
            ),
            "calibrated_branch_count": sum(
                result is not None
                and result.get("probability_calibrated", False)
                for result in (facial_result, texture_result)
            ),
            "method": (
                "confidence_weighted_calibrated_probability_fusion"
                if calibrated
                else (
                    "not_calibrated"
                    if authenticity.get("status") == "unavailable"
                    else "confidence_weighted_uncalibrated_profile_fusion"
                )
            ),
            "warning": (
                None
                if calibrated
                else (
                    "The report has only raw profile evidence. A held-out "
                    "probability calibrator is required before interpreting "
                    "the result as a real-versus-Seedance detector."
                )
            ),
        },
        "scores": {
            "facial_expression_muscle_score_0_1": facial_score,
            "texture_detail_score_0_1": texture_score,
            "real_capture_likelihood_0_1": calibrated_probability,
            "raw_real_domain_evidence_0_1": raw_fused,
            "calibrated_real_probability_0_1": calibrated_probability,
            "ssl_au_score_0_1": (
                facial_result.get("metrics", {}).get("ssl_au_score_0_1")
                if isinstance(facial_result, dict)
                else None
            ),
            "ssl_backbone_score_0_1": (
                facial_result.get("metrics", {}).get("ssl_backbone_score_0_1")
                if isinstance(facial_result, dict)
                else None
            ),
            "physio_rhythm_score_0_1": (
                facial_result.get("metrics", {}).get("physio_rhythm_score_0_1")
                if isinstance(facial_result, dict)
                else None
            ),
            "nr_vqa_score_0_1": (
                texture_result.get("metrics", {}).get("nr_vqa_score_0_1")
                if isinstance(texture_result, dict)
                else None
            ),
            "freq_forensics_score_0_1": (
                texture_result.get("metrics", {}).get("freq_forensics_score_0_1")
                if isinstance(texture_result, dict)
                else None
            ),
            "nr_vqa_backend": (
                texture_result.get("feature_record", {})
                .get("nr_vqa", {})
                .get("backend")
                if isinstance(texture_result, dict)
                else None
            ),
            "nr_vqa_device": (
                texture_result.get("feature_record", {}).get("nr_vqa_device")
                if isinstance(texture_result, dict)
                else None
            ),
            "pose_normalized_frame_ratio": (
                facial_result.get("metrics", {}).get(
                    "pose_normalized_frame_ratio"
                )
                if isinstance(facial_result, dict)
                else None
            ),
        },
        "authenticity": authenticity,
        "window_evidence": {
            "facial_expression_muscle": rank_window_evidence(
                _window_records(facial_result)
            ),
            "texture_detail": rank_window_evidence(
                _window_records(texture_result)
            ),
        },
        "window_summaries": {
            "facial_expression_muscle": {
                "summary": summarize_window_evidence(
                    _window_records(facial_result)
                ),
                "semantics": (
                    "Motion activity coverage and peaks; not an artifact "
                    "probability."
                ),
            },
            "texture_detail": {
                "summary": summarize_window_evidence(
                    _window_records(texture_result)
                ),
                "semantics": (
                    "Texture flicker evidence; higher values indicate "
                    "windows requiring review."
                ),
            },
        },
        "score_semantics": {
            "facial_expression_muscle_score_0_1": (
                "Raw real-versus-Seedance facial-motion profile evidence. "
                "It is not a probability and is not an ordinary expression "
                "correctness score."
            ),
            "texture_detail_score_0_1": (
                "Raw real-versus-Seedance texture profile evidence. It is "
                "not a probability and is not a generic image-quality score."
            ),
            "ssl_au_score_0_1": (
                "Training-free self-supervised AU temporal consistency "
                "(TCAE / VideoMAE style proxies). No manual AU labels."
            ),
            "ssl_backbone_score_0_1": (
                "Trained lightweight AU temporal autoencoder score "
                "(reconstruction + prediction + masking). Optional weights."
            ),
            "physio_rhythm_score_0_1": (
                "Blink / eye-aperture physiological rhythm from landmarks."
            ),
            "nr_vqa_score_0_1": (
                "No-reference VQA score (builtin / pyiqa / optional DOVER, "
                "FAST-VQA, RAPIQUE, SLEEQ). VMAF is not used."
            ),
            "freq_forensics_score_0_1": (
                "No-reference DCT/FFT compression and spectrum forensics."
            ),
            "pose_normalized_frame_ratio": (
                "Fraction of AU/landmark frames pose-normalized via "
                "MediaPipe Face Landmarker / Face Mesh anchors."
            ),
            "real_capture_likelihood_0_1": (
                "Held-out calibrated real-capture probability; null until "
                "a ready probability calibrator is supplied."
            ),
            "not_expression_correctness": True,
            "not_generic_image_quality": True,
            "manual_scores_required": False,
        },
        "auto_pipeline": {
            "pose_normalization": True,
            "self_supervised_au": True,
            "ssl_backbone": True,
            "physiological_rhythm": True,
            "no_reference_vqa": True,
            "frequency_forensics": True,
            "nr_vqa_backend": (
                texture_result.get("feature_record", {})
                .get("nr_vqa", {})
                .get("backend")
                if isinstance(texture_result, dict)
                else None
            ),
            "nr_vqa_device": (
                texture_result.get("feature_record", {}).get("nr_vqa_device")
                if isinstance(texture_result, dict)
                else None
            ),
            "vmaf_used": False,
            "pseudo_label_calibration_supported": True,
            "perturbation_probes_supported": True,
        },
    }
