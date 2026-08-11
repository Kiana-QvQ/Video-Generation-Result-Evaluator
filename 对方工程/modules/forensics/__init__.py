"""Initial forensic branches for real-capture versus generated-video analysis."""

from .au_ssl import extract_self_supervised_au_features, merge_ssl_into_motion_features
from .facial_motion import (
    build_facial_motion_profile,
    build_two_domain_facial_motion_profile,
    extract_facial_motion_features,
    score_facial_motion,
)
from .nr_vqa import extract_nr_vqa_features, resolve_nr_vqa_backend_order
from .perturbation import (
    run_frame_perturbation_battery,
    run_landmark_jitter_probe,
)
from .pseudo_label_calibration import (
    apply_pseudo_calibrator,
    build_pseudo_labeled_samples,
    fit_pseudo_label_calibrator,
)
from .report import analyze_forensics
from .seedance_authenticity import (
    apply_probability_calibrator,
    fit_probability_calibrator,
    fuse_authenticity_evidence,
    rank_window_evidence,
    summarize_window_evidence,
)
from .texture_detail import (
    build_texture_detail_profile,
    extract_texture_detail_features,
    score_texture_detail,
)

__all__ = [
    "analyze_forensics",
    "apply_probability_calibrator",
    "apply_pseudo_calibrator",
    "build_facial_motion_profile",
    "build_pseudo_labeled_samples",
    "build_texture_detail_profile",
    "build_two_domain_facial_motion_profile",
    "extract_facial_motion_features",
    "extract_nr_vqa_features",
    "extract_self_supervised_au_features",
    "extract_texture_detail_features",
    "fit_probability_calibrator",
    "fit_pseudo_label_calibrator",
    "fuse_authenticity_evidence",
    "merge_ssl_into_motion_features",
    "rank_window_evidence",
    "resolve_nr_vqa_backend_order",
    "run_frame_perturbation_battery",
    "run_landmark_jitter_probe",
    "summarize_window_evidence",
    "score_facial_motion",
    "score_texture_detail",
]
