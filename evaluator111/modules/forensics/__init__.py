"""Initial forensic branches for real-capture versus generated-video analysis."""

from .au_ssl import extract_self_supervised_au_features
from .facial_motion import (
    build_facial_motion_profile,
    build_two_domain_facial_motion_profile,
    extract_facial_motion_features,
    score_facial_motion,
)
from .nr_vqa import extract_nr_vqa_features
from .perturbation import run_frame_perturbation_battery
from .pseudo_label_calibration import (
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
    "rank_window_evidence",
    "run_frame_perturbation_battery",
    "summarize_window_evidence",
    "score_facial_motion",
    "score_texture_detail",
]
