"""Initial forensic branches for real-capture versus generated-video analysis."""

from .facial_motion import (
    build_facial_motion_profile,
    build_two_domain_facial_motion_profile,
    extract_facial_motion_features,
    score_facial_motion,
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
    "build_texture_detail_profile",
    "build_two_domain_facial_motion_profile",
    "extract_facial_motion_features",
    "extract_texture_detail_features",
    "fit_probability_calibrator",
    "fuse_authenticity_evidence",
    "rank_window_evidence",
    "summarize_window_evidence",
    "score_facial_motion",
    "score_texture_detail",
]
