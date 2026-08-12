"""Initial forensic branches for real-capture versus generated-video analysis."""

from .au_ssl import extract_self_supervised_au_features, merge_ssl_into_motion_features
from .au_ssl_backbone import (
    extract_backbone_features,
    load_backbone,
    save_backbone,
    train_au_ssl_backbone,
)
from .authenticity_decision import (
    decide_real_vs_generated,
    metrics_from_decisions,
)
from .facial_motion import (
    build_facial_motion_profile,
    build_two_domain_facial_motion_profile,
    extract_facial_motion_features,
    score_facial_motion,
)
from .frequency_forensics import extract_frequency_forensics_features
from .fused_hard_detector import score_fused_hard_detector
from .learned_fusion_head import (
    extract_fusion_features,
    fit_learned_fusion_head,
    load_learned_head,
    save_learned_head,
    score_with_learned_head,
)
from .nr_vqa import extract_nr_vqa_features, resolve_nr_vqa_backend_order
from .perturbation import (
    run_frame_perturbation_battery,
    run_landmark_jitter_probe,
)
from .physiological_rhythm import extract_physiological_rhythm_features
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
    "decide_real_vs_generated",
    "extract_backbone_features",
    "extract_facial_motion_features",
    "extract_frequency_forensics_features",
    "extract_fusion_features",
    "extract_nr_vqa_features",
    "extract_physiological_rhythm_features",
    "extract_self_supervised_au_features",
    "extract_texture_detail_features",
    "fit_learned_fusion_head",
    "fit_probability_calibrator",
    "fit_pseudo_label_calibrator",
    "fuse_authenticity_evidence",
    "load_backbone",
    "load_learned_head",
    "merge_ssl_into_motion_features",
    "metrics_from_decisions",
    "rank_window_evidence",
    "resolve_nr_vqa_backend_order",
    "run_frame_perturbation_battery",
    "run_landmark_jitter_probe",
    "save_backbone",
    "save_learned_head",
    "score_fused_hard_detector",
    "score_with_learned_head",
    "summarize_window_evidence",
    "score_facial_motion",
    "score_texture_detail",
    "train_au_ssl_backbone",
]
