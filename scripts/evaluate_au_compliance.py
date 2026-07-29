from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.au_compliance import (
    AU_EVALUATOR_VERSION,
    fuse_compliance_scores,
    fuse_wangxing_targeted_scores,
    score_au_compliance,
)
from evaluator.holistic_evaluator import evaluate_identity
from evaluator.paths import project_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate identity, AU expression fidelity, leakage and likeness."
    )
    parser.add_argument("--generated-au", required=True)
    parser.add_argument("--au-profile", required=True)
    parser.add_argument(
        "--emotion-profile",
        default="data/au/original_emotion_au_profile.json",
        help="Original AU profile used only for automatic emotion classification.",
    )
    parser.add_argument("--driver-au")
    parser.add_argument("--leakage-classifier")
    parser.add_argument("--expected-class")
    parser.add_argument("--generated-video")
    parser.add_argument("--target-video")
    parser.add_argument("--target-image", action="append")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--identity-threshold", type=float, default=0.75)
    parser.add_argument("--personal-au-threshold", type=float, default=0.50)
    parser.add_argument(
        "--driver-expression-threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument("--leakage-threshold", type=float, default=0.50)
    parser.add_argument("--output")
    args = parser.parse_args()

    generated_au = project_path(args.generated_au)
    au_profile = project_path(args.au_profile)
    emotion_profile = (
        project_path(args.emotion_profile)
        if args.emotion_profile
        else None
    )
    driver_au = project_path(args.driver_au) if args.driver_au else None
    leakage_classifier = (
        project_path(args.leakage_classifier)
        if args.leakage_classifier
        else None
    )
    generated_video = (
        project_path(args.generated_video)
        if args.generated_video
        else None
    )
    target_video = (
        project_path(args.target_video)
        if args.target_video
        else None
    )
    target_images = [
        project_path(value) for value in (args.target_image or [])
    ]

    identity_score = None
    identity_result = None
    if generated_video and (target_video or target_images):
        identity_result = evaluate_identity(
            result_path=generated_video,
            reference_image=target_images,
            reference_video=target_video,
            ground_truth=None,
            max_frames=64,
            device=args.device,
        )
        identity_score = (
            identity_result.get("metrics", {}).get("score_0_1")
            if identity_result.get("status") != "unavailable"
            else None
        )

    au_result = score_au_compliance(
        au_profile,
        generated_au,
        expected_class=args.expected_class,
        driver_au_path=driver_au,
        leakage_classifier_path=leakage_classifier,
        emotion_profile_path=emotion_profile,
    )
    fused = fuse_compliance_scores(
        identity_score_0_1=identity_score,
        personal_au_score_0_1=au_result["personal_au_score_0_1"],
        driver_expression_score_0_1=au_result[
            "driver_expression_score_0_1"
        ],
        leakage_risk_0_1=au_result[
            "driver_identity_leakage_risk_0_1"
        ],
        identity_threshold=args.identity_threshold,
        personal_au_threshold=args.personal_au_threshold,
        driver_expression_threshold=args.driver_expression_threshold,
        leakage_threshold=args.leakage_threshold,
    )
    wangxing_targeted = fuse_wangxing_targeted_scores(
        personal_au_score_0_1=au_result["personal_au_score_0_1"],
        driver_expression_score_0_1=au_result[
            "driver_expression_score_0_1"
        ],
        temporal_alignment_score_0_1=au_result[
            "driver_temporal_alignment_score_0_1"
        ],
        leakage_risk_0_1=au_result[
            "driver_identity_leakage_risk_0_1"
        ],
        evidence_quality_status=au_result.get(
            "evidence_quality_status",
            "available",
        ),
        evidence_confidence_0_1=au_result.get(
            "evidence_confidence_0_1"
        ),
        uncertainty_reasons=au_result.get("uncertainty_reasons", []),
        personal_au_threshold=args.personal_au_threshold,
        driver_expression_threshold=args.driver_expression_threshold,
        leakage_threshold=args.leakage_threshold,
    )
    result = {
        "status": "available",
        "evaluation_meta": {
            "evaluator_version": AU_EVALUATOR_VERSION,
            "profile_schema_version": au_result.get(
                "profile_schema_version"
            ),
            "generated_au_path": str(generated_au),
            "driver_au_path": (
                str(driver_au) if driver_au is not None else None
            ),
        },
        "identity_preservation": identity_result,
        "au_compliance": au_result,
        "wangxing_targeted": wangxing_targeted,
        "fusion": fused,
        "threshold_note": (
            "Automatic evidence mode: low face quality or low usable-frame "
            "coverage is reported as review evidence instead of being "
            "treated as ground truth."
        ),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            serialized + "\n",
            encoding="utf-8",
        )
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
