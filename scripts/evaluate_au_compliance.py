from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.au_compliance import (
    fuse_compliance_scores,
    score_au_compliance,
)
from evaluator.holistic_evaluator import evaluate_identity


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate identity, AU expression fidelity, leakage and likeness."
    )
    parser.add_argument("--generated-au", required=True)
    parser.add_argument("--au-profile", required=True)
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

    identity_score = None
    identity_result = None
    if args.generated_video and (args.target_video or args.target_image):
        identity_result = evaluate_identity(
            result_path=args.generated_video,
            reference_image=args.target_image,
            reference_video=args.target_video,
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
        args.au_profile,
        args.generated_au,
        expected_class=args.expected_class,
        driver_au_path=args.driver_au,
        leakage_classifier_path=args.leakage_classifier,
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
    result = {
        "status": "available",
        "identity_preservation": identity_result,
        "au_compliance": au_result,
        "fusion": fused,
        "threshold_note": (
            "Calibrate hard thresholds on held-out human annotations "
            "before production blocking."
        ),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(
            serialized + "\n",
            encoding="utf-8",
        )
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
