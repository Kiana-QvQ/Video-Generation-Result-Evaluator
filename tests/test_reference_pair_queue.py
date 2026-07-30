from __future__ import annotations

import unittest
from pathlib import Path

from scripts.run_reference_pair_queue_tests import (
    build_cases,
    compact_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReferencePairQueueTests(unittest.TestCase):
    def test_cases_use_gt_and_reference_action_when_available(self) -> None:
        cases = {case["case_id"]: case for case in build_cases()}

        self.assertEqual(
            cases["seedance_test1"]["gt_video"],
            cases["seedance_test1"]["reference_video"],
        )
        self.assertEqual(
            cases["seedance_test2"]["gt_video"],
            cases["seedance_test2"]["reference_video"],
        )
        self.assertIsNone(cases["seedance_test4"]["gt_video"])
        self.assertIsNotNone(cases["seedance_test4"]["reference_video"])
        self.assertIsNone(cases["seedance_test3"]["reference_video"])

    def test_all_case_inputs_exist(self) -> None:
        for case in build_cases():
            result = PROJECT_ROOT / case["result_video"]
            self.assertTrue(result.is_file(), case["case_id"])
            for relative in case["reference_images"]:
                self.assertTrue(
                    (PROJECT_ROOT / relative).is_file(),
                    f"{case['case_id']}: {relative}",
                )
            for key in ("gt_video", "reference_video", "prompt_file"):
                relative = case.get(key)
                if relative:
                    self.assertTrue(
                        (PROJECT_ROOT / relative).is_file(),
                        f"{case['case_id']}: {relative}",
                    )

    def test_compact_result_contains_gt_aware_scores_and_au_scores(self) -> None:
        result = compact_result(
            {
                "status": "partial",
                "evaluation_mode": "full_reference",
                "coverage": "5/5",
                "weighted_score_0_100": 80.0,
                "categories": {
                    name: {"score_0_1": 0.8, "backend": "test"}
                    for name in (
                        "identity",
                        "texture",
                        "expression",
                        "temporal",
                        "aesthetics",
                    )
                },
                "wangxing_au": {
                    "status": "available",
                    "au_compliance": {
                        "selected_expression_class": "smile",
                        "personal_au_score_0_1": 0.7,
                        "driver_expression_score_0_1": 0.8,
                        "driver_temporal_alignment_score_0_1": 0.75,
                        "driver_identity_leakage_risk_0_1": 0.2,
                        "evidence_quality_status": "pass",
                        "evidence_confidence_0_1": 0.9,
                    },
                    "wangxing_targeted": {
                        "wangxing_expression_fit_score_0_1": 0.75,
                        "decision": "allow",
                    },
                    "fusion": {
                        "person_likeness_score_0_1": 0.72,
                        "decision": "allow",
                    },
                },
            }
        )

        self.assertEqual(result["evaluation_mode"], "full_reference")
        self.assertEqual(result["category_scores"]["texture"], 0.8)
        self.assertEqual(result["au_personal_score_0_1"], 0.7)
        self.assertEqual(result["au_driver_expression_score_0_1"], 0.8)
        self.assertEqual(result["au_wangxing_fit_score_0_1"], 0.75)


if __name__ == "__main__":
    unittest.main()
