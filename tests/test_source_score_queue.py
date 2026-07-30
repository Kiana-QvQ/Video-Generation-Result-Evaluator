from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_source_score_queue_tests import (
    compare_groups,
    discover_cases,
    compact_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SourceScoreQueueTests(unittest.TestCase):
    def test_discovers_all_four_source_cohorts(self) -> None:
        cases = discover_cases(PROJECT_ROOT, limit_per_cohort=1)

        self.assertEqual(
            {case["cohort"] for case in cases},
            {"real_video", "real_md_cl", "seedance", "tests_data"},
        )
        self.assertEqual(len(cases), 4)
        self.assertTrue(all(Path(case["path"]).is_file() for case in cases))

    def test_compact_result_keeps_holistic_and_au_scores(self) -> None:
        result = compact_result(
            {
                "status": "partial",
                "coverage": "5/5",
                "weighted_score_0_100": 72.5,
                "categories": {
                    name: {
                        "score_0_1": value,
                        "backend": "test",
                    }
                    for name, value in {
                        "identity": 0.8,
                        "texture": 0.7,
                        "expression": 0.6,
                        "temporal": 0.75,
                        "aesthetics": 0.65,
                    }.items()
                },
                "wangxing_au": {
                    "status": "available",
                    "au_compliance": {
                        "selected_expression_class": "smile",
                        "personal_au_score_0_1": 0.82,
                        "driver_expression_score_0_1": 0.71,
                        "driver_temporal_alignment_score_0_1": 0.68,
                        "driver_identity_leakage_risk_0_1": 0.12,
                        "evidence_quality_status": "pass",
                        "evidence_confidence_0_1": 0.91,
                    },
                    "wangxing_targeted": {
                        "wangxing_expression_fit_score_0_1": 0.74,
                        "decision": "allow",
                    },
                    "fusion": {
                        "person_likeness_score_0_1": 0.79,
                        "decision": "allow",
                    },
                },
            }
        )

        self.assertEqual(result["weighted_score_0_100"], 72.5)
        self.assertEqual(result["category_scores"]["identity"], 0.8)
        self.assertEqual(result["au_personal_score_0_1"], 0.82)
        self.assertEqual(result["au_leakage_risk_0_1"], 0.12)
        self.assertEqual(result["au_wangxing_fit_score_0_1"], 0.74)

    def test_comparison_exposes_score_direction(self) -> None:
        def item(cohort: str, score: float) -> dict:
            return {
                "case": {"cohort": cohort},
                "result": {
                    "weighted_score_0_100": score,
                    "category_scores": {"identity": score / 100.0},
                    "au_personal_score_0_1": score / 100.0,
                    "au_leakage_risk_0_1": 1.0 - score / 100.0,
                    "au_wangxing_fit_score_0_1": score / 100.0,
                    "au_evidence_confidence_0_1": 0.9,
                },
            }

        items = [
            item("real_video", 80),
            item("real_video", 85),
            item("seedance", 30),
            item("seedance", 35),
        ]
        comparison = compare_groups(items, "real_video", "seedance")

        field = comparison["fields"]["au_personal_score_0_1"]
        self.assertEqual(field["auc_left_higher"], 1.0)
        self.assertEqual(field["preferred_higher_cohort"], "real_video")

    def test_output_directory_is_workspace_relative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
