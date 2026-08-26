from __future__ import annotations

import unittest

from scripts.pt_training.evaluate_wangxing_v5_3_runtime import (
    _group_order_satisfied,
    _group_ordering_metrics,
    _pairwise_ordering_metrics,
)
from wangxing_project.runtime_display_v53 import (
    apply_content_gate,
    apply_manifest_display,
    validate_runtime_manifest,
)


def _row(group_id: str, role: str, score: float) -> dict:
    return {
        "group_id": group_id,
        "manifest_role": role,
        "v5": {"score_display_final": score},
    }


class WangxingV53RuntimeTests(unittest.TestCase):
    def test_content_gate_never_flips_v3_decision(self) -> None:
        result = apply_content_gate(
            {
                "decision": "generated",
                "p_v3_real": 0.12,
                "score_display": 0.42,
                "s_realness": 0.91,
                "realness_status": "ok",
                "s_rank": 0.1,
            },
            policy={
                "content_gate": {
                    "enabled": True,
                    "T_high": 0.8,
                    "T_rank_cap": None,
                }
            },
            enabled=True,
        )
        self.assertEqual(result["decision"], "generated")
        self.assertTrue(result["content_gate_applied"])
        self.assertTrue(result["prior_conflict_display"])
        self.assertGreaterEqual(result["score_display_final"], 0.75)

    def test_non_finite_score_degrades_without_ui_score(self) -> None:
        result = apply_content_gate(
            {
                "decision": "generated",
                "score_display": None,
                "s_realness": 0.9,
                "realness_status": "ok",
            },
            policy={"content_gate": {"enabled": True, "T_high": 0.8}},
            enabled=True,
        )
        self.assertEqual(result["runtime_display_mode"], "degraded")
        self.assertEqual(result["fallback_reason"], "non_finite_score")
        self.assertIsNone(result["score_display_final"])

    def test_manifest_real_display_does_not_use_role_anchor(self) -> None:
        result = apply_manifest_display(
            {
                "decision": "generated",
                "score_display": 0.42,
                "s_realness": 0.8,
            },
            role="real",
        )
        self.assertEqual(result["runtime_display_mode"], "manifest_explicit")
        self.assertFalse(result["role_anchor_applied"])
        self.assertTrue(result["prior_conflict_display"])
        self.assertAlmostEqual(result["score_display_final"], 0.95)

    def test_manifest_requires_explicit_roles(self) -> None:
        errors = validate_runtime_manifest(
            {
                "schema_version": "wangxing_v5_3_runtime_manifest_v1",
                "runtime_mode": "web_regression",
                "groups": [
                    {
                        "group_id": "g1",
                        "matching_key": "q1",
                        "runtime_role_source": "manifest_explicit",
                        "completeness": "partial",
                        "videos": {"real": "real.mp4"},
                    }
                ],
            }
        )
        self.assertIn("g1:missing_roles=['lora', 'multiref', 'seedance']", errors)
        self.assertIn("g1:not_full", errors)

    def test_group_order_satisfied_when_scores_monotonic(self) -> None:
        rows = [
            _row("g1", "real", 0.95),
            _row("g1", "lora", 0.64),
            _row("g1", "seedance", 0.35),
            _row("g1", "multiref", 0.01),
        ]
        self.assertTrue(_group_order_satisfied(rows))
        metrics = _group_ordering_metrics(rows)
        self.assertEqual(metrics["complete_group_count"], 1)
        self.assertEqual(metrics["group_order_satisfied_count"], 1)
        pairwise = _pairwise_ordering_metrics(rows)
        self.assertEqual(pairwise["pairwise_total"], 6)
        self.assertEqual(pairwise["pairwise_correct"], 6)
        self.assertAlmostEqual(pairwise["pairwise_ordering_rate"], 1.0)

    def test_group_order_fails_when_seedance_above_lora(self) -> None:
        rows = [
            _row("g1", "real", 0.95),
            _row("g1", "lora", 0.30),
            _row("g1", "seedance", 0.35),
            _row("g1", "multiref", 0.01),
        ]
        self.assertFalse(_group_order_satisfied(rows))
        metrics = _group_ordering_metrics(rows)
        self.assertEqual(metrics["failed_groups"], ["g1"])


if __name__ == "__main__":
    unittest.main()
