from __future__ import annotations

import unittest

from wangxing_project.runtime_display_v53 import (
    apply_content_gate,
    apply_manifest_display,
    validate_runtime_manifest,
)


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


if __name__ == "__main__":
    unittest.main()
