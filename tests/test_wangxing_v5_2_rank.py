"""Unit tests for the V5.2 grouped linear RankHead contract."""

from __future__ import annotations

import unittest

from wangxing_project.cascade_v5 import cascade_score_v51, cascade_score_v52
from wangxing_project.rank_head_v52 import (
    group_pair_rows,
    fit_rank_policy,
    predict_rank_score,
)
from wangxing_project.realness_v5 import realness_feature_dict


def _row(group: str, label: str, value: float) -> dict[str, object]:
    return {
        "group_id": group,
        "label": label,
        "v3": {"p_real": value},
        "realness": {
            "s_realness": value,
            "z_raw": value,
            "features": realness_feature_dict(
                p_drive_eff=value,
                s_direction=value,
                p_v3_real=value,
            ),
        },
        "forensics": {
            "components": {
                "direction_details": {
                    "temporal_naturalness_0_1": value,
                    "texture_stability_0_1": value,
                    "frequency_naturalness_0_1": value,
                    "ai_domain_inverse_0_1": 1.0 - value,
                }
            }
        },
        "v5": {
            "score_display": value,
            "s_rank": None,
        },
    }


class WangxingV52RankTests(unittest.TestCase):
    def test_partial_groups_only_create_available_pairs(self) -> None:
        rows = [
            _row("G1", "lora", 0.6),
            _row("G1", "multiref", 0.3),
        ]
        pairs, inventory = group_pair_rows(rows)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(inventory["complete_group_count"], 0)
        self.assertEqual(inventory["missing_roles"]["G1"], ["real", "seedance"])

    def test_insufficient_data_is_disabled(self) -> None:
        rows = [
            _row("G1", "real", 0.9),
            _row("G1", "lora", 0.7),
            _row("G1", "seedance", 0.5),
            _row("G1", "multiref", 0.2),
        ]
        policy = fit_rank_policy(
            rows=rows,
            fit_groups=["G1"],
            holdout_groups=[],
            min_complete_groups_fit=4,
            min_pairs_fit=12,
        )
        self.assertEqual(
            policy["disabled_reason"],
            "disabled_insufficient_data",
        )
        self.assertFalse(policy["rank_model"]["enabled"])

    def test_linear_rank_can_fit_grouped_pairs(self) -> None:
        rows = []
        for index in range(4):
            group = f"G{index}"
            rows.extend(
                [
                    _row(group, "real", 0.95 - index * 0.01),
                    _row(group, "lora", 0.70 - index * 0.01),
                    _row(group, "seedance", 0.45 - index * 0.01),
                    _row(group, "multiref", 0.20 - index * 0.01),
                ]
            )
        policy = fit_rank_policy(
            rows=rows,
            fit_groups=["G0", "G1", "G2", "G3"],
            holdout_groups=[],
            min_complete_groups_fit=4,
            min_pairs_fit=12,
        )
        self.assertTrue(policy["rank_model"]["enabled"])
        score, status = predict_rank_score(rows[0], policy)
        self.assertEqual(status["status"], "ok")
        self.assertIsNotNone(score)

    def test_rank_disabled_falls_back_to_v51(self) -> None:
        realness = {
            "s_realness": 0.8,
            "s_direction": 0.7,
            "z_raw": 0.7,
            "realness_status": "ok",
        }
        v51 = cascade_score_v51(
            p_v3_real=0.2,
            p_drive=0.9,
            p_drive_eff=0.9,
            realness=realness,
            realness_enabled=True,
        )
        v52 = cascade_score_v52(
            p_v3_real=0.2,
            p_drive=0.9,
            p_drive_eff=0.9,
            realness=realness,
            rank_score=0.9,
            rank_policy={
                "schema_version": "wangxing_v5_2_rank_policy_v1",
                "usable_for_runtime": False,
                "ordering_satisfied": False,
                "rank_model": {"enabled": True},
            },
            rank_enabled=True,
        )
        self.assertEqual(v52["decision"], v51["decision"])
        self.assertAlmostEqual(v52["score_display"], v51["score_display"])
        self.assertIsNone(v52["band_hint"])
        self.assertLess(v52["score_display"], 0.75)

    def test_resolve_disabled_reason_preserves_insufficient_data(self) -> None:
        from wangxing_project.rank_head_v52 import resolve_disabled_reason

        reason = resolve_disabled_reason(
            policy={
                "disabled_reason": "disabled_insufficient_data",
                "rank_model": {"enabled": False},
            },
            rank_usable=False,
        )
        self.assertEqual(reason, "disabled_insufficient_data")

    def test_rank_metrics_without_s_rank_are_not_usable(self) -> None:
        from wangxing_project.rank_head_v52 import rank_metrics

        rows = [
            _row("G1", "real", 0.9),
            _row("G1", "lora", 0.7),
            _row("G1", "seedance", 0.5),
            _row("G1", "multiref", 0.2),
        ]
        metrics = rank_metrics(rows)
        self.assertEqual(metrics["score_source"], "score_display_fallback")
        self.assertFalse(metrics["rank_available"])
        self.assertFalse(metrics["ordering_satisfied"])

    def test_usable_rank_does_not_change_decision_or_realness_display(self) -> None:
        realness = {
            "s_realness": 0.4,
            "s_direction": 0.4,
            "z_raw": 0.4,
            "realness_status": "ok",
        }
        v51 = cascade_score_v51(
            p_v3_real=0.1,
            p_drive=0.8,
            p_drive_eff=0.8,
            realness=realness,
            realness_enabled=True,
        )
        v52 = cascade_score_v52(
            p_v3_real=0.1,
            p_drive=0.8,
            p_drive_eff=0.8,
            realness=realness,
            rank_score=0.99,
            rank_policy={
                "schema_version": "wangxing_v5_2_rank_policy_v1",
                "usable_for_runtime": True,
                "ordering_satisfied": True,
                "rank_model": {"enabled": True},
                "fit_metrics": {
                    "class_mean_scores_0_1": {
                        "lora": 0.8,
                        "seedance": 0.5,
                        "multiref": 0.2,
                    }
                },
            },
            rank_enabled=True,
        )
        self.assertEqual(v52["decision"], "generated")
        self.assertEqual(v52["decision"], v51["decision"])
        self.assertAlmostEqual(v52["score_display"], v51["score_display"])
        self.assertLess(v52["score_display"], 0.75)
        self.assertEqual(v52["band_hint"], "lora")


if __name__ == "__main__":
    unittest.main()
