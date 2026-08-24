"""Unit tests for the V5.1 realness axis and calibrator contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wangxing_project.cascade_v5 import cascade_score, cascade_score_v51
from wangxing_project.realness_v5 import (
    FORBIDDEN_FEATURES,
    REALNESS_FEATURE_NAMES,
    fit_isotonic_calibrator,
    load_calibrator,
    predict_realness,
    realness_feature_dict,
    validate_weights,
    write_calibrator,
)
from wangxing_project.v51_runtime import lexicographic_metrics, rank_metrics


def _row(label: str, value: float) -> dict[str, object]:
    features = realness_feature_dict(
        p_drive_eff=value,
        s_direction=value,
        p_v3_real=value,
    )
    return {
        "label": label,
        "realness_features": features,
    }


class WangxingV51RealnessTests(unittest.TestCase):
    def test_feature_contract_excludes_compatibility(self) -> None:
        self.assertEqual(
            REALNESS_FEATURE_NAMES,
            ("p_drive_eff", "s_direction", "p_v3_real"),
        )
        self.assertTrue(
            "expression_profile.compatibility_0_1"
            in FORBIDDEN_FEATURES
        )

    def test_weight_constraints_limit_v3(self) -> None:
        weights = validate_weights(
            {
                "p_drive_eff": 0.01,
                "s_direction": 0.01,
                "p_v3_real": 0.98,
            }
        )
        self.assertLessEqual(weights["p_v3_real"], 0.30)
        self.assertGreaterEqual(
            weights["p_drive_eff"] + weights["s_direction"],
            0.70,
        )

    def test_calibrator_roundtrip(self) -> None:
        rows = [
            _row("multiref", 0.10),
            _row("seedance", 0.35),
            _row("lora", 0.65),
            _row("real", 0.95),
        ]
        calibrator = fit_isotonic_calibrator(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = write_calibrator(
                Path(directory) / "calibrator.json",
                calibrator,
            )
            loaded = load_calibrator(path)
        self.assertIsNotNone(loaded)
        result = predict_realness(
            features=_row("real", 0.90)["realness_features"],
            calibrator=loaded,
            enabled=True,
        )
        self.assertEqual(result["realness_status"], "ok")
        self.assertGreaterEqual(float(result["s_realness"]), 0.0)
        self.assertLessEqual(float(result["s_realness"]), 1.0)

    def test_v51_fallback_matches_v50_display(self) -> None:
        v50 = cascade_score(
            p_v3_real=0.20,
            p_drive=0.80,
            p_drive_eff=0.80,
        )
        v51 = cascade_score_v51(
            p_v3_real=0.20,
            p_drive=0.80,
            p_drive_eff=0.80,
            realness=None,
            realness_enabled=False,
        )
        self.assertEqual(v51["schema_version"], "wangxing_v5_1_result_v1")
        self.assertEqual(v51["decision"], v50["decision"])
        self.assertAlmostEqual(
            v51["score_display"],
            v50["score_display"],
        )
        self.assertIsNone(v51["s_realness"])

    def test_missing_calibrator_uses_v50_fallback(self) -> None:
        features = realness_feature_dict(
            p_drive_eff=0.80,
            s_direction=0.10,
            p_v3_real=0.20,
        )
        realness = predict_realness(
            features=features,
            calibrator=None,
            enabled=True,
        )
        self.assertIsNone(realness["s_realness"])
        self.assertEqual(realness["realness_status"], "disabled")
        v51 = cascade_score_v51(
            p_v3_real=0.20,
            p_drive=0.80,
            p_drive_eff=0.80,
            realness=realness,
            realness_enabled=True,
        )
        self.assertAlmostEqual(v51["score_display"], 0.74 * 0.80)
        self.assertEqual(v51["rank_reason"], "v5_0_fallback")

    def test_v51_quality_cannot_flip_decision(self) -> None:
        realness = predict_realness(
            features=realness_feature_dict(
                p_drive_eff=0.99,
                s_direction=0.99,
                p_v3_real=0.01,
            ),
            enabled=True,
        )
        result = cascade_score_v51(
            p_v3_real=0.10,
            p_drive=0.99,
            p_drive_eff=0.99,
            realness=realness,
            realness_enabled=True,
        )
        self.assertEqual(result["decision"], "generated")
        self.assertTrue(result["decision_invariant"])
        self.assertLess(result["score_display"], 0.75)

    def test_v51_high_calibrated_realness_stays_below_ai_band(self) -> None:
        rows = [
            _row("multiref", 0.10),
            _row("seedance", 0.35),
            _row("lora", 0.65),
            _row("real", 0.95),
        ]
        calibrator = fit_isotonic_calibrator(rows)
        realness = {
            "s_realness": 0.95,
            "realness_status": "ok",
            "calibrator_id": calibrator["schema_version"],
        }
        result = cascade_score_v51(
            p_v3_real=0.10,
            p_drive=0.99,
            p_drive_eff=0.99,
            realness=realness,
            realness_enabled=True,
        )
        self.assertEqual(result["decision"], "generated")
        self.assertLess(result["score_display"], 0.75)
        self.assertAlmostEqual(result["score_display"], 0.74 * 0.95, places=6)

    def test_v51_prior_conflict_flag_preserved(self) -> None:
        result = cascade_score_v51(
            p_v3_real=0.90,
            p_drive=0.20,
            p_drive_eff=0.20,
            realness={"s_realness": 0.80, "realness_status": "ok"},
            realness_enabled=True,
            prior_conflict=True,
        )
        self.assertTrue(result["prior_conflict"])
        self.assertEqual(result["decision"], "real")

    def test_v51_lexicographic_with_calibrator(self) -> None:
        rows = [
            _row("multiref", 0.10),
            _row("seedance", 0.35),
            _row("lora", 0.65),
            _row("real", 0.95),
        ]
        calibrator = fit_isotonic_calibrator(rows)
        report_rows: list[dict[str, object]] = []
        for label, p_v3 in (("real", 0.90), ("seedance", 0.10)):
            features = realness_feature_dict(
                p_drive_eff=0.80 if label == "real" else 0.20,
                s_direction=0.80 if label == "real" else 0.20,
                p_v3_real=p_v3,
            )
            realness = predict_realness(
                features=features,
                calibrator=calibrator,
                enabled=True,
            )
            cascade = cascade_score_v51(
                p_v3_real=p_v3,
                p_drive=0.80 if label == "real" else 0.20,
                p_drive_eff=0.80 if label == "real" else 0.20,
                realness=realness,
                realness_enabled=True,
            )
            report_rows.append({"label": label, "v5": cascade})
        metrics = lexicographic_metrics(report_rows)
        self.assertTrue(metrics["lexicographic_satisfied"])
        self.assertGreater(
            float(metrics["min_real_score_display"]),
            float(metrics["max_ai_score_display"]),
        )

    def test_rank_metrics_min_pairwise_threshold(self) -> None:
        rows = [
            {
                "label": "real",
                "v5": {"score_display": 0.95, "decision": "real"},
                "decision_matches_v3": True,
            },
            {
                "label": "lora",
                "v5": {"score_display": 0.80, "decision": "generated"},
                "decision_matches_v3": True,
            },
            {
                "label": "lora",
                "v5": {"score_display": 0.75, "decision": "generated"},
                "decision_matches_v3": True,
            },
            {
                "label": "seedance",
                "v5": {"score_display": 0.76, "decision": "generated"},
                "decision_matches_v3": True,
            },
            {
                "label": "multiref",
                "v5": {"score_display": 0.40, "decision": "generated"},
                "decision_matches_v3": True,
            },
        ]
        metrics = rank_metrics(rows, min_pairwise=5.0 / 6.0)
        self.assertAlmostEqual(metrics["min_pairwise_threshold"], 5.0 / 6.0)
        self.assertEqual(metrics["pairwise_total"], 9)
        self.assertEqual(metrics["pairwise_correct"], 8)
        self.assertAlmostEqual(metrics["pairwise_ordering_rate"], 8.0 / 9.0)
        self.assertTrue(metrics["class_ordering_satisfied"])
        self.assertTrue(metrics["ordering_satisfied"])
        strict = rank_metrics(rows, min_pairwise=1.0)
        self.assertFalse(strict["ordering_satisfied"])


if __name__ == "__main__":
    unittest.main()
