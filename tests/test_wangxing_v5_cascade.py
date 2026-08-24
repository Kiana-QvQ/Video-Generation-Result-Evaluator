"""Unit tests for Wang Xing V5 cascade invariants and runtime flags."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from wangxing_project.cascade_v5 import cascade_score
from wangxing_project.drive_head_v5 import (
    apply_drive_quality_gate,
    coverage_from_drive_vector,
    _augment_hardneg_vector,
)
from wangxing_project.v5_flags import v5_runtime_flags


class WangxingV5CascadeTests(unittest.TestCase):
    def test_lexicographic_real_band_above_ai(self) -> None:
        real = cascade_score(p_v3_real=0.92, p_drive=0.40, p_drive_eff=0.40)
        ai = cascade_score(p_v3_real=0.20, p_drive=0.95, p_drive_eff=0.95)
        self.assertEqual(real["decision"], "real")
        self.assertEqual(ai["decision"], "generated")
        self.assertGreaterEqual(real["score_display"], 0.75)
        self.assertLess(ai["score_display"], 0.75)
        self.assertGreater(real["score_display"], ai["score_display"])
        self.assertTrue(real["decision_invariant"])
        self.assertEqual(real["decision_source"], "v3_frozen")

    def test_drive_cannot_flip_v3_decision(self) -> None:
        # Extremely high drive score still cannot turn AI into real.
        result = cascade_score(
            p_v3_real=0.10,
            p_drive=0.99,
            p_drive_eff=0.99,
        )
        self.assertEqual(result["decision"], "generated")
        self.assertEqual(result["score_band"], "ai_unspecified")

    def test_rank_disabled_without_env_flag(self) -> None:
        with patch.dict(os.environ, {"V5_RANK_ENABLED": "0"}, clear=False):
            result = cascade_score(
                p_v3_real=0.10,
                p_drive=0.40,
                p_drive_eff=0.40,
                rank_score=0.90,
                rank_policy={
                    "usable_for_runtime": True,
                    "ordering_satisfied": True,
                },
            )
        self.assertFalse(result["rank_enabled"])
        self.assertEqual(result["score_band"], "ai_unspecified")

    def test_runtime_flags_default_off(self) -> None:
        with patch.dict(
            os.environ,
            {
                "V5_DRIVE_ENABLED": "",
                "V5_RANK_ENABLED": "",
                "V5_DISPLAY_CASCADE": "",
            },
            clear=False,
        ):
            # Empty string should fall back to defaults (false).
            flags = v5_runtime_flags()
        self.assertFalse(flags["V5_DRIVE_ENABLED"])
        self.assertFalse(flags["V5_RANK_ENABLED"])
        self.assertFalse(flags["V5_DISPLAY_CASCADE"])
        self.assertFalse(flags["V5_REALNESS_ENABLED"])

    def test_quality_gate_low_coverage(self) -> None:
        effective, meta = apply_drive_quality_gate(0.90, coverage_q=0.10)
        self.assertEqual(effective, 0.5)
        self.assertEqual(meta["status"], "unavailable")

    def test_hardneg_augmentation_changes_vector(self) -> None:
        rng = np.random.default_rng(0)
        source = np.linspace(0.0, 1.0, 16, dtype=np.float32)
        augmented = _augment_hardneg_vector(source, rng=rng)
        self.assertEqual(augmented.shape, source.shape)
        self.assertFalse(np.allclose(augmented, source))

    def test_coverage_from_named_vector(self) -> None:
        names = (
            "face_geometry_valid_ratio_0_1",
            "face_detection_confidence_0_1",
            "other",
        )
        vector = np.asarray([0.8, 0.6, 0.1], dtype=np.float32)
        self.assertAlmostEqual(
            coverage_from_drive_vector(vector, names),
            0.7,
        )


if __name__ == "__main__":
    unittest.main()
