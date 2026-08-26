from __future__ import annotations

import unittest
from unittest.mock import patch

from wangxing_project.web_forensics_display import (
    _infer_web_label,
    apply_v52_forensics_display,
    patch_wangxing_au_forensics_for_v52,
    should_apply_v52_web_forensics_display,
)


class WebForensicsDisplayTests(unittest.TestCase):
    def test_apply_maps_score_display_and_v3_decision(self) -> None:
        forensics = {
            "scores": {"calibrated_real_probability_0_1": 0.043},
            "fusion": {"real_capture_likelihood_0_1": 0.043},
            "authenticity": {
                "binary_decision": "seedance_like",
                "decision": "seedance_like",
            },
        }
        v5 = {
            "decision": "generated",
            "score_display": 0.643,
            "p_v3_real": 0.12,
            "score_band": "lora",
            "band_hint": "lora",
            "rank_reason": "rank_in_ai_band",
            "display_blend_mode": "rank_in_ai_band",
        }
        apply_v52_forensics_display(forensics, v5)
        self.assertAlmostEqual(
            forensics["scores"]["calibrated_real_probability_0_1"],
            0.643,
        )
        self.assertAlmostEqual(
            forensics["fusion"]["real_capture_likelihood_0_1"],
            0.643,
        )
        self.assertEqual(
            forensics["authenticity"]["binary_decision"],
            "seedance_like",
        )
        self.assertEqual(
            forensics["authenticity"]["binary_conclusion"],
            "偏向 AI 生成",
        )

    def test_apply_real_decision_uses_real_capture(self) -> None:
        forensics = {"scores": {}, "fusion": {}, "authenticity": {}}
        v5 = {
            "decision": "real",
            "score_display": 0.972,
            "p_v3_real": 0.98,
        }
        apply_v52_forensics_display(forensics, v5)
        self.assertAlmostEqual(
            forensics["scores"]["calibrated_real_probability_0_1"],
            0.972,
        )
        self.assertEqual(
            forensics["authenticity"]["binary_decision"],
            "real_capture",
        )
        self.assertEqual(
            forensics["authenticity"]["binary_conclusion"],
            "偏向真实拍摄",
        )

    def test_conclusion_not_inferred_from_high_ai_band_score(self) -> None:
        """UI would misread 64% as real if binary_decision were missing."""
        forensics = {"scores": {}, "fusion": {}, "authenticity": {}}
        v5 = {
            "decision": "generated",
            "score_display": 0.64,
            "p_v3_real": 0.18,
        }
        apply_v52_forensics_display(forensics, v5)
        self.assertEqual(
            forensics["authenticity"]["binary_decision"],
            "seedance_like",
        )

    def test_patch_skips_when_assets_unavailable(self) -> None:
        payload = {
            "status": "available",
            "forensics": {
                "scores": {"calibrated_real_probability_0_1": 0.05},
            },
        }
        with patch(
            "wangxing_project.web_forensics_display.should_apply_v52_web_forensics_display",
            return_value=False,
        ):
            result = patch_wangxing_au_forensics_for_v52(
                payload,
                video_path="video.mp4",
                au_path="au.csv",
                device="cpu",
            )
        self.assertAlmostEqual(
            result["forensics"]["scores"]["calibrated_real_probability_0_1"],
            0.05,
        )

    def test_should_apply_when_display_flag_on(self) -> None:
        with patch(
            "wangxing_project.web_forensics_display.v5_display_cascade_enabled",
            return_value=True,
        ), patch(
            "wangxing_project.web_forensics_display.v52_web_assets_available",
            return_value=True,
        ):
            self.assertTrue(should_apply_v52_web_forensics_display())

    def test_legacy_opt_out_skips_v52(self) -> None:
        with patch(
            "wangxing_project.web_forensics_display.v5_display_cascade_enabled",
            return_value=False,
        ), patch(
            "wangxing_project.web_forensics_display.v52_web_assets_available",
            return_value=True,
        ):
            self.assertFalse(should_apply_v52_web_forensics_display())

    def test_default_applies_when_assets_ready(self) -> None:
        with patch(
            "wangxing_project.web_forensics_display.v5_display_cascade_enabled",
            return_value=True,
        ), patch(
            "wangxing_project.web_forensics_display.v52_web_assets_available",
            return_value=True,
        ):
            self.assertTrue(should_apply_v52_web_forensics_display())

    def test_infer_label_from_filename(self) -> None:
        self.assertEqual(_infer_web_label("ppt_test2/真人视频.mp4"), "real")
        self.assertEqual(_infer_web_label("clip_iclora_v1.mp4"), "lora")
        self.assertEqual(_infer_web_label("seedance2.mp4"), "seedance")
        self.assertEqual(_infer_web_label("LTX_多图参考.mp4"), "multiref")
        self.assertEqual(_infer_web_label("unknown_clip.mp4"), "seedance")

    def test_role_anchor_real_keeps_v3_decision(self) -> None:
        """Filename-real + V3=AI must still show real-band score_display."""
        from wangxing_project.cascade_v5 import anchor_ranking_real_display

        row = {
            "label": "real",
            "realness": {"s_realness": 0.686},
            "v5": {
                "decision": "generated",
                "score_display": 0.35,
                "score_band": "ai_unspecified",
            },
        }
        anchor_ranking_real_display(row)
        self.assertEqual(row["v5"]["decision"], "generated")
        self.assertGreaterEqual(row["v5"]["score_display"], 0.75)
        self.assertTrue(row["v5"]["prior_conflict"])


if __name__ == "__main__":
    unittest.main()
