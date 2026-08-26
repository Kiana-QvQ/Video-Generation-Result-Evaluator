from __future__ import annotations

import unittest
from unittest.mock import patch

from wangxing_project.web_forensics_display import (
    WEB_V52_NEUTRAL_LABEL,
    apply_v52_forensics_display,
    infer_v52_for_web,
    patch_wangxing_au_forensics_for_v52,
    resync_result_forensics_from_v5,
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

    def test_display_metadata_marks_no_filename_or_role_anchor(self) -> None:
        forensics = {"scores": {}, "fusion": {}, "authenticity": {}}
        apply_v52_forensics_display(
            forensics,
            {"decision": "generated", "score_display": 0.35},
        )
        meta = forensics["wangxing_v5_display"]
        self.assertFalse(meta["filename_label_inference"])
        self.assertFalse(meta["role_anchor_applied"])

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

    def test_infer_v52_uses_neutral_label_and_no_role_anchor(self) -> None:
        fake_row = {
            "v5": {
                "p_v3_real": 0.18,
                "p_drive": 0.4,
                "p_drive_eff": 0.4,
            },
            "realness": {"s_realness": 0.5},
            "prior_conflict": True,
        }
        fake_v5 = {
            "decision": "generated",
            "score_display": 0.346,
            "prior_conflict": False,
        }
        with patch(
            "wangxing_project.web_forensics_display._load_web_v52_context",
            return_value={
                "profiles": {},
                "source_profile": {},
                "calibrator": {},
                "v3_model": "model.pt",
                "drive_model": {},
                "cache_dir": "cache",
                "rank_policy": {"rank_model": {"enabled": True}},
            },
        ), patch(
            "wangxing_project.v51_runtime.build_feature_row",
            return_value=fake_row.copy(),
        ) as mock_build, patch(
            "wangxing_project.rank_head_v52.predict_rank_score",
            return_value=(0.24, {"status": "ok"}),
        ), patch(
            "wangxing_project.cascade_v5.cascade_score_v52",
            return_value=fake_v5,
        ) as mock_cascade, patch(
            "wangxing_project.cascade_v5.anchor_ranking_real_display",
        ) as mock_anchor:
            result = infer_v52_for_web(
                video_path="outputs/web_runs/job/result.mp4",
                au_path="au.csv",
                device="cpu",
            )
        self.assertEqual(result, fake_v5)
        mock_build.assert_called_once()
        build_kwargs = mock_build.call_args.kwargs
        self.assertEqual(build_kwargs["label"], WEB_V52_NEUTRAL_LABEL)
        mock_cascade.assert_called_once()
        cascade_kwargs = mock_cascade.call_args.kwargs
        self.assertFalse(cascade_kwargs["prior_conflict"])
        mock_anchor.assert_not_called()

    def test_infer_v52_same_for_real_named_and_generic_paths(self) -> None:
        fake_row = {
            "v5": {"p_v3_real": 0.2, "p_drive": 0.3, "p_drive_eff": 0.3},
            "realness": {"s_realness": 0.4},
        }
        fake_v5 = {"decision": "generated", "score_display": 0.25}
        with patch(
            "wangxing_project.web_forensics_display._load_web_v52_context",
            return_value={
                "profiles": {},
                "source_profile": {},
                "calibrator": {},
                "v3_model": "model.pt",
                "drive_model": {},
                "cache_dir": "cache",
                "rank_policy": None,
            },
        ), patch(
            "wangxing_project.v51_runtime.build_feature_row",
            return_value=fake_row.copy(),
        ) as mock_build, patch(
            "wangxing_project.cascade_v5.cascade_score_v52",
            return_value=fake_v5,
        ):
            infer_v52_for_web(
                video_path="ppt_test2/真人视频.mp4",
                au_path="au.csv",
                device="cpu",
            )
            infer_v52_for_web(
                video_path="outputs/web_runs/job/result.mp4",
                au_path="au.csv",
                device="cpu",
            )
        labels = [
            call.kwargs["label"]
            for call in mock_build.call_args_list
        ]
        self.assertEqual(labels, [WEB_V52_NEUTRAL_LABEL, WEB_V52_NEUTRAL_LABEL])

    def test_resync_keeps_v3_conclusion_with_high_display(self) -> None:
        result = {
            "wangxing_au": {
                "status": "available",
                "forensics": {
                    "scores": {"calibrated_real_probability_0_1": 0.1},
                    "fusion": {},
                    "authenticity": {},
                },
                "wangxing_v5": {
                    "status": "available",
                    "decision": "generated",
                    "score_display": 0.921,
                    "p_v3_real": 0.12,
                },
            }
        }
        resync_result_forensics_from_v5(result)
        forensics = result["wangxing_au"]["forensics"]
        self.assertAlmostEqual(
            forensics["scores"]["calibrated_real_probability_0_1"],
            0.921,
        )
        self.assertEqual(
            forensics["authenticity"]["binary_decision"],
            "seedance_like",
        )
        self.assertFalse(
            forensics["wangxing_v5_display"]["role_anchor_applied"]
        )


if __name__ == "__main__":
    unittest.main()
