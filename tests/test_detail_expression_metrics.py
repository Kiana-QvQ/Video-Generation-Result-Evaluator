from __future__ import annotations

import unittest
from unittest.mock import patch

import evaluator.detail_expression_metrics as public_api
import evaluator.core.detail_expression_runtime as runtime


class DetailExpressionMetricsTests(unittest.TestCase):
    def test_detail_public_api_delegates_to_packaged_runtime(self) -> None:
        generated = object()
        reference = object()
        payload = {
            "score": 0.61,
            "status": "available",
            "details": {"method": "test-detail-runtime"},
        }
        with (
            patch.object(
                public_api,
                "prepare_video_input",
                side_effect=[generated, reference],
            ) as prepare,
            patch.object(
                public_api,
                "score_detail_quality",
                return_value=payload,
            ) as score,
        ):
            result = public_api.compute_detail_metric(
                "generated.mp4",
                "reference.mp4",
                ["reference.png"],
                8,
            )

        self.assertEqual(result.score, 0.61)
        self.assertAlmostEqual(result.score_0_100, 61.0)
        self.assertEqual(result.status, "available")
        self.assertEqual(result.details["method"], "test-detail-runtime")
        self.assertEqual(prepare.call_count, 2)
        score.assert_called_once_with(
            generated=generated,
            reference=reference,
            reference_images=["reference.png"],
            max_frames=8,
        )

    def test_expression_public_api_delegates_without_reference_video(self) -> None:
        generated = object()
        payload = {
            "score": 0.57,
            "status": "partial",
            "details": {"method": "test-expression-runtime"},
        }
        with (
            patch.object(
                public_api,
                "prepare_video_input",
                return_value=generated,
            ) as prepare,
            patch.object(
                public_api,
                "score_face_expression",
                return_value=payload,
            ) as score,
        ):
            result = public_api.compute_face_expression_metric(
                {"path": "generated.mp4"},
                None,
                None,
                4,
            )

        self.assertEqual(result.score, 0.57)
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.details["method"], "test-expression-runtime")
        prepare.assert_called_once_with(
            {"path": "generated.mp4"},
            max_frames=4,
        )
        score.assert_called_once_with(
            generated=generated,
            reference=None,
            reference_images=None,
            max_frames=4,
        )

    def test_public_api_rejects_too_few_frames(self) -> None:
        with self.assertRaises(ValueError):
            public_api.compute_detail_metric("generated.mp4", None, None, 1)

    def test_expression_composite_matches_web_radar_formula(self) -> None:
        generated = runtime.PreparedVideo(
            frames=[object()],
            frame_count=1,
            au_csv="generated.csv",
        )
        forensics = {
            "branches": {
                "facial_motion": {
                    "metrics": {
                        "raw_real_domain_evidence_0_1": 0.542,
                        "au_relation_consistency_0_1": 0.554,
                        "au_dynamics_naturalness_0_1": 0.997,
                        "landmark_valid_frame_ratio": 1.0,
                    },
                },
            },
        }
        expression = {
            "compatibility_0_1": 0.818,
            "selected_profile_display_name": "sadness",
            "selected_profile": "sadness",
            "event_statistics": {"active_ratio": 0.997},
        }
        with (
            patch.object(
                runtime,
                "_load_expression_profile",
                return_value=({}, "expression.json"),
            ),
            patch.object(
                runtime,
                "_load_expression_result",
                return_value=expression,
            ),
            patch.object(
                runtime,
                "_run_specialization_forensics",
                return_value=forensics,
            ),
        ):
            result = runtime.score_face_expression(
                generated=generated,
                reference=None,
                reference_images=None,
                max_frames=16,
            )

        self.assertAlmostEqual(result["score"], 0.7822, places=6)
        self.assertAlmostEqual(
            result["details"]["composite_score_0_100"],
            78.22,
            places=6,
        )
        self.assertEqual(
            result["details"]["selected_profile_display_name"],
            "sadness",
        )

    def test_texture_composite_matches_web_radar_formula(self) -> None:
        generated = runtime.PreparedVideo(
            frames=[object()],
            frame_count=1,
        )
        forensics = {
            "branches": {
                "texture_detail": {
                    "metrics": {
                        "raw_real_domain_evidence_0_1": 0.554,
                        "micro_temporal_naturalness_0_1": 0.887,
                        "optical_flow_homogeneity_0_1": 0.941,
                        "real_domain_fit_0_1": 0.654,
                    },
                },
            },
            "scores": {"raw_real_domain_evidence_0_1": 0.547},
        }
        with (
            patch.object(
                runtime,
                "_load_forensics_profiles",
                return_value={"texture_detail": {}},
            ),
            patch.object(
                runtime,
                "_run_specialization_forensics",
                return_value=forensics,
            ),
        ):
            result = runtime.score_detail_quality(
                generated=generated,
                reference=None,
                reference_images=None,
                max_frames=16,
            )

        self.assertAlmostEqual(result["score"], 0.7166, places=6)
        self.assertAlmostEqual(
            result["details"]["composite_score_0_100"],
            71.66,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
