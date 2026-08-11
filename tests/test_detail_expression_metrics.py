from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import evaluator.detail_expression_metrics as public_api
import evaluator.modules.core.detail_expression_runtime as runtime
import evaluator.modules.core.paths as paths


class DetailExpressionMetricsTests(unittest.TestCase):
    def test_nested_package_resolves_parent_host_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_root = root / "evaluator"
            package_root.mkdir()
            (package_root / "detail_expression_metrics.py").write_text(
                "",
                encoding="utf-8",
            )
            with patch.object(paths, "PACKAGE_ROOT", package_root):
                self.assertEqual(paths._detect_workspace_root(), root)

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

    def test_relative_au_csv_is_resolved_next_to_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "candidate.mp4"
            au_csv = root / "candidate.csv"
            video.write_bytes(b"placeholder")
            au_csv.write_text("AU01_r\n0.5\n", encoding="utf-8")

            prepared = runtime.prepare_video_input(
                {
                    "path": str(video),
                    "au_csv": au_csv.name,
                    "frames": [object(), object()],
                },
                max_frames=2,
            )

        self.assertEqual(prepared.au_csv, str(au_csv.resolve()))

    def test_expression_compatibility_evidence_preserves_zero(self) -> None:
        details = public_api._compat_expression_details(
            0.5,
            {
                "profile_compatibility_0_1": 0.0,
                "muscle_action_evidence_0_1": 0.25,
                "action_coherence_0_1": 0.5,
                "landmark_coverage_0_1": 0.75,
            },
        )

        self.assertEqual(details["evidence"][0]["value"], "0.0%")
        self.assertEqual(details["evidence"][1]["value"], "25.0%")

    def test_expression_reports_actual_valid_face_frame_count(self) -> None:
        generated = runtime.PreparedVideo(
            frames=[np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(8)],
            frame_count=8,
            au_csv="generated.csv",
        )
        forensics = {
            "branches": {
                "facial_motion": {
                    "metrics": {
                        "raw_real_domain_evidence_0_1": 0.5,
                        "au_relation_consistency_0_1": 0.5,
                        "au_dynamics_naturalness_0_1": 0.5,
                        "landmark_valid_frame_ratio": 0.25,
                    },
                },
            },
        }
        expression = {
            "compatibility_0_1": 0.5,
            "event_statistics": {"active_ratio": 0.5},
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
                max_frames=8,
            )

        self.assertEqual(result["status"], "available")
        self.assertEqual(result["details"]["valid_face_frames"], 2)

    def test_expression_without_au_is_explicitly_partial(self) -> None:
        generated = runtime.PreparedVideo(
            frames=[object(), object()],
            frame_count=2,
        )

        with patch.object(
            runtime,
            "_run_specialization_forensics",
            return_value={},
        ):
            result = runtime.score_face_expression(
                generated=generated,
                reference=None,
                reference_images=None,
                max_frames=2,
            )

        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["score"])
        self.assertIn("AU CSV", result["details"]["warning"])

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
