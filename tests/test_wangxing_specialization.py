from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from evaluator.wangxing.wangxing_specialization import (
    _fit_logistic_calibrator,
    _identity_calibration_metrics,
    _weighted_prototype,
    evaluate_identity_profile,
    score_expression_profile,
    sequence_feature_names,
)


class _FakeIdentityBackend:
    backend = "fake"

    def embedding(self, frame: np.ndarray):
        del frame
        return (
            np.asarray([1.0, 0.0], dtype=np.float32),
            (20, 10, 60, 60),
            "fake",
        )


def _write_video(path: Path, frame_count: int = 4) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (96, 96),
    )
    try:
        for _ in range(frame_count):
            writer.write(np.full((96, 96, 3), 128, dtype=np.uint8))
    finally:
        writer.release()


class WangxingSpecializationTests(unittest.TestCase):
    def test_quality_weighted_prototype_prefers_valid_face_frames(self) -> None:
        prototype = _weighted_prototype(
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            [1.0, 0.25],
        )
        self.assertGreater(float(prototype[0]), float(prototype[1]))

    def test_identity_calibrator_separates_positive_and_negative_rows(self) -> None:
        features = np.asarray(
            [
                [0.9, 0.9, 0.9, 0.1, 0.8, 0.95, 1.0, 0.9],
                [0.8, 0.85, 0.82, 0.2, 0.62, 0.90, 1.0, 0.8],
                [0.1, 0.2, 0.15, 0.9, -0.75, 0.92, 1.0, 0.9],
                [0.2, 0.1, 0.15, 0.8, -0.65, 0.88, 1.0, 0.8],
            ],
            dtype=np.float64,
        )
        labels = np.asarray([1, 1, 0, 0], dtype=np.int32)
        calibrator = _fit_logistic_calibrator(features, labels)
        scores = np.asarray(
            [
                # The test uses the public shape of the saved calibrator.
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        location = np.asarray(calibrator["location"])
        scale = np.asarray(calibrator["scale"])
        coefficients = np.asarray(calibrator["coefficients"])
        standardized = (features - location) / scale
        scores = 1.0 / (
            1.0
            + np.exp(
                -(
                    float(calibrator["intercept"])
                    + standardized @ coefficients
                )
            )
        )
        self.assertGreater(float(np.mean(scores[:2])), 0.75)
        self.assertLess(float(np.mean(scores[2:])), 0.25)
        metrics = _identity_calibration_metrics(labels, scores)
        self.assertGreaterEqual(float(metrics["roc_auc"]), 0.99)
        self.assertGreaterEqual(float(metrics["pr_auc"]), 0.99)

    def test_identity_gate_returns_wangxing_for_consistent_positive_track(self) -> None:
        profile = {
            "real_prototype": [1.0, 0.0],
            "generated_prototype": [1.0, 0.0],
            "negative_prototypes": [
                {"name": "actor_01", "prototype": [0.0, 1.0]},
            ],
            "thresholds": {
                "positive_gap_floor": 0.5,
                "negative_gap_ceiling": -0.2,
                "positive_probability_floor": 0.5,
                "negative_probability_ceiling": 0.5,
                "min_valid_frame_count": 3,
                "min_valid_frame_ratio": 0.5,
                "min_frame_consistency": 0.7,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "positive.mp4"
            _write_video(video_path)
            result = evaluate_identity_profile(
                video_path,
                profile,
                _FakeIdentityBackend(),
                max_frames=4,
            )
        self.assertEqual(result["decision"], "wangxing")
        self.assertGreater(result["probability_0_1"], 0.5)
        self.assertGreater(result["frame_consistency"], 0.99)

    def test_expression_compatibility_uses_max_support_domain_score(self) -> None:
        feature_count = len(sequence_feature_names())
        quality = {
            "frame_count": 12,
            "valid_frame_ratio": 1.0,
            "event_statistics": {"event_count": 1},
        }
        classes = {}
        for index, expression_class in enumerate(
            ("smile", "anger", "surprise", "fear", "sadness", "disgust")
        ):
            classes[expression_class] = {
                "display_name": expression_class,
                "location": [float(index)] * feature_count,
                "scale": [1.0] * feature_count,
                "distance_threshold": 1.0,
                "sample_count": 10,
            }
        profile = {"classes": classes}

        with patch(
            "evaluator.wangxing_specialization.extract_sequence_features",
            return_value=(
                np.zeros(feature_count, dtype=np.float32),
                quality,
            ),
        ):
            result = score_expression_profile(
                "unused.csv",
                profile,
                expected_class="anger",
            )

        self.assertEqual(result["profile_winner"], "smile")
        self.assertEqual(result["selected_profile"], "anger")
        self.assertEqual(result["compatibility_0_1"], 1.0)
        self.assertLess(result["expected_profile_score_0_1"], 1.0)
        self.assertEqual(len(result["most_compatible_profiles"]), 2)


if __name__ == "__main__":
    unittest.main()
