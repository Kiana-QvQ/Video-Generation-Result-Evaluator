from __future__ import annotations

import unittest

import cv2
import numpy as np

from evaluator.modules.core.face_landmarker import (
    estimate_similarity_transform,
    normalize_landmarks_pose,
)
from evaluator.modules.forensics import (
    analyze_forensics,
    extract_nr_vqa_features,
    extract_self_supervised_au_features,
    run_frame_perturbation_battery,
)
from evaluator.modules.forensics.perturbation import perturb_frames_blur
from evaluator.modules.forensics.pseudo_label_calibration import (
    build_pseudo_labeled_samples,
    fit_pseudo_label_calibrator,
)


def _frames(count: int = 8) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for index in range(count):
        frame = np.full((96, 96, 3), 55, dtype=np.uint8)
        cv2.rectangle(
            frame,
            (16 + index * 2, 24),
            (68 + index * 2, 72),
            (220, 220, 220),
            -1,
        )
        output.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return output


class AutoForensicsPipelineTests(unittest.TestCase):
    def test_pose_normalization_and_similarity_transform(self) -> None:
        landmarks = np.zeros((478, 3), dtype=np.float32)
        source = np.asarray(
            [
                [0.2, 0.3, 0.0],
                [0.8, 0.3, 0.0],
                [0.5, 0.45, 0.1],
                [0.5, 0.9, -0.05],
            ],
            dtype=np.float32,
        )
        for index, point in zip((33, 263, 1, 152), source):
            landmarks[index] = point
        normalized, matrix = normalize_landmarks_pose(landmarks)
        self.assertEqual(matrix.shape, (4, 4))
        self.assertTrue(np.isfinite(normalized).all())

        target = source * 0.5 + np.asarray([1.0, -1.0, 0.0])
        transform = estimate_similarity_transform(source, target)
        mapped = np.concatenate(
            [source, np.ones((source.shape[0], 1))],
            axis=1,
        ) @ transform.T
        np.testing.assert_allclose(mapped[:, :3], target, atol=1e-6)

    def test_self_supervised_au_features_require_no_manual_labels(self) -> None:
        signal = np.linspace(0.0, 1.0, 40, dtype=np.float64)
        result = extract_self_supervised_au_features(
            np.stack([signal, signal * 0.7], axis=1)
        )
        self.assertEqual(result["status"], "available")
        self.assertFalse(result["manual_au_labels_required"])
        self.assertTrue(
            0.0 <= result["features"]["ssl_au_score_0_1"] <= 1.0
        )

    def test_builtin_nr_vqa_is_reference_free_and_blur_sensitive(self) -> None:
        clean = extract_nr_vqa_features(
            _frames(),
            prefer_backends=("builtin_nr_vqa",),
        )
        blurred = extract_nr_vqa_features(
            perturb_frames_blur(_frames(), kernel_size=21),
            prefer_backends=("builtin_nr_vqa",),
        )
        self.assertEqual(clean["backend"], "builtin_nr_vqa")
        self.assertFalse(clean["vmaf_used"])
        self.assertGreater(clean["score_0_1"], blurred["score_0_1"])

    def test_analyze_forensics_declares_automatic_pipeline(self) -> None:
        report = analyze_forensics(texture_detail=_frames(), detect_faces=False)
        self.assertFalse(report["score_semantics"]["manual_scores_required"])
        self.assertTrue(report["auto_pipeline"]["no_reference_vqa"])
        self.assertFalse(report["auto_pipeline"]["vmaf_used"])
        self.assertIsNotNone(report["branches"]["texture_detail"])

    def test_perturbation_battery_and_pseudo_calibration(self) -> None:
        frames = _frames(10)

        def score(candidate: list[np.ndarray]) -> float:
            return float(
                extract_nr_vqa_features(
                    candidate,
                    prefer_backends=("builtin_nr_vqa",),
                )["score_0_1"]
            )

        battery = run_frame_perturbation_battery(frames, score, min_drop=0.005)
        self.assertIn("blur", battery["results"])
        self.assertGreaterEqual(battery["pass_ratio"], 0.6)

        records = [
            {
                "id": f"real-{index}",
                "source_label": "real",
                "raw_real_domain_evidence_0_1": 0.8,
                "holdout": True,
            }
            for index in range(4)
        ] + [
            {
                "id": f"generated-{index}",
                "source_label": "generated",
                "raw_real_domain_evidence_0_1": 0.2,
                "holdout": True,
            }
            for index in range(4)
        ]
        built = build_pseudo_labeled_samples(records)
        calibrator = fit_pseudo_label_calibrator(built["accepted"])
        self.assertEqual(calibrator["status"], "ready")
        self.assertFalse(calibrator["manual_scores_required"])


if __name__ == "__main__":
    unittest.main()
