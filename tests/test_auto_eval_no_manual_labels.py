from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluator.modules.core.face_landmarker import (
    estimate_similarity_transform,
    normalize_landmarks_pose,
)
from evaluator.modules.forensics.au_ssl import extract_self_supervised_au_features
from evaluator.modules.forensics.facial_motion import extract_facial_motion_features
from evaluator.modules.forensics.nr_vqa import extract_nr_vqa_features
from evaluator.modules.forensics.perturbation import (
    perturb_frames_blur,
    run_frame_perturbation_battery,
)
from evaluator.modules.forensics.pseudo_label_calibration import (
    build_pseudo_labeled_samples,
    consensus_pseudo_label,
    fit_pseudo_label_calibrator,
)
from evaluator.modules.forensics.texture_detail import extract_texture_detail_features


def _synthetic_frames(count: int = 8, *, sharp: bool = True) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for index in range(count):
        frame = np.zeros((96, 96, 3), dtype=np.uint8)
        frame[:, :] = (40, 50, 60)
        if sharp:
            cv_x = 20 + (index % 5) * 8
            frame[30:70, cv_x : cv_x + 4] = (220, 220, 220)
            frame[40:44, 20:70] = (200, 180, 160)
        else:
            frame[:] = (80 + index, 80, 80)
        frames.append(frame)
    return frames


def _write_motion_csv(path: Path) -> None:
    fieldnames = [
        "frame_idx",
        "AU01_r",
        "AU06_r",
        "AU12_r",
        *[
            coordinate
            for index in (10, 13, 14, 33, 61, 152, 234, 263, 291, 454)
            for coordinate in (
                f"lm_mp_{index}_x",
                f"lm_mp_{index}_y",
                f"lm_mp_{index}_z",
            )
        ],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(12):
            signal = index / 11.0
            row = {
                "frame_idx": index,
                "AU01_r": signal,
                "AU06_r": signal * 0.8,
                "AU12_r": signal * 0.6,
                "lm_mp_10_x": 0.5,
                "lm_mp_10_y": 0.10,
                "lm_mp_10_z": 0.0,
                "lm_mp_152_x": 0.5,
                "lm_mp_152_y": 0.90,
                "lm_mp_152_z": 0.0,
                "lm_mp_234_x": 0.10,
                "lm_mp_234_y": 0.5,
                "lm_mp_234_z": 0.0,
                "lm_mp_454_x": 0.90,
                "lm_mp_454_y": 0.5,
                "lm_mp_454_z": 0.0,
                "lm_mp_33_x": 0.30,
                "lm_mp_33_y": 0.40,
                "lm_mp_33_z": 0.0,
                "lm_mp_263_x": 0.70,
                "lm_mp_263_y": 0.40,
                "lm_mp_263_z": 0.0,
                "lm_mp_13_x": 0.5,
                "lm_mp_13_y": 0.48 - signal * 0.02,
                "lm_mp_13_z": 0.02,
                "lm_mp_14_x": 0.5,
                "lm_mp_14_y": 0.52 + signal * 0.02,
                "lm_mp_14_z": 0.02,
                "lm_mp_61_x": 0.40,
                "lm_mp_61_y": 0.50,
                "lm_mp_61_z": 0.01,
                "lm_mp_291_x": 0.60,
                "lm_mp_291_y": 0.50,
                "lm_mp_291_z": 0.01,
            }
            writer.writerow(row)


class AutoEvalNoManualLabelTests(unittest.TestCase):
    def test_pose_normalization_maps_anchors_near_canonical(self) -> None:
        source = np.asarray(
            [
                [0.2, 0.3, 0.0],
                [0.8, 0.3, 0.0],
                [0.5, 0.45, 0.1],
                [0.5, 0.9, -0.05],
            ],
            dtype=np.float64,
        )
        # Build a dense landmark array and place anchors at known indices.
        landmarks = np.zeros((478, 3), dtype=np.float32)
        for index, point in zip((33, 263, 1, 152), source):
            landmarks[index] = point
        normalized, matrix = normalize_landmarks_pose(landmarks)
        self.assertEqual(matrix.shape, (4, 4))
        self.assertLess(
            float(np.linalg.norm(normalized[33] - np.asarray([-0.35, 0.05, 0.0]))),
            0.15,
        )

    def test_similarity_transform_roundtrip_scale(self) -> None:
        source = np.asarray(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=np.float64,
        )
        target = source * 0.5 + np.asarray([1.0, -1.0, 0.0])
        matrix = estimate_similarity_transform(source, target)
        ones = np.ones((3, 1))
        mapped = np.concatenate([source, ones], axis=1) @ matrix.T
        self.assertTrue(np.allclose(mapped[:, :3], target, atol=1e-6))

    def test_facial_motion_includes_ssl_and_pose_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.csv"
            _write_motion_csv(path)
            result = extract_facial_motion_features(path)

        self.assertIn("ssl_au_score_0_1", result["features"])
        self.assertIn("pose_normalized_frame_ratio", result["features"])
        self.assertGreater(result["features"]["pose_normalized_frame_ratio"], 0.5)
        self.assertEqual(result["self_supervised_au"]["status"], "available")

    def test_self_supervised_au_features_are_finite(self) -> None:
        matrix = np.clip(
            np.linspace(0.0, 1.0, 40).reshape(20, 2)
            + 0.05 * np.random.default_rng(0).normal(size=(20, 2)),
            0.0,
            1.0,
        )
        result = extract_self_supervised_au_features(matrix)
        self.assertEqual(result["status"], "available")
        self.assertGreaterEqual(result["features"]["ssl_au_score_0_1"], 0.0)
        self.assertLessEqual(result["features"]["ssl_au_score_0_1"], 1.0)

    def test_nr_vqa_builtin_backend(self) -> None:
        sharp = extract_nr_vqa_features(
            _synthetic_frames(sharp=True),
            prefer_backends=("builtin_nr_vqa",),
        )
        blurry_frames = perturb_frames_blur(_synthetic_frames(sharp=True), kernel_size=21)
        blurry = extract_nr_vqa_features(
            blurry_frames,
            prefer_backends=("builtin_nr_vqa",),
        )
        self.assertEqual(sharp["backend"], "builtin_nr_vqa")
        self.assertFalse(sharp["vmaf_used"])
        self.assertGreater(sharp["score_0_1"], blurry["score_0_1"])

    def test_texture_includes_nr_vqa(self) -> None:
        result = extract_texture_detail_features(
            _synthetic_frames(),
            detect_faces=False,
            include_nr_vqa=True,
        )
        self.assertIn("nr_vqa_score_0_1", result["features"])
        self.assertIsNotNone(result.get("nr_vqa"))

    def test_pseudo_label_calibration_from_source_labels(self) -> None:
        records = []
        for index in range(6):
            records.append(
                {
                    "id": f"real_{index}",
                    "source_label": "real",
                    "raw_real_domain_evidence_0_1": 0.75 + 0.02 * index,
                    "holdout": True,
                }
            )
            records.append(
                {
                    "id": f"gen_{index}",
                    "source_label": "generated",
                    "raw_real_domain_evidence_0_1": 0.25 - 0.02 * index,
                    "holdout": True,
                }
            )
        labeled = build_pseudo_labeled_samples(records)
        calibrator = fit_pseudo_label_calibrator(labeled["accepted"])
        self.assertEqual(calibrator["status"], "ready")
        self.assertFalse(calibrator["manual_scores_required"])

    def test_consensus_pseudo_label_requires_agreement(self) -> None:
        good = consensus_pseudo_label([0.82, 0.80, 0.79])
        bad = consensus_pseudo_label([0.9, 0.1, 0.5])
        self.assertEqual(good["status"], "high_confidence")
        self.assertEqual(good["pseudo_label"], 1)
        self.assertEqual(bad["status"], "low_agreement")
        self.assertIsNone(bad["pseudo_label"])

    def test_perturbation_battery_expects_score_drop(self) -> None:
        frames = _synthetic_frames(count=10, sharp=True)

        def score_fn(candidate):
            return float(
                extract_nr_vqa_features(
                    candidate,
                    prefer_backends=("builtin_nr_vqa",),
                )["score_0_1"]
            )

        report = run_frame_perturbation_battery(
            frames,
            score_fn,
            min_drop=0.005,
        )
        self.assertGreaterEqual(report["pass_ratio"], 0.6)
        self.assertIn("blur", report["results"])
        self.assertTrue(report["results"]["blur"]["passed"])


if __name__ == "__main__":
    unittest.main()
