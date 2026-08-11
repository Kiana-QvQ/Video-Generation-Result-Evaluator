from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from evaluator.modules.forensics import (
    analyze_forensics,
    apply_probability_calibrator,
    build_two_domain_facial_motion_profile,
    extract_facial_motion_features,
    extract_texture_detail_features,
    fit_probability_calibrator,
    score_facial_motion,
    score_texture_detail,
    summarize_window_evidence,
)
from scripts.calibrate_forensics import (
    _brier_score,
    _cached_scored_samples,
    _cross_fitted_probabilities,
    _expected_calibration_error,
)


def _write_motion_csv(path: Path, amplitude: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_idx",
        "AU01_r",
        "AU06_r",
        "AU12_r",
        *[
            coordinate
            for index in range(478)
            for coordinate in (
                f"lm_mp_{index}_x",
                f"lm_mp_{index}_y",
            )
        ],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(12):
            signal = amplitude * (index / 11.0)
            row = {
                "frame_idx": index,
                "AU01_r": signal,
                "AU06_r": signal,
                "AU12_r": signal,
            }
            for landmark_index in range(478):
                row[f"lm_mp_{landmark_index}_x"] = 0.5
                row[f"lm_mp_{landmark_index}_y"] = 0.5
            row.update(
                {
                    "lm_mp_10_x": 0.5,
                    "lm_mp_10_y": 0.10,
                    "lm_mp_152_x": 0.5,
                    "lm_mp_152_y": 0.90,
                    "lm_mp_234_x": 0.10,
                    "lm_mp_234_y": 0.5,
                    "lm_mp_454_x": 0.90,
                    "lm_mp_454_y": 0.5,
                    "lm_mp_13_y": 0.48 - signal * 0.02,
                    "lm_mp_14_y": 0.52 + signal * 0.02,
                    "lm_mp_61_y": 0.50 - signal * 0.01,
                    "lm_mp_291_y": 0.50 - signal * 0.01,
                }
            )
            writer.writerow(row)


class ForensicsInitialTests(unittest.TestCase):
    def test_facial_motion_extracts_temporal_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.csv"
            _write_motion_csv(path, 1.0)
            result = extract_facial_motion_features(path)

        self.assertEqual(result["frame_count"], 12)
        self.assertTrue(result["landmark_available"])
        self.assertGreater(result["features"]["au_event_active_ratio"], 0.0)
        self.assertIn("au_01_velocity_p95", result["features"])
        self.assertIn("landmark_phase_coherence_0_1", result["features"])

    def test_time_aware_derivatives_use_csv_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timed_motion.csv"
            path.write_text(
                "frame_idx,frame_time_in_ms,AU01_r\n"
                "0,0,0.0\n"
                "1,100,1.0\n"
                "2,300,2.0\n",
                encoding="utf-8",
            )
            result = extract_facial_motion_features(
                path,
                time_aware_derivatives=True,
            )

        self.assertEqual(result["timebase"], "frame_time_in_ms_seconds")
        self.assertTrue(result["time_aware_derivatives"])
        self.assertGreater(result["features"]["au_01_velocity_p95"], 9.0)

    def test_facial_motion_two_domain_profile_is_calibrated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real.csv"
            generated = root / "generated.csv"
            candidate = root / "candidate.csv"
            _write_motion_csv(real, 1.0)
            _write_motion_csv(generated, 0.1)
            _write_motion_csv(candidate, 1.0)
            profile = build_two_domain_facial_motion_profile([real], [generated])
            result = score_facial_motion(candidate, profile)

        self.assertEqual(result["status"], "calibrated")
        self.assertIsNotNone(
            result["metrics"]["real_capture_likelihood_0_1"]
        )

    def test_facial_motion_ignores_missing_landmark_features(self) -> None:
        profile = {
            "domain": "real_vs_seedance",
            "feature_names": [
                "au_01_median",
                "landmark_brow_left_median",
                "motion_coherence_0_1",
            ],
            "real": {
                "mean": [0.5, 0.5, 0.8],
                "std": [0.1, 0.1, 0.1],
            },
            "seedance": {
                "mean": [0.2, 0.9, 0.1],
                "std": [0.1, 0.1, 0.1],
            },
        }
        result = score_facial_motion(
            {
                "features": {
                    "au_01_median": 0.5,
                    "landmark_brow_left_median": 0.0,
                    "motion_coherence_0_1": 0.0,
                    "landmark_valid_frame_ratio": 0.0,
                },
                "landmark_available": False,
                "frame_count": 12,
            },
            profile,
        )

        self.assertEqual(result["metrics"]["feature_mode"], "au_only")
        self.assertEqual(result["metrics"]["scored_feature_count"], 1)

    def test_window_summary_reports_mean_and_worst_window(self) -> None:
        summary = summarize_window_evidence(
            [
                {"window_index": 0, "evidence_score_0_1": 0.2},
                {"window_index": 1, "evidence_score_0_1": 0.8},
            ]
        )

        self.assertEqual(summary["window_count"], 2)
        self.assertEqual(summary["worst_window"]["window_index"], 1)
        self.assertAlmostEqual(summary["mean_evidence_score_0_1"], 0.5)
        self.assertAlmostEqual(summary["aggregate_evidence_score_0_1"], 0.65)

    def test_texture_features_have_temporal_residuals(self) -> None:
        frames = []
        for index in range(4):
            frame = np.zeros((96, 96, 3), dtype=np.uint8)
            cv2.rectangle(
                frame,
                (20 + index, 20),
                (76 + index, 76),
                (180, 180, 180),
                -1,
            )
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = extract_texture_detail_features(frames)

        self.assertEqual(result["frame_count"], 4)
        self.assertIn(
            "temporal_warp_residual_mean",
            result["features"],
        )
        scored = score_texture_detail(result)
        self.assertEqual(scored["status"], "features_only")
        self.assertIsNone(
            scored["metrics"]["real_capture_likelihood_0_1"]
        )

    def test_report_keeps_branches_separate(self) -> None:
        frames = [
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.ones((32, 32, 3), dtype=np.uint8) * 10,
        ]
        result = analyze_forensics(texture_detail=frames)

        self.assertEqual(result["schema_version"], "video_forensics_report_v1")
        self.assertIsNotNone(result["branches"]["texture_detail"])
        self.assertIsNone(result["branches"]["facial_motion"])
        self.assertEqual(result["fusion"]["method"], "not_calibrated")
        self.assertIn("facial_expression_muscle_score_0_1", result["scores"])
        self.assertIn("texture_detail_score_0_1", result["scores"])
        self.assertTrue(result["score_semantics"]["not_expression_correctness"])

    def test_uncalibrated_profile_cannot_make_source_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real.csv"
            generated = root / "generated.csv"
            candidate = root / "candidate.csv"
            _write_motion_csv(real, 1.0)
            _write_motion_csv(generated, 0.1)
            _write_motion_csv(candidate, 1.0)
            profile = build_two_domain_facial_motion_profile(
                [real],
                [generated],
            )
            result = analyze_forensics(
                facial_motion=candidate,
                facial_motion_profile=profile,
            )

        self.assertEqual(result["status"], "profile_evidence_only")
        self.assertEqual(result["authenticity"]["decision"], "uncertain")
        self.assertIsNone(
            result["scores"]["real_capture_likelihood_0_1"]
        )
        self.assertIsNotNone(
            result["scores"]["raw_real_domain_evidence_0_1"]
        )

    def test_ready_calibrator_exposes_probability(self) -> None:
        calibrator = fit_probability_calibrator(
            [0.75, 0.85],
            [0.15, 0.25],
        )
        self.assertEqual(calibrator["status"], "ready")
        self.assertIsNotNone(
            apply_probability_calibrator(0.8, calibrator)
        )

    def test_calibration_metrics_are_computed_from_probabilities(self) -> None:
        labels = [1, 1, 0, 0]
        probabilities = [0.9, 0.8, 0.2, 0.1]
        self.assertAlmostEqual(
            _brier_score(labels, probabilities),
            0.025,
        )
        self.assertIsNotNone(
            _expected_calibration_error(labels, probabilities)
        )

    def test_cross_fitted_calibration_probabilities_are_out_of_fold(self) -> None:
        probabilities = _cross_fitted_probabilities(
            [0.70, 0.80, 0.90],
            [0.10, 0.20, 0.30],
        )
        self.assertEqual(len(probabilities), 6)
        self.assertTrue(all(value is not None for value in probabilities))
        self.assertGreater(probabilities[0], probabilities[-1])

    def test_cached_calibration_samples_must_match_holdout_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cached.json"
            real_au = Path(directory) / "real.csv"
            seedance_au = Path(directory) / "seedance.csv"
            real_video = Path(directory) / "real.mp4"
            seedance_video = Path(directory) / "seedance.mp4"
            cached = {
                "samples": [
                    {
                        "domain": "real",
                        "au_path": str(real_au),
                        "video_path": str(real_video),
                        "raw_real_domain_evidence_0_1": 0.8,
                    },
                    {
                        "domain": "seedance",
                        "au_path": str(seedance_au),
                        "video_path": str(seedance_video),
                        "raw_real_domain_evidence_0_1": 0.2,
                    },
                ]
            }
            path.write_text(json.dumps(cached), encoding="utf-8")
            real, seedance = _cached_scored_samples(
                path,
                real_samples=[
                    {
                        "domain": "real",
                        "au_path": real_au,
                        "video_path": real_video,
                    }
                ],
                seedance_samples=[
                    {
                        "domain": "seedance",
                        "au_path": seedance_au,
                        "video_path": seedance_video,
                    }
                ],
            )

        self.assertEqual(len(real), 1)
        self.assertEqual(len(seedance), 1)
        self.assertEqual(
            real[0]["raw_real_domain_evidence_0_1"],
            0.8,
        )
