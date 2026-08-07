from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from evaluator.forensics import (
    analyze_forensics,
    build_two_domain_facial_motion_profile,
    extract_facial_motion_features,
    extract_texture_detail_features,
    score_facial_motion,
    score_texture_detail,
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
