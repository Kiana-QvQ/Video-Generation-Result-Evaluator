from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluator.au_compliance import (
    AU_PROFILE_SCHEMA,
    compare_temporal_events,
    dtw_similarity,
    fit_au_profile,
    fuse_compliance_scores,
    fuse_wangxing_targeted_scores,
    load_au_table,
    score_au_compliance,
    temporal_event_features,
)


def _write_au_csv(path: Path, sequence: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["AU01_r", "AU04_r", "AU06_r", "AU12_r", "AU15_r", "AU25_r"])
        writer.writerows(sequence.tolist())


def _write_libreface_au_csv(path: Path, sequence: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "au_1_intensity",
                "au_4_intensity",
                "au_6_intensity",
                "au_12_intensity",
                "au_15_intensity",
                "au_25_intensity",
            ]
        )
        writer.writerows((sequence * 5.0).tolist())


def _write_quality_au_csv(path: Path, sequence: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_idx",
        "pitch",
        "yaw",
        *[
            coordinate
            for index in range(20)
            for coordinate in (
                f"lm_mp_{index}_x",
                f"lm_mp_{index}_y",
            )
        ],
        "AU01_r",
        "AU04_r",
        "AU06_r",
        "AU12_r",
        "AU15_r",
        "AU25_r",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(sequence):
            payload = {
                "frame_idx": index,
                "pitch": 0.0,
                "yaw": 0.0,
                "AU01_r": float(row[0] * 5.0),
                "AU04_r": float(row[1] * 5.0),
                "AU06_r": float(row[2] * 5.0),
                "AU12_r": float(row[3] * 5.0),
                "AU15_r": float(row[4] * 5.0),
                "AU25_r": float(row[5] * 5.0),
            }
            for landmark_index in range(20):
                payload[f"lm_mp_{landmark_index}_x"] = (
                    0.2 + 0.03 * (landmark_index % 10)
                    if index == 0
                    else 0.0
                )
                payload[f"lm_mp_{landmark_index}_y"] = (
                    0.2 + 0.03 * (landmark_index // 10)
                    if index == 0
                    else 0.0
                )
            writer.writerow(payload)


class AUComplianceTests(unittest.TestCase):
    def test_loads_openface_style_intensity_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            _write_au_csv(path, np.ones((4, 6), dtype=np.float32) * 2.5)
            sequence, au_ids, _ = load_au_table(path)

        self.assertEqual(au_ids, (1, 4, 6, 12, 15, 25))
        self.assertEqual(sequence.shape, (4, 6))
        self.assertTrue(np.allclose(sequence, 0.5))

    def test_loads_libreface_style_intensity_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            _write_libreface_au_csv(
                path,
                np.ones((4, 6), dtype=np.float32) * 0.5,
            )
            sequence, au_ids, _ = load_au_table(path)

        self.assertEqual(au_ids, (1, 4, 6, 12, 15, 25))
        self.assertTrue(np.allclose(sequence, 0.5))

    def test_dtw_identical_sequences_score_one(self) -> None:
        sequence = np.asarray([[0, 1, 0], [0.2, 0.8, 0.1]], dtype=np.float32)
        self.assertAlmostEqual(dtw_similarity(sequence, sequence), 1.0)

    def test_temporal_event_alignment_is_one_for_identical_sequences(self) -> None:
        sequence = np.asarray(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0.8, 0.4, 0, 0.7, 0, 0],
                [0.9, 0.5, 0, 0.8, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        features = temporal_event_features(sequence)
        alignment = compare_temporal_events(features, features)
        self.assertAlmostEqual(
            alignment["event_alignment_score_0_1"],
            1.0,
        )

    def test_face_quality_gate_marks_low_quality_evidence_uncertain(self) -> None:
        base = np.asarray(
            [[0, 0, 1, 1, 0, 0], [0.1, 0, 0.9, 0.8, 0, 0.2]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            generated_path = root / "generated.csv"
            fit_au_profile([("smile", base), ("smile", base)], profile_path)
            _write_quality_au_csv(
                generated_path,
                np.vstack([base, base]),
            )
            result = score_au_compliance(
                profile_path,
                generated_path,
                expected_class="smile",
            )

        self.assertEqual(
            result["generated_au"]["quality"]["status"],
            "uncertain",
        )
        self.assertEqual(result["evidence_quality_status"], "uncertain")
        self.assertIn("face_quality_low", result["uncertainty_reasons"])

    def test_profile_scores_matching_class(self) -> None:
        base = np.asarray(
            [[0, 0, 1, 1, 0, 0], [0.1, 0, 0.9, 0.8, 0, 0.2]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            generated_path = root / "generated.csv"
            fit_au_profile([("smile", base), ("smile", base * 0.98)], profile_path)
            _write_au_csv(generated_path, base)
            result = score_au_compliance(
                profile_path,
                generated_path,
                expected_class="smile",
            )
            payload = json.loads(profile_path.read_text(encoding="utf-8"))

        self.assertEqual(result["selected_expression_class"], "smile")
        self.assertGreater(result["personal_au_score_0_1"], 0.5)
        self.assertEqual(payload["schema_version"], AU_PROFILE_SCHEMA)

    def test_fusion_blocks_high_leakage(self) -> None:
        result = fuse_compliance_scores(
            identity_score_0_1=0.9,
            personal_au_score_0_1=0.8,
            driver_expression_score_0_1=0.8,
            leakage_risk_0_1=0.7,
        )
        self.assertEqual(result["decision"], "block")
        self.assertIn("driver_identity_leakage", result["decision_reasons"])

    def test_wangxing_targeted_fit_does_not_require_identity_or_driver(self) -> None:
        result = fuse_wangxing_targeted_scores(
            personal_au_score_0_1=0.8,
            driver_expression_score_0_1=None,
            leakage_risk_0_1=0.1,
        )
        self.assertEqual(result["decision"], "allow")
        self.assertAlmostEqual(
            result["wangxing_expression_fit_score_0_1"],
            0.8,
        )

    def test_wangxing_targeted_fit_reviews_low_personal_au(self) -> None:
        result = fuse_wangxing_targeted_scores(
            personal_au_score_0_1=0.3,
            driver_expression_score_0_1=None,
            leakage_risk_0_1=0.1,
        )
        self.assertEqual(result["decision"], "review")
        self.assertIn(
            "wangxing_au_below_threshold",
            result["decision_reasons"],
        )

    def test_wangxing_targeted_fit_reviews_uncertain_evidence(self) -> None:
        result = fuse_wangxing_targeted_scores(
            personal_au_score_0_1=0.8,
            driver_expression_score_0_1=None,
            leakage_risk_0_1=0.1,
            evidence_quality_status="uncertain",
            evidence_confidence_0_1=0.2,
            uncertainty_reasons=["face_quality_low"],
        )
        self.assertEqual(result["decision"], "review")
        self.assertIn("evidence_quality_low", result["decision_reasons"])
