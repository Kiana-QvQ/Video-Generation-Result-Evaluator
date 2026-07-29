from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluator.au_compliance import (
    AU_PROFILE_SCHEMA,
    dtw_similarity,
    fit_au_profile,
    fuse_compliance_scores,
    fuse_wangxing_targeted_scores,
    load_au_table,
    score_au_compliance,
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
