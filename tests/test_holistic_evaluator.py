from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from evaluator.holistic_evaluator import WEIGHTS, evaluate_all
from evaluator.video_metrics import VideoInfo, _aligned_sample_indices


def _write_video(path: Path, shift: int = 0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (64, 64),
    )
    if not writer.isOpened():
        raise RuntimeError("Unable to create test video")
    try:
        for index in range(8):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            left = 8 + index * 2 + shift
            cv2.rectangle(frame, (left, 18), (left + 18, 42), (80, 160, 220), -1)
            cv2.circle(frame, (left + 9, 25), 3, (230, 230, 230), -1)
            writer.write(frame)
    finally:
        writer.release()


class HolisticEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Tests should not download optional IQA weights from the network.
        cls._previous_disable_iqa = os.environ.get(
            "EVALUATOR_DISABLE_OPTIONAL_IQA"
        )
        os.environ["EVALUATOR_DISABLE_OPTIONAL_IQA"] = "1"

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._previous_disable_iqa is None:
            os.environ.pop("EVALUATOR_DISABLE_OPTIONAL_IQA", None)
        else:
            os.environ["EVALUATOR_DISABLE_OPTIONAL_IQA"] = cls._previous_disable_iqa

    def test_reference_sampling_uses_common_timestamps(self) -> None:
        result_info = VideoInfo("result", 30.0, 31, 64, 64, 31 / 30)
        gt_info = VideoInfo("gt", 15.0, 16, 64, 64, 16 / 15)

        count, result_indices, gt_indices, timestamps = _aligned_sample_indices(
            result_info,
            gt_info,
            max_frames=4,
        )

        self.assertEqual(count, 4)
        self.assertEqual(result_indices.tolist(), [0, 10, 20, 30])
        self.assertEqual(gt_indices.tolist(), [0, 5, 10, 15])
        self.assertAlmostEqual(float(timestamps[-1]), 1.0, places=5)

    def test_full_reference_path_covers_five_categories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.mp4"
            gt_path = root / "gt.mp4"
            _write_video(result_path)
            _write_video(gt_path, shift=1)

            result = evaluate_all(
                result_path,
                gt_path,
                None,
                None,
                max_frames=4,
                calculate_lpips=False,
                device="cpu",
                manual_expression_score=3,
                manual_aesthetic_score=4,
            )

            self.assertEqual(set(result["categories"]), set(WEIGHTS))
            self.assertEqual(sum(WEIGHTS.values()), 100)
            self.assertEqual(result["categories"]["texture"]["mode"], "full_reference")
            self.assertEqual(result["categories"]["temporal"]["mode"], "reference_flow")
            self.assertEqual(result["evaluation_mode"], "full_reference")
            self.assertIsNotNone(result["categories"]["texture"]["metrics"]["psnr_db"])
            self.assertIsNotNone(result["categories"]["texture"]["metrics"]["ssim"])
            self.assertIn("lpips", result["categories"]["texture"]["metrics"])
            self.assertEqual(len(result["summary"]), 5)
            self.assertGreater(len(result["frame_records"]), 0)
            self.assertIn("weighted_score_0_1", result)
            self.assertIn("gt_frame", result["frame_records"][0])

    def test_no_reference_path_uses_manual_and_self_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.mp4"
            _write_video(result_path)

            result = evaluate_all(
                result_path,
                None,
                None,
                None,
                max_frames=4,
                calculate_lpips=False,
                device="cpu",
                manual_expression_score=2,
                manual_aesthetic_score=5,
            )

            self.assertEqual(result["categories"]["texture"]["mode"], "no_gt")
            self.assertEqual(result["evaluation_mode"], "result_only")
            self.assertIsNone(result["categories"]["texture"]["metrics"]["psnr_db"])
            self.assertIsNone(result["categories"]["texture"]["metrics"]["ssim"])
            self.assertIsNone(result["categories"]["texture"]["metrics"]["lpips"])
            self.assertEqual(result["categories"]["expression"]["mode"], "manual")
            self.assertEqual(result["categories"]["temporal"]["mode"], "self_warping")
            self.assertEqual(result["categories"]["aesthetics"]["status"], "manual")
            self.assertEqual(result["categories"]["aesthetics"]["metrics"]["manual_score_1_to_5"], 5)
            self.assertEqual(len(result["summary"]), 5)
            self.assertIn("weighted_score_weight_coverage", result)


if __name__ == "__main__":
    unittest.main()
