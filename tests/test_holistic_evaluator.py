from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from evaluator.holistic_evaluator import (
    TEMPORAL_LANDMARK_INDICES,
    WEIGHTS,
    _expression_descriptors,
    _face_box_jitter,
    _lower_tail,
    _motion_direction_similarity,
    evaluate_all,
)
from evaluator.video_metrics import VideoInfo, _aligned_sample_indices, _mean


def _write_video(
    path: Path,
    shift: int = 0,
    size: tuple[int, int] = (64, 64),
) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError("Unable to create test video")
    try:
        for index in range(8):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
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

    def test_identity_tail_uses_lowest_scores(self) -> None:
        self.assertEqual(_lower_tail([0.95, 0.2, 0.8, 0.7], 0.5), [0.2, 0.7])

    def test_non_finite_metric_values_do_not_poison_the_mean(self) -> None:
        self.assertEqual(_mean([1.0, float("inf"), None]), 1.0)

    def test_static_motion_matches_static_motion(self) -> None:
        vector = np.zeros(4, dtype=np.float32)
        self.assertEqual(
            _motion_direction_similarity(vector, 0.0, vector, 0.0),
            1.0,
        )
        self.assertEqual(
            _motion_direction_similarity(vector, 0.0, np.ones(4), 1.0),
            0.0,
        )

    def test_expression_proxy_keeps_one_descriptor_shape_when_landmarks_miss(self) -> None:
        class Detector:
            def detect(self, frame: np.ndarray) -> None:
                return None

        class PartialTracker:
            available = True

            def extract(self, frame: np.ndarray) -> np.ndarray | None:
                if int(frame[0, 0, 0]) == 1:
                    return np.ones((468, 3), dtype=np.float32)
                return None

        frames = [
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.ones((16, 16, 3), dtype=np.uint8),
        ]
        descriptors, backend = _expression_descriptors(
            frames,
            Detector(),
            PartialTracker(),
        )
        self.assertEqual(descriptors.shape, (2, 576))
        self.assertEqual(backend, "full_frame_motion_proxy")

    def test_landmark_jitter_removes_rigid_scale_and_rotation(self) -> None:
        base = np.zeros((468, 3), dtype=np.float32)
        points = np.array(
            [
                [-2.0, -1.0],
                [-1.0, 1.0],
                [0.0, -2.0],
                [1.0, 2.0],
                [2.0, -1.0],
            ],
            dtype=np.float32,
        )
        for index, point in zip(TEMPORAL_LANDMARK_INDICES, np.resize(points, (len(TEMPORAL_LANDMARK_INDICES), 2))):
            base[index, :2] = point

        angle = np.deg2rad(25.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )

        class Detector:
            def detect(self, frame: np.ndarray) -> None:
                return None

        class RigidTracker:
            available = True

            def extract(self, frame: np.ndarray) -> np.ndarray:
                marker = int(frame[0, 0, 0])
                transformed = base.copy()
                if marker == 1:
                    transformed[:, :2] = transformed[:, :2] @ rotation * 2.0 + [4.0, -3.0]
                elif marker == 2:
                    transformed[:, :2] = transformed[:, :2] @ rotation * 0.5 + [-2.0, 5.0]
                return transformed

        frames = [
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.ones((16, 16, 3), dtype=np.uint8),
            np.full((16, 16, 3), 2, dtype=np.uint8),
        ]
        jitter, backend = _face_box_jitter(frames, Detector(), RigidTracker())
        self.assertEqual(backend, "mediapipe_landmark_jitter")
        self.assertIsNotNone(jitter)
        self.assertLess(float(jitter), 1e-5)

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
            self.assertEqual(result["coverage"], "3/5")
            self.assertEqual(result["weighted_score_weight_coverage"], 50)
            self.assertEqual(len(result["summary"]), 5)
            self.assertIn("weighted_score_weight_coverage", result)

    def test_invalid_gt_falls_back_without_aborting_other_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.mp4"
            gt_path = root / "gt.mp4"
            _write_video(result_path, size=(64, 64))
            _write_video(gt_path, size=(80, 64))

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

            self.assertEqual(result["evaluation_mode"], "result_only")
            self.assertEqual(result["categories"]["texture"]["mode"], "no_gt")
            self.assertTrue(
                any("GT 全参考指标不可用" in warning for warning in result["warnings"])
            )

    def test_expression_style_and_semantic_scores_are_combined(self) -> None:
        categories = {
            "identity": {
                "status": "available",
                "metrics": {"score_0_1": 0.8},
                "frame_records": [],
                "warnings": [],
            },
            "texture": {
                "status": "available",
                "mode": "no_gt",
                "metrics": {"score_0_1": 0.7},
                "frame_records": [],
                "warnings": [],
            },
            "expression": {
                "status": "manual",
                "mode": "manual",
                "backend": "manual_1_to_5",
                "score_0_1": 0.4,
                "metrics": {},
                "frame_records": [],
                "warnings": [],
            },
            "temporal": {
                "status": "available",
                "metrics": {"stability_score_0_1": 0.9},
                "frame_records": [],
                "warnings": [],
            },
            "aesthetics": {
                "status": "manual",
                "metrics": {
                    "manual_score_1_to_5": 4,
                    "manual_score_0_to_1": 0.8,
                },
                "frame_records": [],
                "warnings": [],
            },
        }
        with (
            patch(
                "evaluator.holistic_evaluator.evaluate_identity",
                return_value=categories["identity"],
            ),
            patch(
                "evaluator.holistic_evaluator.evaluate_texture",
                return_value=categories["texture"],
            ),
            patch(
                "evaluator.holistic_evaluator.evaluate_expression",
                return_value=categories["expression"],
            ),
            patch(
                "evaluator.holistic_evaluator.evaluate_text_alignment",
                return_value={
                    "status": "available",
                    "backend": "viclip_internvid_10m_flt",
                    "metrics": {"score_0_1": 0.9},
                    "frame_records": [],
                },
            ),
            patch(
                "evaluator.holistic_evaluator.evaluate_etva_judge",
                return_value={
                    "status": "unavailable",
                    "score_0_1": None,
                    "warnings": [],
                },
            ),
            patch("evaluator.holistic_evaluator.etva_service_available", return_value=False),
            patch("evaluator.holistic_evaluator.clear_viclip_cache"),
            patch(
                "evaluator.holistic_evaluator.evaluate_temporal",
                return_value=categories["temporal"],
            ),
            patch(
                "evaluator.holistic_evaluator.evaluate_aesthetics",
                return_value=categories["aesthetics"],
            ),
        ):
            result = evaluate_all(
                "unused.mp4",
                None,
                None,
                None,
                prompt_text="人物微笑",
                max_frames=4,
                device="cpu",
            )

        self.assertAlmostEqual(
            result["categories"]["expression"]["score_0_1"],
            0.6,
            places=6,
        )
        self.assertEqual(
            result["categories"]["expression"]["metrics"]["text_video_alignment"],
            0.9,
        )

    def test_etva_replaces_prompt_alignment_as_the_semantic_component(self) -> None:
        categories = {
            "identity": {
                "status": "available",
                "metrics": {"score_0_1": 0.8},
                "frame_records": [],
                "warnings": [],
            },
            "texture": {
                "status": "available",
                "mode": "no_gt",
                "metrics": {"score_0_1": 0.7},
                "frame_records": [],
                "warnings": [],
            },
            "expression": {
                "status": "manual",
                "mode": "manual",
                "backend": "manual_1_to_5",
                "score_0_1": 0.4,
                "metrics": {},
                "frame_records": [],
                "warnings": [],
            },
            "temporal": {
                "status": "available",
                "metrics": {"stability_score_0_1": 0.9},
                "frame_records": [],
                "warnings": [],
            },
            "aesthetics": {
                "status": "manual",
                "metrics": {
                    "manual_score_1_to_5": 4,
                    "manual_score_0_to_1": 0.8,
                },
                "frame_records": [],
                "warnings": [],
            },
        }
        patches = [
            patch(
                "evaluator.holistic_evaluator.evaluate_identity",
                return_value=categories["identity"],
            ),
            patch(
                "evaluator.holistic_evaluator.evaluate_texture",
                return_value=categories["texture"],
            ),
            patch(
                "evaluator.holistic_evaluator.evaluate_expression",
                return_value=categories["expression"],
            ),
            patch(
                "evaluator.holistic_evaluator.evaluate_text_alignment",
                return_value={
                    "status": "available",
                    "backend": "viclip_internvid_10m_flt",
                    "metrics": {"score_0_1": 0.9},
                    "frame_records": [],
                },
            ),
            patch(
                "evaluator.holistic_evaluator.evaluate_etva_judge",
                return_value={
                    "status": "available",
                    "backend": "qwen2_vl_2b_awq_http",
                    "score_0_1": 0.2,
                    "warnings": [],
                },
            ),
            patch("evaluator.holistic_evaluator.etva_service_available", return_value=True),
            patch("evaluator.holistic_evaluator.clear_viclip_cache"),
            patch(
                "evaluator.holistic_evaluator.evaluate_temporal",
                return_value=categories["temporal"],
            ),
            patch(
                "evaluator.holistic_evaluator.evaluate_aesthetics",
                return_value=categories["aesthetics"],
            ),
        ]
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            result = evaluate_all(
                "unused.mp4",
                None,
                None,
                None,
                prompt_text="人物微笑",
                max_frames=4,
                device="cpu",
            )

        self.assertAlmostEqual(
            result["categories"]["expression"]["score_0_1"],
            0.32,
            places=6,
        )
        self.assertIn(
            "qwen2_vl_2b_awq_http",
            result["categories"]["expression"]["backend"],
        )


if __name__ == "__main__":
    unittest.main()
