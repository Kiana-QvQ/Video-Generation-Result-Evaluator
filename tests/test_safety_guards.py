from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import web_app
from backends.etva_judge import _parse_result
from evaluator.core.model_profile import get_recommended_model
from backends.vbench_runner import _matching_score
from evaluator.core.video_metrics import _read_frames


class SafetyGuardTests(unittest.TestCase):
    def test_etva_requires_structured_scores(self) -> None:
        self.assertEqual(
            _parse_result('{"scores":[0,0.5],"overall":0.5}'),
            (0.5, [0.0, 0.5]),
        )
        self.assertEqual(
            _parse_result("The problem description contains 0.5 but no JSON."),
            (None, []),
        )

    def test_vbench_uses_named_dimension_not_overall_score(self) -> None:
        payload = {
            "overall": 0.1,
            "aesthetic_quality": {"score": 0.8},
            "records": [
                {"dimension": "imaging_quality", "score": 0.7},
            ],
        }
        self.assertEqual(_matching_score(payload, "aesthetic_quality"), 0.8)
        self.assertEqual(_matching_score(payload, "imaging_quality"), 0.7)
        self.assertIsNone(_matching_score({"score": 0.9}, "motion_smoothness"))

    def test_model_recommendation_uses_observed_vram(self) -> None:
        self.assertEqual(get_recommended_model(8)["id"], "qwen2_vl_2b_awq")
        self.assertEqual(get_recommended_model(12)["id"], "qwen2_5_vl_3b_awq")
        self.assertEqual(get_recommended_model(24)["id"], "videoscore2_bf16")

    def test_api_key_is_constant_time_compared(self) -> None:
        with patch.dict(os.environ, {"FRAME_AUDIT_API_KEY": "secret"}, clear=False):
            self.assertTrue(web_app._valid_api_key(None, "secret"))
            self.assertTrue(
                web_app._valid_api_key("Bearer secret", None)
            )
            self.assertFalse(web_app._valid_api_key(None, "wrong"))

    def test_queue_lease_is_single_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(web_app, "WEB_RUNS_DIR", Path(directory)):
                web_app.JOB_WORKER_LEASE_HELD = False
                self.assertTrue(web_app._acquire_queue_lease())
                self.assertFalse(web_app._acquire_queue_lease())
                web_app._release_queue_lease()
                self.assertFalse(
                    (Path(directory) / web_app.QUEUE_LEASE_FILENAME).exists()
                )

    def test_frame_reader_preserves_requested_order_without_one_seek_per_frame(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.mp4"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (32, 32),
            )
            for index in range(5):
                frame = np.full((32, 32, 3), index * 40, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            frames = _read_frames(str(path), [4, 0, 4])
            means = [int(round(float(frame.mean()) / 40.0)) for frame in frames]
            self.assertEqual(means, [4, 0, 4])


if __name__ == "__main__":
    unittest.main()
