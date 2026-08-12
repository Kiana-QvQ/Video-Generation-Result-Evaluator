from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from human_review.app import create_app
from human_review.database import ReviewDatabase


class HumanReviewTests(unittest.TestCase):
    def _create_app(self, directory: str):
        root = Path(directory)
        db_path = root / "review.sqlite3"
        database = ReviewDatabase(db_path, ip_secret="test-secret")
        video_a = root / "a.mp4"
        video_b = root / "b.mp4"
        video_a.write_bytes(b"fake-a")
        video_b.write_bytes(b"fake-b")
        database.replace_dataset_bundle(
            {
                "dataset_id": "test_review",
                "name": "Test Review",
                "version": "v1",
                "per_ip_quota": 1,
            },
            [
                {
                    "dataset_id": "test_review",
                    "asset_id": "asset-a",
                    "source_path": str(video_a),
                    "media_type": "video/mp4",
                    "original_name": video_a.name,
                },
                {
                    "dataset_id": "test_review",
                    "asset_id": "asset-b",
                    "source_path": str(video_b),
                    "media_type": "video/mp4",
                    "original_name": video_b.name,
                },
            ],
            [
                {
                    "dataset_id": "test_review",
                    "task_id": "task-1",
                    "candidates": [
                        {
                            "candidate_id": "candidate-a",
                            "asset_id": "asset-a",
                            "origin_type": "ai",
                            "model_id": "model-a",
                        },
                        {
                            "candidate_id": "candidate-b",
                            "asset_id": "asset-b",
                            "origin_type": "real",
                        },
                    ],
                }
            ],
        )
        return create_app(db_path=db_path, dataset_id="test_review")

    def test_review_round_can_restart_without_continue_button(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = self._create_app(directory)
            client = app.test_client()

            page = client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertNotIn('id="next-task"', page.get_data(as_text=True))
            self.assertIn("展示结果约 2 秒，自动进入下一题", page.get_data(as_text=True))

            first = client.get(
                "/api/review/next",
                headers={"X-Review-Round": "round-a"},
            )
            self.assertEqual(first.status_code, 200)
            task = first.get_json()["task"]
            self.assertIsNotNone(task)

            vote = client.post(
                "/api/review/vote",
                json={
                    "task_id": task["task_id"],
                    "choice": "A",
                    "response_ms": 2000,
                },
                headers={"X-Review-Round": "round-a"},
            )
            self.assertEqual(vote.status_code, 200)
            self.assertTrue(vote.get_json()["progress"]["done"])

            completed = client.get(
                "/api/review/next",
                headers={"X-Review-Round": "round-a"},
            )
            self.assertIsNone(completed.get_json()["task"])

            restarted = client.get(
                "/api/review/next",
                headers={"X-Review-Round": "round-b"},
            )
            self.assertIsNotNone(restarted.get_json()["task"])
            self.assertEqual(restarted.get_json()["progress"]["current"], 1)


if __name__ == "__main__":
    unittest.main()
