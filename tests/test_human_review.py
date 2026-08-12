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
            self.assertNotIn('id="previous-task"', page.get_data(as_text=True))
            self.assertNotIn('id="restart-review"', page.get_data(as_text=True))
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
            self.assertIsNone(restarted.get_json()["task"])
            self.assertTrue(restarted.get_json()["progress"]["done"])

            duplicate = client.post(
                "/api/review/vote",
                json={"task_id": task["task_id"], "choice": "B"},
                headers={"X-Review-Round": "round-b"},
            )
            self.assertEqual(duplicate.status_code, 400)

    def test_media_endpoint_only_serves_active_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = self._create_app(directory)
            client = app.test_client()

            denied = client.get("/media/asset/archived_dataset/asset-a")
            self.assertEqual(denied.status_code, 404)

            allowed = client.get("/media/asset/test_review/asset-a")
            try:
                self.assertEqual(allowed.status_code, 200)
                self.assertEqual(allowed.data, b"fake-a")
            finally:
                allowed.close()

    def test_replacing_vote_free_dataset_removes_stale_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = ReviewDatabase(root / "review.sqlite3", ip_secret="test-secret")
            first_video = root / "first.mp4"
            second_video = root / "second.mp4"
            first_video.write_bytes(b"first")
            second_video.write_bytes(b"second")

            bundle = {
                "dataset_id": "replaceable",
                "name": "Replaceable",
                "version": "v1",
                "per_ip_quota": 1,
            }
            db.replace_dataset_bundle(
                bundle,
                [
                    {
                        "dataset_id": "replaceable",
                        "asset_id": "old-asset",
                        "source_path": str(first_video),
                        "media_type": "video/mp4",
                    }
                ],
                [],
            )
            db.replace_dataset_bundle(
                bundle,
                [
                    {
                        "dataset_id": "replaceable",
                        "asset_id": "new-asset",
                        "source_path": str(second_video),
                        "media_type": "video/mp4",
                    }
                ],
                [],
            )

            self.assertIsNone(db.get_asset("old-asset", "replaceable"))
            self.assertIsNotNone(db.get_asset("new-asset", "replaceable"))


if __name__ == "__main__":
    unittest.main()
