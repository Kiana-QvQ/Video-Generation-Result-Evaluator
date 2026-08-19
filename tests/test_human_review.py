from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sqlite3
import gc
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

from human_review.app import create_app
from human_review.database import ReviewDatabase
from human_review import server


class HumanReviewTests(unittest.TestCase):
    def test_server_defaults_to_all_interfaces(self) -> None:
        args = server.build_parser().parse_args([])
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 5001)

        local_args = server.build_parser().parse_args(
            ["--host", "127.0.0.1", "--port", "5002"]
        )
        self.assertEqual(local_args.host, "127.0.0.1")
        self.assertEqual(local_args.port, 5002)

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
                "per_reviewer_quota": 1,
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
        quality_a = root / "quality-a.mp4"
        quality_b = root / "quality-b.mp4"
        quality_a.write_bytes(b"quality-a")
        quality_b.write_bytes(b"quality-b")
        database.replace_quality_dataset_bundle(
            {
                "dataset_id": "test_quality",
                "name": "Test Quality",
                "version": "v1",
                "per_reviewer_quota": 2,
                "per_ip_quota": 2,
            },
            [
                {
                    "dataset_id": "test_quality",
                    "asset_id": "quality-asset-a",
                    "source_path": str(quality_a),
                    "media_type": "video/mp4",
                    "original_name": quality_a.name,
                },
                {
                    "dataset_id": "test_quality",
                    "asset_id": "quality-asset-b",
                    "source_path": str(quality_b),
                    "media_type": "video/mp4",
                    "original_name": quality_b.name,
                },
            ],
            [
                {
                    "dataset_id": "test_quality",
                    "task_id": "quality-task-a",
                    "asset_id": "quality-asset-a",
                    "question": "质量档次？",
                },
                {
                    "dataset_id": "test_quality",
                    "task_id": "quality-task-b",
                    "asset_id": "quality-asset-b",
                    "question": "质量档次？",
                },
            ],
        )
        return create_app(
            db_path=db_path,
            dataset_id="test_review",
            quality_dataset_id="test_quality",
        )

    def test_quality_rating_is_isolated_and_reuses_reviewer_ip_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = self._create_app(directory)
            first_browser = app.test_client()
            second_browser = app.test_client()

            first_task_response = first_browser.get(
                "/api/quality/next",
                headers={"X-Review-Round": "quality-round-a"},
            )
            self.assertEqual(first_task_response.status_code, 200)
            first_payload = first_task_response.get_json()
            first_task = first_payload["task"]
            self.assertEqual(first_payload["dataset_id"], "test_quality")
            self.assertEqual(first_payload["progress"]["total"], 2)
            self.assertEqual(
                {
                    (item["value"], item["label"])
                    for item in first_task["ratings"]
                },
                {
                    ("upper", "上档"),
                    ("middle", "中档"),
                    ("lower", "下档"),
                },
            )

            media = first_browser.get(first_task["video"]["url"])
            try:
                self.assertEqual(media.status_code, 200)
                self.assertIn(media.data, (b"quality-a", b"quality-b"))
            finally:
                media.close()

            invalid = first_browser.post(
                "/api/quality/rate",
                json={
                    "task_id": first_task["task_id"],
                    "rating": "unknown",
                },
            )
            self.assertEqual(invalid.status_code, 400)

            first_rating = first_browser.post(
                "/api/quality/rate",
                json={
                    "task_id": first_task["task_id"],
                    "rating": "upper",
                },
                headers={"X-Review-Round": "quality-round-a"},
            )
            self.assertEqual(first_rating.status_code, 200)
            self.assertEqual(first_rating.get_json()["progress"]["completed"], 1)
            self.assertEqual(
                first_browser.get("/api/review/progress").get_json()["completed"],
                0,
            )

            second_task = second_browser.get("/api/quality/next").get_json()["task"]
            self.assertIsNotNone(second_task)
            second_rating = second_browser.post(
                "/api/quality/rate",
                json={
                    "task_id": second_task["task_id"],
                    "rating": "middle",
                },
            )
            self.assertEqual(second_rating.status_code, 200)

            with closing(sqlite3.connect(Path(directory) / "review.sqlite3")) as connection:
                quality_counts = connection.execute(
                    """
                    SELECT COUNT(*), COUNT(DISTINCT reviewer_id_hash),
                           COUNT(DISTINCT ip_hash)
                    FROM quality_votes
                    WHERE dataset_id = 'test_quality'
                    """
                ).fetchone()
                pairwise_count = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM review_votes
                    WHERE dataset_id = 'test_review'
                    """
                ).fetchone()[0]
            self.assertEqual(tuple(quality_counts), (2, 2, 1))
            self.assertEqual(pairwise_count, 0)

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

    def test_different_browser_users_are_isolated_on_same_ip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = self._create_app(directory)
            first_browser = app.test_client()
            second_browser = app.test_client()
            first_task_response = first_browser.get("/api/review/next")
            second_task_response = second_browser.get("/api/review/next")
            try:
                first_task = first_task_response.get_json()["task"]
                second_task = second_task_response.get_json()["task"]

                self.assertIsNotNone(first_task)
                self.assertEqual(first_task["task_id"], second_task["task_id"])
                self.assertIn(
                    "human_review_reviewer_id=",
                    first_task_response.headers["Set-Cookie"],
                )
                self.assertIn("HttpOnly", first_task_response.headers["Set-Cookie"])
                self.assertIn("SameSite=Lax", first_task_response.headers["Set-Cookie"])
            finally:
                first_task_response.close()
                second_task_response.close()

            first_vote = first_browser.post(
                "/api/review/vote",
                json={"task_id": first_task["task_id"], "choice": "A"},
            )
            try:
                self.assertEqual(first_vote.status_code, 200)
            finally:
                first_vote.close()

            first_progress = first_browser.get("/api/review/progress")
            second_progress = second_browser.get("/api/review/progress")
            try:
                self.assertEqual(first_progress.get_json()["completed"], 1)
                self.assertEqual(second_progress.get_json()["completed"], 0)
            finally:
                first_progress.close()
                second_progress.close()

            second_vote = second_browser.post(
                "/api/review/vote",
                json={"task_id": second_task["task_id"], "choice": "B"},
            )
            try:
                self.assertEqual(second_vote.status_code, 200)
            finally:
                second_vote.close()

            with closing(sqlite3.connect(Path(directory) / "review.sqlite3")) as connection:
                row = connection.execute(
                    """
                    SELECT COUNT(*), COUNT(DISTINCT reviewer_id_hash),
                           COUNT(DISTINCT ip_hash)
                    FROM review_votes
                    WHERE dataset_id = 'test_review'
                    """
                ).fetchone()
            self.assertEqual(tuple(row), (2, 2, 1))
            del first_browser, second_browser, app
            gc.collect()

    def test_legacy_ip_votes_migrate_without_data_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "legacy.sqlite3"
            with closing(sqlite3.connect(db_path)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE datasets (
                        dataset_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        version TEXT NOT NULL,
                        status TEXT NOT NULL,
                        per_ip_quota INTEGER,
                        created_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL
                    );
                    CREATE TABLE tasks (
                        dataset_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        case_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        modality TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        references_json TEXT NOT NULL,
                        candidates_json TEXT NOT NULL,
                        control_type TEXT,
                        task_type TEXT NOT NULL,
                        question TEXT NOT NULL,
                        reveal_mode TEXT NOT NULL,
                        show_context INTEGER NOT NULL,
                        metadata_json TEXT NOT NULL,
                        PRIMARY KEY (dataset_id, task_id)
                    );
                    CREATE TABLE review_votes (
                        vote_id TEXT PRIMARY KEY,
                        dataset_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        ip_hash TEXT NOT NULL,
                        round_id TEXT NOT NULL,
                        choice TEXT NOT NULL,
                        displayed_a_candidate TEXT NOT NULL,
                        displayed_b_candidate TEXT NOT NULL,
                        response_ms INTEGER,
                        created_at TEXT NOT NULL
                    );
                    INSERT INTO datasets VALUES
                        ('test_review', 'Test', 'v1', 'active', 1, 'now', '{}');
                    INSERT INTO tasks VALUES (
                        'test_review', 'task-1', 'case-1', 'ready', 'reference_material',
                        '', '[]', '[]', NULL, 'ai_real_anchor', '', 'origin', 0, '{}'
                    );
                    INSERT INTO review_votes VALUES
                        ('vote-1', 'test_review', 'task-1', 'ip-hash',
                         'legacy', 'A', 'candidate-a', 'candidate-b', 1000, 'now');
                    """
                )

            database = ReviewDatabase(db_path, ip_secret="test-secret")
            with database.connect() as connection:
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(review_votes)")
                }
                vote = connection.execute(
                    """
                    SELECT reviewer_id_hash, ip_hash, choice
                    FROM review_votes
                    WHERE vote_id = 'vote-1'
                    """
                ).fetchone()
            self.assertIn("reviewer_id_hash", columns)
            self.assertEqual(tuple(vote), ("ip-hash", "ip-hash", "A"))
            del database
            gc.collect()

    def test_parallel_votes_cannot_bypass_reviewer_quota(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = self._create_app(directory)
            bootstrap = app.test_client()
            next_response = bootstrap.get("/api/review/next")
            task_id = next_response.get_json()["task"]["task_id"]
            cookie_value = (
                next_response.headers["Set-Cookie"]
                .split(";", 1)[0]
                .split("=", 1)[1]
            )
            next_response.close()
            del bootstrap

            def submit(choice: str) -> int:
                with app.test_client() as client:
                    client.set_cookie("human_review_reviewer_id", cookie_value)
                    response = client.post(
                        "/api/review/vote",
                        json={"task_id": task_id, "choice": choice},
                    )
                    try:
                        return response.status_code
                    finally:
                        response.close()

            with ThreadPoolExecutor(max_workers=2) as executor:
                statuses = list(executor.map(submit, ("A", "B")))

            self.assertEqual(sorted(statuses), [200, 400])
            with closing(sqlite3.connect(Path(directory) / "review.sqlite3")) as connection:
                count = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM review_votes
                    WHERE dataset_id = 'test_review'
                      AND reviewer_id_hash IS NOT NULL
                    """
                ).fetchone()[0]
            self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
