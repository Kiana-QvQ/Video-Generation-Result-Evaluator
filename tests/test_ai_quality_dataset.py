from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from human_review.build_ai_quality_dataset import build_dataset
from human_review.database import ReviewDatabase


class AIQualityDatasetTests(unittest.TestCase):
    def test_build_uses_an_immutable_video_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "videos"
            output_dir = root / "dataset"
            input_dir.mkdir()
            video = input_dir / "sample.mp4"
            video.write_bytes(b"original-video")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "sample_id": "sample_01",
                                "file_name": video.name,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            db_path = root / "review.sqlite3"

            dataset = build_dataset(
                input_dir=input_dir,
                manifest_path=manifest,
                output_dir=output_dir,
                db_path=db_path,
                dataset_id="quality_v1",
                per_reviewer_quota=0,
            )

            assets = [
                json.loads(line)
                for line in (output_dir / "assets.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            snapshot = Path(assets[0]["source_path"])
            self.assertEqual(snapshot.parent.resolve(), (output_dir / "assets").resolve())
            self.assertEqual(snapshot.read_bytes(), b"original-video")
            self.assertEqual(dataset["per_reviewer_quota"], 0)

            video.write_bytes(b"changed-input-video")
            self.assertEqual(snapshot.read_bytes(), b"original-video")

    def test_dataset_with_ratings_cannot_be_rebuilt_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "videos"
            output_dir = root / "dataset"
            input_dir.mkdir()
            video = input_dir / "sample.mp4"
            video.write_bytes(b"original-video")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"items": [{"sample_id": "sample_01", "file_name": video.name}]}),
                encoding="utf-8",
            )
            db_path = root / "review.sqlite3"
            build_dataset(
                input_dir=input_dir,
                manifest_path=manifest,
                output_dir=output_dir,
                db_path=db_path,
                dataset_id="quality_v1",
                per_reviewer_quota=0,
            )

            database = ReviewDatabase(db_path, ip_secret="test-secret")
            database.insert_quality_vote(
                {
                    "dataset_id": "quality_v1",
                    "task_id": "quality_v1__sample_01",
                    "reviewer_id_hash": "reviewer",
                    "ip_hash": "ip",
                    "rating": "upper",
                }
            )

            with self.assertRaises(RuntimeError):
                build_dataset(
                    input_dir=input_dir,
                    manifest_path=manifest,
                    output_dir=output_dir,
                    db_path=db_path,
                    dataset_id="quality_v1",
                    per_reviewer_quota=0,
                )

            with closing(sqlite3.connect(db_path)) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM quality_votes"
                    ).fetchone()[0],
                    1,
                )


if __name__ == "__main__":
    unittest.main()
