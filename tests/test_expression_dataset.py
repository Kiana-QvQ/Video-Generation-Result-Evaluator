from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluator.expression_dataset import (
    SCHEMA_VERSION,
    build_expression_manifest,
    classify_performance,
    validate_expression_manifest,
)


class ExpressionDatasetTests(unittest.TestCase):
    def test_emotion_mapping_keeps_anger_and_annoyance_separate(self) -> None:
        self.assertEqual(classify_performance("Xiao"), "smile")
        self.assertEqual(classify_performance("FenNu"), "anger")
        self.assertEqual(classify_performance("ShengQi"), "annoyance")
        self.assertEqual(classify_performance("BeiShang2"), "sadness")

    def test_support_performances_are_not_emotions(self) -> None:
        self.assertEqual(classify_performance("Neutral"), "neutral")
        self.assertEqual(classify_performance("FACS1"), "facial_action")
        self.assertEqual(classify_performance("FuYin"), "articulation")

    def test_manifest_marks_only_existing_videos_as_usable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Xiao").mkdir()
            (root / "slice_manifest.json").write_text(
                json.dumps(
                    [
                        {
                            "person": "wangxing",
                            "performance": "Xiao",
                            "clip_index": 1,
                            "clip_path": "data/video/Xiao/clip0001.mp4",
                            "clip_len": 89,
                        },
                        {
                            "person": "wangxing",
                            "performance": "FenNu",
                            "clip_index": 1,
                            "clip_path": "data/video/FenNu/clip0001.mp4",
                            "clip_len": 89,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            (root / "Xiao" / "clip0001.mp4").write_bytes(b"test")

            payload = build_expression_manifest(root)

        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(payload["counts"]["emotion"], {"smile": 1})
        self.assertEqual(payload["source"]["usable_rows"], 1)
        self.assertEqual(payload["source"]["missing_video_rows"], 1)
        self.assertEqual(validate_expression_manifest(payload), [])

    def test_filesystem_only_clips_are_added_by_taxonomy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Xiao").mkdir()
            (root / "Xiao" / "clip0000.mp4").write_bytes(b"test")
            (root / "slice_manifest.json").write_text("[]", encoding="utf-8")

            payload = build_expression_manifest(root)

        self.assertEqual(payload["source"]["filesystem_rows"], 1)
        self.assertEqual(payload["source"]["usable_rows"], 1)
        self.assertEqual(payload["counts"]["emotion"], {"smile": 1})
        record = payload["records"][0]
        self.assertEqual(record["label_status"], "taxonomy_only")
        self.assertEqual(record["metadata_source"], "filesystem")
