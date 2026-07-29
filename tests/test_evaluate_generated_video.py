from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_generated_video import _csv_path


class EvaluateGeneratedVideoTests(unittest.TestCase):
    def test_csv_path_uses_video_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "generated"
            video_path = Path(directory) / "nested" / "result.mp4"
            self.assertEqual(
                _csv_path(video_path, output_root),
                output_root / "result.csv",
            )

    def test_csv_path_handles_multiple_suffixes(self) -> None:
        output_root = Path("data/au/generated")
        self.assertEqual(
            _csv_path(Path("result.preview.mp4"), output_root),
            output_root / "result.preview.csv",
        )


if __name__ == "__main__":
    unittest.main()
