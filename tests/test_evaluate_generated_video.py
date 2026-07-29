from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_generated_video import (
    _cached_csv_path,
    _csv_path,
    _run_extraction,
)


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

    def test_content_addressed_cache_hit_skips_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "result.mp4"
            cache_root = root / "cache"
            video_path.write_bytes(b"stable video content")
            cached_path = _cached_csv_path(
                video_path,
                cache_root,
                "generated",
            )
            cached_path.parent.mkdir(parents=True)
            cached_path.write_text("AU01_r\n0.2\n", encoding="utf-8")

            with patch("scripts.evaluate_generated_video.subprocess.run") as run:
                result = _run_extraction(
                    video_path,
                    root / "run",
                    device="cpu",
                    batch_size=1,
                    num_workers=0,
                    force=False,
                    cache_root=cache_root,
                )

            self.assertEqual(result, cached_path)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
