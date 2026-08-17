from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_generated_video import (
    SHARED_AU_CACHE_NAMESPACE,
    _cached_csv_path,
    _cache_debug_meta,
    _csv_path,
    _driver_au_for_video,
    _resolved_cache_namespace,
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

    def test_cache_namespace_includes_extractor_signature(self) -> None:
        with patch(
            "scripts.evaluate_generated_video._au_extraction_cache_signature",
            return_value="abc123def456",
        ):
            self.assertEqual(
                _resolved_cache_namespace("wangxing_specialization_v1"),
                "wangxing_specialization_v1__abc123def456",
            )
            self.assertEqual(
                _cache_debug_meta("wangxing_specialization_v1"),
                {
                    "requested_namespace": "wangxing_specialization_v1",
                    "resolved_namespace": "wangxing_specialization_v1__abc123def456",
                    "schema_version": "libreface_extract_cache_v2",
                    "extractor_signature": "abc123def456",
                },
            )

    def test_cache_miss_when_extractor_signature_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "result.mp4"
            cache_root = root / "cache"
            run_root = root / "run"
            video_path.write_bytes(b"stable video content")
            namespace = "wangxing_specialization_v1"

            with patch(
                "scripts.evaluate_generated_video._au_extraction_cache_signature",
                return_value="oldsignature",
            ):
                old_cached_path = _cached_csv_path(video_path, cache_root, namespace)
            old_cached_path.parent.mkdir(parents=True)
            old_cached_path.write_text("AU01_r\n0.2\n", encoding="utf-8")

            with patch(
                "scripts.evaluate_generated_video._au_extraction_cache_signature",
                return_value="newsignature",
            ):
                expected_output = _cached_csv_path(video_path, cache_root, namespace)
                extraction_root = (
                    expected_output.parent / f".extract_{expected_output.stem}"
                )
                extracted_path = _csv_path(video_path, extraction_root)

                def fake_run(*args: object, **kwargs: object) -> None:
                    extracted_path.parent.mkdir(parents=True, exist_ok=True)
                    extracted_path.write_text("AU01_r\n0.4\n", encoding="utf-8")

                with patch(
                    "scripts.evaluate_generated_video.subprocess.run",
                    side_effect=fake_run,
                ) as run:
                    result = _run_extraction(
                        video_path,
                        run_root,
                        device="cpu",
                        batch_size=1,
                        num_workers=0,
                        force=False,
                        cache_root=cache_root,
                        cache_namespace=namespace,
                    )

            self.assertEqual(result, expected_output)
            self.assertNotEqual(result, old_cached_path)
            self.assertTrue(result.is_file())
            run.assert_called_once()

    def test_same_video_reuses_generated_au_for_driver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated_video = root / "result.mp4"
            driver_video = root / "gt.mp4"
            generated_au = root / "generated.csv"
            generated_video.write_bytes(b"same video")
            driver_video.write_bytes(b"same video")
            generated_au.write_text("AU01_r\n0.2\n", encoding="utf-8")

            with patch(
                "scripts.evaluate_generated_video._run_extraction"
            ) as extract:
                result = _driver_au_for_video(
                    generated_video,
                    generated_au,
                    driver_video,
                    root / "driver",
                    device="cpu",
                    batch_size=1,
                    num_workers=0,
                    force=False,
                    cache_root=root / "cache",
                )

        self.assertEqual(result, generated_au)
        extract.assert_not_called()

    def test_different_driver_uses_shared_cache_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated_video = root / "result.mp4"
            driver_video = root / "gt.mp4"
            generated_au = root / "generated.csv"
            generated_video.write_bytes(b"generated")
            driver_video.write_bytes(b"driver")
            generated_au.write_text("AU01_r\n0.2\n", encoding="utf-8")

            with patch(
                "scripts.evaluate_generated_video._run_extraction",
                return_value=root / "driver.csv",
            ) as extract:
                result = _driver_au_for_video(
                    generated_video,
                    generated_au,
                    driver_video,
                    root / "driver",
                    device="cpu",
                    batch_size=1,
                    num_workers=0,
                    force=False,
                    cache_root=root / "cache",
                )

        self.assertEqual(result, root / "driver.csv")
        self.assertEqual(
            extract.call_args.kwargs["cache_namespace"],
            SHARED_AU_CACHE_NAMESPACE,
        )


if __name__ == "__main__":
    unittest.main()
