from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from scripts import prepare_wangxing_v3_generalization as prepare
from scripts import run_wangxing_v3_pipeline as pipeline


class WangxingV3PipelineTests(unittest.TestCase):
    def test_media_filter_escapes_expression_commas(self) -> None:
        with tempfile.TemporaryDirectory(dir=prepare.PROJECT_ROOT) as directory:
            root = Path(directory)
            source = root / "source.mp4"
            destination = root / "destination.mp4"
            source.write_bytes(b"source")

            def fake_run(command, **_kwargs):
                Path(command[-1]).write_bytes(b"output")
                return CompletedProcess(command, 0)

            with patch(
                "scripts.prepare_wangxing_v3_generalization.subprocess.run",
                side_effect=fake_run,
            ) as run:
                prepare._run_ffmpeg(
                    source,
                    destination,
                    crf=23,
                    long_edge=1280,
                    fps=24,
                )

            command = run.call_args.args[0]
            filter_value = command[command.index("-vf") + 1]
            self.assertIn(r"gt(iw\,ih)", filter_value)
            self.assertIn(r"\,1280\,-2", filter_value)

    def test_relative_manifest_paths_are_stable(self) -> None:
        path = prepare.PROJECT_ROOT / "data" / "example.mp4"
        self.assertEqual(prepare._relative(path), "data/example.mp4")

    def test_manifest_validation_accepts_complete_dataset(self) -> None:
        with tempfile.TemporaryDirectory(dir=pipeline.PROJECT_ROOT) as directory:
            root = Path(directory)
            files = {}
            for name in (
                "real.mp4",
                "fake.mp4",
                "test_real.mp4",
                "test_fake.mp4",
                "media_real.mp4",
                "media_fake.mp4",
                "pseudo.mp4",
                "real.csv",
                "fake.csv",
                "test_real.csv",
                "test_fake.csv",
            ):
                path = root / name
                path.write_bytes(b"fixture")
                files[name] = path.relative_to(pipeline.PROJECT_ROOT).as_posix()

            base = {
                "train": {
                    "real": [files["real.mp4"]],
                    "fake": [files["fake.mp4"]],
                },
                "test": {
                    "real": [files["test_real.mp4"]],
                    "fake": [files["test_fake.mp4"]],
                },
            }
            manifest = {
                "pairs": {
                    "train": {
                        "real": [
                            {"video": files["real.mp4"], "au": files["real.csv"]},
                            {
                                "video": files["media_real.mp4"],
                                "au": files["real.csv"],
                            },
                        ],
                        "fake": [
                            {"video": files["fake.mp4"], "au": files["fake.csv"]},
                            {
                                "video": files["media_fake.mp4"],
                                "au": files["fake.csv"],
                            },
                            {
                                "video": files["pseudo.mp4"],
                                "au": files["real.csv"],
                            },
                        ],
                    },
                    "test": {
                        "real": [
                            {
                                "video": files["test_real.mp4"],
                                "au": files["test_real.csv"],
                            }
                        ],
                        "fake": [
                            {
                                "video": files["test_fake.mp4"],
                                "au": files["test_fake.csv"],
                            }
                        ],
                    },
                },
                "counts": {
                    "pseudo_fake": 1,
                    "media_variants": 2,
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            result = pipeline._validate_v3_manifest(
                manifest_path,
                base_payload=base,
                max_pseudo_fakes=1,
                max_media_per_class=1,
                allow_partial_media=False,
            )

            self.assertEqual(result["train_real"], 2)
            self.assertEqual(result["train_fake"], 3)
            self.assertEqual(result["test_real"], 1)
            self.assertEqual(result["test_fake"], 1)

    def test_dry_run_creates_completed_report(self) -> None:
        with tempfile.TemporaryDirectory(dir=pipeline.PROJECT_ROOT) as directory:
            report = (
                Path(directory)
                / "dry_run_report.json"
            )
            result = pipeline.main(
                [
                    "--dry-run",
                    "--report",
                    str(report),
                ]
            )
            self.assertEqual(result, 0)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(len(payload["stages"]), 0)


if __name__ == "__main__":
    unittest.main()
