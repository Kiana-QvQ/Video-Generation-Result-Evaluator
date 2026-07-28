from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.download_ravdess_negative import (
    _copy_selected_members,
    list_video_members,
    parse_int_list,
    parse_ravdess_member,
    select_balanced_members,
)


class RavdessNegativeTests(unittest.TestCase):
    def test_parses_ravdess_video_name(self) -> None:
        record = parse_ravdess_member(
            "Actor_12/03-01-08-02-02-01-12.mp4"
        )
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["actor"], "12")
        self.assertEqual(record["emotion"], "08")
        self.assertEqual(record["emotion_name"], "surprised")
        self.assertIsNone(parse_ravdess_member("README.txt"))

    def test_parses_integer_lists_without_duplicates(self) -> None:
        self.assertEqual(parse_int_list("1, 2;2 3", name="actors"), [1, 2, 3])

    def test_sampling_is_balanced_over_actor_and_emotion(self) -> None:
        records = []
        for actor in ("01", "02"):
            for emotion in ("01", "02"):
                for repetition in ("01", "02"):
                    records.append(
                        {
                            "actor": actor,
                            "emotion": emotion,
                            "intensity": "01",
                            "filename": (
                                f"03-01-{emotion}-01-01-{repetition}-"
                                f"{actor}.mp4"
                            ),
                        }
                    )
        selected = select_balanced_members(records, max_videos=8, seed=7)
        self.assertEqual(len(selected), 8)
        self.assertEqual(
            {(item["actor"], item["emotion"]) for item in selected},
            {("01", "01"), ("01", "02"), ("02", "01"), ("02", "02")},
        )

    def test_only_selected_zip_members_are_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "Video_Speech_Actor_01.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr(
                    "Actor_01/03-01-01-01-01-01-01.mp4",
                    b"selected",
                )
                handle.writestr(
                    "Actor_01/03-01-02-01-01-01-01.mp4",
                    b"not selected",
                )

            members = list_video_members(
                archive,
                allowed_emotions=[1],
            )
            selected = select_balanced_members(
                members,
                max_videos=1,
                seed=1,
            )
            records = _copy_selected_members(
                {1: archive},
                selected,
                output_root=root / "output",
            )

            self.assertEqual(len(records), 1)
            extracted = root / "output" / "videos" / "actor_01"
            files = list(extracted.glob("*.mp4"))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].read_bytes(), b"selected")


if __name__ == "__main__":
    unittest.main()
