from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.extract_libreface_au import _utf8_environment
from scripts.evaluate_generated_video import _utf8_environment as evaluator_environment
from evaluator.backends import subst


class LibrefacePipelineTests(unittest.TestCase):
    def test_extractor_subprocess_environment_forces_utf8(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PYTHONIOENCODING": "gbk",
                "PYTHONUTF8": "0",
            },
            clear=False,
        ):
            environment = _utf8_environment()

        self.assertEqual(environment["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(environment["PYTHONUTF8"], "1")

    def test_evaluator_subprocess_environment_forces_utf8(self) -> None:
        environment = evaluator_environment()
        self.assertEqual(environment["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(environment["PYTHONUTF8"], "1")

    def test_subst_parser_keeps_mapping_targets(self) -> None:
        output = (
            "R:\\: => D:\\Pycharm2023.21\\project\\视频生成模型结果的评估器\n"
            "S:\\: => D:\\other-project\n"
        )
        with (
            patch.object(subst.os, "name", "nt"),
            patch.object(
                subst.subprocess,
                "run",
                return_value=type("Completed", (), {"stdout": output})(),
            ),
        ):
            mappings = subst.list_subst_mappings()

        self.assertEqual(
            mappings,
            [
                (
                    "R",
                    Path("D:\\Pycharm2023.21\\project\\视频生成模型结果的评估器"),
                ),
                ("S", Path("D:\\other-project")),
            ],
        )

    def test_cleanup_subst_mappings_removes_only_project_target(self) -> None:
        mappings = [
            (
                "R",
                Path("D:\\Pycharm2023.21\\project\\视频生成模型结果的评估器"),
            ),
            ("S", Path("D:\\other-project")),
        ]
        with (
            patch.object(subst.os, "name", "nt"),
            patch.object(subst, "list_subst_mappings", return_value=mappings),
            patch.object(subst, "remove_subst_drive", return_value=True) as remove,
        ):
            removed = subst.cleanup_project_subst_mappings(
                Path("D:\\Pycharm2023.21\\project\\视频生成模型结果的评估器"),
            )

        self.assertEqual(removed, ["R:"])
        remove.assert_called_once_with("R")


if __name__ == "__main__":
    unittest.main()
