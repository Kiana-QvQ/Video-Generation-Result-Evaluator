from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scripts.extract_libreface_au import _utf8_environment
from scripts.evaluate_generated_video import _utf8_environment as evaluator_environment


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


if __name__ == "__main__":
    unittest.main()
