from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluator.core.model_profile import get_recommended_model
from evaluator.backends.vbench_runner import _ensure_dino_compat_source, discover_vbench


class ModelProfileTests(unittest.TestCase):
    def test_default_recommendation_targets_8gb(self) -> None:
        recommendation = get_recommended_model()
        self.assertEqual(recommendation["id"], "qwen2_vl_2b_awq")
        self.assertEqual(recommendation["minimum_vram_gb"], 8)

    def test_larger_hardware_selects_larger_models(self) -> None:
        self.assertEqual(
            get_recommended_model(12)["id"],
            "qwen2_5_vl_3b_awq",
        )
        self.assertEqual(
            get_recommended_model(24)["id"],
            "videoscore2_bf16",
        )

    def test_dino_compatibility_source_is_available_offline(self) -> None:
        self.assertTrue(_ensure_dino_compat_source())

    def test_project_vbench_package_precedes_docker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package_spec = SimpleNamespace(
                submodule_search_locations=[directory],
            )
            real_find_spec = __import__(
                "importlib.util",
                fromlist=["find_spec"],
            ).find_spec

            def find_spec(name: str):
                if name == "vbench":
                    return package_spec
                return real_find_spec(name)

            with (
                patch(
                    "evaluator.backends.vbench_runner._dino_available",
                    return_value=True,
                ),
                patch(
                    "evaluator.backends.vbench_runner.importlib.util.find_spec",
                    side_effect=find_spec,
                ),
                patch(
                    "evaluator.backends.vbench_runner._launch_script",
                    return_value=Path(directory) / "launch" / "evaluate.py",
                ),
                patch(
                    "evaluator.backends.vbench_runner.shutil.which",
                    return_value="docker",
                ),
                patch(
                    "evaluator.backends.vbench_runner._docker_image_available",
                    return_value=True,
                ),
            ):
                result = discover_vbench()

        self.assertEqual(result["backend"], "package")


if __name__ == "__main__":
    unittest.main()
