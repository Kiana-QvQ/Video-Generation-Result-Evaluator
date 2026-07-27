from __future__ import annotations

import unittest

from evaluator.model_profile import get_recommended_model
from evaluator.vbench_runner import _ensure_dino_compat_source


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


if __name__ == "__main__":
    unittest.main()
