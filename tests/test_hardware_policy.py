from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from evaluator.modules.core.hardware_policy import resolve_policy


class HardwarePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_vram = os.environ.get("EVALUATOR_GPU_MEMORY_GB")
        self._previous_device = os.environ.get("CUDA_VISIBLE_DEVICES")

    def tearDown(self) -> None:
        if self._previous_vram is None:
            os.environ.pop("EVALUATOR_GPU_MEMORY_GB", None)
        else:
            os.environ["EVALUATOR_GPU_MEMORY_GB"] = self._previous_vram
        if self._previous_device is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._previous_device

    def test_cpu_request_is_safe(self) -> None:
        policy = resolve_policy("cpu")
        self.assertEqual(policy.tier, "cpu")
        self.assertEqual(policy.resolved_device, "cpu")
        self.assertTrue(policy.serial_models)

    def test_vram_override_selects_compact_tier_when_cuda_is_available(self) -> None:
        os.environ["EVALUATOR_GPU_MEMORY_GB"] = "8"
        policy = resolve_policy("auto")
        if policy.cuda_available:
            self.assertEqual(policy.tier, "compact_8gb")
            self.assertEqual(policy.judge_model, "qwen2_vl_2b_awq")
            self.assertEqual(policy.etva_frames, 4)

    def test_cpu_heavy_model_frame_budget_is_bounded(self) -> None:
        self.assertLessEqual(resolve_policy("cpu").etva_frames, 4)

    def test_critical_free_vram_reduces_etva_window_budget(self) -> None:
        with patch(
            "evaluator.modules.core.hardware_policy._cuda_info",
            return_value=(True, 8.0, 0.5),
        ):
            policy = resolve_policy("auto")

        self.assertEqual(policy.memory_pressure, "critical")
        self.assertEqual(policy.resolved_device, "cpu")
        self.assertEqual(policy.etva_frames, 2)
        self.assertFalse(policy.viclip_enabled_by_default)


if __name__ == "__main__":
    unittest.main()
