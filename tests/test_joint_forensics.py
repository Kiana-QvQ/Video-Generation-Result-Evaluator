from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional test dependency
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class JointForensicsModelTests(unittest.TestCase):
    def test_forward_supports_missing_audio_and_variable_frame_mask(self) -> None:
        from evaluator.modules.forensics.joint_model import JointForensicsModel

        model = JointForensicsModel(
            visual_dim=6,
            facial_dim=4,
            texture_dim=3,
            audio_dim=2,
            hidden_dim=16,
            layers=1,
            attention_heads=4,
            max_frames=8,
        )
        visual = np.zeros((2, 5, 6), dtype=np.float32)
        facial = np.zeros((2, 5, 4), dtype=np.float32)
        outputs = model(
            {
                "visual": visual,
                "facial": facial,
                "texture": None,
                "audio": None,
            },
            frame_mask=np.asarray(
                [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]],
                dtype=np.bool_,
            ),
        )
        self.assertEqual(tuple(outputs["identity_logit"].shape), (2,))
        self.assertEqual(tuple(outputs["expression_logits"].shape), (2, 7))
        probabilities = model.probabilities(outputs)
        self.assertTrue(
            torch.all(
                (probabilities["artifact_0_1"] >= 0.0)
                & (probabilities["artifact_0_1"] <= 1.0)
            )
        )

    def test_multitask_loss_ignores_unlabeled_quality_and_support(self) -> None:
        from evaluator.modules.forensics.joint_model import (
            JointForensicsModel,
            multitask_loss,
        )

        model = JointForensicsModel(
            visual_dim=2,
            hidden_dim=8,
            layers=1,
            attention_heads=2,
        )
        outputs = model(
            {"visual": torch.zeros(2, 4, 2)},
        )
        losses = multitask_loss(
            outputs,
            {
                "identity": torch.tensor([1.0, 0.0]),
                "expression": torch.tensor([2, -1]),
                "expression_support": torch.tensor([float("nan"), float("nan")]),
                "quality": torch.tensor([float("nan"), float("nan")]),
                "artifact": torch.tensor([0.0, 1.0]),
            },
        )
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(float(losses["quality"]), 0.0)
        self.assertEqual(float(losses["expression_support"]), 0.0)

    def test_all_invalid_frames_are_rejected(self) -> None:
        from evaluator.modules.forensics.joint_model import JointForensicsModel

        model = JointForensicsModel(
            visual_dim=2,
            hidden_dim=8,
            layers=1,
            attention_heads=2,
        )
        with self.assertRaisesRegex(ValueError, "valid frame"):
            model(
                {"visual": torch.zeros(1, 2, 2)},
                frame_mask=torch.zeros(1, 2, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
