from __future__ import annotations

import unittest
from unittest.mock import patch

from start import (
    VLMStartupError,
    _parse_args,
    _run_au_training,
    _start_vlm_judge,
    _wait_for_vlm_service,
)


class _FakeProcess:
    def __init__(self, return_code: int | None = None) -> None:
        self.return_code = return_code

    def poll(self) -> int | None:
        return self.return_code


class StartArgumentTests(unittest.TestCase):
    def test_default_transport_remains_http_only(self) -> None:
        args = _parse_args([])
        self.assertEqual(args.transport, "http")
        self.assertEqual(args.http_host, "127.0.0.1")
        self.assertEqual(args.http_port, 7860)
        self.assertFalse(args.with_vlm)

    def test_vlm_can_be_disabled_explicitly(self) -> None:
        self.assertFalse(_parse_args(["--without-vlm"]).with_vlm)

    def test_with_grpc_starts_both_transports(self) -> None:
        args = _parse_args(["--with-grpc"])
        self.assertEqual(args.transport, "both")
        self.assertEqual(args.grpc_host, "127.0.0.1")
        self.assertEqual(args.grpc_port, 50051)

    def test_public_environment_can_be_shared_by_both_transports(self) -> None:
        with patch.dict(
            "os.environ",
            {"EVALUATOR_HOST": "0.0.0.0"},
            clear=False,
        ):
            args = _parse_args(["--with-grpc"])
        self.assertEqual(args.http_host, "0.0.0.0")
        self.assertEqual(args.grpc_host, "0.0.0.0")

    def test_train_au_arguments_build_the_shared_runner_command(self) -> None:
        args = _parse_args(
            [
                "--train-au",
                "--negative-dataset",
                "RAVDESS",
                "--ravdess-actors",
                "1,2",
                "--max-negative-videos",
                "12",
                "--au-device",
                "cpu",
            ]
        )
        with patch("start.subprocess.call", return_value=0) as call:
            result = _run_au_training(args)

        self.assertEqual(result, 0)
        command = call.call_args.args[0]
        self.assertIn("scripts\\run_au_training_pipeline.py", command[1])
        self.assertIn("--max-negative-videos", command)
        self.assertIn("12", command)
        self.assertIn("--device", command)
        self.assertIn("cpu", command)

    def test_vlm_waits_for_a_model_to_appear(self) -> None:
        process = _FakeProcess()
        with (
            patch(
                "start._vlm_service_models",
                side_effect=[[], [], ["qwen2-vl-2b-awq"]],
            ),
            patch("start.time.sleep") as sleep,
            patch.dict(
                "os.environ",
                {"EVALUATOR_VLM_STARTUP_TIMEOUT_SECONDS": "5"},
                clear=False,
            ),
        ):
            models = _wait_for_vlm_service(process, "2b", "local")

        self.assertEqual(models, ["qwen2-vl-2b-awq"])
        self.assertEqual(sleep.call_count, 2)

    def test_vlm_startup_fails_when_local_process_exits_early(self) -> None:
        process = _FakeProcess(return_code=1)
        with (
            patch("start._vlm_service_models", return_value=[]),
            patch.dict(
                "os.environ",
                {"EVALUATOR_VLM_STARTUP_TIMEOUT_SECONDS": "5"},
                clear=False,
            ),
        ):
            with self.assertRaisesRegex(
                VLMStartupError,
                "exited before /v1/models became ready",
            ):
                _wait_for_vlm_service(process, "2b", "local")

    def test_vlm_weights_missing_is_a_startup_error(self) -> None:
        with (
            patch("start._vlm_service_models", return_value=[]),
            patch("start._vlm_model_weights_available", return_value=False),
        ):
            with self.assertRaisesRegex(
                VLMStartupError,
                "download the model or start with --without-vlm",
            ):
                _start_vlm_judge("2b", "local")


if __name__ == "__main__":
    unittest.main()
