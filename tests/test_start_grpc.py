from __future__ import annotations

import unittest
from unittest.mock import patch

from start_grpc import _parse_args


class GrpcStartArgumentTests(unittest.TestCase):
    def test_defaults_to_localhost_and_port_50051(self) -> None:
        args = _parse_args([])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 50051)
        self.assertTrue(args.with_vlm)

    def test_public_binds_all_interfaces(self) -> None:
        args = _parse_args(["--public", "--port", "50054"])
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 50054)

    def test_environment_values_are_used(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "EVALUATOR_GRPC_HOST": "192.168.80.88",
                "EVALUATOR_GRPC_PORT": "50054",
            },
            clear=False,
        ):
            args = _parse_args([])
        self.assertEqual(args.host, "192.168.80.88")
        self.assertEqual(args.port, 50054)

    def test_explicit_host_overrides_public_flag(self) -> None:
        args = _parse_args(["--public", "--host", "127.0.0.1"])
        self.assertEqual(args.host, "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
