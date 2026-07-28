from __future__ import annotations

import unittest
from unittest.mock import patch

from start import _parse_args


class StartArgumentTests(unittest.TestCase):
    def test_default_transport_remains_http_only(self) -> None:
        args = _parse_args([])
        self.assertEqual(args.transport, "http")
        self.assertEqual(args.http_host, "127.0.0.1")
        self.assertEqual(args.http_port, 7860)

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


if __name__ == "__main__":
    unittest.main()
