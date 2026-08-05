from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from start import VLMStartupError, _start_vlm_judge, _stop_vlm_judge


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def _restart_in_project_venv() -> None:
    """Run this entry point with the project's Python environment on Windows."""
    if os.name != "nt":
        return
    if not VENV_PYTHON.is_file():
        raise SystemExit("Project environment is missing. Run setup.ps1 first.")
    project_venv = (ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() == project_venv:
        return
    return_code = subprocess.call(
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=ROOT,
    )
    raise SystemExit(return_code)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the Frame Audit gRPC service only."
    )
    parser.add_argument(
        "--host",
        "--bind-host",
        "--grpc-host",
        dest="host",
        default=None,
        help="gRPC bind address; defaults to EVALUATOR_GRPC_HOST or 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        "--grpc-port",
        dest="port",
        type=int,
        default=None,
        help="gRPC port; defaults to EVALUATOR_GRPC_PORT or 50051.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Bind to 0.0.0.0 so other machines can connect.",
    )
    parser.add_argument("--tls-certfile", default=None)
    parser.add_argument("--tls-keyfile", default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print the selected Python environment and endpoint, then exit.",
    )
    vlm_group = parser.add_mutually_exclusive_group()
    vlm_group.add_argument(
        "--with-vlm",
        dest="with_vlm",
        action="store_true",
        help="Start the cached Qwen VLM Judge on port 30000 (default).",
    )
    vlm_group.add_argument(
        "--without-vlm",
        "--no-vlm",
        dest="with_vlm",
        action="store_false",
        help="Do not start the Qwen VLM Judge.",
    )
    parser.set_defaults(with_vlm=False)
    parser.add_argument(
        "--vlm-backend",
        choices=("local", "docker"),
        default=None,
        help="Qwen Judge backend. Local transformers is the default.",
    )
    parser.add_argument(
        "--vlm-model",
        choices=("2b", "2.5-3b"),
        default="2b",
        help="Cached Qwen Judge model to start.",
    )
    args = parser.parse_args(argv)
    if args.vlm_backend is None:
        args.vlm_backend = os.environ.get("EVALUATOR_VLM_BACKEND", "local")

    args.host = args.host or (
        "0.0.0.0"
        if args.public
        else os.environ.get("EVALUATOR_GRPC_HOST", "127.0.0.1")
    )
    args.port = args.port or int(
        os.environ.get("EVALUATOR_GRPC_PORT", "50051")
    )
    args.tls_certfile = args.tls_certfile or os.environ.get(
        "EVALUATOR_GRPC_TLS_CERT",
        "",
    )
    args.tls_keyfile = args.tls_keyfile or os.environ.get(
        "EVALUATOR_GRPC_TLS_KEY",
        "",
    )
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    _restart_in_project_venv()
    os.chdir(ROOT)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.setdefault("EVALUATOR_FACE_DEVICE", "auto")
    os.environ.setdefault("EVALUATOR_IQA_DEVICE", "auto")
    os.environ.setdefault("EVALUATOR_SEMANTIC_DEVICE", "auto")

    args = _parse_args(argv)
    if args.check:
        print(f"executable={sys.executable}")
        print(f"prefix={sys.prefix}")
        print(f"grpc_host={args.host}")
        print(f"grpc_port={args.port}")
        return
    try:
        is_public = not ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        is_public = args.host != "localhost"
    if is_public:
        if not os.environ.get("FRAME_AUDIT_API_KEY", "").strip():
            raise SystemExit("Public binding requires FRAME_AUDIT_API_KEY.")
        if (
            (not args.tls_certfile or not args.tls_keyfile)
            and os.environ.get("EVALUATOR_ALLOW_INSECURE_PUBLIC", "").lower()
            not in {"1", "true", "yes", "on"}
        ):
            raise SystemExit(
                "Public gRPC binding requires --tls-certfile/--tls-keyfile "
                "or an explicit insecure override."
            )
        os.environ["FRAME_AUDIT_REQUIRE_AUTH"] = "1"

    os.environ["EVALUATOR_GRPC_HOST"] = args.host
    os.environ["EVALUATOR_GRPC_PORT"] = str(args.port)
    if args.tls_certfile:
        os.environ["EVALUATOR_GRPC_TLS_CERT"] = args.tls_certfile
    if args.tls_keyfile:
        os.environ["EVALUATOR_GRPC_TLS_KEY"] = args.tls_keyfile
    vlm_handle = None
    try:
        if args.with_vlm:
            vlm_handle = _start_vlm_judge(
                args.vlm_model,
                args.vlm_backend,
            )
        print(f"gRPC starting on {args.host}:{args.port}", flush=True)
        from grpc_server import serve

        serve()
    except VLMStartupError as exc:
        print(f"Qwen Judge failed to start: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    finally:
        _stop_vlm_judge(vlm_handle)


if __name__ == "__main__":
    main()
