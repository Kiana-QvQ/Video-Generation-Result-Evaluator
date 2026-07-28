from __future__ import annotations

import os
import argparse
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def _restart_in_project_venv() -> None:
    if os.name != "nt":
        return
    if not VENV_PYTHON.is_file():
        raise SystemExit("Project environment is missing. Run setup.ps1 first.")
    project_venv = (ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() == project_venv:
        return
    child = subprocess.Popen(
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]]
    )
    try:
        return_code = child.wait()
    except KeyboardInterrupt:
        child.terminate()
        try:
            return_code = child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            return_code = child.wait(timeout=5)
        raise
    raise SystemExit(return_code)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Frame Audit services.")
    parser.add_argument(
        "--transport",
        choices=("http", "grpc", "both"),
        default="http",
        help="Service transport to start. Defaults to HTTP only.",
    )
    parser.add_argument(
        "--with-grpc",
        action="store_true",
        help="Start HTTP and gRPC together.",
    )
    parser.add_argument(
        "--http-host",
        default=None,
        help="HTTP bind address; defaults to EVALUATOR_HOST or 127.0.0.1.",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=None,
        help="HTTP port; defaults to EVALUATOR_PORT or 7860.",
    )
    parser.add_argument(
        "--grpc-host",
        default=None,
        help="gRPC bind address; defaults to EVALUATOR_GRPC_HOST.",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=None,
        help="gRPC port; defaults to EVALUATOR_GRPC_PORT or 50051.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print the selected Python environment and exit.",
    )
    parser.add_argument(
        "--train-au",
        action="store_true",
        help="Run the bounded AU leakage training pipeline and exit.",
    )
    parser.add_argument(
        "--negative-dataset",
        choices=("RAVDESS", "MetaHuman"),
        default="RAVDESS",
    )
    parser.add_argument("--metahuman-archive", default="")
    parser.add_argument("--metahuman-url", default="")
    parser.add_argument("--ravdess-actors", default="1,2")
    parser.add_argument(
        "--ravdess-source",
        choices=("ZENODO", "HUGGINGFACE"),
        default="ZENODO",
    )
    parser.add_argument(
        "--ravdess-emotions",
        default="1,2,3,4,5,6,7,8",
    )
    parser.add_argument("--ravdess-cache-root", default="data/cache/ravdess")
    parser.add_argument("--max-negative-videos", type=int, default=48)
    parser.add_argument(
        "--au-device",
        choices=("cpu", "cuda", "auto"),
        default="cuda",
    )
    parser.add_argument("--au-batch-size", type=int, default=64)
    parser.add_argument("--au-num-workers", type=int, default=2)
    parser.add_argument("--skip-negative-preparation", action="store_true")
    parser.add_argument("--force-au-extraction", action="store_true")
    args = parser.parse_args(argv)
    if args.with_grpc:
        args.transport = "both"
    args.http_host = args.http_host or os.environ.get(
        "EVALUATOR_HOST",
        "127.0.0.1",
    )
    args.http_port = args.http_port or int(
        os.environ.get("EVALUATOR_PORT", "7860")
    )
    args.grpc_host = args.grpc_host or os.environ.get(
        "EVALUATOR_GRPC_HOST",
        args.http_host if args.transport == "both" else "127.0.0.1",
    )
    args.grpc_port = args.grpc_port or int(
        os.environ.get("EVALUATOR_GRPC_PORT", "50051")
    )
    return args


def _start_grpc_process(args: argparse.Namespace) -> subprocess.Popen[bytes]:
    environment = os.environ.copy()
    environment["EVALUATOR_GRPC_HOST"] = args.grpc_host
    environment["EVALUATOR_GRPC_PORT"] = str(args.grpc_port)
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "grpc_server.py")],
        cwd=ROOT,
        env=environment,
    )
    return process


def _run_au_training(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_au_training_pipeline.py"),
        "--negative-dataset",
        args.negative_dataset,
        "--ravdess-actors",
        args.ravdess_actors,
        "--ravdess-source",
        args.ravdess_source,
        "--ravdess-emotions",
        args.ravdess_emotions,
        "--ravdess-cache-root",
        args.ravdess_cache_root,
        "--max-negative-videos",
        str(args.max_negative_videos),
        "--device",
        args.au_device,
        "--batch-size",
        str(args.au_batch_size),
        "--num-workers",
        str(args.au_num_workers),
    ]
    if args.metahuman_archive:
        command.extend(["--metahuman-archive", args.metahuman_archive])
    if args.metahuman_url:
        command.extend(["--metahuman-url", args.metahuman_url])
    if args.skip_negative_preparation:
        command.append("--skip-negative-preparation")
    if args.force_au_extraction:
        command.append("--force-au-extraction")
    return subprocess.call(command, cwd=ROOT)


def _stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


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
        print(f"venv={VENV_PYTHON}")
        return

    if args.train_au:
        raise SystemExit(_run_au_training(args))

    grpc_process = None
    if args.transport in {"grpc", "both"}:
        print(
            f"gRPC listening on {args.grpc_host}:{args.grpc_port}",
            flush=True,
        )
        grpc_process = _start_grpc_process(args)
        if grpc_process.poll() is not None:
            raise SystemExit("gRPC service failed to start.")

    if args.transport == "grpc":
        try:
            return_code = grpc_process.wait() if grpc_process else 0
            if return_code:
                raise SystemExit(return_code)
            return
        except KeyboardInterrupt:
            return
        finally:
            _stop_process(grpc_process)

    try:
        import uvicorn

        print(
            f"HTTP listening on {args.http_host}:{args.http_port}",
            flush=True,
        )
        uvicorn.run(
            "web_app:app",
            host=args.http_host,
            port=args.http_port,
            reload=False,
        )
    finally:
        _stop_process(grpc_process)


if __name__ == "__main__":
    main()
