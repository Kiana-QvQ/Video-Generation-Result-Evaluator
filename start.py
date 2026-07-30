from __future__ import annotations

import os
import argparse
import importlib
import importlib.util
import json
import urllib.error
import urllib.request
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VLM_SCRIPT = ROOT / "scripts" / "run-vlm-judge-docker.ps1"
VLM_LOCAL_SCRIPT = ROOT / "scripts" / "run-vlm-judge-local.py"
VLM_MODEL_PATHS = {
    "2b": ROOT / "model_cache" / "vlm_judge" / "Qwen2-VL-2B-Instruct-AWQ",
    "2.5-3b": ROOT / "model_cache" / "vlm_judge" / "Qwen2.5-VL-3B-Instruct-AWQ",
}
VLM_SERVED_NAMES = {
    "2b": "qwen2-vl-2b-awq",
    "2.5-3b": "qwen2.5-vl-3b-awq",
}
VLM_CONTAINER_NAMES = {
    "2b": "frame-audit-qwen-2b",
    "2.5-3b": "frame-audit-qwen-2.5-3b",
}


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


def _vlm_service_models() -> list[str]:
    judge_url = os.environ.get(
        "ETVA_JUDGE_URL",
        "http://127.0.0.1:30000/v1/chat/completions",
    )
    models_url = judge_url.replace(
        "/v1/chat/completions",
        "/v1/models",
    )
    try:
        with urllib.request.urlopen(
            models_url,
            timeout=0.4,
        ) as response:
            if not 200 <= int(response.status) < 300:
                return []
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("data", []) if isinstance(payload, dict) else []
        return [
            str(item.get("id"))
            for item in models
            if isinstance(item, dict) and item.get("id")
        ]
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError):
        return []


def _vlm_service_available() -> bool:
    return bool(_vlm_service_models())


def _vlm_model_weights_available(model_path: Path) -> bool:
    """Accept both single-file and sharded Hugging Face checkpoints."""
    if not model_path.is_dir():
        return False
    return any(model_path.glob("*.safetensors")) or (
        model_path / "pytorch_model.bin"
    ).is_file()


def _local_vlm_missing_dependencies() -> list[str]:
    required = {
        "qwen_vl_utils": "qwen-vl-utils",
        "awq": "autoawq",
    }
    missing: list[str] = []
    for module, package in required.items():
        if importlib.util.find_spec(module) is None:
            missing.append(package)
            continue
        try:
            importlib.import_module(module)
        except Exception as exc:
            missing.append(f"{package} (import failed: {type(exc).__name__})")
    return missing


def _docker_ready() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _start_vlm_judge(
    model: str,
    backend: str = "local",
) -> tuple[subprocess.Popen[bytes], str, str] | None:
    service_models = _vlm_service_models()
    if service_models:
        requested_model = VLM_SERVED_NAMES[model]
        active_model = (
            requested_model
            if requested_model in service_models
            else service_models[0]
        )
        os.environ["ETVA_JUDGE_MODEL"] = active_model
        print(
            "Qwen Judge is already available on 127.0.0.1:30000 "
            f"({', '.join(service_models)}).",
            flush=True,
        )
        return None
    model_path = VLM_MODEL_PATHS[model]
    if not _vlm_model_weights_available(model_path):
        print(
            f"Qwen Judge weights are missing at {model_path}; "
            "HTTP service will remain disabled.",
            flush=True,
        )
        return None
    served_name = VLM_SERVED_NAMES[model]
    os.environ.setdefault("ETVA_JUDGE_MODEL", served_name)
    environment = os.environ.copy()
    environment.setdefault("ETVA_JUDGE_MODEL", served_name)

    backend = backend.lower()
    if backend == "local":
        missing_dependencies = _local_vlm_missing_dependencies()
        if missing_dependencies:
            print(
                "Local Qwen Judge dependencies are missing: "
                f"{', '.join(missing_dependencies)}. "
                "Run .\\setup.ps1 -VLM once, then restart the evaluator.",
                flush=True,
            )
            return None
        if not VLM_LOCAL_SCRIPT.is_file():
            print(f"Local Qwen Judge launcher is missing: {VLM_LOCAL_SCRIPT}", flush=True)
            return None
        command = [
            sys.executable,
            str(VLM_LOCAL_SCRIPT),
            "--model-path",
            str(model_path),
            "--served-model-name",
            served_name,
            "--host",
            "127.0.0.1",
            "--port",
            "30000",
        ]
        print(
            f"Starting local Qwen Judge ({model}) on 127.0.0.1:30000...",
            flush=True,
        )
        process = subprocess.Popen(command, cwd=ROOT, env=environment)
        return process, "local", ""

    if backend != "docker":
        print(
            f"Unsupported Qwen Judge backend {backend!r}; "
            "HTTP service will remain disabled.",
            flush=True,
        )
        return None
    if not VLM_SCRIPT.is_file():
        print(f"Qwen Judge launcher is missing: {VLM_SCRIPT}", flush=True)
        return None
    if shutil.which("docker") is None:
        print(
            "Docker is not available; Qwen Judge HTTP service will remain "
            "disabled.",
            flush=True,
        )
        return None
    if not _docker_ready():
        print(
            "Docker Desktop/daemon is not running; Qwen Judge HTTP service "
            "will remain disabled.",
            flush=True,
        )
        return None
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(VLM_SCRIPT),
        "-JudgeModel",
        model,
    ]
    print(
        f"Starting Qwen Judge ({model}) on 127.0.0.1:30000...",
        flush=True,
    )
    process = subprocess.Popen(command, cwd=ROOT, env=environment)
    return process, "docker", VLM_CONTAINER_NAMES[model]


def _stop_vlm_judge(
    handle: tuple[subprocess.Popen[bytes], str, str] | None,
) -> None:
    if handle is None:
        return
    process, backend, resource = handle
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    if backend == "docker" and resource:
        subprocess.run(
            ["docker", "stop", resource],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


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
    parser.set_defaults(with_vlm=True)
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
    parser.add_argument(
        "--original-au-root",
        default="data/au/MD_CL",
        help="Original AU CSV root used for general emotion classification.",
    )
    parser.add_argument(
        "--emotion-profile-output",
        default="data/au/original_emotion_au_profile.json",
    )
    parser.add_argument("--emotion-min-samples-per-class", type=int, default=3)
    args = parser.parse_args(argv)
    if args.vlm_backend is None:
        args.vlm_backend = os.environ.get("EVALUATOR_VLM_BACKEND", "local")
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
    command.extend(["--original-au-root", args.original_au_root])
    command.extend(["--emotion-profile-output", args.emotion_profile_output])
    command.extend(
        [
            "--emotion-min-samples-per-class",
            str(args.emotion_min_samples_per_class),
        ]
    )
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

    vlm_handle = (
        _start_vlm_judge(args.vlm_model, args.vlm_backend)
        if args.with_vlm
        else None
    )
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
            _stop_vlm_judge(vlm_handle)

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
        _stop_vlm_judge(vlm_handle)


if __name__ == "__main__":
    main()
