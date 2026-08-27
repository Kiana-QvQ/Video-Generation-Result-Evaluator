from __future__ import annotations

import os
import argparse
import ipaddress
import importlib
import importlib.util
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
HUMAN_REVIEW_MODULE = "human_review.server"
PUBLIC_SHOWCASE_BUILDER = (
    ROOT / "scripts" / "web_forensics" / "build_public_showcase.py"
)
VLM_SCRIPT = ROOT / "scripts" / "tools" / "run-vlm-judge-docker.ps1"
VLM_LOCAL_SCRIPT = ROOT / "scripts" / "tools" / "run-vlm-judge-local.py"
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
VLM_STARTUP_TIMEOUT_SECONDS = 300.0
VLM_STARTUP_POLL_SECONDS = 0.5


class VLMStartupError(RuntimeError):
    """Raised when the Qwen Judge cannot become OpenAI-compatible and ready."""


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


def _vlm_startup_timeout_seconds() -> float:
    raw_value = os.environ.get(
        "EVALUATOR_VLM_STARTUP_TIMEOUT_SECONDS",
        str(VLM_STARTUP_TIMEOUT_SECONDS),
    )
    try:
        value = float(raw_value)
    except ValueError:
        value = VLM_STARTUP_TIMEOUT_SECONDS
    return max(1.0, value)


def _wait_for_vlm_service(
    process: subprocess.Popen[bytes],
    model: str,
    backend: str,
    resource: str = "",
) -> list[str]:
    """Wait until /v1/models exposes a usable model before starting evaluation."""
    deadline = time.monotonic() + _vlm_startup_timeout_seconds()
    last_models: list[str] = []
    while True:
        last_models = _vlm_service_models()
        expected_model = VLM_SERVED_NAMES[model]
        if expected_model in last_models:
            print(
                "Qwen Judge is ready on 127.0.0.1:30000 "
                f"({', '.join(last_models)}).",
                flush=True,
            )
            return last_models

        return_code = process.poll()
        if return_code is not None and (
            backend == "local" or return_code != 0
        ):
            raise VLMStartupError(
                "Qwen Judge process exited before /v1/models became ready "
                f"(backend={backend}, return_code={return_code}, "
                f"model={expected_model})."
            )
        if time.monotonic() >= deadline:
            resource_note = f" resource={resource!r}" if resource else ""
            raise VLMStartupError(
                "Timed out waiting for Qwen Judge /v1/models after "
                f"{_vlm_startup_timeout_seconds():.0f}s "
                f"(backend={backend}{resource_note}, "
                f"last_models={last_models or 'none'})."
            )
        time.sleep(VLM_STARTUP_POLL_SECONDS)


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
        raise VLMStartupError(
            f"Qwen Judge weights are missing at {model_path}; "
            "download the model or start with --without-vlm.",
        )
    served_name = VLM_SERVED_NAMES[model]
    os.environ.setdefault("ETVA_JUDGE_MODEL", served_name)
    environment = os.environ.copy()
    environment.setdefault("ETVA_JUDGE_MODEL", served_name)

    backend = backend.lower()
    if backend == "local":
        missing_dependencies = _local_vlm_missing_dependencies()
        if missing_dependencies:
            raise VLMStartupError(
                "Local Qwen Judge dependencies are missing: "
                f"{', '.join(missing_dependencies)}. "
                "Run .\\setup.ps1 -VLM once, then restart the evaluator, "
                "or use --without-vlm.",
            )
        if not VLM_LOCAL_SCRIPT.is_file():
            raise VLMStartupError(
                f"Local Qwen Judge launcher is missing: {VLM_LOCAL_SCRIPT}."
            )
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
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=environment)
        except OSError as exc:
            raise VLMStartupError(
                f"Unable to launch the local Qwen Judge: {exc}"
            ) from exc
        handle = (process, "local", "")
        try:
            _wait_for_vlm_service(process, model, "local")
        except VLMStartupError:
            _stop_vlm_judge(handle)
            raise
        return handle

    if backend != "docker":
        raise VLMStartupError(
            f"Unsupported Qwen Judge backend {backend!r}; "
            "use --vlm-backend local or docker.",
        )
    if not VLM_SCRIPT.is_file():
        raise VLMStartupError(f"Qwen Judge launcher is missing: {VLM_SCRIPT}.")
    if shutil.which("docker") is None:
        raise VLMStartupError(
            "Docker is not available; install/start Docker or use "
            "--vlm-backend local."
        )
    if not _docker_ready():
        raise VLMStartupError(
            "Docker Desktop/daemon is not running; start Docker or use "
            "--vlm-backend local."
        )
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
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=environment)
    except OSError as exc:
        raise VLMStartupError(
            f"Unable to launch the Docker Qwen Judge: {exc}"
        ) from exc
    resource = VLM_CONTAINER_NAMES[model]
    handle = (process, "docker", resource)
    try:
        _wait_for_vlm_service(process, model, "docker", resource)
    except VLMStartupError:
        _stop_vlm_judge(handle)
        raise
    return handle


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
    human_review_group = parser.add_mutually_exclusive_group()
    human_review_group.add_argument(
        "--with-human-review",
        dest="with_human_review",
        action="store_true",
        help="Start the standalone human-review website alongside Frame Audit.",
    )
    human_review_group.add_argument(
        "--without-human-review",
        "--no-human-review",
        dest="with_human_review",
        action="store_false",
        help="Start only the main Frame Audit website.",
    )
    parser.set_defaults(with_human_review=True)
    parser.add_argument(
        "--human-review-host",
        default=None,
        help=(
            "Human-review bind address; defaults to HUMAN_REVIEW_HOST "
            "or 0.0.0.0 (本机+局域网可访问)."
        ),
    )
    parser.add_argument(
        "--human-review-port",
        type=int,
        default=None,
        help="Human-review port; defaults to HUMAN_REVIEW_PORT or 5001.",
    )
    parser.add_argument(
        "--skip-public-showcase-refresh",
        action="store_true",
        help="Keep the existing public showcase index when starting.",
    )
    v5_display_group = parser.add_mutually_exclusive_group()
    v5_display_group.add_argument(
        "--v5-display",
        dest="v5_display",
        action="store_true",
        help=(
            "Use Wang Xing V5.2 score_display on the live web UI "
            "(default for the public display path)."
        ),
    )
    v5_display_group.add_argument(
        "--no-v5-display",
        "--legacy-forensics-display",
        dest="v5_display",
        action="store_false",
        help=(
            "Fall back to legacy forensics calibrated probability "
            "instead of V5.2 score_display."
        ),
    )
    parser.set_defaults(v5_display=True)
    v5_3_gate_group = parser.add_mutually_exclusive_group()
    v5_3_gate_group.add_argument(
        "--v5-3-content-gate",
        dest="v5_3_content_gate",
        action="store_true",
        help="Enable the validated V5.3 public content gate.",
    )
    v5_3_gate_group.add_argument(
        "--no-v5-3-content-gate",
        dest="v5_3_content_gate",
        action="store_false",
        help="Disable the V5.3 public content gate.",
    )
    parser.set_defaults(v5_3_content_gate=None)
    parser.add_argument(
        "--v5-3-runtime-manifest",
        default=None,
        help="Optional internal V5.3 explicit-role runtime manifest.",
    )
    parser.add_argument(
        "--require-api-key",
        action="store_true",
        help="Require FRAME_AUDIT_API_KEY for public API access.",
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
    parser.add_argument(
        "--http-host",
        default=None,
        help=(
            "HTTP bind address; defaults to EVALUATOR_HOST or 0.0.0.0 "
            "(本机+局域网可访问)."
        ),
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
        "--localhost-only",
        action="store_true",
        help=(
            "Bind Frame Audit and Human Review to 127.0.0.1 only "
            "(disables LAN URLs)."
        ),
    )
    parser.add_argument("--tls-certfile", default=None)
    parser.add_argument("--tls-keyfile", default=None)
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
    default_bind = "127.0.0.1" if args.localhost_only else "0.0.0.0"
    args.http_host = args.http_host or os.environ.get(
        "EVALUATOR_HOST",
        default_bind,
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
    args.human_review_host = args.human_review_host or os.environ.get(
        "HUMAN_REVIEW_HOST",
        default_bind,
    )
    args.human_review_port = args.human_review_port or int(
        os.environ.get("HUMAN_REVIEW_PORT", "5001")
    )
    args.tls_certfile = args.tls_certfile or os.environ.get(
        "EVALUATOR_TLS_CERTFILE",
        "",
    )
    args.tls_keyfile = args.tls_keyfile or os.environ.get(
        "EVALUATOR_TLS_KEYFILE",
        "",
    )
    return args


def _start_grpc_process(args: argparse.Namespace) -> subprocess.Popen[bytes]:
    environment = os.environ.copy()
    environment["EVALUATOR_GRPC_HOST"] = args.grpc_host
    environment["EVALUATOR_GRPC_PORT"] = str(args.grpc_port)
    if args.tls_certfile:
        environment["EVALUATOR_GRPC_TLS_CERT"] = args.tls_certfile
    if args.tls_keyfile:
        environment["EVALUATOR_GRPC_TLS_KEY"] = args.tls_keyfile
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "grpc_server.py")],
        cwd=ROOT,
        env=environment,
    )
    return process


def _start_human_review_process(
    args: argparse.Namespace,
) -> subprocess.Popen[bytes]:
    environment = os.environ.copy()
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    environment.setdefault("PYTHONUTF8", "1")
    command = [
        sys.executable,
        "-m",
        HUMAN_REVIEW_MODULE,
        "--host",
        args.human_review_host,
        "--port",
        str(args.human_review_port),
    ]
    print(
        "Human Review website starting on "
        f"{args.human_review_host}:{args.human_review_port}",
        flush=True,
    )
    return subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
    )


def _local_lan_ip() -> str:
    """Resolve the LAN address used for the startup access hint."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return str(probe.getsockname()[0])
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "本机局域网IP"
    finally:
        probe.close()


def _print_web_endpoints(args: argparse.Namespace) -> None:
    """Print both website endpoints in the same style as human_review/server."""
    lan_ip = _local_lan_ip()
    main_scheme = (
        "https"
        if args.tls_certfile and args.tls_keyfile
        else "http"
    )
    print(
        """
============================================================
Frame Audit / 视频评估网页
------------------------------------------------------------"""
    )
    print(f"本机访问:     {main_scheme}://127.0.0.1:{args.http_port}")
    print(f"绑定地址:     {main_scheme}://{args.http_host}:{args.http_port}")
    if args.http_host in {"127.0.0.1", "::1", "localhost"}:
        print("局域网访问:   当前绑定本机地址，局域网设备不可访问")
    else:
        print(f"局域网访问:   {main_scheme}://{lan_ip}:{args.http_port}")
    v5_on = os.environ.get("V5_DISPLAY_CASCADE", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if v5_on:
        print(
            "王兴展示分:   V5.3 已启用"
            "（真实拍摄概率=score_display；取证结论跟展示分≥0.75）"
        )
    else:
        print(
            "王兴展示分:   旧取证概率"
            "（已用 --no-v5-display / legacy 回退）"
        )
    if args.with_human_review:
        print(
            """
------------------------------------------------------------
Human Review / 人工审核网页
------------------------------------------------------------"""
        )
        print(
            f"本机访问:     http://127.0.0.1:{args.human_review_port}"
        )
        print(
            f"绑定地址:     http://{args.human_review_host}:"
            f"{args.human_review_port}"
        )
        if args.human_review_host in {"127.0.0.1", "::1", "localhost"}:
            print("局域网访问:   当前绑定本机地址，局域网设备不可访问")
        else:
            print(
                f"局域网访问:   http://{lan_ip}:{args.human_review_port}"
            )
    print(
        """
------------------------------------------------------------
一键可访问地址（编辑器直接运行 start.py，无需额外参数）
------------------------------------------------------------"""
    )
    print(f"评分本机:     {main_scheme}://127.0.0.1:{args.http_port}/")
    if args.http_host not in {"127.0.0.1", "::1", "localhost"}:
        print(f"评分局域网:   {main_scheme}://{lan_ip}:{args.http_port}/")
    if args.with_human_review:
        print(f"人工本机:     http://127.0.0.1:{args.human_review_port}/")
        if args.human_review_host not in {"127.0.0.1", "::1", "localhost"}:
            print(
                f"人工局域网:   http://{lan_ip}:{args.human_review_port}/"
            )
    print(
        """------------------------------------------------------------
按 Ctrl+C 停止两个网页
============================================================"""
    )


def _refresh_public_showcase() -> None:
    if not PUBLIC_SHOWCASE_BUILDER.is_file():
        print(
            f"Public showcase builder missing: {PUBLIC_SHOWCASE_BUILDER}",
            file=sys.stderr,
        )
        return
    command = [
        sys.executable,
        str(PUBLIC_SHOWCASE_BUILDER),
        "--max-items",
        "1000",
    ]
    index_path = ROOT / "outputs" / "public_showcase" / "index.json"
    if index_path.is_file():
        try:
            selection = json.loads(
                index_path.read_text(encoding="utf-8-sig")
            ).get("selection", {})
        except (OSError, json.JSONDecodeError):
            selection = {}
        if isinstance(selection, dict) and selection.get("mode") == "selected_jobs":
            for job_id in selection.get("job_ids", []):
                command.extend(["--job-id", str(job_id)])
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        print(
            "Public showcase refresh failed; keeping the existing index.",
            file=sys.stderr,
        )


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

    # V5.2 display is the default; --no-v5-display keeps legacy forensics.
    if args.v5_display:
        os.environ["V5_DISPLAY_CASCADE"] = "1"
    else:
        os.environ["V5_DISPLAY_CASCADE"] = "0"
    if args.v5_3_content_gate is not None:
        os.environ["V5_3_CONTENT_GATE"] = (
            "1" if args.v5_3_content_gate else "0"
        )
    if args.v5_3_runtime_manifest is not None:
        os.environ["V5_3_RUNTIME_MANIFEST"] = args.v5_3_runtime_manifest

    # IDE / 一键启动默认绑 0.0.0.0，允许本机+局域网明文访问。
    # 仅本机请加 --localhost-only；公网部署请配 TLS。
    public_hosts: list[str] = []
    for host in (args.http_host, args.grpc_host, args.human_review_host):
        try:
            if not ipaddress.ip_address(host).is_loopback:
                public_hosts.append(host)
        except ValueError:
            if host not in {"localhost"}:
                public_hosts.append(host)
    if public_hosts:
        os.environ.setdefault("EVALUATOR_ALLOW_INSECURE_PUBLIC", "1")
        if (
            args.transport in {"http", "both"}
            and (not args.tls_certfile or not args.tls_keyfile)
            and os.environ.get("EVALUATOR_ALLOW_INSECURE_PUBLIC", "").lower()
            not in {"1", "true", "yes", "on"}
        ):
            raise SystemExit(
                "Public HTTP binding requires --tls-certfile/--tls-keyfile "
                "or an explicit insecure override."
            )
        if args.require_api_key:
            if not os.environ.get("FRAME_AUDIT_API_KEY", "").strip():
                raise SystemExit(
                    "--require-api-key requires FRAME_AUDIT_API_KEY."
                )
            os.environ["FRAME_AUDIT_REQUIRE_AUTH"] = "1"
        else:
            # Public HTTP is intentionally address-only by default. Keep an
            # explicit opt-in for deployments that want API-key protection.
            os.environ.pop("FRAME_AUDIT_API_KEY", None)
            os.environ.pop("FRAME_AUDIT_REQUIRE_AUTH", None)

    grpc_process = None
    human_review_process = None
    vlm_handle = None
    if not args.skip_public_showcase_refresh:
        _refresh_public_showcase()
    if args.transport == "grpc":
        try:
            if args.with_human_review:
                human_review_process = _start_human_review_process(args)
            if args.with_vlm:
                vlm_handle = _start_vlm_judge(
                    args.vlm_model,
                    args.vlm_backend,
                )
            print(
                f"gRPC listening on {args.grpc_host}:{args.grpc_port}",
                flush=True,
            )
            grpc_process = _start_grpc_process(args)
            if grpc_process.poll() is not None:
                raise SystemExit("gRPC service failed to start.")
            if args.with_human_review:
                _print_web_endpoints(args)
            return_code = grpc_process.wait() if grpc_process else 0
            if return_code:
                raise SystemExit(return_code)
            return
        except VLMStartupError as exc:
            print(f"Qwen Judge failed to start: {exc}", file=sys.stderr, flush=True)
            raise SystemExit(1) from exc
        except KeyboardInterrupt:
            return
        finally:
            _stop_process(grpc_process)
            _stop_process(human_review_process)
            _stop_vlm_judge(vlm_handle)

    try:
        if args.with_human_review:
            human_review_process = _start_human_review_process(args)
        if args.with_vlm:
            vlm_handle = _start_vlm_judge(
                args.vlm_model,
                args.vlm_backend,
            )
        if args.transport in {"grpc", "both"}:
            print(
                f"gRPC listening on {args.grpc_host}:{args.grpc_port}",
                flush=True,
            )
            grpc_process = _start_grpc_process(args)
            if grpc_process.poll() is not None:
                raise SystemExit("gRPC service failed to start.")
        import uvicorn

        print(
            f"HTTP listening on {args.http_host}:{args.http_port}",
            flush=True,
        )
        _print_web_endpoints(args)
        uvicorn.run(
            "web_app:app",
            host=args.http_host,
            port=args.http_port,
            reload=False,
            ssl_certfile=args.tls_certfile or None,
            ssl_keyfile=args.tls_keyfile or None,
        )
    except VLMStartupError as exc:
        print(f"Qwen Judge failed to start: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    finally:
        _stop_process(grpc_process)
        _stop_process(human_review_process)
        _stop_vlm_judge(vlm_handle)

if __name__ == "__main__":
    main()
