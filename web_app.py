from __future__ import annotations

import json
import hmac
import ipaddress
import math
import logging
from multiprocessing import Process
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from pydantic import BaseModel, Field

from evaluator.modules.core.holistic_evaluator import (
    WEIGHTS,
    evaluate_all,
    get_model_inventory,
    get_model_recommendation,
)
from evaluator.modules.core.evaluation_lock import serialized_evaluation
from evaluator.modules.core.hardware_policy import resolve_policy
from evaluator.modules.core.media import concatenate_videos, transcode_video_for_browser
from evaluator.modules.core.paths import resolve_profile
from evaluator.modules.core.public_showcase import (
    get_public_showcase as _get_public_showcase,
    list_public_showcase as _list_public_showcase,
    public_showcase_status as _public_showcase_status,
    resolve_public_showcase_file as _resolve_public_showcase_file,
)
from evaluator.modules.core.runtime import OUTPUT_DIR, PROJECT_ROOT
from backends.subst import cleanup_project_subst_mappings
from evaluator.modules.core.video_metrics import is_video_path, probe_video
from evaluator.modules.forensics import analyze_forensics
from evaluator.modules.wangxing.wangxing_specialization import (
    EXPRESSION_DISPLAY_NAMES,
    SPECIALIZATION_EVALUATOR_VERSION,
)


WEB_DIR = PROJECT_ROOT / "web"
WEB_RUNS_DIR = OUTPUT_DIR / "web_runs"
JOB_STATUS_FILENAME = "status.json"
JOB_PARAMS_FILENAME = "params.json"
JOB_OWNER_FILENAME = "owner.json"
TRUST_PROXY_HEADERS = os.getenv("FRAME_AUDIT_TRUST_PROXY_HEADERS", "").lower() in {
    "1",
    "true",
    "yes",
}
MAX_UPLOAD_BYTES = 1_500 * 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = int(
    os.getenv("FRAME_AUDIT_MAX_TOTAL_UPLOAD_BYTES", str(4 * MAX_UPLOAD_BYTES))
)
MAX_UPLOAD_FILES = int(
    os.getenv("FRAME_AUDIT_MAX_UPLOAD_FILES", "16")
)
MAX_REFERENCE_IMAGES = int(
    os.getenv("FRAME_AUDIT_MAX_REFERENCE_IMAGES", "8")
)
MAX_VIDEO_DURATION_SECONDS = float(
    os.getenv("FRAME_AUDIT_MAX_VIDEO_DURATION_SECONDS", "900")
)
MAX_VIDEO_FRAME_COUNT = int(
    os.getenv("FRAME_AUDIT_MAX_VIDEO_FRAME_COUNT", "100000")
)
MAX_VIDEO_PIXELS = int(
    os.getenv("FRAME_AUDIT_MAX_VIDEO_PIXELS", str(3840 * 2160))
)
RUN_RETENTION_SECONDS = float(
    os.getenv("FRAME_AUDIT_RUN_RETENTION_SECONDS", str(7 * 24 * 3600))
)
MAX_RUNS_BYTES = int(
    os.getenv("FRAME_AUDIT_MAX_RUNS_BYTES", str(50 * 1024**3))
)
UPLOADS_PER_MINUTE = int(
    os.getenv("FRAME_AUDIT_UPLOADS_PER_MINUTE", "60")
)
RUN_CLEANUP_INTERVAL_SECONDS = float(
    os.getenv("FRAME_AUDIT_RUN_CLEANUP_INTERVAL_SECONDS", "60")
)
QUEUE_LEASE_FILENAME = ".queue_worker.lock"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".m4v"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
WANGXING_AU_CLASSES = {
    "smile",
    "anger",
    "surprise",
    "fear",
    "sadness",
    "disgust",
}
WANGXING_AU_PROFILE_PATH = resolve_profile(
    "wangxing_au_profile.json",
    "data/au/wangxing_au_profile.json",
) or (PROJECT_ROOT / "data/au/wangxing_au_profile.json")
WANGXING_AU_CLASSIFIER_PATH = PROJECT_ROOT / "data/au/au_leakage_classifier.json"
ORIGINAL_EMOTION_AU_PROFILE_PATH = (
    PROJECT_ROOT / "data/au/original_emotion_au_profile.json"
)
WANGXING_IDENTITY_PROFILE_PATH = resolve_profile(
    "wangxing_identity_profile.json",
    "data/au/wangxing_identity_profile.json",
) or (PROJECT_ROOT / "data/au/wangxing_identity_profile.json")
WANGXING_EXPRESSION_PROFILE_PATH = resolve_profile(
    "wangxing_expression_profile.json",
    "data/au/wangxing_expression_profile.json",
) or (PROJECT_ROOT / "data/au/wangxing_expression_profile.json")
WANGXING_SOURCE_PROFILE_PATH = resolve_profile(
    "wangxing_source_profile.json",
    "data/au/wangxing_source_profile.json",
) or (PROJECT_ROOT / "data/au/wangxing_source_profile.json")
WANGXING_AU_CACHE_ROOT = OUTPUT_DIR / "au_cache"
FORENSICS_PROFILE_PATH = resolve_profile(
    "forensics_profiles.json",
    "outputs/forensics/forensics_profiles.json",
) or (PROJECT_ROOT / "outputs/forensics/forensics_profiles.json")
GENERATED_REPORT_FILES = {
    "summary.csv",
    "frame_metrics.csv",
    "result.json",
    "wangxing_au_result.json",
}

WEB_RUNS_DIR.mkdir(parents=True, exist_ok=True)

GENERIC_EXPRESSION_EVIDENCE_WEIGHTS = {
    "reference_style": 0.60,
    "prompt_semantic": 0.40,
}
EXPRESSION_TRACK_WEIGHTS = {
    "generic_expression": 0.50,
    "wangxing_au": 0.50,
}

JOB_DISPATCH_QUEUE: queue.Queue[str] = queue.Queue()
JOB_QUEUES_BY_IP: dict[str, deque[str]] = {}
JOB_SCHEDULED_IPS: set[str] = set()
JOB_LOCK = threading.RLock()
JOB_WORKER: threading.Thread | None = None
JOB_QUEUE_RECOVERED = False
JOB_PROCESSES: dict[str, Process] = {}
JOB_WORKER_STOP = threading.Event()
JOB_WORKER_STATE = "stopped"
JOB_WORKER_LAST_ERROR: str | None = None
JOB_WORKER_HEARTBEAT = 0.0
JOB_RECONCILE_INTERVAL_SECONDS = 2.0
JOB_STALE_RUNNING_SECONDS = 10 * 60
JOB_PROCESS_JOIN_TIMEOUT_SECONDS = 5
JOB_SCHEDULER_NAME = "hrrn_per_client_fifo_v1"
JOB_WORKER_LEASE_HELD = False
JOB_WORKER_LAST_CLEANUP = 0.0
LOGGER = logging.getLogger("frame_audit.web")
RATE_LIMIT_LOCK = threading.Lock()
RATE_LIMIT_BUCKETS: dict[str, deque[float]] = {}


class JobUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    action: Literal["cancel", "retry"] | None = None
    prompt_text: str | None = Field(default=None, max_length=10_000)
    max_frames: int | None = Field(default=None, ge=2, le=256)
    calculate_lpips: bool | None = None
    device: Literal["cpu", "auto", "cuda"] | None = None
    manual_expression_score: float | None = Field(default=None, ge=1, le=5)
    manual_aesthetic_score: float | None = Field(default=None, ge=1, le=5)
    wangxing_au_enabled: bool | None = None
    wangxing_expected_class: str | None = None


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    global JOB_WORKER
    cleanup_project_subst_mappings(PROJECT_ROOT)
    _ensure_queue_worker()
    try:
        yield
    finally:
        JOB_WORKER_STOP.set()
        _terminate_all_job_processes()
        cleanup_project_subst_mappings(PROJECT_ROOT)
        with JOB_LOCK:
            worker = JOB_WORKER
        if worker is not None and worker.is_alive():
            worker.join(timeout=3)
        with JOB_LOCK:
            JOB_WORKER = None
            JOB_WORKER_STOP.clear()
        _release_queue_lease()

app = FastAPI(
    title="Frame Audit",
    description="Local video generation evaluation workspace.",
    lifespan=app_lifespan,
)
app.mount("/assets", StaticFiles(directory=str(WEB_DIR)), name="assets")


@app.middleware("http")
async def disable_asset_cache(request: Any, call_next: Any) -> Any:
    if request.url.path.startswith("/api/") and _auth_required():
        if not _valid_api_key(request.headers.get("authorization"), request.headers.get("x-api-key")):
            return JSONResponse(
                status_code=401,
                content={"detail": "API authentication is required."},
                headers={"WWW-Authenticate": "Bearer"},
            )
    if request.url.path in {"/api/jobs", "/api/evaluate"}:
        if not _allow_upload_request(_client_ip(request)):
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        "Upload rate limit exceeded. "
                        f"Limit: {UPLOADS_PER_MINUTE} requests per minute."
                    )
                },
                headers={"Retry-After": "60"},
            )
    response = await call_next(request)
    if request.url.path.startswith("/assets/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def _auth_required() -> bool:
    configured_key = os.getenv("FRAME_AUDIT_API_KEY", "").strip()
    explicit = os.getenv("FRAME_AUDIT_REQUIRE_AUTH", "").strip().lower()
    return bool(configured_key) or explicit in {"1", "true", "yes", "on"}


def _valid_api_key(
    authorization: str | None,
    x_api_key: str | None,
) -> bool:
    expected = os.getenv("FRAME_AUDIT_API_KEY", "").strip()
    if not expected:
        return False
    supplied = (x_api_key or "").strip()
    if not supplied and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.casefold() == "bearer":
            supplied = token.strip()
    return bool(supplied) and hmac.compare_digest(supplied, expected)


def _allow_upload_request(client_ip: str) -> bool:
    now = time.monotonic()
    cutoff = now - 60.0
    with RATE_LIMIT_LOCK:
        bucket = RATE_LIMIT_BUCKETS.setdefault(client_ip, deque())
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= max(1, UPLOADS_PER_MINUTE):
            return False
        bucket.append(now)
        if len(RATE_LIMIT_BUCKETS) > 2048:
            for key in list(RATE_LIMIT_BUCKETS):
                if not RATE_LIMIT_BUCKETS[key]:
                    RATE_LIMIT_BUCKETS.pop(key, None)
        return True


def authenticate_grpc(context: Any) -> None:
    """Apply the same API-key policy to gRPC metadata."""
    if not _auth_required():
        return
    metadata = {
        str(key).casefold(): str(value)
        for key, value in context.invocation_metadata()
    }
    if not _valid_api_key(
        metadata.get("authorization"),
        metadata.get("x-api-key") or metadata.get("api-key"),
    ):
        raise HTTPException(
            status_code=401,
            detail="API authentication is required.",
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            # JSON has no native representation for infinities. Keep the
            # metric's meaning instead of turning a perfect PSNR into null.
            return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_job_id(job_id: str) -> str:
    safe_job_id = Path(job_id).name
    if safe_job_id != job_id or not safe_job_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return safe_job_id


def _client_ip(request: Request) -> str:
    host = request.client.host if request.client else "unknown"
    if TRUST_PROXY_HEADERS:
        forwarded_for = request.headers.get("x-forwarded-for", "")
        if forwarded_for:
            host = forwarded_for.split(",", 1)[0].strip() or host
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return host


def _job_owner(job: dict[str, Any]) -> str:
    return str(job.get("client_ip") or "legacy")


def _assert_job_owner(job: dict[str, Any], client_ip: str) -> None:
    if _job_owner(job) != client_ip:
        raise HTTPException(status_code=404, detail="Job not found")


def _job_dir(job_id: str) -> Path:
    safe_job_id = _safe_job_id(job_id)
    run_dir = (WEB_RUNS_DIR / safe_job_id).resolve()
    if run_dir.parent != WEB_RUNS_DIR.resolve():
        raise HTTPException(status_code=404, detail="Job not found")
    return run_dir


def _job_status_path(job_id: str) -> Path:
    return _job_dir(job_id) / JOB_STATUS_FILENAME


def _job_owner_path(job_id: str) -> Path:
    return _job_dir(job_id) / JOB_OWNER_FILENAME


def _write_job_owner(job_id: str, client_ip: str) -> None:
    run_dir = _job_dir(job_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(run_dir / JOB_OWNER_FILENAME, {"client_ip": client_ip})


def _read_job_owner(job_id: str) -> str | None:
    owner_path = _job_owner_path(job_id)
    if not owner_path.is_file():
        return None
    try:
        payload = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    owner = payload.get("client_ip") if isinstance(payload, dict) else None
    return str(owner) if owner else None


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        last_error: PermissionError | None = None
        for attempt in range(6):
            try:
                temporary_path.replace(path)
                return
            except PermissionError as exc:
                last_error = exc
                if attempt == 5:
                    raise
                # Windows antivirus/indexers and another short-lived worker
                # can briefly hold status.json during an atomic replacement.
                time.sleep(0.1 * (attempt + 1))
        if last_error is not None:
            raise last_error
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_job(job_id: str) -> dict[str, Any] | None:
    status_path = _job_status_path(job_id)
    if not status_path.exists():
        return None
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_job(job: dict[str, Any]) -> None:
    run_dir = _job_dir(str(job["job_id"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(run_dir / JOB_STATUS_FILENAME, job)


def _write_job_params(job: dict[str, Any]) -> None:
    run_dir = _job_dir(str(job["job_id"]))
    _atomic_write_json(run_dir / JOB_PARAMS_FILENAME, job.get("parameters", {}))


def _stored_file_name(run_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    resolved_path = path.resolve()
    resolved_root = run_dir.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError("Uploaded file escaped the job directory.")
    return resolved_path.relative_to(resolved_root).as_posix()


def _job_file_path(job: dict[str, Any], key: str) -> Path | None:
    relative_name = job.get("files", {}).get(key)
    if not relative_name:
        return None
    run_dir = _job_dir(str(job["job_id"]))
    path = (run_dir / str(relative_name)).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise ValueError("Job file escaped the job directory.")
    return path


def _job_file_paths(job: dict[str, Any], key: str) -> list[Path]:
    relative_names = job.get("files", {}).get(key) or []
    run_dir = _job_dir(str(job["job_id"]))
    paths: list[Path] = []
    for relative_name in relative_names:
        path = (run_dir / str(relative_name)).resolve()
        if not path.is_relative_to(run_dir.resolve()):
            raise ValueError("Job file escaped the job directory.")
        paths.append(path)
    return paths


def _job_uploaded_urls(job: dict[str, Any]) -> dict[str, Any]:
    run_id = str(job["job_id"])
    uploaded: dict[str, Any] = {}
    for key, relative_name in job.get("files", {}).items():
        if isinstance(relative_name, list):
            uploaded[key] = [
                _file_url(run_id, _job_file_paths(job, key)[index])
                for index in range(len(relative_name))
            ]
        else:
            uploaded[key] = _file_url(run_id, _job_file_path(job, key))
    return uploaded


def _result_downloads(job: dict[str, Any]) -> dict[str, str]:
    run_id = str(job["job_id"])
    run_dir = _job_dir(run_id)
    filenames = {
        "summary_csv": "summary.csv",
        "frame_csv": "frame_metrics.csv",
        "result_json": "result.json",
        "wangxing_au_json": "wangxing_au_result.json",
    }
    return {
        key: _file_url(run_id, run_dir / filename)
        for key, filename in filenames.items()
        if (run_dir / filename).is_file()
    }


def _job_name_fallback(job: dict[str, Any]) -> str:
    original_files = job.get("original_files")
    original_name = (
        original_files.get("result_video")
        if isinstance(original_files, dict)
        else None
    )
    return Path(
        str(original_name or job.get("name") or "result video")
    ).name or "result video"


def _job_display_name(job: dict[str, Any]) -> str:
    name = str(job.get("name") or "").strip()
    return name or _job_name_fallback(job)


def _normalize_wangxing_class(value: str | None) -> str | None:
    normalized = str(value or "").strip().lower()
    if not normalized or normalized == "auto":
        return None
    if normalized not in WANGXING_AU_CLASSES:
        raise HTTPException(
            status_code=422,
            detail=(
                "wangxing_expected_class must be auto, smile, anger, "
                "surprise, fear, sadness, or disgust."
            ),
        )
    return normalized


def _wangxing_au_status() -> dict[str, Any]:
    identity_profile_exists = WANGXING_IDENTITY_PROFILE_PATH.is_file()
    expression_profile_exists = WANGXING_EXPRESSION_PROFILE_PATH.is_file()
    ready = (
        identity_profile_exists
        and expression_profile_exists
    )
    return {
        "ready": ready,
        "evaluator_version": SPECIALIZATION_EVALUATOR_VERSION,
        "identity_profile": str(WANGXING_IDENTITY_PROFILE_PATH),
        "expression_profile": str(WANGXING_EXPRESSION_PROFILE_PATH),
        "source_profile": str(WANGXING_SOURCE_PROFILE_PATH),
        "identity_profile_exists": identity_profile_exists,
        "expression_profile_exists": expression_profile_exists,
        "source_profile_exists": WANGXING_SOURCE_PROFILE_PATH.is_file(),
        "forensics_profile": str(FORENSICS_PROFILE_PATH),
        "forensics_profile_exists": FORENSICS_PROFILE_PATH.is_file(),
        # Keep legacy paths in the payload for older clients.
        "profile": str(WANGXING_AU_PROFILE_PATH),
        "classifier": str(WANGXING_AU_CLASSIFIER_PATH),
        "emotion_profile": str(ORIGINAL_EMOTION_AU_PROFILE_PATH),
        "classes": sorted(WANGXING_AU_CLASSES),
        "note": (
            "ArcFace 身份画像和王兴真人表情画像已就绪。"
            "Seedance 仅用于身份校准。"
            if ready
            else (
                "请先训练王兴身份画像和真人表情画像，再启用该专项能力。"
            )
        ),
    }


def _run_forensics_assessment(
    *,
    result_path: Path,
    au_path: Path | None,
    device: str = "auto",
) -> dict[str, Any]:
    if not FORENSICS_PROFILE_PATH.is_file():
        return {
            "status": "unavailable",
            "reason": "真实性取证画像缺失。",
        }
    try:
        profiles = json.loads(
            FORENSICS_PROFILE_PATH.read_text(encoding="utf-8-sig")
        )
        result = analyze_forensics(
            facial_motion=au_path if au_path is not None else None,
            facial_motion_profile=profiles.get("facial_motion"),
            texture_detail=result_path,
            texture_detail_profile=profiles.get("texture_detail"),
            authenticity_calibrator=profiles.get(
                "authenticity_calibrator"
            ),
            max_frames=32,
            sample_fps=8.0,
            device=device,
        )
        result["auto_invoked_by"] = "wangxing_specialization_web_flow"
        return result
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"真实性取证评分失败：{exc}",
        }


@serialized_evaluation
def _run_wangxing_au_assessment(
    *,
    result_path: Path,
    reference_image_paths: list[Path],
    expected_class: str | None,
    device: str,
    run_dir: Path,
    reference_video_path: Path | None = None,
    prompt_text: str | None = None,
    driver_source: str | None = None,
) -> dict[str, Any]:
    # The specialization is self-contained; normal reference inputs remain
    # available to the five ordinary scores.
    del reference_video_path, driver_source
    status = _wangxing_au_status()
    if not status["ready"]:
        return {
            "status": "unavailable",
            "reason": status["note"],
        }

    au_device = resolve_policy(device).resolved_device
    output_path = run_dir / "wangxing_au_result.json"
    output_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/wangxing/evaluate_wangxing_specialization.py"),
        "--generated-video",
        str(result_path),
        "--identity-profile",
        str(WANGXING_IDENTITY_PROFILE_PATH),
        "--expression-profile",
        str(WANGXING_EXPRESSION_PROFILE_PATH),
        "--source-profile",
        str(WANGXING_SOURCE_PROFILE_PATH),
        "--output-root",
        str(run_dir / "wangxing_specialization"),
        "--cache-root",
        str(WANGXING_AU_CACHE_ROOT),
        "--output",
        str(output_path),
        "--device",
        au_device,
        "--identity-frames",
        "16",
    ]
    for reference_image_path in reference_image_paths:
        command.extend(["--target-image", str(reference_image_path)])
    if expected_class is not None:
        command.extend(["--expected-class", expected_class])

    evaluator_script = Path(command[1])
    if not evaluator_script.is_file():
        return {
            "status": "unavailable",
            "reason": (
                "Wang Xing AU 评估脚本不存在，已停止任务。"
                f"当前路径：{evaluator_script}"
            ),
            "error_type": "missing_evaluator_script",
        }

    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    try:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
    except subprocess.CalledProcessError as exc:
        diagnostics = "\n".join(
            value
            for value in (exc.stdout, exc.stderr)
            if value
        ).strip()
        normalized_diagnostics = diagnostics.lower()
        if (
            "codec can't encode" in normalized_diagnostics
            or "unicodeencodeerror" in normalized_diagnostics
            or "unicode decode error" in normalized_diagnostics
        ):
            reason = (
                "Wang Xing AU 子进程输出编码失败，已阻止本次评估结果生成。"
                "请重试；如果问题持续，请检查 Python/LibreFace 的 UTF-8 环境。"
            )
        elif (
            "no face detected" in normalized_diagnostics
            or "no landmarks detected" in normalized_diagnostics
        ):
            reason = (
                "AU 提取未检测到可用人脸关键点。系统已尝试原始视频和 "
                "标准化尺寸；请确认脸部没有被裁切，并保持正面、清晰、 "
                "足够大的脸部画面。"
            )
        else:
            reason = (
                "Wang Xing AU 提取失败。请检查 LibreFace、ffmpeg "
                "和输入视频格式。"
            )
        diagnostic_preview = " ".join(
            line.strip()
            for line in diagnostics.splitlines()[-3:]
            if line.strip()
        )[:800]
        if diagnostic_preview:
            reason = f"{reason} 具体错误：{diagnostic_preview}"
        return {
            "status": "unavailable",
            "reason": reason,
            "diagnostics": diagnostics[-2000:],
            "error_type": type(exc).__name__,
        }
    except OSError as exc:
        return {
            "status": "unavailable",
            "reason": (
                "Wang Xing AU 提取进程无法启动，请检查 LibreFace 和 ffmpeg。"
            ),
            "error_type": type(exc).__name__,
        }

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        payload.setdefault("driver_source", None)
        payload.setdefault("reference_action_used", False)
        payload.setdefault(
            "facial_dynamics_evidence_source",
            "wangxing_training_profile_dynamic_statistics",
        )
        payload.setdefault(
            "action_evidence_source",
            "wangxing_training_profile_dynamic_statistics",
        )
        generated_au = (
            payload.get("evaluation_meta", {}).get("generated_au")
            if isinstance(payload.get("evaluation_meta"), dict)
            else None
        )
        payload["forensics"] = _run_forensics_assessment(
            result_path=result_path,
            au_path=(
                Path(str(generated_au))
                if generated_au and Path(str(generated_au)).is_file()
                else None
            ),
            device=au_device,
        )
        payload["prompt_evidence"] = {
            "provided": bool((prompt_text or "").strip()),
            "note": (
                "Prompt 只参与通用表情语义评估，"
                "不会改变王兴身份门控结果。"
            ),
        }
        return payload
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "unavailable",
            "reason": "未生成王兴专项报告。",
            "error_type": type(exc).__name__,
        }


def _finite_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _attach_wangxing_evidence(
    result: dict[str, Any],
    *,
    wangxing_au: dict[str, Any],
    prompt_text: str | None,
    driver_source: str | None,
) -> None:
    """Expose Wang Xing evidence without changing normal evaluation scores."""
    categories = result.get("categories")
    if not isinstance(categories, dict):
        return
    expression = categories.get("expression")
    if not isinstance(expression, dict):
        return
    metrics = expression.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    targeted = wangxing_au.get("wangxing_targeted", {})
    if not isinstance(targeted, dict):
        targeted = {}
    au_score = _finite_score(
        targeted.get("wangxing_expression_fit_score_0_1")
    )
    au_coverage = _finite_score(
        targeted.get(
            "score_weight_coverage",
            targeted.get("evidence_coverage_0_1"),
        )
    )
    if au_coverage is None:
        evidence = targeted.get("evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}
        au_coverage = sum(
            weight
            for name, weight in (
                ("personal_au", 0.40),
                ("facial_dynamics", 0.30),
            )
            if _finite_score(evidence.get(name)) is not None
        )
    au_missing = list(targeted.get("missing_evidence", []))
    if not au_missing and wangxing_au.get("status") != "available":
        au_missing = ["wangxing_au"]

    reference_source = str(expression.get("reference_source") or "")
    generic_style = _finite_score(
        metrics.get("generic_style_score_0_1")
    )
    prompt_score = _finite_score(
        metrics.get("prompt_semantic_score_0_1")
    )
    generic_coverage = (
        0.60 * (1.0 if generic_style is not None else 0.0)
        if reference_source not in {"", "none"}
        else 0.0
    ) + 0.40 * (1.0 if prompt_score is not None else 0.0)

    if wangxing_au.get("schema_version") == "wangxing_specialization_v1":
        identity = wangxing_au.get("identity", {})
        expression_profile = wangxing_au.get("expression_profile", {})
        if not isinstance(identity, dict):
            identity = {}
        if not isinstance(expression_profile, dict):
            expression_profile = {}
        compatibility = _finite_score(
            expression_profile.get("compatibility_0_1")
        )
        result["expression_evidence"] = {
            "generic": {
                "score_0_1": _finite_score(expression.get("score_0_1")),
                "coverage_0_1": generic_coverage,
                "missing_evidence": [
                    name
                    for name, present in (
                        (
                            "reference_style",
                            reference_source not in {"", "none"}
                            and generic_style is not None,
                        ),
                        ("prompt_semantic", prompt_score is not None),
                    )
                    if not present
                ],
            },
            "wangxing_specialization": {
                "identity": identity,
                "source": wangxing_au.get("source", {}),
                "source_domain_evidence": wangxing_au.get("source", {}),
                "expression": expression_profile,
                "decision": wangxing_au.get("decision"),
                "forensics": wangxing_au.get("forensics"),
            },
            "wangxing_au": {
                "score_0_1": compatibility,
                "coverage_0_1": 1.0 if compatibility is not None else 0.0,
                "status": wangxing_au.get("status"),
                "decision": wangxing_au.get("decision"),
            },
            "scope": "separate_targeted_specialization",
            "normal_expression_unchanged": True,
            "prompt_semantic_backend": (
                metrics.get("prompt_semantic_backend")
                if prompt_text and prompt_text.strip()
                else None
            ),
        }
        return

    result["expression_evidence"] = {
        "generic": {
            "score_0_1": _finite_score(expression.get("score_0_1")),
            "coverage_0_1": generic_coverage,
            "missing_evidence": [
                name
                for name, present in (
                    (
                        "reference_style",
                        reference_source not in {"", "none"}
                        and generic_style is not None,
                    ),
                    ("prompt_semantic", prompt_score is not None),
                )
                if not present
            ],
        },
        "wangxing_au": {
            "score_0_1": au_score,
            "coverage_0_1": max(0.0, min(1.0, au_coverage or 0.0)),
            "missing_evidence": au_missing,
            "status": targeted.get("status"),
            "decision": targeted.get("decision"),
        },
        "scope": "separate_targeted_specialization",
        "normal_expression_unchanged": True,
        "driver_source": None,
        "reference_action_used": False,
        "facial_dynamics_evidence_source": (
            "wangxing_training_profile_dynamic_statistics"
        ),
        "action_evidence_source": (
            "wangxing_training_profile_dynamic_statistics"
        ),
        "prompt_semantic_backend": (
            metrics.get("prompt_semantic_backend")
            if prompt_text and prompt_text.strip()
            else None
        ),
    }


def _legacy_fuse_expression_evidence(
    result: dict[str, Any],
    *,
    wangxing_au: dict[str, Any],
    prompt_text: str | None,
    driver_source: str | None,
) -> None:
    """Merge generic expression and AU evidence without hiding gaps."""
    categories = result.get("categories")
    if not isinstance(categories, dict):
        return
    expression = categories.get("expression")
    if not isinstance(expression, dict):
        return

    metrics = expression.setdefault("metrics", {})
    old_expression_status = str(expression.get("status", "unavailable"))
    generic_score = _finite_score(expression.get("score_0_1"))
    style_score = _finite_score(metrics.get("generic_style_score_0_1"))
    prompt_score = _finite_score(metrics.get("prompt_semantic_score_0_1"))
    reference_source = str(expression.get("reference_source") or "")
    has_reference_source = reference_source not in {"", "none"}
    reference_style_score = style_score if has_reference_source else None
    if style_score is None and prompt_score is None and generic_score is not None:
        style_score = generic_score

    generic_components = {
        "reference_style": reference_style_score,
        "prompt_semantic": prompt_score,
    }
    generic_weight_coverage = sum(
        GENERIC_EXPRESSION_EVIDENCE_WEIGHTS[name]
        for name, score in generic_components.items()
        if score is not None
    )
    generic_missing = [
        name for name, score in generic_components.items() if score is None
    ]
    generic_available = [
        (
            GENERIC_EXPRESSION_EVIDENCE_WEIGHTS[name],
            score,
        )
        for name, score in generic_components.items()
        if score is not None
    ]
    if generic_available:
        generic_score = sum(weight * score for weight, score in generic_available) / sum(
            weight for weight, _ in generic_available
        )

    targeted = wangxing_au.get("wangxing_targeted", {})
    au_score = _finite_score(
        targeted.get("wangxing_expression_fit_score_0_1")
    )
    au_coverage = _finite_score(
        targeted.get(
            "score_weight_coverage",
            targeted.get("evidence_coverage_0_1"),
        )
    )
    if au_coverage is None:
        au_components = (
            ("personal_au", targeted.get("evidence", {}).get("personal_au")),
            (
                "facial_dynamics",
                targeted.get("evidence", {}).get("facial_dynamics"),
            ),
        )
        au_coverage = sum(
            weight
            for name, weight in (
                ("personal_au", 0.70),
                ("facial_dynamics", 0.30),
            )
            if _finite_score(dict(au_components).get(name)) is not None
        )
    au_coverage = max(0.0, min(1.0, au_coverage or 0.0))
    au_missing = list(targeted.get("missing_evidence", []))
    if not au_missing and wangxing_au.get("status") != "available":
        au_missing = ["wangxing_au"]

    tracks = [
        (EXPRESSION_TRACK_WEIGHTS["generic_expression"], generic_score),
        (EXPRESSION_TRACK_WEIGHTS["wangxing_au"], au_score),
    ]
    available_tracks = [
        (weight, score)
        for weight, score in tracks
        if score is not None
    ]
    fused_score = (
        sum(weight * score for weight, score in available_tracks)
        / sum(weight for weight, _ in available_tracks)
        if available_tracks
        else None
    )
    expression_coverage = (
        EXPRESSION_TRACK_WEIGHTS["generic_expression"]
        * generic_weight_coverage
        + EXPRESSION_TRACK_WEIGHTS["wangxing_au"] * au_coverage
    )
    missing_evidence = [
        f"expression_{name}" for name in generic_missing
    ] + [f"au_{name}" for name in au_missing]
    complete = (
        generic_weight_coverage >= 1.0
        and au_coverage >= 1.0
        and has_reference_source
        and old_expression_status in {"available", "manual"}
        and targeted.get("status") == "complete"
    )
    expression_status = (
        "complete"
        if complete
        else ("partial" if fused_score is not None else "unavailable")
    )
    expression["score_0_1"] = fused_score
    expression["status"] = expression_status
    expression["backend"] = (
        f'{expression.get("backend", "generic_expression")} + '
        "wangxing_au_evidence_fusion"
    )
    expression["evidence_coverage_0_1"] = expression_coverage
    expression["missing_evidence"] = missing_evidence
    metrics.update(
        {
            "generic_expression_score_0_1": generic_score,
            "wangxing_au_score_0_1": au_score,
            "generic_evidence_coverage_0_1": generic_weight_coverage,
            "wangxing_au_evidence_coverage_0_1": au_coverage,
            "expression_evidence_coverage_0_1": expression_coverage,
        }
    )
    expression["evidence_sources"] = {
        "reference_style": driver_source or (
            "ground_truth"
            if result.get("ground_truth_video")
            else None
        ),
        "prompt_semantic": (
            metrics.get("prompt_semantic_backend")
            if prompt_text and prompt_text.strip()
            else None
        ),
        "wangxing_au": (
            "generated_au_vs_profile"
            if au_score is not None
            else None
        ),
    }
    if missing_evidence:
        expression["note"] = (
            f'{expression.get("note", "")} '
            f"缺少证据：{', '.join(missing_evidence)}。"
        ).strip()

    categories["expression"] = expression
    category_scores = {
        "identity": categories["identity"].get("metrics", {}).get("score_0_1"),
        "texture": categories["texture"].get("metrics", {}).get("score_0_1"),
        "expression": expression.get("score_0_1"),
        "temporal": categories["temporal"].get("metrics", {}).get(
            "stability_score_0_1"
        ),
        "aesthetics": categories["aesthetics"].get("score_0_1"),
    }
    if category_scores["aesthetics"] is None:
        category_scores["aesthetics"] = categories["aesthetics"].get(
            "metrics", {}
        ).get("manual_score_0_to_1")
    result["category_scores"] = category_scores
    valid_scores = [
        (WEIGHTS[name], _finite_score(score))
        for name, score in category_scores.items()
        if _finite_score(score) is not None
    ]
    weight_coverage = sum(weight for weight, _ in valid_scores)
    result["weighted_score_0_1"] = (
        sum(weight * score for weight, score in valid_scores) / weight_coverage
        if weight_coverage
        else None
    )
    result["weighted_score_0_100"] = (
        result["weighted_score_0_1"] * 100
        if result["weighted_score_0_1"] is not None
        else None
    )
    result["weighted_score_weight_coverage"] = weight_coverage
    result["coverage"] = f"{len(valid_scores)}/5"
    result["status"] = (
        "complete"
        if len(valid_scores) == len(WEIGHTS)
        and all(
            categories[name].get("status") in {"available", "manual"}
            for name in WEIGHTS
        )
        else "partial"
    )
    result["weighted_score_status"] = result["status"]
    result["expression_evidence"] = {
        "generic": {
            "score_0_1": generic_score,
            "coverage_0_1": generic_weight_coverage,
            "missing_evidence": generic_missing,
        },
        "wangxing_au": {
            "score_0_1": au_score,
            "coverage_0_1": au_coverage,
            "missing_evidence": au_missing,
        },
        "fused": {
            "score_0_1": fused_score,
            "coverage_0_1": expression_coverage,
            "missing_evidence": missing_evidence,
        },
    }
    if isinstance(result.get("summary"), list) and len(result["summary"]) > 2:
        summary_entry = result["summary"][2]
        summary_keys = list(summary_entry)
        if len(summary_keys) >= 6:
            summary_entry[summary_keys[2]] = expression_status
            summary_entry[summary_keys[3]] = _format_score_for_report(
                fused_score
            )
            summary_entry[summary_keys[4]] = (
                f"普通表情 {_format_score_for_report(generic_score)}；"
                f"王兴 AU {_format_score_for_report(au_score)}；"
                f"证据覆盖 {expression_coverage:.2f}"
            )
            summary_entry[summary_keys[5]] = expression["backend"]


def _format_score_for_report(value: float | None) -> str:
    return "不可用" if value is None else f"{value:.4f}"


def _job_response(
    job: dict[str, Any],
    *,
    include_result: bool = False,
    queue_position: int | None = None,
) -> dict[str, Any]:
    result_path = _job_dir(str(job["job_id"])) / "result.json"
    result_available = (
        job.get("status") == "completed" and result_path.is_file()
    )
    response: dict[str, Any] = {
        "job_id": job["job_id"],
        "run_id": job["job_id"],
        "name": _job_display_name(job),
        "storage_path": f"web_runs/{job['job_id']}",
        "status": job.get("status", "queued"),
        "stage": job.get("stage", "queued"),
        "progress": float(job.get("progress", 0)),
        "queue_position": queue_position,
        "scheduler": JOB_SCHEDULER_NAME,
        "estimated_seconds": _estimate_job_seconds(job),
        "wait_seconds": round(_job_wait_seconds(job), 1),
        "created_at": job.get("created_at"),
        "queued_at": job.get("queued_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "updated_at": job.get("updated_at"),
        "error": job.get("error"),
        "parameters": job.get("parameters", {}),
        "original_files": job.get("original_files", {}),
        "uploaded_files": _job_uploaded_urls(job),
        "result_available": result_available,
    }
    if job.get("status") == "completed":
        response["downloads"] = _result_downloads(job)
    if include_result and job.get("status") == "completed":
        if result_path.is_file():
            try:
                response["result"] = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                response["result"] = None
    return _json_safe(response)


def _all_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for status_path in WEB_RUNS_DIR.glob(f"*/{JOB_STATUS_FILENAME}"):
        try:
            job = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(job, dict) and job.get("job_id"):
            jobs.append(job)
    return sorted(
        jobs,
        key=lambda item: str(item.get("created_at", "")),
        reverse=True,
    )


def _jobs_for_ip(jobs: list[dict[str, Any]], client_ip: str) -> list[dict[str, Any]]:
    return [job for job in jobs if _job_owner(job) == client_ip]


def _queued_positions(
    jobs: list[dict[str, Any]],
    client_ip: str | None = None,
) -> dict[str, int]:
    queued = sorted(
        (
            job
            for job in jobs
            if job.get("status") == "queued"
            and (client_ip is None or _job_owner(job) == client_ip)
        ),
        key=lambda item: str(item.get("queued_at") or item.get("created_at", "")),
    )
    return {
        str(job["job_id"]): index
        for index, job in enumerate(queued, start=1)
    }


def _estimate_job_seconds(job: dict[str, Any]) -> float:
    """Estimate relative job cost without reading media or loading models."""
    parameters = job.get("parameters", {})
    max_frames = max(1, int(parameters.get("max_frames", 8)))
    estimate = 8.0 + max_frames * 0.45
    if bool(parameters.get("calculate_lpips", True)):
        estimate += 10.0
    files = job.get("files", {})
    if files.get("gt_video"):
        estimate += 8.0
    if files.get("reference_video"):
        estimate += 7.0
    if files.get("reference_images"):
        estimate += min(8.0, 2.0 * len(files["reference_images"]))
    if bool(parameters.get("wangxing_au_enabled", False)):
        estimate += 20.0
    if parameters.get("prompt_text"):
        estimate += 4.0
    return round(max(1.0, estimate), 1)


def _job_wait_seconds(job: dict[str, Any]) -> float:
    timestamp = job.get("queued_at") or job.get("created_at")
    if not timestamp:
        return 0.0
    try:
        parsed = datetime.fromisoformat(str(timestamp))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return max(0.0, (datetime.now().astimezone() - parsed).total_seconds())


def _scheduler_score(job: dict[str, Any]) -> tuple[float, float]:
    """Use HRRN so short jobs finish quickly while old jobs gain priority."""
    service_seconds = _estimate_job_seconds(job)
    wait_seconds = _job_wait_seconds(job)
    response_ratio = (wait_seconds + service_seconds) / service_seconds
    return response_ratio, wait_seconds


def _peek_queued_job(client_ip: str) -> dict[str, Any] | None:
    with JOB_LOCK:
        ip_queue = JOB_QUEUES_BY_IP.get(client_ip)
        while ip_queue:
            job = _read_job(str(ip_queue[0]))
            if job is not None and job.get("status") == "queued":
                return job
            ip_queue.popleft()
        if ip_queue is not None and not ip_queue:
            JOB_SCHEDULED_IPS.discard(client_ip)
            JOB_QUEUES_BY_IP.pop(client_ip, None)
    return None


def _drain_dispatch_tokens() -> None:
    while True:
        try:
            JOB_DISPATCH_QUEUE.get_nowait()
        except queue.Empty:
            return
        else:
            JOB_DISPATCH_QUEUE.task_done()


def _select_next_dispatch_ip() -> str | None:
    """Pick one FIFO head per client using HRRN with aging."""
    _drain_dispatch_tokens()
    with JOB_LOCK:
        client_ips = sorted(JOB_SCHEDULED_IPS)
    candidates = [
        (client_ip, job)
        for client_ip in client_ips
        if (job := _peek_queued_job(client_ip)) is not None
    ]
    if not candidates:
        return None

    candidates.sort(
        key=lambda item: str(item[1].get("queued_at") or "")
    )
    selected_ip, _ = max(
        candidates,
        key=lambda item: _scheduler_score(item[1]),
    )
    with JOB_LOCK:
        for client_ip, _ in candidates:
            if client_ip != selected_ip:
                JOB_DISPATCH_QUEUE.put(client_ip)
    return selected_ip


def _display_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active = [job for job in jobs if job.get("status") == "running"]
    queued = sorted(
        (job for job in jobs if job.get("status") == "queued"),
        key=lambda item: str(item.get("queued_at") or item.get("created_at", "")),
    )
    history = sorted(
        (
            job
            for job in jobs
            if job.get("status") not in {"running", "queued"}
        ),
        key=lambda item: str(item.get("updated_at") or item.get("created_at", "")),
        reverse=True,
    )
    return active + queued + history


def _enqueue_job(job_id: str, client_ip: str | None = None) -> None:
    with JOB_LOCK:
        if client_ip is None:
            job = _read_job(job_id)
            if job is None:
                return
            client_ip = _job_owner(job)
        ip_queue = JOB_QUEUES_BY_IP.setdefault(client_ip, deque())
        if job_id not in ip_queue:
            ip_queue.append(job_id)
        if client_ip not in JOB_SCHEDULED_IPS:
            JOB_SCHEDULED_IPS.add(client_ip)
            JOB_DISPATCH_QUEUE.put(client_ip)


def _take_next_job(client_ip: str) -> str | None:
    with JOB_LOCK:
        ip_queue = JOB_QUEUES_BY_IP.get(client_ip)
        if not ip_queue:
            JOB_SCHEDULED_IPS.discard(client_ip)
            JOB_QUEUES_BY_IP.pop(client_ip, None)
            return None
        return ip_queue.popleft()


def _reschedule_ip(client_ip: str) -> None:
    with JOB_LOCK:
        ip_queue = JOB_QUEUES_BY_IP.get(client_ip)
        if ip_queue:
            JOB_DISPATCH_QUEUE.put(client_ip)
        else:
            JOB_SCHEDULED_IPS.discard(client_ip)
            JOB_QUEUES_BY_IP.pop(client_ip, None)


def _remove_job_from_dispatch(job_id: str, client_ip: str) -> None:
    ip_queue = JOB_QUEUES_BY_IP.get(client_ip)
    if not ip_queue:
        return
    remaining = deque(item for item in ip_queue if item != job_id)
    if remaining:
        JOB_QUEUES_BY_IP[client_ip] = remaining
        return
    JOB_QUEUES_BY_IP.pop(client_ip, None)
    JOB_SCHEDULED_IPS.discard(client_ip)


def _clear_dispatch_state() -> None:
    """Drop stale in-memory dispatch entries before rebuilding them."""
    with JOB_LOCK:
        while True:
            try:
                JOB_DISPATCH_QUEUE.get_nowait()
            except queue.Empty:
                break
            else:
                JOB_DISPATCH_QUEUE.task_done()
        JOB_QUEUES_BY_IP.clear()
        JOB_SCHEDULED_IPS.clear()


def _queue_lease_path() -> Path:
    return WEB_RUNS_DIR / QUEUE_LEASE_FILENAME


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError, SystemError):
        return False
    return True


def _acquire_queue_lease() -> bool:
    """Allow exactly one process to own the in-memory GPU queue."""
    global JOB_WORKER_LEASE_HELD
    lease_path = _queue_lease_path()
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            with lease_path.open("x", encoding="ascii") as handle:
                handle.write(str(os.getpid()))
            JOB_WORKER_LEASE_HELD = True
            return True
        except FileExistsError:
            try:
                owner_pid = int(lease_path.read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                owner_pid = 0
            if owner_pid and _pid_is_alive(owner_pid):
                return False
            lease_path.unlink(missing_ok=True)
    return False


def _release_queue_lease() -> None:
    global JOB_WORKER_LEASE_HELD
    if not JOB_WORKER_LEASE_HELD:
        return
    lease_path = _queue_lease_path()
    try:
        owner_pid = int(lease_path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        owner_pid = 0
    if owner_pid == os.getpid():
        lease_path.unlink(missing_ok=True)
    JOB_WORKER_LEASE_HELD = False


def _job_claim_path(job_id: str) -> Path:
    return _job_dir(job_id) / ".worker_claim"


def _claim_job(job_id: str) -> bool:
    claim_path = _job_claim_path(job_id)
    try:
        with claim_path.open("x", encoding="ascii") as handle:
            handle.write(f"{os.getpid()}\n{time.time():.6f}\n")
        return True
    except FileExistsError:
        try:
            owner_pid = int(
                claim_path.read_text(encoding="ascii").splitlines()[0]
            )
        except (OSError, ValueError, IndexError):
            owner_pid = 0
        if owner_pid and _pid_is_alive(owner_pid):
            return False
        claim_path.unlink(missing_ok=True)
        try:
            with claim_path.open("x", encoding="ascii") as handle:
                handle.write(f"{os.getpid()}\n{time.time():.6f}\n")
            return True
        except FileExistsError:
            return False


def _release_job_claim(job_id: str) -> None:
    claim_path = _job_claim_path(job_id)
    try:
        owner_pid = int(
            claim_path.read_text(encoding="ascii").splitlines()[0]
        )
    except (OSError, ValueError, IndexError):
        owner_pid = 0
    if owner_pid == os.getpid():
        claim_path.unlink(missing_ok=True)


def _directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _cleanup_expired_runs(force: bool = False) -> None:
    global JOB_WORKER_LAST_CLEANUP
    now = time.monotonic()
    if (
        not force
        and now - JOB_WORKER_LAST_CLEANUP < RUN_CLEANUP_INTERVAL_SECONDS
    ):
        return
    JOB_WORKER_LAST_CLEANUP = now
    terminal = {"completed", "failed", "canceled"}
    candidates: list[tuple[float, Path, int]] = []
    total_size = 0
    wall_clock = time.time()
    for run_dir in WEB_RUNS_DIR.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        job = _read_job(run_dir.name)
        if not job or job.get("status") not in terminal:
            continue
        size = _directory_size(run_dir)
        total_size += size
        try:
            updated = datetime.fromisoformat(
                str(job.get("updated_at") or job.get("created_at"))
            ).timestamp()
        except (TypeError, ValueError, OSError):
            updated = run_dir.stat().st_mtime
        candidates.append((updated, run_dir, size))

    candidates.sort(key=lambda item: item[0])
    for updated, run_dir, size in candidates:
        expired = (
            RUN_RETENTION_SECONDS >= 0
            and wall_clock - updated >= RUN_RETENTION_SECONDS
        )
        over_quota = total_size > MAX_RUNS_BYTES
        if not expired and not over_quota:
            continue
        try:
            _remove_job_directory(run_dir)
        except OSError as exc:
            LOGGER.warning("Unable to clean run directory %s: %s", run_dir, exc)
            continue
        total_size = max(0, total_size - size)


def _record_worker_error(exc: Exception) -> None:
    global JOB_WORKER_LAST_ERROR
    with JOB_LOCK:
        JOB_WORKER_LAST_ERROR = f"{type(exc).__name__}: {exc}"


def _worker_snapshot() -> dict[str, Any]:
    with JOB_LOCK:
        worker = JOB_WORKER
        heartbeat = JOB_WORKER_HEARTBEAT
        state = JOB_WORKER_STATE
        last_error = JOB_WORKER_LAST_ERROR
    return {
        "state": state if worker is not None and worker.is_alive() else "stopped",
        "alive": bool(worker is not None and worker.is_alive()),
        "scheduler": JOB_SCHEDULER_NAME,
        "heartbeat_age_seconds": (
            round(max(0.0, time.monotonic() - heartbeat), 2)
            if heartbeat
            else None
        ),
        "last_error": last_error,
    }


def _job_age_seconds(job: dict[str, Any]) -> float | None:
    timestamp = job.get("updated_at") or job.get("started_at")
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(timestamp))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return max(0.0, (datetime.now().astimezone() - parsed).total_seconds())


def _is_stale_running_job(job: dict[str, Any]) -> bool:
    if job.get("status") != "running":
        return False
    with JOB_LOCK:
        if str(job.get("job_id")) in JOB_PROCESSES:
            return False
    age = _job_age_seconds(job)
    return age is not None and age >= JOB_STALE_RUNNING_SECONDS


def _restore_queued_jobs() -> None:
    """Rebuild dispatch entries from persisted jobs after a worker loss."""
    with JOB_LOCK:
        persisted_jobs = sorted(
            _all_jobs(),
            key=lambda item: str(
                item.get("queued_at") or item.get("created_at", "")
            ),
        )
        for job in persisted_jobs:
            if job.get("status") != "queued":
                continue
            client_ip = job.get("client_ip")
            if not client_ip:
                continue
            _enqueue_job(str(job["job_id"]), str(client_ip))


def _reconcile_queued_jobs() -> None:
    """Recover jobs that were persisted but lost from the in-memory queue."""
    _cleanup_expired_runs()
    for job in _all_jobs():
        if _is_stale_running_job(job):
            job_id = str(job["job_id"])
            now = _now_iso()
            job.update(
                {
                    "status": "queued",
                    "stage": "queued",
                    "progress": 0.0,
                    "started_at": None,
                    "queued_at": now,
                    "updated_at": now,
                    "error": "Recovered a stale running task; queued for retry.",
                    "cancel_requested": False,
                }
            )
            if not _safe_recovery_write(job):
                continue
            _enqueue_job(job_id, _job_owner(job))
            continue
        if job.get("status") != "queued":
            continue
        client_ip = job.get("client_ip")
        if client_ip:
            _enqueue_job(str(job["job_id"]), str(client_ip))


def _safe_recovery_write(job: dict[str, Any]) -> bool:
    try:
        _write_job(job)
    except Exception as exc:
        _record_worker_error(exc)
        return False
    return True


def _remove_job_directory(run_dir: Path) -> None:
    last_error: OSError | None = None
    for attempt in range(5):
        try:
            shutil.rmtree(run_dir)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            if attempt == 4:
                break
            time.sleep(0.1 * (attempt + 1))
    if last_error is not None:
        raise last_error


def _update_job_state(job_id: str, **changes: Any) -> dict[str, Any]:
    with JOB_LOCK:
        job = _read_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.get("cancel_requested") and changes.get("status") != "canceled":
            return job
        job.update(changes)
        job["updated_at"] = _now_iso()
        _write_job(job)
        return job


def _optional_float(value: str | float | int | None) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid score value: {value}",
        ) from exc
    if not math.isfinite(parsed):
        raise HTTPException(
            status_code=422,
            detail=f"Score must be finite: {value}",
        )
    return parsed


def _optional_score(
    value: str | float | int | None,
    field_name: str,
) -> float | None:
    score = _optional_float(value)
    if score is not None and not 1.0 <= score <= 5.0:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be between 1 and 5.",
        )
    return score


def _validate_upload_count(
    *,
    result_video: UploadFile,
    gt_video: UploadFile | None,
    reference_images: list[UploadFile] | None,
    reference_video: list[UploadFile] | None,
) -> None:
    uploads = [
        upload
        for upload in (
            result_video,
            gt_video,
            *(reference_images or []),
            *(reference_video or []),
        )
        if upload is not None and upload.filename
    ]
    if len(uploads) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"A request cannot contain more than {MAX_UPLOAD_FILES} files.",
        )
    if len(reference_images or []) > MAX_REFERENCE_IMAGES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"A request cannot contain more than "
                f"{MAX_REFERENCE_IMAGES} reference images."
            ),
        )


def _upload_suffix(upload: UploadFile, allowed: set[str]) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Expected: {allowed_text}",
        )
    return suffix


def _save_upload(
    upload: UploadFile | None,
    run_dir: Path,
    stem: str,
    allowed: set[str],
    *,
    total_bytes: list[int] | None = None,
) -> Path | None:
    if upload is None or not upload.filename:
        return None
    suffix = _upload_suffix(upload, allowed)
    target = run_dir / f"{stem}.source{suffix}"
    total = 0
    try:
        with target.open("wb") as output:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="A single upload cannot exceed 1.5 GB.",
                    )
                if total_bytes is not None:
                    next_total = total_bytes[0] + len(chunk)
                    if next_total > MAX_TOTAL_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "The total upload size for one request cannot "
                                f"exceed {MAX_TOTAL_UPLOAD_BYTES / (1024**3):.1f} GB."
                            ),
                        )
                    total_bytes[0] = next_total
                output.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    if allowed == VIDEO_SUFFIXES:
        _validate_video(target, f"Uploaded file {upload.filename}")
        # Most generated MP4 files are already browser/OpenCV compatible.
        # Avoid blocking job creation with a second full video encode.
        if suffix == ".mp4":
            normalized = run_dir / f"{stem}.mp4"
            target.replace(normalized)
            return normalized
        normalized = run_dir / f"{stem}.mp4"
        transcode_video_for_browser(target, normalized)
        _validate_video(normalized, f"Normalized file {upload.filename}")
        return normalized
    try:
        encoded = np.fromfile(str(target), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except (OSError, ValueError):
        image = None
    if image is None:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f"Uploaded image is not readable: {upload.filename}",
        )
    return target


def _save_reference_videos(
    uploads: list[UploadFile] | None,
    run_dir: Path,
    *,
    total_bytes: list[int] | None = None,
) -> Path | None:
    valid_uploads = [
        upload for upload in (uploads or []) if upload is not None and upload.filename
    ]
    if not valid_uploads:
        return None
    if len(valid_uploads) == 1:
        return _save_upload(
            valid_uploads[0],
            run_dir,
            "reference_motion",
            VIDEO_SUFFIXES,
            total_bytes=total_bytes,
        )

    segment_paths = [
        saved
        for index, upload in enumerate(valid_uploads, start=1)
        if (
            saved := _save_upload(
                upload,
                run_dir,
                f"reference_motion_{index:02d}",
                VIDEO_SUFFIXES,
                total_bytes=total_bytes,
            )
        )
    ]
    if not segment_paths:
        return None
    return concatenate_videos(segment_paths, run_dir / "reference_motion.mp4")


def _original_reference_video_names(
    uploads: list[UploadFile] | None,
) -> str | list[str] | None:
    names = [
        Path(upload.filename).name
        for upload in (uploads or [])
        if upload is not None and upload.filename
    ]
    if not names:
        return None
    return names[0] if len(names) == 1 else names


def _reuse_optional_uploads(
    source_job: dict[str, Any],
    run_dir: Path,
    uploaded: dict[str, Any],
) -> dict[str, Any]:
    """Copy optional inputs from a prior job when replacing its result video."""
    source_original_files = source_job.get("original_files")
    source_original_files = (
        source_original_files if isinstance(source_original_files, dict) else {}
    )
    reused_original_files: dict[str, Any] = {}

    def copy_file(source_path: Path, target_path: Path) -> Path:
        if not source_path.is_file():
            raise HTTPException(
                status_code=409,
                detail="A saved reference file is no longer available.",
            )
        try:
            shutil.copy2(source_path, target_path)
        except OSError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Unable to reuse a saved reference file: {exc}",
            ) from exc
        return target_path

    if uploaded.get("gt_video") is None:
        source_path = _job_file_path(source_job, "gt_video")
        if source_path is not None:
            uploaded["gt_video"] = copy_file(source_path, run_dir / "gt.mp4")
            reused_original_files["gt_video"] = (
                source_original_files.get("gt_video") or source_path.name
            )

    if not uploaded.get("reference_images"):
        source_paths = _job_file_paths(source_job, "reference_images")
        if source_paths:
            copied_paths = []
            original_names = source_original_files.get("reference_images")
            for index, source_path in enumerate(source_paths, start=1):
                suffix = source_path.suffix.lower() or ".png"
                copied_paths.append(
                    copy_file(
                        source_path,
                        run_dir / f"reference_{index:02d}{suffix}",
                    )
                )
            uploaded["reference_images"] = copied_paths
            reused_original_files["reference_images"] = (
                original_names
                if isinstance(original_names, list) and original_names
                else [path.name for path in source_paths]
            )

    if uploaded.get("reference_video") is None:
        source_path = _job_file_path(source_job, "reference_video")
        if source_path is not None:
            uploaded["reference_video"] = copy_file(
                source_path,
                run_dir / "reference_motion.mp4",
            )
            reused_original_files["reference_video"] = (
                source_original_files.get("reference_video") or source_path.name
            )

    return reused_original_files


def _validate_video(path: Path, label: str) -> None:
    try:
        info = probe_video(path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{label} is not a readable video: {exc}",
        ) from exc
    if info.duration_seconds > MAX_VIDEO_DURATION_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"{label} exceeds the maximum duration of "
                f"{MAX_VIDEO_DURATION_SECONDS:.0f} seconds."
            ),
        )
    if info.frame_count > MAX_VIDEO_FRAME_COUNT:
        raise HTTPException(
            status_code=413,
            detail=(
                f"{label} exceeds the maximum of "
                f"{MAX_VIDEO_FRAME_COUNT} frames."
            ),
        )
    if info.width * info.height > MAX_VIDEO_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"{label} exceeds the maximum resolution of "
                f"{MAX_VIDEO_PIXELS:,} pixels per frame."
            ),
        )
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{label} exceeds the maximum upload size.",
        )


def _file_url(run_id: str, path: Path | None) -> str | None:
    if path is None:
        return None
    return f"/api/runs/{run_id}/{path.name}"


def _file_urls(
    run_id: str,
    paths: list[Path] | None,
) -> list[str]:
    if not paths:
        return []
    return [
        url
        for path in paths
        if (url := _file_url(run_id, path)) is not None
    ]


def _result_payload(
    run_id: str,
    result: dict[str, Any],
    csv_path: Path,
    json_path: Path,
    uploaded: dict[str, Any],
) -> dict[str, Any]:
    downloads = {
        "summary_csv": _file_url(run_id, csv_path),
        "result_json": _file_url(run_id, json_path),
    }
    wangxing_au_path = _job_dir(run_id) / "wangxing_au_result.json"
    if wangxing_au_path.is_file():
        downloads["wangxing_au_json"] = _file_url(
            run_id,
            wangxing_au_path,
        )
    payload = {
        "run_id": run_id,
        "result": result,
        "downloads": downloads,
        "uploaded_files": {
            key: (
                _file_urls(run_id, path)
                if isinstance(path, list)
                else _file_url(run_id, path)
            )
            for key, path in uploaded.items()
        },
    }
    return _json_safe(payload)


def _validate_evaluation_request(
    result_video: UploadFile,
    max_frames: int,
    device: str,
) -> str:
    result_suffix = _upload_suffix(result_video, VIDEO_SUFFIXES)
    if not is_video_path(f"result{result_suffix}"):
        raise HTTPException(status_code=415, detail="Result upload must be a video.")
    if max_frames < 2 or max_frames > 256:
        raise HTTPException(
            status_code=422,
            detail="max_frames must be between 2 and 256.",
        )
    if device not in {"cpu", "auto", "cuda"}:
        raise HTTPException(status_code=422, detail="Unsupported inference device.")
    return result_suffix


def _prepare_job(
    *,
    run_id: str,
    client_ip: str,
    name: str,
    reuse_source_job: dict[str, Any] | None,
    result_video: UploadFile,
    gt_video: UploadFile | None,
    reference_images: list[UploadFile] | None,
    reference_video: list[UploadFile] | None,
    prompt_text: str,
    max_frames: int,
    calculate_lpips: bool,
    device: str,
    manual_expression_score: str,
    manual_aesthetic_score: str,
    wangxing_au_enabled: bool,
    wangxing_expected_class: str,
) -> dict[str, Any]:
    _validate_evaluation_request(result_video, max_frames, device)
    _validate_upload_count(
        result_video=result_video,
        gt_video=gt_video,
        reference_images=reference_images,
        reference_video=reference_video,
    )
    prompt = prompt_text.strip()
    if len(prompt) > 10_000:
        raise HTTPException(
            status_code=422,
            detail="Prompt must be 10,000 characters or fewer.",
        )
    expression_score = _optional_score(
        manual_expression_score,
        "manual_expression_score",
    )
    aesthetic_score = _optional_score(
        manual_aesthetic_score,
        "manual_aesthetic_score",
    )
    expected_class = _normalize_wangxing_class(wangxing_expected_class)
    requested_name = name.strip() if isinstance(name, str) else ""
    display_name = requested_name or Path(
        result_video.filename or "result video"
    ).name

    run_dir = _job_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    uploaded: dict[str, Any] = {
        "result_video": None,
        "gt_video": None,
        "reference_images": [],
        "reference_video": None,
    }
    upload_budget = [0]
    try:
        uploaded["result_video"] = _save_upload(
            result_video,
            run_dir,
            "result",
            VIDEO_SUFFIXES,
            total_bytes=upload_budget,
        )
        uploaded["gt_video"] = _save_upload(
            gt_video,
            run_dir,
            "gt",
            VIDEO_SUFFIXES,
            total_bytes=upload_budget,
        )
        uploaded["reference_images"] = [
            saved
            for index, reference_image in enumerate(reference_images or [], start=1)
            if (
                saved := _save_upload(
                    reference_image,
                    run_dir,
                    f"reference_{index:02d}",
                    IMAGE_SUFFIXES,
                    total_bytes=upload_budget,
                )
            )
        ]
        uploaded["reference_video"] = _save_reference_videos(
            reference_video,
            run_dir,
            total_bytes=upload_budget,
        )
        reused_original_files: dict[str, Any] = {}
        if reuse_source_job is not None:
            reused_original_files = _reuse_optional_uploads(
                reuse_source_job,
                run_dir,
                uploaded,
            )
        result_path = uploaded["result_video"]
        if result_path is None:
            raise HTTPException(status_code=422, detail="Result video is required.")
        _validate_video(result_path, "Result video")
        if uploaded["gt_video"]:
            _validate_video(uploaded["gt_video"], "GT video")
        if uploaded["reference_video"]:
            _validate_video(uploaded["reference_video"], "Reference video")
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise

    files = {
        key: (
            [
                _stored_file_name(run_dir, path)
                for path in paths
                if path is not None
            ]
            if isinstance(paths, list)
            else _stored_file_name(run_dir, paths)
        )
        for key, paths in uploaded.items()
    }
    original_files = {
        "result_video": Path(result_video.filename or "result video").name,
        "gt_video": (
            Path(gt_video.filename).name
            if gt_video is not None and gt_video.filename
            else None
        ),
        "reference_images": [
            Path(image.filename).name
            for image in (reference_images or [])
            if image.filename
        ],
        "reference_video": _original_reference_video_names(reference_video),
    }
    for key, value in reused_original_files.items():
        original_files[key] = value
    parameters = {
        "prompt_text": prompt,
        "max_frames": max_frames,
        "calculate_lpips": calculate_lpips,
        "device": device,
        "manual_expression_score": expression_score,
        "manual_aesthetic_score": aesthetic_score,
        "wangxing_au_enabled": wangxing_au_enabled,
        "wangxing_expected_class": expected_class or "auto",
    }
    created_at = _now_iso()
    job = {
        "job_id": run_id,
        "client_ip": client_ip,
        "name": display_name,
        "status": "queued",
        "stage": "queued",
        "progress": 0.0,
        "created_at": created_at,
        "queued_at": created_at,
        "started_at": None,
        "finished_at": None,
        "updated_at": created_at,
        "error": None,
        "files": files,
        "original_files": original_files,
        "parameters": parameters,
    }
    _write_job_params(job)
    return job


def _execute_job(job_id: str) -> None:
    job = _read_job(job_id)
    if job is None or job.get("status") != "running":
        return
    try:
        result_path = _job_file_path(job, "result_video")
        gt_path = _job_file_path(job, "gt_video")
        reference_paths = _job_file_paths(job, "reference_images")
        reference_video_path = _job_file_path(job, "reference_video")
        if result_path is None or not result_path.is_file():
            raise ValueError("Result video is missing from the job directory.")

        _update_job_state(job_id, stage="sampling", progress=0.25)
        parameters = job.get("parameters", {})
        _update_job_state(job_id, stage="models", progress=0.55)
        result = evaluate_all(
            result_path=result_path,
            ground_truth=gt_path,
            reference_image=reference_paths,
            reference_video=reference_video_path,
            prompt_text=parameters.get("prompt_text") or None,
            max_frames=int(parameters.get("max_frames", 8)),
            calculate_lpips=bool(parameters.get("calculate_lpips", True)),
            device=str(parameters.get("device", "auto")),
            manual_expression_score=parameters.get("manual_expression_score"),
            manual_aesthetic_score=parameters.get("manual_aesthetic_score"),
            vbench_output_root=_job_dir(job_id),
        )
        if bool(parameters.get("wangxing_au_enabled", False)):
            _update_job_state(
                job_id,
                stage="wangxing_au",
                progress=0.72,
            )
            expected_class = _normalize_wangxing_class(
                str(parameters.get("wangxing_expected_class", "auto"))
            )
            wangxing_au = _run_wangxing_au_assessment(
                result_path=result_path,
                reference_image_paths=reference_paths,
                expected_class=expected_class,
                device=str(parameters.get("device", "auto")),
                run_dir=_job_dir(job_id),
                prompt_text=parameters.get("prompt_text"),
            )
            result["wangxing_au"] = wangxing_au
            _attach_wangxing_evidence(
                result,
                wangxing_au=wangxing_au,
                prompt_text=parameters.get("prompt_text"),
                driver_source=None,
            )
        else:
            result["wangxing_au"] = {
                "status": "not_applicable",
                "scope": "wangxing_specialization_only",
                "reason": (
                    "Wang Xing AU specialization was not selected. "
                    "Generic evaluation is unaffected."
                ),
            }
        result["web_run_id"] = job_id
        result["result_video"] = probe_video(result_path).to_dict()
        if gt_path:
            result["ground_truth_video"] = probe_video(gt_path).to_dict()

        run_dir = _job_dir(job_id)
        summary_path = run_dir / "summary.csv"
        frame_path = run_dir / "frame_metrics.csv"
        json_path = run_dir / "result.json"
        _update_job_state(job_id, stage="report", progress=0.88)
        pd.DataFrame(result["summary"]).to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(result.get("frame_records", [])).to_csv(
            frame_path,
            index=False,
            encoding="utf-8-sig",
        )
        json_path.write_text(
            json.dumps(_json_safe(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _update_job_state(
            job_id,
            status="completed",
            stage="completed",
            progress=1.0,
            finished_at=_now_iso(),
            error=None,
        )
    except Exception as exc:
        if isinstance(exc, HTTPException):
            error_message = str(exc.detail)
        else:
            error_message = f"{type(exc).__name__}: {exc}"
        try:
            _update_job_state(
                job_id,
                status="failed",
                stage="failed",
                progress=0.0,
                finished_at=_now_iso(),
                error=error_message,
            )
        except Exception as state_exc:
            _record_worker_error(state_exc)


def _process_job(job_id: str) -> None:
    if not _claim_job(job_id):
        return
    with JOB_LOCK:
        job = _read_job(job_id)
        if job is None or job.get("status") != "queued":
            _release_job_claim(job_id)
            return
        now = _now_iso()
        job.update(
            {
                "status": "running",
                "stage": "preparing",
                "progress": 0.08,
                "started_at": now,
                "updated_at": now,
                "error": None,
                "cancel_requested": False,
            }
        )
        _write_job(job)
        try:
            process = Process(
                target=_execute_job,
                args=(job_id,),
                name=f"frame-audit-job-{job_id}",
            )
        except Exception:
            _release_job_claim(job_id)
            raise
        process.daemon = True
        JOB_PROCESSES[job_id] = process
        try:
            process.start()
        except Exception as exc:
            JOB_PROCESSES.pop(job_id, None)
            process.close()
            _release_job_claim(job_id)
            try:
                _update_job_state(
                    job_id,
                    status="failed",
                    stage="failed",
                    progress=0.0,
                    finished_at=_now_iso(),
                    error=f"{type(exc).__name__}: {exc}",
                )
            except Exception as state_exc:
                _record_worker_error(state_exc)
            return

    try:
        process.join()
        job_after = _read_job(job_id)
        if (
            job_after is not None
            and job_after.get("status") == "running"
            and process.exitcode not in {0, None}
        ):
            _update_job_state(
                job_id,
                status="failed",
                stage="failed",
                progress=0.0,
                finished_at=_now_iso(),
                error=f"评估进程异常退出，退出码: {process.exitcode}",
            )
    finally:
        with JOB_LOCK:
            JOB_PROCESSES.pop(job_id, None)
        process.close()
        _release_job_claim(job_id)


def _terminate_job_process(job_id: str) -> None:
    with JOB_LOCK:
        process = JOB_PROCESSES.get(job_id)
    try:
        if process is None:
            raise HTTPException(
                status_code=409,
                detail="评估进程尚未准备好，稍后再试。",
            )
        if process.is_alive():
            if os.name == "nt" and getattr(process, "pid", None):
                # The evaluator can launch ffmpeg, VBench, or Docker children.
                # Terminating only the Python parent leaves those processes alive.
                subprocess.run(
                    [
                        "taskkill",
                        "/PID",
                        str(process.pid),
                        "/T",
                        "/F",
                    ],
                    capture_output=True,
                    check=False,
                )
            else:
                process.terminate()
        process.join(timeout=JOB_PROCESS_JOIN_TIMEOUT_SECONDS)
        if process.is_alive():
            process.kill()
            process.join(timeout=JOB_PROCESS_JOIN_TIMEOUT_SECONDS)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail=f"无法中断评估进程: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        # taskkill /T /F can bypass the child's Python finally block.
        cleanup_project_subst_mappings(PROJECT_ROOT)


def _terminate_all_job_processes() -> None:
    with JOB_LOCK:
        job_ids = list(JOB_PROCESSES)
    for job_id in job_ids:
        try:
            _terminate_job_process(job_id)
        except HTTPException:
            continue


def _mark_dispatch_failure(job_id: str, exc: Exception) -> None:
    try:
        _update_job_state(
            job_id,
            status="failed",
            stage="failed",
            progress=0.0,
            finished_at=_now_iso(),
            error=f"Queue worker failed: {type(exc).__name__}: {exc}",
        )
    except Exception as mark_exc:
        _record_worker_error(mark_exc)
        job = _read_job(job_id)
        if job is not None and job.get("status") == "running":
            try:
                _update_job_state(
                    job_id,
                    status="queued",
                    stage="queued",
                    progress=0.0,
                    started_at=None,
                    queued_at=_now_iso(),
                    error=f"Will retry after worker failure: {type(exc).__name__}: {exc}",
                )
            except Exception as retry_exc:
                _record_worker_error(retry_exc)


def _queue_worker_loop() -> None:
    global JOB_WORKER_HEARTBEAT, JOB_WORKER_STATE
    next_reconcile = 0.0
    while not JOB_WORKER_STOP.is_set():
        now = time.monotonic()
        with JOB_LOCK:
            JOB_WORKER_HEARTBEAT = now
            JOB_WORKER_STATE = "idle"
        if now >= next_reconcile:
            try:
                _reconcile_queued_jobs()
            except Exception as exc:
                _record_worker_error(exc)
            next_reconcile = now + JOB_RECONCILE_INTERVAL_SECONDS

        job_id: str | None = None
        client_ip: str | None = None
        try:
            JOB_DISPATCH_QUEUE.get(timeout=0.5)
            JOB_DISPATCH_QUEUE.task_done()
            client_ip = _select_next_dispatch_ip()
            job_id = _take_next_job(client_ip) if client_ip is not None else None
            if job_id is not None:
                with JOB_LOCK:
                    JOB_WORKER_STATE = "running"
                _process_job(job_id)
        except queue.Empty:
            continue
        except Exception as exc:
            _record_worker_error(exc)
            if job_id is not None:
                _mark_dispatch_failure(job_id, exc)
        finally:
            if client_ip is not None:
                try:
                    _reschedule_ip(client_ip)
                except Exception as exc:
                    _record_worker_error(exc)
            with JOB_LOCK:
                JOB_WORKER_HEARTBEAT = time.monotonic()
                JOB_WORKER_STATE = "idle"
    with JOB_LOCK:
        JOB_WORKER_STATE = "stopped"


def _recover_persisted_jobs() -> None:
    global JOB_QUEUE_RECOVERED
    with JOB_LOCK:
        if JOB_QUEUE_RECOVERED:
            return
        persisted_jobs = sorted(
            _all_jobs(),
            key=lambda item: str(
                item.get("queued_at") or item.get("created_at", "")
            ),
        )
        for job in persisted_jobs:
            if job.get("status") != "running":
                _release_job_claim(str(job["job_id"]))
            if not job.get("client_ip") and job.get("status") in {"queued", "running"}:
                now = _now_iso()
                job.update(
                    {
                        "status": "canceled",
                        "stage": "canceled",
                        "finished_at": now,
                        "updated_at": now,
                        "error": "无法确认任务来源，已停止恢复。",
                        "cancel_requested": False,
                    }
                )
                _safe_recovery_write(job)
                continue
            if job.get("status") == "canceling":
                now = _now_iso()
                job.update(
                    {
                        "status": "canceled",
                        "stage": "canceled",
                        "finished_at": now,
                        "updated_at": now,
                        "error": "服务重启后确认任务已中断。",
                        "cancel_requested": False,
                    }
                )
                _safe_recovery_write(job)
                continue
            if job.get("status") == "running":
                _release_job_claim(str(job["job_id"]))
                now = _now_iso()
                job.update(
                    {
                        "status": "queued",
                        "stage": "queued",
                        "progress": 0.0,
                        "queued_at": now,
                        "started_at": None,
                        "updated_at": now,
                        "error": "Requeued after service restart.",
                        "cancel_requested": False,
                    }
                )
                if not _safe_recovery_write(job):
                    continue
            if job.get("status") == "queued":
                _enqueue_job(str(job["job_id"]), _job_owner(job))
        JOB_QUEUE_RECOVERED = True


def _ensure_queue_worker() -> None:
    global JOB_WORKER, JOB_WORKER_LAST_ERROR, JOB_WORKER_STATE
    with JOB_LOCK:
        JOB_WORKER_STOP.clear()
        if JOB_WORKER is None or not JOB_WORKER.is_alive():
            if not _acquire_queue_lease():
                JOB_WORKER_STATE = "external_worker"
                return
            _recover_persisted_jobs()
            _clear_dispatch_state()
            _restore_queued_jobs()
            JOB_WORKER_LAST_ERROR = None
            JOB_WORKER = threading.Thread(
                target=_queue_worker_loop,
                name="frame-audit-job-worker",
                daemon=True,
            )
            JOB_WORKER.start()


@app.get("/", response_class=FileResponse)
def index() -> Path:
    return WEB_DIR / "index.html"


@app.get("/showcase", response_class=FileResponse)
def public_showcase_page() -> Path:
    return WEB_DIR / "showcase.html"


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "frame-audit",
        "queue_worker": _worker_snapshot(),
        "public_showcase": _public_showcase_status(),
    }


@app.get("/api/public-showcase")
def public_showcase(
    limit: int = 50,
    query: str = "",
    category: str = "",
) -> dict[str, Any]:
    try:
        return _list_public_showcase(
            limit=limit,
            query=query,
            category=category,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/public-showcase/{item_id}")
def get_public_showcase(item_id: str) -> dict[str, Any]:
    try:
        return _get_public_showcase(item_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/public-showcase/{item_id}/files/{file_key}")
def download_public_showcase_file(
    item_id: str,
    file_key: str,
) -> FileResponse:
    try:
        path = _resolve_public_showcase_file(item_id, file_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path)


@app.get("/api/models")
def models() -> dict[str, Any]:
    policy = resolve_policy("auto")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": get_model_inventory(),
        "recommendation": get_model_recommendation(policy.vram_gb),
        "hardware_policy": policy.to_dict(),
        "wangxing_au": _wangxing_au_status(),
    }


@app.get("/api/hardware")
def hardware(device: str = "auto") -> dict[str, Any]:
    return resolve_policy(device).to_dict()


@app.post("/api/jobs", status_code=202)
def create_job(
    request: Request,
    name: str = Form(""),
    reuse_job_id: str = Form(""),
    result_video: UploadFile = File(...),
    gt_video: UploadFile | None = File(None),
    reference_images: list[UploadFile] | None = File(None),
    reference_video: list[UploadFile] | None = File(None),
    prompt_text: str = Form(""),
    max_frames: int = Form(8),
    calculate_lpips: bool = Form(True),
    device: str = Form("auto"),
    manual_expression_score: str = Form(""),
    manual_aesthetic_score: str = Form(""),
    wangxing_au_enabled: bool = Form(False),
    wangxing_expected_class: str = Form("auto"),
) -> JSONResponse:
    client_ip = _client_ip(request)
    normalized_reuse_job_id = (
        reuse_job_id if isinstance(reuse_job_id, str) else ""
    ).strip()
    reuse_source_job = None
    if normalized_reuse_job_id:
        reuse_source_job = _read_job(normalized_reuse_job_id)
        if reuse_source_job is None:
            raise HTTPException(status_code=404, detail="Source job not found")
        _assert_job_owner(reuse_source_job, client_ip)
        if reuse_source_job.get("status") not in {
            "completed",
            "failed",
            "canceled",
        }:
            raise HTTPException(
                status_code=409,
                detail="Only finished jobs can provide reusable reference files.",
            )
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid4().hex[:12]
    )
    try:
        job = _prepare_job(
            run_id=run_id,
            client_ip=client_ip,
            name=name,
            reuse_source_job=reuse_source_job,
            result_video=result_video,
            gt_video=gt_video,
            reference_images=reference_images,
            reference_video=reference_video,
            prompt_text=prompt_text,
            max_frames=max_frames,
            calculate_lpips=calculate_lpips,
            device=device,
            manual_expression_score=manual_expression_score,
            manual_aesthetic_score=manual_aesthetic_score,
            wangxing_au_enabled=wangxing_au_enabled,
            wangxing_expected_class=wangxing_expected_class,
        )
        with JOB_LOCK:
            _write_job(job)
            _enqueue_job(run_id, client_ip)
        _ensure_queue_worker()
        jobs = _jobs_for_ip(_all_jobs(), client_ip)
        positions = _queued_positions(jobs, client_ip)
        return JSONResponse(
            status_code=202,
            content=_job_response(
                job,
                queue_position=positions.get(run_id),
            ),
        )
    except HTTPException:
        shutil.rmtree(WEB_RUNS_DIR / run_id, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(WEB_RUNS_DIR / run_id, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


@app.get("/api/jobs")
def list_jobs(
    request: Request,
    limit: int = 20,
    *,
    ensure_worker: bool = True,
) -> dict[str, Any]:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100.")
    if ensure_worker:
        _ensure_queue_worker()
    client_ip = _client_ip(request)
    jobs = _jobs_for_ip(_all_jobs(), client_ip)
    display_jobs = _display_jobs(jobs)
    positions = _queued_positions(jobs, client_ip)
    items = [
        _job_response(
            job,
            queue_position=positions.get(str(job["job_id"])),
        )
        for job in display_jobs[:limit]
    ]
    active = next(
        (item for item in items if item["status"] == "running"),
        None,
    )
    queued_count = sum(1 for job in jobs if job.get("status") == "queued")
    return {
        "jobs": items,
        "active_job_id": active["job_id"] if active else None,
        "queued_count": queued_count,
        "total_count": len(jobs),
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    _ensure_queue_worker()
    job = _read_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    client_ip = _client_ip(request)
    _assert_job_owner(job, client_ip)
    positions = _queued_positions(_jobs_for_ip(_all_jobs(), client_ip), client_ip)
    return _job_response(
        job,
        include_result=True,
        queue_position=positions.get(str(job["job_id"])),
    )


@app.patch("/api/jobs/{job_id}")
def update_job(
    job_id: str,
    request: Request,
    update: JobUpdate,
) -> dict[str, Any]:
    _ensure_queue_worker()

    cancel_running = False
    enqueue_after_update = False
    with JOB_LOCK:
        job = _read_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        _assert_job_owner(job, _client_ip(request))
        status = str(job.get("status"))
        requested_name: str | None = None
        if update.name is not None:
            requested_name = update.name.strip()
            if not requested_name and update.action == "retry":
                requested_name = _job_name_fallback(job)
            if not requested_name:
                raise HTTPException(status_code=422, detail="Job name cannot be empty.")
        if update.action == "cancel":
            if status == "canceled":
                pass
            elif status == "running":
                if JOB_PROCESSES.get(job_id) is None:
                    if not _is_stale_running_job(job):
                        raise HTTPException(
                            status_code=409,
                            detail="评估进程尚未准备好，稍后再试。",
                        )
                    job.update(
                        {
                            "status": "canceled",
                            "stage": "canceled",
                            "progress": float(job.get("progress", 0)),
                            "finished_at": _now_iso(),
                            "updated_at": _now_iso(),
                            "error": "Canceled a stale task with no live process.",
                            "cancel_requested": False,
                        }
                    )
                else:
                    cancel_running = True
                    job.update(
                        {
                            "status": "canceling",
                            "stage": "canceled",
                            "cancel_previous_stage": job.get("stage", "models"),
                            "cancel_requested": True,
                            "updated_at": _now_iso(),
                            "error": "正在中断评估进程。",
                        }
                    )
                    # Persist the cancellation before killing the worker. The
                    # child may otherwise write a stale completed state.
                    _write_job(job)
            elif status == "queued":
                job.update(
                    {
                        "status": "canceled",
                        "stage": "canceled",
                        "progress": 0.0,
                        "finished_at": _now_iso(),
                        "error": "已由用户取消。",
                    }
                )
            else:
                raise HTTPException(
                    status_code=409,
                    detail="只有排队中或运行中的任务可以取消。",
                )
        elif update.action == "retry":
            if status not in {"failed", "canceled", "completed"}:
                raise HTTPException(
                    status_code=409,
                    detail="只有已完成、失败或已取消的任务可以重试。",
                )
            parameters = dict(job.get("parameters", {}))
            parameter_fields = {
                "prompt_text",
                "max_frames",
                "calculate_lpips",
                "device",
                "manual_expression_score",
                "manual_aesthetic_score",
                "wangxing_au_enabled",
                "wangxing_expected_class",
            }
            for field_name in update.model_fields_set & parameter_fields:
                value = getattr(update, field_name)
                if field_name == "prompt_text":
                    value = value or ""
                elif field_name == "wangxing_expected_class":
                    value = _normalize_wangxing_class(value) or "auto"
                parameters[field_name] = value
            job["parameters"] = parameters
            run_dir = _job_dir(str(job["job_id"]))
            for output_name in (
                "result.json",
                "summary.csv",
                "frame_metrics.csv",
                "wangxing_au_result.json",
            ):
                (run_dir / output_name).unlink(missing_ok=True)
            shutil.rmtree(run_dir / "wangxing_au", ignore_errors=True)
            now = _now_iso()
            job.update(
                {
                    "status": "queued",
                    "stage": "queued",
                    "progress": 0.0,
                    "queued_at": now,
                    "started_at": None,
                    "finished_at": None,
                    "error": None,
                    "cancel_requested": False,
                }
            )
            enqueue_after_update = True
            _write_job_params(job)
        if requested_name is not None:
            job["name"] = requested_name
        if not cancel_running:
            job["updated_at"] = _now_iso()
            _write_job(job)

    if cancel_running:
        try:
            _terminate_job_process(job_id)
        except HTTPException as exc:
            with JOB_LOCK:
                job = _read_job(job_id)
                if job is not None and job.get("status") == "canceling":
                    job.update(
                        {
                            "status": "running",
                            "stage": job.get("cancel_previous_stage", "models"),
                            "cancel_requested": False,
                            "updated_at": _now_iso(),
                            "error": str(exc.detail),
                        }
                    )
                    _write_job(job)
            raise
        with JOB_LOCK:
            job = _read_job(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            if job.get("status") in {"running", "canceling"}:
                job.update(
                    {
                        "status": "canceled",
                        "stage": "canceled",
                        "progress": float(job.get("progress", 0)),
                        "finished_at": _now_iso(),
                        "updated_at": _now_iso(),
                        "error": "已由用户中断。",
                        "cancel_requested": False,
                        "cancel_previous_stage": None,
                    }
                )
                if requested_name is not None:
                    job["name"] = requested_name
                _write_job(job)
    if enqueue_after_update:
        _enqueue_job(str(job["job_id"]), _job_owner(job))
        _ensure_queue_worker()
    client_ip = _client_ip(request)
    positions = _queued_positions(
        _jobs_for_ip(_all_jobs(), client_ip),
        client_ip,
    )
    return _job_response(
        job,
        queue_position=positions.get(str(job["job_id"])),
    )


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, request: Request) -> dict[str, Any]:
    with JOB_LOCK:
        job = _read_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        _assert_job_owner(job, _client_ip(request))
        if job.get("status") in {"running", "canceling"}:
            raise HTTPException(
                status_code=409,
                detail="运行中的任务不能删除，请先中断任务。",
            )
        run_dir = _job_dir(job_id)
        try:
            _remove_job_directory(run_dir)
        except OSError as exc:
            raise HTTPException(
                status_code=409,
                detail="任务文件正在释放，请稍后重试。",
            ) from exc
        _remove_job_from_dispatch(job_id, _job_owner(job))
    return {"job_id": job_id, "deleted": True}


@app.get("/api/runs/{run_id}/{filename}")
def download_run_file(
    run_id: str,
    filename: str,
    request: Request,
) -> FileResponse:
    safe_run_id = Path(run_id).name
    safe_filename = Path(filename).name
    if safe_run_id != run_id or safe_filename != filename:
        raise HTTPException(status_code=404, detail="File not found")
    target = (WEB_RUNS_DIR / safe_run_id / safe_filename).resolve()
    run_root = (WEB_RUNS_DIR / safe_run_id).resolve()
    if run_root.parent != WEB_RUNS_DIR.resolve() or not target.is_relative_to(run_root):
        raise HTTPException(status_code=404, detail="File not found")
    job = _read_job(run_id)
    if job is not None:
        _assert_job_owner(job, _client_ip(request))
        if filename in GENERATED_REPORT_FILES and job.get("status") != "completed":
            raise HTTPException(status_code=404, detail="File not found")
    else:
        owner = _read_job_owner(run_id)
        if owner is not None and owner != _client_ip(request):
            raise HTTPException(status_code=404, detail="File not found")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


@app.post("/api/evaluate")
def evaluate(
    request: Request,
    result_video: UploadFile = File(...),
    gt_video: UploadFile | None = File(None),
    reference_images: list[UploadFile] | None = File(None),
    reference_video: list[UploadFile] | None = File(None),
    prompt_text: str = Form(""),
    max_frames: int = Form(8),
    calculate_lpips: bool = Form(True),
    device: str = Form("auto"),
    manual_expression_score: str = Form(""),
    manual_aesthetic_score: str = Form(""),
    wangxing_au_enabled: bool = Form(False),
    wangxing_expected_class: str = Form("auto"),
) -> JSONResponse:
    client_ip = _client_ip(request)
    _validate_upload_count(
        result_video=result_video,
        gt_video=gt_video,
        reference_images=reference_images,
        reference_video=reference_video,
    )
    result_suffix = _upload_suffix(result_video, VIDEO_SUFFIXES)
    if not is_video_path(f"result{result_suffix}"):
        raise HTTPException(status_code=415, detail="Result upload must be a video.")
    if max_frames < 2 or max_frames > 256:
        raise HTTPException(status_code=422, detail="max_frames must be between 2 and 256.")
    if device not in {"cpu", "auto", "cuda"}:
        raise HTTPException(status_code=422, detail="Unsupported inference device.")

    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid4().hex[:12]
    )
    run_dir = WEB_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    uploaded: dict[str, Any] = {
        "result_video": None,
        "gt_video": None,
        "reference_images": [],
        "reference_video": None,
    }
    upload_budget = [0]
    try:
        _write_job_owner(run_id, client_ip)
        uploaded["result_video"] = _save_upload(
            result_video,
            run_dir,
            "result",
            VIDEO_SUFFIXES,
            total_bytes=upload_budget,
        )
        uploaded["gt_video"] = _save_upload(
            gt_video,
            run_dir,
            "gt",
            VIDEO_SUFFIXES,
            total_bytes=upload_budget,
        )
        uploaded["reference_images"] = [
            saved
            for index, reference_image in enumerate(reference_images or [], start=1)
            if (
                saved := _save_upload(
                    reference_image,
                    run_dir,
                    f"reference_{index:02d}",
                    IMAGE_SUFFIXES,
                    total_bytes=upload_budget,
                )
            )
        ]
        uploaded["reference_video"] = _save_reference_videos(
            reference_video,
            run_dir,
            total_bytes=upload_budget,
        )
        result_path = uploaded["result_video"]
        if result_path is None:
            raise HTTPException(status_code=422, detail="Result video is required.")
        _validate_video(result_path, "Result video")
        if uploaded["gt_video"]:
            _validate_video(uploaded["gt_video"], "GT video")
        if uploaded["reference_video"]:
            _validate_video(uploaded["reference_video"], "Reference video")

        prompt = prompt_text.strip()
        if len(prompt) > 10_000:
            raise HTTPException(
                status_code=422,
                detail="Prompt must be 10,000 characters or fewer.",
            )

        try:
            result = evaluate_all(
                result_path=result_path,
                ground_truth=uploaded["gt_video"],
                reference_image=uploaded["reference_images"],
                reference_video=uploaded["reference_video"],
                prompt_text=prompt or None,
                max_frames=max_frames,
                calculate_lpips=calculate_lpips,
                device=device,
                manual_expression_score=_optional_score(
                    manual_expression_score,
                    "manual_expression_score",
                ),
                manual_aesthetic_score=_optional_score(
                    manual_aesthetic_score,
                    "manual_aesthetic_score",
                ),
                vbench_output_root=run_dir,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Input videos cannot be evaluated: {exc}",
            ) from exc
        if wangxing_au_enabled:
            wangxing_au = _run_wangxing_au_assessment(
                result_path=result_path,
                reference_image_paths=uploaded["reference_images"],
                expected_class=_normalize_wangxing_class(
                    wangxing_expected_class
                ),
                device=device,
                run_dir=run_dir,
                prompt_text=prompt or None,
            )
            result["wangxing_au"] = wangxing_au
            _attach_wangxing_evidence(
                result,
                wangxing_au=wangxing_au,
                prompt_text=prompt or None,
                driver_source=None,
            )
        else:
            result["wangxing_au"] = {
                "status": "not_applicable",
                "scope": "wangxing_specialization_only",
                "reason": (
                    "Wang Xing AU specialization was not selected. "
                    "Generic evaluation is unaffected."
                ),
            }
        result["web_run_id"] = run_id
        result["result_video"] = probe_video(result_path).to_dict()
        if uploaded["gt_video"]:
            result["ground_truth_video"] = probe_video(uploaded["gt_video"]).to_dict()

        summary_path = run_dir / "summary.csv"
        frame_path = run_dir / "frame_metrics.csv"
        json_path = run_dir / "result.json"
        pd.DataFrame(result["summary"]).to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(result.get("frame_records", [])).to_csv(
            frame_path,
            index=False,
            encoding="utf-8-sig",
        )
        json_path.write_text(
            json.dumps(_json_safe(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        payload = _result_payload(run_id, result, summary_path, json_path, uploaded)
        payload["downloads"]["frame_csv"] = _file_url(run_id, frame_path)
        return JSONResponse(content=payload)
    except HTTPException:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    except (FileNotFoundError, ValueError) as exc:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(
            status_code=422,
            detail=f"Input video is not a readable video or cannot be decoded: {exc}",
        ) from exc
    except Exception as exc:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("EVALUATOR_HOST", "127.0.0.1")
    certfile = os.environ.get("EVALUATOR_TLS_CERTFILE", "").strip() or None
    keyfile = os.environ.get("EVALUATOR_TLS_KEYFILE", "").strip() or None
    if host not in {"127.0.0.1", "::1", "localhost"}:
        if not os.environ.get("FRAME_AUDIT_API_KEY", "").strip():
            raise SystemExit("Public binding requires FRAME_AUDIT_API_KEY.")
        if (
            (not certfile or not keyfile)
            and os.environ.get("EVALUATOR_ALLOW_INSECURE_PUBLIC", "").lower()
            not in {"1", "true", "yes", "on"}
        ):
            raise SystemExit(
                "Public binding requires EVALUATOR_TLS_CERTFILE/KEYFILE "
                "or an explicit insecure override."
            )
        os.environ["FRAME_AUDIT_REQUIRE_AUTH"] = "1"
    uvicorn.run(
        "web_app:app",
        host=host,
        port=int(os.environ.get("EVALUATOR_PORT", "7860")),
        reload=False,
        ssl_certfile=certfile,
        ssl_keyfile=keyfile,
    )
