from __future__ import annotations

import json
import ipaddress
import math
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

import numpy as np
import pandas as pd
from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from pydantic import BaseModel, Field

from evaluator.holistic_evaluator import (
    evaluate_all,
    get_model_inventory,
    get_model_recommendation,
)
from evaluator.hardware_policy import resolve_policy
from evaluator.media import transcode_video_for_browser
from evaluator.runtime import OUTPUT_DIR, PROJECT_ROOT
from evaluator.video_metrics import is_video_path, probe_video


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
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".m4v"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
WANGXING_AU_CLASSES = {
    "smile",
    "anger",
    "surprise",
    "fear",
    "annoyance",
    "sadness",
}
WANGXING_AU_PROFILE_PATH = PROJECT_ROOT / "data/au/wangxing_au_profile.json"
WANGXING_AU_CLASSIFIER_PATH = PROJECT_ROOT / "data/au/au_leakage_classifier.json"

WEB_RUNS_DIR.mkdir(parents=True, exist_ok=True)

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


class JobUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    action: Literal["cancel", "retry"] | None = None


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    global JOB_WORKER
    _ensure_queue_worker()
    try:
        yield
    finally:
        JOB_WORKER_STOP.set()
        _terminate_all_job_processes()
        with JOB_LOCK:
            worker = JOB_WORKER
        if worker is not None and worker.is_alive():
            worker.join(timeout=3)
        with JOB_LOCK:
            JOB_WORKER = None
            JOB_WORKER_STOP.clear()

app = FastAPI(
    title="Frame Audit",
    description="Local video generation evaluation workspace.",
    lifespan=app_lifespan,
)
app.mount("/assets", StaticFiles(directory=str(WEB_DIR)), name="assets")


@app.middleware("http")
async def disable_asset_cache(request: Any, call_next: Any) -> Any:
    response = await call_next(request)
    if request.url.path.startswith("/assets/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
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


def _normalize_wangxing_class(value: str | None) -> str | None:
    normalized = str(value or "").strip().lower()
    if not normalized or normalized == "auto":
        return None
    if normalized not in WANGXING_AU_CLASSES:
        raise HTTPException(
            status_code=422,
            detail=(
                "wangxing_expected_class must be auto, smile, anger, "
                "surprise, fear, annoyance, or sadness."
            ),
        )
    return normalized


def _wangxing_au_status() -> dict[str, Any]:
    ready = (
        WANGXING_AU_PROFILE_PATH.is_file()
        and WANGXING_AU_CLASSIFIER_PATH.is_file()
    )
    return {
        "ready": ready,
        "profile": str(WANGXING_AU_PROFILE_PATH),
        "classifier": str(WANGXING_AU_CLASSIFIER_PATH),
        "classes": sorted(WANGXING_AU_CLASSES),
        "note": (
            "Uses Wang Xing AU profile as the primary expression-fit signal."
            if ready
            else "Train the Wang Xing AU profile and classifier first."
        ),
    }


def _run_wangxing_au_assessment(
    *,
    result_path: Path,
    reference_image_paths: list[Path],
    reference_video_path: Path | None,
    expected_class: str | None,
    device: str,
    run_dir: Path,
) -> dict[str, Any]:
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
        str(PROJECT_ROOT / "scripts/evaluate_generated_video.py"),
        "--generated-video",
        str(result_path),
        "--output-root",
        str(run_dir / "wangxing_au"),
        "--au-profile",
        str(WANGXING_AU_PROFILE_PATH),
        "--leakage-classifier",
        str(WANGXING_AU_CLASSIFIER_PATH),
        "--output",
        str(output_path),
        "--device",
        au_device,
    ]
    for reference_image_path in reference_image_paths:
        command.extend(["--target-image", str(reference_image_path)])
    if reference_video_path is not None:
        command.extend(["--driver-video", str(reference_video_path)])
    if expected_class is not None:
        command.extend(["--expected-class", expected_class])

    try:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "status": "unavailable",
            "reason": (
                "Wang Xing AU extraction could not complete. "
                "Check that the result video contains a visible face."
            ),
            "error_type": type(exc).__name__,
        }

    try:
        return json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "unavailable",
            "reason": "Wang Xing AU report was not written.",
            "error_type": type(exc).__name__,
        }


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
        "name": job.get("name") or "result video",
        "storage_path": f"web_runs/{job['job_id']}",
        "status": job.get("status", "queued"),
        "stage": job.get("stage", "queued"),
        "progress": float(job.get("progress", 0)),
        "queue_position": queue_position,
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
                output.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    if allowed == VIDEO_SUFFIXES:
        # Most generated MP4 files are already browser/OpenCV compatible.
        # Avoid blocking job creation with a second full video encode.
        if suffix == ".mp4":
            try:
                probe_video(target)
            except (FileNotFoundError, ValueError):
                pass
            else:
                normalized = run_dir / f"{stem}.mp4"
                target.replace(normalized)
                return normalized
        normalized = run_dir / f"{stem}.mp4"
        transcode_video_for_browser(target, normalized)
        return normalized
    return target


def _validate_video(path: Path, label: str) -> None:
    try:
        probe_video(path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{label} is not a readable video: {exc}",
        ) from exc


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
    result_video: UploadFile,
    gt_video: UploadFile | None,
    reference_images: list[UploadFile] | None,
    reference_video: UploadFile | None,
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

    run_dir = _job_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    uploaded: dict[str, Any] = {
        "result_video": None,
        "gt_video": None,
        "reference_images": [],
        "reference_video": None,
    }
    try:
        uploaded["result_video"] = _save_upload(
            result_video,
            run_dir,
            "result",
            VIDEO_SUFFIXES,
        )
        uploaded["gt_video"] = _save_upload(gt_video, run_dir, "gt", VIDEO_SUFFIXES)
        uploaded["reference_images"] = [
            saved
            for index, reference_image in enumerate(reference_images or [], start=1)
            if (
                saved := _save_upload(
                    reference_image,
                    run_dir,
                    f"reference_{index:02d}",
                    IMAGE_SUFFIXES,
                )
            )
        ]
        uploaded["reference_video"] = _save_upload(
            reference_video,
            run_dir,
            "reference_motion",
            VIDEO_SUFFIXES,
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
        "reference_video": (
            Path(reference_video.filename).name
            if reference_video is not None and reference_video.filename
            else None
        ),
    }
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
        "name": Path(result_video.filename or "result video").name,
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
            max_frames=int(parameters.get("max_frames", 64)),
            calculate_lpips=bool(parameters.get("calculate_lpips", True)),
            device=str(parameters.get("device", "cpu")),
            manual_expression_score=parameters.get("manual_expression_score"),
            manual_aesthetic_score=parameters.get("manual_aesthetic_score"),
            vbench_output_root=_job_dir(job_id),
        )
        if bool(parameters.get("wangxing_au_enabled", True)):
            _update_job_state(
                job_id,
                stage="wangxing_au",
                progress=0.72,
            )
            expected_class = _normalize_wangxing_class(
                str(parameters.get("wangxing_expected_class", "auto"))
            )
            result["wangxing_au"] = _run_wangxing_au_assessment(
                result_path=result_path,
                reference_image_paths=reference_paths,
                reference_video_path=reference_video_path,
                expected_class=expected_class,
                device=str(parameters.get("device", "cpu")),
                run_dir=_job_dir(job_id),
            )
        else:
            result["wangxing_au"] = {
                "status": "disabled",
                "reason": "Wang Xing AU assessment was disabled for this job.",
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
        except Exception:
            pass


def _process_job(job_id: str) -> None:
    with JOB_LOCK:
        job = _read_job(job_id)
        if job is None or job.get("status") != "queued":
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
        process = Process(
            target=_execute_job,
            args=(job_id,),
            name=f"frame-audit-job-{job_id}",
        )
        process.daemon = True
        JOB_PROCESSES[job_id] = process
        try:
            process.start()
        except Exception as exc:
            JOB_PROCESSES.pop(job_id, None)
            process.close()
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


def _terminate_job_process(job_id: str) -> None:
    with JOB_LOCK:
        process = JOB_PROCESSES.get(job_id)
    if process is None:
        raise HTTPException(
            status_code=409,
            detail="评估进程尚未准备好，稍后再试。",
        )
    try:
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
    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail=f"无法中断评估进程: {type(exc).__name__}: {exc}",
        ) from exc


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
            client_ip = JOB_DISPATCH_QUEUE.get(timeout=0.5)
            job_id = _take_next_job(client_ip)
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
                finally:
                    JOB_DISPATCH_QUEUE.task_done()
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
    global JOB_WORKER, JOB_WORKER_LAST_ERROR
    with JOB_LOCK:
        JOB_WORKER_STOP.clear()
        _recover_persisted_jobs()
        if JOB_WORKER is None or not JOB_WORKER.is_alive():
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


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "frame-audit",
        "queue_worker": _worker_snapshot(),
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    policy = resolve_policy("auto")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": get_model_inventory(),
        "recommendation": get_model_recommendation(),
        "hardware_policy": policy.to_dict(),
        "wangxing_au": _wangxing_au_status(),
    }


@app.get("/api/hardware")
def hardware(device: str = "auto") -> dict[str, Any]:
    return resolve_policy(device).to_dict()


@app.post("/api/jobs", status_code=202)
def create_job(
    request: Request,
    result_video: UploadFile = File(...),
    gt_video: UploadFile | None = File(None),
    reference_images: list[UploadFile] | None = File(None),
    reference_video: UploadFile | None = File(None),
    prompt_text: str = Form(""),
    max_frames: int = Form(64),
    calculate_lpips: bool = Form(True),
    device: str = Form("auto"),
    manual_expression_score: str = Form(""),
    manual_aesthetic_score: str = Form(""),
    wangxing_au_enabled: bool = Form(True),
    wangxing_expected_class: str = Form("auto"),
) -> JSONResponse:
    client_ip = _client_ip(request)
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid4().hex[:12]
    )
    try:
        job = _prepare_job(
            run_id=run_id,
            client_ip=client_ip,
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
def list_jobs(request: Request, limit: int = 20) -> dict[str, Any]:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100.")
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
    if update.name is not None and not update.name.strip():
        raise HTTPException(status_code=422, detail="Job name cannot be empty.")

    cancel_running = False
    enqueue_after_update = False
    with JOB_LOCK:
        job = _read_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        _assert_job_owner(job, _client_ip(request))
        status = str(job.get("status"))
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
            if status not in {"failed", "canceled"}:
                raise HTTPException(
                    status_code=409,
                    detail="只有失败或已取消的任务可以重试。",
                )
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
        if update.name is not None:
            job["name"] = update.name.strip()
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
                if update.name is not None:
                    job["name"] = update.name.strip()
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
        shutil.rmtree(run_dir, ignore_errors=False)
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
    reference_video: UploadFile | None = File(None),
    prompt_text: str = Form(""),
    max_frames: int = Form(64),
    calculate_lpips: bool = Form(True),
    device: str = Form("auto"),
    manual_expression_score: str = Form(""),
    manual_aesthetic_score: str = Form(""),
    wangxing_au_enabled: bool = Form(True),
    wangxing_expected_class: str = Form("auto"),
) -> JSONResponse:
    client_ip = _client_ip(request)
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
    try:
        _write_job_owner(run_id, client_ip)
        uploaded["result_video"] = _save_upload(
            result_video,
            run_dir,
            "result",
            VIDEO_SUFFIXES,
        )
        uploaded["gt_video"] = _save_upload(gt_video, run_dir, "gt", VIDEO_SUFFIXES)
        uploaded["reference_images"] = [
            saved
            for index, reference_image in enumerate(reference_images or [], start=1)
            if (
                saved := _save_upload(
                    reference_image,
                    run_dir,
                    f"reference_{index:02d}",
                    IMAGE_SUFFIXES,
                )
            )
        ]
        uploaded["reference_video"] = _save_upload(
            reference_video,
            run_dir,
            "reference_motion",
            VIDEO_SUFFIXES,
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
            result["wangxing_au"] = _run_wangxing_au_assessment(
                result_path=result_path,
                reference_image_paths=uploaded["reference_images"],
                reference_video_path=uploaded["reference_video"],
                expected_class=_normalize_wangxing_class(
                    wangxing_expected_class
                ),
                device=device,
                run_dir=run_dir,
            )
        else:
            result["wangxing_au"] = {
                "status": "disabled",
                "reason": "Wang Xing AU assessment was disabled for this job.",
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

    uvicorn.run(
        "web_app:app",
        host=os.environ.get("EVALUATOR_HOST", "127.0.0.1"),
        port=int(os.environ.get("EVALUATOR_PORT", "7860")),
        reload=False,
    )
