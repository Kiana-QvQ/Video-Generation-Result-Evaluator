from __future__ import annotations

import json
import math
import queue
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import File, Form, HTTPException, UploadFile
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
MAX_UPLOAD_BYTES = 1_500 * 1024 * 1024
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".m4v"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

WEB_RUNS_DIR.mkdir(parents=True, exist_ok=True)

JOB_QUEUE: queue.Queue[str] = queue.Queue()
JOB_LOCK = threading.RLock()
JOB_WORKER: threading.Thread | None = None
JOB_QUEUE_RECOVERED = False


class JobUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    action: Literal["cancel", "retry"] | None = None


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    _ensure_queue_worker()
    yield

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


def _job_dir(job_id: str) -> Path:
    safe_job_id = _safe_job_id(job_id)
    run_dir = (WEB_RUNS_DIR / safe_job_id).resolve()
    if run_dir.parent != WEB_RUNS_DIR.resolve():
        raise HTTPException(status_code=404, detail="Job not found")
    return run_dir


def _job_status_path(job_id: str) -> Path:
    return _job_dir(job_id) / JOB_STATUS_FILENAME


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


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
    }
    return {
        key: _file_url(run_id, run_dir / filename)
        for key, filename in filenames.items()
        if (run_dir / filename).is_file()
    }


def _job_response(
    job: dict[str, Any],
    *,
    include_result: bool = False,
    queue_position: int | None = None,
) -> dict[str, Any]:
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
        "uploaded_files": _job_uploaded_urls(job),
        "result_available": job.get("status") == "completed",
    }
    if job.get("status") == "completed":
        response["downloads"] = _result_downloads(job)
    if include_result and job.get("status") == "completed":
        result_path = _job_dir(str(job["job_id"])) / "result.json"
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


def _queued_positions(jobs: list[dict[str, Any]]) -> dict[str, int]:
    queued = sorted(
        (job for job in jobs if job.get("status") == "queued"),
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


def _update_job_state(job_id: str, **changes: Any) -> dict[str, Any]:
    with JOB_LOCK:
        job = _read_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
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
    payload = {
        "run_id": run_id,
        "result": result,
        "downloads": {
            "summary_csv": _file_url(run_id, csv_path),
            "result_json": _file_url(run_id, json_path),
        },
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
    parameters = {
        "prompt_text": prompt,
        "max_frames": max_frames,
        "calculate_lpips": calculate_lpips,
        "device": device,
        "manual_expression_score": expression_score,
        "manual_aesthetic_score": aesthetic_score,
    }
    created_at = _now_iso()
    job = {
        "job_id": run_id,
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
        "parameters": parameters,
    }
    _write_job_params(job)
    return job


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
            }
        )
        _write_job(job)

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
        )
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
        except HTTPException:
            pass


def _queue_worker_loop() -> None:
    while True:
        job_id = JOB_QUEUE.get()
        try:
            _process_job(job_id)
        finally:
            JOB_QUEUE.task_done()


def _recover_persisted_jobs() -> None:
    global JOB_QUEUE_RECOVERED
    with JOB_LOCK:
        if JOB_QUEUE_RECOVERED:
            return
        for job in _all_jobs():
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
                    }
                )
                _write_job(job)
            if job.get("status") == "queued":
                JOB_QUEUE.put(str(job["job_id"]))
        JOB_QUEUE_RECOVERED = True


def _ensure_queue_worker() -> None:
    global JOB_WORKER
    with JOB_LOCK:
        _recover_persisted_jobs()
        if JOB_QUEUE.empty():
            return
        if JOB_WORKER is None or not JOB_WORKER.is_alive():
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
def health() -> dict[str, str]:
    return {"status": "ok", "service": "frame-audit"}


@app.get("/api/models")
def models() -> dict[str, Any]:
    policy = resolve_policy("auto")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": get_model_inventory(),
        "recommendation": get_model_recommendation(),
        "hardware_policy": policy.to_dict(),
    }


@app.get("/api/hardware")
def hardware(device: str = "auto") -> dict[str, Any]:
    return resolve_policy(device).to_dict()


@app.post("/api/jobs", status_code=202)
def create_job(
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
) -> JSONResponse:
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid4().hex[:12]
    )
    try:
        job = _prepare_job(
            run_id=run_id,
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
        )
        _ensure_queue_worker()
        with JOB_LOCK:
            _write_job(job)
        JOB_QUEUE.put(run_id)
        _ensure_queue_worker()
        jobs = _all_jobs()
        positions = _queued_positions(jobs)
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
def list_jobs(limit: int = 20) -> dict[str, Any]:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100.")
    _ensure_queue_worker()
    jobs = _all_jobs()
    display_jobs = _display_jobs(jobs)
    positions = _queued_positions(jobs)
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
def get_job(job_id: str) -> dict[str, Any]:
    _ensure_queue_worker()
    job = _read_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    positions = _queued_positions(_all_jobs())
    return _job_response(
        job,
        include_result=True,
        queue_position=positions.get(str(job["job_id"])),
    )


@app.patch("/api/jobs/{job_id}")
def update_job(job_id: str, update: JobUpdate) -> dict[str, Any]:
    _ensure_queue_worker()
    if update.name is not None and not update.name.strip():
        raise HTTPException(status_code=422, detail="Job name cannot be empty.")

    with JOB_LOCK:
        job = _read_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        status = str(job.get("status"))
        if update.action == "cancel":
            if status == "canceled":
                return _job_response(job)
            if status != "queued":
                raise HTTPException(
                    status_code=409,
                    detail="Only queued jobs can be canceled.",
                )
            job.update(
                {
                    "status": "canceled",
                    "stage": "canceled",
                    "progress": 0.0,
                    "finished_at": _now_iso(),
                    "error": "Canceled by user.",
                }
            )
        elif update.action == "retry":
            if status not in {"failed", "canceled"}:
                raise HTTPException(
                    status_code=409,
                    detail="Only failed or canceled jobs can be retried.",
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
                }
            )
            JOB_QUEUE.put(str(job["job_id"]))
        if update.name is not None:
            job["name"] = update.name.strip()
        job["updated_at"] = _now_iso()
        _write_job(job)

    if update.action == "retry":
        _ensure_queue_worker()
    positions = _queued_positions(_all_jobs())
    return _job_response(
        job,
        queue_position=positions.get(str(job["job_id"])),
    )


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, Any]:
    with JOB_LOCK:
        job = _read_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.get("status") == "running":
            raise HTTPException(
                status_code=409,
                detail="Running jobs cannot be deleted.",
            )
        run_dir = _job_dir(job_id)
        shutil.rmtree(run_dir, ignore_errors=False)
    return {"job_id": job_id, "deleted": True}


@app.get("/api/runs/{run_id}/{filename}")
def download_run_file(run_id: str, filename: str) -> FileResponse:
    safe_run_id = Path(run_id).name
    safe_filename = Path(filename).name
    if safe_run_id != run_id or safe_filename != filename:
        raise HTTPException(status_code=404, detail="File not found")
    target = (WEB_RUNS_DIR / safe_run_id / safe_filename).resolve()
    run_root = (WEB_RUNS_DIR / safe_run_id).resolve()
    if run_root.parent != WEB_RUNS_DIR.resolve() or not target.is_relative_to(run_root):
        raise HTTPException(status_code=404, detail="File not found")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


@app.post("/api/evaluate")
def evaluate(
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
) -> JSONResponse:
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
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Input videos cannot be evaluated: {exc}",
            ) from exc
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
        host="127.0.0.1",
        port=7860,
        reload=False,
    )
