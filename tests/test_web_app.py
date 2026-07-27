from __future__ import annotations

import shutil
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient

import web_app
from web_app import _json_safe, app


class WebAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_homepage_is_available_without_authentication(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("FRAME AUDIT", response.text)
        self.assertIn("process-progress", response.text)
        self.assertIn("radar-chart", response.text)
        self.assertIn("评分原理与准则", response.text)
        self.assertIn("preflight-list", response.text)
        self.assertIn("process-queue", response.text)
        self.assertIn("new-evaluation", response.text)
        self.assertIn("KEY EVIDENCE", response.text)
        self.assertIn("关键证据", response.text)
        self.assertIn("加权总分", response.text)
        self.assertIn("五维评分雷达图", response.text)
        self.assertIn("已覆盖", response.text)
        self.assertIn("中断任务", response.text)
        self.assertIn("处理队列", response.text)
        self.assertIn("看清视频，", response.text)
        self.assertIn("上传视频后开始评分。", response.text)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_web_assets_are_not_cached_during_local_development(self) -> None:
        response = self.client.get("/assets/styles.css")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("cache-control"), "no-store")

    def test_model_inventory_exposes_readiness(self) -> None:
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreaterEqual(len(payload["models"]), 7)
        self.assertTrue(
            all(
                {"name", "purpose", "status", "ready", "note"}
                <= set(model)
                for model in payload["models"]
            )
        )
        self.assertEqual(
            payload["recommendation"]["id"],
            "qwen2_vl_2b_awq",
        )

    def test_run_file_path_is_sandboxed(self) -> None:
        response = self.client.get("/api/runs/../requirements.txt")
        self.assertIn(response.status_code, {404, 422})

    def test_invalid_video_upload_returns_client_error(self) -> None:
        response = self.client.post(
            "/api/evaluate",
            files={"result_video": ("broken.mp4", b"not a video", "video/mp4")},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("readable video", response.json()["detail"])

    def test_unsupported_upload_extension_returns_unsupported_media(self) -> None:
        response = self.client.post(
            "/api/evaluate",
            files={"result_video": ("result.txt", b"not a video", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)

    def test_manual_scores_are_range_checked(self) -> None:
        video_path = Path("outputs/test_result.mp4")
        with video_path.open("rb") as video:
            response = self.client.post(
                "/api/evaluate",
                files={"result_video": ("result.mp4", video, "video/mp4")},
                data={
                    "manual_expression_score": "6",
                    "manual_aesthetic_score": "4",
                    "calculate_lpips": "false",
                    "max_frames": "2",
                    "device": "cpu",
                },
            )
        self.assertEqual(response.status_code, 422)
        self.assertIn("between 1 and 5", response.json()["detail"])

    def test_job_resource_supports_create_read_update_cancel_retry_and_delete(self) -> None:
        video_path = Path("outputs/test_result.mp4")
        with patch("web_app._ensure_queue_worker"):
            with video_path.open("rb") as video:
                create_response = self.client.post(
                    "/api/jobs",
                    files={"result_video": ("result.mp4", video, "video/mp4")},
                    data={
                        "calculate_lpips": "false",
                        "max_frames": "2",
                        "device": "cpu",
                    },
                )
            self.assertEqual(create_response.status_code, 202)
            job = create_response.json()
            job_id = job["job_id"]
            job_dir = Path("outputs/web_runs") / job_id
            self.assertEqual(job["original_files"]["result_video"], "result.mp4")
            self.assertTrue((job_dir / "status.json").is_file())
            self.assertTrue((job_dir / "params.json").is_file())

            list_response = self.client.get("/api/jobs")
            self.assertEqual(list_response.status_code, 200)
            self.assertTrue(
                any(item["job_id"] == job_id for item in list_response.json()["jobs"])
            )

            detail_response = self.client.get(f"/api/jobs/{job_id}")
            self.assertEqual(detail_response.status_code, 200)
            self.assertEqual(detail_response.json()["status"], "queued")

            rename_response = self.client.patch(
                f"/api/jobs/{job_id}",
                json={"name": "renamed review"},
            )
            self.assertEqual(rename_response.status_code, 200)
            self.assertEqual(rename_response.json()["name"], "renamed review")

            cancel_response = self.client.patch(
                f"/api/jobs/{job_id}",
                json={"action": "cancel"},
            )
            self.assertEqual(cancel_response.status_code, 200)
            self.assertEqual(cancel_response.json()["status"], "canceled")

            retry_response = self.client.patch(
                f"/api/jobs/{job_id}",
                json={"action": "retry"},
            )
            self.assertEqual(retry_response.status_code, 200)
            self.assertEqual(retry_response.json()["status"], "queued")

            delete_response = self.client.delete(f"/api/jobs/{job_id}")
            self.assertEqual(delete_response.status_code, 200)
            self.assertFalse(job_dir.exists())
            self.assertEqual(
                self.client.get(f"/api/jobs/{job_id}").status_code,
                404,
            )

    def test_jobs_are_isolated_by_client_ip(self) -> None:
        video_path = Path("outputs/test_result.mp4")
        with patch("web_app._ensure_queue_worker"), patch(
            "web_app._client_ip",
            return_value="192.0.2.10",
        ):
            with video_path.open("rb") as video:
                create_response = self.client.post(
                    "/api/jobs",
                    files={"result_video": ("result.mp4", video, "video/mp4")},
                    data={
                        "calculate_lpips": "false",
                        "max_frames": "2",
                        "device": "cpu",
                    },
                )
            self.assertEqual(create_response.status_code, 202)
            job_id = create_response.json()["job_id"]
            status = web_app._read_job(job_id)
            self.assertEqual(status["client_ip"], "192.0.2.10")

            with patch("web_app._client_ip", return_value="192.0.2.11"):
                self.assertEqual(
                    self.client.get("/api/jobs").json()["jobs"],
                    [],
                )
                self.assertEqual(
                    self.client.get(f"/api/jobs/{job_id}").status_code,
                    404,
                )
                self.assertEqual(
                    self.client.patch(
                        f"/api/jobs/{job_id}",
                        json={"action": "cancel"},
                    ).status_code,
                    404,
                )
                self.assertEqual(
                    self.client.delete(f"/api/jobs/{job_id}").status_code,
                    404,
                )
                self.assertEqual(
                    self.client.get(
                        f"/api/runs/{job_id}/result.mp4"
                    ).status_code,
                    404,
                )

            with patch("web_app._client_ip", return_value="192.0.2.10"):
                self.assertEqual(
                    self.client.get(
                        f"/api/runs/{job_id}/result.mp4"
                    ).status_code,
                    200,
                )
            self.client.delete(f"/api/jobs/{job_id}")

    def test_ip_scheduler_is_fifo_and_round_robin(self) -> None:
        with web_app.JOB_LOCK:
            while True:
                try:
                    web_app.JOB_DISPATCH_QUEUE.get_nowait()
                    web_app.JOB_DISPATCH_QUEUE.task_done()
                except Exception:
                    break
            web_app.JOB_QUEUES_BY_IP.clear()
            web_app.JOB_SCHEDULED_IPS.clear()

        try:
            web_app._enqueue_job("ip-a-job-1", "192.0.2.10")
            web_app._enqueue_job("ip-a-job-2", "192.0.2.10")
            web_app._enqueue_job("ip-b-job-1", "192.0.2.11")

            first_ip = web_app.JOB_DISPATCH_QUEUE.get_nowait()
            first_job = web_app._take_next_job(first_ip)
            web_app._reschedule_ip(first_ip)
            web_app.JOB_DISPATCH_QUEUE.task_done()

            second_ip = web_app.JOB_DISPATCH_QUEUE.get_nowait()
            second_job = web_app._take_next_job(second_ip)
            web_app._reschedule_ip(second_ip)
            web_app.JOB_DISPATCH_QUEUE.task_done()

            third_ip = web_app.JOB_DISPATCH_QUEUE.get_nowait()
            third_job = web_app._take_next_job(third_ip)
            web_app._reschedule_ip(third_ip)
            web_app.JOB_DISPATCH_QUEUE.task_done()

            self.assertEqual((first_ip, first_job), ("192.0.2.10", "ip-a-job-1"))
            self.assertEqual((second_ip, second_job), ("192.0.2.11", "ip-b-job-1"))
            self.assertEqual((third_ip, third_job), ("192.0.2.10", "ip-a-job-2"))
        finally:
            with web_app.JOB_LOCK:
                while True:
                    try:
                        web_app.JOB_DISPATCH_QUEUE.get_nowait()
                        web_app.JOB_DISPATCH_QUEUE.task_done()
                    except Exception:
                        break
                web_app.JOB_QUEUES_BY_IP.clear()
                web_app.JOB_SCHEDULED_IPS.clear()

    def test_running_job_can_be_interrupted(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.terminated = False

            def is_alive(self) -> bool:
                return not self.terminated

            def terminate(self) -> None:
                self.terminated = True

            def join(self, timeout: int | None = None) -> None:
                return None

            def kill(self) -> None:
                self.terminated = True

        video_path = Path("outputs/test_result.mp4")
        with patch("web_app._ensure_queue_worker"):
            with video_path.open("rb") as video:
                create_response = self.client.post(
                    "/api/jobs",
                    files={"result_video": ("result.mp4", video, "video/mp4")},
                    data={
                        "prompt_text": "保持镜头稳定",
                        "calculate_lpips": "false",
                        "max_frames": "2",
                        "device": "cpu",
                    },
                )
            self.assertEqual(create_response.status_code, 202)
            job_id = create_response.json()["job_id"]
            web_app._update_job_state(
                job_id,
                status="running",
                stage="models",
                progress=0.55,
                started_at="2026-07-27T12:00:00+08:00",
            )
            fake_process = FakeProcess()
            web_app.JOB_PROCESSES[job_id] = fake_process
            try:
                cancel_response = self.client.patch(
                    f"/api/jobs/{job_id}",
                    json={"action": "cancel"},
                )
                self.assertEqual(cancel_response.status_code, 200)
                self.assertEqual(cancel_response.json()["status"], "canceled")
                self.assertTrue(fake_process.terminated)
            finally:
                web_app.JOB_PROCESSES.pop(job_id, None)
                self.client.delete(f"/api/jobs/{job_id}")

    def test_job_worker_marks_missing_input_as_failed(self) -> None:
        job_id = f"worker-smoke-{uuid4().hex}"
        job_dir = Path("outputs/web_runs") / job_id
        created_at = "2026-07-27T12:00:00+08:00"
        job = {
            "job_id": job_id,
            "name": "missing.mp4",
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "created_at": created_at,
            "queued_at": created_at,
            "started_at": None,
            "finished_at": None,
            "updated_at": created_at,
            "error": None,
            "files": {"result_video": None},
            "original_files": {"result_video": "missing.mp4"},
            "parameters": {"device": "cpu", "max_frames": 2},
        }
        web_app._write_job(job)
        try:
            web_app._process_job(job_id)
            failed_job = web_app._read_job(job_id)
            self.assertIsNotNone(failed_job)
            self.assertEqual(failed_job["status"], "failed")
            self.assertIn("missing", failed_job["error"])
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)

    def test_readable_mp4_is_not_reencoded_when_queued(self) -> None:
        video_path = Path("outputs/test_result.mp4")
        with patch("web_app._ensure_queue_worker"), patch(
            "web_app.transcode_video_for_browser"
        ) as transcode:
            with video_path.open("rb") as video:
                response = self.client.post(
                    "/api/jobs",
                    files={"result_video": ("result.mp4", video, "video/mp4")},
                    data={
                        "calculate_lpips": "false",
                        "max_frames": "2",
                        "device": "cpu",
                    },
                )
            self.assertEqual(response.status_code, 202)
            transcode.assert_not_called()
            self.client.delete(f"/api/jobs/{response.json()['job_id']}")

    def test_json_safe_handles_numpy_and_paths(self) -> None:
        payload = _json_safe(
            {
                "path": Path("result.mp4"),
                "array": np.array([1, 2], dtype=np.int64),
                "score": float("inf"),
            }
        )
        self.assertEqual(payload["path"], "result.mp4")
        self.assertEqual(payload["array"], [1, 2])
        self.assertIsNone(payload["score"])


if __name__ == "__main__":
    unittest.main()
