from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

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
