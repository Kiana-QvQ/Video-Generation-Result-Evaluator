from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from evaluator.media import concatenate_videos, find_ffmpeg
from evaluator.video_metrics import probe_video
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
        self.assertIn('name="name"', response.text)
        self.assertIn("wangxing-au-card", response.text)
        self.assertIn("wangxing_expected_class", response.text)
        self.assertIn("wangxing-result", response.text)
        self.assertIn('<option value="cuda" selected>', response.text)
        self.assertIn("KEY EVIDENCE", response.text)
        self.assertIn("关键证据", response.text)
        self.assertIn("加权总分", response.text)
        self.assertIn("五维评分雷达图", response.text)
        self.assertIn("已覆盖", response.text)
        self.assertIn("中断任务", response.text)
        self.assertIn("处理队列", response.text)
        self.assertIn("看清视频，", response.text)
        self.assertIn("上传视频后开始评分。", response.text)
        self.assertIn("参考视频 (可选，支持多段)", response.text)
        self.assertIn("多段参考视频", response.text)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_web_assets_are_not_cached_during_local_development(self) -> None:
        response = self.client.get("/assets/styles.css")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("cache-control"), "no-store")

    def test_gt_report_no_longer_mentions_crop_alignment(self) -> None:
        response = self.client.get("/assets/app.js")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("人脸保护裁剪", response.text)
        self.assertNotIn("居中裁剪对齐", response.text)
        self.assertNotIn("face_protected_crop_gt_to_result_aspect", response.text)

    def test_wangxing_result_copy_is_utf8_and_not_mojibake(self) -> None:
        response = self.client.get("/assets/app.js")
        self.assertEqual(response.status_code, 200)
        self.assertIn("缺少证据：", response.text)
        self.assertIn("证据覆盖 ", response.text)
        self.assertNotIn("缂哄皯璇佹嵁", response.text)
        self.assertNotIn("璇佹嵁瑕嗙洊", response.text)

    def test_wangxing_result_keeps_unavailable_state_and_forensics_visible(self) -> None:
        response = self.client.get("/assets/app.js")
        self.assertEqual(response.status_code, 200)
        self.assertIn('if (payload.status !== "available")', response.text)
        self.assertNotIn(
            'return;\n  if (payload.status !== "available")',
            response.text,
        )
        self.assertIn("raw_real_domain_evidence_0_1", response.text)
        self.assertIn("NOT CALIBRATED", response.text)

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
        self.assertIn("wangxing_au", payload)
        self.assertIn("ready", payload["wangxing_au"])
        self.assertIn("evaluator_version", payload["wangxing_au"])

    def test_wangxing_au_runner_passes_identity_images_without_driver_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            result_path = run_dir / "result.mp4"
            reference_image = run_dir / "reference_01.png"
            reference_video = run_dir / "reference_motion.mp4"
            result_path.write_bytes(b"result")
            reference_image.write_bytes(b"image")
            reference_video.write_bytes(b"driver")

            def fake_run(command: list[str], **_: object) -> None:
                output_path = Path(command[command.index("--output") + 1])
                output_path.write_text(
                    json.dumps({"status": "available"}),
                    encoding="utf-8",
                )

            with patch(
                "web_app._wangxing_au_status",
                return_value={"ready": True},
            ), patch("web_app.subprocess.run", side_effect=fake_run) as run:
                result = web_app._run_wangxing_au_assessment(
                    result_path=result_path,
                    reference_image_paths=[reference_image],
                    reference_video_path=reference_video,
                    expected_class="smile",
                    device="cpu",
                    run_dir=run_dir,
                )

            command = run.call_args.args[0]
            self.assertEqual(result["status"], "available")
            self.assertIn("--target-image", command)
            self.assertIn(str(reference_image), command)
            self.assertIn("--cache-root", command)
            self.assertNotIn("--driver-video", command)
            self.assertFalse(result["reference_action_used"])
            self.assertEqual(
                result["action_evidence_source"],
                "wangxing_training_profile_dynamic_statistics",
            )
            self.assertEqual(
                command[command.index("--expected-class") + 1],
                "smile",
            )
            self.assertEqual(run.call_args.kwargs["env"]["PYTHONIOENCODING"], "utf-8")
            self.assertEqual(run.call_args.kwargs["env"]["PYTHONUTF8"], "1")

    def test_wangxing_evidence_does_not_change_normal_expression_score(self) -> None:
        result = {
            "status": "partial",
            "categories": {
                "identity": {
                    "status": "available",
                    "metrics": {"score_0_1": 0.8},
                },
                "texture": {
                    "status": "available",
                    "metrics": {"score_0_1": 0.7},
                },
                "expression": {
                    "status": "partial",
                    "reference_source": "none",
                    "score_0_1": 0.7,
                    "backend": "prompt_clip",
                    "metrics": {
                        "generic_style_score_0_1": None,
                        "prompt_semantic_score_0_1": 0.7,
                        "prompt_semantic_backend": "clip",
                    },
                },
                "temporal": {
                    "status": "available",
                    "metrics": {"stability_score_0_1": 0.9},
                },
                "aesthetics": {
                    "status": "manual",
                    "metrics": {"manual_score_0_to_1": 0.8},
                },
            },
            "summary": [
                {},
                {},
                {
                    "类别": "3. 表情准确",
                    "权重": "15%",
                    "状态": "partial",
                    "标准化分数": "0.7000",
                    "核心结果": "prompt",
                    "后端": "prompt_clip",
                },
            ],
        }
        au = {
            "status": "available",
            "wangxing_targeted": {
                "status": "partial",
                "wangxing_expression_fit_score_0_1": 0.55,
                "score_weight_coverage": 0.4,
                "missing_evidence": [
                    "driver_expression",
                    "temporal_alignment",
                ],
                "evidence": {
                    "personal_au": 0.55,
                    "driver_expression": None,
                    "temporal_alignment": None,
                },
            },
        }
        result["category_scores"] = {"expression": 0.7}
        result["weighted_score_0_1"] = 0.82
        result["weighted_score_0_100"] = 82.0

        normal_score = result["categories"]["expression"]["score_0_1"]
        normal_status = result["categories"]["expression"]["status"]
        normal_weighted_score = result["weighted_score_0_1"]

        web_app._attach_wangxing_evidence(
            result,
            wangxing_au=au,
            prompt_text="人物微笑",
            driver_source=None,
        )

        expression = result["categories"]["expression"]
        self.assertEqual(expression["status"], normal_status)
        self.assertEqual(expression["score_0_1"], normal_score)
        self.assertNotIn("wangxing_au_score_0_1", expression["metrics"])
        self.assertEqual(
            result["weighted_score_0_1"],
            normal_weighted_score,
        )
        self.assertEqual(result["weighted_score_0_100"], 82.0)
        self.assertEqual(
            result["expression_evidence"]["wangxing_au"]["score_0_1"],
            0.55,
        )
        self.assertEqual(
            result["expression_evidence"]["scope"],
            "separate_targeted_specialization",
        )
        self.assertTrue(
            result["expression_evidence"]["normal_expression_unchanged"]
        )

    def test_multiple_reference_videos_are_joined_for_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            segment_paths = [
                run_dir / "reference_motion_01.mp4",
                run_dir / "reference_motion_02.mp4",
            ]
            merged_path = run_dir / "reference_motion.mp4"
            uploads = [
                SimpleNamespace(filename="segment-a.mp4"),
                SimpleNamespace(filename="segment-b.webm"),
            ]
            with patch(
                "web_app._save_upload",
                side_effect=segment_paths,
            ) as save_upload, patch(
                "web_app.concatenate_videos",
                return_value=merged_path,
            ) as concatenate:
                result = web_app._save_reference_videos(uploads, run_dir)

            self.assertEqual(result, merged_path)
            self.assertEqual(save_upload.call_count, 2)
            concatenate.assert_called_once_with(segment_paths, merged_path)

    def test_multiple_reference_videos_keep_all_frames_when_sizes_differ(self) -> None:
        source = Path("outputs/test_result.mp4")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resized = root / "resized.mp4"
            merged = root / "merged.mp4"
            subprocess.run(
                [
                    find_ffmpeg(),
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    "-vf",
                    "scale=80:60",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(resized),
                ],
                check=True,
            )
            concatenate_videos([source, resized], merged)

            merged_info = probe_video(merged)
            self.assertEqual(merged_info.frame_count, 16)
            self.assertEqual(merged_info.width, 64)
            self.assertEqual(merged_info.height, 48)

    def test_evaluate_accepts_multiple_reference_video_parts(self) -> None:
        captured: dict[str, list[str]] = {}
        base_result = {
            "summary": [],
            "frame_records": [],
            "categories": {},
            "weighted_score_0_100": 0.0,
            "coverage": "0/5",
        }

        def capture_reference_videos(uploads, _run_dir, **_kwargs):
            captured["names"] = [upload.filename for upload in uploads]
            return None

        with patch(
            "web_app._save_reference_videos",
            side_effect=capture_reference_videos,
        ), patch("web_app.evaluate_all", return_value=base_result):
            with Path("outputs/test_result.mp4").open("rb") as result_file:
                response = self.client.post(
                    "/api/evaluate",
                    files=[
                        ("result_video", ("result.mp4", result_file, "video/mp4")),
                        ("reference_video", ("a.mp4", b"a", "video/mp4")),
                        ("reference_video", ("b.mp4", b"b", "video/mp4")),
                    ],
                    data={
                        "max_frames": "2",
                        "device": "cpu",
                        "calculate_lpips": "false",
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["names"], ["a.mp4", "b.mp4"])

    def test_sync_evaluation_can_disable_wangxing_au(self) -> None:
        video_path = Path("outputs/test_result.mp4")
        base_result = {
            "summary": [],
            "frame_records": [],
            "categories": {},
            "weighted_score_0_100": 0.0,
            "coverage": "0/5",
        }
        with patch("web_app.evaluate_all", return_value=base_result), patch(
            "web_app._run_wangxing_au_assessment"
        ) as runner:
            with video_path.open("rb") as video:
                response = self.client.post(
                    "/api/evaluate",
                    files={"result_video": ("result.mp4", video, "video/mp4")},
                    data={
                        "calculate_lpips": "false",
                        "max_frames": "2",
                        "device": "cpu",
                        "wangxing_au_enabled": "false",
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["result"]["wangxing_au"]["status"],
            "not_applicable",
        )
        runner.assert_not_called()

    def test_wangxing_au_runner_explains_face_detection_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            result_path = run_dir / "result.mp4"
            result_path.write_bytes(b"result")
            failure = subprocess.CalledProcessError(
                2,
                ["evaluate_generated_video"],
                stderr="No face detected in the provided video.",
            )
            with patch(
                "web_app._wangxing_au_status",
                return_value={"ready": True},
            ), patch(
                "web_app.subprocess.run",
                side_effect=failure,
            ):
                result = web_app._run_wangxing_au_assessment(
                    result_path=result_path,
                    reference_image_paths=[],
                    reference_video_path=None,
                    expected_class=None,
                    device="cpu",
                    run_dir=run_dir,
                )

        self.assertEqual(result["status"], "unavailable")
        self.assertIn("未检测到可用人脸关键点", result["reason"])

    def test_wangxing_au_runner_does_not_mislabel_encoding_failure_as_face_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            result_path = run_dir / "result.mp4"
            result_path.write_bytes(b"result")
            failure = subprocess.CalledProcessError(
                2,
                ["evaluate_generated_video"],
                stderr=(
                    "ERROR: 'gbk' codec can't encode character "
                    "'\\ufffd' in position 835: illegal multibyte sequence"
                ),
            )
            with patch(
                "web_app._wangxing_au_status",
                return_value={"ready": True},
            ), patch(
                "web_app.subprocess.run",
                side_effect=failure,
            ):
                result = web_app._run_wangxing_au_assessment(
                    result_path=result_path,
                    reference_image_paths=[],
                    reference_video_path=None,
                    expected_class=None,
                    device="cpu",
                    run_dir=run_dir,
                )

        self.assertEqual(result["status"], "unavailable")
        self.assertIn("编码失败", result["reason"])
        self.assertNotIn("未检测到可用人脸关键点", result["reason"])

    def test_forensics_failure_is_optional_and_does_not_abort_specialization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.mp4"
            au_path = root / "generated.csv"
            result_path.write_bytes(b"result")
            au_path.write_text("frame_idx\n0\n", encoding="utf-8")
            with patch(
                "web_app.FORENSICS_PROFILE_PATH",
                root / "forensics_profiles.json",
            ), patch(
                "web_app.analyze_forensics",
                side_effect=TypeError("malformed forensic profile"),
            ):
                (root / "forensics_profiles.json").write_text(
                    json.dumps({"facial_motion": {}, "texture_detail": {}}),
                    encoding="utf-8",
                )
                result = web_app._run_forensics_assessment(
                    result_path=result_path,
                    au_path=au_path,
                )

        self.assertEqual(result["status"], "unavailable")
        self.assertIn("malformed forensic profile", result["reason"])

    def test_invalid_wangxing_expression_class_is_rejected_before_queueing(
        self,
    ) -> None:
        video_path = Path("outputs/test_result.mp4")
        with patch("web_app._ensure_queue_worker"):
            with video_path.open("rb") as video:
                response = self.client.post(
                    "/api/jobs",
                    files={"result_video": ("result.mp4", video, "video/mp4")},
                    data={
                        "calculate_lpips": "false",
                        "max_frames": "2",
                        "device": "cpu",
                        "wangxing_expected_class": "not_an_expression",
                    },
                )

        self.assertEqual(response.status_code, 422)
        self.assertIn("wangxing_expected_class", response.json()["detail"])

    def test_run_file_path_is_sandboxed(self) -> None:
        response = self.client.get("/api/runs/../requirements.txt")
        self.assertIn(response.status_code, {404, 422})

    def test_incomplete_job_cannot_download_stale_report_files(self) -> None:
        job_id = f"report-guard-{uuid4().hex}"
        job_dir = Path("outputs/web_runs") / job_id
        job = {
            "job_id": job_id,
            "client_ip": "127.0.0.1",
            "name": "report.mp4",
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "created_at": "2026-07-31T12:00:00+08:00",
            "queued_at": "2026-07-31T12:00:00+08:00",
            "started_at": None,
            "finished_at": None,
            "updated_at": "2026-07-31T12:00:00+08:00",
            "error": None,
            "files": {},
            "original_files": {},
            "parameters": {},
        }
        web_app._write_job(job)
        (job_dir / "result.json").write_text("stale", encoding="utf-8")
        try:
            response = self.client.get(f"/api/runs/{job_id}/result.json")
            self.assertEqual(response.status_code, 404)
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)

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
        with patch("web_app._ensure_queue_worker"), patch(
            "web_app._enqueue_job"
        ):
            with video_path.open("rb") as video:
                create_response = self.client.post(
                    "/api/jobs",
                    files={"result_video": ("result.mp4", video, "video/mp4")},
                    data={
                        "name": "consistency-v3",
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
            self.assertEqual(job["name"], "consistency-v3")
            self.assertFalse(job["parameters"]["wangxing_au_enabled"])
            self.assertEqual(
                job["parameters"]["wangxing_expected_class"],
                "auto",
            )
            self.assertTrue((job_dir / "status.json").is_file())
            self.assertTrue((job_dir / "params.json").is_file())

            list_response = self.client.get("/api/jobs")
            self.assertEqual(list_response.status_code, 200)
            self.assertTrue(
                any(item["job_id"] == job_id for item in list_response.json()["jobs"])
            )

            detail_response = self.client.get(f"/api/jobs/{job_id}")
            self.assertEqual(detail_response.status_code, 200)
            self.assertIn(
                detail_response.json()["status"],
                {"queued", "running"},
            )

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
                json={
                    "action": "retry",
                    "name": "consistency-v4",
                    "prompt_text": "保持镜头稳定并自然微笑",
                    "max_frames": 12,
                    "calculate_lpips": True,
                    "device": "auto",
                    "manual_expression_score": 4,
                    "manual_aesthetic_score": 5,
                    "wangxing_au_enabled": True,
                    "wangxing_expected_class": "smile",
                },
            )
            self.assertEqual(retry_response.status_code, 200)
            self.assertEqual(retry_response.json()["status"], "queued")
            self.assertEqual(retry_response.json()["name"], "consistency-v4")
            self.assertEqual(
                retry_response.json()["parameters"]["prompt_text"],
                "保持镜头稳定并自然微笑",
            )
            self.assertEqual(retry_response.json()["parameters"]["max_frames"], 12)
            self.assertEqual(
                retry_response.json()["parameters"]["wangxing_expected_class"],
                "smile",
            )

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

    def test_completed_job_can_be_retried_with_edited_parameters(self) -> None:
        video_path = Path("outputs/test_result.mp4")
        with patch("web_app._ensure_queue_worker"), patch(
            "web_app._enqueue_job"
        ):
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
            job_id = response.json()["job_id"]
            job_dir = Path("outputs/web_runs") / job_id
            try:
                web_app._update_job_state(
                    job_id,
                    status="completed",
                    stage="completed",
                    progress=1.0,
                )
                (job_dir / "result.json").write_text("old report", encoding="utf-8")
                (job_dir / "summary.csv").write_text("old summary", encoding="utf-8")
                retry = self.client.patch(
                    f"/api/jobs/{job_id}",
                    json={
                        "action": "retry",
                        "name": "",
                        "prompt_text": "重新检查镜头稳定性",
                        "max_frames": 16,
                    },
                )
                self.assertEqual(retry.status_code, 200)
                self.assertEqual(retry.json()["status"], "queued")
                self.assertEqual(retry.json()["name"], "result.mp4")
                self.assertEqual(
                    retry.json()["parameters"]["prompt_text"],
                    "重新检查镜头稳定性",
                )
                self.assertEqual(retry.json()["parameters"]["max_frames"], 16)
                params = json.loads(
                    (job_dir / "params.json").read_text(encoding="utf-8")
                )
                self.assertEqual(params["max_frames"], 16)
                self.assertFalse((job_dir / "result.json").exists())
                self.assertFalse((job_dir / "summary.csv").exists())
            finally:
                shutil.rmtree(job_dir, ignore_errors=True)

    def test_replacing_result_video_reuses_optional_inputs(self) -> None:
        video_path = Path("outputs/test_result.mp4")
        with patch("web_app._ensure_queue_worker"), patch(
            "web_app._enqueue_job"
        ):
            with (
                video_path.open("rb") as result_video,
                video_path.open("rb") as gt_video,
            ):
                source_response = self.client.post(
                    "/api/jobs",
                    files=[
                        ("result_video", ("source-result.mp4", result_video, "video/mp4")),
                        ("gt_video", ("source-gt.mp4", gt_video, "video/mp4")),
                        (
                            "reference_images",
                            (
                                "source-face.png",
                                (Path("tests/data/front.png").read_bytes()),
                                "image/png",
                            ),
                        ),
                    ],
                    data={
                        "name": "source-evaluation",
                        "calculate_lpips": "false",
                        "max_frames": "2",
                        "device": "cpu",
                    },
                )
            self.assertEqual(source_response.status_code, 202)
            source_job_id = source_response.json()["job_id"]
            source_dir = Path("outputs/web_runs") / source_job_id
            web_app._update_job_state(
                source_job_id,
                status="completed",
                stage="completed",
                progress=1.0,
            )

            with video_path.open("rb") as replacement_video:
                replacement_response = self.client.post(
                    "/api/jobs",
                    files={
                        "result_video": (
                            "replacement-result.mp4",
                            replacement_video,
                            "video/mp4",
                        )
                    },
                    data={
                        "reuse_job_id": source_job_id,
                        "calculate_lpips": "false",
                        "max_frames": "2",
                        "device": "cpu",
                    },
                )
            self.assertEqual(replacement_response.status_code, 202)
            replacement_job = replacement_response.json()
            replacement_dir = Path("outputs/web_runs") / replacement_job["job_id"]
            try:
                self.assertEqual(
                    replacement_job["original_files"]["gt_video"],
                    "source-gt.mp4",
                )
                self.assertEqual(
                    replacement_job["original_files"]["reference_images"],
                    ["source-face.png"],
                )
                self.assertEqual(
                    (replacement_dir / "gt.mp4").read_bytes(),
                    (source_dir / "gt.mp4").read_bytes(),
                )
                self.assertEqual(
                    (replacement_dir / "reference_01.png").read_bytes(),
                    Path("tests/data/front.png").read_bytes(),
                )
            finally:
                shutil.rmtree(source_dir, ignore_errors=True)
                shutil.rmtree(replacement_dir, ignore_errors=True)

    def test_deleting_job_removes_its_stale_dispatch_entry(self) -> None:
        job_id = f"queued-delete-{uuid4().hex}"
        client_ip = "testclient"
        job_dir = Path("outputs/web_runs") / job_id
        job = {
            "job_id": job_id,
            "client_ip": client_ip,
            "name": "queued-delete.mp4",
            "status": "completed",
            "stage": "completed",
            "progress": 1.0,
            "created_at": "2026-07-31T12:00:00+08:00",
            "queued_at": "2026-07-31T12:00:00+08:00",
            "started_at": None,
            "finished_at": "2026-07-31T12:01:00+08:00",
            "updated_at": "2026-07-31T12:01:00+08:00",
            "error": None,
            "files": {},
            "original_files": {},
            "parameters": {},
        }
        web_app._write_job(job)
        try:
            with web_app.JOB_LOCK:
                web_app.JOB_QUEUES_BY_IP[client_ip] = web_app.deque([job_id])
                web_app.JOB_SCHEDULED_IPS.add(client_ip)
            delete_response = self.client.delete(f"/api/jobs/{job_id}")
            self.assertEqual(delete_response.status_code, 200)
            with web_app.JOB_LOCK:
                self.assertNotIn(
                    job_id,
                    web_app.JOB_QUEUES_BY_IP.get(client_ip, ()),
                )
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
            with web_app.JOB_LOCK:
                web_app.JOB_QUEUES_BY_IP.pop(client_ip, None)
                web_app.JOB_SCHEDULED_IPS.discard(client_ip)

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

    def test_scheduler_estimates_reference_and_au_jobs_as_heavier(self) -> None:
        light = {
            "parameters": {
                "max_frames": 8,
                "calculate_lpips": False,
                "wangxing_au_enabled": False,
            },
            "files": {},
        }
        heavy = {
            "parameters": {
                "max_frames": 64,
                "calculate_lpips": True,
                "wangxing_au_enabled": True,
                "prompt_text": "compare the expression",
            },
            "files": {
                "gt_video": "gt.mp4",
                "reference_video": "reference.mp4",
                "reference_images": ["reference_01.png", "reference_02.png"],
            },
        }

        self.assertLess(
            web_app._estimate_job_seconds(light),
            web_app._estimate_job_seconds(heavy),
        )

    def test_single_persisted_queued_job_is_restored_for_dispatch(self) -> None:
        job_id = f"queued-recovery-{uuid4().hex}"
        job_dir = Path("outputs/web_runs") / job_id
        created_at = "2026-07-27T12:00:00+08:00"
        job = {
            "job_id": job_id,
            "client_ip": "192.0.2.10",
            "name": "queued.mp4",
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
            "original_files": {"result_video": "queued.mp4"},
            "parameters": {"device": "cpu", "max_frames": 2},
        }
        web_app._write_job(job)
        try:
            with web_app.JOB_LOCK:
                web_app._clear_dispatch_state()
                web_app._restore_queued_jobs()

            client_ip = web_app.JOB_DISPATCH_QUEUE.get_nowait()
            restored_job_id = web_app._take_next_job(client_ip)
            web_app._reschedule_ip(client_ip)
            web_app.JOB_DISPATCH_QUEUE.task_done()
            self.assertEqual(client_ip, "192.0.2.10")
            self.assertEqual(restored_job_id, job_id)
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
            with web_app.JOB_LOCK:
                web_app._clear_dispatch_state()

    def test_running_job_can_be_interrupted(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.terminated = False
                self.job_id = None
                self.cancel_state = None

            def is_alive(self) -> bool:
                return not self.terminated

            def terminate(self) -> None:
                self.terminated = True
                if self.job_id:
                    self.cancel_state = web_app._read_job(self.job_id)

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
            fake_process.job_id = job_id
            web_app.JOB_PROCESSES[job_id] = fake_process
            try:
                cancel_response = self.client.patch(
                    f"/api/jobs/{job_id}",
                    json={"action": "cancel"},
                )
                self.assertEqual(cancel_response.status_code, 200)
                self.assertEqual(cancel_response.json()["status"], "canceled")
                self.assertTrue(fake_process.terminated)
                self.assertEqual(fake_process.cancel_state["status"], "canceling")
                self.assertTrue(fake_process.cancel_state["cancel_requested"])
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
        self.assertEqual(payload["score"], "inf")

    def test_atomic_status_write_retries_windows_permission_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "status.json"
            original_replace = Path.replace
            attempts = 0

            def replace_with_transient_lock(
                source: Path,
                destination: Path,
            ) -> Path:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise PermissionError(5, "access denied")
                return original_replace(source, destination)

            with patch.object(
                Path,
                "replace",
                new=replace_with_transient_lock,
            ):
                web_app._atomic_write_json(target, {"status": "ok"})

            self.assertEqual(attempts, 2)
            self.assertEqual(
                target.read_text(encoding="utf-8"),
                '{\n  "status": "ok"\n}',
            )

    def test_stale_running_job_without_process_is_detected(self) -> None:
        job = {
            "job_id": f"stale-{uuid4().hex}",
            "status": "running",
            "updated_at": "2020-01-01T00:00:00+08:00",
        }
        self.assertTrue(web_app._is_stale_running_job(job))


if __name__ == "__main__":
    unittest.main()
