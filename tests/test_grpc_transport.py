from __future__ import annotations

from concurrent import futures
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import grpc

import grpc_server
from grpc_api import frame_audit_pb2 as pb2
from grpc_api import frame_audit_pb2_grpc as pb2_grpc


class GrpcTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        pb2_grpc.add_FrameAuditServicer_to_server(
            grpc_server.FrameAuditService(),
            cls.server,
        )
        cls.port = cls.server.add_insecure_port("127.0.0.1:0")
        cls.server.start()
        cls.channel = grpc.insecure_channel(f"127.0.0.1:{cls.port}")
        cls.stub = pb2_grpc.FrameAuditStub(cls.channel)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.channel.close()
        cls.server.stop(0).wait()

    def test_health_returns_the_http_payload(self) -> None:
        response = self.stub.Health(pb2.Empty())
        self.assertEqual(response.http_status, 200)
        self.assertEqual(json.loads(response.json)["status"], "ok")

    def test_hardware_uses_the_existing_policy(self) -> None:
        response = self.stub.Hardware(pb2.HardwareRequest(device="cpu"))
        payload = json.loads(response.json)
        self.assertEqual(response.http_status, 200)
        self.assertEqual(payload["resolved_device"], "cpu")

    def test_list_jobs_preserves_the_http_response_shape(self) -> None:
        response = self.stub.ListJobs(pb2.ListJobsRequest(limit=1))
        payload = json.loads(response.json)
        self.assertEqual(response.http_status, 200)
        self.assertTrue(
            {"jobs", "active_job_id", "queued_count", "total_count"}
            <= set(payload)
        )

    def test_streamed_invalid_video_maps_to_invalid_argument(self) -> None:
        def requests():
            yield pb2.UploadRequest(
                options=pb2.JobOptions(
                    max_frames=2,
                    calculate_lpips=False,
                    device="cpu",
                )
            )
            yield pb2.UploadRequest(
                chunk=pb2.UploadChunk(
                    file_id="result-1",
                    field_name="result_video",
                    filename="result.txt",
                    content_type="text/plain",
                    data=b"not a video",
                    first=True,
                    last=True,
                )
            )

        with self.assertRaises(grpc.RpcError) as raised:
            self.stub.CreateJob(requests())
        self.assertEqual(
            raised.exception.code(),
            grpc.StatusCode.INVALID_ARGUMENT,
        )

    def test_streamed_valid_video_creates_the_same_job_shape(self) -> None:
        video_path = Path("outputs/test_result.mp4")

        def requests():
            yield pb2.UploadRequest(
                options=pb2.JobOptions(
                    max_frames=2,
                    calculate_lpips=False,
                    device="cpu",
                )
            )
            with video_path.open("rb") as source:
                data = source.read(1024 * 1024)
                first = True
                while data:
                    next_data = source.read(1024 * 1024)
                    yield pb2.UploadRequest(
                        chunk=pb2.UploadChunk(
                            file_id="result-1",
                            field_name="result_video",
                            filename="result.mp4",
                            content_type="video/mp4",
                            data=data,
                            first=first,
                            last=not next_data,
                        )
                    )
                    first = False
                    data = next_data

        job_id = None
        try:
            with patch("web_app._ensure_queue_worker"):
                response = self.stub.CreateJob(requests())
            payload = json.loads(response.json)
            job_id = payload["job_id"]
            self.assertEqual(response.http_status, 202)
            self.assertEqual(payload["status"], "queued")
            self.assertIn("result_video", payload["uploaded_files"])
            self.assertTrue(payload["parameters"]["wangxing_au_enabled"])
            self.assertEqual(
                payload["parameters"]["wangxing_expected_class"],
                "auto",
            )
        finally:
            if job_id:
                try:
                    self.stub.UpdateJob(
                        pb2.UpdateJobRequest(
                            job_id=job_id,
                            action="cancel",
                        )
                    )
                except grpc.RpcError:
                    # The short CPU job may finish before cleanup runs.
                    pass
                self.stub.DeleteJob(pb2.JobRequest(job_id=job_id))


if __name__ == "__main__":
    unittest.main()
