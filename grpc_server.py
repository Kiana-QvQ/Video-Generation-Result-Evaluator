from __future__ import annotations

import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from typing import BinaryIO, Iterable
from concurrent import futures

import grpc
from fastapi import HTTPException, UploadFile

import web_app
from grpc_api import frame_audit_pb2 as pb2
from grpc_api import frame_audit_pb2_grpc as pb2_grpc


CHUNK_SIZE = 1024 * 1024
_ALLOWED_FIELDS = {
    "result_video": web_app.VIDEO_SUFFIXES,
    "gt_video": web_app.VIDEO_SUFFIXES,
    "reference_images": web_app.IMAGE_SUFFIXES,
    "reference_video": web_app.VIDEO_SUFFIXES,
}


class _GrpcRequestError(Exception):
    def __init__(self, code: grpc.StatusCode, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass
class _StagedUpload:
    file_id: str
    field_name: str
    filename: str
    content_type: str
    path: Path
    handle: BinaryIO
    size: int = 0
    complete: bool = False


def _peer_ip(context: grpc.ServicerContext) -> str:
    peer = context.peer()
    if peer.startswith("ipv4:"):
        return peer[5:].rsplit(":", 1)[0]
    if peer.startswith("ipv6:"):
        return peer[5:].rsplit(":", 1)[0].strip("[]")
    match = re.search(r"(?P<host>\d+\.\d+\.\d+\.\d+)", peer)
    return match.group("host") if match else "grpc"


def _request_for(context: grpc.ServicerContext) -> SimpleNamespace:
    return SimpleNamespace(
        client=SimpleNamespace(host=_peer_ip(context)),
        headers={},
    )


def _json_response(value: object, status_code: int = 200) -> pb2.JsonResponse:
    if isinstance(value, web_app.JSONResponse):
        payload = json.loads(value.body.decode("utf-8"))
        status_code = value.status_code
    else:
        payload = web_app._json_safe(value)
    return pb2.JsonResponse(
        json=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        http_status=status_code,
    )


def _status_code(status_code: int) -> grpc.StatusCode:
    return {
        400: grpc.StatusCode.INVALID_ARGUMENT,
        404: grpc.StatusCode.NOT_FOUND,
        409: grpc.StatusCode.ABORTED,
        413: grpc.StatusCode.RESOURCE_EXHAUSTED,
        415: grpc.StatusCode.INVALID_ARGUMENT,
        422: grpc.StatusCode.INVALID_ARGUMENT,
    }.get(status_code, grpc.StatusCode.INTERNAL)


def _abort(context: grpc.ServicerContext, exc: Exception) -> None:
    if isinstance(exc, grpc.RpcError):
        raise exc
    if isinstance(exc, _GrpcRequestError):
        context.abort(exc.code, exc.detail)
    if isinstance(exc, HTTPException):
        context.abort(_status_code(exc.status_code), str(exc.detail))
    if isinstance(exc, ValueError):
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
    context.abort(grpc.StatusCode.INTERNAL, f"{type(exc).__name__}: {exc}")


def _options(request: pb2.JobOptions | None) -> dict[str, object]:
    if request is None:
        return {
            "prompt_text": "",
            "max_frames": 64,
            "calculate_lpips": True,
            "device": "auto",
            "manual_expression_score": "",
            "manual_aesthetic_score": "",
            "wangxing_au_enabled": True,
            "wangxing_expected_class": "auto",
        }
    return {
        "prompt_text": request.prompt_text,
        "max_frames": request.max_frames if request.HasField("max_frames") else 64,
        "calculate_lpips": (
            request.calculate_lpips
            if request.HasField("calculate_lpips")
            else True
        ),
        "device": request.device or "auto",
        "manual_expression_score": request.manual_expression_score,
        "manual_aesthetic_score": request.manual_aesthetic_score,
        "wangxing_au_enabled": True,
        "wangxing_expected_class": "auto",
    }


def _collect_uploads(
    requests: Iterable[pb2.UploadRequest],
    temp_root: Path,
    context: grpc.ServicerContext,
) -> tuple[pb2.JobOptions | None, list[_StagedUpload]]:
    options: pb2.JobOptions | None = None
    uploads: list[_StagedUpload] = []
    active: dict[str, _StagedUpload] = {}

    for request in requests:
        if request.HasField("options"):
            if options is not None:
                raise _GrpcRequestError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Upload options may only be sent once.",
                )
            options = request.options
            continue
        if not request.HasField("chunk"):
            raise _GrpcRequestError(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Each upload request must contain options or a file chunk.",
            )

        chunk = request.chunk
        if not chunk.file_id:
            raise _GrpcRequestError(
                grpc.StatusCode.INVALID_ARGUMENT,
                "file_id is required.",
            )
        if chunk.field_name not in _ALLOWED_FIELDS:
            raise _GrpcRequestError(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Unsupported upload field: {chunk.field_name}",
            )
        if chunk.first:
            if chunk.file_id in active or any(
                upload.file_id == chunk.file_id for upload in uploads
            ):
                raise _GrpcRequestError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Duplicate file_id: {chunk.file_id}",
                )
            filename = Path(chunk.filename or "").name
            suffix = Path(filename).suffix.lower()
            if not filename or suffix not in _ALLOWED_FIELDS[chunk.field_name]:
                raise _GrpcRequestError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Unsupported filename for {chunk.field_name}: {chunk.filename}",
                )
            path = temp_root / f"{len(uploads):04d}{suffix}"
            staged = _StagedUpload(
                file_id=chunk.file_id,
                field_name=chunk.field_name,
                filename=filename,
                content_type=chunk.content_type or "application/octet-stream",
                path=path,
                handle=path.open("wb"),
            )
            active[chunk.file_id] = staged
            uploads.append(staged)
        else:
            staged = active.get(chunk.file_id)
            if staged is None:
                raise _GrpcRequestError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Unknown file_id: {chunk.file_id}",
                )

        staged = active[chunk.file_id]
        staged.handle.write(chunk.data)
        staged.size += len(chunk.data)
        if staged.size > web_app.MAX_UPLOAD_BYTES:
            raise _GrpcRequestError(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "A single upload cannot exceed 1.5 GB.",
            )
        if chunk.last:
            staged.handle.close()
            staged.complete = True
            active.pop(chunk.file_id, None)

    for staged in active.values():
        staged.handle.close()
        raise _GrpcRequestError(
            grpc.StatusCode.INVALID_ARGUMENT,
            f"Upload did not finish: {staged.file_id}",
        )
    if any(not upload.complete for upload in uploads):
        raise _GrpcRequestError(
            grpc.StatusCode.INVALID_ARGUMENT,
            "Every upload must end with last=true.",
        )
    return options, uploads


def _upload_objects(
    uploads: list[_StagedUpload],
) -> tuple[UploadFile, UploadFile | None, list[UploadFile], UploadFile | None]:
    def open_upload(upload: _StagedUpload) -> UploadFile:
        return UploadFile(
            file=upload.path.open("rb"),
            size=upload.size,
            filename=upload.filename,
        )

    by_field = {
        field: [upload for upload in uploads if upload.field_name == field]
        for field in _ALLOWED_FIELDS
    }
    result = by_field["result_video"]
    if len(result) != 1:
        raise HTTPException(
            status_code=422,
            detail="Exactly one result_video upload is required.",
        )
    gt = by_field["gt_video"]
    motion = by_field["reference_video"]
    return (
        open_upload(result[0]),
        open_upload(gt[0]) if gt else None,
        [open_upload(upload) for upload in by_field["reference_images"]],
        open_upload(motion[0]) if motion else None,
    )


def _close_uploads(
    result_video: UploadFile,
    gt_video: UploadFile | None,
    reference_images: list[UploadFile],
    reference_video: UploadFile | None,
) -> None:
    uploads = [result_video, gt_video, reference_video, *reference_images]
    for upload in uploads:
        if upload is not None:
            upload.file.close()


class FrameAuditService(pb2_grpc.FrameAuditServicer):
    def Health(self, request: pb2.Empty, context: grpc.ServicerContext) -> pb2.JsonResponse:
        try:
            return _json_response(web_app.health())
        except Exception as exc:
            _abort(context, exc)
        raise AssertionError("unreachable")

    def Models(self, request: pb2.Empty, context: grpc.ServicerContext) -> pb2.JsonResponse:
        try:
            return _json_response(web_app.models())
        except Exception as exc:
            _abort(context, exc)
        raise AssertionError("unreachable")

    def Hardware(
        self,
        request: pb2.HardwareRequest,
        context: grpc.ServicerContext,
    ) -> pb2.JsonResponse:
        try:
            return _json_response(web_app.hardware(request.device or "auto"))
        except Exception as exc:
            _abort(context, exc)
        raise AssertionError("unreachable")

    def CreateJob(
        self,
        requests: Iterable[pb2.UploadRequest],
        context: grpc.ServicerContext,
    ) -> pb2.JsonResponse:
        try:
            with tempfile.TemporaryDirectory(prefix="frame-audit-grpc-") as directory:
                options_message, staged = _collect_uploads(
                    requests,
                    Path(directory),
                    context,
                )
                options = _options(options_message)
                result_video, gt_video, reference_images, reference_video = (
                    _upload_objects(staged)
                )
                try:
                    response = web_app.create_job(
                        _request_for(context),
                        result_video=result_video,
                        gt_video=gt_video,
                        reference_images=reference_images,
                        reference_video=reference_video,
                        prompt_text=str(options["prompt_text"]),
                        max_frames=int(options["max_frames"]),
                        calculate_lpips=bool(options["calculate_lpips"]),
                        device=str(options["device"]),
                        manual_expression_score=str(
                            options["manual_expression_score"]
                        ),
                        manual_aesthetic_score=str(
                            options["manual_aesthetic_score"]
                        ),
                        wangxing_au_enabled=bool(
                            options["wangxing_au_enabled"]
                        ),
                        wangxing_expected_class=str(
                            options["wangxing_expected_class"]
                        ),
                    )
                    return _json_response(response)
                finally:
                    _close_uploads(
                        result_video,
                        gt_video,
                        reference_images,
                        reference_video,
                    )
        except Exception as exc:
            _abort(context, exc)
        raise AssertionError("unreachable")

    def Evaluate(
        self,
        requests: Iterable[pb2.UploadRequest],
        context: grpc.ServicerContext,
    ) -> pb2.JsonResponse:
        try:
            with tempfile.TemporaryDirectory(prefix="frame-audit-grpc-") as directory:
                options_message, staged = _collect_uploads(
                    requests,
                    Path(directory),
                    context,
                )
                options = _options(options_message)
                result_video, gt_video, reference_images, reference_video = (
                    _upload_objects(staged)
                )
                try:
                    response = web_app.evaluate(
                        _request_for(context),
                        result_video=result_video,
                        gt_video=gt_video,
                        reference_images=reference_images,
                        reference_video=reference_video,
                        prompt_text=str(options["prompt_text"]),
                        max_frames=int(options["max_frames"]),
                        calculate_lpips=bool(options["calculate_lpips"]),
                        device=str(options["device"]),
                        manual_expression_score=str(
                            options["manual_expression_score"]
                        ),
                        manual_aesthetic_score=str(
                            options["manual_aesthetic_score"]
                        ),
                        wangxing_au_enabled=bool(
                            options["wangxing_au_enabled"]
                        ),
                        wangxing_expected_class=str(
                            options["wangxing_expected_class"]
                        ),
                    )
                    return _json_response(response)
                finally:
                    _close_uploads(
                        result_video,
                        gt_video,
                        reference_images,
                        reference_video,
                    )
        except Exception as exc:
            _abort(context, exc)
        raise AssertionError("unreachable")

    def ListJobs(
        self,
        request: pb2.ListJobsRequest,
        context: grpc.ServicerContext,
    ) -> pb2.JsonResponse:
        try:
            limit = request.limit or 20
            return _json_response(
                web_app.list_jobs(_request_for(context), limit=limit)
            )
        except Exception as exc:
            _abort(context, exc)
        raise AssertionError("unreachable")

    def GetJob(
        self,
        request: pb2.JobRequest,
        context: grpc.ServicerContext,
    ) -> pb2.JsonResponse:
        try:
            return _json_response(
                web_app.get_job(request.job_id, _request_for(context))
            )
        except Exception as exc:
            _abort(context, exc)
        raise AssertionError("unreachable")

    def UpdateJob(
        self,
        request: pb2.UpdateJobRequest,
        context: grpc.ServicerContext,
    ) -> pb2.JsonResponse:
        try:
            name = request.name if request.HasField("name") else None
            update = web_app.JobUpdate(
                name=name,
                action=request.action or None,
            )
            return _json_response(
                web_app.update_job(
                    request.job_id,
                    _request_for(context),
                    update,
                )
            )
        except Exception as exc:
            _abort(context, exc)
        raise AssertionError("unreachable")

    def DeleteJob(
        self,
        request: pb2.JobRequest,
        context: grpc.ServicerContext,
    ) -> pb2.JsonResponse:
        try:
            return _json_response(
                web_app.delete_job(request.job_id, _request_for(context))
            )
        except Exception as exc:
            _abort(context, exc)
        raise AssertionError("unreachable")

    def DownloadRunFile(
        self,
        request: pb2.DownloadRequest,
        context: grpc.ServicerContext,
    ) -> Iterable[pb2.DownloadChunk]:
        try:
            response = web_app.download_run_file(
                request.run_id,
                request.filename,
                _request_for(context),
            )
            path = Path(response.path)
            if not path.is_file():
                raise _GrpcRequestError(
                    grpc.StatusCode.NOT_FOUND,
                    "File not found",
                )
            content_type = response.media_type or mimetypes.guess_type(
                path.name
            )[0] or "application/octet-stream"
            with path.open("rb") as source:
                while True:
                    data = source.read(CHUNK_SIZE)
                    if not data:
                        break
                    yield pb2.DownloadChunk(
                        filename=path.name,
                        content_type=content_type,
                        data=data,
                        eof=False,
                    )
            yield pb2.DownloadChunk(
                filename=path.name,
                content_type=content_type,
                eof=True,
            )
        except Exception as exc:
            _abort(context, exc)


def serve() -> None:
    host = os.environ.get("EVALUATOR_GRPC_HOST", "127.0.0.1")
    port = int(os.environ.get("EVALUATOR_GRPC_PORT", "50051"))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb2_grpc.add_FrameAuditServicer_to_server(FrameAuditService(), server)
    bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"Unable to bind gRPC endpoint {host}:{port}")
    server.start()
    print(f"Frame Audit gRPC listening on {host}:{bound_port}", flush=True)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5).wait()


if __name__ == "__main__":
    serve()
