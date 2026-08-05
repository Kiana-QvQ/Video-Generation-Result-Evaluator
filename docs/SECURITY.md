# Security Deployment

The evaluator can upload videos, start GPU work, read reports, and delete
stored runs. Do not expose HTTP or gRPC directly to the public internet.

## Required Settings

Set an API key before any non-loopback bind:

```powershell
$env:FRAME_AUDIT_API_KEY = "replace-with-a-long-random-secret"
.\run.ps1 -Public -ApiKey $env:FRAME_AUDIT_API_KEY `
  -TlsCertfile "C:\certs\evaluator.crt" `
  -TlsKeyfile "C:\certs\evaluator.key"
```

The HTTP API accepts `Authorization: Bearer <key>` or `X-API-Key: <key>`.
gRPC clients must send the same value in `authorization` or `x-api-key`
metadata. Requests without the key receive `401` or
`UNAUTHENTICATED`.

Public HTTP and gRPC binds require TLS by default. The explicit
`-AllowInsecurePublic` switch is only for a controlled private network and
should not be used for internet-facing services.

## Resource Controls

Uploads are bounded by file count, total request bytes, duration, frame count,
resolution, and FFmpeg timeout. Configure them with:

```text
FRAME_AUDIT_MAX_TOTAL_UPLOAD_BYTES
FRAME_AUDIT_MAX_UPLOAD_FILES
FRAME_AUDIT_MAX_VIDEO_DURATION_SECONDS
FRAME_AUDIT_MAX_VIDEO_FRAME_COUNT
FRAME_AUDIT_MAX_VIDEO_PIXELS
FRAME_AUDIT_FFMPEG_TIMEOUT_SECONDS
FRAME_AUDIT_UPLOADS_PER_MINUTE
```

Completed, failed, and canceled runs are cleaned according to
`FRAME_AUDIT_RUN_RETENTION_SECONDS` and `FRAME_AUDIT_MAX_RUNS_BYTES`.

## Queue Model

The queue is intentionally single-owner. A filesystem lease and an atomic
per-job claim prevent duplicate execution when multiple Uvicorn/Gunicorn
workers share the output directory. Use an external queue/database if
horizontal scaling is required.

## Dependencies

The main evaluator, VBench, the local VLM, and LibreFace are not one
environment:

- Main evaluator: `requirements.txt`, Torch 2.5.
- VBench: Docker or `requirements/vbench.txt`, Transformers 4.33.2.
- Local VLM: `requirements/vlm_local.txt`, Transformers 4.46.3.
- LibreFace: `.\setup-libreface.ps1`, Torch 2.0.

Run `python scripts/audit_dataset.py --strict` before training. Missing clips
or AU CSV files must be repaired or deliberately excluded; they are not
silently treated as complete training data.

For large datasets, follow [`DATA_POLICY.md`](DATA_POLICY.md) and keep media
outside ordinary Git blobs.
