# Public Showcase Queue

The management showcase is a read-only queue separate from the per-client
evaluation queue.

## Build Or Refresh

Run from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\web_forensics\build_public_showcase.py
```

The index is written to:

```text
outputs/public_showcase/index.json
```

The builder imports:

- Current completed jobs under `outputs/web_runs`
- Archived completed jobs under `outputs/历史归档/缓存/web_runs`
- Web forensic reports under `outputs/forensics/**/all_results.json`
- Selected Wang Xing model metric reports

The original personal queue remains private and is not merged into the
public scheduler.

`start.py` refreshes this index automatically on startup unless
`--skip-public-showcase-refresh` is supplied.

## HTTP

Open the management page:

```text
/showcase
```

To start the main evaluation website and the separate human-review website
from one command:

```powershell
.\.venv\Scripts\python.exe start.py `
  --transport http `
  --http-host 127.0.0.1 `
  --http-port 7860 `
  --with-human-review `
  --human-review-host 127.0.0.1 `
  --human-review-port 5001
```

The public showcase is on the main site at `/showcase`; the human-review
website is on port `5001`.

Read-only endpoints:

```text
GET /api/public-showcase?limit=1000
GET /api/public-showcase/{item_id}
GET /api/public-showcase/{item_id}/files/{file_key}
```

## gRPC

The protobuf service adds:

```text
ListPublicShowcase
GetPublicShowcase
DownloadPublicShowcaseFile
```

The generated bindings are in `grpc_api/frame_audit_pb2.py` and
`grpc_api/frame_audit_pb2_grpc.py`.

Public gRPC binding still requires the existing API-key and TLS protections.
Do not expose an unauthenticated public port.

## Wang Xing Score Explanation

The three evidence scores have different meanings:

- Identity evidence: whether the face matches the Wang Xing identity profile.
- Expression evidence: whether facial movement matches the Wang Xing expression
  support profiles.
- Forensics evidence: strength of real-versus-generated forensic signals.
- Real-capture probability: an independent calibrated real-versus-AI
  probability from motion, texture residual, frequency, and temporal evidence.

Therefore, high identity or expression evidence does not imply high
real-capture probability. A generated video can preserve Wang Xing's identity
and expression while still showing AI-generation artifacts. The public card
now states this distinction directly. The UI always presents a binary
conclusion (`偏向 AI 生成` or `偏向真实拍摄`); the original uncertainty reasons
remain in the JSON for diagnostics.
