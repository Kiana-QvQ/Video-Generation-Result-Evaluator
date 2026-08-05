# gRPC 备选接口

gRPC 是现有 HTTP/FastAPI 接口之外的备用传输方式。启用 gRPC 不会改变 HTTP 的 `7860` 端口、路径、浏览器页面或 JSON 内容。

## 启动

首次安装 gRPC 依赖：

```powershell
.\setup.ps1 -Grpc
```

默认只监听本机 `127.0.0.1:50051`：

```powershell
.\run-grpc.ps1
```

供公司局域网内的其他电脑访问：

```powershell
.\run-grpc.ps1 -Public
```

也可以指定地址和端口：

```powershell
.\run-grpc.ps1 -BindHost 0.0.0.0 -Port 50051
```

一键同时启动 HTTP 和 gRPC：

```powershell
python .\start.py --with-grpc
```

如果使用启动脚本并让两个服务都监听局域网：

```powershell
.\run.ps1 -Public -WithGrpc
```

`.\run.ps1` 仍然只负责 HTTP 服务，默认地址仍是 `127.0.0.1:7860`。

## 方法对应关系

| gRPC 方法 | 现有 HTTP 请求 | 返回内容 |
| --- | --- | --- |
| `Health` | `GET /api/health` | 相同 JSON |
| `Models` | `GET /api/models` | 相同 JSON |
| `Hardware` | `GET /api/hardware?device=...` | 相同硬件策略 JSON |
| `CreateJob` | `POST /api/jobs` | 相同任务 JSON，推荐使用 |
| `Evaluate` | `POST /api/evaluate` | 相同同步评估 JSON |
| `ListJobs` | `GET /api/jobs?limit=...` | 相同任务列表 JSON |
| `GetJob` | `GET /api/jobs/{job_id}` | 相同任务详情和结果 JSON |
| `UpdateJob` | `PATCH /api/jobs/{job_id}` | 相同取消、重试、改名结果 |
| `DeleteJob` | `DELETE /api/jobs/{job_id}` | 相同删除结果 |
| `DownloadRunFile` | `GET /api/runs/{run_id}/{filename}` | 分块文件流 |

除文件下载外，所有响应都放在 `JsonResponse.json` 中，内容是原 HTTP 接口返回的 JSON 字符串；`JsonResponse.http_status` 保留原 HTTP 状态码语义。校验失败会转换为 gRPC 状态码。

## 上传协议

`CreateJob` 和 `Evaluate` 都是 client-streaming RPC。客户端先发送一次 `UploadRequest.options`，再发送一个或多个 `UploadRequest.chunk`：

```text
options
result_video: first chunk ... last chunk
gt_video: first chunk ... last chunk          optional
reference_images: first chunk ... last chunk  optional, can repeat
reference_video: first chunk ... last chunk   optional, can repeat
```

每个文件使用唯一的 `file_id`。第一个分块设置 `first=true`，最后一个分块设置 `last=true`，中间分块两个标志都为 `false`。单个文件仍限制为 1.5 GB，扩展名校验与 HTTP 接口一致。重复的 `reference_video` 会按上传顺序拼接；`gt_video` 最多只能上传一个。

gRPC 与 HTTP 共用认证、文件数量、总请求大小、视频时长、分辨率、帧数和
FFmpeg 超时限制。公网监听时必须配置 TLS 和 `FRAME_AUDIT_API_KEY`；客户端
通过 `authorization: Bearer <key>` 或 `x-api-key: <key>` metadata 发送密钥。

`JobOptions` 对应 HTTP 表单字段：

| JobOptions | HTTP 表单字段 | 默认值 |
| --- | --- | --- |
| `prompt_text` | `prompt_text` | 空字符串 |
| `max_frames` | `max_frames` | `8` |
| `calculate_lpips` | `calculate_lpips` | `true` |
| `device` | `device` | `auto` |
| `manual_expression_score` | `manual_expression_score` | 空字符串 |
| `manual_aesthetic_score` | `manual_aesthetic_score` | 空字符串 |
| `wangxing_au_enabled` | `wangxing_au_enabled` | `false` |
| `wangxing_expected_class` | `wangxing_expected_class` | `auto` |

## Python 客户端最小示例

先生成或安装客户端依赖后，在项目根目录运行：

```python
from pathlib import Path
import json
import grpc

from grpc_api import frame_audit_pb2 as pb2
from grpc_api import frame_audit_pb2_grpc as pb2_grpc


def upload(path: Path, field_name: str):
    chunk_size = 1024 * 1024
    with path.open("rb") as source:
        data = source.read(chunk_size)
        if not data:
            yield pb2.UploadRequest(
                chunk=pb2.UploadChunk(
                    file_id=field_name,
                    field_name=field_name,
                    filename=path.name,
                    first=True,
                    last=True,
                )
            )
            return
        first = True
        while data:
            next_data = source.read(chunk_size)
            yield pb2.UploadRequest(
                chunk=pb2.UploadChunk(
                    file_id=field_name,
                    field_name=field_name,
                    filename=path.name,
                    data=data,
                    first=first,
                    last=not next_data,
                )
            )
            first = False
            data = next_data


channel = grpc.insecure_channel("127.0.0.1:50051")
stub = pb2_grpc.FrameAuditStub(channel)
response = stub.Health(pb2.Empty())
print(json.loads(response.json))
channel.close()
```

生产客户端应使用 `DownloadRunFile` 读取结果文件，不应直接拼接 HTTP 下载地址。`CreateJob` 是异步队列接口，提交后使用 `GetJob` 轮询状态；GPU 评估仍由同一个队列串行处理。
