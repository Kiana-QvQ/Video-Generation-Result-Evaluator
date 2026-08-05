# 公网部署

## 公司内网部署（推荐）

如果员工都在同一个公司局域网或 VPN 中，不需要云服务器、域名、登录和注册。大显存 Windows 电脑就是服务端，员工通过它的内网 IP 访问。

### 1. 准备目标电脑

在目标电脑安装：

- NVIDIA 驱动，并确认 `nvidia-smi` 能看到显卡。
- Python 3.11（建议与开发电脑保持同一小版本）。
- FFmpeg，并确保 `ffmpeg -version` 可以执行。
- Git，或者直接把整个项目目录复制过去。

不要复制开发电脑的 `.venv`。Windows 虚拟环境中的路径和二进制依赖通常不可迁移，应在目标电脑重新创建。

### 2. 复制项目和模型

可以复制整个项目目录，但至少需要复制：

- 项目源代码。
- `model_cache/` 中已经下载的模型权重。
- `config/`、`requirements.txt` 和 `requirements/`。

不要复制运行中的 `.venv/`。`outputs/` 可以按需要复制；如果不需要历史任务结果，可以只创建一个空目录。

### 3. 在目标电脑安装环境

在项目根目录打开 PowerShell：

```powershell
.\setup.ps1 -Cuda -Optional
```

安装完成后验证 GPU：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

第二条命令应包含 `CUDAExecutionProvider`。如果目标电脑使用 CUDA 11.x，安装 ONNX Runtime GPU 时需要使用项目安装脚本中对应的 CUDA 11 源。

### 4. 设置固定内网 IP

让公司网络管理员为目标电脑配置 DHCP 保留或固定 IP，例如：

```text
192.168.1.50
```

员工以后访问：

```text
http://192.168.1.50:7860/
```

不要依赖经常变化的 DHCP 地址。

### 5. 只开放公司内网端口

以管理员身份打开 PowerShell，在目标电脑执行：

```powershell
New-NetFirewallRule `
    -DisplayName "Video Evaluator Intranet 7860" `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort 7860 `
    -Profile Domain,Private `
    -RemoteAddress LocalSubnet
```

这条规则只允许 Windows 判断为本地子网的设备访问，不开放公网。不要选择 `Public` 网络配置文件，也不要做家用路由器端口映射。

### 6. 启动内网服务

在目标电脑执行：

```powershell
.\run.ps1 -Public
```

看到 `Listening on http://0.0.0.0:7860` 后，员工使用目标电脑的固定内网 IP 访问。目标电脑关机、睡眠或服务窗口关闭时，网页都不可用。

### 7. 配置开机自启

最简单的方式是使用“任务计划程序”：

1. 创建任务，例如 `Video Evaluator`。
2. 触发器选择“计算机启动时”。
3. 勾选“无论用户是否登录都运行”。
4. 操作选择启动程序 `powershell.exe`。
5. 参数填写：

   ```text
   -NoProfile -ExecutionPolicy Bypass -File ".\run.ps1" -Public
   ```

6. 工作目录设置为项目根目录。
7. 使用目标电脑上的固定服务账号运行，并授予该账号读写项目目录的权限。

### 8. 访问限制

项目当前没有登录和注册功能，因此“谁能访问”由公司网络和 Windows 防火墙控制。至少应满足：

- 只允许公司内网/VPN 网段访问。
- 不把 `7860` 映射到公网。
- 不把带有敏感视频的结果目录共享给不需要的人。
- 给 `outputs/` 和临时上传目录预留足够磁盘空间。
- 确认单 GPU 任务队列不会被多人同时提交压满。

## 先确认部署方式

当前项目的 GPU 和模型都在这台 Windows 电脑上。把网页发布到公网，并不会把推理迁移到云端；这台电脑必须保持开机、服务运行，并且能访问外网。

项目没有内置登录系统。不要把未加保护的地址长期公开，否则任何拿到地址的人都可以上传视频并消耗本机 GPU 和磁盘。

## 临时分享：Cloudflare Quick Tunnel

适合演示和短期让别人访问，不需要路由器端口映射或公网 IP。

1. 正常启动项目：

   ```powershell
   .\run.ps1
   ```

2. 在第二个 PowerShell 窗口运行：

   ```powershell
   cloudflared tunnel --url http://127.0.0.1:7860
   ```

3. 复制命令输出的 `https://*.trycloudflare.com` 地址给别人。

这个地址会在 `cloudflared` 进程结束后失效，适合测试，不适合正式服务。

## 局域网访问

如果访问者和这台电脑在同一个局域网：

```powershell
.\run.ps1 -Public
```

在 Windows 防火墙中允许 TCP `7860` 后，访问者打开：

```text
http://这台电脑的局域网IP:7860
```

可用下面的命令查看局域网 IPv4 地址：

```powershell
(Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object {$_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -ne "WellKnown"}).IPAddress
```

## 长期公网服务：云服务器反向代理

推荐使用一台有公网 IP 的 Linux 云服务器作为入口，域名指向云服务器，Nginx 或 Caddy 负责 HTTPS 和登录保护；GPU Windows 电脑通过 VPN、专线或已认证的隧道连接到入口服务器。不要直接把 `7860` 端口暴露到公网。

云服务器安全组只放行 `80/443`，SSH 仅允许你的固定 IP；应用内部端口只允许反向代理访问。

如果要让公网入口直接转发到当前电脑，至少需要：

1. 公网 IP 或稳定隧道。
2. 域名和 HTTPS。
3. 反向代理认证或 Cloudflare Access。
4. 限制上传大小、请求频率和允许的来源。
5. 将 `outputs/` 和上传目录放到有容量监控的磁盘。

## 自定义绑定地址和端口

默认仍然是本机访问：

```powershell
.\run.ps1
```

也可以显式指定：

```powershell
.\run.ps1 -BindHost 0.0.0.0 -Port 7860
```

`-Public` 只改变监听地址，不会自动创建公网入口，也不会自动配置防火墙、域名或 HTTPS。

当前服务已拒绝无保护的公网启动：非回环地址需要
`FRAME_AUDIT_API_KEY`，并默认需要 TLS 证书和私钥。详细的 API key、限流、
上传限制、任务清理、单 worker 队列和依赖隔离说明见
[`SECURITY.md`](SECURITY.md)。
