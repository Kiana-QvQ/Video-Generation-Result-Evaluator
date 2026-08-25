# Blender + Codex

这个目录提供一个不依赖第三方 Python 包的本地桥接：

```text
Codex MCP STDIO -> mcp_server.py -> TCP localhost -> Blender bpy bridge
```

## 已确认的本机路径

- Blender: `D:\Steam\steamapps\common\Blender\blender.exe`
- Blender 版本: `4.2.23 LTS`
- Python: `D:\Anaconda3\python.exe`
- uvx: `D:\Anaconda3\Scripts\uvx.exe`

## 启动

先启动带桥接的 Blender。推荐双击或在终端运行 `.cmd` 包装器，它会绕过
Windows 默认禁止脚本执行的策略：

```powershell
.\tools\blender_codex\launch_blender_bridge.cmd
```

如果当前 PowerShell 允许脚本执行，也可以直接运行：

```powershell
.\tools\blender_codex\launch_blender_bridge.ps1
```

也可以打开已有的 `.blend`：

```powershell
.\tools\blender_codex\launch_blender_bridge.cmd -BlendFile "D:\path\scene.blend"
```

然后重启 Codex 或重新打开这个项目。项目配置位于 `.codex/config.toml`，会把
`mcp_server.py` 注册为 `blender_codex` MCP server。

## 可用工具

- `blender_status`: 检查 Blender 是否在线。
- `blender_scene_summary`: 查看当前场景和物体。
- `blender_execute`: 在 Blender 主线程执行 `bpy` Python 脚本。
- `blender_save`: 保存当前 `.blend`。
- `blender_render_preview`: 渲染当前场景到图片。

建模时让 Codex 调用 `blender_execute`，把生成的模型脚本放在一次调用中执行；
脚本可以设置 `result` 变量返回摘要。执行完成后再调用 `blender_save` 和
`blender_render_preview` 验证结果。

## 本地安全边界

桥接只绑定 `127.0.0.1`，但 `blender_execute` 有能力执行 Blender Python，
因此只应在自己的机器和可信项目中启用。不要把端口暴露到局域网或公网。

## 离线自检

下面的命令不需要启动 MCP：

```powershell
& "D:\Steam\steamapps\common\Blender\blender.exe" `
  --background `
  --python ".\tools\blender_codex\examples\create_demo_scene.py"
```

输出会写入 `outputs/blender_codex_demo/`，包括 `.blend` 和渲染预览图。
