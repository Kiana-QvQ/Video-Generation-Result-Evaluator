"""Local Blender bridge for Codex.

The bridge listens only on localhost and executes requests on Blender's main
thread through a timer, which keeps bpy access on the thread Blender expects.
"""

from __future__ import annotations

import io
import json
import os
import queue
import socket
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import bpy


bl_info = {
    "name": "Codex Blender Bridge",
    "author": "Local project integration",
    "version": (0, 1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Codex",
    "description": "Expose a localhost Blender bridge for Codex MCP tools",
    "category": "Development",
}


HOST = os.environ.get("BLENDER_CODEX_HOST", "127.0.0.1")
PORT = int(os.environ.get("BLENDER_CODEX_PORT", "9876"))
TIMER_INTERVAL = 0.05

_jobs: queue.Queue[tuple[dict[str, Any], queue.Queue[dict[str, Any]]]] = queue.Queue()
_server_socket: socket.socket | None = None
_server_thread: threading.Thread | None = None
_stop_event = threading.Event()
_running = False


def _json_line(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _scene_summary() -> dict[str, Any]:
    objects = []
    for obj in bpy.context.scene.objects:
        objects.append(
            {
                "name": obj.name,
                "type": obj.type,
                "location": [round(float(v), 5) for v in obj.location],
                "dimensions": [round(float(v), 5) for v in obj.dimensions],
                "visible": bool(obj.visible_get()),
            }
        )
    return {
        "blend_file": bpy.data.filepath,
        "scene": bpy.context.scene.name,
        "object_count": len(objects),
        "objects": objects,
    }


def _execute_code(code: str) -> dict[str, Any]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    namespace: dict[str, Any] = {
        "bpy": bpy,
        "json": json,
        "os": os,
    }
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile(code, "<codex-blender>", "exec"), namespace, namespace)
        return {
            "ok": True,
            "result": namespace.get("result"),
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
        }
    except Exception as exc:  # Blender script errors must return to Codex.
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
        }


def _dispatch(request: dict[str, Any]) -> dict[str, Any]:
    method = request.get("method")
    params = request.get("params") or {}

    if method == "status":
        return {
            "ok": True,
            "host": HOST,
            "port": PORT,
            "blender_version": ".".join(str(v) for v in bpy.app.version),
            "blend_file": bpy.data.filepath,
            "scene": bpy.context.scene.name,
        }
    if method == "scene_summary":
        return {"ok": True, "scene": _scene_summary()}
    if method == "execute":
        code = params.get("code")
        if not isinstance(code, str) or not code.strip():
            return {"ok": False, "error": "params.code must be a non-empty string"}
        return _execute_code(code)
    if method == "save":
        filepath = params.get("filepath")
        if not isinstance(filepath, str) or not filepath.strip():
            return {"ok": False, "error": "params.filepath must be a non-empty string"}
        try:
            bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(filepath))
            return {"ok": True, "filepath": bpy.data.filepath}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if method == "render":
        filepath = params.get("filepath")
        if not isinstance(filepath, str) or not filepath.strip():
            return {"ok": False, "error": "params.filepath must be a non-empty string"}
        try:
            scene = bpy.context.scene
            scene.render.filepath = os.path.abspath(filepath)
            bpy.ops.render.render(write_still=True)
            return {"ok": True, "filepath": scene.render.filepath}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": False, "error": f"Unknown bridge method: {method}"}


def _client_worker(connection: socket.socket) -> None:
    with connection:
        connection.settimeout(300)
        buffer = b""
        while b"\n" not in buffer:
            chunk = connection.recv(65536)
            if not chunk:
                return
            buffer += chunk
        raw_request = buffer.split(b"\n", 1)[0]
        try:
            request = json.loads(raw_request.decode("utf-8"))
        except json.JSONDecodeError as exc:
            connection.sendall(_json_line({"ok": False, "error": f"Invalid JSON: {exc}"}))
            return

        response_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        _jobs.put((request, response_queue))
        try:
            response = response_queue.get(timeout=300)
        except queue.Empty:
            response = {"ok": False, "error": "Blender main-thread request timed out"}
        connection.sendall(_json_line(response))


def _server_worker() -> None:
    global _server_socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(8)
    server.settimeout(0.5)
    _server_socket = server
    try:
        while not _stop_event.is_set():
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=_client_worker,
                args=(connection,),
                daemon=True,
                name="codex-blender-client",
            ).start()
    finally:
        try:
            server.close()
        finally:
            _server_socket = None


def _process_jobs() -> float | None:
    if not _running:
        return None
    while True:
        try:
            request, response_queue = _jobs.get_nowait()
        except queue.Empty:
            break
        try:
            response_queue.put(_dispatch(request))
        except Exception as exc:
            response_queue.put(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
    return TIMER_INTERVAL


def start_bridge() -> None:
    global _running, _server_thread
    if _running:
        return
    _stop_event.clear()
    _running = True
    _server_thread = threading.Thread(
        target=_server_worker,
        daemon=True,
        name="codex-blender-server",
    )
    _server_thread.start()
    try:
        bpy.app.timers.register(_process_jobs, first_interval=TIMER_INTERVAL)
    except ValueError:
        # The timer is already registered after an add-on reload.
        pass
    print(f"[Codex Blender Bridge] listening on {HOST}:{PORT}")


def stop_bridge() -> None:
    global _running
    _running = False
    _stop_event.set()
    if _server_socket is not None:
        try:
            _server_socket.close()
        except OSError:
            pass
    try:
        bpy.app.timers.unregister(_process_jobs)
    except ValueError:
        pass


class CODEX_OT_start_bridge(bpy.types.Operator):
    bl_idname = "codex_bridge.start"
    bl_label = "Start Codex Bridge"

    def execute(self, _context):
        start_bridge()
        return {"FINISHED"}


class CODEX_OT_stop_bridge(bpy.types.Operator):
    bl_idname = "codex_bridge.stop"
    bl_label = "Stop Codex Bridge"

    def execute(self, _context):
        stop_bridge()
        return {"FINISHED"}


class CODEX_PT_bridge_panel(bpy.types.Panel):
    bl_label = "Codex Blender Bridge"
    bl_idname = "CODEX_PT_bridge_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Codex"

    def draw(self, _context):
        layout = self.layout
        status = "Running" if _running else "Stopped"
        layout.label(text=f"{status}: {HOST}:{PORT}")
        row = layout.row(align=True)
        row.operator(CODEX_OT_start_bridge.bl_idname)
        row.operator(CODEX_OT_stop_bridge.bl_idname)


_CLASSES = (
    CODEX_OT_start_bridge,
    CODEX_OT_stop_bridge,
    CODEX_PT_bridge_panel,
)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    start_bridge()


def unregister() -> None:
    stop_bridge()
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
