"""Minimal stdio MCP server that forwards Codex tools to Blender."""

from __future__ import annotations

import json
import os
import socket
import sys
from typing import Any


HOST = os.environ.get("BLENDER_CODEX_HOST", "127.0.0.1")
PORT = int(os.environ.get("BLENDER_CODEX_PORT", "9876"))
SERVER_NAME = "blender-codex-bridge"
SERVER_VERSION = "0.1.0"


def _log(message: str) -> None:
    print(f"[{SERVER_NAME}] {message}", file=sys.stderr, flush=True)


def _send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _call_blender(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    request = {"method": method, "params": params or {}}
    try:
        with socket.create_connection((HOST, PORT), timeout=5) as connection:
            connection.settimeout(300)
            connection.sendall(
                (json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
            )
            chunks: list[bytes] = []
            while True:
                chunk = connection.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
        raw = b"".join(chunks).split(b"\n", 1)[0]
        if not raw:
            return {"ok": False, "error": "Blender bridge closed the connection"}
        return json.loads(raw.decode("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "error": (
                f"Cannot reach Blender at {HOST}:{PORT}. "
                "Launch tools/blender_codex/launch_blender_bridge.ps1 first. "
                f"{type(exc).__name__}: {exc}"
            ),
        }


TOOLS = [
    {
        "name": "blender_status",
        "description": "Check whether the local Blender bridge is running.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "blender_scene_summary",
        "description": "List the active Blender scene and its objects.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "blender_execute",
        "description": (
            "Execute a Python script in Blender's main thread. "
            "The script has bpy, json, and os available. "
            "Set a variable named result to return structured data."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code using Blender's bpy API."},
                "description": {
                    "type": "string",
                    "description": "Short description for logs and human review.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
    {
        "name": "blender_save",
        "description": "Save the current Blender file to an absolute path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Absolute .blend output path."},
            },
            "required": ["filepath"],
            "additionalProperties": False,
        },
    },
    {
        "name": "blender_render_preview",
        "description": "Render the active scene to an absolute image path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Absolute PNG/JPG output path."},
            },
            "required": ["filepath"],
            "additionalProperties": False,
        },
    },
]


def _text_result(payload: dict[str, Any]) -> dict[str, Any]:
    is_error = not payload.get("ok", False)
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, ensure_ascii=False, indent=2),
            }
        ],
        "isError": is_error,
    }


def _handle(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}

    if request_id is None:
        return None
    if method == "initialize":
        requested_version = params.get("protocolVersion", "2024-11-05")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": requested_version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name == "blender_status":
            payload = _call_blender("status")
        elif name == "blender_scene_summary":
            payload = _call_blender("scene_summary")
        elif name == "blender_execute":
            payload = _call_blender(
                "execute",
                {"code": arguments.get("code", ""), "description": arguments.get("description", "")},
            )
        elif name == "blender_save":
            payload = _call_blender("save", {"filepath": arguments.get("filepath", "")})
        elif name == "blender_render_preview":
            payload = _call_blender("render", {"filepath": arguments.get("filepath", "")})
        else:
            payload = {"ok": False, "error": f"Unknown tool: {name}"}
        return {"jsonrpc": "2.0", "id": request_id, "result": _text_result(payload)}
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    _log(f"starting; Blender endpoint is {HOST}:{PORT}")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = _handle(request)
            if response is not None:
                _send(response)
        except json.JSONDecodeError as exc:
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Invalid JSON: {exc}"},
                }
            )
        except Exception as exc:
            _log(f"request failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
