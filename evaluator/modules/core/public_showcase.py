"""Persistent read-only public showcase index for web and gRPC clients.

The normal job queue is intentionally private to the submitting client IP.
This module exposes a separate curated index for management/demo clients.
Entries point to existing result files instead of duplicating large videos.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from .runtime import OUTPUT_DIR, PROJECT_ROOT

PUBLIC_SHOWCASE_DIR = OUTPUT_DIR / "public_showcase"
PUBLIC_SHOWCASE_INDEX = PUBLIC_SHOWCASE_DIR / "index.json"
PUBLIC_SHOWCASE_SCHEMA = "public_showcase_v1"

_PUBLIC_FILE_ROOTS = (
    (PROJECT_ROOT / "outputs").resolve(),
    (PROJECT_ROOT / "data" / "test").resolve(),
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _safe_item_id(item_id: str) -> str:
    value = Path(str(item_id)).name
    if not value or value != str(item_id):
        raise ValueError("Invalid public showcase item id.")
    return value


def _project_path(value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _relative_project_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _read_index() -> dict[str, Any]:
    if not PUBLIC_SHOWCASE_INDEX.is_file():
        return {
            "schema_version": PUBLIC_SHOWCASE_SCHEMA,
            "queue_name": "public_showcase",
            "updated_at": None,
            "items": [],
        }
    try:
        payload = _load_json(PUBLIC_SHOWCASE_INDEX)
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": PUBLIC_SHOWCASE_SCHEMA,
            "queue_name": "public_showcase",
            "updated_at": None,
            "items": [],
        }
    if not isinstance(payload, dict):
        return {
            "schema_version": PUBLIC_SHOWCASE_SCHEMA,
            "queue_name": "public_showcase",
            "updated_at": None,
            "items": [],
        }
    items = payload.get("items")
    payload["items"] = items if isinstance(items, list) else []
    return payload


def _find_item(item_id: str) -> dict[str, Any]:
    safe_id = _safe_item_id(item_id)
    for item in _read_index().get("items", []):
        if isinstance(item, dict) and str(item.get("item_id")) == safe_id:
            return item
    raise FileNotFoundError(f"Public showcase item not found: {safe_id}")


def _source_payload(item: dict[str, Any]) -> Any:
    source = item.get("source") or {}
    if not isinstance(source, dict):
        return None
    source_path = _project_path(str(source.get("path") or ""))
    if not source_path.is_file():
        return None
    try:
        payload = _load_json(source_path)
    except (OSError, json.JSONDecodeError):
        return None
    kind = str(source.get("kind") or "")
    if kind == "forensics_results":
        rows = payload.get("results") if isinstance(payload, dict) else None
        index = int(source.get("index", -1))
        if isinstance(rows, list) and 0 <= index < len(rows):
            return rows[index]
        return None
    if kind == "forensics_summary":
        if isinstance(payload, dict):
            return payload.get("summary") or payload
        return payload
    if kind == "web_run":
        return payload
    return payload


def _effective_item(item: dict[str, Any]) -> dict[str, Any]:
    """Refresh live web-run metadata without rebuilding the showcase index."""
    refreshed = dict(item)
    source = item.get("source") or {}
    if not isinstance(source, dict) or source.get("kind") != "web_run":
        return refreshed
    result_path = _project_path(str(source.get("path") or ""))
    status_path = result_path.parent / "status.json"
    if not status_path.is_file():
        return refreshed
    try:
        status = _load_json(status_path)
    except (OSError, json.JSONDecodeError):
        return refreshed
    if not isinstance(status, dict):
        return refreshed
    for key in (
        "name",
        "status",
        "stage",
        "progress",
        "created_at",
        "queued_at",
        "started_at",
        "finished_at",
        "updated_at",
        "error",
    ):
        if key in status:
            target_key = "title" if key == "name" else key
            refreshed[target_key] = status[key]
    refreshed["result_available"] = (
        str(status.get("status")) == "completed"
        and result_path.is_file()
    )
    return refreshed


def _item_files(item: dict[str, Any]) -> dict[str, Path]:
    files = item.get("files") or {}
    if not isinstance(files, dict):
        return {}
    resolved: dict[str, Path] = {}
    for key, value in files.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        path = _project_path(value)
        if not path.is_file():
            continue
        if not any(path.is_relative_to(root) for root in _PUBLIC_FILE_ROOTS):
            continue
        resolved[key] = path
    return resolved


def _public_file_url(item_id: str, file_key: str) -> str:
    return f"/api/public-showcase/{item_id}/files/{file_key}"


def list_public_showcase(
    *,
    limit: int = 50,
    query: str = "",
    category: str = "",
) -> dict[str, Any]:
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000.")
    payload = _read_index()
    items = [
        item
        for item in payload.get("items", [])
        if isinstance(item, dict)
    ]
    query_value = str(query or "").strip().casefold()
    category_value = str(category or "").strip().casefold()
    if query_value:
        items = [
            item
            for item in items
            if query_value in " ".join(
                str(item.get(key) or "")
                for key in ("item_id", "title", "category", "sample_id")
            ).casefold()
        ]
    if category_value:
        items = [
            item
            for item in items
            if str(item.get("category") or "").casefold() == category_value
        ]
    items.sort(
        key=lambda item: str(
            item.get("published_at")
            or item.get("created_at")
            or item.get("updated_at")
            or ""
        ),
        reverse=True,
    )
    visible = []
    for item in items[:limit]:
        copy = _effective_item(item)
        copy["result_available"] = (
            copy.get("result_available", False)
            or _source_payload(item) is not None
        )
        copy["downloads"] = {
            key: _public_file_url(str(item["item_id"]), key)
            for key in _item_files(item)
        }
        visible.append(_json_safe(copy))
    return {
        "schema_version": PUBLIC_SHOWCASE_SCHEMA,
        "queue_name": payload.get("queue_name", "public_showcase"),
        "updated_at": payload.get("updated_at"),
        "total_count": len(items),
        "items": visible,
    }


def get_public_showcase(item_id: str) -> dict[str, Any]:
    item = _effective_item(_find_item(item_id))
    result = _source_payload(item)
    downloads = {
        key: _public_file_url(str(item["item_id"]), key)
        for key in _item_files(item)
    }
    return _json_safe(
        {
            "schema_version": PUBLIC_SHOWCASE_SCHEMA,
            "item": item,
            "result": result,
            "downloads": downloads,
        }
    )


def resolve_public_showcase_file(item_id: str, file_key: str) -> Path:
    item = _find_item(item_id)
    key = str(file_key)
    if not key or Path(key).name != key:
        raise FileNotFoundError("Invalid public showcase file key.")
    path = _item_files(item).get(key)
    if path is None:
        raise FileNotFoundError("Public showcase file not found.")
    return path


def public_showcase_status() -> dict[str, Any]:
    payload = _read_index()
    items = payload.get("items", [])
    return {
        "ready": bool(items),
        "schema_version": payload.get("schema_version", PUBLIC_SHOWCASE_SCHEMA),
        "queue_name": payload.get("queue_name", "public_showcase"),
        "updated_at": payload.get("updated_at"),
        "item_count": len(items) if isinstance(items, list) else 0,
        "index_path": str(PUBLIC_SHOWCASE_INDEX),
    }


def write_public_showcase_index(
    items: list[dict[str, Any]],
    *,
    queue_name: str = "领导公共展示队列",
    selection: dict[str, Any] | None = None,
) -> Path:
    PUBLIC_SHOWCASE_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = {
        "schema_version": PUBLIC_SHOWCASE_SCHEMA,
        "queue_name": queue_name,
        "updated_at": now,
        "selection": selection or {"mode": "all_results"},
        "items": _json_safe(items),
    }
    temporary = PUBLIC_SHOWCASE_INDEX.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(PUBLIC_SHOWCASE_INDEX)
    return PUBLIC_SHOWCASE_INDEX
