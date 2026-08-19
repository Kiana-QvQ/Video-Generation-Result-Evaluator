#!/usr/bin/env python3
"""Launch the standalone human review web application."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

try:
    from .app import create_app
    from .build_ai_quality_dataset import (
        collect_videos,
        load_source_queue,
        source_signature,
        build_dataset,
    )
    from .database import ReviewDatabase
except ImportError:
    from app import create_app
    from build_ai_quality_dataset import (
        collect_videos,
        load_source_queue,
        source_signature,
        build_dataset,
    )
    from database import ReviewDatabase


ROOT_DIR = Path(__file__).resolve().parent
QUALITY_QUEUE = ROOT_DIR / "data" / "ai_quality" / "source_queue.json"
QUALITY_STATE = ROOT_DIR / "data" / "ai_quality" / "active_dataset.json"
DEFAULT_DB = ROOT_DIR / "data" / "review.sqlite3"


def _resolve_queue_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT_DIR.parent / path
    return path.resolve()


def _next_dataset_id(database: ReviewDatabase, base_id: str) -> str:
    with database.connect() as connection:
        ids = [
            str(row["dataset_id"])
            for row in connection.execute(
                "SELECT dataset_id FROM quality_datasets"
            )
        ]
    if base_id not in ids:
        return base_id
    match = re.match(r"^(.*)_v(\d+)$", base_id)
    prefix = match.group(1) if match else base_id
    version = int(match.group(2)) if match else 1
    while f"{prefix}_v{version + 1}" in ids:
        version += 1
    return f"{prefix}_v{version + 1}"


def _load_quality_state() -> dict[str, str]:
    if not QUALITY_STATE.is_file():
        return {}
    try:
        value = json.loads(QUALITY_STATE.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_quality_state(dataset_id: str, signature: str) -> None:
    QUALITY_STATE.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "source_signature": signature,
                "updated_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def sync_quality_dataset() -> str | None:
    """Build/select the quality dataset before the server starts.

    The source queue is the only input the user needs to edit. Existing
    ratings are never rebuilt in place; a changed source creates the next
    dataset version automatically.
    """

    if str(os.getenv("HUMAN_REVIEW_AUTO_SYNC", "true")).lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return os.getenv("HUMAN_REVIEW_QUALITY_DATASET")

    queue = load_source_queue(QUALITY_QUEUE)
    input_dirs = [
        (
            _resolve_queue_path(item.get("path")),
            item.get("label"),
        )
        for item in queue.get("input_dirs", [])
        if item.get("enabled", True)
    ]
    input_dirs = [(path, label) for path, label in input_dirs if path]
    videos = collect_videos(input_dirs)
    signature = source_signature(videos)

    database = ReviewDatabase(
        Path(os.getenv("HUMAN_REVIEW_DB", DEFAULT_DB)),
        ip_secret=os.getenv("HUMAN_REVIEW_IP_SECRET", "human-review-local-v1"),
    )
    state = _load_quality_state()
    base_id = str(queue.get("dataset_id", "ai_quality_auto_v1"))
    dataset_id = str(state.get("dataset_id") or base_id)
    selected = database.get_active_quality_dataset(dataset_id)
    selected_signature = ""
    if selected:
        try:
            selected_signature = json.loads(
                selected["metadata_json"] or "{}"
            ).get("source_signature", "")
        except json.JSONDecodeError:
            selected_signature = ""

    if selected and selected_signature == signature:
        os.environ["HUMAN_REVIEW_QUALITY_DATASET"] = dataset_id
        return dataset_id

    if selected and database.count_quality_dataset_votes(dataset_id):
        dataset_id = _next_dataset_id(database, dataset_id)

    output_dir = ROOT_DIR / "data" / "datasets" / dataset_id
    manifest_value = queue.get("manifest")
    manifest_path = (
        _resolve_queue_path(manifest_value)
        if manifest_value
        else ROOT_DIR / "data" / "ai_quality" / "manifest.json"
    )
    build_dataset(
        input_dir=None,
        input_dirs=input_dirs,
        manifest_path=manifest_path,
        output_dir=output_dir,
        db_path=database.path,
        dataset_id=dataset_id,
        per_reviewer_quota=0,
    )
    _save_quality_state(dataset_id, signature)
    os.environ["HUMAN_REVIEW_QUALITY_DATASET"] = dataset_id
    return dataset_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the standalone human video review service."
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="监听地址，默认 0.0.0.0（允许局域网访问）",
    )
    parser.add_argument("--port", type=int, default=5001, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        quality_dataset_id = sync_quality_dataset()
        if quality_dataset_id:
            print(f"AI quality dataset: {quality_dataset_id}")
    except Exception as exc:
        print(f"AI quality dataset auto-sync skipped: {exc}")

    print(
        f"""
    ============================================================
    Human Review / 人工视频评测
    ------------------------------------------------------------
    本机访问:     http://127.0.0.1:{args.port}
    绑定地址:     http://{args.host}:{args.port}
    局域网访问:   http://<本机局域网IP>:{args.port}
    ------------------------------------------------------------
    按 Ctrl+C 停止服务
    ============================================================
    """
    )
    create_app().run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
    )


if __name__ == "__main__":
    main()
