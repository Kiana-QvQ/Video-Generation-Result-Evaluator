"""Build the read-only public showcase queue from preserved result files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.public_showcase import (
    _json_safe,
    _relative_project_path,
    write_public_showcase_index,
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _relative(path: Path) -> str:
    return _relative_project_path(path)


def _item_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    safe = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in prefix
    ).strip("_")
    return f"{safe}_{digest}"


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(
        path.stat().st_mtime,
    ).astimezone().isoformat(timespec="seconds")


def _label_from_row(row: dict[str, Any]) -> str:
    label = str(row.get("label") or "").lower()
    if label in {"ai", "generated", "seedance"}:
        return "AI 生成"
    if label in {"real", "real_capture"}:
        return "实拍"
    return label or "未标注"


def _preview(row: dict[str, Any]) -> dict[str, Any]:
    card = row.get("web_card")
    if isinstance(card, dict):
        return {
            "title": card.get("title"),
            "status": card.get("status"),
            "conclusion": (
                card.get("forensics", {}).get("conclusion")
                if isinstance(card.get("forensics"), dict)
                else None
            ),
            "real_probability": (
                card.get("forensics", {}).get("calibrated_real_probability")
                if isinstance(card.get("forensics"), dict)
                else None
            ),
            "identity_score": (
                card.get("radar", {}).get("identity", {}).get("score")
                if isinstance(card.get("radar"), dict)
                else None
            ),
            "expression_score": (
                card.get("radar", {}).get("expression", {}).get("score")
                if isinstance(card.get("radar"), dict)
                else None
            ),
            "forensics_score": (
                card.get("radar", {}).get("forensics", {}).get("score")
                if isinstance(card.get("radar"), dict)
                else None
            ),
        }
    return {
        "title": "王兴专项评估",
        "status": (row.get("wangxing") or {}).get("summary", {}).get(
            "identity_conclusion"
        ),
        "conclusion": (
            (row.get("forensics") or {}).get("summary", {}).get("conclusion")
        ),
        "real_probability": (
            (row.get("forensics") or {}).get("scores", {}).get(
                "calibrated_real_probability_0_1"
            )
        ),
    }


def _add_web_run_items(
    items: list[dict[str, Any]],
    *,
    root: Path,
    source_label: str,
) -> None:
    if not root.is_dir():
        return
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        status_path = run_dir / "status.json"
        result_path = run_dir / "result.json"
        if not result_path.is_file():
            continue
        status: dict[str, Any] = {}
        if status_path.is_file():
            try:
                status = _load(status_path)
            except (OSError, json.JSONDecodeError):
                status = {}
        if status.get("status") not in {None, "completed"}:
            continue
        job_id = str(status.get("job_id") or run_dir.name)
        item = {
            "item_id": _item_id("web_run_" + job_id, source_label),
            "title": str(
                status.get("name")
                or status.get("original_files", {}).get("result_video")
                or job_id
            ),
            "category": "网页任务",
            "sample_id": job_id,
            "label": "历史网页任务",
            "status": "completed",
            "created_at": status.get("created_at") or _mtime(result_path),
            "published_at": status.get("finished_at") or _mtime(result_path),
            "source": {
                "kind": "web_run",
                "path": _relative(result_path),
            },
            "files": {},
            "preview": {},
            "source_label": source_label,
        }
        for key, filename in (
            ("video", "result.mp4"),
            ("result_json", "result.json"),
            ("summary_csv", "summary.csv"),
            ("frame_csv", "frame_metrics.csv"),
            ("wangxing_json", "wangxing_au_result.json"),
        ):
            candidate = run_dir / filename
            if candidate.is_file():
                item["files"][key] = _relative(candidate)
        try:
            result = _load(result_path)
        except (OSError, json.JSONDecodeError):
            result = {}
        if isinstance(result, dict):
            item["preview"] = _preview(result)
        items.append(item)


def _add_forensics_items(
    items: list[dict[str, Any]],
    *,
    root: Path,
) -> None:
    if not root.is_dir():
        return
    for result_path in sorted(root.rglob("all_results.json")):
        try:
            payload = _load(result_path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        rows = payload.get("results")
        if not isinstance(rows, list):
            continue
        report_name = result_path.parent.name
        summary_id = _item_id(
            "report_" + report_name,
            str(result_path),
        )
        items.append(
            {
                "item_id": summary_id,
                "title": f"{report_name} 汇总",
                "category": "网页测试汇总",
                "sample_id": report_name,
                "label": "测试集汇总",
                "status": "completed",
                "created_at": _mtime(result_path),
                "published_at": _mtime(result_path),
                "source": {
                    "kind": "forensics_summary",
                    "path": _relative(result_path),
                },
                "files": {"result_json": _relative(result_path)},
                "preview": payload.get("summary", {}),
                "source_label": "outputs/forensics",
            }
        )
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            sample_id = str(
                row.get("sample_id")
                or Path(str(row.get("video") or f"sample_{index}")).stem
            )
            item = {
                "item_id": _item_id(
                    "report_" + report_name + "_" + sample_id,
                    f"{result_path}:{index}",
                ),
                "title": f"{report_name} / {sample_id}",
                "category": "网页测试",
                "sample_id": sample_id,
                "label": _label_from_row(row),
                "status": str(row.get("status") or "completed"),
                "created_at": _mtime(result_path),
                "published_at": _mtime(result_path),
                "source": {
                    "kind": "forensics_results",
                    "path": _relative(result_path),
                    "index": index,
                },
                "files": {"result_json": _relative(result_path)},
                "preview": _preview(row),
                "source_label": "outputs/forensics",
            }
            video = Path(str(row.get("video") or ""))
            if video.is_file() and video.is_relative_to(PROJECT_ROOT):
                item["files"]["video"] = _relative(video)
            items.append(item)


def _add_metric_items(items: list[dict[str, Any]], root: Path) -> None:
    patterns = (
        "wangxing_v3_*metrics.json",
        "wangxing_v4_*metrics.json",
        "wangxing_web_v3_*metrics.json",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.glob(pattern))
    for path in sorted(paths):
        try:
            payload = _load(path)
        except (OSError, json.JSONDecodeError):
            continue
        items.append(
            {
                "item_id": _item_id("metrics_" + path.stem, str(path)),
                "title": path.stem,
                "category": "模型指标",
                "sample_id": path.stem,
                "label": "指标报告",
                "status": "completed",
                "created_at": _mtime(path),
                "published_at": _mtime(path),
                "source": {"kind": "json_document", "path": _relative(path)},
                "files": {"result_json": _relative(path)},
                "preview": payload.get("headline", payload.get("summary", {}))
                if isinstance(payload, dict)
                else {},
                "source_label": "outputs/forensics",
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the public management showcase queue."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/public_showcase",
    )
    parser.add_argument("--max-items", type=int, default=1000)
    parser.add_argument(
        "--without-archived",
        action="store_true",
        help="Do not import the archived web_runs history.",
    )
    parser.add_argument(
        "--without-metrics",
        action="store_true",
        help="Do not import PT/metric JSON summaries.",
    )
    args = parser.parse_args()

    items: list[dict[str, Any]] = []
    _add_web_run_items(
        items,
        root=PROJECT_ROOT / "outputs" / "web_runs",
        source_label="outputs/web_runs",
    )
    if not args.without_archived:
        _add_web_run_items(
            items,
            root=PROJECT_ROOT / "outputs" / "历史归档" / "缓存" / "web_runs",
            source_label="outputs/历史归档/缓存/web_runs",
        )
    _add_forensics_items(
        items,
        root=PROJECT_ROOT / "outputs" / "forensics",
    )
    if not args.without_metrics:
        _add_metric_items(
            items,
            PROJECT_ROOT / "outputs" / "forensics",
        )

    unique: dict[str, dict[str, Any]] = {}
    for item in items:
        unique[str(item["item_id"])] = item
    ordered = sorted(
        unique.values(),
        key=lambda item: str(item.get("published_at") or ""),
        reverse=True,
    )
    if args.max_items > 0:
        ordered = ordered[: args.max_items]
    path = write_public_showcase_index(ordered)
    print(
        json.dumps(
            {
                "item_count": len(ordered),
                "index": str(path),
                "categories": sorted(
                    {
                        str(item.get("category") or "")
                        for item in ordered
                    }
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
