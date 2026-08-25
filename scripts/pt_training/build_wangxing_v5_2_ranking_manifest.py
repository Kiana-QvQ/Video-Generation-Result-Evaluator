"""Build a grouped V5.2 ranking manifest from ppt and optional LTX videos.

Partial LTX groups (LoRA + multiref only) can be completed by drawing real /
Seedance proxies from existing project pools that are outside the final
binary holdouts.  This unlocks RankHead training (complete train groups >= 4)
while keeping ppt_test2 as a native same-prompt holdout.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wangxing_project.v52_ranking_data import (
    DEFAULT_REAL_POOLS,
    DEFAULT_SEEDANCE_POOLS,
    ORDER,
    complete_partial_groups,
    is_complete,
)


def _label(path: Path) -> str | None:
    name = path.name.casefold()
    if "真人" in path.name or "real" in name:
        return "real"
    if "iclora" in name or "lora" in name:
        return "lora"
    if "seedance" in name:
        return "seedance"
    if "多图" in path.name or "multiref" in name:
        return "multiref"
    return None


def _group_id(path: Path, root: Path, prefix: str) -> str:
    relative = path.relative_to(root)
    if prefix == "ppt":
        return f"ppt_{relative.parts[0]}"
    match = re.search(r"[_-](\d+)(?:\.[^.]+)$", path.name)
    suffix = match.group(1) if match else path.stem
    if suffix.isdigit():
        suffix = suffix.zfill(2)
    return f"ltx_{suffix}"


def _collect_root(root: Path, prefix: str) -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    if not root.is_dir():
        return groups
    for video in sorted(root.rglob("*.mp4")):
        label = _label(video)
        if label is None:
            continue
        group = _group_id(video, root, prefix)
        groups.setdefault(group, {})
        if label not in groups[group]:
            groups[group][label] = str(video.resolve())
    return groups


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build V5.2 grouped ranking manifest; optionally complete "
            "partial LTX groups from real/Seedance pools."
        )
    )
    parser.add_argument(
        "--ppt-root",
        default=r"C:\Users\zhanghaotian\Desktop\ppt_video",
    )
    parser.add_argument("--ltx-root", default="data/LTX")
    parser.add_argument("--holdout-group", default="ppt_test2")
    parser.add_argument("--holdout-group-count", type=int, default=1)
    parser.add_argument(
        "--complete-from-pools",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill missing real/seedance for train partial groups from pools.",
    )
    parser.add_argument(
        "--min-complete-train",
        type=int,
        default=5,
        help="Require at least this many complete train groups after fill.",
    )
    parser.add_argument(
        "--real-pool",
        action="append",
        default=None,
        help="Real video pool root (repeatable). Defaults to MD_CL + video.",
    )
    parser.add_argument(
        "--seedance-pool",
        action="append",
        default=None,
        help="Seedance video pool root (repeatable).",
    )
    parser.add_argument(
        "--output",
        default="data/ranking/wangxing_v5_2/manifest.json",
    )
    parser.add_argument(
        "--completion-report",
        default="data/ranking/wangxing_v5_2/completion_report.json",
    )
    args = parser.parse_args(argv)

    ppt_root = Path(args.ppt_root).expanduser().resolve()
    ltx_root = _resolve_path(args.ltx_root)

    ppt_groups = _collect_root(ppt_root, "ppt")
    ltx_groups = _collect_root(ltx_root, "ltx")
    merged = {**ppt_groups, **ltx_groups}
    if not merged:
        raise SystemExit("No labeled ranking videos were found.")

    full_groups = [
        group for group, videos in merged.items() if is_complete(videos)
    ]
    if args.holdout_group in merged:
        holdout = [args.holdout_group]
    else:
        holdout = sorted(full_groups)[-max(1, args.holdout_group_count) :]

    rows: list[dict[str, Any]] = []
    for group_id in sorted(merged):
        videos = merged[group_id]
        split = "holdout" if group_id in holdout else "train"
        native_full = is_complete(videos)
        rows.append(
            {
                "group_id": group_id,
                "split": split,
                "matching_key": group_id,
                "completeness": "full" if native_full else "partial",
                "completion_mode": (
                    "native_full" if native_full else "partial"
                ),
                "same_prompt_matched": native_full,
                "filled_roles": {},
                "videos": {role: videos.get(role) for role in ORDER},
            }
        )

    completion_report: dict[str, Any] = {
        "enabled": False,
        "completions": [],
    }
    if args.complete_from_pools:
        completion_report = complete_partial_groups(
            rows,
            project_root=PROJECT_ROOT,
            real_pools=args.real_pool or DEFAULT_REAL_POOLS,
            seedance_pools=args.seedance_pool or DEFAULT_SEEDANCE_POOLS,
            min_complete_train=args.min_complete_train,
        )
        completion_report["enabled"] = True

    final_markers = (
        "data\\test\\single_video",
        "data/test/single_video",
        "data\\test\\wangxing_32x32",
        "data/test/wangxing_32x32",
    )
    for row in rows:
        for path in (row["videos"] or {}).values():
            if path and any(
                marker.casefold() in str(path).casefold()
                for marker in final_markers
            ):
                raise SystemExit(
                    f"Final-test video entered ranking manifest: {path}"
                )

    train_complete = sum(
        row["split"] == "train" and row["completeness"] == "full"
        for row in rows
    )
    payload = {
        "schema_version": "wangxing_v5_2_ranking_manifest_v1",
        "order_prior": list(ORDER),
        "forbidden_final_test": [
            "data/test/single_video",
            "data/test/wangxing_32x32",
        ],
        "source_roots": {
            "ppt": str(ppt_root),
            "ltx": str(ltx_root),
        },
        "completion": {
            "enabled": bool(args.complete_from_pools),
            "mode": "pool_fill_dev" if args.complete_from_pools else "none",
            "min_complete_train": int(args.min_complete_train),
            "train_complete_groups": train_complete,
            "report": completion_report,
            "note": (
                "Pool-filled real/seedance are development proxies to unlock "
                "RankHead fitting. Holdout ppt group stays native same-prompt."
            ),
        },
        "groups": rows,
        "counts": {
            "total_groups": len(rows),
            "train_groups": sum(row["split"] == "train" for row in rows),
            "holdout_groups": sum(row["split"] == "holdout" for row in rows),
            "complete_groups": sum(
                row["completeness"] == "full" for row in rows
            ),
            "partial_groups": sum(
                row["completeness"] == "partial" for row in rows
            ),
            "train_complete_groups": train_complete,
            "pool_filled_groups": sum(
                row.get("completion_mode") == "pool_fill_dev" for row in rows
            ),
        },
    }

    output = _resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report_path = _resolve_path(args.completion_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "manifest": str(output),
                "counts": payload["counts"],
                "completion": payload["completion"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "train_complete_groups": train_complete,
                "pool_filled_groups": payload["counts"]["pool_filled_groups"],
                "completion_enabled": bool(args.complete_from_pools),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Wrote {output}")
    print(f"Completion report: {report_path}")
    if train_complete < args.min_complete_train:
        raise SystemExit(
            f"train_complete_groups={train_complete} < "
            f"min_complete_train={args.min_complete_train}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
