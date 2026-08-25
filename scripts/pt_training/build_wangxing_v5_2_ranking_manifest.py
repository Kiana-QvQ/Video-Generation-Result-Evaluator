"""Build a grouped V5.2 ranking manifest from ppt and optional LTX videos."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

ORDER = ("real", "lora", "seedance", "multiref")


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build V5.2 grouped ranking manifest; no final test inputs."
    )
    parser.add_argument(
        "--ppt-root",
        default=r"C:\Users\zhanghaotian\Desktop\ppt_video",
    )
    parser.add_argument(
        "--ltx-root",
        default="data/LTX",
    )
    parser.add_argument("--holdout-group", default="ppt_test2")
    parser.add_argument("--holdout-group-count", type=int, default=1)
    parser.add_argument(
        "--output",
        default="data/ranking/wangxing_v5_2/manifest.json",
    )
    args = parser.parse_args(argv)

    ppt_groups = _collect_root(Path(args.ppt_root).expanduser().resolve(), "ppt")
    ltx_groups = _collect_root(Path(args.ltx_root).expanduser().resolve(), "ltx")
    merged = {**ppt_groups, **ltx_groups}
    if not merged:
        raise SystemExit("No labeled ranking videos were found.")

    full_groups = [
        group for group, videos in merged.items()
        if all(role in videos for role in ORDER)
    ]
    if args.holdout_group in merged:
        holdout = [args.holdout_group]
    else:
        holdout = sorted(full_groups)[-max(1, args.holdout_group_count):]
    rows: list[dict[str, Any]] = []
    for group_id in sorted(merged):
        videos = merged[group_id]
        split = "holdout" if group_id in holdout else "train"
        rows.append(
            {
                "group_id": group_id,
                "split": split,
                "matching_key": group_id,
                "completeness": (
                    "full"
                    if all(role in videos for role in ORDER)
                    else "partial"
                ),
                "videos": {
                    role: videos.get(role)
                    for role in ORDER
                },
            }
        )
    final_markers = (
        "data\\test\\single_video",
        "data/test/single_video",
        "data\\test\\wangxing_32x32",
        "data/test/wangxing_32x32",
    )
    for row in rows:
        for path in (row["videos"] or {}).values():
            if path and any(marker.casefold() in path.casefold() for marker in final_markers):
                raise SystemExit(f"Final-test video entered ranking manifest: {path}")
    payload = {
        "schema_version": "wangxing_v5_2_ranking_manifest_v1",
        "order_prior": list(ORDER),
        "forbidden_final_test": [
            "data/test/single_video",
            "data/test/wangxing_32x32",
        ],
        "source_roots": {
            "ppt": str(Path(args.ppt_root).expanduser().resolve()),
            "ltx": str(Path(args.ltx_root).expanduser().resolve()),
        },
        "groups": rows,
        "counts": {
            "total_groups": len(rows),
            "train_groups": sum(row["split"] == "train" for row in rows),
            "holdout_groups": sum(row["split"] == "holdout" for row in rows),
            "complete_groups": sum(row["completeness"] == "full" for row in rows),
            "partial_groups": sum(row["completeness"] == "partial" for row in rows),
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
