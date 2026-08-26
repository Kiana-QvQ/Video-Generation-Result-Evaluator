"""Convert a V5.2 ranking manifest to the V5.3 explicit-role manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path


def _convert_group(source_group: dict[str, Any]) -> dict[str, Any]:
    videos = source_group.get("videos") or {}
    return {
        "group_id": source_group.get("group_id"),
        "split": source_group.get("split"),
        "matching_key": source_group.get("matching_key")
        or source_group.get("group_id"),
        "completeness": source_group.get("completeness"),
        "runtime_role_source": "manifest_explicit",
        "same_prompt_matched": bool(
            source_group.get("same_prompt_matched")
        ),
        "videos": {
            role: videos.get(role)
            for role in ("real", "lora", "seedance", "multiref")
            if videos.get(role)
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--runtime-mode",
        choices=("web_regression", "offline_eval", "public_single"),
        default="web_regression",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Keep only completeness=full groups with four roles.",
    )
    parser.add_argument(
        "--same-prompt-only",
        action="store_true",
        help="Keep only same_prompt_matched=true groups (formal ORDER).",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        help="Optional split filter (train/holdout). Repeatable.",
    )
    args = parser.parse_args(argv)
    source_path = project_path(args.input)
    source = json.loads(source_path.read_text(encoding="utf-8-sig"))
    groups: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    allowed_splits = {str(item) for item in args.split if str(item).strip()}
    for source_group in source.get("groups") or []:
        group = _convert_group(source_group)
        roles = set((group.get("videos") or {}).keys())
        reasons: list[str] = []
        if allowed_splits and str(group.get("split")) not in allowed_splits:
            reasons.append("split_filtered")
        if args.full_only:
            if group.get("completeness") != "full":
                reasons.append("not_full")
            if roles != {"real", "lora", "seedance", "multiref"}:
                reasons.append("missing_roles")
        if args.same_prompt_only and not group.get("same_prompt_matched"):
            reasons.append("not_same_prompt")
        if reasons:
            skipped.append(
                {
                    "group_id": group.get("group_id"),
                    "reasons": reasons,
                }
            )
            continue
        groups.append(group)
    if not groups:
        raise SystemExit("No groups left after filters; refuse empty manifest.")
    result = {
        "schema_version": "wangxing_v5_3_runtime_manifest_v1",
        "compatible_with": "wangxing_v5_2_ranking_manifest_v1",
        "runtime_mode": args.runtime_mode,
        "runtime_defaults": {
            "label_inference": "forbidden",
            "role_anchor": "forbidden",
            "content_gate": "off",
            "group_rescore": "off",
        },
        "source_manifest": str(source_path),
        "filters": {
            "full_only": bool(args.full_only),
            "same_prompt_only": bool(args.same_prompt_only),
            "splits": sorted(allowed_splits),
        },
        "skipped_groups": skipped,
        "groups": groups,
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "runtime_mode": args.runtime_mode,
                "group_count": len(groups),
                "skipped_count": len(skipped),
                "group_ids": [g.get("group_id") for g in groups],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
