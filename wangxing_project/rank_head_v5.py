"""V5 ranking-policy adapter.

The current repository has only a small ``ppt_video`` ranking development
set.  This module therefore creates an explicitly disabled policy until the
minimum grouped-query gate is met, instead of silently overfitting eight
videos.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from wangxing_project.cascade_v5 import V5_SCHEMA

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


def ranking_inventory(root: str | Path) -> dict[str, Any]:
    input_root = Path(root).expanduser().resolve()
    groups: dict[str, set[str]] = {}
    for video in sorted(input_root.rglob("*.mp4")):
        label = _label(video)
        if label is None:
            continue
        groups.setdefault(video.parent.name, set()).add(label)
    complete = [
        group
        for group, labels in groups.items()
        if set(ORDER).issubset(labels)
    ]
    return {
        "root": str(input_root),
        "group_count": len(groups),
        "complete_query_count": len(complete),
        "complete_groups": sorted(complete),
        "labels_seen": sorted({label for labels in groups.values() for label in labels}),
    }


def disabled_rank_policy(
    *,
    root: str | Path,
    inventory: dict[str, Any],
    minimum_queries: int,
) -> dict[str, Any]:
    return {
        "schema_version": V5_SCHEMA,
        "decision_source": "v3_frozen",
        "development_only": True,
        "expected_order": list(ORDER),
        "ordering_satisfied": False,
        "usable_for_runtime": False,
        "rank_model": {"enabled": False},
        "minimum_queries": int(minimum_queries),
        "ranking_inventory": inventory,
        "disabled_reason": (
            "insufficient_complete_grouped_queries"
            if inventory["complete_query_count"] < minimum_queries
            else "ordering_not_validated"
        ),
        "development_root": str(Path(root).expanduser().resolve()),
        "test_sets_excluded": [
            "data/test/single_video",
            "data/test/wangxing_32x32",
        ],
    }


def build_rank_policy(
    *,
    root: str | Path,
    output: str | Path,
    minimum_queries: int = 30,
    forensics_profile: str | Path | None = None,
    source_profile: str | Path | None = None,
    expression_only: bool = True,
) -> dict[str, Any]:
    inventory = ranking_inventory(root)
    output_path = Path(output).expanduser().resolve()
    if inventory["complete_query_count"] < minimum_queries:
        payload = disabled_rank_policy(
            root=root,
            inventory=inventory,
            minimum_queries=minimum_queries,
        )
    else:
        project_root = Path(__file__).resolve().parents[1]
        legacy_output = output_path.with_name(output_path.stem + "_legacy.json")
        command = [
            sys.executable,
            str(project_root / "scripts" / "web_forensics" / "train_wangxing_weighted_policy.py"),
            "--input-root",
            str(Path(root).expanduser().resolve()),
            "--policy-output",
            str(legacy_output),
        ]
        if forensics_profile:
            command.extend(["--forensics-profile", str(forensics_profile)])
        if source_profile:
            command.extend(["--source-profile", str(source_profile)])
        if expression_only:
            command.append("--expression-only")
        completed = subprocess.run(command, cwd=project_root, check=False)
        if completed.returncode != 0 or not legacy_output.is_file():
            payload = disabled_rank_policy(
                root=root,
                inventory=inventory,
                minimum_queries=minimum_queries,
            )
            payload["disabled_reason"] = "legacy_rank_fit_failed"
        else:
            legacy = json.loads(legacy_output.read_text(encoding="utf-8-sig"))
            ordering = bool(legacy.get("ordering_satisfied"))
            payload = {
                "schema_version": V5_SCHEMA,
                "decision_source": "v3_frozen",
                "development_only": True,
                "expected_order": list(ORDER),
                "ordering_satisfied": ordering,
                "usable_for_runtime": ordering,
                "class_ordering_satisfied": bool(
                    legacy.get("class_ordering_satisfied")
                ),
                "pairwise_ordering_rate": legacy.get(
                    "pairwise_ordering_rate"
                ),
                "class_mean_scores_0_1": legacy.get(
                    "class_mean_scores_0_1", {}
                ),
                "rank_model": legacy.get("rank_model", {}),
                "ranking_inventory": inventory,
                "minimum_queries": int(minimum_queries),
                "development_root": str(Path(root).expanduser().resolve()),
                "test_sets_excluded": [
                    "data/test/single_video",
                    "data/test/wangxing_32x32",
                ],
                "disabled_reason": None if ordering else "ordering_not_satisfied",
            }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload
