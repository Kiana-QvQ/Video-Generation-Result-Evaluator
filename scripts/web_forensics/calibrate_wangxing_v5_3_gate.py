"""Fit the V5.3 public content-gate threshold from train rows only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from numpy import percentile

from evaluator.modules.core.paths import project_path
from wangxing_project.drive_head_v5 import load_drive_head
from wangxing_project.realness_v5 import load_calibrator
from wangxing_project.v51_runtime import (
    build_feature_row,
    extract_au_for_video,
    load_json,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if isinstance(rows, list):
        return rows
    holdout = payload.get("holdout")
    if isinstance(holdout, dict) and isinstance(holdout.get("rows"), list):
        # Never use holdout for fitting; return empty so caller fails loudly.
        return []
    raise SystemExit("--rows must contain a top-level rows array")


def _score_train_rows_from_ranking_manifest(
    *,
    ranking_manifest: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    manifest = _load(ranking_manifest)
    profiles = load_json(args.forensics_profile)
    source_profile = load_json(args.source_profile)
    calibrator = load_calibrator(project_path(args.calibrator))
    if calibrator is None:
        raise SystemExit("Missing or invalid V5.1 calibrator.")
    v3_model = project_path(args.v3_model)
    drive_model = load_drive_head(project_path(args.drive_model))
    cache_dir = project_path(args.cache_dir)
    au_root = project_path(args.au_output_root)
    rows: list[dict[str, Any]] = []
    for group in manifest.get("groups") or []:
        if str(group.get("split") or "") != "train":
            continue
        group_id = str(group.get("group_id") or "")
        for label, video_value in (group.get("videos") or {}).items():
            if label not in {"lora", "seedance", "multiref"}:
                continue
            if not video_value:
                continue
            video = Path(str(video_value)).expanduser().resolve()
            if not video.is_file():
                raise SystemExit(f"Train video missing: {video}")
            au = extract_au_for_video(
                video=video,
                au_output_root=au_root,
                cache_dir=cache_dir,
                device=args.wangxing_device,
            )
            row = build_feature_row(
                video=video,
                label=label,
                group=group_id,
                au_path=au,
                v3_model=v3_model,
                drive_model=drive_model,
                drive_cache=cache_dir,
                source_profile=source_profile,
                forensics_profile=profiles,
                device=args.device,
                wangxing_device=args.wangxing_device,
                calibrator=calibrator,
                realness_enabled=True,
            )
            row["split"] = "train"
            row["group_id"] = group_id
            rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        help="JSON with top-level rows (train only used for fit)",
    )
    parser.add_argument(
        "--ranking-manifest",
        help="V5.2 ranking manifest; scores train AI roles for gate fit",
    )
    parser.add_argument(
        "--output",
        default="outputs/forensics/wangxing_v5_3_display_gate.json",
    )
    parser.add_argument(
        "--rows-output",
        default="outputs/forensics/wangxing_v5_3_gate_train_rows.json",
    )
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument(
        "--calibrator",
        default="outputs/forensics/wangxing_v5_realness_calibrator.json",
    )
    parser.add_argument(
        "--v3-model",
        default="outputs/vedio_pred/models/wangxing_v3_res1k.pt",
    )
    parser.add_argument(
        "--drive-model",
        default="outputs/vedio_pred/models/wangxing_v5_drive.json",
    )
    parser.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles_web_v3_test_excluded.json",
    )
    parser.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_web_v3_test_excluded.json",
    )
    parser.add_argument(
        "--cache-dir",
        default="outputs/forensics/cache_wangxing_v5_2",
    )
    parser.add_argument(
        "--au-output-root",
        default="outputs/forensics/cache_wangxing_v5_2/au",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wangxing-device", default="cuda")
    args = parser.parse_args(argv)

    if args.ranking_manifest:
        rows = _score_train_rows_from_ranking_manifest(
            ranking_manifest=project_path(args.ranking_manifest),
            args=args,
        )
        rows_path = project_path(args.rows_output)
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        rows_path.write_text(
            json.dumps(
                {
                    "schema_version": "wangxing_v5_3_gate_train_rows_v1",
                    "source_manifest": str(
                        project_path(args.ranking_manifest)
                    ),
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    elif args.rows:
        payload = _load(Path(args.rows).expanduser().resolve())
        rows = _rows_from_payload(payload)
    else:
        raise SystemExit("Provide --rows or --ranking-manifest")

    values: dict[str, list[float]] = {
        "lora": [],
        "seedance": [],
        "multiref": [],
    }
    for row in rows:
        if str(row.get("split") or row.get("runtime_split") or "train") != "train":
            continue
        label = str(row.get("label") or "")
        value = (row.get("realness") or {}).get("s_realness")
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and label in values:
            values[label].append(parsed)
    if any(not values[label] for label in values):
        raise SystemExit(
            "Each AI role needs at least one finite train s_realness value: "
            + json.dumps({k: len(v) for k, v in values.items()})
        )
    p95 = {label: float(percentile(items, 95)) for label, items in values.items()}
    threshold = max(p95.values()) + float(args.margin)
    result = {
        "schema_version": "wangxing_v5_3_display_gate_v1",
        "development_only": True,
        "content_gate": {
            "enabled": True,
            "T_high": min(1.0, threshold),
            "T_rank_cap": None,
            "margin": float(args.margin),
            "source": "ranking_train_only",
            "role_p95": p95,
            "sample_counts": {
                label: len(items) for label, items in values.items()
            },
        },
        "holdout_used_for_fit": False,
        "test_sets_excluded": [
            "data/test/single_video",
            "data/test/wangxing_32x32",
            "ppt_test2",
        ],
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
