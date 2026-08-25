"""Fit the V5.2 grouped linear pairwise RankHead."""

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
from wangxing_project.drive_head_v5 import load_drive_head
from wangxing_project.rank_head_v52 import (
    DEFAULT_MIN_COMPLETE_GROUPS_FIT,
    DEFAULT_MIN_COMPLETE_GROUPS_RUNTIME,
    DEFAULT_MIN_PAIRS_FIT,
    DEFAULT_MIN_PAIRWISE_RUNTIME,
    fit_rank_policy,
    write_rank_policy,
)
from wangxing_project.realness_v5 import load_calibrator
from wangxing_project.v51_runtime import (
    build_feature_row,
    extract_au_for_video,
    load_json,
)

def _manifest_rows(
    manifest: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    profiles = load_json(args.forensics_profile)
    source_profile = load_json(args.source_profile)
    calibrator = load_calibrator(project_path(args.calibrator))
    if calibrator is None:
        raise SystemExit("V5.1 calibrator is missing or schema-invalid.")
    v3_model = project_path(args.v3_model)
    drive_model = load_drive_head(project_path(args.drive_model))
    cache_dir = project_path(args.cache_dir)
    au_root = project_path(args.au_output_root)
    rows: list[dict[str, Any]] = []
    for group in manifest.get("groups") or []:
        if str(group.get("split")) != "train":
            continue
        group_id = str(group.get("group_id"))
        for label, video_value in (group.get("videos") or {}).items():
            if not video_value:
                continue
            video = Path(str(video_value)).expanduser().resolve()
            if not video.is_file():
                raise SystemExit(f"Ranking video not found: {video}")
            if label not in {"real", "lora", "seedance", "multiref"}:
                raise SystemExit(f"Unknown ranking label: {label}")
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
            row["group_id"] = group_id
            rows.append(row)
            print(
                f"[V5.2 rank features] {group_id}/{label}",
                flush=True,
            )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train V5.2 linear pairwise RankHead on train groups only."
    )
    parser.add_argument("--manifest", required=True)
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
    parser.add_argument("--C", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-complete-groups-fit",
        type=int,
        default=DEFAULT_MIN_COMPLETE_GROUPS_FIT,
    )
    parser.add_argument(
        "--min-pairs-fit",
        type=int,
        default=DEFAULT_MIN_PAIRS_FIT,
    )
    parser.add_argument(
        "--min-complete-groups-runtime",
        type=int,
        default=DEFAULT_MIN_COMPLETE_GROUPS_RUNTIME,
    )
    parser.add_argument(
        "--min-pairwise",
        type=float,
        default=DEFAULT_MIN_PAIRWISE_RUNTIME,
    )
    parser.add_argument(
        "--output",
        default="outputs/forensics/wangxing_v5_2_rank_policy.json",
    )
    args = parser.parse_args(argv)

    manifest_path = project_path(args.manifest)
    manifest = load_json(manifest_path)
    rows = _manifest_rows(manifest, args=args)
    fit_groups = [
        str(group["group_id"])
        for group in manifest.get("groups") or []
        if str(group.get("split")) == "train"
    ]
    holdout_groups = [
        str(group["group_id"])
        for group in manifest.get("groups") or []
        if str(group.get("split")) == "holdout"
    ]
    policy = fit_rank_policy(
        rows=rows,
        fit_groups=fit_groups,
        holdout_groups=holdout_groups,
        seed=args.seed,
        C=args.C,
        min_complete_groups_fit=args.min_complete_groups_fit,
        min_pairs_fit=args.min_pairs_fit,
        min_complete_groups_runtime=args.min_complete_groups_runtime,
        min_pairwise_runtime=args.min_pairwise,
    )
    policy["manifest"] = str(manifest_path)
    policy["feature_cache"] = str(project_path(args.cache_dir))
    policy["train_row_count"] = len(rows)
    output = write_rank_policy(args.output, policy)
    print(json.dumps(
        {
            "schema_version": policy["schema_version"],
            "fit_groups": policy["fit_groups"],
            "holdout_groups": policy["holdout_groups"],
            "pair_count_fit": policy["pair_count_fit"],
            "disabled_reason": policy["disabled_reason"],
            "usable_for_runtime": policy["usable_for_runtime"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
