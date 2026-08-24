"""Fit the V5.1 realness calibrator on ppt test1 only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wangxing_project.realness_v5 import (
    fit_isotonic_calibrator,
    write_calibrator,
)
from wangxing_project.v51_runtime import (
    build_feature_row,
    collect_videos,
    extract_au_for_video,
    load_json,
)
from wangxing_project.drive_head_v5 import load_drive_head
from evaluator.modules.core.paths import project_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit V5.1 realness on ppt test1 only."
    )
    parser.add_argument(
        "--ranking-root",
        default=r"C:\Users\zhanghaotian\Desktop\ppt_video",
    )
    parser.add_argument("--fit-group", default="test1")
    parser.add_argument("--holdout-group", default="test2")
    parser.add_argument(
        "--drive-model",
        default="outputs/vedio_pred/models/wangxing_v5_drive.json",
    )
    parser.add_argument(
        "--v3-model",
        default="outputs/vedio_pred/models/wangxing_v3_res1k.pt",
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
        default="outputs/forensics/cache_wangxing_v5_1_ppt",
    )
    parser.add_argument(
        "--au-output-root",
        default="outputs/forensics/cache_wangxing_v5_1_ppt/au",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wangxing-device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="outputs/forensics/wangxing_v5_realness_calibrator.json",
    )
    args = parser.parse_args(argv)

    ranking_root = Path(args.ranking_root).expanduser().resolve()
    if not ranking_root.is_dir():
        raise SystemExit(f"Ranking root not found: {ranking_root}")
    videos = collect_videos(ranking_root, args.fit_group)
    if len(videos) < 4:
        raise SystemExit(
            f"{args.fit_group} has fewer than four complete videos."
        )
    # Explicitly touch the holdout inventory, but never use it for fitting.
    holdout_videos = collect_videos(ranking_root, args.holdout_group)
    if len(holdout_videos) < 4:
        raise SystemExit(
            f"{args.holdout_group} has fewer than four complete videos."
        )

    source_profile = load_json(args.source_profile)
    forensics_profile = load_json(args.forensics_profile)
    v3_model = project_path(args.v3_model)
    drive_model = load_drive_head(project_path(args.drive_model))
    cache_dir = project_path(args.cache_dir)
    au_root = project_path(args.au_output_root)
    fit_rows: list[dict[str, Any]] = []
    for index, (video, label) in enumerate(videos, start=1):
        print(
            f"[V5.1 calibrate] {args.fit_group} "
            f"{index}/{len(videos)} {video.name}",
            flush=True,
        )
        au = extract_au_for_video(
            video=video,
            au_output_root=au_root,
            cache_dir=cache_dir,
            device=args.wangxing_device,
        )
        fit_rows.append(
            build_feature_row(
                video=video,
                label=label,
                group=args.fit_group,
                au_path=au,
                v3_model=v3_model,
                drive_model=drive_model,
                drive_cache=cache_dir,
                source_profile=source_profile,
                forensics_profile=forensics_profile,
                device=args.device,
                wangxing_device=args.wangxing_device,
                realness_enabled=False,
            )
        )

    calibrator = fit_isotonic_calibrator(
        fit_rows,
        fit_split=args.fit_group,
        holdout_split=args.holdout_group,
        seed=args.seed,
    )
    calibrator["fit_rows"] = [
        {
            "video": row["video"],
            "group": row["group"],
            "label": row["label"],
            "z_raw": row["realness"]["z_raw"],
            "features": row["realness"]["features"],
        }
        for row in fit_rows
    ]
    output = write_calibrator(args.output, calibrator)
    print(json.dumps(
        {
            "schema_version": calibrator["schema_version"],
            "fit_split": calibrator["fit_split"],
            "holdout_split": calibrator["holdout_split"],
            "fit_count": calibrator["fit_count"],
            "feature_names": calibrator["feature_names"],
            "forbidden_features": calibrator["forbidden_features"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
