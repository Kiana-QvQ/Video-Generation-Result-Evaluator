from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.paths import project_path
from evaluator.wangxing_specialization import (
    build_expression_profile,
    score_expression_profile,
)
from scripts.evaluate_generated_video import _run_extraction


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Seedance facial features and create auditable "
            "expression pseudo-labels."
        )
    )
    parser.add_argument("--video-root", default="data/WangXing_Seedance")
    parser.add_argument("--au-root", default="data/au/WangXing_Seedance")
    parser.add_argument(
        "--expression-profile",
        default="data/au/wangxing_expression_profile.json",
    )
    parser.add_argument(
        "--output",
        default="data/au/WangXing_Seedance/pseudo_expression_manifest.json",
    )
    parser.add_argument("--cache-root", default="outputs/au_cache")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--min-valid-frame-ratio", type=float, default=0.50)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def _stable_au_path(
    video_path: Path,
    video_root: Path,
    au_root: Path,
) -> Path:
    relative = video_path.relative_to(video_root).with_suffix(".csv")
    return au_root / relative


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.min_score < 0.0 or args.min_score > 1.0:
        raise SystemExit("--min-score must be between 0 and 1.")
    if args.min_margin < 0.0 or args.min_margin > 1.0:
        raise SystemExit("--min-margin must be between 0 and 1.")
    if args.min_valid_frame_ratio < 0.0 or args.min_valid_frame_ratio > 1.0:
        raise SystemExit("--min-valid-frame-ratio must be between 0 and 1.")

    video_root = project_path(args.video_root)
    au_root = project_path(args.au_root)
    expression_profile_path = project_path(args.expression_profile)
    output_path = project_path(args.output)
    cache_root = project_path(args.cache_root)
    videos = [
        path
        for path in sorted(video_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]
    if args.max_videos is not None:
        videos = videos[: max(0, args.max_videos)]
    if not videos:
        raise SystemExit(f"No videos found below {video_root}.")

    if not expression_profile_path.is_file():
        print("Expression profile is missing; building the real-only baseline.")
        build_expression_profile(
            project_path("data/au/MD_CL"),
            expression_profile_path,
        )
    profile = json.loads(
        expression_profile_path.read_text(encoding="utf-8-sig")
    )

    records: list[dict[str, object]] = []
    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video_path.name}", flush=True)
        cached_au = _run_extraction(
            video_path,
            au_root / "_extraction",
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            force=args.force,
            cache_root=cache_root,
            cache_namespace="wangxing_seedance_expression_v1",
        )
        stable_au = _stable_au_path(video_path, video_root, au_root)
        stable_au.parent.mkdir(parents=True, exist_ok=True)
        if cached_au.resolve() != stable_au.resolve():
            shutil.copy2(cached_au, stable_au)

        try:
            scored = score_expression_profile(stable_au, profile)
        except (OSError, ValueError, RuntimeError) as exc:
            records.append(
                {
                    "video_path": str(video_path),
                    "au_path": str(stable_au),
                    "source_type": "generated_wangxing",
                    "label_status": "unknown",
                    "pseudo_label": None,
                    "error": str(exc),
                }
            )
            continue

        top_profiles = scored.get("top_profiles", [])
        top = top_profiles[0] if top_profiles else {}
        second = top_profiles[1] if len(top_profiles) > 1 else {}
        score = float(top.get("score_0_1", 0.0) or 0.0)
        margin = float(
            scored.get("margin_0_1", top.get("score_0_1", 0.0))
            or 0.0
        )
        quality = scored.get("quality", {})
        valid_ratio = float(
            quality.get("valid_frame_ratio", 0.0)
            if isinstance(quality, dict)
            else 0.0
        )
        if valid_ratio < args.min_valid_frame_ratio:
            label_status = "unknown"
        elif score < args.min_score:
            label_status = "low_confidence"
        elif margin < args.min_margin:
            label_status = "ambiguous"
        else:
            label_status = "high_confidence"

        records.append(
            {
                "video_path": str(video_path),
                "au_path": str(stable_au),
                "source_type": "generated_wangxing",
                "pseudo_label": top.get("class"),
                "label_status": label_status,
                "use_for_training": label_status == "high_confidence",
                "confidence_0_1": score,
                "compatibility_0_1": scored.get("compatibility_0_1"),
                "margin_0_1": margin,
                "valid_frame_ratio": valid_ratio,
                "top_profiles": top_profiles[:2],
                "selected_profile": scored.get("selected_profile"),
                "second_profile": second.get("class"),
                "event_statistics": scored.get("event_statistics"),
            }
        )

    counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("label_status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    class_counts: dict[str, int] = {}
    for record in records:
        if record.get("label_status") != "high_confidence":
            continue
        label = str(record.get("pseudo_label"))
        class_counts[label] = class_counts.get(label, 0) + 1

    payload = {
        "schema_version": "wangxing_seedance_expression_pseudo_labels_v1",
        "source_type": "generated_wangxing",
        "label_method": (
            "nearest real Wang Xing expression support domain using "
            "AU + Face Mesh + pose + temporal features"
        ),
        "thresholds": {
            "min_score": args.min_score,
            "min_margin": args.min_margin,
            "min_valid_frame_ratio": args.min_valid_frame_ratio,
        },
        "summary": {
            "video_count": len(records),
            "status_counts": counts,
            "high_confidence_class_counts": class_counts,
        },
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
