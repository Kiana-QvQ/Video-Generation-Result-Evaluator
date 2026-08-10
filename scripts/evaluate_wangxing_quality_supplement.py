from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.paths import project_path
from evaluator.wangxing_quality_supplement import evaluate_quality_supplement
from scripts.evaluate_generated_video import _run_extraction


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Additive Wang Xing facial-expression / texture quality "
            "supplement. Does not modify ordinary scores or web UI."
        )
    )
    parser.add_argument("--generated-video", required=True)
    parser.add_argument(
        "--au-csv",
        help="Reuse an existing AU CSV and skip LibreFace extraction.",
    )
    parser.add_argument(
        "--expression-profile",
        default="data/au/wangxing_expression_profile.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/wangxing_quality_supplement.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/au_cache/wangxing_quality_supplement",
    )
    parser.add_argument("--cache-root", default="outputs/au_cache")
    parser.add_argument("--expected-class")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-texture-frames", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    video = project_path(args.generated_video)
    if not video.is_file():
        raise SystemExit(f"generated video not found: {video}")
    expression_profile = project_path(args.expression_profile)
    if not expression_profile.is_file():
        raise SystemExit(f"expression profile not found: {expression_profile}")

    if args.au_csv:
        au_csv = project_path(args.au_csv)
        if not au_csv.is_file():
            raise SystemExit(f"AU CSV not found: {au_csv}")
    else:
        au_csv = _run_extraction(
            video,
            project_path(args.output_root),
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            force=args.force,
            cache_root=project_path(args.cache_root),
            cache_namespace="wangxing_specialization_v1",
        )

    result = evaluate_quality_supplement(
        au_csv=au_csv,
        video_path=video,
        expression_profile_path=expression_profile,
        expected_class=args.expected_class,
        max_texture_frames=max(4, int(args.max_texture_frames)),
    )
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    facial = result["facial_expression_muscle"]
    texture = result["texture_detail_quality"]
    print(f"Wrote {output}")
    print(
        "facial_expression_muscle="
        f"{facial.get('score_100')} "
        "texture_detail_quality="
        f"{texture.get('score_100')}"
    )
    metrics = facial.get("metrics") or {}
    labels = facial.get("metric_labels") or {}
    for key, label in labels.items():
        value = metrics.get(key)
        rendered = "--" if value is None else f"{100.0 * float(value):.1f}"
        print(f"  {label}: {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
