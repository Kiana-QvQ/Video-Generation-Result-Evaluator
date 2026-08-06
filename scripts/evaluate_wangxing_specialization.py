from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.paths import project_path
from evaluator.wangxing_specialization import (
    evaluate_specialization,
)
from scripts.evaluate_generated_video import _run_extraction


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Wang Xing identity and facial-expression compatibility."
    )
    parser.add_argument("--generated-video", required=True)
    parser.add_argument(
        "--identity-profile",
        default="data/au/wangxing_identity_profile.json",
    )
    parser.add_argument(
        "--expression-profile",
        default="data/au/wangxing_expression_profile.json",
    )
    parser.add_argument(
        "--output-root",
        default="data/au/generated_specialization",
    )
    parser.add_argument("--cache-root", default="outputs/au_cache")
    parser.add_argument("--output")
    parser.add_argument("--expected-class")
    parser.add_argument(
        "--target-image",
        action="append",
        help="Legacy compatibility input; built-in identity profile is used.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--identity-frames", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    del args.target_image
    if args.identity_frames < 3:
        raise SystemExit("--identity-frames must be at least 3.")
    generated_video = project_path(args.generated_video)
    identity_profile = project_path(args.identity_profile)
    expression_profile = project_path(args.expression_profile)
    output_root = project_path(args.output_root)
    cache_root = project_path(args.cache_root)
    output = project_path(args.output) if args.output else None
    for path, label in (
        (generated_video, "generated video"),
        (identity_profile, "identity profile"),
        (expression_profile, "expression profile"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} was not found: {path}")

    generated_au = _run_extraction(
        generated_video,
        output_root,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force=args.force,
        cache_root=cache_root,
        cache_namespace="wangxing_specialization_v1",
    )
    result = evaluate_specialization(
        video_path=generated_video,
        au_path=generated_au,
        identity_profile_path=identity_profile,
        expression_profile_path=expression_profile,
        expected_class=args.expected_class,
        device=args.device,
        max_identity_frames=args.identity_frames,
    )
    result["evaluation_meta"] = {
        "generated_video": str(generated_video),
        "generated_au": str(generated_au),
        "identity_profile": str(identity_profile),
        "expression_profile": str(expression_profile),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
