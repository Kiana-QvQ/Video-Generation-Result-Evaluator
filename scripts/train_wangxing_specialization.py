from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.core.paths import project_path
from evaluator.wangxing.wangxing_specialization import (
    build_expression_profile,
    build_identity_profile,
    build_source_profile,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Wang Xing open-set identity profile and the real "
            "facial-expression support domains."
        )
    )
    parser.add_argument("--real-root", default="data/MD_CL")
    parser.add_argument("--generated-root", default="data/WangXing_Seedance")
    parser.add_argument(
        "--negative-root",
        default="data/negative/ravdess/videos",
        help="Existing automatically labeled non-Wang Xing videos.",
    )
    parser.add_argument(
        "--au-root",
        default="data/au/MD_CL",
        help="AU CSV root for real Wang Xing expression profiles.",
    )
    parser.add_argument(
        "--identity-output",
        default="data/au/wangxing_identity_profile.json",
    )
    parser.add_argument(
        "--expression-output",
        default="data/au/wangxing_expression_profile.json",
    )
    parser.add_argument(
        "--source-profile-output",
        default="data/au/wangxing_source_profile.json",
    )
    parser.add_argument(
        "--seedance-label-manifest",
        default="",
        help=(
            "Optional manifest produced by label_seedance_expressions.py. "
            "Only high-confidence records join expression training."
        ),
    )
    parser.add_argument("--max-pseudo-per-class", type=int, default=40)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--identity-frames",
        type=int,
        default=8,
        help="Uniformly sampled frames per training video.",
    )
    parser.add_argument(
        "--identity-limit",
        type=int,
        help="Optional development limit per source root.",
    )
    parser.add_argument(
        "--skip-identity",
        action="store_true",
        help="Only rebuild the expression profile from existing AU CSVs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.identity_frames < 1:
        raise SystemExit("--identity-frames must be positive.")
    if args.max_pseudo_per_class <= 0:
        raise SystemExit("--max-pseudo-per-class must be positive.")
    pseudo_manifest = (
        project_path(args.seedance_label_manifest)
        if args.seedance_label_manifest
        else None
    )
    expression = build_expression_profile(
        project_path(args.au_root),
        project_path(args.expression_output),
        pseudo_label_manifest=pseudo_manifest,
        max_pseudo_per_class=args.max_pseudo_per_class,
    )
    result: dict[str, object] = {
        "expression_profile": {
            "output": str(project_path(args.expression_output)),
            "class_counts": expression["class_counts"],
            "sample_count": expression["provenance"]["sample_count"],
            "pseudo_sample_count": expression["provenance"][
                "pseudo_sample_count"
            ],
        }
    }
    if pseudo_manifest is not None:
        source = build_source_profile(
            real_au_root=project_path(args.au_root),
            seedance_label_manifest=pseudo_manifest,
            output_path=project_path(args.source_profile_output),
        )
        result["source_profile"] = {
            "output": str(project_path(args.source_profile_output)),
            "sample_counts": source["provenance"]["sample_counts"],
            "failed_count": source["provenance"]["skipped_count"],
        }
    if not args.skip_identity:
        identity = build_identity_profile(
            real_root=project_path(args.real_root),
            generated_root=project_path(args.generated_root),
            negative_root=project_path(args.negative_root),
            output_path=project_path(args.identity_output),
            device=args.device,
            max_frames=args.identity_frames,
            limit=(
                args.identity_limit
                if args.identity_limit is not None and args.identity_limit > 0
                else None
            ),
        )
        result["identity_profile"] = {
            "output": str(project_path(args.identity_output)),
            "backend": identity["backend"],
            "positive_count": identity["calibration"]["positive_count"],
            "negative_count": identity["calibration"]["negative_count"],
            "failed_count": identity["provenance"]["failed_count"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
