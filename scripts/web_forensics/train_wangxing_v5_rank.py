"""Build the guarded V5 ranking policy for the web/PT cascade."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wangxing_project.rank_head_v5 import build_rank_policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit or explicitly disable the guarded V5 rank policy."
    )
    parser.add_argument(
        "--ranking-root",
        default=r"C:\Users\zhanghaotian\Desktop\ppt_video",
    )
    parser.add_argument(
        "--output",
        default="outputs/forensics/wangxing_authenticity_policy_v5.json",
    )
    parser.add_argument("--min-queries", type=int, default=30)
    parser.add_argument(
        "--forensics-profile",
        default="outputs/forensics/forensics_profiles_web_v3_test_excluded.json",
    )
    parser.add_argument(
        "--source-profile",
        default="outputs/forensics/wangxing_source_profile_web_v3_test_excluded.json",
    )
    parser.add_argument("--expression-only", action="store_true")
    args = parser.parse_args(argv)
    payload = build_rank_policy(
        root=args.ranking_root,
        output=args.output,
        minimum_queries=args.min_queries,
        forensics_profile=args.forensics_profile,
        source_profile=args.source_profile,
        expression_only=args.expression_only,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {Path(args.output).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
