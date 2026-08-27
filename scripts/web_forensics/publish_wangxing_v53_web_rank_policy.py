"""Publish the best validated RankHead policy for the public web UI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wangxing_project.web_forensics_display import (  # noqa: E402
    WEB_V53_RANK_POLICY,
    publish_web_rank_policy,
    resolve_web_rank_policy_path,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="outputs/forensics/wangxing_v5_2_rank_policy_overnight.json",
        help="Validated rank policy to copy into the stable web runtime path.",
    )
    args = parser.parse_args(argv)
    target = publish_web_rank_policy(args.source)
    print(
        {
            "published_to": str(target),
            "web_runtime_path": WEB_V53_RANK_POLICY,
            "resolved_web_policy": str(resolve_web_rank_policy_path()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
