from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from evaluator.modules.wangxing.wangxing_redesign_check import inspect_wangxing_redesign


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check completion of the Wang Xing specialization redesign."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 1 unless every required check is complete.",
    )
    args = parser.parse_args()
    report = inspect_wangxing_redesign(project_path(args.project_root))
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0 if not args.strict or report["overall_status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
