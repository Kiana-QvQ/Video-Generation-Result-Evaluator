from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.au_compliance import fit_leakage_classifier


def _csv_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.csv"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit AU leakage classifier: label 0=target real, "
            "label 1=other/AI leakage or abnormal."
        )
    )
    parser.add_argument("--positive-root", required=True)
    parser.add_argument("--negative-root", required=True)
    parser.add_argument(
        "--output",
        default="data/au/au_leakage_classifier.json",
    )
    args = parser.parse_args()

    positive = _csv_files(Path(args.positive_root))
    negative = _csv_files(Path(args.negative_root))
    if not positive:
        raise SystemExit("No positive AU CSV files found.")
    if not negative:
        raise SystemExit("No negative AU CSV files found.")
    model = fit_leakage_classifier(
        positive,
        negative,
        args.output,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "positive_count": model["positive_count"],
                "negative_count": model["negative_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
