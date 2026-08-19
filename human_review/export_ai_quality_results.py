#!/usr/bin/env python3
"""Export aggregate results for the isolated AI quality review dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DB = ROOT_DIR / "data" / "review.sqlite3"
DEFAULT_OUTPUT = (
    ROOT_DIR / "data" / "reports" / "ai_quality_25plus5_v1_summary.json"
)
RATINGS = ("upper", "middle", "lower")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def export_results(
    db_path: Path,
    dataset_id: str,
    output_path: Path,
    csv_path: Path | None = None,
) -> dict[str, Any]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                t.task_id,
                t.case_id,
                t.asset_id,
                t.metadata_json,
                a.original_name,
                v.rating
            FROM quality_tasks t
            JOIN quality_assets a
              ON a.dataset_id = t.dataset_id
             AND a.asset_id = t.asset_id
            LEFT JOIN quality_votes v
              ON v.dataset_id = t.dataset_id
             AND v.task_id = t.task_id
            WHERE t.dataset_id = ?
              AND t.status IN ('ready', 'active')
            ORDER BY t.task_id, v.created_at
            """,
            (dataset_id,),
        ).fetchall()

    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        metadata = json.loads(row["metadata_json"] or "{}")
        item = grouped.setdefault(
            row["task_id"],
            {
                "task_id": row["task_id"],
                "sample_id": metadata.get("sample_id") or row["case_id"],
                "file_name": row["original_name"],
                "cohort": metadata.get("cohort"),
                "source_domain": metadata.get("source_domain"),
                "human_band": metadata.get("human_band"),
                "program_band": metadata.get("program_band"),
                "expression_score": metadata.get("expression_score"),
                "rating_counts": {rating: 0 for rating in RATINGS},
            },
        )
        if row["rating"] in item["rating_counts"]:
            item["rating_counts"][row["rating"]] += 1

    items = []
    for item in grouped.values():
        counts = item["rating_counts"]
        vote_count = sum(counts.values())
        majority = None
        agreement = None
        if vote_count:
            majority = max(RATINGS, key=lambda rating: counts[rating])
            agreement = round(counts[majority] / vote_count, 4)
        item["vote_count"] = vote_count
        item["majority_rating"] = majority
        item["agreement"] = agreement
        items.append(item)

    total_counts = Counter()
    for item in items:
        total_counts.update(
            {
                rating: item["rating_counts"][rating]
                for rating in RATINGS
            }
        )
    rated_items = [item for item in items if item["vote_count"]]
    summary = {
        "dataset_id": dataset_id,
        "generated_at": utc_now(),
        "task_count": len(items),
        "rated_task_count": len(rated_items),
        "total_vote_count": sum(total_counts.values()),
        "rating_counts": dict(total_counts),
        "items": sorted(items, key=lambda item: item["sample_id"]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            fields = [
                "sample_id",
                "file_name",
                "cohort",
                "source_domain",
                "human_band",
                "program_band",
                "expression_score",
                "vote_count",
                "upper",
                "middle",
                "lower",
                "majority_rating",
                "agreement",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in summary["items"]:
                writer.writerow(
                    {
                        "sample_id": item["sample_id"],
                        "file_name": item["file_name"],
                        "cohort": item["cohort"],
                        "source_domain": item["source_domain"],
                        "human_band": item["human_band"],
                        "program_band": item["program_band"],
                        "expression_score": item["expression_score"],
                        "vote_count": item["vote_count"],
                        "upper": item["rating_counts"]["upper"],
                        "middle": item["rating_counts"]["middle"],
                        "lower": item["rating_counts"]["lower"],
                        "majority_rating": item["majority_rating"],
                        "agreement": item["agreement"],
                    }
                )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export aggregate AI quality review results."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--dataset-id",
        default="ai_quality_25plus5_v1",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_results(
        db_path=args.db,
        dataset_id=args.dataset_id,
        output_path=args.output,
        csv_path=args.csv,
    )
    print(
        json.dumps(
            {
                "dataset_id": summary["dataset_id"],
                "task_count": summary["task_count"],
                "rated_task_count": summary["rated_task_count"],
                "total_vote_count": summary["total_vote_count"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
