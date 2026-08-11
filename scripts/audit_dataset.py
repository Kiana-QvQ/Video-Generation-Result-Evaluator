from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.wangxing.expression_dataset import build_expression_manifest  # noqa: E402


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def audit_au(video_root: Path, au_root: Path) -> dict[str, object]:
    videos = sorted(video_root.rglob("*.mp4"))
    missing_csv: list[str] = []
    for video in videos:
        relative = video.relative_to(video_root).with_suffix(".csv")
        if not (au_root / relative).is_file():
            missing_csv.append(relative.as_posix())
    return {
        "video_root": str(video_root),
        "au_root": str(au_root),
        "video_count": len(videos),
        "csv_count": len(list(au_root.rglob("*.csv"))),
        "missing_csv_count": len(missing_csv),
        "missing_csv": missing_csv,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit dataset manifests and AU/video completeness."
    )
    parser.add_argument("--expression-root", default="data/video")
    parser.add_argument("--au-video-root", default="data/MD_CL")
    parser.add_argument("--au-root", default="data/au/MD_CL")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    expression_root = (PROJECT_ROOT / args.expression_root).resolve()
    expression = build_expression_manifest(expression_root)
    au = audit_au(
        (PROJECT_ROOT / args.au_video_root).resolve(),
        (PROJECT_ROOT / args.au_root).resolve(),
    )
    report = {
        "schema_version": "dataset_audit_v1",
        "expression": {
            "manifest_rows": expression["source"]["manifest_rows"],
            "actual_video_rows": expression["source"]["actual_video_rows"],
            "missing_video_rows": expression["source"]["missing_video_rows"],
            "usable_rows": expression["source"]["usable_rows"],
        },
        "au": au,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and (
        report["expression"]["missing_video_rows"]
        or report["au"]["missing_csv_count"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
