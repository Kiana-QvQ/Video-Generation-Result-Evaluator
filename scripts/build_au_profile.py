from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.au_compliance import DEFAULT_AU_IDS, fit_au_profile, load_au_table
from evaluator.paths import project_path


def _find_au_file(au_root: Path, relative_path: str) -> Path | None:
    relative = Path(relative_path)
    candidates = [
        au_root / relative.with_suffix(".csv"),
        au_root / relative.with_suffix(".tsv"),
        au_root / relative.parent / relative.stem / "au.csv",
    ]
    return next((path for path in candidates if path.is_file()), None)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Wang Xing AU personal-expression distributions."
    )
    parser.add_argument(
        "--manifest",
        default="data/video/expression_reference_manifest.json",
    )
    parser.add_argument("--au-root", required=True)
    parser.add_argument(
        "--output",
        default="data/video/wangxing_au_profile.json",
    )
    args = parser.parse_args()

    manifest = json.loads(
        project_path(args.manifest).read_text(encoding="utf-8-sig")
    )
    au_root = project_path(args.au_root)
    output = project_path(args.output)
    labeled_sequences: list[tuple[str, object]] = []
    missing: list[str] = []
    for record in manifest["records"]:
        if (
            not record.get("phase1_usable")
            or not record.get("is_emotion")
        ):
            continue
        au_path = _find_au_file(au_root, record["relative_path"])
        if au_path is None:
            missing.append(record["relative_path"])
            continue
        sequence, _, _ = load_au_table(au_path, DEFAULT_AU_IDS)
        labeled_sequences.append(
            (record["expression_class"], sequence)
        )

    if missing:
        print(f"Missing AU files: {len(missing)}")
        for path in missing[:20]:
            print(f"  {path}")
    if not labeled_sequences:
        raise SystemExit(
            "No labeled AU files found. Run a mature AU extractor first."
        )

    profile = fit_au_profile(labeled_sequences, output)
    print(json.dumps({
        "classes": {
            name: model["sample_count"]
            for name, model in profile["classes"].items()
        },
        "output": output.relative_to(project_path("." ).resolve()).as_posix(),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
