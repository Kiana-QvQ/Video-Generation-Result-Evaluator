from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.au_compliance import (  # noqa: E402
    DEFAULT_AU_IDS,
    DEFAULT_PRESENCE_AU_IDS,
    fit_au_profile,
    load_au_table,
)
from evaluator.paths import project_path  # noqa: E402


ORIGINAL_CLASS_PREFIXES = {
    "kaixin": "smile",
    "fennu": "anger",
    "jingya": "surprise",
    "kongju": "fear",
    "shengqi": "annoyance",
    "beishang": "sadness",
}


def _class_from_path(path: Path) -> str | None:
    directory = re.sub(r"^cl_", "", path.parent.name.casefold())
    for prefix, class_name in ORIGINAL_CLASS_PREFIXES.items():
        if directory.startswith(prefix):
            return class_name
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the general emotion AU profile from original AU CSV files. "
            "This profile is separate from the Wang Xing personal AU profile."
        )
    )
    parser.add_argument("--au-root", default="data/au/MD_CL")
    parser.add_argument(
        "--output",
        default="data/au/original_emotion_au_profile.json",
    )
    parser.add_argument("--min-samples-per-class", type=int, default=3)
    args = parser.parse_args(argv)
    if args.min_samples_per_class <= 0:
        raise ValueError("--min-samples-per-class must be positive.")

    au_root = project_path(args.au_root)
    output = project_path(args.output)
    labeled_sequences: list[tuple[str, object]] = []
    presence_sequences: list[tuple[str, object]] = []
    skipped: list[str] = []
    for path in sorted(au_root.rglob("*.csv")):
        expression_class = _class_from_path(path)
        if expression_class is None:
            skipped.append(f"{path}: unknown class directory")
            continue
        try:
            sequence, supported, _ = load_au_table(
                path,
                DEFAULT_AU_IDS,
                feature_type="intensity",
                strict=False,
            )
            if not supported:
                skipped.append(f"{path}: no supported intensity AU")
                continue
            labeled_sequences.append((expression_class, sequence))
            try:
                presence, _, _ = load_au_table(
                    path,
                    DEFAULT_PRESENCE_AU_IDS,
                    feature_type="presence",
                    strict=False,
                    intensity_scale=1.0,
                )
            except ValueError:
                continue
            presence_sequences.append((expression_class, presence))
        except (OSError, ValueError) as exc:
            skipped.append(f"{path}: {exc}")

    if not labeled_sequences:
        raise SystemExit(
            f"No original AU CSV files with recognized emotion labels were found "
            f"under {au_root}."
        )

    profile = fit_au_profile(
        labeled_sequences,
        output,
        au_ids=DEFAULT_AU_IDS,
        presence_labeled_sequences=presence_sequences,
        presence_au_ids=DEFAULT_PRESENCE_AU_IDS,
    )
    counts = Counter(class_name for class_name, _ in labeled_sequences)
    ready = (
        len(counts) >= 2
        and min(counts.values()) >= args.min_samples_per_class
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload.update(
        {
            "profile_role": "original_emotion_reference",
            "source_root": str(au_root),
            "sample_counts": dict(sorted(counts.items())),
            "auto_classification_ready": ready,
            "auto_classification_reason": (
                "Original AU emotion profile is ready."
                if ready
                else (
                    "Need at least two emotion classes with "
                    f"{args.min_samples_per_class} samples per class."
                )
            ),
            "skipped_file_count": len(skipped),
        }
    )
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": output.relative_to(project_path(".").resolve()).as_posix(),
                "sample_counts": dict(sorted(counts.items())),
                "auto_classification_ready": ready,
                "skipped_file_count": len(skipped),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
