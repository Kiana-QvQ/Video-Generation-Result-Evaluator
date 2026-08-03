from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import re
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.au_compliance import (  # noqa: E402
    DEFAULT_AU_IDS,
    DEFAULT_PRESENCE_AU_IDS,
    DEFAULT_FACE_QUALITY_THRESHOLD,
    atomic_write_text,
    fit_au_profile,
    load_au_table,
    load_au_profile_tables,
)
from evaluator.au_dataset import (  # noqa: E402
    DEFAULT_MIN_AU_ROWS,
    DEFAULT_MIN_FRAME_COVERAGE,
    validate_au_csv,
)
from evaluator.paths import project_path  # noqa: E402


ORIGINAL_CLASS_PREFIXES = {
    "kaixin": "smile",
    "fennu": "anger",
    "jingya": "surprise",
    "kongju": "fear",
    "shengqi": "annoyance",
    "beishang": "sadness",
    "yanwu": "disgust",
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
    parser.add_argument("--video-root", default="data/MD_CL")
    parser.add_argument("--min-samples-per-class", type=int, default=3)
    parser.add_argument("--min-output-rows", type=int, default=DEFAULT_MIN_AU_ROWS)
    parser.add_argument(
        "--min-frame-coverage",
        type=float,
        default=DEFAULT_MIN_FRAME_COVERAGE,
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Build a partial profile instead of failing the completeness gate.",
    )
    args = parser.parse_args(argv)
    if args.min_samples_per_class <= 0:
        raise ValueError("--min-samples-per-class must be positive.")
    if args.min_output_rows <= 0:
        raise ValueError("--min-output-rows must be positive.")
    if not 0.0 <= args.min_frame_coverage <= 1.0:
        raise ValueError("--min-frame-coverage must be between 0 and 1.")

    au_root = project_path(args.au_root)
    output = project_path(args.output)
    video_root = project_path(args.video_root)
    labeled_sequences: list[tuple[str, object]] = []
    presence_sequences: list[tuple[str, object]] = []
    skipped: list[str] = []
    expected_outputs: list[Path] = []
    incomplete_outputs: list[str] = []
    quality_filtered_file_count = 0
    quality_dropped_frame_count = 0
    quality_source_file_counts: Counter[str] = Counter()
    if not video_root.is_dir() and not args.allow_incomplete:
        raise SystemExit(f"Original video root was not found: {video_root}")
    if video_root.is_dir():
        for video_path in sorted(video_root.rglob("*.mp4")):
            if _class_from_path(video_path) is None:
                continue
            relative = video_path.relative_to(video_root)
            au_path = au_root / relative.with_suffix(".csv")
            expected_outputs.append(au_path)
            valid, reason = validate_au_csv(
                au_path,
                min_rows=args.min_output_rows,
                min_frame_coverage=args.min_frame_coverage,
            )
            if not valid:
                incomplete_outputs.append(f"{au_path}: {reason}")
    if incomplete_outputs and not args.allow_incomplete:
        preview = "\n".join(incomplete_outputs[:10])
        more = len(incomplete_outputs) - min(len(incomplete_outputs), 10)
        suffix = f"\n... and {more} more." if more else ""
        raise SystemExit(
            "AU extraction is incomplete or contains low-quality outputs. "
            f"Missing/invalid files: {len(incomplete_outputs)}.\n"
            f"{preview}{suffix}\n"
            "Finish extraction first, or pass --allow-incomplete explicitly."
        )

    for path in sorted(au_root.rglob("*.csv")):
        expression_class = _class_from_path(path)
        if expression_class is None:
            skipped.append(f"{path}: unknown class directory")
            continue
        valid, reason = validate_au_csv(
            path,
            min_rows=args.min_output_rows,
            min_frame_coverage=args.min_frame_coverage,
        )
        if not valid:
            skipped.append(f"{path}: {reason}")
            continue
        try:
            sequence, supported, presence, presence_supported = load_au_profile_tables(
                path,
                intensity_au_ids=DEFAULT_AU_IDS,
                presence_au_ids=DEFAULT_PRESENCE_AU_IDS,
            )
            if not supported:
                skipped.append(f"{path}: no supported intensity AU")
                continue
            _, _, quality_metadata = load_au_table(
                path,
                DEFAULT_AU_IDS,
                feature_type="intensity",
                strict=False,
            )
            quality = quality_metadata.get("quality", {})
            quality_source_file_counts[
                str(quality.get("source", "unknown"))
            ] += 1
            quality_mask = np.asarray(
                quality_metadata.get(
                    "_frame_quality",
                    np.ones(len(sequence), dtype=np.float32),
                ),
                dtype=np.float32,
            ) >= DEFAULT_FACE_QUALITY_THRESHOLD
            if (
                bool(quality.get("available"))
                and len(quality_mask) == len(sequence)
                and int(np.sum(quality_mask)) >= args.min_output_rows
                and int(np.sum(quality_mask)) < len(sequence)
            ):
                dropped = len(sequence) - int(np.sum(quality_mask))
                sequence = sequence[quality_mask]
                if presence is not None and len(presence) == len(quality_mask):
                    presence = presence[quality_mask]
                quality_filtered_file_count += 1
                quality_dropped_frame_count += dropped
            labeled_sequences.append((expression_class, sequence))
            if presence is not None and presence_supported:
                presence_sequences.append((expression_class, presence))
        except (OSError, ValueError) as exc:
            skipped.append(f"{path}: {exc}")

    if not labeled_sequences:
        raise SystemExit(
            f"No original AU CSV files with recognized emotion labels were found "
            f"under {au_root}."
        )

    fit_au_profile(
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
            "video_root": str(video_root),
            "expected_output_count": len(expected_outputs),
            "incomplete_output_count": len(incomplete_outputs),
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
            "quality_filtered_file_count": quality_filtered_file_count,
            "quality_dropped_frame_count": quality_dropped_frame_count,
            "quality_source_file_counts": dict(
                sorted(quality_source_file_counts.items())
            ),
        }
    )
    atomic_write_text(
        output,
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
