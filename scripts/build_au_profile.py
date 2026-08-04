from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.au_compliance import (
    DEFAULT_AU_IDS,
    DEFAULT_PRESENCE_AU_IDS,
    atomic_write_text,
    fit_au_profile,
    load_au_table,
    load_au_profile_tables,
    sha256_file,
)
from evaluator.paths import project_path


FULL_DATASET_CLASS_PREFIXES = {
    "kaixin": "smile",
    "fennu": "anger",
    "jingya": "surprise",
    "kongju": "fear",
    "shengqi": "annoyance",
    "beishang": "sadness",
    "yanwu": "disgust",
}


def _find_au_file(au_root: Path, relative_path: str) -> Path | None:
    relative = Path(relative_path)
    candidates = [
        au_root / relative.with_suffix(".csv"),
        au_root / relative.with_suffix(".tsv"),
        au_root / relative.parent / relative.stem / "au.csv",
    ]
    return next((path for path in candidates if path.is_file()), None)


def _full_dataset_class(path: Path) -> str | None:
    directory = re.sub(r"^cl_", "", path.parent.name.casefold())
    for prefix, expression_class in FULL_DATASET_CLASS_PREFIXES.items():
        if directory.startswith(prefix):
            return expression_class
    return None


def _build_from_full_dataset(
    *,
    au_root: Path,
    video_root: Path,
) -> tuple[
    list[tuple[str, object]],
    list[tuple[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    labeled_sequences: list[tuple[str, object]] = []
    presence_sequences: list[tuple[str, object]] = []
    sample_metadata: list[dict[str, object]] = []
    skipped: list[str] = []
    quality_source_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()

    for au_path in sorted(au_root.rglob("*.csv")):
        expression_class = _full_dataset_class(au_path)
        if expression_class is None:
            continue
        relative = au_path.relative_to(au_root)
        video_path = video_root / relative.with_suffix(".mp4")
        try:
            sequence, supported, presence, presence_supported = (
                load_au_profile_tables(
                    au_path,
                    intensity_au_ids=DEFAULT_AU_IDS,
                    presence_au_ids=DEFAULT_PRESENCE_AU_IDS,
                )
            )
            if not supported or presence is None or not presence_supported:
                raise ValueError("missing intensity or presence AU columns")
            quality_source_counts["profile_tables_one_pass"] += 1
        except (OSError, ValueError) as exc:
            skipped.append(f"{relative.as_posix()}: {exc}")
            continue

        labeled_sequences.append((expression_class, sequence))
        presence_sequences.append((expression_class, presence))
        class_counts[expression_class] += 1
        metadata: dict[str, object] = {
            "source_id": relative.with_suffix("").as_posix(),
            "source_path": relative.with_suffix(".mp4").as_posix(),
            "au_path": str(au_path),
            "au_sha256": sha256_file(au_path),
        }
        if video_path.is_file():
            metadata["video_path"] = str(video_path)
            metadata["video_sha256"] = sha256_file(video_path)
        else:
            skipped.append(
                f"{relative.with_suffix('.mp4').as_posix()}: video file missing"
            )
        sample_metadata.append(metadata)

    return (
        labeled_sequences,
        presence_sequences,
        sample_metadata,
        {
            "dataset_mode": "full_au_dataset",
            "source_root": str(au_root),
            "video_root": str(video_root),
            "class_mapping": FULL_DATASET_CLASS_PREFIXES,
            "class_counts": dict(sorted(class_counts.items())),
            "skipped_count": len(skipped),
            "skipped_preview": skipped[:20],
            "quality_source_file_counts": dict(
                sorted(quality_source_counts.items())
            ),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Wang Xing AU personal-expression distributions."
    )
    parser.add_argument(
        "--manifest",
        default="data/video/expression_reference_manifest.json",
        help="Legacy expression manifest. Ignored when --input-root is used.",
    )
    parser.add_argument("--au-root", required=True)
    parser.add_argument(
        "--input-root",
        help=(
            "Full AU dataset root, for example data/au/MD_CL. "
            "When supplied, scan all recognized emotion directories instead "
            "of using the 85-record expression manifest."
        ),
    )
    parser.add_argument("--video-root", default="data/MD_CL")
    parser.add_argument(
        "--output",
        default="data/video/wangxing_au_profile.json",
    )
    args = parser.parse_args()

    au_root = project_path(args.au_root)
    output = project_path(args.output)
    if args.input_root:
        (
            labeled_sequences,
            presence_sequences,
            sample_metadata,
            dataset_provenance,
        ) = _build_from_full_dataset(
            au_root=project_path(args.input_root),
            video_root=project_path(args.video_root),
        )
        if not labeled_sequences:
            raise SystemExit(
                "No recognized full-dataset AU CSV files were found."
            )
    else:
        manifest = json.loads(
            project_path(args.manifest).read_text(encoding="utf-8-sig")
        )
        labeled_sequences = []
        presence_sequences = []
        sample_metadata = []
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
            sequence, _, _ = load_au_table(
                au_path,
                DEFAULT_AU_IDS,
                feature_type="intensity",
                strict=True,
            )
            labeled_sequences.append(
                (record["expression_class"], sequence)
            )
            metadata: dict[str, object] = {
                "source_id": record.get("clip_id", record["relative_path"]),
                "source_path": record.get(
                    "local_path",
                    record["relative_path"],
                ),
                "au_path": str(au_path),
                "au_sha256": sha256_file(au_path),
            }
            source_video = project_path("data/video") / Path(
                record.get("local_path", record["relative_path"])
            )
            if source_video.is_file():
                metadata["video_path"] = str(source_video)
                metadata["video_sha256"] = sha256_file(source_video)
            sample_metadata.append(metadata)
            try:
                presence, _, _ = load_au_table(
                    au_path,
                    DEFAULT_PRESENCE_AU_IDS,
                    feature_type="presence",
                    strict=True,
                    intensity_scale=1.0,
                )
            except ValueError:
                continue
            presence_sequences.append(
                (record["expression_class"], presence)
            )

        if missing:
            print(f"Missing AU files: {len(missing)}")
            for path in missing[:20]:
                print(f"  {path}")
    if not labeled_sequences:
        raise SystemExit(
            "No labeled AU files found. Run a mature AU extractor first."
        )

    profile = fit_au_profile(
        labeled_sequences,
        output,
        au_ids=DEFAULT_AU_IDS,
        presence_labeled_sequences=presence_sequences,
        presence_au_ids=DEFAULT_PRESENCE_AU_IDS,
        sample_metadata=sample_metadata,
    )
    profile["provenance"].update(
        {
            "profile_role": "wangxing_personal_expression",
            "au_root": str(au_root),
            **(
                dataset_provenance
                if args.input_root
                else {
                    "dataset_mode": "expression_reference_manifest",
                    "manifest_path": str(project_path(args.manifest)),
                    "manifest_sha256": sha256_file(
                        project_path(args.manifest)
                    ),
                }
            ),
        }
    )
    atomic_write_text(
        output,
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps({
        "classes": {
            name: model["sample_count"]
            for name, model in profile["classes"].items()
        },
        "sample_count": len(labeled_sequences),
        "output": output.relative_to(project_path("." ).resolve()).as_posix(),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
