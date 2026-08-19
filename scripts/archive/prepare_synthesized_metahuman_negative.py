from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VIDEO_SUFFIXES = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
}


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(
                    f"Unsafe archive member path: {member.filename}"
                )
        handle.extractall(destination)


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "VideoEvaluator/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def _video_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def _sample_videos(
    files: list[Path],
    *,
    max_videos: int,
    seed: int,
) -> list[Path]:
    if max_videos <= 0 or len(files) <= max_videos:
        return files
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:max_videos])


def _write_manifest(
    files: list[Path],
    source_root: Path,
    output_root: Path,
) -> Path:
    records = []
    for index, source in enumerate(files, start=1):
        relative = source.relative_to(source_root).as_posix()
        target = output_root / "videos" / f"negative_{index:05d}{source.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "person": f"metahuman_negative_{index:05d}",
                "performance": "synthesized_metahuman",
                "relative_path": target.relative_to(output_root).as_posix(),
                "local_path": target.relative_to(output_root).as_posix(),
                "source_relative_path": relative,
                "phase1_usable": True,
                "is_emotion": False,
                "expression_class": "negative_identity",
                "metadata_source": "synthesized_metahuman",
            }
        )

    manifest = {
        "schema_version": "negative_video_manifest_v1",
        "source": "Synthesized MetaHuman",
        "records": records,
    }
    manifest_path = output_root / "negative_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a licensed Synthesized MetaHuman subset as negative "
            "videos for AU leakage classification."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", help="Official ZIP received after approval.")
    source.add_argument(
        "--url",
        help="Official download URL received after approval.",
    )
    parser.add_argument(
        "--output-root",
        default="data/negative/synthesized_metahuman",
    )
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    output_root = _project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="metahuman_prepare_") as temp:
        temp_root = Path(temp)
        if args.archive:
            archive = _project_path(args.archive)
            if not archive.is_file():
                raise SystemExit(f"Archive not found: {archive}")
            archive_path = archive
        else:
            archive_path = temp_root / "synthesized_metahuman.zip"
            print("Downloading the user-provided official URL...")
            _download(args.url, archive_path)

        extracted = temp_root / "extracted"
        extracted.mkdir()
        _safe_extract(archive_path, extracted)
        files = _video_files(extracted)
        if not files:
            raise SystemExit(
                "No supported videos were found in the official archive."
            )
        selected = _sample_videos(
            files,
            max_videos=int(args.max_videos),
            seed=int(args.seed),
        )
        manifest_path = _write_manifest(
            selected,
            extracted,
            output_root,
        )

    print(
        json.dumps(
            {
                "selected_videos": len(selected),
                "manifest": str(manifest_path),
                "output_root": str(output_root),
                "note": (
                    "This is a negative cross-identity/synthetic set; "
                    "it is not a Wang Xing expression ground truth set."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
