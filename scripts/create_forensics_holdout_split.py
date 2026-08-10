from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".wmv"}
AU_SUFFIXES = {".csv", ".tsv"}


def _files(root: Path, suffixes: set[str]) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _matched_pairs(video_root: Path, au_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for au_path in _files(au_root, AU_SUFFIXES):
        relative = au_path.relative_to(au_root)
        video_path = next(
            (
                candidate
                for suffix in sorted(VIDEO_SUFFIXES)
                if (
                    candidate := video_root / relative.with_suffix(suffix)
                ).is_file()
            ),
            None,
        )
        if video_path is not None:
            pairs.append((video_path, au_path))
    return pairs


def _uniform_select(
    pairs: list[tuple[Path, Path]],
    count: int,
) -> list[tuple[Path, Path]]:
    if count <= 0:
        raise ValueError("Holdout count must be positive.")
    if len(pairs) < count:
        raise ValueError(
            f"Requested {count} holdout samples but only {len(pairs)} "
            "matched video/AU pairs are available."
        )
    if count == len(pairs):
        return pairs
    if count == 1:
        return [pairs[0]]
    indexes = [
        round(index * (len(pairs) - 1) / (count - 1))
        for index in range(count)
    ]
    return [pairs[index] for index in indexes]


def _project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _records(pairs: Iterable[tuple[Path, Path]]) -> list[dict[str, str]]:
    return [
        {
            "video": _project_relative(video_path),
            "au": _project_relative(au_path),
        }
        for video_path, au_path in pairs
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic source-video holdout pairs for forensic "
            "profile training and probability calibration."
        )
    )
    parser.add_argument("--real-video-root", default="data/MD_CL")
    parser.add_argument("--real-au-root", default="data/au/MD_CL")
    parser.add_argument(
        "--seedance-video-root",
        default="data/WangXing_Seedance",
    )
    parser.add_argument(
        "--seedance-au-root",
        default="data/au/WangXing_Seedance",
    )
    parser.add_argument("--real-count", type=int, default=25)
    parser.add_argument("--seedance-count", type=int, default=25)
    parser.add_argument(
        "--output",
        default="data/forensics/holdout_split.json",
    )
    args = parser.parse_args()

    roots = {
        "real_video": PROJECT_ROOT / args.real_video_root,
        "real_au": PROJECT_ROOT / args.real_au_root,
        "seedance_video": PROJECT_ROOT / args.seedance_video_root,
        "seedance_au": PROJECT_ROOT / args.seedance_au_root,
    }
    for label, root in roots.items():
        if not root.is_dir():
            raise SystemExit(f"{label} root was not found: {root}")

    real_pairs = _matched_pairs(roots["real_video"], roots["real_au"])
    seedance_pairs = _matched_pairs(
        roots["seedance_video"],
        roots["seedance_au"],
    )
    real_holdout = _uniform_select(real_pairs, args.real_count)
    seedance_holdout = _uniform_select(seedance_pairs, args.seedance_count)

    payload = {
        "schema_version": "forensics_holdout_split_v1",
        "protocol": {
            "split_unit": "source_video",
            "selection": "uniform_sorted_source_paths",
            "profile_sources_excluded": True,
            "calibrator_uses_only_holdout_pairs": True,
        },
        "real": _records(real_holdout),
        "seedance": _records(seedance_holdout),
        "summary": {
            "real_available_pairs": len(real_pairs),
            "seedance_available_pairs": len(seedance_pairs),
            "real_holdout_count": len(real_holdout),
            "seedance_holdout_count": len(seedance_holdout),
        },
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
