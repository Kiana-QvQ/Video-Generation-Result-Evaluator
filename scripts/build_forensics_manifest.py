from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.paths import project_path
from evaluator.video_metrics import probe_video
from evaluator.wangxing_specialization import _expression_class_from_path

VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".wmv"}


def _files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def _short_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(path.resolve()).encode("utf-8", errors="replace"))
    digest.update(str(path.stat().st_size).encode("ascii"))
    return digest.hexdigest()[:16]


def _au_path(
    path: Path,
    *,
    video_root: Path,
    au_root: Path,
) -> Path | None:
    relative = path.relative_to(video_root)
    candidate = au_root / relative.with_suffix(".csv")
    return candidate if candidate.is_file() else None


def _record(
    path: Path,
    *,
    domain: str,
    video_root: Path,
    au_root: Path | None,
    seedance_version: str | None,
    generation_mode: str | None,
    input_type: str | None,
    prompt_id: str | None,
    driver_id: str | None,
    codec: str | None,
) -> dict[str, Any]:
    info = probe_video(path)
    au_path = (
        _au_path(path, video_root=video_root, au_root=au_root)
        if au_root is not None
        else None
    )
    expression_class = (
        _expression_class_from_path(au_path)
        if au_path is not None
        else None
    )
    metadata = {
        "seedance_version": seedance_version,
        "generation_mode": generation_mode,
        "input_type": input_type,
        "prompt_id": prompt_id,
        "driver_id": driver_id,
        "codec": codec,
        "metadata_complete": all(
            value is not None
            for value in (
                seedance_version,
                generation_mode,
                input_type,
                codec,
            )
        )
        if domain == "generated_wangxing"
        else True,
    }
    return {
        "sample_id": _short_hash(path),
        "domain": domain,
        "video_path": str(path),
        "relative_video_path": path.relative_to(video_root).as_posix(),
        "au_path": str(au_path) if au_path is not None else None,
        "expression_class": expression_class,
        "video": info.to_dict(),
        "generation_metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the real/Seedance forensic data manifest."
    )
    parser.add_argument("--real-video-root", default="data/MD_CL")
    parser.add_argument(
        "--seedance-video-root",
        default="data/WangXing_Seedance",
    )
    parser.add_argument("--real-au-root", default="data/au/MD_CL")
    parser.add_argument(
        "--seedance-au-root",
        default="data/au/WangXing_Seedance",
    )
    parser.add_argument(
        "--seedance-version",
        default=None,
        help="Known Seedance version; unknown is kept null.",
    )
    parser.add_argument("--generation-mode", default=None)
    parser.add_argument("--input-type", default=None)
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--driver-id", default=None)
    parser.add_argument("--codec", default=None)
    parser.add_argument(
        "--output",
        default="data/forensics/forensics_manifest.json",
    )
    args = parser.parse_args()

    real_root = project_path(args.real_video_root)
    seedance_root = project_path(args.seedance_video_root)
    real_au_root = project_path(args.real_au_root)
    seedance_au_root = project_path(args.seedance_au_root)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    inputs = [
        ("real_wangxing", real_root, real_au_root, None),
        (
            "generated_wangxing",
            seedance_root,
            seedance_au_root,
            {
                "version": args.seedance_version,
                "mode": args.generation_mode,
                "input_type": args.input_type,
                "prompt_id": args.prompt_id,
                "driver_id": args.driver_id,
                "codec": args.codec,
            },
        ),
    ]
    for domain, video_root, au_root, metadata in inputs:
        paths = _files(video_root)
        for index, path in enumerate(paths, start=1):
            if index == 1 or index % 25 == 0 or index == len(paths):
                print(
                    f"Manifest {domain}: {index}/{len(paths)}",
                    flush=True,
                )
            try:
                records.append(
                    _record(
                        path,
                        domain=domain,
                        video_root=video_root,
                        au_root=au_root,
                        seedance_version=(
                            metadata["version"] if metadata else None
                        ),
                        generation_mode=(
                            metadata["mode"] if metadata else None
                        ),
                        input_type=(
                            metadata["input_type"] if metadata else None
                        ),
                        prompt_id=(
                            metadata["prompt_id"] if metadata else None
                        ),
                        driver_id=(
                            metadata["driver_id"] if metadata else None
                        ),
                        codec=metadata["codec"] if metadata else None,
                    )
                )
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append({"path": str(path), "error": str(exc)})

    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "forensics_manifest_v1",
        "records": records,
        "summary": {
            "record_count": len(records),
            "real_count": sum(
                record["domain"] == "real_wangxing" for record in records
            ),
            "generated_count": sum(
                record["domain"] == "generated_wangxing"
                for record in records
            ),
            "au_linked_count": sum(
                record["au_path"] is not None for record in records
            ),
            "metadata_complete_generated_count": sum(
                record["generation_metadata"]["metadata_complete"]
                for record in records
                if record["domain"] == "generated_wangxing"
            ),
            "failure_count": len(failures),
        },
        "failures": failures[:100],
        "protocol": {
            "split_key": "source video or generation batch",
            "seedance_metadata_required_for_production": True,
            "unknown_metadata_is_not_inferred": True,
            "required_generated_fields": [
                "seedance_version",
                "generation_mode",
                "input_type",
                "codec",
            ],
        },
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
