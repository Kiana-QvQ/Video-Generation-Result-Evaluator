from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _manifest_inputs(
    manifest_path: Path,
    *,
    only_emotions: bool,
) -> list[tuple[Path, str]]:
    payload = json.loads(
        manifest_path.read_text(encoding="utf-8-sig")
    )
    inputs: list[tuple[Path, str]] = []
    for record in payload.get("records", []):
        if not record.get("phase1_usable"):
            continue
        if only_emotions and not record.get("is_emotion"):
            continue
        inputs.append(
            (
                Path(record["local_path"]),
                str(record["relative_path"]),
            )
        )
    return inputs


def _find_executable() -> str:
    executable = (
        shutil.which("libreface")
        or shutil.which("libreface.exe")
    )
    if executable is None:
        raise SystemExit(
            "LibreFace CLI was not found. Install LibreFace in its "
            "supported environment, then make `libreface` available on PATH."
        )
    return executable


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract per-frame AU CSV files with official LibreFace."
    )
    parser.add_argument(
        "--manifest",
        default="data/video/expression_reference_manifest.json",
    )
    parser.add_argument("--input", action="append")
    parser.add_argument("--output-root", default="data/au/libreface")
    parser.add_argument("--only-emotions", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    executable = _find_executable()
    output_root = Path(args.output_root)
    if args.input:
        inputs = [
            (Path(value), Path(value).name)
            for value in args.input
        ]
    else:
        inputs = _manifest_inputs(
            Path(args.manifest),
            only_emotions=args.only_emotions,
        )
    if args.limit is not None:
        inputs = inputs[: max(0, int(args.limit))]

    completed = 0
    skipped = 0
    for video_path, relative_name in inputs:
        if not video_path.is_file():
            print(f"SKIP missing video: {video_path}")
            skipped += 1
            continue
        relative = Path(relative_name)
        output_path = output_root / relative.with_suffix(".csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.force:
            print(f"SKIP existing: {output_path}")
            skipped += 1
            continue

        command = [
            executable,
            f"--input_path={video_path}",
            f"--output_path={output_path}",
            f"--device={args.device}",
            f"--batch_size={max(1, int(args.batch_size))}",
            f"--num_workers={max(0, int(args.num_workers))}",
        ]
        print("RUN", " ".join(str(part) for part in command))
        subprocess.run(command, check=True)
        completed += 1

    print(
        json.dumps(
            {
                "completed": completed,
                "skipped": skipped,
                "output_root": str(output_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
