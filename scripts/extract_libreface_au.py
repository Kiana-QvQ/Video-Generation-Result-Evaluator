from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MPL_CONFIG_DIR = PROJECT_ROOT / ".tmp" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
if os.name == "nt":
    LIBREFACE_STAGE_ROOT = Path(
        os.environ.get(
            "FRAME_AUDIT_LIBREFACE_TMP",
            str(Path(tempfile.gettempdir()) / "frame_audit_libreface"),
        )
    )
else:
    LIBREFACE_STAGE_ROOT = Path(tempfile.gettempdir()) / "frame_audit_libreface"
LIBREFACE_STAGE_ROOT.mkdir(parents=True, exist_ok=True)
LIBREFACE_WEIGHTS_ROOT = LIBREFACE_STAGE_ROOT / "weights_libreface"
LIBREFACE_WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _manifest_inputs(
    manifest_path: Path,
    *,
    only_emotions: bool,
) -> list[tuple[Path, str]]:
    manifest_path = _project_path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    inputs: list[tuple[Path, str]] = []
    for record in payload.get("records", []):
        if not record.get("phase1_usable"):
            continue
        if only_emotions and not record.get("is_emotion"):
            continue
        local_path = Path(record["local_path"])
        if not local_path.is_absolute():
            local_path = manifest_path.parent / local_path
        inputs.append((local_path, str(record["relative_path"])))
    return inputs


def _find_executable() -> str:
    executable_candidates = [
        Path(sys.executable).with_name("libreface.exe"),
        Path(sys.executable).with_name("libreface"),
    ]
    executable = next(
        (
            str(path)
            for path in executable_candidates
            if path.is_file()
        ),
        None,
    )
    executable = executable or shutil.which("libreface")
    executable = executable or shutil.which("libreface.exe")
    if not executable:
        raise SystemExit(
            "LibreFace CLI was not found. Install the `libreface` package "
            "in the same Python environment or add `libreface` to PATH."
        )
    return executable


def _free_subst_drive() -> str | None:
    if os.name != "nt":
        return None
    result = subprocess.run(
        ["subst"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    occupied = result.stdout.upper()
    for letter in "RSTUVWXYZ":
        if f"{letter}:\\" not in occupied:
            return f"{letter}:"
    return None


def _run_libreface(
    video_path: Path,
    output_path: Path,
    *,
    device: str,
    batch_size: int,
    num_workers: int,
) -> None:
    # LibreFace's temporary frame handling is unreliable under non-ASCII
    # Windows paths, so stage both input and output in an ASCII directory.
    with tempfile.TemporaryDirectory(
        prefix="run_",
        dir=str(LIBREFACE_STAGE_ROOT),
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        staged_input = temporary_root / f"input{video_path.suffix.lower()}"
        staged_output = temporary_root / "output.csv"
        shutil.copy2(video_path, staged_input)
        drive = _free_subst_drive()
        mapped = False
        if drive is not None:
            mount = subprocess.run(
                ["subst", drive, str(PROJECT_ROOT)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            mapped = mount.returncode == 0
        if mapped:
            mapped_root = Path(f"{drive}\\")
            runtime_python = mapped_root / ".venv/Scripts/python.exe"
            worker = mapped_root / "scripts/libreface_worker.py"
        else:
            runtime_python = Path(sys.executable)
            worker = PROJECT_ROOT / "scripts/libreface_worker.py"
        command = [
            str(runtime_python),
            str(worker),
            "--input-path",
            staged_input.as_posix(),
            "--output-path",
            staged_output.as_posix(),
            "--temp",
            temporary_root.as_posix(),
            "--weights-dir",
            LIBREFACE_WEIGHTS_ROOT.as_posix(),
            "--device",
            device,
            "--batch-size",
            str(max(1, int(batch_size))),
            "--num-workers",
            str(max(0, int(num_workers))),
        ]
        print("RUN", " ".join(str(part) for part in command))
        environment = os.environ.copy()
        environment.setdefault("PYTHONIOENCODING", "utf-8")
        environment.setdefault("PYTHONUTF8", "1")
        try:
            subprocess.run(command, check=True, env=environment)
            if not staged_output.is_file() or staged_output.stat().st_size == 0:
                raise RuntimeError(
                    "LibreFace exited without producing an AU CSV. "
                    f"Input: {video_path}"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged_output, output_path)
        finally:
            if mapped and drive is not None:
                subprocess.run(
                    ["subst", drive, "/D"],
                    capture_output=True,
                    check=False,
                )


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

    _find_executable()
    if args.input:
        inputs = [
            (_project_path(value), Path(value).name)
            for value in args.input
        ]
    else:
        inputs = _manifest_inputs(
            _project_path(args.manifest),
            only_emotions=args.only_emotions,
        )
    output_root = _project_path(args.output_root)
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

        _run_libreface(
            video_path,
            output_path,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
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
