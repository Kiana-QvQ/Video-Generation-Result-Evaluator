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
LIBREFACE_MAX_LONG_SIDE = 960


def _configure_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue


def _utf8_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    return environment


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


def _input_root_inputs(input_root: Path) -> list[tuple[Path, str]]:
    input_root = _project_path(input_root)
    if not input_root.is_dir():
        raise SystemExit(f"Input root was not found: {input_root}")

    inputs: list[tuple[Path, str]] = []
    for path in sorted(input_root.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".mp4":
            inputs.append((path, path.relative_to(input_root).as_posix()))
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
        environment = _utf8_environment()

        def run_worker(input_path: Path) -> None:
            staged_output.unlink(missing_ok=True)
            command = [
                str(runtime_python),
                str(worker),
                "--input-path",
                input_path.as_posix(),
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
            try:
                completed = subprocess.run(
                    command,
                    check=True,
                    env=environment,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except subprocess.CalledProcessError as exc:
                if exc.stdout:
                    print(exc.stdout, end="")
                if exc.stderr:
                    print(exc.stderr, end="", file=sys.stderr)
                raise
            if completed.stdout:
                print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr)

        def normalise_input() -> Path:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError(
                    "ffmpeg is required to normalize high-resolution AU input."
                )
            normalized_input = temporary_root / "normalized.mp4"
            try:
                completed = subprocess.run(
                    [
                        ffmpeg,
                        "-y",
                        "-i",
                        str(staged_input),
                        "-vf",
                        "scale=540:-2",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        "-an",
                        str(normalized_input),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except subprocess.CalledProcessError as exc:
                if exc.stderr:
                    print(exc.stderr, end="", file=sys.stderr)
                raise
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr)
            return normalized_input

        try:
            try:
                run_worker(staged_input)
            except subprocess.CalledProcessError as first_error:
                print(
                    "LibreFace could not process the original video. "
                    "Retrying with a normalized AU input.",
                    file=sys.stderr,
                )
                try:
                    normalized_input = normalise_input()
                    run_worker(normalized_input)
                except Exception as normalized_error:
                    raise normalized_error from first_error
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
    _configure_utf8_streams()
    parser = argparse.ArgumentParser(
        description="Extract per-frame AU CSV files with official LibreFace."
    )
    parser.add_argument(
        "--manifest",
        default="data/video/expression_reference_manifest.json",
    )
    parser.add_argument("--input", action="append")
    parser.add_argument(
        "--input-root",
        help="Recursively process all MP4 files below this directory.",
    )
    parser.add_argument("--output-root", default="data/au/libreface")
    parser.add_argument("--only-emotions", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed videos and continue with the remaining inputs.",
    )
    parser.add_argument(
        "--failure-log",
        help="JSON path for failures; defaults to <output-root>/_failures.json.",
    )
    args = parser.parse_args()

    _find_executable()
    if args.input_root:
        if args.input:
            raise SystemExit("--input-root cannot be combined with --input.")
        inputs = _input_root_inputs(_project_path(args.input_root))
    elif args.input:
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
    failures: list[dict[str, str]] = []
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

        try:
            _run_libreface(
                video_path,
                output_path,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        except Exception as exc:
            failure = {
                "input": str(video_path),
                "output": str(output_path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(
                f"FAIL {video_path}: {failure['error']}",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                raise
            continue
        completed += 1

    failure_log: Path | None = None
    if failures:
        failure_log = _project_path(
            args.failure_log
            if args.failure_log
            else output_root / "_failures.json"
        )
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        failure_log.write_text(
            json.dumps(
                {
                    "failed": len(failures),
                    "records": failures,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "completed": completed,
                "skipped": skipped,
                "failed": len(failures),
                "output_root": str(output_root),
                "failure_log": str(failure_log) if failure_log else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
