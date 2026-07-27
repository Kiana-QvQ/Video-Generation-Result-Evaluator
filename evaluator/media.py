from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_ffmpeg() -> str | None:
    configured = os.environ.get("FFMPEG_BIN")
    if configured and Path(configured).exists():
        return configured

    found = shutil.which("ffmpeg")
    if found:
        return found

    candidates = [
        PROJECT_ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
        Path(r"D:\ffmpeg\ffmpeg-2025-08-11-git-3542260376-full_build\bin\ffmpeg.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def transcode_video_for_browser(
    source: str | Path,
    destination: str | Path,
) -> Path:
    """Create a browser/OpenCV-friendly H.264 MP4 without changing the source."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise FileNotFoundError(
            "FFmpeg is required to normalize HEVC/AV1/VP9 uploads, but it was not found."
        )

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or not destination.exists() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        detail = completed.stderr.strip() or "unknown FFmpeg error"
        raise ValueError(f"Video normalization failed: {detail[-2000:]}")
    return destination
