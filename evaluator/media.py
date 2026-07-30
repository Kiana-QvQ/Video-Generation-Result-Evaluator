from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .video_metrics import probe_video


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


def concatenate_videos(
    sources: list[str | Path] | tuple[str | Path, ...],
    destination: str | Path,
) -> Path:
    """Join normalized video segments into one evaluation reference video."""
    if not sources:
        raise ValueError("At least one video segment is required.")
    if len(sources) == 1:
        return Path(sources[0])

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise FileNotFoundError(
            "FFmpeg is required to join multiple reference video segments, "
            "but it was not found."
        )

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    first_info = probe_video(sources[0])
    target_width = max(2, first_info.width - first_info.width % 2)
    target_height = max(2, first_info.height - first_info.height % 2)
    target_fps = max(first_info.fps, 1.0)
    filter_parts = []
    for index in range(len(sources)):
        filter_parts.append(
            f"[{index}:v:0]"
            f"scale={target_width}:{target_height}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,fps={target_fps:g},format=yuv420p,"
            f"setpts=PTS-STARTPTS[v{index}]"
        )
    concat_inputs = "".join(f"[v{index}]" for index in range(len(sources)))
    filter_complex = (
        ";".join(filter_parts)
        + ";"
        + f"{concat_inputs}concat=n={len(sources)}:v=1:a=0[outv]"
    )

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    for source in sources:
        command.extend(["-i", str(source)])
    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if (
        completed.returncode != 0
        or not destination.exists()
        or destination.stat().st_size == 0
    ):
        destination.unlink(missing_ok=True)
        detail = completed.stderr.strip() or "unknown FFmpeg error"
        raise ValueError(f"Video concatenation failed: {detail[-2000:]}")
    return destination
