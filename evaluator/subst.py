from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


_SUBST_LINE = re.compile(
    r"^\s*([A-Za-z]):\\:\s*=>\s*(.+?)\s*$",
)


def _decode_subst_output(value: bytes | str | None) -> str:
    if isinstance(value, str):
        return value
    if not value:
        return ""
    encodings = ("utf-8", "mbcs", "oem") if os.name == "nt" else ("utf-8",)
    for encoding in encodings:
        try:
            return value.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return value.decode("utf-8", errors="replace")


def list_subst_mappings() -> list[tuple[str, Path]]:
    """Return drive-letter mappings created by the Windows subst command."""
    if os.name != "nt":
        return []
    try:
        result = subprocess.run(
            ["subst"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return []

    mappings: list[tuple[str, Path]] = []
    for line in _decode_subst_output(result.stdout).splitlines():
        match = _SUBST_LINE.match(line)
        if match is None:
            continue
        mappings.append((match.group(1).upper(), Path(match.group(2).strip())))
    return mappings


def remove_subst_drive(drive: str) -> bool:
    """Remove one subst mapping without touching the mapped target."""
    normalized = drive.rstrip(":\\").upper()
    if len(normalized) != 1 or not normalized.isalpha() or os.name != "nt":
        return False
    try:
        result = subprocess.run(
            ["subst", f"{normalized}:", "/D"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))


def cleanup_project_subst_mappings(project_root: str | Path) -> list[str]:
    """Remove only subst mappings whose target is exactly project_root."""
    if os.name != "nt":
        return []
    target = _normalized_path(Path(project_root))
    removed: list[str] = []
    for drive, mapped_path in list_subst_mappings():
        if _normalized_path(mapped_path) != target:
            continue
        if remove_subst_drive(drive):
            removed.append(f"{drive}:")
    return removed
