from __future__ import annotations

import csv
from pathlib import Path


DEFAULT_MIN_AU_ROWS = 32
DEFAULT_MIN_FRAME_COVERAGE = 0.10


def validate_au_csv(
    path: str | Path,
    *,
    min_rows: int = DEFAULT_MIN_AU_ROWS,
    min_frame_coverage: float = DEFAULT_MIN_FRAME_COVERAGE,
) -> tuple[bool, str]:
    """Validate the minimum temporal quality needed for an AU sequence."""
    path = Path(path)
    if min_rows <= 0:
        raise ValueError("min_rows must be positive.")
    if not 0.0 <= min_frame_coverage <= 1.0:
        raise ValueError("min_frame_coverage must be between 0 and 1.")
    if not path.is_file():
        return False, "file does not exist"
    if path.stat().st_size == 0:
        return False, "file is empty"

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            frame_field = next(
                (
                    name
                    for name in fieldnames
                    if str(name).casefold()
                    in {"frame_idx", "frame_index", "frame"}
                ),
                None,
            )
            frame_indices: list[int] = []
            row_count = 0
            for row in reader:
                row_count += 1
                if frame_field is None:
                    continue
                try:
                    frame_indices.append(int(float(row[frame_field])))
                except (TypeError, ValueError):
                    continue
    except (OSError, csv.Error) as exc:
        return False, f"CSV read failed: {exc}"

    if row_count < min_rows:
        return False, f"only {row_count} rows; need at least {min_rows}"

    if frame_indices and min_frame_coverage > 0.0:
        frame_span = max(frame_indices) - min(frame_indices) + 1
        coverage = len(set(frame_indices)) / max(frame_span, 1)
        if coverage < min_frame_coverage:
            return False, (
                f"frame coverage {coverage:.3f} is below "
                f"{min_frame_coverage:.3f}"
            )

    return True, f"{row_count} rows"
