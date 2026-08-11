"""Train the unlabeled AU SSL temporal backbone on data/au CSVs.

No manual AU intensity labels are used. Reconstruction + future prediction +
masked-frame losses learn natural muscle dynamics from trajectories alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from evaluator.modules.core.paths import project_path  # noqa: E402
    from evaluator.modules.forensics.au_ssl_backbone import (  # noqa: E402
        default_backbone_path,
        save_backbone,
        train_au_ssl_backbone,
    )
    from evaluator.modules.forensics.facial_motion import (  # noqa: E402
        DEFAULT_AU_IDS,
        _column_for_au,
        _fill_missing,
        _finite,
        _read_rows,
    )
except ImportError:
    from modules.core.paths import project_path  # noqa: E402
    from modules.forensics.au_ssl_backbone import (  # noqa: E402
        default_backbone_path,
        save_backbone,
        train_au_ssl_backbone,
    )
    from modules.forensics.facial_motion import (  # noqa: E402
        DEFAULT_AU_IDS,
        _column_for_au,
        _fill_missing,
        _finite,
        _read_rows,
    )


def _collect_csvs(roots: Sequence[Path], limit: int) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() == ".csv":
            paths.append(root)
            continue
        paths.extend(sorted(root.rglob("*.csv")))
    paths = sorted({path.resolve() for path in paths})
    if limit > 0:
        paths = paths[:limit]
    return paths


def _load_au_matrix(csv_path: Path) -> np.ndarray | None:
    rows, fieldnames = _read_rows(csv_path)
    if len(rows) < 8:
        return None
    columns = {
        au_id: _column_for_au(fieldnames, au_id) for au_id in DEFAULT_AU_IDS
    }
    matrix = np.full((len(rows), len(DEFAULT_AU_IDS)), np.nan, dtype=np.float32)
    for row_index, row in enumerate(rows):
        for column_index, au_id in enumerate(DEFAULT_AU_IDS):
            column = columns[au_id]
            if column is None:
                continue
            value = _finite(row.get(column), float("nan"))
            if np.isfinite(value) and value > 1.0:
                value /= 5.0
            if np.isfinite(value):
                matrix[row_index, column_index] = float(
                    max(0.0, min(1.0, value))
                )
    if not np.isfinite(matrix).any():
        return None
    return _fill_missing(matrix)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train TCAE-style AU SSL backbone on unlabeled AU CSVs under data/."
        )
    )
    parser.add_argument(
        "--au-roots",
        nargs="+",
        default=["data/au/MD_CL", "data/au/WangXing_Seedance"],
        help="CSV roots used as unlabeled SSL training corpora.",
    )
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--output",
        default=str(default_backbone_path()),
        help="Where to save au_ssl_tcae.pt",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    roots = [project_path(root) for root in args.au_roots]
    csv_paths = _collect_csvs(roots, args.limit)
    if not csv_paths:
        raise SystemExit(f"No AU CSV found under: {roots}")

    sequences: list[np.ndarray] = []
    used: list[str] = []
    for path in csv_paths:
        matrix = _load_au_matrix(path)
        if matrix is None:
            continue
        sequences.append(matrix)
        used.append(str(path))
    if len(sequences) < 4:
        raise SystemExit(
            f"Need >=4 usable AU sequences; got {len(sequences)} from {len(csv_paths)} CSVs."
        )

    print(
        f"Training AU SSL backbone on {len(sequences)} sequences "
        f"(from {len(csv_paths)} CSVs)..."
    )
    payload = train_au_ssl_backbone(
        sequences,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    output = project_path(args.output)
    save_backbone(payload, output)
    summary = {
        "output": str(output),
        "sequence_count": len(sequences),
        "csv_count_scanned": len(csv_paths),
        "final_loss": payload.get("final_loss"),
        "train_window_count": payload.get("train_window_count"),
        "manual_labels_required": False,
        "sample_sources": used[:12],
    }
    meta_path = output.with_suffix(".json")
    meta_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
