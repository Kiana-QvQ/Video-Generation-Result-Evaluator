from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_holdout_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Holdout manifest must be an object: {manifest_path}")
    return payload


def holdout_paths(
    manifest_path: str | Path,
    *,
    domain: str,
    kind: str,
) -> set[str]:
    payload = load_holdout_manifest(manifest_path)
    records = payload.get(domain, [])
    if not isinstance(records, list):
        raise ValueError(
            f"Holdout domain must be a list: {domain} in {manifest_path}"
        )
    values: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        value = record.get(kind)
        if value is None:
            value = record.get(f"{kind}_path")
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        values.add(str(candidate.resolve()))
    return values
