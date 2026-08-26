"""Validate a V5.3 explicit-role runtime manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path
from wangxing_project.runtime_display_v53 import validate_runtime_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output")
    parser.add_argument(
        "--strict-unique-sha256",
        action="store_true",
        help=(
            "Fail on cross-group duplicate content hashes. "
            "Default only forbids duplicates inside the same group "
            "(LTX may be byte-copies of ppt clips)."
        ),
    )
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest).expanduser().resolve()
    payload: dict[str, Any] = json.loads(
        manifest_path.read_text(encoding="utf-8-sig")
    )
    errors = validate_runtime_manifest(payload)
    warnings: list[str] = []
    global_seen: dict[str, str] = {}
    for group in payload.get("groups") or []:
        group_id = str(group.get("group_id") or "<missing>")
        group_seen: set[str] = set()
        for role, value in (group.get("videos") or {}).items():
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = (manifest_path.parent / path).resolve()
            key = f"{group_id}/{role}"
            if not path.is_file():
                errors.append(f"{key}:missing_file")
                continue
            digest = _sha256(path)
            if digest in group_seen:
                errors.append(f"{key}:duplicate_sha256_in_group")
            group_seen.add(digest)
            prior = global_seen.get(digest)
            if prior is not None:
                message = f"{key}:duplicate_sha256_of={prior}"
                if args.strict_unique_sha256:
                    errors.append(message)
                else:
                    warnings.append(message)
            else:
                global_seen[digest] = key
    result = {
        "schema_version": "wangxing_v5_3_manifest_validation_v1",
        "manifest": str(manifest_path),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "strict_unique_sha256": bool(args.strict_unique_sha256),
        "sha256_count": len(global_seen),
        "note": (
            "Cross-group duplicate hashes are warnings by default because "
            "data/LTX may contain byte-identical copies of ppt clips. "
            "Use --strict-unique-sha256 for same-prompt formal jobs."
        ),
    }
    if args.output:
        output = project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
