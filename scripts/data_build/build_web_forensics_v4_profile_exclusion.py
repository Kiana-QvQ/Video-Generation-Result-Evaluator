"""Merge old exclusions with all final web-test source videos/AU files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.modules.core.paths import project_path


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _items(payload: dict[str, Any]) -> list[tuple[str, dict[str, str]]]:
    rows: list[tuple[str, dict[str, str]]] = []
    if isinstance(payload.get("samples"), list):
        for item in payload["samples"]:
            if item.get("source_video") and item.get("source_au"):
                rows.append(
                    (
                        "seedance"
                        if item.get("label") in {"ai", "generated"}
                        else "real",
                        {
                            "video": str(item["source_video"]),
                            "au": str(item["source_au"]),
                        },
                    )
                )
    for domain in ("real", "seedance", "fake"):
        if isinstance(payload.get(domain), list):
            for item in payload[domain]:
                if item.get("video") and item.get("au"):
                    rows.append(
                        (
                            "seedance" if domain in {"seedance", "fake"} else "real",
                            {
                                "video": str(item["video"]),
                                "au": str(item["au"]),
                            },
                        )
                    )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build v4 web profile exclusion from final test manifests."
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    base_path = project_path(args.base)
    payload = _load(base_path)
    merged: dict[str, tuple[str, dict[str, str]]] = {}
    for domain in ("real", "seedance"):
        for item in payload.get(domain, []):
            if item.get("video") and item.get("au"):
                merged[f"{item['video']}|{item['au']}"] = (
                    domain,
                    {
                        "video": str(item["video"]),
                        "au": str(item["au"]),
                    },
                )
    for manifest in args.manifest:
        path = project_path(manifest)
        if not path.is_file():
            raise FileNotFoundError(f"Test manifest not found: {path}")
        for domain, item in _items(_load(path)):
            merged[f"{item['video']}|{item['au']}"] = (domain, item)

    output = {
        "schema_version": "web_forensics_v4_profile_exclusion_v1",
        "base_exclusion": str(base_path),
        "test_manifests": [
            str(project_path(path)) for path in args.manifest
        ],
        "note": (
            "Includes the previous web/official exclusions plus all "
            "source videos and AU files from the v4 final test manifests."
        ),
        "real": [],
        "seedance": [],
    }
    for domain, item in merged.values():
        output[domain].append(item)
    output["real"].sort(key=lambda item: item["video"])
    output["seedance"].sort(key=lambda item: item["video"])
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "real": len(output["real"]),
                "seedance": len(output["seedance"]),
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
