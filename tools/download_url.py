from __future__ import annotations

import sys
import hashlib
from pathlib import Path

import requests


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) not in {3, 4}:
        raise SystemExit(
            "usage: download_url.py URL DESTINATION [EXPECTED_SHA256]"
        )

    url = sys.argv[1]
    destination = Path(sys.argv[2])
    expected_sha256 = sys.argv[3].lower() if len(sys.argv) == 4 else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    try:
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Downloaded file is empty: {temporary}")
        if expected_sha256 and _sha256(temporary).lower() != expected_sha256:
            raise RuntimeError(
                f"SHA256 mismatch for {url}: expected {expected_sha256}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
