from __future__ import annotations

import sys
from pathlib import Path

import requests


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: download_url.py URL DESTINATION")

    url = sys.argv[1]
    destination = Path(sys.argv[2])
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with destination.open("wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)


if __name__ == "__main__":
    main()
