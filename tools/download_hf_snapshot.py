from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: download_hf_snapshot.py REPOSITORY LOCAL_DIRECTORY"
        )
    repository = sys.argv[1]
    local_directory = Path(sys.argv[2]).resolve()
    local_directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repository,
        local_dir=str(local_directory),
        cache_dir=str(local_directory.parent / "_hf_cache"),
        resume_download=True,
    )
    print(f"Downloaded {repository} to {local_directory}")


if __name__ == "__main__":
    main()
