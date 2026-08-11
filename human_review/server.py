#!/usr/bin/env python3
"""Launch the standalone human review web application."""

from __future__ import annotations

import argparse

from app import app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the standalone human video review service."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"Human review: http://127.0.0.1:{args.port}")
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
    )


if __name__ == "__main__":
    main()

