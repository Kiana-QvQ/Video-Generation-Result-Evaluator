#!/usr/bin/env python3
"""Launch the standalone human review web application."""

from __future__ import annotations

import argparse

try:
    from .app import create_app
except ImportError:
    from app import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the standalone human video review service."
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="监听地址，默认 0.0.0.0（允许局域网访问）",
    )
    parser.add_argument("--port", type=int, default=5001, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print(
        f"""
    ============================================================
    Human Review / 人工视频评测
    ------------------------------------------------------------
    本机访问:     http://127.0.0.1:{args.port}
    绑定地址:     http://{args.host}:{args.port}
    局域网访问:   http://<本机局域网IP>:{args.port}
    ------------------------------------------------------------
    按 Ctrl+C 停止服务
    ============================================================
    """
    )
    create_app().run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
    )


if __name__ == "__main__":
    main()
