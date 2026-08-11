#!/usr/bin/env python3
"""
启动视频评估Web服务器 - 支持外部访问
"""

import argparse
from app import app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='启动视频评估Web服务器')
    parser.add_argument('--host', default='0.0.0.0', help='监听地址，默认0.0.0.0（所有接口）')
    parser.add_argument('--port', type=int, default=5000, help='监听端口，默认5000')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')

    args = parser.parse_args()

    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║  视频生成模型评估系统 - Web服务器                        ║
    ╠══════════════════════════════════════════════════════════╣
    ║  本地访问: http://127.0.0.1:{args.port}                   ║
    ║  外部访问: http://{args.host}:{args.port}   ║
    ║  局域网访问: http://<本机IP>:{args.port}                 ║
    ╠══════════════════════════════════════════════════════════╣
    ║  按 Ctrl+C 停止服务                                      ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )
