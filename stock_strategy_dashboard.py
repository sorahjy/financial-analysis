"""Compatibility launcher for the Flask stock strategy dashboard.

The dashboard HTML/CSS/JS now lives under ``app/templates`` and ``app/static``.
This module is kept so older commands such as
``python stock_strategy_dashboard.py --port 8765`` still start the unified app.
"""

from __future__ import annotations

import argparse

from run import serve


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Flask stock strategy dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-stock-warmup", "--skip-warmup", action="store_true")
    parser.add_argument("--skip-refresh", action="store_true", help="kept for old scripts; Flask does not refresh on startup")
    parser.add_argument("--refresh-mode", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--strict-refresh", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--refresh-timeout", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--refresh-python", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--index-workers", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--index-limit", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    serve(host=args.host, port=args.port, debug=args.debug,
          skip_stock_warmup=args.skip_stock_warmup)


if __name__ == "__main__":
    main()
