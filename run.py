from __future__ import annotations

import argparse
import os

from app import create_app


def serve(*, host: str = "127.0.0.1", port: int = 8765, debug: bool = False,
          skip_stock_warmup: bool = False) -> None:
    """Create and run the Flask app. Shared by run.py and the compat launcher."""
    should_warmup_stock = not skip_stock_warmup and (
        not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    )
    app = create_app(warmup_stock_strategy=should_warmup_stock)
    app.run(host=host, port=port, debug=debug)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the financial-analysis Flask app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-stock-warmup", action="store_true")
    args = parser.parse_args()

    serve(host=args.host, port=args.port, debug=args.debug,
          skip_stock_warmup=args.skip_stock_warmup)


if __name__ == "__main__":
    main()
