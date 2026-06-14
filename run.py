from __future__ import annotations

import argparse

from app import create_app


def serve(*, host: str = "127.0.0.1", port: int = 8765, debug: bool = False) -> None:
    """Create and run the Flask app. Shared by run.py and the compat launcher."""
    app = create_app()
    app.run(host=host, port=port, debug=debug)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the financial-analysis Flask app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    serve(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
