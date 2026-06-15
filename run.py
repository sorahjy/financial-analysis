from __future__ import annotations

import argparse
import os
import threading

from app import create_app


def _warm_stock_caches_async() -> None:
    """后台预热个股数据缓存，让首个「计算策略结果」请求不必现场冷载 ~6s。"""
    def _warm() -> None:
        try:
            from stock_advanced_strategies import warm_caches
            warm_caches()
        except Exception:
            pass  # 预热失败不应影响服务启动

    threading.Thread(target=_warm, name="stock-cache-warmup", daemon=True).start()


def serve(*, host: str = "127.0.0.1", port: int = 8765, debug: bool = False) -> None:
    """Create and run the Flask app. Shared by run.py and the compat launcher."""
    app = create_app()
    # 仅在真正提供服务的进程预热（非 debug 单进程；或 reloader 的子进程），避免重复加载。
    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        _warm_stock_caches_async()
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
