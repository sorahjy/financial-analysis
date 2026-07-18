#!/bin/sh
# Unix 兼容入口；Windows 和网页刷新直接运行 stock_radar_fresh_data.py。
set -eu
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" -B stock_radar_fresh_data.py "$@"
