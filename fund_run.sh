#!/bin/sh
# Unix 兼容入口；Windows 和网页刷新直接运行 fund_data_refresh.py。
set -eu
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" -B fund_data_refresh.py "$@"
