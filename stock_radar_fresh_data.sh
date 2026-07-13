#!/usr/bin/env bash
set -euo pipefail

export STOCK_CRAWL_NO_PROXY=1
export NO_PROXY="*"
export no_proxy="*"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

python stock_data_refresh.py --mode full --no-proxy
python plate_crawl_history.py --no-proxy
python stock_theme_candidates.py
python stock_hot_money_radar.py ambush --no-exclude-large-cap --pool leader
