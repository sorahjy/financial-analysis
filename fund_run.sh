#!/bin/sh
# 任一步失败立即停止，避免用残缺数据覆盖上一次的正常报告
set -e
cd "$(dirname "$0")"

# FUND_CRAWL_NO_PROXY=1 时绕过系统代理直连（天天基金均为境内接口，
# 经本地代理转发易出现 ProxyError；NO_PROXY=* 同时屏蔽 macOS 系统代理）
if [ "$FUND_CRAWL_NO_PROXY" = "1" ]; then
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
    export NO_PROXY="*" no_proxy="*"
    echo "已绕过代理直连境内数据源"
fi

# 只解析严格校验过的基金代码列表，不执行 funds.py；同时兼容直接运行本脚本。
python -c 'from app.routes.fund import _load_validated_fund_config, _sync_fund_codes; _sync_fund_codes(_load_validated_fund_config())'

# Scrapy 缓存每日失效：非今日缓存自动清除
CACHE_DIR=".scrapy/httpcache/jijin"
if [ -d "$CACHE_DIR" ]; then
    CACHE_DATE=$(stat -f "%Sm" -t "%Y-%m-%d" "$CACHE_DIR" 2>/dev/null || stat -c "%y" "$CACHE_DIR" 2>/dev/null | cut -d' ' -f1)
    TODAY=$(date "+%Y-%m-%d")
    if [ "$CACHE_DATE" != "$TODAY" ]; then
        echo "缓存非今日，清除: $CACHE_DIR"
        rm -rf "$CACHE_DIR"
    else
        echo "今日缓存有效，复用"
    fi
fi

scrapy crawl jijin
python fund_fetch_data.py
python fund_technical_analysis.py
python fund_generate_output.py
