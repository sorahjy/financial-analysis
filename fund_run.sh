python funds.py

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
python fund_fetch_nav_history.py
python fund_fetch_realtime_estimate.py
python fund_technical_analysis.py
python fund_generate_output.py
