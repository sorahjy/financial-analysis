"""
股票爬虫公共工具：被 stock_crawl_hot_money / stock_crawl_fundamentals /
stock_crawl_price_valuation 共用，消除三者间的重复实现。

包含：代理绕过、安全数值转换、带退避重试、交易所前缀、指数成分获取、
通用 JSON 读写、按日期合并、history 记录规范化等。

注意语义差异（故意保留，各调用方依赖）：
  - safe_float: NaN/inf → None（财报/估值留空）。
  - safe_num: None/NaN/非数字 → 0.0（短线资金与雷达填零）。
  - retry_fetch: 末次失败 raise。
  - retry_fetch_or_none: 末次失败返回 None。
"""

import atexit
import json
import math
import multiprocessing
import os
import statistics
import subprocess
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import requests

MAX_RETRIES = 3
INDEX_CONS_PRIMARY_TIMEOUT_SEC = int(os.getenv("STOCK_INDEX_CONS_PRIMARY_TIMEOUT", "30"))
_PRINT_LOCK = threading.Lock()

HISTORY_DAILY_FIELD_ALIASES = {
    "daily_open": ("daily_open", "open"),
    "daily_high": ("daily_high", "high"),
    "daily_low": ("daily_low", "low"),
    "daily_close": ("daily_close", "close"),
    "daily_volume": ("daily_volume", "volume"),
    "daily_amount": ("daily_amount", "amount", "turnover"),
    "daily_change_pct": ("daily_change_pct", "change_pct"),
    "daily_turnover_rate": ("daily_turnover_rate", "turnover_rate"),
}
HISTORY_PRICE_FIELDS = tuple(HISTORY_DAILY_FIELD_ALIASES)
ANALYSIS_DAILY_FIELD_ALIASES = {
    "open": ("daily_open", "open"),
    "high": ("daily_high", "high"),
    "low": ("daily_low", "low"),
    "close": ("daily_close", "close"),
    "volume": ("daily_volume", "volume"),
    "amount": ("daily_amount", "amount", "turnover"),
    "change_pct": ("daily_change_pct", "change_pct"),
    "turnover_rate": ("daily_turnover_rate", "turnover_rate"),
}
HISTORY_VALUATION_FIELDS = (
    "market_cap",
    "pe_ttm",
    "pe_static",
    "pb",
    "pcf",
)
HISTORY_DAILY_OHLCV_FIELDS = (
    "daily_open",
    "daily_high",
    "daily_low",
    "daily_close",
    "daily_volume",
    "daily_amount",
)
HISTORY_SNAPSHOT_NON_CLOSE_FIELDS = (
    "daily_open",
    "daily_high",
    "daily_low",
    "daily_volume",
    "daily_amount",
    "daily_change_pct",
    "daily_turnover_rate",
)
HISTORY_DATA_FIELDS = tuple(HISTORY_DAILY_FIELD_ALIASES) + HISTORY_VALUATION_FIELDS


def strip_proxy_env():
    """STOCK_CRAWL_NO_PROXY=1 时绕过系统/环境变量代理直连境内接口。

    东财/腾讯/新浪/百度均为境内接口，经本地代理转发常出现
    ProxyError(RemoteDisconnected)；NO_PROXY=* 同时屏蔽 macOS 系统代理。
    """
    if os.getenv("STOCK_CRAWL_NO_PROXY", "").strip().lower() not in ("1", "true", "yes"):
        return
    for var in ("http_proxy", "https_proxy", "all_proxy",
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(var, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def safe_print(*args, **kwargs):
    """Thread-safe print used by concurrent crawlers."""
    with _PRINT_LOCK:
        print(*args, **kwargs)


def safe_float(val):
    """转 float；None/NaN/inf → None（便于 JSON 序列化），保留 6 位小数。"""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    except (ValueError, TypeError):
        return None


def safe_num(val, default=0.0):
    """转 float；None/NaN/非数字 → default，供短线资金口径使用。"""
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (ValueError, TypeError):
        return default


def retry_fetch(func, *args, retries=MAX_RETRIES, **kwargs):
    """重试数据获取；末次失败抛出异常（应对连接不稳定）。"""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception:
            if attempt == retries - 1:
                raise


def retry_fetch_or_none(func, *args, retries=MAX_RETRIES, **kwargs):
    """重试数据获取；末次失败返回 None，适合非关键行情补充。"""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception:
            if attempt == retries - 1:
                return None
    return None


def is_bse_stock(code):
    """北交所股票：代码以 4 / 8 / 9 开头（430xxx, 830xxx, 920xxx 等）。"""
    code = str(code)
    return code.startswith("4") or code.startswith("8") or code.startswith("9")


def exchange_prefix(code):
    """6 位 code → 交易所前缀 sh/sz/bj。"""
    code = str(code).zfill(6)
    if code.startswith(("60", "68", "9")):
        return "sh"
    if code.startswith(("00", "30", "20")):
        return "sz"
    if code.startswith(("8", "4", "92")):
        return "bj"
    return "sh"


def symbol_with_prefix(code):
    """6 位 code → 带交易所前缀的 symbol，如 sh600000（新浪/腾讯接口要求）。"""
    return exchange_prefix(code) + str(code).zfill(6)


def _fetch_csindex_constituents_with_timeout(symbol, timeout_sec=INDEX_CONS_PRIMARY_TIMEOUT_SEC):
    """Call AkShare's CSIndex constituents API in a subprocess.

    ak.index_stock_cons_csindex can hang inside HTTP/client internals without raising,
    especially for 000985. A subprocess timeout keeps refresh_stock_universe moving so
    the normal fallback path can run.
    """
    script = r"""
import contextlib
import json
import os
import sys

if os.getenv("STOCK_CRAWL_NO_PROXY", "").strip().lower() in ("1", "true", "yes"):
    for var in ("http_proxy", "https_proxy", "all_proxy",
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(var, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"

import akshare as ak

symbol = sys.argv[1]
with contextlib.redirect_stdout(sys.stderr):
    df = ak.index_stock_cons_csindex(symbol=symbol)
sys.stdout.write(json.dumps(df.to_dict(orient="records"), ensure_ascii=False, default=str))
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script, str(symbol)],
            cwd=str(Path(__file__).resolve().parent),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"超过 {timeout_sec}s 未返回") from exc

    if completed.returncode != 0:
        err = (completed.stderr or completed.stdout or "").strip()
        if len(err) > 600:
            err = "..." + err[-600:]
        raise RuntimeError(f"子进程退出 {completed.returncode}: {err or '无错误输出'}")

    try:
        rows = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        snippet = (completed.stdout or "").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        raise RuntimeError(f"中证官网成分接口返回无法解析: {snippet!r}") from exc

    return {
        str(row["成分券代码"]).zfill(6): str(row["成分券名称"])
        for row in rows
        if row.get("成分券代码") is not None and row.get("成分券名称") is not None
    }


def fetch_index_constituents(symbol):
    """指数全量成分 → {code: name}。

    优先中证官网接口 index_stock_cons_csindex(全量)；失败回退新浪
    index_stock_cons。
    """
    try:
        cons = _fetch_csindex_constituents_with_timeout(symbol)
        if cons:
            return cons
    except Exception as e:
        print(f"  [WARN] 中证官网成分接口失败/超时({symbol}): {e}, 回退新浪接口")
    df = retry_fetch(ak.index_stock_cons, symbol=symbol)
    return {
        str(row["品种代码"]).zfill(6): str(row["品种名称"])
        for _, row in df.iterrows()
    }


# ─── A 股前复权日线工具 ───────────────────────────────────────

CHANGE_PCT_LOOKBACK_DAYS = 15


def _date_dash(value):
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text[:10]


def _date_compact(value):
    return _date_dash(value).replace("-", "")


def latest_weekday_date(value=None):
    """返回不晚于 value 的最近工作日日期；避免周末刷新时补不存在的行情。"""
    if value is None:
        dt = datetime.now()
    elif isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.strptime(_date_dash(value), "%Y-%m-%d")
    while dt.weekday() >= 5:
        dt -= timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def normalize_history_record(row, *, include_valuation=True):
    """Normalize one row to the project-wide stock_data.history daily schema.

    Keep unknown OHLCV fields as None instead of fabricating prices from close.
    Legacy daily.recent_daily used "turnover" as amount; map it to amount.
    """
    if not isinstance(row, dict) or not row.get("date"):
        return None
    item = {"date": _date_dash(row.get("date"))}
    for field, aliases in HISTORY_DAILY_FIELD_ALIASES.items():
        value = next((row.get(alias) for alias in aliases if row.get(alias) is not None), None)
        item[field] = safe_float(value)
    if include_valuation:
        for field in HISTORY_VALUATION_FIELDS:
            item[field] = safe_float(row.get(field))
    return item


def normalize_history_records(records, *, include_valuation=True, drop_future=True):
    latest_date = latest_weekday_date() if drop_future else None
    normalized = []
    for row in records or []:
        item = normalize_history_record(row, include_valuation=include_valuation)
        if not item:
            continue
        if latest_date and item["date"] > latest_date:
            continue
        normalized.append(item)
    return sorted(normalized, key=lambda row: row["date"])


def history_record_has_daily_ohlcv(row):
    """True when one history row has the daily fields needed by K-line users."""
    item = normalize_history_record(row, include_valuation=False)
    if not item:
        return False
    return all(item.get(field) is not None for field in HISTORY_DAILY_OHLCV_FIELDS)


def history_record_is_snapshot_only(row):
    """Detect market_snapshot-only points: close/valuation without daily market fields.

    These rows came from the old local fallback and should not live in
    stock_data.history, because they poison OHLCV completeness checks.
    """
    item = normalize_history_record(row)
    if not item or item.get("daily_close") is None:
        return False
    return all(item.get(field) is None for field in HISTORY_SNAPSHOT_NON_CLOSE_FIELDS)


def history_record_is_empty_stub(row):
    """Detect date-only/all-null rows produced by obsolete local fallbacks."""
    item = normalize_history_record(row)
    if not item:
        return False
    return all(item.get(field) is None for field in HISTORY_DATA_FIELDS)


def prune_snapshot_only_history_records(records, *, include_valuation=True):
    """Normalize history records and drop rows that are not usable daily bars."""
    kept = []
    removed = 0
    for row in records or []:
        item = normalize_history_record(row, include_valuation=include_valuation)
        if not item:
            continue
        if history_record_is_snapshot_only(item) or history_record_is_empty_stub(item):
            removed += 1
            continue
        kept.append(item)
    return sorted(kept, key=lambda row: row["date"]), removed


def daily_stats_from_history_records(records):
    """Derive the daily.stats cache from stock_data.history daily records."""
    rows, _ = prune_snapshot_only_history_records(records, include_valuation=False)
    if not rows:
        return {}

    closes = [safe_float(row.get("daily_close")) for row in rows]
    closes = [value for value in closes if value is not None and value > 0]
    daily_returns = [
        closes[i] / closes[i - 1] - 1
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]
    daily_std = statistics.stdev(daily_returns) if len(daily_returns) > 1 else None
    volumes = [
        safe_float(row.get("daily_volume"))
        for row in rows
        if safe_float(row.get("daily_volume")) is not None
    ]
    amounts = [
        safe_float(row.get("daily_amount"))
        for row in rows
        if safe_float(row.get("daily_amount")) is not None
    ]

    return {
        "history_window_trading_days": len(rows),
        "history_window_avg_daily_volume": safe_float(statistics.fmean(volumes)) if volumes else None,
        "history_window_avg_daily_amount": safe_float(statistics.fmean(amounts)) if amounts else None,
        "history_window_annualized_volatility": safe_float(daily_std * math.sqrt(252)) if daily_std is not None else None,
        "history_window_daily_return_std": safe_float(daily_std),
        "latest_daily_close": safe_float(closes[-1]) if closes else None,
        "latest_trade_date": rows[-1]["date"],
        "history_window_start_date": rows[0]["date"],
        "history_window_end_date": rows[-1]["date"],
        "price_adjust": "qfq",
        "change_pct_basis": "close_to_close",
    }


def daily_payload_from_history_records(records):
    """Return the compact daily cache; raw recent_daily is intentionally omitted."""
    return {"stats": daily_stats_from_history_records(records)}


def history_payload_from_records(code, name, records, source):
    rows, _ = prune_snapshot_only_history_records(records)
    payload = {
        "symbol": str(code).zfill(6),
        "name": name,
        "records": rows,
        "source": source,
        "price_adjust": "qfq",
        "change_pct_basis": "close_to_close",
    }
    if rows:
        payload["start_date"] = rows[0]["date"]
        payload["end_date"] = rows[-1]["date"]
    return payload


def analysis_record_from_history_record(row):
    """Convert canonical daily_* history rows to legacy algorithm row names.

    Disk data keeps explicit daily_* keys; strategy/radar math can keep using
    compact open/high/low/close names after passing through this boundary.
    """
    item = normalize_history_record(row)
    if not item:
        return None
    out = {"date": item["date"]}
    for field, aliases in ANALYSIS_DAILY_FIELD_ALIASES.items():
        out[field] = next(
            (safe_float(row.get(alias)) for alias in aliases if row.get(alias) is not None),
            item.get(f"daily_{field}") if field != "amount" else item.get("daily_amount"),
        )
    for field in HISTORY_VALUATION_FIELDS:
        out[field] = safe_float(row.get(field))
    return out


def analysis_records_from_history_records(records):
    rows = []
    for row in records or []:
        item = analysis_record_from_history_record(row)
        if item:
            rows.append(item)
    return sorted(rows, key=lambda item: item["date"])


def _lookback_start(start_date):
    return (
        datetime.strptime(_date_dash(start_date), "%Y-%m-%d")
        - timedelta(days=CHANGE_PCT_LOOKBACK_DAYS)
    ).strftime("%Y%m%d")


def _records_with_computed_change(rows, start_date, end_date):
    """用相邻前复权收盘价计算 close-to-close change_pct 后截取日期区间。"""
    start = _date_dash(start_date)
    end = _date_dash(end_date)
    records = []
    prev_close = None
    for row in rows:
        date_str = _date_dash(row.get("date"))
        close = row.get("close")
        change_pct = None
        if close is not None and prev_close:
            change_pct = round((close / prev_close - 1) * 100, 6)
        if start <= date_str <= end:
            item = dict(row)
            item["date"] = date_str
            item["change_pct"] = change_pct
            records.append(item)
        if close is not None:
            prev_close = close
    return records


def _fetch_daily_eastmoney_qfq(symbol, start_date, end_date, include_trading_value=False):
    df = ak.stock_zh_a_hist(
        symbol=str(symbol).zfill(6),
        period="daily",
        start_date=_date_compact(start_date),
        end_date=_date_compact(end_date),
        adjust="qfq",
    )
    if df is None or df.empty:
        return []

    records = []
    for _, row in df.iterrows():
        record = {
            "date": str(row["日期"])[:10],
            "close": safe_float(row["收盘"]),
            "change_pct": safe_float(row["涨跌幅"]),
            "turnover_rate": safe_float(row["换手率"]),
        }
        if include_trading_value:
            record["open"] = safe_float(row.get("开盘"))
            record["high"] = safe_float(row.get("最高"))
            record["low"] = safe_float(row.get("最低"))
            record["volume"] = safe_float(row.get("成交量"))
            record["amount"] = safe_float(row.get("成交额"))
        records.append(record)
    return records


def _normalize_volume_to_hands(volume, amount=None, close=None):
    """成交量统一归一到「手」；当成交额能识别出「股」单位时自动除以 100。"""
    volume = safe_float(volume)
    if volume is None or volume <= 0:
        return None
    amount = safe_float(amount)
    close = safe_float(close)
    if amount is None or amount <= 0 or close is None or close <= 0:
        return volume

    share_avg = amount / volume
    hand_avg = amount / (volume * 100.0)
    if abs(share_avg - close) < abs(hand_avg - close):
        return round(volume / 100.0, 2)
    return volume


def _fetch_daily_tencent_qfq(symbol, start_date, end_date, include_trading_value=False):
    tx_symbol = symbol_with_prefix(symbol)
    start_compact = _lookback_start(start_date)
    end_compact = _date_compact(end_date)
    range_start = int(start_compact[:4])
    range_end = int(end_compact[:4]) + 1
    raw_rows = []
    url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"

    for year in range(range_start, range_end):
        params = {
            "_var": f"kline_dayqfq{year}",
            "param": f"{tx_symbol},day,{year}-01-01,{year + 1}-12-31,640,qfq",
            "r": "0.8205512681390605",
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        text = resp.text
        start_idx = text.find("={")
        if start_idx < 0:
            continue
        payload = json.loads(text[start_idx + 1:])
        data = payload.get("data", {}).get(tx_symbol, {})
        raw_rows.extend(data.get("qfqday") or data.get("day") or [])

    if not raw_rows:
        return []

    rows = []
    seen_dates = set()
    for row in sorted(raw_rows, key=lambda item: str(item[0]) if item else ""):
        if not row or len(row) < 6:
            continue
        date_str = _date_dash(row[0])
        if date_str in seen_dates:
            continue
        seen_dates.add(date_str)
        close = safe_float(row[2])
        volume = safe_float(row[5])
        turnover_rate = safe_float(row[7]) if len(row) > 7 else None
        amount_wan = safe_float(row[8]) if len(row) > 8 else None
        amount = round(amount_wan * 10000, 6) if amount_wan is not None else None
        record = {
            "date": date_str,
            "open": safe_float(row[1]) if include_trading_value else None,
            "high": safe_float(row[3]) if include_trading_value else None,
            "low": safe_float(row[4]) if include_trading_value else None,
            "close": close,
            "turnover_rate": turnover_rate,
        }
        if include_trading_value:
            record["volume"] = _normalize_volume_to_hands(volume, amount, close)
            record["amount"] = (
                amount
                if amount is not None
                else round(record["volume"] * 100 * close, 6)
                if volume is not None and close is not None
                else None
            )
        rows.append(record)
    return _records_with_computed_change(rows, start_date, end_date)


_SINA_DAILY_LOCK = threading.Lock()


def _fetch_daily_sina_qfq(symbol, start_date, end_date, include_trading_value=False):
    # akshare 的新浪日线接口内部会用 py_mini_racer；并发初始化会触发
    # libmini_racer native fatal crash，因此该源必须串行调用。
    with _SINA_DAILY_LOCK:
        df = ak.stock_zh_a_daily(
            symbol=symbol_with_prefix(symbol),
            start_date=_lookback_start(start_date),
            end_date=_date_compact(end_date),
            adjust="qfq",
        )
    if df is None or df.empty:
        return []

    rows = []
    has_turnover = "turnover" in df.columns
    for _, row in df.sort_values("date").iterrows():
        turnover = safe_float(row["turnover"]) if has_turnover else None
        record = {
            "date": str(row["date"])[:10],
            "open": safe_float(row.get("open")) if include_trading_value else None,
            "high": safe_float(row.get("high")) if include_trading_value else None,
            "low": safe_float(row.get("low")) if include_trading_value else None,
            "close": safe_float(row["close"]),
            "turnover_rate": round(turnover * 100, 6) if turnover is not None else None,
        }
        if include_trading_value:
            # 新浪 stock_zh_a_daily 的 volume 单位是「股」——统一归一到手(÷100)。
            sina_vol = safe_float(row.get("volume"))
            record["volume"] = round(sina_vol / 100.0, 2) if sina_vol is not None else None
            record["amount"] = safe_float(row.get("amount"))
        rows.append(record)
    return _records_with_computed_change(rows, start_date, end_date)


DAILY_QFQ_SOURCES = (
    ("东财", _fetch_daily_eastmoney_qfq),
    ("腾讯", _fetch_daily_tencent_qfq),
    ("新浪", _fetch_daily_sina_qfq),
)
_DAILY_QFQ_SOURCE_MAP = {source: fetcher for source, fetcher in DAILY_QFQ_SOURCES}
DAILY_QFQ_PRIORITY = [["新浪"], ["腾讯"], ["东财"]]

_DAILY_SOURCE_LOCK = threading.Lock()
_DAILY_SOURCE_FAILURES = {source: 0 for source, _ in DAILY_QFQ_SOURCES}
_DAILY_SOURCE_DISABLED = set()
_DAILY_PRIORITY_CURSORS = {}
try:
    _DAILY_SOURCE_DISABLE_AFTER = max(
        1,
        int(os.getenv("STOCK_DAILY_QFQ_DISABLE_AFTER", "6") or "6"),
    )
except ValueError:
    _DAILY_SOURCE_DISABLE_AFTER = 6

def _env_text(names, default):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


try:
    _DAILY_PROCESS_WORKERS = max(
        0,
        int(
            _env_text(
                ("STOCK_DAILY_PROCESS_WORKERS", "STOCK_DAILY_FALLBACK_PROCESS_WORKERS"),
                "32",
            )
        ),
    )
except ValueError:
    _DAILY_PROCESS_WORKERS = 32

_DAILY_PROCESS_SOURCES = {
    item.strip()
    for item in _env_text(
        ("STOCK_DAILY_PROCESS_SOURCES", "STOCK_DAILY_FALLBACK_PROCESS_SOURCES"),
        "腾讯,新浪",
    ).split(",")
    if item.strip()
}
_DAILY_PROCESS_POOL = None
_DAILY_PROCESS_POOL_LOCK = threading.Lock()


def _daily_priority_groups():
    groups = []
    for item in DAILY_QFQ_PRIORITY:
        if isinstance(item, str):
            groups.append((item,))
        else:
            groups.append(tuple(item))
    return groups


def _enabled_daily_source_groups(include_trading_value=False):
    with _DAILY_SOURCE_LOCK:
        disabled = set(_DAILY_SOURCE_DISABLED)
    priority_groups = _daily_priority_groups()
    active_groups = []
    for group in priority_groups:
        active = tuple(source for source in group if source not in disabled)
        if active:
            active_groups.append(active)
    return active_groups or priority_groups


def _enabled_daily_sources(include_trading_value=False):
    return [
        (source, _DAILY_QFQ_SOURCE_MAP[source])
        for group in _enabled_daily_source_groups(include_trading_value=include_trading_value)
        for source in group
    ]


def _ordered_daily_group_sources(group):
    group = tuple(group)
    if len(group) <= 1:
        return group
    with _DAILY_SOURCE_LOCK:
        cursor = _DAILY_PRIORITY_CURSORS.get(group, 0)
        _DAILY_PRIORITY_CURSORS[group] = cursor + 1
    start = cursor % len(group)
    return group[start:] + group[:start]


def _record_daily_source_success(source):
    with _DAILY_SOURCE_LOCK:
        _DAILY_SOURCE_FAILURES[source] = 0


def _record_daily_source_failure(source):
    with _DAILY_SOURCE_LOCK:
        failures = _DAILY_SOURCE_FAILURES.get(source, 0) + 1
        _DAILY_SOURCE_FAILURES[source] = failures
        if failures >= _DAILY_SOURCE_DISABLE_AFTER and source not in _DAILY_SOURCE_DISABLED:
            _DAILY_SOURCE_DISABLED.add(source)
            return True
    return False


def _daily_source_uses_process(source):
    return (
        _DAILY_PROCESS_WORKERS > 1
        and source in _DAILY_PROCESS_SOURCES
        and multiprocessing.current_process().name == "MainProcess"
    )


def _daily_process_start_method():
    method = _env_text(
        ("STOCK_DAILY_PROCESS_START_METHOD", "STOCK_DAILY_FALLBACK_PROCESS_START_METHOD"),
        "spawn",
    ).strip()
    methods = multiprocessing.get_all_start_methods()
    if method in methods:
        return method
    return "spawn" if "spawn" in methods else methods[0]


def _get_daily_process_pool():
    global _DAILY_PROCESS_POOL
    with _DAILY_PROCESS_POOL_LOCK:
        if _DAILY_PROCESS_POOL is None:
            ctx = multiprocessing.get_context(_daily_process_start_method())
            _DAILY_PROCESS_POOL = ProcessPoolExecutor(
                max_workers=_DAILY_PROCESS_WORKERS,
                mp_context=ctx,
            )
        return _DAILY_PROCESS_POOL


def _reset_daily_process_pool():
    global _DAILY_PROCESS_POOL
    with _DAILY_PROCESS_POOL_LOCK:
        pool = _DAILY_PROCESS_POOL
        _DAILY_PROCESS_POOL = None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


def _shutdown_daily_process_pool():
    _reset_daily_process_pool()


atexit.register(_shutdown_daily_process_pool)


def _fetch_daily_source_worker(source, symbol, start_date, end_date, include_trading_value):
    fetcher = _DAILY_QFQ_SOURCE_MAP[source]
    return fetcher(
        symbol,
        start_date,
        end_date,
        include_trading_value=include_trading_value,
    )


def _fetch_daily_source(source, symbol, start_date, end_date, include_trading_value=False):
    if not _daily_source_uses_process(source):
        return _fetch_daily_source_worker(
            source,
            symbol,
            start_date,
            end_date,
            include_trading_value,
        )
    pool = _get_daily_process_pool()
    try:
        return pool.submit(
            _fetch_daily_source_worker,
            source,
            symbol,
            start_date,
            end_date,
            include_trading_value,
        ).result()
    except BrokenProcessPool:
        _reset_daily_process_pool()
        raise


def fetch_qfq_daily_records(symbol, start_date, end_date, include_trading_value=False, warn=None):
    """A 股前复权日线，change_pct 统一为 close-to-close 日涨跌幅。"""
    last_err = None
    for group in _enabled_daily_source_groups(include_trading_value=include_trading_value):
        for source in _ordered_daily_group_sources(group):
            try:
                records = retry_fetch(
                    _fetch_daily_source,
                    source,
                    symbol,
                    start_date,
                    end_date,
                    include_trading_value=include_trading_value,
                )
                _record_daily_source_success(source)
                return records
            except Exception as exc:
                last_err = exc
                disabled_now = _record_daily_source_failure(source)
                if warn:
                    warn(f"{symbol} {source}行情失败({exc})，切换下一数据源")
                    if disabled_now:
                        warn(f"{source}行情连续失败，当前进程后续请求将先跳过该源")
    raise last_err


# ─── 通用 JSON 文件工具 ───────────────────────────────────────

def load_json_file(path, default=None):
    """读取 JSON 文件；不存在或损坏时返回 default。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def write_json_file(path, payload, indent=2):
    """写 JSON 文件，自动创建父目录。"""
    fp = Path(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)


def merge_records_by_date(existing, new_records, date_key="date", overwrite_none=True):
    """按 date_key 合并记录并升序返回。

    overwrite_none=True 保持浅合并语义：新记录的 None 也会覆盖旧值。
    overwrite_none=False 用于本地兜底数据：新记录为 None 时保留旧字段。
    """
    by_date = {}
    for row in existing or []:
        if not isinstance(row, dict):
            continue
        date = row.get(date_key)
        if date:
            by_date[str(date)] = dict(row)
    for row in new_records or []:
        if not isinstance(row, dict):
            continue
        date = row.get(date_key)
        if not date:
            continue
        merged = by_date.setdefault(str(date), {})
        for key, value in row.items():
            if overwrite_none or value is not None or key not in merged:
                merged[key] = value
    return sorted(by_date.values(), key=lambda item: str(item.get(date_key, "")))
