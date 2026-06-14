"""
爬取中证800成分股历史日频数据（最多10年）。

数据字段:
  行情: 单日开高低收/成交量/成交额/涨跌幅/换手率
        (daily_open/daily_high/daily_low/daily_close/daily_volume/daily_amount/
         daily_change_pct/daily_turnover_rate)
  估值: 总市值(market_cap,亿元)、PE(TTM)(pe_ttm)、PE(静)(pe_static)、PB(pb)、PCF(pcf)
        — 来自百度接口，只覆盖近5年（非逐日，约914点），更早记录该字段为 None
存储路径: data/stock_data/CN_{code}_{name}.json 的 history 字段
增量更新: 已有文件只补充 end_date+1 至今的缺失数据
并发: 默认6线程，可通过 STOCK_THREAD_COUNT 环境变量调整

附带:
  - fetch_csi300_etf(): 爬取 510310 沪深300 ETF 累计净值作为基准，供数据刷新的基准对比
"""

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak

from stock_crawl_common import (
    strip_proxy_env,
    safe_float as _safe_float,
    retry_fetch as _retry_fetch,
    fetch_index_constituents,
    fetch_qfq_daily_records,
    find_stock_json,
    daily_payload_from_history_records,
    history_record_has_daily_ohlcv,
    history_record_is_snapshot_only,
    history_payload_from_records,
    latest_weekday_date,
    merge_records_by_date,
    prune_snapshot_only_history_records,
    stock_json_path,
    load_json_file,
    write_json_file,
)

DATA_DIR = Path("data/stock_data")
MAX_YEARS = 10
MAX_RETRIES = 3
DEFAULT_THREAD_COUNT = 6   # 不要调高：>6 易被东财/百度等境内接口限流甚至跑挂
VALUATION_FRESH_DAYS = 7   # 最近一条估值在该天数内则跳过重抓（百度估值非逐日，几天延迟对长线 PB/PE 选股可忽略）


def _env_int(name, default, minimum=1, maximum=None):
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < minimum:
        return default
    if maximum is not None:
        return min(value, maximum)
    return value


THREAD_COUNT = _env_int("STOCK_THREAD_COUNT", DEFAULT_THREAD_COUNT, maximum=20)


strip_proxy_env()

_print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


# ─── 中证800成分股 ────────────────────────────────────────────

def get_csi800_stocks():
    """获取中证800成分股列表，返回 {code: name}（中证官网优先，新浪回退）"""
    safe_print("正在获取中证800成分股列表...")
    result = fetch_index_constituents("000906")
    safe_print(f"共 {len(result)} 只成分股")
    return result


# ─── 文件 I/O（长历史写入 stock_data.history）──

def find_stock_file(code, name=None):
    return find_stock_json(DATA_DIR, code, name)


def load_stock_file(code, name):
    fp = find_stock_json(DATA_DIR, code, name)
    if fp is not None:
        data = load_json_file(fp, {})
        history = data.get("history") if isinstance(data, dict) else None
        if isinstance(history, dict):
            return history
        if isinstance(data, dict) and isinstance(data.get("records"), list):
            return data
    return {}


def save_stock_file(code, name, data):
    fp = find_stock_json(DATA_DIR, code, name) or stock_json_path(DATA_DIR, code, name)
    payload = load_json_file(fp, {})
    if not isinstance(payload, dict):
        payload = {}
    history = history_payload_from_records(
        code, name, data.get("records", []), "stock_crawl_price_valuation"
    )

    payload.setdefault("symbol", str(code).zfill(6))
    payload.setdefault("name", name)
    payload["history"] = history
    payload["daily"] = daily_payload_from_history_records(history.get("records", []))
    payload["history_refetched_at"] = datetime.now().isoformat()
    write_json_file(fp, payload)


# ─── 数据爬取 ─────────────────────────────────────────────────


def fetch_daily_range(symbol, start_date, end_date):
    """爬取 [start_date, end_date] 的日频数据，返回 records 列表。

    数据源回退链：东财 → 腾讯 → 新浪。腾讯无换手率；
    腾讯/新浪的涨跌幅由相邻收盘价推算，新浪源串行兜底以避免 MiniRacer 并发崩溃。
    """
    return fetch_qfq_daily_records(
        symbol,
        start_date,
        end_date,
        include_trading_value=True,
        warn=lambda message: safe_print(f"  [FALLBACK] {message}"),
    )


# 百度估值指标映射: 存盘字段名 → akshare indicator 参数
VALUATION_INDICATORS = {
    "market_cap": "总市值",
    "pe_ttm": "市盈率(TTM)",
    "pe_static": "市盈率(静)",
    "pb": "市净率",
    "pcf": "市现率",
}
VALUATION_FIELDS = tuple(VALUATION_INDICATORS.keys())


def fetch_valuation_data(symbol, period="近五年"):
    """拉取 5 个百度估值指标，返回 {date_str: {field: value, ...}}

    某个 indicator 失败不影响整体，该字段对所有日期置空。
    """
    by_date = {}
    for field, indicator in VALUATION_INDICATORS.items():
        try:
            df = _retry_fetch(
                ak.stock_zh_valuation_baidu,
                symbol=symbol, indicator=indicator, period=period,
            )
        except Exception as e:
            safe_print(f"  [VALUATION WARN] {symbol} {indicator}: {e}")
            continue

        if df is None or df.empty:
            continue

        for _, row in df.iterrows():
            d = row["date"]
            date_str = d.isoformat() if hasattr(d, "isoformat") else str(d)[:10]
            by_date.setdefault(date_str, {})[field] = _safe_float(row["value"])

        time.sleep(random.uniform(0.1, 0.25))

    return by_date


def merge_records(existing, new_records):
    """按日期字段级合并：同日期 dict 浅合并（新值覆盖同名字段、保留其他字段），按日期升序"""
    return merge_records_by_date(existing, new_records, overwrite_none=True)


def attach_valuation(records, val_by_date):
    """给 records 每条按 date 补齐 5 个估值字段（匹配不到的留 None）"""
    all_fields = list(VALUATION_INDICATORS.keys())
    for r in records:
        vals = val_by_date.get(r["date"], {})
        for f in all_fields:
            # 已有非 None 值时保留（避免被本次 partial 拉取的 None 覆盖）
            if r.get(f) is None:
                r[f] = vals.get(f)
    return records


def records_need_ohlcv_backfill(records):
    """Whether existing history is structurally missing OHLCV.

    Ignore market_snapshot-only rows and require a majority of rows to be
    incomplete before triggering a 10-year refetch. One dirty latest close
    should not cause a full backfill.
    """
    rows = [
        row for row in (records or [])
        if isinstance(row, dict) and not history_record_is_snapshot_only(row)
    ]
    if not rows:
        return False
    missing = sum(1 for row in rows if not history_record_has_daily_ohlcv(row))
    return missing > len(rows) / 2


# ─── 单股增量更新 ─────────────────────────────────────────────

def _decide_valuation_period(records, today):
    """决定估值拉取策略：
      - None    : 最后一条已有完整估值，或最近一条估值在 VALUATION_FRESH_DAYS 天内 → 跳过
      - "近一年": 估值落后 (VALUATION_FRESH_DAYS, 300] 天
      - "近五年": 首次 / 估值缺失 / 落后过多（百度接口最远可取 5 年）
    """
    if not records:
        return "近五年"

    # 最新一条已完整覆盖 → 无需再爬。选股依赖 PB/PE，不能只看 market_cap。
    if all(records[-1].get(field) is not None for field in VALUATION_FIELDS):
        return None

    # 找最新一条估值字段完整的日期
    latest_val_date = None
    for r in reversed(records):
        if all(r.get(field) is not None for field in VALUATION_FIELDS):
            latest_val_date = r["date"]
            break

    if latest_val_date is None:
        return "近五年"

    try:
        gap_days = (datetime.strptime(today, "%Y-%m-%d")
                    - datetime.strptime(latest_val_date, "%Y-%m-%d")).days
    except ValueError:
        return "近五年"

    # 估值足够新则跳过重抓：百度估值非逐日，几天延迟对长线 PB/PE 选股可忽略；
    # 超过阈值后再批量回补，不影响历史估值序列密度与优化器的 PIT 回测。
    if gap_days <= VALUATION_FRESH_DAYS:
        return None
    return "近一年" if gap_days <= 300 else "近五年"


def process_stock(code, name, idx, total):
    existing = load_stock_file(code, name)
    existing_records = existing.get("records", [])
    snapshot_removed = 0
    if existing_records:
        existing_records, snapshot_removed = prune_snapshot_only_history_records(existing_records)
    start_date = existing.get("start_date")
    end_date = existing.get("end_date")
    today = latest_weekday_date()
    if existing_records:
        existing_records = sorted(
            [
                row for row in existing_records
                if str(row.get("date", ""))[:10] <= today
            ],
            key=lambda row: str(row.get("date", "")),
        )
        if existing_records:
            start_date = start_date or existing_records[0].get("date")
            end_date = end_date or existing_records[-1].get("date")
        else:
            start_date = None
            end_date = None

    max_start = (
        datetime.strptime(today, "%Y-%m-%d") - timedelta(days=365 * MAX_YEARS)
    ).strftime("%Y-%m-%d")

    hist_new_records = []

    full_daily_backfill = bool(existing_records and records_need_ohlcv_backfill(existing_records))
    if full_daily_backfill:
        safe_print(f"[{idx}/{total}] {code} {name}: 回补OHLCV {max_start} ~ {today}")
        hist_new_records.extend(fetch_daily_range(code, max_start, today))

    if not full_daily_backfill and existing_records and start_date and start_date > max_start:
        prev_day = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        if max_start <= prev_day:
            safe_print(f"[{idx}/{total}] {code} {name}: 回补历史 {max_start} ~ {prev_day}")
            hist_new_records.extend(fetch_daily_range(code, max_start, prev_day))

    if full_daily_backfill:
        pass
    elif end_date:
        if end_date < today:
            next_day = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            safe_print(f"[{idx}/{total}] {code} {name}: 补行情 {next_day} ~ {today}")
            hist_new_records.extend(fetch_daily_range(code, next_day, today))
    else:
        safe_print(f"[{idx}/{total}] {code} {name}: 首次爬取行情 {max_start} ~ {today}")
        hist_new_records.extend(fetch_daily_range(code, max_start, today))

    merged = merge_records(existing_records, hist_new_records)
    if not merged:
        safe_print(f"[{idx}/{total}] {code} {name}: 无数据")
        return

    # 估值：按 merged（合并后最新状态）判断是否需要拉取及拉取哪个period
    val_period = _decide_valuation_period(merged, today)
    if val_period is not None:
        val_by_date = fetch_valuation_data(code, period=val_period)
        if val_by_date:
            merged = attach_valuation(merged, val_by_date)

    result = {
        "symbol": code,
        "name": name,
        "start_date": merged[0]["date"],
        "end_date": merged[-1]["date"],
        "records": merged,
    }
    save_stock_file(code, name, result)

    n_new = len(merged) - len(existing_records)
    val_info = f"估值={val_period}" if val_period else "估值=skip"
    clean_info = f", 清理snapshot={snapshot_removed}" if snapshot_removed else ""
    safe_print(
        f"[{idx}/{total}] {code} {name}: "
        f"{merged[0]['date']} ~ {merged[-1]['date']} "
        f"({len(merged)}条, +{n_new}, {val_info}{clean_info})"
    )


# ─── 沪深300 ETF 基准 ─────────────────────────────────────────
#
# 用 510310（华泰柏瑞沪深300ETF）作为沪深300近似基准，爬近12年日频累计净值，
# 持久化到 data/csi300_etf_nav.json。由 stock_data_refresh 的基准对比步骤调用。

CSI300_ETF_CODE = "510310"
CSI300_ETF_YEARS = 12
CSI300_FILE = Path(__file__).resolve().parent / "data" / "csi300_etf_nav.json"


def fetch_csi300_etf(years=CSI300_ETF_YEARS):
    """爬取 510310 近 N 年累计净值并持久化，返回 records 列表"""
    # 基金净值爬取/合并工具在 fund 侧，延迟导入避免常规爬取路径背上依赖
    from fund_fetch_data import fetch_range, merge_records as merge_nav_records

    CSI300_FILE.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if CSI300_FILE.exists():
        with open(CSI300_FILE, encoding="utf-8") as f:
            existing = [
                r for r in json.load(f).get("records", [])
                if isinstance(r, dict) and r.get("date")
            ]
            existing.sort(key=lambda r: r["date"])

    today = latest_weekday_date()
    start = (
        datetime.strptime(today, "%Y-%m-%d") - timedelta(days=365 * years + 10)
    ).strftime("%Y-%m-%d")

    rows = []
    if not existing:
        safe_print(f"[沪深300] 首次爬取 {CSI300_ETF_CODE} {start} ~ {today} …")
        rows.extend(fetch_range(CSI300_ETF_CODE, start, today))
    else:
        first_date = existing[0]["date"]
        last_date = existing[-1]["date"]
        if first_date > start:
            before_first = (
                datetime.strptime(first_date, "%Y-%m-%d") - timedelta(days=1)
            ).strftime("%Y-%m-%d")
            if start <= before_first:
                safe_print(f"[沪深300] 补前段 {CSI300_ETF_CODE} {start} ~ {before_first} …")
                rows.extend(fetch_range(CSI300_ETF_CODE, start, before_first))
        if last_date < today:
            after_last = (
                datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            if after_last <= today:
                safe_print(f"[沪深300] 补后段 {CSI300_ETF_CODE} {after_last} ~ {today} …")
                rows.extend(fetch_range(CSI300_ETF_CODE, after_last, today))
        if not rows:
            safe_print(f"[沪深300] 已覆盖 {CSI300_ETF_CODE} {first_date} ~ {last_date}")

    merged = merge_nav_records(existing, rows)

    if not merged:
        raise RuntimeError("510310 爬取失败，无数据")

    with open(CSI300_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "code": CSI300_ETF_CODE,
            "target_years": years,
            "start_date": merged[0]["date"],
            "end_date": merged[-1]["date"],
            "records": merged,
        }, f, ensure_ascii=False)

    safe_print(f"[沪深300] 共 {len(merged)} 条（{merged[0]['date']} ~ {merged[-1]['date']}）")
    return merged


# ─── 主流程 ───────────────────────────────────────────────────

def main():
    stocks = get_csi800_stocks()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    items = list(stocks.items())
    total = len(items)
    completed = 0
    failed = []
    counter_lock = threading.Lock()

    def worker(args):
        nonlocal completed
        idx_local, (code, name) = args
        try:
            process_stock(code, name, idx_local, total)
            time.sleep(random.uniform(0.1, 0.3))
        except Exception as e:
            safe_print(f"[ERROR] {code} {name}: {e}")
            with counter_lock:
                failed.append((code, name))
        finally:
            with counter_lock:
                completed += 1
                if completed % 100 == 0:
                    safe_print(f"── 进度 {completed}/{total} ──")

    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        futures = [executor.submit(worker, (i + 1, item)) for i, item in enumerate(items)]
        for f in as_completed(futures):
            pass

    safe_print(f"\n完成！成功: {total - len(failed)}/{total}")
    if failed:
        safe_print(f"失败 {len(failed)} 只:")
        for code, name in failed:
            safe_print(f"  {code} {name}")
    return failed


if __name__ == "__main__":
    failed_stocks = main()
    if failed_stocks:
        raise SystemExit(1)
