"""
爬取细分行业龙头成分股历史日频数据（最多10年）。

股票集来自 `stock_crawl_segment_leaders.py`（细分行业龙头爬虫）的选股结果，
不再用中证800成分。get_segment_leader_stocks() 直接调用 build_segment_leader_pool 函数。
另对龙头池做**财报增量刷新**：yjbb 检测有新报告 / 超 90 天 / 从未爬过的票，才在 process_stock
里逐只补财务三表/指标/分红(嵌入式)；检测用 yjbb 公告日，数据仍逐只爬(yjbb 不当数据源)。

数据字段:
  行情: 单日开高低收/成交量/成交额/涨跌幅/换手率
        (daily_open/daily_high/daily_low/daily_close/daily_volume/daily_amount/
         daily_change_pct/daily_turnover_rate)
  估值: 总市值(market_cap,亿元)、PE(TTM)(pe_ttm)、PE(静)(pe_static)、PB(pb)、PCF(pcf)
        — 来自百度接口，只覆盖近5年（非逐日，约914点），更早记录该字段为 None
存储路径: data/stock_data/CN_{code}_{name}.json 的 history 字段
增量更新: 已有文件只补充 end_date+1 至今的缺失数据
并发: 默认32线程(stock 级)，可通过 STOCK_THREAD_COUNT 环境变量调整(上限64)；日线源另有独立进程池

附带:
  - fetch_benchmark_etfs(): 爬取 510310 沪深300 ETF、510580 中证500 ETF 累计净值作为基准
"""

import json
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak

import stock_storage as ss
from stock_crawl_common import (
    strip_proxy_env,
    safe_print,
    safe_float as _safe_float,
    retry_fetch as _retry_fetch,
    fetch_qfq_daily_records,
    daily_payload_from_history_records,
    history_record_has_daily_ohlcv,
    history_record_is_snapshot_only,
    history_payload_from_records,
    latest_weekday_date,
    merge_records_by_date,
    prune_snapshot_only_history_records,
)

MAX_YEARS = 10
MAX_RETRIES = 3
# stock 级线程多为网络等待 + GIL 绑定的 Python 规范化/合并：按 ~2× 核数自适应并封顶，
# 远超核数只会让线程抢 GIL 互拖、CPU 空转（境内行情接口也更易触发限流）。可用 STOCK_THREAD_COUNT 覆盖。
DEFAULT_THREAD_COUNT = min(32, (os.cpu_count() or 4) * 2)
VALUATION_FRESH_DAYS = 7   # 最近一条估值在该天数内则跳过重抓（百度估值非逐日，几天延迟对长线 PB/PE 选股可忽略）


def _date_dash(value):
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text[:10]


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


THREAD_COUNT = _env_int("STOCK_THREAD_COUNT", DEFAULT_THREAD_COUNT, maximum=64)
DB_WRITE_QUEUE_SIZE = _env_int("STOCK_DB_WRITE_QUEUE_SIZE", 64, minimum=1, maximum=512)
SEGMENT_REFRESH_SLICE = _env_int("STOCK_SEGMENT_REFRESH_SLICE", 15, minimum=0, maximum=336)


strip_proxy_env()


# ─── 细分行业龙头选股 ─────────────────────────────────────────

def get_segment_leader_stocks():
    """运行细分行业龙头选股，返回 {code: name}。

    直接调用 build_segment_leader_pool 函数(非子进程)拿到龙头池，再把各申万三级行业的
    top 龙头去重，作为本次要爬历史的股票集。默认滚动刷新最旧的 15 个 membership；
    可用 STOCK_SEGMENT_REFRESH_SLICE 覆盖。
    """
    from stock_crawl_segment_leaders import (
        build_segment_leader_pool,
        DEFAULT_TOP_PER_SEGMENT,
    )
    safe_print("正在运行细分行业龙头选股(crawl)...")
    payload = build_segment_leader_pool(
        top_per_segment=DEFAULT_TOP_PER_SEGMENT,
        refresh_slice=SEGMENT_REFRESH_SLICE,
    )
    stocks = {}
    for seg in payload.get("segments", []):
        for leader in seg.get("leaders", []):
            code = str(leader.get("code", "")).zfill(6)
            if code:
                stocks.setdefault(code, leader.get("name", ""))
    safe_print(f"共 {len(stocks)} 只细分龙头候选")
    return stocks


# ─── 数据 I/O（长历史写入 stock_history，SQLite）──

def load_stock_file(code, name):
    conn = ss.thread_conn()
    records = ss.load_history_records(conn, str(code).zfill(6))
    if not records:
        return {}
    return {
        "records": records,
        "start_date": records[0]["date"],
        "end_date": records[-1]["date"],
    }


def save_stock_file(code, name, data):
    conn = ss.thread_conn()
    full_records = data.get("records", [])
    has_fundamentals = (
        any(data.get(k) is not None for k in ("financials", "indicators", "dividends", "pledge"))
        or data.get("financials_refetched_at")
    )
    # 增量快路：纯日线 append(history_replace=False)且本轮没刷财报时，直接只 upsert 新增日线行
    # + 刷新 daily.stats/history 元，跳过「load_stock 读回整只 → 把没变的 financials/indicators/
    # dividends JSON blob parse 再 dump → 整行重写 stock_meta」这一整套往返开销。
    if full_records and data.get("history_replace") is False and not has_fundamentals:
        write_records = data.get("history_write_records", full_records)
        ss.upsert_history_records(
            conn,
            str(code).zfill(6),
            name,
            write_records,
            source="stock_crawl_price_valuation",
            daily_stats=daily_payload_from_history_records(full_records).get("stats"),
        )
        return

    # 读回整只(保留已有 financials/indicators/... 等 meta blob)，按需更新 history/daily 与财报。
    # include_history=False：本次的全量历史已在内存 data["records"] 里，不必再读回数万行日线。
    payload = ss.load_stock(conn, str(code).zfill(6), include_history=False) or {}
    payload.setdefault("symbol", str(code).zfill(6))
    payload.setdefault("name", name)
    replace_history = True
    if full_records:
        # history 元信息(start/end)与 daily.stats 取全量序列；实际写入的行可能只是增量(append)。
        write_records = data.get("history_write_records", full_records)
        replace_history = data.get("history_replace", True)
        history = history_payload_from_records(code, name, full_records, "stock_crawl_price_valuation")
        history["start_date"] = full_records[0]["date"]
        history["end_date"] = full_records[-1]["date"]
        # fetch_daily_range 返回 open/high/low/close/volume/amount 原始键；
        # SQLite 只写 daily_* canonical 列，增量写入也必须先规范化。
        history["records"] = history_payload_from_records(
            code, name, write_records, "stock_crawl_price_valuation"
        ).get("records", [])
        payload["history"] = history
        payload["daily"] = daily_payload_from_history_records(full_records)
        payload["history_refetched_at"] = datetime.now().isoformat()
    # 本次若顺带嵌入式刷新了财报，覆盖写入对应字段 + 新鲜度时间戳
    for key in ("financials", "indicators", "dividends", "pledge"):
        if data.get(key) is not None:
            payload[key] = data[key]
    if data.get("financials_refetched_at"):
        payload["financials_refetched_at"] = data["financials_refetched_at"]
    # 无新日线(仅刷财报)→不写 history；增量→replace_history=False 只 upsert 新增行，不整段重写
    ss.save_stock(conn, payload, replace_history=replace_history, write_history=bool(full_records))


_DB_WRITE_STOP = object()


class StockDbWriter:
    """单写线程：抓取线程只入队，SQLite 写入在这里顺序落库。"""

    def __init__(self, maxsize=DB_WRITE_QUEUE_SIZE):
        self.queue = queue.Queue(maxsize=maxsize)
        self.enqueued = 0
        self.completed = 0
        self.failed = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="stock-db-writer", daemon=True)

    def start(self):
        self._thread.start()

    def enqueue(self, code, name, data):
        self.queue.put((code, name, data))
        with self._lock:
            self.enqueued += 1
            enqueued = self.enqueued
            completed = self.completed
        if enqueued % 100 == 0:
            safe_print(f"── 写库排队 {enqueued}，已写 {completed}，队列 {self.queue.qsize()} ──")

    def wait(self):
        self.queue.join()
        self.queue.put(_DB_WRITE_STOP)
        self._thread.join()

    def _run(self):
        while True:
            task = self.queue.get()
            try:
                if task is _DB_WRITE_STOP:
                    return
                code, name, data = task
                try:
                    save_stock_file(code, name, data)
                except Exception as exc:
                    safe_print(f"[WRITE-ERROR] {code} {name}: {exc}")
                    with self._lock:
                        self.failed.append((code, name))
                else:
                    with self._lock:
                        self.completed += 1
                        completed = self.completed
                    if completed % 100 == 0:
                        safe_print(f"── 写库进度 {completed}/{self.enqueued}，队列 {self.queue.qsize()} ──")
            finally:
                self.queue.task_done()


# ─── 数据爬取 ─────────────────────────────────────────────────


def fetch_daily_range(symbol, start_date, end_date):
    """爬取 [start_date, end_date] 的日频数据，返回 records 列表。
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


# 估值指标共享线程池：把单只的 5 个百度指标并发拉(单只延迟 ~5x→1x)，同时把对百度的
# 全局并发钳在 VALUATION_POOL_WORKERS(默认=THREAD_COUNT)，不随 stock 线程数 ×5 爆炸。
VALUATION_POOL_WORKERS = _env_int("STOCK_VALUATION_WORKERS", THREAD_COUNT, maximum=64)
_VALUATION_POOL = None
_VALUATION_POOL_LOCK = threading.Lock()


def _valuation_pool():
    global _VALUATION_POOL
    if _VALUATION_POOL is None:
        with _VALUATION_POOL_LOCK:
            if _VALUATION_POOL is None:
                _VALUATION_POOL = ThreadPoolExecutor(
                    max_workers=VALUATION_POOL_WORKERS, thread_name_prefix="valuation")
    return _VALUATION_POOL


def _fetch_one_valuation_indicator(symbol, field, indicator, period):
    try:
        df = _retry_fetch(ak.stock_zh_valuation_baidu, symbol=symbol, indicator=indicator, period=period)
    except Exception as exc:
        safe_print(f"  [VALUATION WARN] {symbol} {indicator}: {exc}")
        return field, None
    return field, df


def fetch_valuation_data(symbol, period="近五年"):
    """拉取 5 个百度估值指标，返回 {date_str: {field: value, ...}}。

    5 个指标走共享线程池并发拉；某个失败不影响整体，该字段对所有日期置空。
    """
    by_date = {}
    futures = [
        _valuation_pool().submit(_fetch_one_valuation_indicator, symbol, field, indicator, period)
        for field, indicator in VALUATION_INDICATORS.items()
    ]
    for future in futures:
        field, df = future.result()
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            d = row["date"]
            date_str = d.isoformat() if hasattr(d, "isoformat") else str(d)[:10]
            by_date.setdefault(date_str, {})[field] = _safe_float(row["value"])
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


def process_stock(code, name, idx, total, *, need_fundamentals=False, pledge_info=None, save_callback=None):
    existing = load_stock_file(code, name)
    existing_records = existing.get("records", [])
    raw_count = len(existing_records)  # C: 增量快路用——现有行是否被 prune/未来日期过滤改动
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

    result = {"symbol": code, "name": name}
    val_period = None
    if merged:
        # 估值：按 merged（合并后最新状态）判断是否需要拉取及拉取哪个period
        val_period = _decide_valuation_period(merged, today)
        if val_period is not None:
            val_by_date = fetch_valuation_data(code, period=val_period)
            if val_by_date:
                merged = attach_valuation(merged, val_by_date)
        result["start_date"] = merged[0]["date"]
        result["end_date"] = merged[-1]["date"]
        result["records"] = merged
        # C: 增量 append 快路——估值未变(val_period=None)、现有行未被裁剪/过滤、且非全量回补
        # (全量回补会覆盖已有日期) 时，只 upsert 新增日期行，不 DELETE+重写整段 10 年历史。
        existing_unchanged = len(existing_records) == raw_count
        existing_dates = {r["date"] for r in existing_records}
        new_rows = [r for r in merged if r["date"] not in existing_dates]
        # 全无变化(估值新、无新交易日、未 prune、非回补、不刷财报)→ 重写会原样落回同样数据，
        # 直接跳过：免去对「已最新」股票的整段 DELETE+重插（重复跑/周末/停牌股尤其明显）。
        if (val_period is None and existing_unchanged and not new_rows
                and not full_daily_backfill and not need_fundamentals):
            safe_print(f"[{idx}/{total}] {code} {name}: 已最新({end_date})，跳过写库")
            return
        if val_period is None and existing_unchanged and new_rows and not full_daily_backfill:
            result["history_write_records"] = new_rows
            result["history_replace"] = False
        else:
            result["history_write_records"] = merged
            result["history_replace"] = True

    # 嵌入式财报刷新：仅当本只被判定需要刷(新报告/超期/从未爬)时才逐只爬
    if need_fundamentals:
        from stock_crawl_fundamentals import fetch_fundamentals
        result.update(fetch_fundamentals(code, pledge_info))

    if not merged and not need_fundamentals:
        safe_print(f"[{idx}/{total}] {code} {name}: 无数据")
        return

    queued_write = save_callback is not None
    if save_callback is None:
        save_stock_file(code, name, result)
    else:
        save_callback(code, name, result)

    parts = []
    if merged:
        n_new = len(merged) - len(existing_records)
        val_info = f"估值={val_period}" if val_period else "估值=skip"
        clean_info = f", 清理snapshot={snapshot_removed}" if snapshot_removed else ""
        write_mode = "增量" if result.get("history_replace") is False else "重写"
        parts.append(
            f"{merged[0]['date']} ~ {merged[-1]['date']} ({len(merged)}条, +{n_new}, {write_mode}, {val_info}{clean_info})"
        )
    if need_fundamentals:
        parts.append("财报已刷新")
    write_state = "已入写库队列" if queued_write else "已写库"
    safe_print(f"[{idx}/{total}] {code} {name}: {write_state} | " + " | ".join(parts))


# ─── ETF 基准 ─────────────────────────────────────────────────
#
# 用 510310（华泰柏瑞沪深300ETF）和 510580（易方达中证500ETF）作为
# 长线优化器的 50/50 等权基准成分，从 2012-01-01 起爬日频累计净值并落库 index_nav。

CSI300_ETF_CODE = "510310"
BENCHMARK_ETF_START_DATE = "2012-01-01"
CSI300_FILE = Path(__file__).resolve().parent / "data" / "csi300_etf_nav.json"
CSI500_ETF_CODE = "510580"
CSI500_FILE = Path(__file__).resolve().parent / "data" / "csi500_etf_nav.json"
BENCHMARK_ETFS = (
    {
        "code": CSI300_ETF_CODE,
        "label": "沪深300",
        "file": CSI300_FILE,
        "start_date": BENCHMARK_ETF_START_DATE,
    },
    {
        "code": CSI500_ETF_CODE,
        "label": "中证500",
        "file": CSI500_FILE,
        "start_date": BENCHMARK_ETF_START_DATE,
    },
)


def fetch_index_etf_nav(code, output_file, *, label, start_date=BENCHMARK_ETF_START_DATE, years=None):
    """爬取单只指数 ETF 累计净值并持久化，返回 records 列表。"""
    # 基金净值爬取/合并工具在 fund 侧，延迟导入避免常规爬取路径背上依赖
    from fund_fetch_data import fetch_range, merge_records as merge_nav_records

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            existing = [
                r for r in json.load(f).get("records", [])
                if isinstance(r, dict) and r.get("date")
            ]
            existing.sort(key=lambda r: r["date"])

    today = latest_weekday_date()
    if start_date:
        start = _date_dash(start_date)
    elif years:
        start = (
            datetime.strptime(today, "%Y-%m-%d") - timedelta(days=365 * years + 10)
        ).strftime("%Y-%m-%d")
    else:
        start = BENCHMARK_ETF_START_DATE

    rows = []
    if not existing:
        safe_print(f"[{label}] 首次爬取 {code} {start} ~ {today} …")
        rows.extend(fetch_range(code, start, today))
    else:
        first_date = existing[0]["date"]
        last_date = existing[-1]["date"]
        if first_date > start:
            before_first = (
                datetime.strptime(first_date, "%Y-%m-%d") - timedelta(days=1)
            ).strftime("%Y-%m-%d")
            if start <= before_first:
                safe_print(f"[{label}] 补前段 {code} {start} ~ {before_first} …")
                rows.extend(fetch_range(code, start, before_first))
        if last_date < today:
            after_last = (
                datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            if after_last <= today:
                safe_print(f"[{label}] 补后段 {code} {after_last} ~ {today} …")
                rows.extend(fetch_range(code, after_last, today))
        if not rows:
            safe_print(f"[{label}] 已覆盖 {code} {first_date} ~ {last_date}")

    merged = merge_nav_records(existing, rows)

    if not merged:
        raise RuntimeError(f"{code} 爬取失败，无数据")

    payload = {
        "code": code,
        "target_years": years,
        "target_start_date": start,
        "start_date": merged[0]["date"],
        "end_date": merged[-1]["date"],
        "records": merged,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    # 同步落库 index_nav（优化器基准走 DB，JSON 留作回滚）
    nav_conn = ss.connect()
    try:
        ss.save_index_nav(nav_conn, payload)
    finally:
        nav_conn.close()

    safe_print(f"[{label}] 共 {len(merged)} 条（{merged[0]['date']} ~ {merged[-1]['date']}）")
    return merged


def fetch_csi300_etf(start_date=BENCHMARK_ETF_START_DATE, years=None):
    """爬取 510310 累计净值并持久化，返回 records 列表。"""
    return fetch_index_etf_nav(CSI300_ETF_CODE, CSI300_FILE, label="沪深300", start_date=start_date, years=years)


def fetch_csi500_etf(start_date=BENCHMARK_ETF_START_DATE, years=None):
    """爬取 510580 累计净值并持久化，返回 records 列表。"""
    return fetch_index_etf_nav(CSI500_ETF_CODE, CSI500_FILE, label="中证500", start_date=start_date, years=years)


def fetch_benchmark_etfs():
    """爬取优化器所需的全部 ETF 基准成分，返回 {code: records}。"""
    results = {}
    for item in BENCHMARK_ETFS:
        results[item["code"]] = fetch_index_etf_nav(
            item["code"],
            item["file"],
            label=item["label"],
            start_date=item["start_date"],
        )
    return results


# ─── 主流程 ───────────────────────────────────────────────────

def _plan_fundamentals_refresh(codes):
    """进线程池前 bulk 准备财报刷新计划：DB 上次刷新时间 + yjbb 公告日 + 全市场质押，
    算出每只龙头是否需要刷财报。返回 (needs:{code:bool}, pledge_map)。

    yjbb/质押 接口失败时(返回空)自动降级为'仅超期兜底刷新'，不影响主流程。
    """
    from stock_crawl_fundamentals import (
        fetch_pledge_data_bulk,
        fetch_latest_report_announce_dates,
        needs_fundamentals_refresh,
        FUNDAMENTALS_EXPIRE_DAYS,
    )
    conn = ss.connect()
    try:
        refetched = ss.financials_refetched_map(conn, list(codes))
    finally:
        conn.close()
    safe_print("[财报] 拉取业绩报表(yjbb)检测新报告 + 全市场质押...")
    announce = fetch_latest_report_announce_dates()
    pledge_map = fetch_pledge_data_bulk()
    needs = {
        code: needs_fundamentals_refresh(refetched.get(code), announce.get(code),
                                         expire_days=FUNDAMENTALS_EXPIRE_DAYS)
        for code in codes
    }
    n = sum(1 for v in needs.values() if v)
    safe_print(f"[财报] {n}/{len(codes)} 只需刷新财报(新报告/超{FUNDAMENTALS_EXPIRE_DAYS}天/从未爬)")
    return needs, pledge_map


def main():
    stocks = get_segment_leader_stocks()
    needs_fund, pledge_map = _plan_fundamentals_refresh(set(stocks))

    items = list(stocks.items())
    total = len(items)
    completed = 0
    failed = []
    counter_lock = threading.Lock()
    db_writer = StockDbWriter()
    db_writer.start()

    def worker(args):
        nonlocal completed
        idx_local, (code, name) = args
        try:
            process_stock(code, name, idx_local, total,
                          need_fundamentals=needs_fund.get(code, False),
                          pledge_info=pledge_map.get(code),
                          save_callback=db_writer.enqueue)
        except Exception as e:
            safe_print(f"[ERROR] {code} {name}: {e}")
            with counter_lock:
                failed.append((code, name))
        finally:
            with counter_lock:
                completed += 1
                if completed % 100 == 0:
                    safe_print(
                        f"── 抓取进度 {completed}/{total}，"
                        f"写库已排队 {db_writer.enqueued} / 已写 {db_writer.completed} "
                        f"/ 队列 {db_writer.queue.qsize()} ──"
                    )

    try:
        with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            futures = [executor.submit(worker, (i + 1, item)) for i, item in enumerate(items)]
            for f in as_completed(futures):
                pass
    finally:
        safe_print(
            f"抓取线程完成，等待写库队列清空："
            f"已排队 {db_writer.enqueued} / 已写 {db_writer.completed} / 队列 {db_writer.queue.qsize()}"
        )
        db_writer.wait()

    # 整轮写完后批量把最新总市值同步进 sw3_member（取代旧的每只 save 都查大表同步单只）。
    try:
        synced = ss.sync_sw3_member_market_caps(ss.thread_conn())
        safe_print(f"sw3 市值批量同步: {synced} 行")
    except Exception as exc:
        safe_print(f"[WARN] sw3 市值批量同步失败: {exc}")

    fetch_failed_count = len(failed)
    failed.extend(db_writer.failed)
    safe_print(
        f"\n完成！抓取成功: {total - fetch_failed_count}/{total}；"
        f"写库成功: {db_writer.completed}/{db_writer.enqueued}；"
        f"写库失败: {len(db_writer.failed)}"
    )
    if failed:
        safe_print(f"失败 {len(failed)} 只:")
        for code, name in failed:
            safe_print(f"  {code} {name}")
    return failed


if __name__ == "__main__":
    failed_stocks = main()
    if failed_stocks:
        raise SystemExit(1)
