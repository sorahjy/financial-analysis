"""
爬取中证800成分股历史日频数据（最多10年）。

数据字段:
  行情: 收盘价(close)、涨跌幅(change_pct)、换手率(turnover_rate)
  估值: 总市值(market_cap,亿元)、PE(TTM)(pe_ttm)、PE(静)(pe_static)、PB(pb)、PCF(pcf)
        — 来自百度接口，只覆盖近5年（非逐日，约914点），更早记录该字段为 None
存储路径: data/CN_stock/CN_{code}_{name}.json
增量更新: 已有文件只补充 end_date+1 至今的缺失数据
并发: 默认6线程，可通过 STOCK_THREAD_COUNT 环境变量调整
"""

import json
import math
import os
import random
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd

DATA_DIR = Path("data/CN_stock")
MAX_YEARS = 10
MAX_RETRIES = 3
DEFAULT_THREAD_COUNT = 6


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

_print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def _safe_float(val):
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    except (ValueError, TypeError):
        return None


def _retry_fetch(func, *args, retries=MAX_RETRIES, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt + 1)
            else:
                raise


# ─── 中证800成分股 ────────────────────────────────────────────

def get_csi800_stocks():
    """获取中证800成分股列表，返回 {code: name}"""
    safe_print("正在获取中证800成分股列表...")
    df = _retry_fetch(ak.index_stock_cons, symbol="000906")
    result = {}
    for _, row in df.iterrows():
        code = str(row["品种代码"]).zfill(6)
        name = str(row["品种名称"])
        result[code] = name
    safe_print(f"共 {len(result)} 只成分股")
    return result


# ─── 文件 I/O ────────────────────────────────────────────────

def _safe_name_component(name):
    text = str(name).strip()
    text = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", text)
    return text.strip(" ._") or "UNKNOWN"


def _filepath(code, name):
    return DATA_DIR / f"CN_{code}_{_safe_name_component(name)}.json"


def find_stock_file(code, name=None):
    candidates = []
    if name is not None:
        candidates.append(_filepath(code, name))
        legacy = DATA_DIR / f"CN_{code}_{name}.json"
        if legacy not in candidates:
            candidates.append(legacy)
    candidates.extend(sorted(DATA_DIR.glob(f"CN_{code}_*.json")))

    for fp in candidates:
        if fp.exists():
            return fp
    return None


def load_stock_file(code, name):
    fp = find_stock_file(code, name)
    if fp is not None:
        with open(fp, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_stock_file(code, name, data):
    fp = _filepath(code, name)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# ─── 数据爬取 ─────────────────────────────────────────────────

def fetch_daily_range(symbol, start_date, end_date):
    """爬取 [start_date, end_date] 的日频数据，返回 records 列表"""
    df = _retry_fetch(
        ak.stock_zh_a_hist,
        symbol=symbol,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="qfq",
    )
    if df is None or df.empty:
        return []

    records = []
    for _, row in df.iterrows():
        records.append({
            "date": str(row["日期"])[:10],
            "close": _safe_float(row["收盘"]),
            "change_pct": _safe_float(row["涨跌幅"]),
            "turnover_rate": _safe_float(row["换手率"]),
        })
    return records


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
    by_date = {}
    for r in existing:
        by_date[r["date"]] = dict(r)
    for r in new_records:
        d = r["date"]
        if d in by_date:
            by_date[d].update(r)
        else:
            by_date[d] = dict(r)
    return sorted(by_date.values(), key=lambda x: x["date"])


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


# ─── 单股增量更新 ─────────────────────────────────────────────

def _decide_valuation_period(records, today):
    """决定估值拉取策略：
      - None    : 最后一条（end_date）已有完整估值字段 → 跳过
      - "近一年": 估值落后 ≤300 天
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

    return "近一年" if gap_days <= 300 else "近五年"


def process_stock(code, name, idx, total):
    existing = load_stock_file(code, name)
    existing_records = existing.get("records", [])
    end_date = existing.get("end_date")

    today = datetime.now().strftime("%Y-%m-%d")
    max_start = (datetime.now() - timedelta(days=365 * MAX_YEARS)).strftime("%Y-%m-%d")

    hist_new_records = []

    if end_date:
        if end_date < today:
            next_day = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            safe_print(f"[{idx}/{total}] {code} {name}: 补行情 {next_day} ~ {today}")
            hist_new_records = fetch_daily_range(code, next_day, today)
    else:
        safe_print(f"[{idx}/{total}] {code} {name}: 首次爬取行情 {max_start} ~ {today}")
        hist_new_records = fetch_daily_range(code, max_start, today)

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
    safe_print(
        f"[{idx}/{total}] {code} {name}: "
        f"{merged[0]['date']} ~ {merged[-1]['date']} "
        f"({len(merged)}条, +{n_new}, {val_info})"
    )


# ─── 选股分析 ─────────────────────────────────────────────────

def _visual_width(s):
    """字符串在终端的显示宽度：东亚全/宽字符计 2，其他计 1"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in ("F", "W") else 1 for c in str(s))


def _pad_visual(s, width):
    """按显示宽度在右侧补空格，使不同中英文长度的列能对齐"""
    s = str(s)
    return s + " " * max(0, width - _visual_width(s))


def select_by_drawdown(start_time=None, max_drawdown_pct=40,
                       min_market_cap=500, max_market_cap=None,
                       max_pb=8, filter_negative_pe=True,
                       print_details=True, persist=True):
    """扫描 data/CN_stock/ 下所有股票，选出同时满足：
       (1) [start_time-2年, start_time] 窗口内最高收盘价 → start_time 当日收盘价的
           跌幅 > `max_drawdown_pct`
       (2) 市值（亿元）区间 `(min_market_cap, max_market_cap)`；任一端为 None 表示不限
       (3) PB < `max_pb`
       (4) 若 `filter_negative_pe=True`，剔除 PE(TTM) 为负的股票
       的股票；按跌幅降序打印 代码/名称/跌幅/市值/PE(TTM)/PB。

    Args:
        start_time: 窗口右端（'YYYY-MM-DD' 字符串或 datetime 对象），默认今天。
        max_drawdown_pct: 跌幅阈值（百分比），默认 40。
        min_market_cap: 市值下限（亿元），默认 500。None 表示不设下限。
        max_market_cap: 市值上限（亿元），默认 None 表示不设上限。
        max_pb: PB 上限，默认 8。PB 缺失的股票会被过滤掉。
        filter_negative_pe: 默认 True，过滤 PE(TTM) 为负的股票。
        print_details: 默认 True，在终端打印完整选股明细表；
            False 时只打印新增/保留/删除的 diff 摘要。

    Returns:
        list[dict]: 命中股票列表，按跌幅从大到小排序。
    """
    if start_time is None:
        start_time = datetime.now().strftime("%Y-%m-%d")
    elif isinstance(start_time, datetime):
        start_time = start_time.strftime("%Y-%m-%d")

    window_start = (datetime.strptime(start_time, "%Y-%m-%d")
                    - timedelta(days=365 * 2)).strftime("%Y-%m-%d")

    selected = []
    for fp in sorted(DATA_DIR.glob("CN_*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        records = data.get("records", [])
        window = [r for r in records if window_start <= r["date"] <= start_time]
        closes = [r.get("close") for r in window if r.get("close") is not None]
        if len(closes) < 2:
            continue

        # 窗口最高价 → 窗口末端价 的跌幅
        peak = max(closes)
        end_price = closes[-1]
        if peak <= 0:
            continue
        dd_pct = (peak - end_price) / peak * 100

        if dd_pct <= max_drawdown_pct:
            continue

        # 窗口内最新一条带 PE(TTM) / PB / market_cap 的记录（各字段独立取最新）
        pe = pb = mc = None
        for r in reversed(window):
            if pe is None and r.get("pe_ttm") is not None:
                pe = r["pe_ttm"]
            if pb is None and r.get("pb") is not None:
                pb = r["pb"]
            if mc is None and r.get("market_cap") is not None:
                mc = r["market_cap"]
            if pe is not None and pb is not None and mc is not None:
                break

        # 市值区间过滤：下限/上限任一为 None 表示不限
        if min_market_cap is not None and (mc is None or mc <= min_market_cap):
            continue
        if max_market_cap is not None and (mc is None or mc >= max_market_cap):
            continue
        # PB 过滤（缺失视为不合格）
        if pb is None or pb >= max_pb:
            continue
        # 负 PE 过滤（缺失不过滤）
        if filter_negative_pe and pe is not None and pe < 0:
            continue

        selected.append({
            "code": data.get("symbol", ""),
            "name": data.get("name", ""),
            "drop_from_peak": round(dd_pct, 2),
            "peak": round(peak, 2),
            "end_price": round(end_price, 2),
            "market_cap": mc,
            "pe_ttm": pe,
            "pb": pb,
        })

    selected.sort(key=lambda x: -x["drop_from_peak"])

    # 回测等场景下禁用持久化与终端输出
    if not persist:
        return selected

    out_fp = DATA_DIR / "selected_stocks.json"
    backup_fp = DATA_DIR / "selected_stocks_old.json"
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    # 读取旧选股并计算 diff
    old_stocks = []
    if out_fp.exists():
        try:
            with open(out_fp, encoding="utf-8") as f:
                old_stocks = json.load(f).get("stocks", [])
        except (json.JSONDecodeError, OSError):
            old_stocks = []
    old_by_code = {s.get("code", ""): s for s in old_stocks}
    new_by_code = {s["code"]: s for s in selected}
    old_codes = set(old_by_code)
    new_codes = set(new_by_code)
    added_codes = new_codes - old_codes
    removed_codes = old_codes - new_codes
    kept_codes = new_codes & old_codes

    # 备份旧文件再写新文件
    if out_fp.exists():
        shutil.copy2(out_fp, backup_fp)

    with open(out_fp, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "params": {
                "start_time": start_time,
                "window_start": window_start,
                "max_drawdown_pct": max_drawdown_pct,
                "min_market_cap": min_market_cap,
                "max_market_cap": max_market_cap,
                "max_pb": max_pb,
                "filter_negative_pe": filter_negative_pe,
            },
            "count": len(selected),
            "stocks": selected,
        }, f, ensure_ascii=False, indent=2)

    # 终端对齐输出
    mc_desc_parts = []
    if min_market_cap is not None:
        mc_desc_parts.append(f"市值>{min_market_cap}亿")
    if max_market_cap is not None:
        mc_desc_parts.append(f"市值<{max_market_cap}亿")
    mc_desc = " 且 ".join(mc_desc_parts) if mc_desc_parts else "市值不限"
    pe_desc = " 且 排除负PE" if filter_negative_pe else ""
    print("=" * 80)
    print(f"  近2年高点→{start_time} 跌幅 > {max_drawdown_pct}% "
          f"且 {mc_desc} 且 PB<{max_pb}{pe_desc} 的股票")
    print(f"  窗口 {window_start} ~ {start_time} · 命中 {len(selected)} 只")
    print("=" * 80)

    # Diff 摘要（对比上次 selected_stocks.json）
    print(
        f"变化：新增 {len(added_codes)} · 保留 {len(kept_codes)} · 删除 {len(removed_codes)}"
    )
    if added_codes:
        added_items = sorted(
            (new_by_code[c] for c in added_codes),
            key=lambda x: -x["drop_from_peak"],
        )
        print("  [新增]")
        for s in added_items:
            print(f"    + {s['code']} {s['name']}  跌幅 {s['drop_from_peak']:.2f}%")
    if removed_codes:
        removed_items = sorted(
            (old_by_code[c] for c in removed_codes),
            key=lambda x: -x.get("drop_from_peak", 0),
        )
        print("  [删除]")
        for s in removed_items:
            dd = s.get("drop_from_peak")
            dd_s = f"{dd:.2f}%" if isinstance(dd, (int, float)) else "N/A"
            print(f"    - {s.get('code','')} {s.get('name','')}  上次跌幅 {dd_s}")
    print("-" * 80)

    if print_details:
        header = (
            _pad_visual("代码", 8)
            + _pad_visual("名称", 14)
            + _pad_visual("跌幅%", 10)
            + _pad_visual("市值(亿)", 14)
            + _pad_visual("PE(TTM)", 12)
            + _pad_visual("PB", 8)
        )
        print(header)
        print("-" * 80)
        for s in selected:
            pe_s = f"{s['pe_ttm']:.2f}" if s["pe_ttm"] is not None else "N/A"
            pb_s = f"{s['pb']:.2f}" if s["pb"] is not None else "N/A"
            mc_s = f"{s['market_cap']:.2f}" if s["market_cap"] is not None else "N/A"
            dd_s = f"{s['drop_from_peak']:.2f}"
            line = (
                _pad_visual(s["code"], 8)
                + _pad_visual(s["name"], 14)
                + _pad_visual(dd_s, 10)
                + _pad_visual(mc_s, 14)
                + _pad_visual(pe_s, 12)
                + _pad_visual(pb_s, 8)
            )
            print(line)

    return selected


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
    select_by_drawdown()
    if failed_stocks:
        raise SystemExit(1)
