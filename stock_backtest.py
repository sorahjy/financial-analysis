"""
select_by_drawdown 选股策略的超额收益回测。

流程：
  1. 用 510310（华泰柏瑞沪深300ETF）作为沪深300近似基准，通过
     fund_fetch_nav_history.fetch_range 爬近5年日频累计净值，持久化到
     data/csi300_etf_nav.json。
  2. 在 [今天-3年, 今天-0.5年] 区间按 STEP_DAYS 滚动取时间点 t，对每个 t
     调用 select_by_drawdown(start_time=t, persist=False) 拿到选股结果。
  3. 等权买入持有 HOLD_DAYS 天，计算组合收益率、同期沪深300收益率、超额收益。
  4. 打印逐期明细并给出平均超额收益。
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

from fund_fetch_nav_history import fetch_range, merge_records
from stock_crawl_top_800_data import select_by_drawdown, DATA_DIR

CSI300_ETF_CODE = "510310"
CSI300_FILE = Path("data/csi300_etf_nav.json")

HOLD_DAYS = 182        # 持有约半年
STEP_DAYS = 30         # 回测时间点间隔（约每月1次）
LOOKBACK_MIN_DAYS = int(365 * 0.5)   # 最近端：半年前（留出持有期）
LOOKBACK_MAX_DAYS = 365 * 3          # 最远端：3年前


# ─── 沪深300 ETF 净值 ────────────────────────────────────────

def fetch_csi300_etf(years=5):
    """爬取 510310 近 N 年累计净值并持久化，返回 records 列表"""
    CSI300_FILE.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if CSI300_FILE.exists():
        with open(CSI300_FILE, encoding="utf-8") as f:
            existing = json.load(f).get("records", [])

    today = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * years + 10)).strftime("%Y-%m-%d")

    print(f"[沪深300] 爬取 {CSI300_ETF_CODE} {start} ~ {today} …")
    rows = fetch_range(CSI300_ETF_CODE, start, today)
    merged = merge_records(existing, rows)

    if not merged:
        raise RuntimeError(f"510310 爬取失败，无数据")

    with open(CSI300_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "code": CSI300_ETF_CODE,
            "start_date": merged[0]["date"],
            "end_date": merged[-1]["date"],
            "records": merged,
        }, f, ensure_ascii=False)

    print(f"[沪深300] 共 {len(merged)} 条（{merged[0]['date']} ~ {merged[-1]['date']}）")
    return merged


def _load_csi300_series():
    """返回按日期排序的 [(date, nav_acc)]"""
    with open(CSI300_FILE, encoding="utf-8") as f:
        records = json.load(f)["records"]
    series = []
    for r in records:
        try:
            nav = float(r.get("nav_acc"))
        except (TypeError, ValueError):
            continue
        series.append((r["date"], nav))
    series.sort(key=lambda x: x[0])
    return series


# ─── 选股收益计算 ────────────────────────────────────────────

_stock_cache = {}


def _load_stock_series(code, name):
    """返回按日期排序的 [(date, close)]，带缓存。"""
    key = (code, name)
    if key in _stock_cache:
        return _stock_cache[key]
    fp = DATA_DIR / f"CN_{code}_{name}.json"
    if not fp.exists():
        _stock_cache[key] = []
        return []
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    series = []
    for r in data.get("records", []):
        c = r.get("close")
        if c is not None:
            series.append((r["date"], c))
    series.sort(key=lambda x: x[0])
    _stock_cache[key] = series
    return series


def _price_on_or_before(series, target_date):
    """series 是升序 [(date, value)]。返回 date<=target 的最后一条 value；没有则 None。"""
    if not series:
        return None
    # 二分的简化版：线性倒扫足够（序列不大）
    for d, v in reversed(series):
        if d <= target_date:
            return v
    return None


def _portfolio_return(selected, t_str, hold_end_str):
    """等权组合在 [t_str, hold_end_str] 的收益率。返回 (ret, 有效数)"""
    rets = []
    for s in selected:
        series = _load_stock_series(s["code"], s["name"])
        p0 = _price_on_or_before(series, t_str)
        p1 = _price_on_or_before(series, hold_end_str)
        if p0 is None or p1 is None or p0 <= 0:
            continue
        rets.append(p1 / p0 - 1)
    if not rets:
        return None, 0
    return sum(rets) / len(rets), len(rets)


def _benchmark_return(series, t_str, hold_end_str):
    p0 = _price_on_or_before(series, t_str)
    p1 = _price_on_or_before(series, hold_end_str)
    if p0 is None or p1 is None or p0 <= 0:
        return None
    return p1 / p0 - 1


# ─── 主流程 ───────────────────────────────────────────────

def backtest(step_days=STEP_DAYS, hold_days=HOLD_DAYS):
    fetch_csi300_etf(years=5)
    csi = _load_csi300_series()

    today = datetime.now().date()
    t_start = today - timedelta(days=LOOKBACK_MAX_DAYS)
    t_end = today - timedelta(days=LOOKBACK_MIN_DAYS)

    print(
        f"\n回测区间: {t_start} ~ {t_end} · "
        f"步长 {step_days} 天 · 持有 {hold_days} 天\n"
    )
    print(
        f"{'日期':<12}{'选股数':>6}{'有效数':>6}"
        f"{'组合收益':>12}{'沪深300':>12}{'超额':>10}"
    )
    print("-" * 60)

    excess_list = []
    t = t_start
    while t <= t_end:
        t_str = t.strftime("%Y-%m-%d")
        hold_end_str = (t + timedelta(days=hold_days)).strftime("%Y-%m-%d")

        selected = select_by_drawdown(start_time=t_str, persist=False)
        if not selected:
            print(f"{t_str:<12}{'0':>6}{'-':>6}{'未选出':>12}")
            t += timedelta(days=step_days)
            continue

        port_ret, n_eff = _portfolio_return(selected, t_str, hold_end_str)
        bench_ret = _benchmark_return(csi, t_str, hold_end_str)

        if port_ret is None or bench_ret is None:
            print(f"{t_str:<12}{len(selected):>6}{n_eff:>6}{'价格缺失':>12}")
            t += timedelta(days=step_days)
            continue

        excess = port_ret - bench_ret
        excess_list.append(excess)
        print(
            f"{t_str:<12}{len(selected):>6}{n_eff:>6}"
            f"{port_ret*100:>+11.2f}%{bench_ret*100:>+11.2f}%"
            f"{excess*100:>+9.2f}%"
        )
        t += timedelta(days=step_days)

    print("-" * 60)
    if excess_list:
        avg = sum(excess_list) / len(excess_list)
        win = sum(1 for e in excess_list if e > 0)
        print(
            f"样本数 {len(excess_list)} · 胜率 {win}/{len(excess_list)} "
            f"({win/len(excess_list)*100:.1f}%) · "
            f"平均超额收益 {avg*100:+.2f}%"
        )
    else:
        print("无有效样本")


if __name__ == "__main__":
    backtest()
