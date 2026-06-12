
"""
T+1 / T+2 龙虎榜共振策略回测。

模式
─────
  replay   从 data/capital/snapshots/scored_<date>.json 读取已保存的当日评分。
           适合：日常跑 stock_crawl_capital.py 后累积快照，每周/每月汇总验收。
  rebuild  对历史日期重新调用 crawl_main_capital_stocks(as_of_date=t)，
           现拉当时的龙虎榜 + K 线（K 线会按 as_of_date 截断防止未来函数）。
           适合：一次性补全历史回测，每个时间点 1~5 分钟。

执行模型（贴近实盘）
──────────────────
  T 日（上榜日）收盘后，运行评分得出 Strong Buy 名单
  T+1 日开盘价买入（开盘集合竞价，加 0.1% 滑点）
  T+1+hold 日收盘价卖出（默认 hold=1 即 T+2 收盘；hold=2 即 T+3 收盘）
  跳过：T+1 一字板（无法成交）、停牌（K 线缺失）
  等权分配资金，按日复利累计

输出
────
  - 每个买点的实际收益
  - 整体胜率、平均盈/亏比、累计收益、最大回撤
  - 与基准（默认沪深 300）超额收益对比
  - 各因子（连板/共振席位数/评级）分组胜率，定位策略弱点
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median

import akshare as ak

from stock_crawl_capital import (
    DATA_DIR, SNAPSHOT_DIR,
    _num, _retry,
    crawl_main_capital_stocks,
)

DEFAULT_BENCHMARK = "sh000300"   # 沪深300指数
DEFAULT_SLIPPAGE = 0.001          # 0.1% 单边滑点
DEFAULT_COMMISSION = 0.0003       # 万3 单边手续费，买卖各收一次

BACKTEST_DIR = DATA_DIR / "backtest"


# ─── K 线缓存 ──────────────────────────────────────────────────

_kline_cache = {}


def fetch_kline_range(code, start_date, end_date):
    """拉 [start, end] 日 K 线（前复权，东财源）。内存缓存。"""
    key = (code, start_date, end_date)
    if key in _kline_cache:
        return _kline_cache[key]
    df = _retry(
        ak.stock_zh_a_hist,
        symbol=code,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="qfq",
    )
    if df is None or df.empty:
        _kline_cache[key] = []
        return []
    records = []
    prev_c = None
    for _, row in df.iterrows():
        c = _num(row["收盘"])
        chg = _num(row.get("涨跌幅"))
        if chg == 0 and prev_c:
            chg = (c / prev_c - 1) * 100
        records.append({
            "date":  str(row["日期"])[:10],
            "open":  _num(row["开盘"]),
            "high":  _num(row["最高"]),
            "low":   _num(row["最低"]),
            "close": c,
            "change_pct": chg,
        })
        prev_c = c
    _kline_cache[key] = records
    return records


# ─── 交易日工具 ────────────────────────────────────────────────

def next_calendar_day(date_str, n=1):
    """字符串日期 + n 天（自然日，不考虑节假日）"""
    d = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=n)
    return d.strftime("%Y-%m-%d")


# ─── 单笔模拟 ──────────────────────────────────────────────────

def simulate_trade(code, name, buy_intent_date, hold_days=1,
                   slippage=DEFAULT_SLIPPAGE, commission=DEFAULT_COMMISSION):
    """模拟一笔交易：buy_intent_date 之后第 1 个交易日开盘买入，再过 hold_days 个交易日的收盘卖出。

    Returns:
        dict: {code, name, buy_date, sell_date, buy_price, sell_price, ret_pct}
        或   {code, name, buy_date, skipped: <reason>}
    """
    # 留足 hold_days + 缓冲（跨周末/节假日）
    end = next_calendar_day(buy_intent_date, hold_days + 10)
    start = next_calendar_day(buy_intent_date, -5)
    recs = fetch_kline_range(code, start, end)
    recs = [r for r in recs if r["date"] > buy_intent_date]
    if len(recs) < hold_days + 1:
        return {"code": code, "name": name, "buy_date": buy_intent_date,
                "skipped": "K线不足"}

    buy_rec = recs[0]
    # T+1 一字板：开盘=最高=最低，集合竞价根本买不到
    if buy_rec["high"] > 0:
        spread = (buy_rec["high"] - buy_rec["low"]) / buy_rec["high"]
        if spread < 0.005 and buy_rec["change_pct"] >= 9.5:
            return {"code": code, "name": name, "buy_date": buy_rec["date"],
                    "skipped": "T+1 一字板"}

    sell_rec = recs[hold_days]
    buy_price = buy_rec["open"] * (1 + slippage)
    sell_price = sell_rec["close"] * (1 - slippage)
    ret = (sell_price / buy_price - 1 - commission * 2) * 100
    return {
        "code":       code,
        "name":       name,
        "buy_date":   buy_rec["date"],
        "sell_date":  sell_rec["date"],
        "buy_price":  round(buy_price, 3),
        "sell_price": round(sell_price, 3),
        "ret_pct":    round(ret, 3),
    }


# ─── 单快照回测 ────────────────────────────────────────────────

def backtest_snapshot(snapshot, top_n=10, hold_days=1, only_strong=True,
                      min_concurrent=None, max_consec_lb=None):
    """对单个 scored snapshot 做模拟，返回 [trade, ...]。

    Args:
        only_strong:     只买 Strong Buy（默认 True）
        min_concurrent:  筛掉并发席位<min_concurrent 的（None 不过滤）
        max_consec_lb:   筛掉连板>max_consec_lb 的（None 不过滤）
    """
    picks = snapshot.get("stocks", [])
    if only_strong:
        picks = [s for s in picks if s["scores"]["verdict"] == "Strong Buy"]
    if min_concurrent is not None:
        picks = [s for s in picks if s.get("concurrent_count", 0) >= min_concurrent]
    if max_consec_lb is not None:
        picks = [s for s in picks
                 if s["scores"].get("consecutive_limit_up", 0) <= max_consec_lb]
    picks = picks[:top_n]

    buy_intent = next_calendar_day(snapshot["as_of_date"], 1)
    trades = []
    for p in picks:
        t = simulate_trade(p["code"], p["name"], buy_intent, hold_days)
        t["score"] = p["scores"]["total"]
        t["verdict"] = p["scores"]["verdict"]
        t["concurrent_count"] = p.get("concurrent_count", 0)
        t["consec_lb"] = p["scores"].get("consecutive_limit_up", 0)
        t["weighted_score"] = p.get("weighted_score", 0)
        trades.append(t)
    return trades


# ─── 汇总指标 ──────────────────────────────────────────────────

def summarize_trades(trades):
    valid = [t for t in trades if "ret_pct" in t]
    skipped = [t for t in trades if "skipped" in t]
    if not valid:
        return {"n_trades": 0, "n_skipped": len(skipped),
                "skip_reasons": dict(_count_reasons(skipped))}

    rets = [t["ret_pct"] for t in valid]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]

    # 等权按日复利（同日多笔取均值作组合当日回报）
    by_buy_date = defaultdict(list)
    for t in valid:
        by_buy_date[t["buy_date"]].append(t["ret_pct"])

    cumulative = 1.0
    cum_curve = []
    peak = 1.0
    max_dd = 0.0
    for d in sorted(by_buy_date):
        daily_ret = mean(by_buy_date[d]) / 100
        cumulative *= (1 + daily_ret)
        cum_curve.append((d, round(cumulative, 5)))
        peak = max(peak, cumulative)
        dd = (peak - cumulative) / peak
        max_dd = max(max_dd, dd)

    return {
        "n_trades":           len(valid),
        "n_skipped":          len(skipped),
        "skip_reasons":       dict(_count_reasons(skipped)),
        "win_rate":           round(len(wins) / len(valid) * 100, 2),
        "avg_ret":            round(mean(rets), 3),
        "median_ret":         round(median(rets), 3),
        "avg_win":            round(mean(wins), 3) if wins else 0,
        "avg_loss":           round(mean(losses), 3) if losses else 0,
        "profit_loss_ratio":  round(abs(mean(wins) / mean(losses)), 2)
                              if (wins and losses) else None,
        "cum_return":         round((cumulative - 1) * 100, 2),
        "max_drawdown":       round(max_dd * 100, 2),
        "cum_curve":          cum_curve,
    }


def _count_reasons(skipped):
    counts = defaultdict(int)
    for t in skipped:
        counts[t["skipped"]] += 1
    return counts


# ─── 因子分组分析（定位策略弱点）──────────────────────────────

def factor_breakdown(trades):
    """按 评级 / 并发数 / 连板数 分组算胜率与平均收益。"""
    valid = [t for t in trades if "ret_pct" in t]
    out = {}

    def _group(label, key_fn, buckets):
        grouped = defaultdict(list)
        for t in valid:
            key = key_fn(t)
            for bucket_name, pred in buckets:
                if pred(key):
                    grouped[bucket_name].append(t["ret_pct"])
                    break
        rows = []
        for name, _ in buckets:
            rs = grouped[name]
            if not rs:
                rows.append((name, 0, None, None))
            else:
                wins = [r for r in rs if r > 0]
                rows.append((name, len(rs), round(len(wins)/len(rs)*100, 1),
                             round(mean(rs), 2)))
        out[label] = rows

    _group("评级", lambda t: t["verdict"], [
        ("Strong Buy", lambda v: v == "Strong Buy"),
        ("Watch",      lambda v: v == "Watch"),
    ])
    _group("并发席位", lambda t: t.get("concurrent_count", 0), [
        ("≥4",  lambda x: x >= 4),
        ("3",   lambda x: x == 3),
        ("2",   lambda x: x == 2),
        ("≤1",  lambda x: x <= 1),
    ])
    _group("连板数", lambda t: t.get("consec_lb", 0), [
        ("0 (普通)",  lambda x: x == 0),
        ("1 (首板)",  lambda x: x == 1),
        ("2 (二板)",  lambda x: x == 2),
        ("≥3",        lambda x: x >= 3),
    ])
    _group("总分桶", lambda t: t["score"], [
        ("≥85",     lambda x: x >= 85),
        ("80~85",   lambda x: 80 <= x < 85),
        ("78~80",   lambda x: 78 <= x < 80),
        ("<78",     lambda x: x < 78),
    ])
    return out


# ─── 基准 ──────────────────────────────────────────────────────

def fetch_benchmark(symbol, start, end):
    df = _retry(ak.stock_zh_index_daily, symbol=symbol)
    if df is None or df.empty:
        return None
    df["date"] = df["date"].astype(str).str[:10]
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    if len(df) < 2:
        return None
    o = float(df.iloc[0]["open"])
    c = float(df.iloc[-1]["close"])
    return round((c / o - 1) * 100, 2)


# ─── 模式入口 ──────────────────────────────────────────────────

def list_snapshots(start=None, end=None):
    snaps = []
    for fp in sorted(SNAPSHOT_DIR.glob("scored_*.json")):
        date_str = fp.stem.replace("scored_", "")
        if start and date_str < start:
            continue
        if end and date_str > end:
            continue
        snaps.append((date_str, fp))
    return snaps


def run_replay(top_n, hold_days, start, end, only_strong,
               min_concurrent, max_consec_lb):
    snaps = list_snapshots(start, end)
    if not snaps:
        print(f"[ERROR] {SNAPSHOT_DIR} 内找不到快照（范围 {start}~{end}）")
        print("        请先运行 stock_crawl_capital.py 累积几天的 snapshots，")
        print("        或使用 --mode rebuild 重建历史。")
        return None, []
    print(f"加载 {len(snaps)} 个快照: {snaps[0][0]} ~ {snaps[-1][0]}")

    all_trades = []
    for date, fp in snaps:
        snap = json.load(open(fp))
        trades = backtest_snapshot(snap, top_n=top_n, hold_days=hold_days,
                                   only_strong=only_strong,
                                   min_concurrent=min_concurrent,
                                   max_consec_lb=max_consec_lb)
        valid = [t for t in trades if "ret_pct" in t]
        skipped = len(trades) - len(valid)
        avg = round(mean([t["ret_pct"] for t in valid]), 2) if valid else None
        print(f"  {date}: {len(valid)} trades (avg {avg}%), {skipped} skipped")
        all_trades.extend(trades)
    return summarize_trades(all_trades), all_trades


def run_rebuild(top_n, hold_days, start, end, step, only_strong,
                min_concurrent, max_consec_lb,
                days=14, top_n_yyb=20, sort_by="appearances",
                min_followers=2, window_days=2):
    """对 [start, end] 区间每 step 天重建一次评分快照，然后回测。"""
    if not start or not end:
        raise ValueError("rebuild 模式必须传 --start 与 --end")
    cur = datetime.strptime(start, "%Y-%m-%d")
    stop = datetime.strptime(end, "%Y-%m-%d")
    all_trades = []
    while cur <= stop:
        as_of = cur.strftime("%Y-%m-%d")
        if cur.weekday() >= 5:   # 周末跳过
            cur += timedelta(days=1)
            continue
        print(f"\n=== Rebuild as_of {as_of} ===")
        try:
            ranked = crawl_main_capital_stocks(
                days=days, top_n_yyb=top_n_yyb, sort_by=sort_by,
                min_followers=min_followers, score_top=top_n,
                window_days=window_days, exclude_channel=True,
                as_of_date=as_of, persist=True,
            )
        except Exception as e:
            print(f"  [SKIP] {as_of} 重建失败: {e}")
            cur += timedelta(days=step)
            continue
        snap = {"as_of_date": as_of, "stocks": _ranked_to_snapshot(ranked)}
        trades = backtest_snapshot(snap, top_n=top_n, hold_days=hold_days,
                                   only_strong=only_strong,
                                   min_concurrent=min_concurrent,
                                   max_consec_lb=max_consec_lb)
        all_trades.extend(trades)
        cur += timedelta(days=step)
    return summarize_trades(all_trades), all_trades


def _ranked_to_snapshot(ranked):
    """把 in-memory ranked 列表压成 backtest_snapshot 能用的 stocks 字段。"""
    out = []
    for s in ranked:
        cinfo = s.get("concurrent_info") or {}
        out.append({
            "code": s["code"], "name": s["name"],
            "scores": s["scores"],
            "concurrent_count": cinfo.get("concurrent_count", 0),
            "weighted_score":   cinfo.get("weighted_score", 0),
        })
    return out


# ─── 打印 / 落盘 ───────────────────────────────────────────────

def print_summary(s, title=""):
    print()
    print("=" * 80)
    print(f" {title}")
    print("=" * 80)
    if s.get("n_trades", 0) == 0:
        print(f"  无有效交易（跳过 {s.get('n_skipped', 0)}）")
        for reason, n in s.get("skip_reasons", {}).items():
            print(f"    {reason}: {n}")
        return
    print(f"  交易数:    {s['n_trades']} (跳过 {s['n_skipped']})")
    for reason, n in s.get("skip_reasons", {}).items():
        print(f"    └─ {reason}: {n}")
    print(f"  胜率:      {s['win_rate']}%")
    print(f"  平均收益:  {s['avg_ret']}%   中位数: {s['median_ret']}%")
    print(f"  均胜:      {s['avg_win']}%   均亏: {s['avg_loss']}%   盈亏比: {s.get('profit_loss_ratio')}")
    print(f"  累计收益:  {s['cum_return']}%   最大回撤: {s['max_drawdown']}%")


def print_factor_breakdown(bk):
    print()
    print("=" * 80)
    print(" 因子分组（定位策略弱点）")
    print("=" * 80)
    for label, rows in bk.items():
        print(f"\n  [{label}]")
        print(f"    {'桶':<14}{'笔数':>6}{'胜率%':>10}{'均收益%':>12}")
        for name, n, wr, avg in rows:
            wr_s = f"{wr}" if wr is not None else "-"
            avg_s = f"{avg}" if avg is not None else "-"
            print(f"    {name:<14}{n:>6}{wr_s:>10}{avg_s:>12}")


def persist_backtest(summary, trades, factor_bk, tag):
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    fp = BACKTEST_DIR / f"backtest_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fp, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "factor_breakdown": factor_bk,
                   "trades": trades}, f, ensure_ascii=False, indent=2)
    print(f"\n  → 回测结果落盘 {fp}")


def main():
    parser = argparse.ArgumentParser(description="龙虎榜共振策略回测")
    parser.add_argument("--mode", choices=["replay", "rebuild"], default="replay")
    parser.add_argument("--top", type=int, default=10, help="每日取 Top N picks")
    parser.add_argument("--hold", type=int, default=1,
                        help="持有交易日数：1=T+2收盘卖, 2=T+3收盘卖")
    parser.add_argument("--start", default=None, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--step", type=int, default=3,
                        help="rebuild 模式：每隔多少天跑一次评分，默认 3")
    parser.add_argument("--all-verdicts", action="store_true",
                        help="扩到所有评级（含 Watch），默认仅 Strong Buy")
    parser.add_argument("--min-concurrent", type=int, default=None,
                        help="筛掉并发席位 < 此值的股票")
    parser.add_argument("--max-consec-lb", type=int, default=None,
                        help="筛掉连板 > 此值的股票")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK,
                        help="对比基准指数代码（默认 sh000300 沪深300）")
    parser.add_argument("--no-persist", action="store_true",
                        help="不把回测结果落盘到 data/capital/backtest/")
    args = parser.parse_args()

    print(f"模式: {args.mode} · Top {args.top} · 持有 {args.hold} 交易日 (T+{args.hold+1} 卖)")
    print(f"筛选: only_strong={not args.all_verdicts}"
          f" min_concurrent={args.min_concurrent} max_consec_lb={args.max_consec_lb}")

    if args.mode == "replay":
        summary, trades = run_replay(
            top_n=args.top, hold_days=args.hold,
            start=args.start, end=args.end,
            only_strong=not args.all_verdicts,
            min_concurrent=args.min_concurrent,
            max_consec_lb=args.max_consec_lb,
        )
    else:
        summary, trades = run_rebuild(
            top_n=args.top, hold_days=args.hold,
            start=args.start, end=args.end, step=args.step,
            only_strong=not args.all_verdicts,
            min_concurrent=args.min_concurrent,
            max_consec_lb=args.max_consec_lb,
        )

    if not summary:
        return

    title = f"策略回测汇总 (Top{args.top} · 持{args.hold}日)"
    print_summary(summary, title=title)
    bk = factor_breakdown(trades)
    print_factor_breakdown(bk)

    # 基准对比
    valid = [t for t in trades if "ret_pct" in t]
    if valid:
        first_buy = min(t["buy_date"] for t in valid)
        last_sell = max(t["sell_date"] for t in valid)
        try:
            bm = fetch_benchmark(args.benchmark, first_buy, last_sell)
            if bm is not None:
                excess = round(summary["cum_return"] - bm, 2)
                print(f"\n  基准 {args.benchmark} 同期: {bm}%   超额: {excess}%")
        except Exception as e:
            print(f"\n  [WARN] 基准获取失败: {e}")

    if not args.no_persist:
        persist_backtest(summary, trades, bk, tag=args.mode)


if __name__ == "__main__":
    main()