"""
爬取龙虎榜数据：活跃营业部（主力席位）、每日明细、机构买卖统计、个股上榜统计，
按共振席位聚合候选池，并落盘短线策略需要的原始 signals（龙虎榜/机构/统计/技术面）。

数据来源（东方财富网 - 数据中心 - 龙虎榜单）:
    akshare.stock_lhb_hyyyb_em            每日活跃营业部（席位 → 买入股票）
    akshare.stock_lhb_detail_em           每日明细（净买额/上榜原因/流通市值）
    akshare.stock_lhb_jgmmtj_em           机构买卖每日统计
    akshare.stock_lhb_stock_statistic_em  个股上榜统计（近一月）
    akshare.stock_lhb_yybph_em            营业部排行（用于标记高活跃/知名席位）
"""

import argparse
import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak

from stock_crawl_common import (
    analysis_records_from_history_records,
    find_stock_json,
    history_record_has_daily_ohlcv,
    load_json_file,
    normalize_history_records,
    retry_fetch_or_none as _retry,
    safe_num as _num,
    strip_proxy_env,
    symbol_with_prefix as _symbol_with_prefix,
)

KLINE_LOOKBACK = 60      # 取最近 60 个交易日（足够算 MA20 / RSI / 距高点）
MAX_RETRIES = 3
THREAD_COUNT = 6         # 不要调高：>6 易被龙虎榜/K线接口限流甚至跑挂
DATA_DIR = Path("data/capital")
STOCK_DATA_DIR = Path("data/stock_data")
CANDIDATES_FILE = DATA_DIR / "hot_money_candidates.json"
KNOWN_SEAT_TOP_N = 100   # 营业部排行前 N 标记为高活跃/知名席位


strip_proxy_env()

# 新浪 K 线接口底层走 mini_racer / V8 执行 JS 加密，V8 不是线程安全
# 多线程并发会 segfault → 用全局锁串行化 JS 执行段。
_kline_lock = threading.Lock()


def _visual_width(s):
    """字符串在终端的显示宽度：东亚全/宽字符计 2，其他计 1"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in ("F", "W") else 1 for c in str(s))


def _pad_visual(s, width):
    """按显示宽度在右侧补空格，使不同中英文长度的列能对齐"""
    s = str(s)
    return s + " " * max(0, width - _visual_width(s))


def fetch_active_yyb(days=14):
    """拉取最近 `days` 个自然日内每日活跃营业部记录。

    Returns:
        pandas.DataFrame: 列含 营业部名称/上榜日/买入个股数/卖出个股数/
            买入总金额/卖出总金额/总买卖净额/买入股票/营业部代码
    """
    end = datetime.now()
    start = end - timedelta(days=days)
    return ak.stock_lhb_hyyyb_em(
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
    )


def fetch_lhb_detail(days=14):
    """龙虎榜每日明细，按股票聚合。

    Returns:
        dict[code]: {count, total_net_buy, max_net_pct, reasons,
                     float_cap, max_buy_ratio, net_buy_to_float_pct}
    """
    end = datetime.now()
    start = end - timedelta(days=days)
    df = _retry(ak.stock_lhb_detail_em,
                start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
    out = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        code = str(row.get("代码", "")).zfill(6)
        if len(code) != 6 or not code.isdigit():
            continue
        d = out.setdefault(code, {
            "count": 0, "total_net_buy": 0.0, "max_net_pct": None,
            "reasons": [], "float_cap": None, "max_buy_ratio": None,
        })
        d["count"] += 1
        d["total_net_buy"] += _num(row.get("龙虎榜净买额"))
        pct = _num(row.get("净买额占总成交比"))
        if d["max_net_pct"] is None or pct > d["max_net_pct"]:
            d["max_net_pct"] = pct
        reason = str(row.get("上榜原因") or "").strip()
        if reason and reason not in d["reasons"]:
            d["reasons"].append(reason)
        float_cap = _num(row.get("流通市值"))
        if float_cap > 0:
            d["float_cap"] = float_cap
        lhb_buy = _num(row.get("龙虎榜买入额"))
        lhb_amount = _num(row.get("龙虎榜成交额"))
        if lhb_amount > 0:
            ratio = lhb_buy / lhb_amount
            if d["max_buy_ratio"] is None or ratio > d["max_buy_ratio"]:
                d["max_buy_ratio"] = round(ratio, 4)
    for d in out.values():
        d["total_net_buy"] = round(d["total_net_buy"], 2)
        d["net_buy_to_float_pct"] = (
            round(d["total_net_buy"] / d["float_cap"] * 100, 4) if d.get("float_cap") else None
        )
    return out


def fetch_inst_stats(days=14):
    """机构买卖每日统计，按股票聚合 → {code: {inst_buy_count, inst_net_buy}}"""
    end = datetime.now()
    start = end - timedelta(days=days)
    df = _retry(ak.stock_lhb_jgmmtj_em,
                start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
    out = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        code = str(row.get("代码", "")).zfill(6)
        if len(code) != 6 or not code.isdigit():
            continue
        d = out.setdefault(code, {"inst_buy_count": 0, "inst_net_buy": 0.0})
        buyer_count = _num(row.get("买方机构数"))
        if buyer_count > 0:
            d["inst_buy_count"] += int(buyer_count)
        elif _num(row.get("机构买入总额")) > 0:
            d["inst_buy_count"] += 1
        d["inst_net_buy"] += _num(row.get("机构买入净额"))
    for d in out.values():
        d["inst_net_buy"] = round(d["inst_net_buy"], 2)
    return out


def fetch_stock_lhb_stat(symbol="近一月"):
    """个股上榜统计 → {code: {stat_count, stat_net_buy}}"""
    df = _retry(ak.stock_lhb_stock_statistic_em, symbol=symbol)
    out = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        code = str(row.get("代码", "")).zfill(6)
        if len(code) != 6 or not code.isdigit():
            continue
        out[code] = {
            "stat_count": _num(row.get("上榜次数")),
            "stat_net_buy": _num(row.get("龙虎榜净买额")),
        }
    return out


def fetch_known_seats(top_n=KNOWN_SEAT_TOP_N):
    """营业部排行（近一月）按买入活跃度取前 N，作为高活跃/知名席位名单。"""
    df = _retry(ak.stock_lhb_yybph_em, symbol="近一月")
    if df is None or df.empty:
        return set()
    col = "上榜后1天-买入次数"
    if col in df.columns:
        df = df.sort_values(col, ascending=False)
    return set(str(name) for name in df["营业部名称"].head(top_n))


def aggregate_yyb(df):
    """按营业部聚合：上榜次数、活跃日数、净买入、买入金额、买入股票合集。

    Returns:
        dict[str, dict]: 营业部名称 → 聚合字段
    """
    agg = defaultdict(lambda: {
        "appearances": 0,
        "net_amount": 0.0,
        "buy_amount": 0.0,
        "sell_amount": 0.0,
        "stocks": set(),
        "dates": set(),
    })
    for _, row in df.iterrows():
        name = str(row["营业部名称"])
        a = agg[name]
        a["appearances"] += 1
        a["net_amount"] += _num(row["总买卖净额"])
        a["buy_amount"] += _num(row["买入总金额"])
        a["sell_amount"] += _num(row["卖出总金额"])
        a["dates"].add(str(row["上榜日"])[:10])
        for s in str(row["买入股票"] or "").split():
            if s:
                a["stocks"].add(s)
    return agg


def print_top_yyb(agg, top_n=20, sort_by="appearances", stock_limit=20):
    """按 `sort_by` 排序打印前 `top_n` 个活跃席位与其涉及股票。

    Args:
        sort_by: "appearances"(上榜次数) / "net"(净买入) / "buy"(买入金额)
        stock_limit: 每个席位最多展示多少只买入股票，超出截断
    """
    items = _sort_items(agg, sort_by)

    print("=" * 100)
    print(f"  最近活跃营业部（主力席位）Top {top_n} · 按 {sort_by} 排序")
    print("=" * 100)
    header = (
        _pad_visual("营业部", 40)
        + _pad_visual("上榜次数", 10)
        + _pad_visual("活跃日数", 10)
        + _pad_visual("净买入(亿)", 14)
        + _pad_visual("买入额(亿)", 14)
    )
    print(header)
    print("-" * 100)
    for item in items[:top_n]:
        print(
            _pad_visual(item["name"], 40)
            + _pad_visual(str(item["appearances"]), 10)
            + _pad_visual(str(len(item["dates"])), 10)
            + _pad_visual(f"{item['net_amount']/1e8:.2f}", 14)
            + _pad_visual(f"{item['buy_amount']/1e8:.2f}", 14)
        )
        stocks = sorted(item["stocks"])
        shown = stocks[:stock_limit]
        more = f" ...+{len(stocks) - stock_limit}" if len(stocks) > stock_limit else ""
        print(f"    买入股票: {' '.join(shown)}{more}")
    print("=" * 100)


def _sort_items(agg, sort_by):
    """把 aggregate 字典展平为 list 并按指定字段排序。"""
    items = [{"name": n, **info} for n, info in agg.items()]
    keys = {
        "appearances": lambda x: (-x["appearances"], -x["net_amount"]),
        "net": lambda x: -x["net_amount"],
        "buy": lambda x: -x["buy_amount"],
    }
    items.sort(key=keys[sort_by])
    return items


def collect_followers_from_top_yyb(df, agg, top_n=20, sort_by="appearances", known_seats=None):
    """从 Top N 席位的逐日记录构造 股票名 → [席位明细] 映射。

    buy_est 为估算值：该席位当日买入总金额 / 当日买入个股数（东财活跃营业部
    接口不提供席位级单票金额）。category: knownHotMoney(高活跃席位) / top_active_seat。

    Returns:
        dict[str, list[dict]]: 股票名称 → [{seat, date, buy_est, category}, ...]
    """
    known_seats = known_seats or set()
    top_names = set(item["name"] for item in _sort_items(agg, sort_by)[:top_n])
    followers = defaultdict(list)
    for _, row in df.iterrows():
        name = str(row["营业部名称"])
        if name not in top_names:
            continue
        stocks = [s for s in str(row["买入股票"] or "").split() if s]
        if not stocks:
            continue
        buy_total = _num(row["买入总金额"])
        est = round(buy_total / len(stocks), 2) if buy_total > 0 else None
        date = str(row["上榜日"])[:10]
        category = "knownHotMoney" if name in known_seats else "top_active_seat"
        for stock in stocks:
            followers[stock].append(
                {"seat": name, "date": date, "buy_est": est, "category": category}
            )
    return dict(followers)


def summarize_followers(flist):
    """席位明细聚合：3日共振窗口、买方席位数、估算买入额与席位加权分。"""
    seats = set(f["seat"] for f in flist)
    dates = sorted(set(f["date"] for f in flist if f.get("date")))
    buy_total = sum(f["buy_est"] or 0.0 for f in flist)
    known_count = sum(1 for f in flist if f.get("category") == "knownHotMoney")

    concurrent = 0
    best_window = None
    for d0 in dates:
        d_end = (datetime.strptime(d0, "%Y-%m-%d") + timedelta(days=2)).strftime("%Y-%m-%d")
        win_seats = set(f["seat"] for f in flist if f.get("date") and d0 <= f["date"] <= d_end)
        if len(win_seats) > concurrent:
            concurrent = len(win_seats)
            win_dates = [d for d in dates if d0 <= d <= d_end]
            best_window = [win_dates[0], win_dates[-1]]

    # 启发式席位加权分：共振、席位广度、知名席位数与买入额体量的线性组合，
    # 只用于横截面百分位排序，绝对数值无含义。
    weighted = round(
        concurrent * 10 + len(seats) * 4 + known_count * 6 + min(buy_total, 2e9) / 1e8 * 8, 2
    )
    return {
        "concurrent_count": concurrent,
        "total_buyers": len(seats),
        "buy_amount_total": round(buy_total, 2) if buy_total > 0 else None,
        "weighted_score": weighted,
        "best_window": best_window,
        "known_seat_count": known_count,
    }


# ─── 股票名 → 代码映射（带缓存）─────────────────────────────────

_name_code_cache = None


def get_name_code_map():
    """全 A 股 股票名称 → 6 位代码 映射，进程内缓存一次。"""
    global _name_code_cache
    if _name_code_cache is None:
        df = ak.stock_info_a_code_name()  # 列: code, name
        _name_code_cache = {
            str(row["name"]): str(row["code"]).zfill(6)
            for _, row in df.iterrows()
        }
    return _name_code_cache


# ─── 单只股票数据爬取（K 线）────────────────────────────────────

def _has_complete_ohlcv(records):
    required = ("open", "high", "low", "close", "volume", "amount")
    return bool(records) and all(
        all(row.get(field) is not None for field in required)
        for row in records
    )


def load_local_kline_records(code, name=None, *, lookback=None, start_date=None):
    """Read complete OHLCV rows from data/stock_data.history when available."""
    fp = find_stock_json(STOCK_DATA_DIR, str(code).zfill(6), name)
    if fp is None:
        return []
    payload = load_json_file(fp, {})
    records = normalize_history_records(
        (payload.get("history") or {}).get("records", []),
        include_valuation=False,
    )
    if start_date:
        records = [row for row in records if row.get("date") >= start_date]
    records = [row for row in records if history_record_has_daily_ohlcv(row)]
    if lookback:
        records = records[-lookback:]
        if len(records) < lookback:
            return []
    analysis_records = analysis_records_from_history_records(records)
    return analysis_records if _has_complete_ohlcv(analysis_records) else []


def fetch_stock_data(code, name):
    """拉取最近 `KLINE_LOOKBACK` 个交易日的前复权日 K 线（新浪源）。

    Returns:
        dict | None: {code, name, records: [{date,open,high,low,close,volume,
            amount,change_pct,turnover_rate}, ...]}；失败返回 None。
    """
    local_records = load_local_kline_records(code, name, lookback=KLINE_LOOKBACK)
    if local_records:
        return {"code": code, "name": name, "records": local_records}

    with _kline_lock:
        df = _retry(ak.stock_zh_a_daily,
                    symbol=_symbol_with_prefix(code), adjust="qfq")
    if df is None or df.empty:
        return None

    df = df.tail(KLINE_LOOKBACK).reset_index(drop=True)
    records = []
    prev_close = None
    for _, row in df.iterrows():
        close = _num(row["close"])
        change_pct = ((close / prev_close - 1) * 100) if prev_close else 0.0
        records.append({
            "date":          str(row["date"])[:10],
            "open":          _num(row["open"]),
            "high":          _num(row["high"]),
            "low":           _num(row["low"]),
            "close":         close,
            "volume":        _num(row["volume"]),
            "amount":        _num(row["amount"]),
            "change_pct":    round(change_pct, 4),
            "turnover_rate": _num(row.get("turnover", 0)) * 100,
        })
        prev_close = close
    return {"code": code, "name": name, "records": records}


# ─── 打分模型 ───────────────────────────────────────────────────
# 目标：T+1 买入、T+2 卖出。短线对基本面不敏感，权重压在「资金共振 + 动量 + 位置 + 风控」。
# 借鉴 financial-services 仓库的「评分卡 / 多维度评估」范式，因子替换为短线指标。

def _ma(values, n):
    """简单移动均线最后一个值；不足 n 返回 None。"""
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def _rsi(closes, n=14):
    """RSI(n) 最后一个值，不足返回 None。"""
    if len(closes) <= n:
        return None
    gains = losses = 0.0
    for i in range(-n, 0):
        d = closes[i] - closes[i - 1]
        if d > 0:
            gains += d
        else:
            losses -= d
    if gains + losses == 0:
        return 50.0
    rs = gains / (losses + 1e-9)
    return 100 - 100 / (1 + rs)


def _ema(values, n):
    """简单指数移动平均（最后一个值）。"""
    if not values:
        return None
    k = 2 / (n + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def compute_tech_signals(records, code):
    """从 60 日 K 线提取短线技术面信号，落盘到 signals.tech。"""
    if len(records) < 21:
        return {}
    closes = [r["close"] for r in records]
    vols = [r["volume"] for r in records]
    limit_pct = 19.5 if code.startswith(("30", "68")) else 9.5  # 创业板/科创板20cm

    chg_today = records[-1]["change_pct"]
    chg_5d = (closes[-1] / closes[-6] - 1) * 100 if closes[-6] > 0 else None
    ma5, ma10, ma20 = _ma(closes, 5), _ma(closes, 10), _ma(closes, 20)
    dist_ma20 = (closes[-1] / ma20 - 1) * 100 if ma20 else None
    avg5 = sum(vols[-6:-1]) / 5 if len(vols) >= 6 else 0
    vol_ratio = vols[-1] / avg5 if avg5 > 0 else None
    rsi = _rsi(closes, 14)

    recent_limit_up = sum(1 for r in records[-10:] if r["change_pct"] >= limit_pct)
    consecutive = 0
    for r in reversed(records):
        if r["change_pct"] >= limit_pct:
            consecutive += 1
        else:
            break

    last = records[-1]
    spread = (last["high"] - last["low"]) / last["high"] if last["high"] > 0 else 1.0
    is_yizi = spread < 0.005 and last["change_pct"] >= limit_pct
    is_t = (
        not is_yizi
        and last["change_pct"] >= limit_pct
        and last["high"] > 0
        and abs(last["close"] - last["high"]) / last["high"] < 0.002
        and abs(last["open"] - last["high"]) / last["high"] < 0.002
    )

    macd_dif = None
    if len(closes) >= 26:
        macd_dif = round(_ema(closes, 12) - _ema(closes, 26), 4)

    return {
        "chg_today": round(chg_today, 3),
        "chg_5d": round(chg_5d, 3) if chg_5d is not None else None,
        "dist_from_ma20_pct": round(dist_ma20, 3) if dist_ma20 is not None else None,
        "rsi": round(rsi, 2) if rsi is not None else None,
        "vol_ratio": round(vol_ratio, 3) if vol_ratio is not None else None,
        "turnover_today": records[-1]["turnover_rate"] or None,
        "recent_limit_up": recent_limit_up,
        "ma_bull": bool(ma5 and ma10 and ma20 and ma5 > ma10 > ma20 and closes[-1] > ma5),
        "macd_dif": macd_dif,
        "consecutive_limit_up": consecutive,
        "is_yizi_ban": is_yizi,
        "is_t_ban": is_t,
    }


# ─── 候选排序 / 输出 / 落盘 ─────────────────────────────────────

def _candidate_sort_key(stock):
    """仅用于 CLI 展示和落盘稳定排序，不作为策略分数。"""
    return (
        -(stock.get("weighted_score") or 0.0),
        -(stock.get("concurrent_count") or 0),
        -(stock.get("buy_amount_total") or 0.0),
        stock.get("code") or "",
    )


def sort_candidates(stocks):
    """按席位聚合强度做稳定排序；正式分数由 stock_advanced_strategies 计算。"""
    return sorted(stocks, key=_candidate_sort_key)


def print_candidates(stocks, top_n=20):
    """打印游资候选池 Top N，不输出策略总分。"""
    ranked = sort_candidates(stocks)

    print("\n" + "=" * 110)
    print(f"  游资候选池 Top {top_n}（仅按席位聚合强度展示，非策略评分）")
    print("=" * 110)
    header = (
        _pad_visual("代码", 8)
        + _pad_visual("名称", 14)
        + _pad_visual("席位加权", 12)
        + _pad_visual("共振", 8)
        + _pad_visual("席位数", 8)
        + _pad_visual("知名席位", 10)
        + _pad_visual("估算买入(亿)", 14)
        + _pad_visual("窗口", 22)
    )
    print(header)
    print("-" * 110)
    for s in ranked[:top_n]:
        window = s.get("best_window") or []
        window_text = "~".join(window) if len(window) >= 2 else ""
        print(
            _pad_visual(s["code"], 8)
            + _pad_visual(s["name"], 14)
            + _pad_visual(f"{(s.get('weighted_score') or 0):.1f}", 12)
            + _pad_visual(str(s.get("concurrent_count") or 0), 8)
            + _pad_visual(str(s.get("total_buyers") or len(s.get("followers") or [])), 8)
            + _pad_visual(str(s.get("known_seat_count") or 0), 10)
            + _pad_visual(f"{(s.get('buy_amount_total') or 0)/1e8:.2f}", 14)
            + _pad_visual(window_text, 22)
        )
    print("=" * 110)
    print("注: 这里不再预打分；短线最终分数由 stock_advanced_strategies.py 统一计算。")
    return ranked


def persist_candidates(stocks, fp=None):
    """落盘游资候选池与原始 signals。"""
    fp = Path(fp) if fp is not None else CANDIDATES_FILE
    fp.parent.mkdir(parents=True, exist_ok=True)
    ranked = sort_candidates(stocks)
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of_date": datetime.now().strftime("%Y-%m-%d"),
        "schema": "hot_money_candidates.v1",
        "count": len(ranked),
        "stocks": [
            {
                "code": s["code"],
                "name": s["name"],
                "followers": s["followers"],
                "concurrent_count": s.get("concurrent_count"),
                "total_buyers": s.get("total_buyers"),
                "buy_amount_total": s.get("buy_amount_total"),
                "weighted_score": s.get("weighted_score"),
                "best_window": s.get("best_window"),
                "known_seat_count": s.get("known_seat_count"),
                "signals": s.get("signals"),
                # 不落 records 防止文件过大
            }
            for s in ranked
        ],
    }
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  → 已落盘 {fp} (共 {len(ranked)} 条)")


# ─── 编排：席位 → 股票 → 数据爬取 ───────────────────────────────

def crawl_main_capital_stocks(days=14, top_n_yyb=20, sort_by="appearances",
                              min_followers=2, score_top=20):
    """主流程：龙虎榜 → 共振股票池 → 并发拉 K 线 → 原始信号落盘。

    Args:
        min_followers: 至少被多少个 Top 席位共买才纳入候选池，默认 2（更聚焦共振）。
        score_top:     终端候选池展示前 N 条；保留参数名兼容原 CLI。

    Returns:
        list[dict]: 候选股票列表（已按席位聚合强度排序）。
    """
    df = fetch_active_yyb(days=days)
    print(f"共 {len(df)} 条上榜记录 · {df['营业部名称'].nunique()} 个不同营业部 · "
          f"{df['上榜日'].nunique()} 个交易日")

    agg = aggregate_yyb(df)
    print_top_yyb(agg, top_n=top_n_yyb, sort_by=sort_by)

    # 龙虎榜明细/机构/统计/活跃席位名单，任一失败不阻塞主流程（对应 signals 字段置空）
    print("\n拉取龙虎榜明细、机构统计与活跃席位名单...")

    def _safe_fetch(label, func, default, *args, **kwargs):
        try:
            out = func(*args, **kwargs)
            print(f"  {label}: {len(out)} 条")
            return out
        except Exception as e:
            print(f"  [WARN] {label} 获取失败: {e}")
            return default

    lhb_map = _safe_fetch("每日明细", fetch_lhb_detail, {}, days=days)
    inst_map = _safe_fetch("机构买卖统计", fetch_inst_stats, {}, days=days)
    stat_map = _safe_fetch("个股上榜统计", fetch_stock_lhb_stat, {})
    known_seats = _safe_fetch("高活跃席位名单", fetch_known_seats, set())

    follower_map = collect_followers_from_top_yyb(
        df, agg, top_n=top_n_yyb, sort_by=sort_by, known_seats=known_seats
    )
    pool = {
        s: flist for s, flist in follower_map.items()
        if len(set(f["seat"] for f in flist)) >= min_followers
    }
    print(f"\nTop {top_n_yyb} 席位共买入 {len(follower_map)} 只去重股票"
          f"（≥{min_followers} 个席位共买：{len(pool)} 只）")

    name_map = get_name_code_map()
    tasks, missing = [], []
    for stock_name in sorted(pool, key=lambda s: -len(set(f["seat"] for f in pool[s]))):
        code = name_map.get(stock_name)
        if not code:
            missing.append(stock_name)
            continue
        tasks.append((code, stock_name, sorted(pool[stock_name], key=lambda f: f["date"])))

    print(f"\n开始并发拉 K 线 + 生成原始信号（{len(tasks)} 只 / {THREAD_COUNT} 线程）...")
    results, failed = [], []
    completed = 0

    def worker(code, name, followers):
        data = fetch_stock_data(code, name)
        if not data or not data.get("records"):
            return ("FAIL", code, name, "empty kline")
        data["followers"] = followers
        data.update(summarize_followers(followers))
        tech = compute_tech_signals(data["records"], code)
        data["signals"] = {
            "lhb": lhb_map.get(code),
            "inst": inst_map.get(code),
            "stat": stat_map.get(code),
            "tech": tech or None,
        }
        return ("OK", data)

    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as ex:
        futures = [ex.submit(worker, c, n, f) for c, n, f in tasks]
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                failed.append(("?", "?", str(e)))
                res = None
            if res and res[0] == "OK":
                results.append(res[1])
            elif res:
                failed.append((res[1], res[2], res[3]))
            completed += 1
            if completed % 50 == 0:
                print(f"  进度 {completed}/{len(tasks)}")

    print(f"\n已生成候选 {len(results)} 只 · 跳过(无代码) {len(missing)} · 失败 {len(failed)}")
    if missing:
        sample = missing[:10]
        more = f" ...+{len(missing) - 10}" if len(missing) > 10 else ""
        print(f"  无代码样本: {' '.join(sample)}{more}")
    if failed:
        for code, name, err in failed[:5]:
            print(f"  [FAIL] {code} {name}: {err}")

    if not results:
        return []

    ranked = print_candidates(results, top_n=score_top)
    persist_candidates(ranked)
    return ranked


def main():
    parser = argparse.ArgumentParser(
        description="主力席位龙虎榜 → 共振股票池 → 原始信号候选池")
    parser.add_argument("--days", type=int, default=14,
                        help="拉取最近多少自然日的龙虎榜数据，默认 14")
    parser.add_argument("--top-yyb", type=int, default=20,
                        help="取活跃营业部 Top N，默认 20")
    parser.add_argument("--sort", choices=["appearances", "net", "buy"],
                        default="appearances",
                        help="席位排序依据：上榜次数/净买入/买入金额")
    parser.add_argument("--min-followers", type=int, default=2,
                        help="股票至少被多少个 Top 席位共买才纳入候选池，默认 2")
    parser.add_argument("--score-top", type=int, default=20,
                        help="候选池展示前 N 条，默认 20；参数名保留兼容原命令")
    args = parser.parse_args()

    print(f"拉取最近 {args.days} 天的活跃营业部数据...")
    crawl_main_capital_stocks(
        days=args.days,
        top_n_yyb=args.top_yyb,
        sort_by=args.sort,
        min_followers=args.min_followers,
        score_top=args.score_top,
    )


if __name__ == "__main__":
    main()
