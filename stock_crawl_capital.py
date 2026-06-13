"""
爬取龙虎榜数据：活跃营业部（主力席位）、每日明细、机构买卖统计、个股上榜统计，
按共振席位聚合打分，并落盘短线策略需要的 signals（龙虎榜/机构/统计/技术面）。

数据来源（东方财富网 - 数据中心 - 龙虎榜单）:
    akshare.stock_lhb_hyyyb_em            每日活跃营业部（席位 → 买入股票）
    akshare.stock_lhb_detail_em           每日明细（净买额/上榜原因/流通市值）
    akshare.stock_lhb_jgmmtj_em           机构买卖每日统计
    akshare.stock_lhb_stock_statistic_em  个股上榜统计（近一月）
    akshare.stock_lhb_yybph_em            营业部排行（用于标记高活跃/知名席位）
"""

import argparse
import json
import math
import os
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak

KLINE_LOOKBACK = 60      # 取最近 60 个交易日（足够算 MA20 / RSI / 距高点）
MAX_RETRIES = 3
THREAD_COUNT = 6
DATA_DIR = Path("data/capital")
KNOWN_SEAT_TOP_N = 100   # 营业部排行前 N 标记为高活跃/知名席位


def _strip_proxy_env():
    """STOCK_CRAWL_NO_PROXY=1 时绕过系统/环境变量代理直连（东财为境内接口）。"""
    if os.getenv("STOCK_CRAWL_NO_PROXY", "").strip().lower() not in ("1", "true", "yes"):
        return
    for var in ("http_proxy", "https_proxy", "all_proxy",
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(var, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


_strip_proxy_env()

# 新浪 K 线接口底层走 mini_racer / V8 执行 JS 加密，V8 不是线程安全
# 多线程并发会 segfault → 用全局锁串行化 JS 执行段。
_kline_lock = threading.Lock()


def _num(v):
    """把 None / NaN / 字符串都转成 float，无法解析返回 0.0。"""
    if v is None:
        return 0.0
    try:
        f = float(v)
        return 0.0 if math.isnan(f) else f
    except (ValueError, TypeError):
        return 0.0


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

def _retry(func, *args, retries=MAX_RETRIES, **kwargs):
    """退避重试包装；末次失败返回 None 而非抛错。"""
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception:
            if i == retries - 1:
                return None
            time.sleep(0.5 * (i + 1) + random.uniform(0, 0.5))
    return None


def _symbol_with_prefix(code):
    """6 位 code → 新浪接口要求的 sh/sz/bj 前缀格式。"""
    if code.startswith(("60", "68", "9")):
        return "sh" + code
    if code.startswith(("00", "30", "20")):
        return "sz" + code
    if code.startswith(("8", "4", "92")):
        return "bj" + code
    return "sh" + code


def fetch_stock_data(code, name):
    """拉取最近 `KLINE_LOOKBACK` 个交易日的前复权日 K 线（新浪源）。

    Returns:
        dict | None: {code, name, records: [{date,open,high,low,close,volume,
            amount,change_pct,turnover_rate}, ...]}；失败返回 None。
    """
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


def score_resonance(follower_count):
    """主力共振分：被几个 Top 席位共同买入。"""
    table = {0: 0, 1: 30, 2: 55, 3: 70, 4: 85}
    return table.get(follower_count, 100)


def score_momentum(records):
    """动量分 = 平均(5日涨幅分, 量比分, 均线多头分)。"""
    if len(records) < 20:
        return 0.0
    closes = [r["close"] for r in records]
    vols = [r["volume"] for r in records]

    # 1) 5 日涨幅：钟形得分，2~10% 最佳（动量但未过热）
    if closes[-6] > 0:
        ret5 = (closes[-1] / closes[-6] - 1) * 100
    else:
        ret5 = 0
    if 2 <= ret5 <= 10:
        r_score = 100
    elif ret5 < -10 or ret5 > 25:
        r_score = 0
    elif ret5 < 2:
        r_score = max(0, 50 + ret5 * 5)        # -10→0, 2→60
    else:
        r_score = max(0, 100 - (ret5 - 10) * 5) # 10→100, 25→25

    # 2) 量比：今日量 vs 5 日均量
    avg5 = sum(vols[-6:-1]) / 5 if len(vols) >= 6 else 0
    vol_ratio = (vols[-1] / avg5) if avg5 > 0 else 1
    if vol_ratio < 0.5: v_score = 20
    elif vol_ratio < 0.8: v_score = 40
    elif vol_ratio < 1.2: v_score = 60
    elif vol_ratio < 2:   v_score = 80
    elif vol_ratio < 4:   v_score = 100
    else:                 v_score = 70         # 过度天量警惕

    # 3) 均线多头：MA5 > MA10 > MA20 且 close > MA5
    ma5, ma10, ma20 = _ma(closes, 5), _ma(closes, 10), _ma(closes, 20)
    if ma5 and ma10 and ma20:
        bull = ma5 > ma10 > ma20
        above5 = closes[-1] > ma5
        ma_score = 100 if (bull and above5) else (60 if above5 else 30)
    else:
        ma_score = 50

    return (r_score + v_score + ma_score) / 3


def score_position(records):
    """位置分 = 平均(距 60 日高点位置分, RSI 分)。"""
    if len(records) < 20:
        return 50.0
    closes = [r["close"] for r in records]
    highs = [r["high"] for r in records]

    look = min(60, len(records))
    peak = max(highs[-look:])
    last = closes[-1]
    if peak <= 0:
        p_score = 50
    else:
        drop = (peak - last) / peak * 100  # 距高点跌幅 %
        if drop < 0:      p_score = 80     # 已破前高（继续强但不追)
        elif drop < 5:    p_score = 90
        elif drop < 15:   p_score = 100    # 上攻空间最佳
        elif drop < 30:   p_score = 70
        elif drop < 50:   p_score = 40
        else:             p_score = 20

    r = _rsi(closes, 14)
    if r is None:        rsi_score = 50
    elif r < 30:         rsi_score = 100   # 超卖反弹
    elif r < 50:         rsi_score = 90
    elif r < 70:         rsi_score = 70
    elif r < 80:         rsi_score = 40
    else:                rsi_score = 10    # 极度超买

    return (p_score + rsi_score) / 2


def score_risk_penalty(name, records):
    """风险扣分汇总（越大越坏，上限 100）。"""
    penalty = 0
    upper = name.upper()
    if "ST" in upper or "*ST" in upper:
        penalty += 60

    if len(records) >= 6 and records[-6]["close"] > 0:
        ret5 = (records[-1]["close"] / records[-6]["close"] - 1) * 100
        if ret5 > 30:    penalty += 30      # 短期过热
        elif ret5 > 20:  penalty += 15

    if records and records[-1]["change_pct"] < -5:
        penalty += 30                       # 昨日大跌（可能炸板）

    # 流动性
    if len(records) >= 5:
        avg_amt = sum(r["amount"] for r in records[-5:]) / 5
        if avg_amt < 5e7:    penalty += 20  # 日均成交 <5000 万
        elif avg_amt < 1e8:  penalty += 10

    # 长上影线（昨日上影 > 实体 1.5 倍）
    if records:
        r = records[-1]
        body = abs(r["close"] - r["open"])
        up_wick = r["high"] - max(r["close"], r["open"])
        if body > 0 and up_wick > body * 1.5:
            penalty += 15

    return min(penalty, 100)


WEIGHTS = {
    "resonance": 0.25,  # 主力共振
    "momentum":  0.30,  # 动量
    "position":  0.20,  # 位置（距高点/RSI）
    "risk":      0.25,  # 风控（反向）
}


def score_stock(stock):
    """对单只股票打分。

    Args:
        stock: 含 followers / records / name 的字典
    Returns:
        dict: {resonance, momentum, position, risk_penalty, total, verdict}
    """
    records = stock.get("records") or []
    name = stock.get("name", "")
    followers = stock.get("followers") or []
    # followers 是席位级明细（同一席位多日重复出现），共振按去重席位数算
    n_followers = len(set(
        f.get("seat") if isinstance(f, dict) else str(f) for f in followers
    ))

    r = score_resonance(n_followers)
    m = score_momentum(records)
    p = score_position(records)
    risk = score_risk_penalty(name, records)

    total = (
        WEIGHTS["resonance"] * r
        + WEIGHTS["momentum"] * m
        + WEIGHTS["position"] * p
        + WEIGHTS["risk"] * max(0, 100 - risk)
    )

    if total >= 75:   verdict = "Strong Buy"
    elif total >= 60: verdict = "Watch"
    else:             verdict = "Skip"

    return {
        "resonance":     round(r, 1),
        "momentum":      round(m, 1),
        "position":      round(p, 1),
        "risk_penalty":  round(risk, 1),
        "total":         round(total, 1),
        "verdict":       verdict,
    }


# ─── 排序 / 输出 / 落盘 ─────────────────────────────────────────

def rank_and_print(stocks, top_n=20):
    """对带 scores 的 stocks 排序并打印 Top N。"""
    ranked = sorted(stocks, key=lambda s: -s["scores"]["total"])

    print("\n" + "=" * 110)
    print(f"  T+1 买 / T+2 卖 综合打分 Top {top_n}")
    print(f"  权重: 共振{int(WEIGHTS['resonance']*100)}% 动量{int(WEIGHTS['momentum']*100)}% "
          f"位置{int(WEIGHTS['position']*100)}% 风控{int(WEIGHTS['risk']*100)}%")
    print("=" * 110)
    header = (
        _pad_visual("代码", 8)
        + _pad_visual("名称", 14)
        + _pad_visual("总分", 8)
        + _pad_visual("评级", 14)
        + _pad_visual("共振", 8)
        + _pad_visual("动量", 8)
        + _pad_visual("位置", 8)
        + _pad_visual("风控", 8)
        + _pad_visual("席位数", 8)
    )
    print(header)
    print("-" * 110)
    for s in ranked[:top_n]:
        sc = s["scores"]
        print(
            _pad_visual(s["code"], 8)
            + _pad_visual(s["name"], 14)
            + _pad_visual(f"{sc['total']:.1f}", 8)
            + _pad_visual(sc["verdict"], 14)
            + _pad_visual(f"{sc['resonance']:.0f}", 8)
            + _pad_visual(f"{sc['momentum']:.0f}", 8)
            + _pad_visual(f"{sc['position']:.0f}", 8)
            + _pad_visual(f"{100 - sc['risk_penalty']:.0f}", 8)
            + _pad_visual(str(s.get("total_buyers") or len(s["followers"])), 8)
        )
    print("=" * 110)
    print("注: 分数仅供参考；T+1 实际买点请结合开盘竞价与盘口。")
    return ranked


def persist_results(stocks, fp=None):
    """落盘排序结果。"""
    if fp is None:
        fp = DATA_DIR / "scored_stocks.json"
    else:
        fp = Path(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(stocks, key=lambda s: -s["scores"]["total"])
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of_date": datetime.now().strftime("%Y-%m-%d"),
        "weights": WEIGHTS,
        "count": len(ranked),
        "stocks": [
            {
                "code": s["code"],
                "name": s["name"],
                "scores": s["scores"],
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
    """主流程：龙虎榜 → 共振股票池 → 并发拉 K 线 → 打分 → 排序输出。

    Args:
        min_followers: 至少被多少个 Top 席位共买才纳入打分，默认 2（更聚焦共振）。
        score_top:     最终打分榜展示前 N 条。

    Returns:
        list[dict]: 含 scores 的股票列表（已按 total 降序）。
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

    print(f"\n开始并发拉 K 线 + 打分（{len(tasks)} 只 / {THREAD_COUNT} 线程）...")
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
        data["scores"] = score_stock(data)
        if tech:
            data["scores"].update({
                "consecutive_limit_up": tech.get("consecutive_limit_up"),
                "is_yizi_ban": tech.get("is_yizi_ban"),
                "is_t_ban": tech.get("is_t_ban"),
            })
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

    print(f"\n已打分 {len(results)} 只 · 跳过(无代码) {len(missing)} · 失败 {len(failed)}")
    if missing:
        sample = missing[:10]
        more = f" ...+{len(missing) - 10}" if len(missing) > 10 else ""
        print(f"  无代码样本: {' '.join(sample)}{more}")
    if failed:
        for code, name, err in failed[:5]:
            print(f"  [FAIL] {code} {name}: {err}")

    if not results:
        return []

    ranked = rank_and_print(results, top_n=score_top)
    persist_results(ranked)
    return ranked


def main():
    parser = argparse.ArgumentParser(
        description="主力席位龙虎榜 → 共振股票池 → T+1/T+2 打分排序")
    parser.add_argument("--days", type=int, default=14,
                        help="拉取最近多少自然日的龙虎榜数据，默认 14")
    parser.add_argument("--top-yyb", type=int, default=20,
                        help="取活跃营业部 Top N，默认 20")
    parser.add_argument("--sort", choices=["appearances", "net", "buy"],
                        default="appearances",
                        help="席位排序依据：上榜次数/净买入/买入金额")
    parser.add_argument("--min-followers", type=int, default=2,
                        help="股票至少被多少个 Top 席位共买才纳入打分，默认 2")
    parser.add_argument("--score-top", type=int, default=20,
                        help="打分榜展示前 N 条，默认 20")
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
