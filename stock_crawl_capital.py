"""
爬取龙虎榜活跃营业部（主力席位）数据，按上榜次数与净买入聚合，
打印最近活跃席位及其买入的股票。

数据来源:
    东方财富网 - 数据中心 - 龙虎榜单 - 每日活跃营业部
接口:
    akshare.stock_lhb_hyyyb_em(start_date, end_date)
"""

import argparse
import json
import math
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


def collect_stocks_from_top_yyb(agg, top_n=20, sort_by="appearances"):
    """从 Top N 席位的「买入股票」集合中抽出去重股票，并保留反向映射。

    Returns:
        dict[str, set[str]]: 股票名称 → 推荐它的席位集合
    """
    items = _sort_items(agg, sort_by)[:top_n]
    stock_to_yyb = defaultdict(set)
    for item in items:
        for stock in item["stocks"]:
            stock_to_yyb[stock].add(item["name"])
    return dict(stock_to_yyb)


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
    n_followers = len(stock.get("followers", []))

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
            + _pad_visual(str(len(s["followers"])), 8)
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
        "weights": WEIGHTS,
        "count": len(ranked),
        "stocks": [
            {
                "code": s["code"],
                "name": s["name"],
                "scores": s["scores"],
                "followers": s["followers"],
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

    stock_to_yyb = collect_stocks_from_top_yyb(agg, top_n=top_n_yyb, sort_by=sort_by)
    pool = {s: yybs for s, yybs in stock_to_yyb.items() if len(yybs) >= min_followers}
    print(f"\nTop {top_n_yyb} 席位共买入 {len(stock_to_yyb)} 只去重股票"
          f"（≥{min_followers} 个席位共买：{len(pool)} 只）")

    name_map = get_name_code_map()
    tasks, missing = [], []
    for stock_name in sorted(pool, key=lambda s: -len(pool[s])):
        code = name_map.get(stock_name)
        if not code:
            missing.append(stock_name)
            continue
        tasks.append((code, stock_name, sorted(pool[stock_name])))

    print(f"\n开始并发拉 K 线 + 打分（{len(tasks)} 只 / {THREAD_COUNT} 线程）...")
    results, failed = [], []
    completed = 0

    def worker(code, name, followers):
        data = fetch_stock_data(code, name)
        if not data or not data.get("records"):
            return ("FAIL", code, name, "empty kline")
        data["followers"] = followers
        data["scores"] = score_stock(data)
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
