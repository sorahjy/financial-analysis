"""游资小盘 universe 选股 + 建池。

目标：产出一个【稳定的小盘题材/游资活跃股票池】，供 stock_hot_money_radar.py 做截面打分
和多年 verify 回测——区别于 stock_crawl_hot_money.py（基于龙虎榜席位共振的【每日 T+1 交易信号】、
只爬 60 日 K 线、不做 PIT 回测）。

口径（用户确定）：
  · 游资活跃度主筛 = 近 1 年龙虎榜上榜 ≥ N 次（N 默认 5；龙虎榜高频 = 游资活跃 + 题材热门二合一）。
  · 标准 A 股前缀（沪 60 / 深 00 / 创业 30 / 科创 68）；排除 B 股(9*)、北交所、ST。
  · 爬 K 线（≥3.5 年，供 verify）后按【流通市值 ≤ 100 亿】过滤（流通市值=成交额/换手率反推，近 20 日中位）。
  · 选出标记 sw3_member.is_hot_money=1（复刻 is_leader 模式，radar 用 --pool hotmoney 取）。

MVP：先用「在 sw3_member 的票」（有 name/行业归属、能落 is_hot_money）；不在 sw3_member 的票
记到日志作为后补 TODO（需补行业归属 + sw3_segment 外键）。
"""
from __future__ import annotations

import argparse
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional

os.environ.setdefault("STOCK_CRAWL_NO_PROXY", "1")

import stock_storage
from stock_crawl_common import (
    fetch_qfq_daily_records,
    latest_weekday_date,
    retry_fetch_or_none as _retry,
    strip_proxy_env,
)

strip_proxy_env()

VALID_PREFIXES = ("60", "00", "30", "68")
DEFAULT_MIN_LHB = 5
DEFAULT_MAX_CAP_YI = 100.0
DEFAULT_WORKERS = 6          # 别调高：>6 易把龙虎榜/K线接口跑挂(限流)
DEFAULT_YEARS = 4.0         # 覆盖 verify 的 LOOKBACK(90)+WINDOW(750)+前向(40) ≈ 880 交易日
MIN_BARS = 250              # 上市/数据不足(次新)门槛：有效日线 < 250 视为次新剔除


def _lookback_start(years: float) -> str:
    return (datetime.now() - timedelta(days=int(years * 365))).strftime("%Y-%m-%d")


def select_seed(conn, min_lhb: int, since: str) -> Dict[str, List[str]]:
    """近 since 起龙虎榜上榜 ≥ min_lhb 次的 code；拆成 在册(in sw3_member) / 不在册(TODO)。"""
    rows = conn.execute(
        "SELECT code FROM lhb_all WHERE date >= ? GROUP BY code HAVING COUNT(*) >= ?",
        (since, min_lhb),
    ).fetchall()
    codes = [str(r["code"]).zfill(6) for r in rows]
    codes = [c for c in codes if c.startswith(VALID_PREFIXES)]
    members = {str(r["code"]).zfill(6): (r["name"] or "")
               for r in conn.execute("SELECT code, name FROM sw3_member").fetchall()}
    in_member, not_member = [], []
    for c in codes:
        if c in members:
            if "ST" in members[c].upper():
                continue
            in_member.append(c)
        else:
            not_member.append(c)
    return {"in_member": in_member, "not_member": not_member, "names": members}


def _bars_count(conn, code: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM stock_history WHERE code=? AND daily_close IS NOT NULL", (code,)
    ).fetchone()[0]


def crawl_one(code: str, name: str, start: str, end: str) -> tuple:
    """爬单只 ≥start K 线(含量额换手)并 upsert。返回 (code, status, bars)。"""
    try:
        recs = _retry(fetch_qfq_daily_records, code, start, end, include_trading_value=True)
    except Exception as e:
        return (code, f"ERR:{type(e).__name__}", 0)
    if not recs:
        return (code, "EMPTY", 0)
    conn = stock_storage.thread_conn()
    stock_storage.upsert_history_records(conn, code, name, recs, source="hot_money_universe")
    return (code, "OK", len(recs))


def float_cap_yi(conn, code: str) -> Optional[float]:
    """流通市值(亿) = 成交额 / (换手率/100)，取近 20 个有效日中位数(抗一字板/停牌噪声)。"""
    rows = conn.execute(
        "SELECT daily_amount, daily_turnover_rate FROM stock_history "
        "WHERE code=? AND daily_amount>0 AND daily_turnover_rate>0 "
        "ORDER BY date DESC LIMIT 20", (code,)
    ).fetchall()
    caps = [r["daily_amount"] / (r["daily_turnover_rate"] / 100.0) / 1e8
            for r in rows if r["daily_amount"] and r["daily_turnover_rate"]]
    return statistics.median(caps) if caps else None


def main() -> None:
    ap = argparse.ArgumentParser(description="游资小盘 universe 选股建池")
    ap.add_argument("--min-lhb", type=int, default=DEFAULT_MIN_LHB, help="近1年龙虎榜最少上榜次数")
    ap.add_argument("--max-cap", type=float, default=DEFAULT_MAX_CAP_YI, help="流通市值上限(亿)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="爬取线程数(别>6)")
    ap.add_argument("--years", type=float, default=DEFAULT_YEARS, help="爬取历史年数")
    ap.add_argument("--since", default=None, help="龙虎榜起始日(默认今天往前365天)")
    ap.add_argument("--dry-run", action="store_true", help="只选股不爬K线不标记")
    args = ap.parse_args()

    since = args.since or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    start, end = _lookback_start(args.years), latest_weekday_date()
    conn = stock_storage.connect()
    seed = select_seed(conn, args.min_lhb, since)
    in_member, not_member, names = seed["in_member"], seed["not_member"], seed["names"]
    print(f"近1年(>= {since})龙虎榜≥{args.min_lhb}次 ∩ 标准前缀 ∩ 非ST：")
    print(f"  在 sw3_member(可建池): {len(in_member)}  ·  不在 sw3_member(TODO后补): {len(not_member)}")

    if args.dry_run:
        need = [c for c in in_member if _bars_count(conn, c) < MIN_BARS]
        print(f"  [dry-run] 其中需补 K 线(<{MIN_BARS}日): {len(need)}  ·  已够: {len(in_member)-len(need)}")
        if not_member[:10]:
            print(f"  [dry-run] 不在册样本: {' '.join(not_member[:10])}")
        conn.close()
        return

    # 1) 加 is_hot_money 列
    stock_storage._ensure_table_columns(conn, "sw3_member", {"is_hot_money": "INTEGER NOT NULL DEFAULT 0"})

    # 2) 爬缺/不足历史的 K 线
    need = [c for c in in_member if _bars_count(conn, c) < MIN_BARS or _bars_count(conn, c) < args.years * 200]
    print(f"\n爬 K 线：{len(need)} 只待补 / {args.workers} 线程 / 历史 {start}..{end}")
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(crawl_one, c, names.get(c, ""), start, end): c for c in need}
        for i, fut in enumerate(as_completed(futs), 1):
            code, status, n = fut.result()
            if status == "OK":
                ok += 1
            else:
                fail += 1
                if fail <= 8:
                    print(f"  [FAIL] {code}: {status}")
            if i % 50 == 0:
                print(f"  进度 {i}/{len(need)}  ok={ok} fail={fail}")
    print(f"爬取完成 ok={ok} fail={fail}")

    # 3) 流通市值过滤 + 次新过滤 → 标记 is_hot_money
    conn.execute("UPDATE sw3_member SET is_hot_money = 0 WHERE is_hot_money = 1")
    selected, skip_cap, skip_new, skip_nocap = [], 0, 0, 0
    for c in in_member:
        bars = _bars_count(conn, c)
        if bars < MIN_BARS:
            skip_new += 1
            continue
        cap = float_cap_yi(conn, c)
        if cap is None:
            skip_nocap += 1
            continue
        if cap > args.max_cap:
            skip_cap += 1
            continue
        selected.append(c)
    if selected:
        conn.executemany("UPDATE sw3_member SET is_hot_money = 1 WHERE code = ?", [(c,) for c in selected])
    conn.commit()
    print(f"\n=== 建池完成 ===")
    print(f"  入池(is_hot_money=1): {len(selected)}")
    print(f"  剔除: 流通市值>{args.max_cap}亿 {skip_cap} · 次新(<{MIN_BARS}日) {skip_new} · 无市值数据 {skip_nocap}")
    print(f"  不在 sw3_member 未处理(TODO): {len(not_member)}")
    conn.close()


if __name__ == "__main__":
    main()
