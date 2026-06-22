"""资金/内部人信号数据层：龙虎榜(全榜) + 回购。爬取 + 入库 + 事件研究验证。

这两类都是稀疏事件信号（只覆盖上榜/有公告的票），价量代理不了，正交于吸筹分。
用事件研究口径验证：命中 = 近 EVENT_LOOKBACK_DAYS 日内有该事件；excess = 命中样本前向收益 −
同日全体均值（剔大盘 beta），跨日平均 + t。与 patterns 模式同口径。
PIT：龙虎榜用「上榜日」、回购用「公告日」作为信息可用日。

⚠️ 实测结论(纪要 item 13)：龙虎榜(全榜或机构、净买正负都一样)命中后全周期显著负(40日-2%/t-3)——
   「上榜=异动/追高」在龙头池系统性反转，故龙虎榜是**反向/避雷**信号，非买点。全榜比机构口径更显著、
   样本更大，故采全榜 stock_lhb_detail_em。回购 20-40日近显著正(+2%)=干净弱买点。

用法：
  python stock_crawl_capital.py --no-proxy            # 爬龙虎榜全榜 + 回购(至少近10年)入库
  python stock_crawl_capital.py --validate             # 读库事件研究(剔大盘)，不联网
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import stock_storage
from stock_crawl_common import retry_fetch_or_none, safe_float, safe_print, strip_proxy_env

HISTORY_YEARS = 10
CACHE_TTL_DAYS = 1
EVENT_LOOKBACK_DAYS = 30          # as-of 当日往前 30 自然日内有事件 → 视为"近期命中"


def history_start_date() -> str:
    return (datetime.now() - timedelta(days=366 * HISTORY_YEARS)).strftime("%Y-%m-%d")


def ensure_tables(conn) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS lhb_all (
            code TEXT, date TEXT, net_buy REAL, net_pct REAL, updated_at TEXT,
            PRIMARY KEY (code, date)
        );
        CREATE TABLE IF NOT EXISTS repurchase (
            code TEXT, disclose_date TEXT, amount REAL, status TEXT, updated_at TEXT,
            PRIMARY KEY (code, disclose_date)
        );
        """
    )
    _ensure_columns(conn, "lhb_all", {
        "net_buy": "REAL",
        "net_pct": "REAL",
        "updated_at": "TEXT",
    })
    _ensure_columns(conn, "repurchase", {
        "amount": "REAL",
        "status": "TEXT",
        "updated_at": "TEXT",
    })
    conn.commit()


def _table_columns(conn, table: str) -> set:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _ensure_columns(conn, table: str, columns: Dict[str, str]) -> None:
    existing = _table_columns(conn, table)
    for name, col_type in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")


def _norm(code: Any) -> str:
    return stock_storage._normalize_code(code)


def _updated_at_fresh(value: Any, ttl_days: int = CACHE_TTL_DAYS) -> bool:
    text = str(value or "")[:19]
    if not text:
        return False
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            sample = text[:19] if fmt.endswith("%S") else text[:10]
            updated = datetime.strptime(sample, fmt)
            return updated >= datetime.now() - timedelta(days=ttl_days)
        except ValueError:
            continue
    return False


def _cache_window(conn, table: str, date_col: str):
    columns = _table_columns(conn, table)
    updated_expr = "MAX(updated_at)" if "updated_at" in columns else "NULL"
    return conn.execute(
        f"SELECT COUNT(*) AS c, MIN({date_col}) AS min_date, MAX({date_col}) AS max_date, "
        f"{updated_expr} AS max_updated FROM {table}"
    ).fetchone()


def _cache_is_fresh(conn, table: str, date_col: str, start_date: str) -> bool:
    row = _cache_window(conn, table, date_col)
    if not row or int(row["c"] or 0) <= 0:
        return False
    min_date = str(row["min_date"] or "")[:10]
    return bool(min_date and min_date <= start_date and _updated_at_fresh(row["max_updated"]))


def _fetch_start_date(conn, table: str, date_col: str, start_date: str) -> str:
    row = _cache_window(conn, table, date_col)
    if not row or int(row["c"] or 0) <= 0:
        return start_date
    min_date = str(row["min_date"] or "")[:10]
    max_date = str(row["max_date"] or "")[:10]
    if not min_date or min_date > start_date or not max_date:
        return start_date
    try:
        backfill = datetime.strptime(max_date, "%Y-%m-%d") - timedelta(days=3)
        return max(start_date, backfill.strftime("%Y-%m-%d"))
    except ValueError:
        return start_date


def _year_ranges(start_date: str, end_date: str) -> List[Tuple[str, str]]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    ranges: List[Tuple[str, str]] = []
    for year in range(start.year, end.year + 1):
        s = max(start, datetime(year, 1, 1))
        e = min(end, datetime(year, 12, 31))
        if s <= e:
            ranges.append((s.strftime("%Y%m%d"), e.strftime("%Y%m%d")))
    return ranges


def _parse_lhb_df(df) -> List[Tuple[str, str, Optional[float], Optional[float]]]:
    rows: List[Tuple[str, str, Optional[float], Optional[float]]] = []
    if df is None or df.empty:
        return rows
    for _, r in df.iterrows():
        code = _norm(r.get("代码") or r.get("股票代码"))
        d = str(r.get("上榜日") or r.get("交易日") or "")[:10]
        if not code or not d:
            continue
        rows.append((code, d, safe_float(r.get("龙虎榜净买额")), safe_float(r.get("净买额占总成交比"))))
    return rows


def _fetch_lhb_direct_eastmoney(start_date: str, end_date: str) -> List[Tuple[str, str, Optional[float], Optional[float]]]:
    import requests

    s = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    e = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "sortColumns": "SECURITY_CODE,TRADE_DATE",
        "sortTypes": "1,-1",
        "pageSize": "5000",
        "pageNumber": "1",
        "reportName": "RPT_DAILYBILLBOARD_DETAILSNEW",
        "columns": (
            "SECURITY_CODE,SECURITY_NAME_ABBR,TRADE_DATE,BILLBOARD_NET_AMT,"
            "DEAL_NET_RATIO"
        ),
        "source": "WEB",
        "client": "WEB",
        "filter": f"(TRADE_DATE<='{e}')(TRADE_DATE>='{s}')",
    }
    try:
        first = requests.get(url, params=params, timeout=25)
        first.raise_for_status()
        result = (first.json().get("result") or {})
        pages = int(result.get("pages") or 1)
        data = list(result.get("data") or [])
        for page in range(2, pages + 1):
            params["pageNumber"] = str(page)
            resp = requests.get(url, params=params, timeout=25)
            resp.raise_for_status()
            data.extend(((resp.json().get("result") or {}).get("data") or []))
    except Exception as exc:
        safe_print(f"  [WARN] 龙虎榜东财直连 {s}~{e} 失败: {exc}")
        return []
    rows = []
    for r in data:
        code = _norm(r.get("SECURITY_CODE"))
        d = str(r.get("TRADE_DATE") or "")[:10]
        if code and d:
            rows.append((code, d, safe_float(r.get("BILLBOARD_NET_AMT")), safe_float(r.get("DEAL_NET_RATIO"))))
    return rows


def _fetch_lhb_sina_daily(start_date: str, end_date: str) -> List[Tuple[str, str, Optional[float], Optional[float]]]:
    """备用接口：新浪每日龙虎榜。只用于短缺口，避免十年级别逐日请求。"""
    import akshare as ak

    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    if (end - start).days > 120:
        return []
    rows: List[Tuple[str, str, Optional[float], Optional[float]]] = []
    cur = start
    while cur <= end:
        ymd = cur.strftime("%Y%m%d")
        df = retry_fetch_or_none(ak.stock_lhb_detail_daily_sina, date=ymd, retries=1)
        if df is not None and not df.empty:
            for _, r in df.iterrows():
                code = _norm(r.get("股票代码"))
                if code:
                    rows.append((code, cur.strftime("%Y-%m-%d"), None, None))
        cur += timedelta(days=1)
    return rows


def _fetch_lhb_range(start_date: str, end_date: str) -> List[Tuple[str, str, Optional[float], Optional[float]]]:
    import akshare as ak

    df = retry_fetch_or_none(ak.stock_lhb_detail_em, start_date=start_date, end_date=end_date)
    rows = _parse_lhb_df(df)
    if rows:
        return rows
    rows = _fetch_lhb_direct_eastmoney(start_date, end_date)
    if rows:
        return rows
    return _fetch_lhb_sina_daily(start_date, end_date)


def crawl_lhb(conn) -> int:
    """全榜龙虎榜（stock_lhb_detail_em，不限机构）。同股同日多条上榜原因→取净买额绝对值最大的一条。"""
    start_date = history_start_date()
    if _cache_is_fresh(conn, "lhb_all", "date", start_date):
        safe_print(f"[lhb_all] 缓存已覆盖≥{HISTORY_YEARS}年且今日新鲜，跳过联网爬取")
        return 0

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    end_date = datetime.now().strftime("%Y-%m-%d")
    fetch_start = _fetch_start_date(conn, "lhb_all", "date", start_date)
    best: Dict[tuple, tuple] = {}
    for s, e in _year_ranges(fetch_start, end_date):   # 按年分块拉，避免区间过大
        for code, d, nb, net_pct in _fetch_lhb_range(s, e):
            rec = (code, d, nb, net_pct, now)
            key = (code, d)
            if key not in best or abs(nb or 0) > abs(best[key][2] or 0):
                best[key] = rec
    if best:
        conn.executemany("INSERT OR REPLACE INTO lhb_all VALUES (?,?,?,?,?)", list(best.values()))
        conn.commit()
    safe_print(f"[lhb_all] {len(best)} 行（全榜上榜净买，避雷信号，{fetch_start} 起补齐）")
    return len(best)


def _fetch_repurchase_direct_eastmoney() -> List[tuple]:
    import requests

    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "sortColumns": "UPD,DIM_DATE,DIM_SCODE",
        "sortTypes": "-1,-1,-1",
        "pageSize": "500",
        "pageNumber": "1",
        "reportName": "RPTA_WEB_GETHGLIST_NEW",
        "columns": "ALL",
        "source": "WEB",
    }
    progress_map = {
        "001": "董事会预案",
        "002": "股东大会通过",
        "003": "股东大会否决",
        "004": "实施中",
        "005": "停止实施",
        "006": "完成实施",
    }
    try:
        first = requests.get(url, params=params, timeout=25)
        first.raise_for_status()
        result = (first.json().get("result") or {})
        pages = int(result.get("pages") or 1)
        data = list(result.get("data") or [])
        for page in range(2, pages + 1):
            params["pageNumber"] = str(page)
            resp = requests.get(url, params=params, timeout=25)
            resp.raise_for_status()
            data.extend(((resp.json().get("result") or {}).get("data") or []))
    except Exception as exc:
        safe_print(f"  [WARN] 回购东财直连失败: {exc}")
        return []
    rows = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for r in data:
        code = _norm(r.get("DIM_SCODE"))
        d = str(r.get("UPDATEDATE") or r.get("UPD") or r.get("DIM_DATE") or "")[:10]
        if not code or not d:
            continue
        status_raw = str(r.get("REPURPROGRESS") or "")
        rows.append((
            code,
            d,
            safe_float(r.get("REPURAMOUNT")),
            progress_map.get(status_raw, status_raw),
            now,
        ))
    return rows


def crawl_repurchase(conn) -> int:
    import akshare as ak
    start_date = history_start_date()
    if _cache_is_fresh(conn, "repurchase", "disclose_date", start_date):
        safe_print(f"[repurchase] 缓存已覆盖≥{HISTORY_YEARS}年且今日新鲜，跳过联网爬取")
        return 0

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = retry_fetch_or_none(ak.stock_repurchase_em)
    rows: List[tuple] = []
    if df is not None and not df.empty:
        for _, r in df.iterrows():
            code = _norm(r.get("股票代码"))
            d = str(r.get("最新公告日期") or "")[:10]
            if code and d and d >= start_date:
                rows.append((code, d, safe_float(r.get("已回购金额")), str(r.get("实施进度") or ""), now))
    if not rows:
        rows = [r for r in _fetch_repurchase_direct_eastmoney() if str(r[1]) >= start_date]
    # 同股同日去重（PK），保留任一
    uniq = {(c, d): (c, d, a, s, u) for c, d, a, s, u in rows}
    if uniq:
        conn.executemany("INSERT OR REPLACE INTO repurchase VALUES (?,?,?,?,?)", list(uniq.values()))
        conn.commit()
    safe_print(f"[repurchase] {len(uniq)} 行（回购公告，{start_date}起）")
    return len(uniq)


def crawl() -> None:
    conn = stock_storage.connect()
    ensure_tables(conn)
    crawl_lhb(conn)
    crawl_repurchase(conn)
    conn.close()
    safe_print("[capital] 完成")


def _event_dates(conn, table: str, date_col: str, where: str = "") -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    sql = f"SELECT code, {date_col} d FROM {table} WHERE {date_col} IS NOT NULL {where} ORDER BY code, {date_col}"
    for row in conn.execute(sql).fetchall():
        out.setdefault(_norm(row["code"]), []).append(row["d"])
    return out


def validate(max_cap: Optional[float]) -> None:
    import stock_hot_money_radar as R

    conn = stock_storage.connect()
    signals = {
        "龙虎榜全榜(避雷)": _event_dates(conn, "lhb_all", "date"),
        "回购": _event_dates(conn, "repurchase", "disclose_date"),
    }
    if not any(signals.values()):
        safe_print("[validate] 库里无数据，先爬取。")
        conn.close()
        return
    cands = R.load_leader_candidates(conn, max_cap=max_cap)
    HZ = R.VERIFY_HORIZONS
    max_h = max(HZ)
    series = {}
    for c in cands:
        b = R._all_bars(conn, c["code"])
        if len(b) >= R.LOOKBACK + max_h + 1:
            series[c["code"]] = (b, {x["date"]: i for i, x in enumerate(b)})
    conn.close()

    all_dates = sorted({d for b, _ in series.values() for d in (x["date"] for x in b)})
    as_of = all_dates[:-max_h][-R.VERIFY_WINDOW_DAYS:][::R.VERIFY_STEP]

    def has_event(dates: List[str], d: str) -> bool:
        lo = (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=EVENT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        # dates 升序；找是否存在 lo < e <= d
        return any(lo < e <= d for e in dates)

    # 收集每个 as-of 截面：全体 forward rets + 各信号命中集合
    sections = []
    for d in as_of:
        cohort = []
        for code, (b, idx) in series.items():
            i = idx.get(d)
            if i is None or i < R.LOOKBACK - 1 or i + max_h >= len(b):
                continue
            c0 = b[i]["close"]
            if not c0:
                continue
            rets = {}
            ok = True
            for h in HZ:
                cf = b[i + h]["close"]
                if not cf:
                    ok = False
                    break
                rets[h] = cf / c0 - 1.0
            if ok:
                cohort.append((code, rets))
        if len(cohort) >= R.VERIFY_MIN_NAMES:
            sections.append((d, cohort))

    def event_study(ev: Dict[str, List[str]], h: int):
        excess, n_hits = [], 0
        for d, cohort in sections:
            sec_mean = R._mean([r[h] for _, r in cohort])
            hit = [r[h] for code, r in cohort if code in ev and has_event(ev[code], d)]
            if not hit:
                continue
            excess.append(R._mean(hit) - sec_mean)
            n_hits += len(hit)
        if not excess:
            return 0, None, None
        m = R._mean(excess)
        sd = (sum((x - m) ** 2 for x in excess) / len(excess)) ** 0.5 if len(excess) > 1 else None
        t = (m / sd * math.sqrt(len(excess))) if sd else None
        return n_hits, m, t

    safe_print(f"[validate] 事件研究·命中后相对同日全体超额(剔大盘={max_cap}) · 截面 {len(sections)} 日 · 近{EVENT_LOOKBACK_DAYS}日内有事件=命中\n")
    print(f"  {'信号':<14}{'命中':>7}" + "".join(f"{str(h)+'日':>13}" for h in HZ))
    for name, ev in signals.items():
        cells = ""
        nh0 = 0
        for h in HZ:
            n, m, t = event_study(ev, h)
            nh0 = max(nh0, n)
            cells += f"{(f'{m*100:+.2f}%({t:+.1f})' if m is not None else '-'):>13}"
        print(f"  {name:<14}{nh0:>7}{cells}")
    print("  (回购看正超额=弱买点；龙虎榜全榜实测负=反向避雷信号；t≥1.5 视为显著)")


def main() -> None:
    p = argparse.ArgumentParser(description="资金/内部人信号：龙虎榜全榜 + 回购")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--max-cap-yi", type=float, default=300.0, help="验证市值上限(亿)；0=全市值")
    p.add_argument("--no-proxy", action="store_true")
    args = p.parse_args()
    if args.no_proxy:
        os.environ["STOCK_CRAWL_NO_PROXY"] = "1"
    strip_proxy_env()
    if args.validate:
        validate(None if args.max_cap_yi == 0 else args.max_cap_yi)
    else:
        crawl()


if __name__ == "__main__":
    main()
