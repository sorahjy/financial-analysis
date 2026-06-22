"""股东户数（筹码集中度）数据层：爬取 + 入库 + IC 验证。最小管线。

动机：P1-P24 buy 侧价量因子在龙头池普遍失效（多为反转），真正"谁在吸筹"的信息价量代理不了。
股东户数下降 = 筹码向少数人集中 = 主力吸筹的直接基本面证据（深交所实证：集中度+10%→后续收益+10.68%）。

数据源：akshare stock_zh_a_gdhs_detail_em（东财·个股股东户数详情，季度全历史）。
PIT：用「股东户数公告日期」作为信息可用日（不是统计截止日），避免前视。

用法：
  python stock_crawl_holders.py --no-proxy                 # 爬全部龙头(is_leader=1)
  python stock_crawl_holders.py --no-proxy --limit 30       # 先小样本试
  python stock_crawl_holders.py --validate                  # 读库跑截面 IC(剔大盘)，不联网
"""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import stock_storage
from stock_crawl_common import retry_fetch_or_none, safe_float, safe_print, strip_proxy_env

TABLE = "shareholder_count"
HISTORY_YEARS = 10
CACHE_TTL_DAYS = 30


def history_start_date() -> str:
    return (datetime.now() - timedelta(days=366 * HISTORY_YEARS)).strftime("%Y-%m-%d")


def _updated_at_fresh(value: Any, ttl_days: int) -> bool:
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


def ensure_table(conn) -> None:
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {TABLE} (
            code TEXT NOT NULL,
            report_date TEXT NOT NULL,      -- 股东户数统计截止日
            disclose_date TEXT,             -- 股东户数公告日期（PIT 信息可用日）
            holders REAL,                   -- 股东户数·本次
            change_pct REAL,                -- 股东户数·增减比例(%)
            avg_shares REAL,                -- 户均持股数量
            updated_at TEXT,
            PRIMARY KEY (code, report_date)
        )"""
    )
    _ensure_columns(conn, {
        "disclose_date": "TEXT",
        "holders": "REAL",
        "change_pct": "REAL",
        "avg_shares": "REAL",
        "updated_at": "TEXT",
    })
    conn.commit()


def _table_columns(conn) -> set:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({TABLE})").fetchall()}


def _ensure_columns(conn, columns: Dict[str, str]) -> None:
    existing = _table_columns(conn)
    for name, col_type in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {TABLE} ADD COLUMN {name} {col_type}")


def _cache_is_fresh(conn, code: str, start_date: str) -> bool:
    columns = _table_columns(conn)
    updated_expr = "MAX(updated_at)" if "updated_at" in columns else "NULL"
    row = conn.execute(
        f"SELECT COUNT(*) AS c, MIN(report_date) AS min_report, {updated_expr} AS max_updated "
        f"FROM {TABLE} WHERE code = ?",
        (code,),
    ).fetchone()
    if not row or int(row["c"] or 0) <= 0:
        return False
    min_report = str(row["min_report"] or "")[:10]
    return bool(min_report and min_report <= start_date and _updated_at_fresh(row["max_updated"], CACHE_TTL_DAYS))


def _fetch_one_akshare(code: str) -> List[Dict[str, Any]]:
    import akshare as ak
    df = retry_fetch_or_none(ak.stock_zh_a_gdhs_detail_em, symbol=str(code).zfill(6))
    if df is None or df.empty:
        return []
    recs: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        report = str(row.get("股东户数统计截止日") or "")[:10]
        if not report:
            continue
        recs.append({
            "report_date": report,
            "disclose_date": str(row.get("股东户数公告日期") or "")[:10] or None,
            "holders": safe_float(row.get("股东户数-本次")),
            "change_pct": safe_float(row.get("股东户数-增减比例")),
            "avg_shares": safe_float(row.get("户均持股数量")),
        })
    return recs


def _fetch_one_direct_eastmoney(code: str) -> List[Dict[str, Any]]:
    import requests

    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "sortColumns": "END_DATE",
        "sortTypes": "-1",
        "pageSize": "500",
        "pageNumber": "1",
        "reportName": "RPT_HOLDERNUM_DET",
        "columns": (
            "SECURITY_CODE,SECURITY_NAME_ABBR,END_DATE,AVG_HOLD_NUM,HOLD_NOTICE_DATE,"
            "HOLDER_NUM,HOLDER_NUM_RATIO"
        ),
        "filter": f'(SECURITY_CODE="{str(code).zfill(6)}")',
        "source": "WEB",
        "client": "WEB",
    }
    try:
        first = requests.get(url, params=params, timeout=20)
        first.raise_for_status()
        payload = first.json()
        result = payload.get("result") or {}
        pages = int(result.get("pages") or 1)
        data = list(result.get("data") or [])
        for page in range(2, pages + 1):
            params["pageNumber"] = str(page)
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data.extend(((resp.json().get("result") or {}).get("data") or []))
    except Exception as exc:
        safe_print(f"  [WARN] {code} 东财直连股东户数失败: {exc}")
        return []

    recs: List[Dict[str, Any]] = []
    for row in data:
        report = str(row.get("END_DATE") or "")[:10]
        if not report:
            continue
        recs.append({
            "report_date": report,
            "disclose_date": str(row.get("HOLD_NOTICE_DATE") or "")[:10] or None,
            "holders": safe_float(row.get("HOLDER_NUM")),
            "change_pct": safe_float(row.get("HOLDER_NUM_RATIO")),
            "avg_shares": safe_float(row.get("AVG_HOLD_NUM")),
        })
    return recs


def _fetch_one_sina_main_holder(code: str) -> List[Dict[str, Any]]:
    """备用：新浪主要股东页含股东总数/公告日，可还原股东户数变化。"""
    import akshare as ak

    df = retry_fetch_or_none(ak.stock_main_stock_holder, stock=str(code).zfill(6), retries=2)
    if df is None or df.empty:
        return []
    by_report: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        report = str(row.get("截至日期") or row.get("截止日期") or "")[:10]
        holders = safe_float(row.get("股东总数"))
        if not report or holders is None:
            continue
        by_report[report] = {
            "report_date": report,
            "disclose_date": str(row.get("公告日期") or "")[:10] or None,
            "holders": holders,
            "avg_shares": safe_float(row.get("平均持股数")),
        }
    recs: List[Dict[str, Any]] = []
    prev_holders: Optional[float] = None
    for report in sorted(by_report):
        rec = by_report[report]
        holders = safe_float(rec.get("holders"))
        change_pct = None
        if holders is not None and prev_holders and abs(prev_holders) > 1e-9:
            change_pct = (holders - prev_holders) / abs(prev_holders) * 100.0
        recs.append({
            "report_date": rec["report_date"],
            "disclose_date": rec["disclose_date"],
            "holders": holders,
            "change_pct": change_pct,
            "avg_shares": rec["avg_shares"],
        })
        if holders is not None:
            prev_holders = holders
    return recs


def _fetch_one(code: str) -> List[Dict[str, Any]]:
    recs = _fetch_one_akshare(code)
    if recs:
        return recs
    recs = _fetch_one_direct_eastmoney(code)
    if recs:
        return recs
    return _fetch_one_sina_main_holder(code)


def _save(conn, code: str, recs: List[Dict[str, Any]]) -> int:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.executemany(
        f"INSERT OR REPLACE INTO {TABLE} "
        "(code, report_date, disclose_date, holders, change_pct, avg_shares, updated_at) "
        "VALUES (?,?,?,?,?,?,?)",
        [(code, r["report_date"], r["disclose_date"], r["holders"], r["change_pct"], r["avg_shares"], now)
         for r in recs],
    )
    conn.commit()
    return len(recs)


def crawl(limit: Optional[int], workers: int, force: bool = False) -> None:
    conn = stock_storage.connect()
    ensure_table(conn)
    codes = [stock_storage._normalize_code(m["code"]) for m in stock_storage.leader_members(conn)]
    codes = [c for c in codes if c]
    if limit:
        codes = codes[:limit]
    start_date = history_start_date()
    skipped = 0
    if not force:
        fresh_codes = []
        for code in codes:
            if _cache_is_fresh(conn, code, start_date):
                skipped += 1
            else:
                fresh_codes.append(code)
        codes = fresh_codes
    safe_print(
        f"[holders] 历史窗口≥{HISTORY_YEARS}年({start_date}起) · 待爬 {len(codes)} 只龙头 "
        f"· 缓存命中 {skipped} · workers={workers}"
    )
    if not codes:
        conn.close()
        safe_print(f"[holders] 缓存已新鲜：跳过联网爬取（TTL={CACHE_TTL_DAYS}天）")
        return
    done = ok = total_rows = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, c): c for c in codes}
        for fut in as_completed(futs):
            code = futs[fut]
            done += 1
            try:
                recs = fut.result()
            except Exception as exc:
                recs = []
                safe_print(f"  [WARN] {code} 失败: {exc}")
            if recs:
                total_rows += _save(conn, code, recs)
                ok += 1
            if done % 50 == 0 or done == len(codes):
                safe_print(f"  进度 {done}/{len(codes)} · 有数据 {ok} · 累计 {total_rows} 行")
    conn.close()
    safe_print(f"[holders] 完成：{ok}/{len(codes)} 只有数据，共 {total_rows} 行入库 {TABLE}")


def validate(max_cap: Optional[float]) -> None:
    """读库 → PIT 对齐 → 户数变化因子的截面 RankIC（剔大盘）。不联网。"""
    import stock_hot_money_radar as R

    conn = stock_storage.connect()
    # 每只票按公告日排序的 (disclose_date, change_pct, holders)
    hist: Dict[str, List[tuple]] = {}
    for row in conn.execute(
        f"SELECT code, disclose_date, change_pct, holders FROM {TABLE} "
        "WHERE disclose_date IS NOT NULL AND change_pct IS NOT NULL ORDER BY code, disclose_date"
    ).fetchall():
        hist.setdefault(stock_storage._normalize_code(row["code"]), []).append(
            (row["disclose_date"], row["change_pct"], row["holders"]))
    if not hist:
        safe_print("[validate] 库里没有股东户数数据，先跑爬取。")
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

    def latest_change(code: str, d: str) -> Optional[float]:
        """as-of 当日已公告的最近一期股东户数增减比例(%)。"""
        recs = hist.get(code)
        if not recs:
            return None
        prior = [cp for dd, cp, _ in recs if dd <= d]
        return prior[-1] if prior else None

    by_date: Dict[str, List[tuple]] = {}
    for d in as_of:
        rows = []
        for code, (b, idx) in series.items():
            i = idx.get(d)
            if i is None or i < R.LOOKBACK - 1 or i + max_h >= len(b):
                continue
            cp = latest_change(code, d)
            if cp is None:
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
            if not ok:
                continue
            rows.append((-cp, rets))   # 因子=−增减比例：户数降幅越大(越负)→因子越高→看多
        if len(rows) >= R.VERIFY_MIN_NAMES:
            by_date[d] = rows

    safe_print(f"[validate] 因子=「−股东户数增减比例」(户数降=看多) · 截面 {len(by_date)} 日 · 剔大盘={max_cap}\n")
    safe_print(f"  {'持有期':>6} {'RankIC':>9} {'IC_t':>7} {'覆盖股均值':>9}")
    cover = R._mean([len(r) for r in by_date.values()]) if by_date else 0
    for h in HZ:
        vals = []
        for rows in by_date.values():
            xs = [f for f, _ in rows]
            ys = [r[h] for _, r in rows]
            s = R._spearman(xs, ys)
            if s is not None:
                vals.append(s)
        if not vals:
            continue
        m = R._mean(vals)
        sd = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5 if len(vals) > 1 else None
        t = (m / sd * math.sqrt(len(vals))) if sd else None
        safe_print(f"  {str(h)+'日':>6} {m:>9.4f} {(t or 0):>7.2f} {cover:>9.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="股东户数数据层：爬取 / 验证")
    parser.add_argument("--validate", action="store_true", help="读库跑截面 IC（不联网）")
    parser.add_argument("--limit", type=int, default=None, help="只爬前 N 只（试跑）")
    parser.add_argument("--workers", type=int, default=5, help="并发数（≤6，避免限流）")
    parser.add_argument("--force", action="store_true", help="忽略缓存，强制重爬")
    parser.add_argument("--max-cap-yi", type=float, default=300.0, help="验证时市值上限(亿)；0=全市值")
    parser.add_argument("--no-proxy", action="store_true", help="绕过代理直连境内接口")
    args = parser.parse_args()

    if args.no_proxy:
        os.environ["STOCK_CRAWL_NO_PROXY"] = "1"
    strip_proxy_env()

    if args.validate:
        validate(None if args.max_cap_yi == 0 else args.max_cap_yi)
    else:
        crawl(args.limit, min(args.workers, 6), force=args.force)


if __name__ == "__main__":
    main()
