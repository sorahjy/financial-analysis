"""短线行业热度数据准备入口。

抓取申万二级行业日度分析数据，落库到 data/plate_data.sqlite3。
默认从 2016-01-01 开始；后续运行从所有板块共同覆盖水位之后继续，自动补齐尾部缺口。
"""

from __future__ import annotations

import argparse
import math
import os
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import plate_storage
from stock_crawl_common import strip_proxy_env


DEFAULT_START = date(2016, 1, 1)
DEFAULT_CHUNK_DAYS = 365
DEFAULT_PAGE_SIZE = 500
SW2_API_URL = "https://www.swsresearch.com/institute-sw/api/index_analysis/index_analysis_report/"


def parse_yyyymmdd(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def format_yyyymmdd(value: date) -> str:
    return value.strftime("%Y%m%d")


def date_dash(value: date) -> str:
    return value.strftime("%Y-%m-%d")


def parse_dash(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_chunks(start: date, end: date, chunk_days: int) -> Iterable[Tuple[date, date]]:
    if chunk_days <= 0:
        chunk_days = DEFAULT_CHUNK_DAYS
    cur = start
    while cur <= end:
        chunk_end = min(end, cur + timedelta(days=chunk_days - 1))
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def resolve_incremental_start(
    latest_trade_date: Optional[str],
    requested_start: date,
    requested_end: date,
) -> Optional[date]:
    if latest_trade_date:
        next_start = parse_dash(latest_trade_date) + timedelta(days=1)
        requested_start = max(requested_start, next_start)
    return requested_start if requested_start <= requested_end else None


def fetch_sw2_daily_rows(start: date, end: date, *, page_size: int = DEFAULT_PAGE_SIZE) -> List[Mapping[str, Any]]:
    """Fetch SW second-level industry rows from the same endpoint AkShare wraps."""
    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    page_size = max(int(page_size or DEFAULT_PAGE_SIZE), 50)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        )
    }
    base_params = {
        "page_size": str(page_size),
        "index_type": "二级行业",
        "start_date": date_dash(start),
        "end_date": date_dash(end),
        "type": "DAY",
        "swindexcode": "all",
    }
    params = {**base_params, "page": "1"}
    response = requests.get(SW2_API_URL, params=params, headers=headers, verify=False, timeout=30)
    response.raise_for_status()
    data = response.json()
    count = int(((data.get("data") or {}).get("count")) or 0)
    total_pages = math.ceil(count / page_size) if count > 0 else 0
    rows = list(((data.get("data") or {}).get("results")) or [])

    for page in range(2, total_pages + 1):
        params = {**base_params, "page": str(page)}
        response = requests.get(SW2_API_URL, params=params, headers=headers, verify=False, timeout=30)
        response.raise_for_status()
        data = response.json()
        rows.extend(((data.get("data") or {}).get("results")) or [])
    return rows


def fetch_sw2_daily_analysis(
    start: date,
    end: date,
    *,
    chunk_days: int,
    sleep_sec: float,
    page_size: int = DEFAULT_PAGE_SIZE,
    db_file: str | None = None,
) -> Dict[str, Any]:
    conn = plate_storage.connect(db_file or plate_storage.DEFAULT_DB_FILE)
    try:
        latest = plate_storage.latest_trade_date(conn, plate_storage.PLATE_TYPE_SW2)
        coverage_latest = plate_storage.oldest_latest_trade_date(
            conn, plate_storage.PLATE_TYPE_SW2
        )
        coverage_gap = plate_storage.recent_incomplete_trade_date(
            conn, plate_storage.PLATE_TYPE_SW2
        )
        frontier_start = resolve_incremental_start(coverage_latest or latest, start, end)
        gap_start = None
        if coverage_gap:
            candidate = max(start, parse_dash(coverage_gap))
            gap_start = candidate if candidate <= end else None
        candidates = [candidate for candidate in (frontier_start, gap_start) if candidate]
        fetch_start = min(candidates) if candidates else None
        if fetch_start is None:
            total = plate_storage.table_count(conn, "plate_daily")
            print(
                f"[sw2] no new dates; latest={latest}, "
                f"coverage_latest={coverage_latest}, total={total}",
                flush=True,
            )
            return {
                "requested_start": format_yyyymmdd(start),
                "requested_end": format_yyyymmdd(end),
                "fetch_start": None,
                "fetch_end": format_yyyymmdd(end),
                "fetched": 0,
                "inserted": 0,
                "updated": 0,
                "errors": [],
                "total_records": total,
                "latest_trade_date": latest,
                "coverage_trade_date": coverage_latest,
                "coverage_gap_date": coverage_gap,
            }

        totals = {"fetched": 0, "inserted": 0, "updated": 0}
        errors: List[Dict[str, Any]] = []
        for chunk_start, chunk_end in date_chunks(fetch_start, end, chunk_days):
            start_text = format_yyyymmdd(chunk_start)
            end_text = format_yyyymmdd(chunk_end)
            try:
                rows = fetch_sw2_daily_rows(chunk_start, chunk_end, page_size=page_size)
                stats = plate_storage.save_sw2_daily_rows(conn, rows)
                for key in totals:
                    totals[key] += stats[key]
                print(
                    f"[sw2] {start_text}-{end_text} fetched={stats['fetched']} "
                    f"inserted={stats['inserted']} updated={stats['updated']}",
                    flush=True,
                )
            except Exception as exc:  # pragma: no cover - network/data-source boundary
                errors.append({
                    "start_date": start_text,
                    "end_date": end_text,
                    "error": str(exc),
                    "failed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                print(f"[sw2] {start_text}-{end_text} failed: {exc}", flush=True)
                # 不允许后续分块越过失败区间推进全局最新日期；否则下次增量无法知道中间
                # 有一整段缺口。停在第一处失败，下轮仍会从共同覆盖水位继续补抓。
                break
            if sleep_sec > 0:
                time.sleep(sleep_sec)

        total_records = plate_storage.table_count(conn, "plate_daily")
        latest_after = plate_storage.latest_trade_date(conn, plate_storage.PLATE_TYPE_SW2)
        return {
            "requested_start": format_yyyymmdd(start),
            "requested_end": format_yyyymmdd(end),
            "fetch_start": format_yyyymmdd(fetch_start),
            "fetch_end": format_yyyymmdd(end),
            "fetched": totals["fetched"],
            "inserted": totals["inserted"],
            "updated": totals["updated"],
            "errors": errors,
            "total_records": total_records,
            "latest_trade_date": latest_after,
            "coverage_trade_date": plate_storage.oldest_latest_trade_date(
                conn, plate_storage.PLATE_TYPE_SW2
            ),
            "coverage_gap_date": plate_storage.recent_incomplete_trade_date(
                conn, plate_storage.PLATE_TYPE_SW2
            ),
        }
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抓取申万二级行业日度分析数据并落库 plate_data.sqlite3")
    parser.add_argument("--start-date", help="开始日期 YYYYMMDD；默认 20160101")
    parser.add_argument("--end-date", help="结束日期 YYYYMMDD；默认今天")
    parser.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS, help="分块天数，默认 365")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="接口分页大小，默认 500")
    parser.add_argument("--sleep-sec", type=float, default=0.0, help="分块请求间隔秒数，默认 0")
    parser.add_argument("--no-proxy", action="store_true", help="绕过代理直连境内数据源")
    return parser


def main() -> Dict[str, Any]:
    args = build_parser().parse_args()
    if args.no_proxy:
        os.environ["STOCK_CRAWL_NO_PROXY"] = "1"
        strip_proxy_env()
    today = date.today()
    start = parse_yyyymmdd(args.start_date) if args.start_date else DEFAULT_START
    end = parse_yyyymmdd(args.end_date) if args.end_date else today
    if start > end:
        raise SystemExit("start-date must be <= end-date")
    result = fetch_sw2_daily_analysis(
        start,
        end,
        chunk_days=args.chunk_days,
        sleep_sec=args.sleep_sec,
        page_size=args.page_size,
    )
    print(
        f"[sw2] done fetched={result['fetched']} inserted={result['inserted']} "
        f"updated={result['updated']} total={result['total_records']} "
        f"latest={result['latest_trade_date']} -> {plate_storage.DEFAULT_DB_FILE}",
        flush=True,
    )
    if result["errors"]:
        print(f"[sw2] errors={len(result['errors'])}", flush=True)
    return result


if __name__ == "__main__":
    _result = main()
    if _result["errors"]:
        raise SystemExit(1)
