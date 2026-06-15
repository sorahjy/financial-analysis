"""
统一基金数据抓取脚本。

默认一次完成两类数据：
  1. 从天天基金 F10DataApi 增量抓取历史净值，写入 data/fund_data.sqlite3，
  2. 从天天基金估值接口抓取实时净值估算，写入 SQLite。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

import requests

from fund_storage import (
    connect as connect_fund_db,
    import_nav_store_payload,
    load_nav_store as db_load_nav_store,
    save_nav_entry,
    save_realtime_estimates,
)


FUND_CODES_FILE = os.path.join("data", "fund_codes.json")

NAV_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://fundf10.eastmoney.com/",
}
NAV_API_URL = "https://fundf10.eastmoney.com/F10DataApi.aspx"

ESTIMATE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://fund.eastmoney.com/",
}
ESTIMATE_API_URL = "https://fundgz.1234567.com.cn/js/{code}.js"

MAX_RETRIES = 3
META_RE = re.compile(r"records:(\d+),pages:(\d+),curpage:(\d+)")
ROW_RE = re.compile(r"<td[^>]*>(.*?)</td>")
JSONP_RE = re.compile(r"jsonpgz\((.*)\)")


def load_fund_codes() -> List[str]:
    with open(FUND_CODES_FILE, encoding="utf-8") as f:
        return json.load(f)


def parse_response(text: str) -> tuple[List[Dict[str, str]], int, int]:
    """解析 F10DataApi 返回的 JS 变量，提取数据行和分页信息。"""
    meta_match = META_RE.search(text)
    if not meta_match:
        return [], 0, 0

    total_records = int(meta_match.group(1))
    total_pages = int(meta_match.group(2))

    rows = []
    tr_parts = text.split("<tr>")[2:]
    for tr in tr_parts:
        cells = ROW_RE.findall(tr)
        if len(cells) >= 4:
            rows.append(
                {
                    "date": cells[0].strip(),
                    "nav": cells[1].strip(),
                    "nav_acc": cells[2].strip(),
                    "daily_growth_rate": cells[3].strip().replace("%", ""),
                }
            )

    return rows, total_records, total_pages


def _get_nav_with_retry(params: Dict[str, Any]) -> str:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(NAV_API_URL, params=params, headers=NAV_HEADERS, timeout=15)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_err = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt + random.uniform(0, 1))
    if last_err:
        raise last_err
    raise RuntimeError("基金历史净值请求失败")


def fetch_range(code: str, sdate: str, edate: str) -> List[Dict[str, str]]:
    """爬取单只基金 [sdate, edate] 区间内的全部净值数据。"""
    collected = []
    page = 1

    while True:
        params = {
            "type": "lsjz",
            "code": code,
            "page": page,
            "per": 20,
            "sdate": sdate,
            "edate": edate,
        }
        text = _get_nav_with_retry(params)
        rows, _total_records, total_pages = parse_response(text)

        if not rows:
            break

        collected.extend(rows)

        if page >= total_pages:
            break
        page += 1
        time.sleep(random.uniform(0.15, 0.35))

    return collected


def merge_records(existing: Iterable[Dict[str, Any]], new_records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """合并两组记录，按日期去重并排序。"""
    by_date = {}
    for record in existing:
        by_date[record["date"]] = record
    for record in new_records:
        by_date[record["date"]] = record
    return sorted(by_date.values(), key=lambda item: item["date"])


def load_store() -> Dict[str, Any]:
    conn = connect_fund_db()
    try:
        return db_load_nav_store(conn)
    finally:
        conn.close()


def save_store(store: Dict[str, Any]) -> None:
    conn = connect_fund_db()
    try:
        import_nav_store_payload(conn, store)
    finally:
        conn.close()


def fetch_fund_incremental(code: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    增量爬取单只基金：
      1. 向后补：end_date+1 -> 今天
      2. 向前追溯：从 start_date 往前，每批1年，直到无数据
    """
    today = datetime.now().strftime("%Y-%m-%d")
    existing_records = entry.get("records", [])
    start_date = entry.get("start_date")
    end_date = entry.get("end_date")

    new_records = []

    if end_date and end_date < today:
        next_day = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"  -> 补后段 {next_day} ~ {today}", end="", flush=True)
        try:
            rows = fetch_range(code, next_day, today)
        except Exception as exc:
            rows = []
            print(f" 请求失败({exc})，本次跳过")
        if rows:
            new_records.extend(rows)
            print(f" +{len(rows)}条")
        else:
            print(" 无新数据")
    elif not end_date:
        sdate = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        print(f"  -> 首次爬取 {sdate} ~ {today}", end="", flush=True)
        try:
            rows = fetch_range(code, sdate, today)
        except Exception as exc:
            print(f" 请求失败({exc})，本次跳过")
            return entry
        if rows:
            new_records.extend(rows)
            print(f" +{len(rows)}条")
        else:
            print(" 无数据")
            return entry

    all_so_far = merge_records(existing_records, new_records)
    if all_so_far:
        earliest = all_so_far[0]["date"]
    elif start_date:
        earliest = start_date
    else:
        return entry

    batch = 0
    max_back_batches = 20
    while batch < max_back_batches:
        batch_end = (datetime.strptime(earliest, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        batch_start = (datetime.strptime(batch_end, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")

        print(f"  <- 追溯 {batch_start} ~ {batch_end}", end="", flush=True)
        try:
            rows = fetch_range(code, batch_start, batch_end)
        except Exception as exc:
            print(f" 请求失败({exc})，本次中止追溯")
            break
        if not rows:
            print(" 已到最早")
            break

        new_records.extend(rows)
        print(f" +{len(rows)}条")
        earliest = batch_start
        batch += 1
        time.sleep(random.uniform(0.2, 0.5))

    merged = merge_records(existing_records, new_records)

    if merged:
        return {
            "start_date": merged[0]["date"],
            "end_date": merged[-1]["date"],
            "records": merged,
        }
    return entry


def fetch_nav_history(codes: Optional[List[str]] = None) -> Dict[str, Any]:
    codes = codes or load_fund_codes()
    conn = connect_fund_db()
    try:
        store = db_load_nav_store(conn)
    finally:
        conn.close()
    total = len(codes)

    print("\n========== 基金历史净值 ==========")
    for index, code in enumerate(codes):
        entry = store.get(code, {})
        start_date = entry.get("start_date", "无")
        end_date = entry.get("end_date", "无")
        old_count = len(entry.get("records", []))
        print(f"[{index + 1}/{total}] {code}  已有: {start_date} ~ {end_date} ({old_count}条)")

        updated = fetch_fund_incremental(code, entry)
        store[code] = updated

        new_count = len(updated.get("records", []))
        if new_count > old_count:
            print(f'  ✓ 更新后: {updated["start_date"]} ~ {updated["end_date"]} ({new_count}条)')
        else:
            print("  - 无新增")

        if updated.get("records"):
            conn = connect_fund_db()
            try:
                save_nav_entry(conn, code, updated)
            finally:
                conn.close()

    store = load_store()

    print(f"\n历史净值完成！共 {len(store)} 只基金")
    for code in codes:
        entry = store.get(code, {})
        print(
            f'  {code}: {entry.get("start_date", "?")} ~ '
            f'{entry.get("end_date", "?")} ({len(entry.get("records", []))}条)'
        )
    return store


def fetch_estimate(code: str) -> Optional[Dict[str, str]]:
    """获取单只基金的实时估算数据，返回 dict 或 None。"""
    url = ESTIMATE_API_URL.format(code=code)
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=ESTIMATE_HEADERS, timeout=10)
            response.encoding = "utf-8"
            match = JSONP_RE.search(response.text)
            if not match:
                return None
            data = json.loads(match.group(1))
            return {
                "gsz": data.get("gsz", ""),
                "gszzl": data.get("gszzl", ""),
                "gztime": data.get("gztime", ""),
                "dwjz": data.get("dwjz", ""),
            }
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt + random.uniform(0, 0.5))
            else:
                print(f"  [ERROR] {code}: {exc}")
    return None


def fetch_realtime_estimates(codes: Optional[List[str]] = None) -> Dict[str, Dict[str, str]]:
    codes = codes or load_fund_codes()
    estimates = {}
    total = len(codes)

    print("\n========== 基金实时估算 ==========")
    for index, code in enumerate(codes):
        print(f"[{index + 1}/{total}] 获取估算: {code}", end="", flush=True)
        result = fetch_estimate(code)
        if result:
            estimates[code] = result
            print(f'  gsz={result["gsz"]} gszzl={result["gszzl"]}% gztime={result["gztime"]}')
        else:
            print("  无数据")
        if index < total - 1:
            time.sleep(random.uniform(0.1, 0.3))

    conn = connect_fund_db()
    try:
        save_realtime_estimates(conn, estimates)
    finally:
        conn.close()

    print(f"\n实时估算完成！共获取 {len(estimates)}/{total} 只基金")
    return estimates


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch fund NAV history and realtime estimates")
    parser.add_argument(
        "--mode",
        choices=("all", "nav-history", "realtime-estimate"),
        default="all",
        help="要抓取的数据类型，默认 all",
    )
    args = parser.parse_args(argv)

    codes = load_fund_codes()
    if args.mode in ("all", "nav-history"):
        fetch_nav_history(codes)
    if args.mode in ("all", "realtime-estimate"):
        fetch_realtime_estimates(codes)


if __name__ == "__main__":
    main()
