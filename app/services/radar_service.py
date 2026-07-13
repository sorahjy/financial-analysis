from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, Dict, List

from app.config import ROOT_DIR
from app.services.job_service import get_job_state, start_command_job

import stock_storage
from stock_crawl_common import load_json_file
from stock_hot_money_radar import (
    AMBUSH_RESULT_FILE,
    fetch_realtime_a_quotes,
    pattern_catalog,
    scoring_model_catalog,
    realtime_rescore_payload,
)


RADAR_RUN_JOB_ID = "radar-run"          # 刷新结果：跑 ambush 重新打分
RADAR_DATA_JOB_ID = "radar-data"        # 刷新数据：跑 stock_radar_fresh_data.sh 重爬行情/板块/题材
RADAR_REFRESH_SCRIPT = "stock_radar_fresh_data.sh"
RADAR_REALTIME_CACHE_SECONDS = 30
RADAR_REALTIME_INTERVAL_SECONDS = 120
_REALTIME_LOCK = threading.Lock()
_REALTIME_CACHE: Dict[str, Any] = {"fetched_at_ts": 0.0, "fetched_at": "", "codes_key": (), "quotes": None}


def radar_payload() -> Dict[str, Any]:
    """游资雷达 ambush 结果（stock_hot_money_radar.py 落盘的 hot_money_ambush.json）。"""
    return load_json_file(AMBUSH_RESULT_FILE, {}) or {}


def _payload_codes(payload: Dict[str, Any]) -> List[str]:
    return [str(row.get("code") or "").zfill(6) for row in payload.get("stocks", []) if row.get("code")]


def _cached_realtime_quotes(codes: List[str]) -> tuple[Dict[str, Dict[str, Any]], str]:
    now = time.time()
    codes_key = tuple(sorted(dict.fromkeys(codes)))
    with _REALTIME_LOCK:
        cached_quotes = _REALTIME_CACHE.get("quotes")
        cached_ts = float(_REALTIME_CACHE.get("fetched_at_ts") or 0.0)
        if (
            cached_quotes is not None
            and _REALTIME_CACHE.get("codes_key") == codes_key
            and now - cached_ts <= RADAR_REALTIME_CACHE_SECONDS
        ):
            return cached_quotes, str(_REALTIME_CACHE.get("fetched_at") or "")
        quotes = fetch_realtime_a_quotes(codes)
        fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _REALTIME_CACHE.update({
            "fetched_at_ts": now,
            "fetched_at": fetched_at,
            "codes_key": codes_key,
            "quotes": quotes,
        })
        return quotes, fetched_at


def radar_realtime_payload() -> Dict[str, Any]:
    """把批量实时行情叠加到当前雷达结果，仅返回页面展示，不写回结果文件。"""
    payload = radar_payload()
    try:
        quotes, fetched_at = _cached_realtime_quotes(_payload_codes(payload))
        return realtime_rescore_payload(payload, quotes, fetched_at=fetched_at)
    except Exception as exc:
        out = dict(payload)
        out["realtime_quote"] = {
            "available": False,
            "source": "realtime_batch",
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "interval_seconds": RADAR_REALTIME_INTERVAL_SECONDS,
            "error": str(exc),
        }
        return out


def radar_pattern_catalog() -> List[Dict[str, Any]]:
    """全部游资形态的结构化说明（编号/名称/类别/信号/命中条件/是否实测有效）。"""
    return pattern_catalog()


def radar_scoring_model() -> Dict[str, Any]:
    return scoring_model_catalog()


def start_radar_run(include_large_cap: bool = True, pool: str = "leader") -> bool:
    """后台跑 `python stock_hot_money_radar.py ambush`（刷新吸筹分/反转分+形态结果）。

    显式传市值口径与候选池（不依赖 CLI 默认）：含大盘=纳入全市值，否则剔除大盘(≤300亿)；
    pool='leader' 细分龙头(吸筹/机会分) / 'hotmoney' 游资小盘universe(超短反转分)。
    """
    pool = pool if pool in ("leader", "hotmoney") else "leader"
    cmd = ["python", "stock_hot_money_radar.py", "ambush",
           "--no-exclude-large-cap" if include_large_cap else "--exclude-large-cap",
           "--pool", pool]
    return start_command_job(
        RADAR_RUN_JOB_ID,
        cmd,
        cwd=ROOT_DIR,
        timeout=1800,
        # The full data-refresh script also rebuilds this same radar artifact.
        # Share its lock so two differently named UI jobs cannot write it at once.
        resource_key="stock-data-refresh",
    )


def radar_run_state() -> Dict[str, Any]:
    return get_job_state(RADAR_RUN_JOB_ID)


def start_radar_data_refresh() -> bool:
    """后台跑 stock_radar_fresh_data.sh（行情全量 + 板块历史 + 题材候选，较重）。"""
    return start_command_job(
        RADAR_DATA_JOB_ID,
        ["bash", RADAR_REFRESH_SCRIPT],
        cwd=ROOT_DIR,
        timeout=10800,                            # 全量爬取耗时长，给 3 小时上限
        resource_key="stock-data-refresh",
    )


def radar_data_state() -> Dict[str, Any]:
    return get_job_state(RADAR_DATA_JOB_ID)


def _period_key(date_text: str, period: str) -> str:
    if period == "week":
        dt = datetime.strptime(date_text, "%Y-%m-%d").date()
        iso = dt.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if period == "month":
        return date_text[:7]
    return date_text


def _aggregate_kline_rows(rows: List[Any], period: str) -> List[Dict[str, Any]]:
    bars: List[Dict[str, Any]] = []
    current_key = ""
    current: Dict[str, Any] | None = None

    for row in rows:
        date_text = row[0]
        key = _period_key(date_text, period)
        if key != current_key:
            if current:
                bars.append(current)
            current_key = key
            current = {
                "date": date_text,
                "start_date": date_text,
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5] or 0,
            }
            continue

        if current is None:
            continue
        current["date"] = date_text
        current["high"] = max(current["high"], row[2])
        current["low"] = min(current["low"], row[3])
        current["close"] = row[4]
        current["volume"] += row[5] or 0

    if current:
        bars.append(current)
    return bars


def kline_bars(code: str, limit: int = 3600, period: str = "day") -> Dict[str, Any]:
    """取单只个股 K 线（日/周/月，升序）供前端画图。"""
    code = stock_storage._normalize_code(code)
    if not code:
        return {"code": "", "period": period, "bars": []}
    period = period if period in {"day", "week", "month"} else "day"
    limit = max(60, min(int(limit or 3600), 6000))
    conn = stock_storage.connect()
    try:
        rows = conn.execute(
            "SELECT date, daily_open, daily_high, daily_low, daily_close, daily_volume "
            "FROM stock_history "
            "WHERE code = ? "
            "AND daily_open IS NOT NULL AND daily_high IS NOT NULL "
            "AND daily_low IS NOT NULL AND daily_close IS NOT NULL "
            "ORDER BY date DESC LIMIT ?",
            (code, limit),
        ).fetchall()
    finally:
        conn.close()
    daily_rows = list(reversed(rows))
    bars = [
        {"date": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5] or 0}
        for r in daily_rows
    ] if period == "day" else _aggregate_kline_rows(daily_rows, period)
    return {"code": code, "period": period, "bars": bars}
