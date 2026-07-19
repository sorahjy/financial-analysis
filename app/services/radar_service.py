from __future__ import annotations

import threading
import time
import sys
from datetime import datetime
from typing import Any, Dict, List, Mapping

from app.config import ROOT_DIR
from app.services.job_service import get_job_state, start_command_job

import stock_storage
import stock_hot_money_radar as hot_money_radar
from stock_crawl_common import load_json_file
from stock_hot_money_radar import (
    AMBUSH_RESULT_FILE,
    fetch_realtime_a_quotes,
    pattern_catalog,
    scoring_model_catalog,
    realtime_rescore_payload,
)
from sw3_industry_heat import (
    SW3_INDUSTRY_HEAT_LEGACY_SCHEMA,
    SW3_INDUSTRY_HEAT_SCHEMA,
)


RADAR_RUN_JOB_ID = "radar-run"          # 刷新结果：跑 ambush 重新打分
RADAR_DATA_JOB_ID = "radar-data"        # 刷新数据：统一脚本刷新股票、ETF、板块和题材
RADAR_REFRESH_SCRIPT = "stock_radar_fresh_data.py"
RADAR_REALTIME_CACHE_SECONDS = 30
RADAR_REALTIME_INTERVAL_SECONDS = 120
RADAR_PATTERN_BACKTEST_YEARS = 6
RADAR_INDUSTRY_HEAT_FILE = ROOT_DIR / "data/capital/sw3_industry_heat.json"
_REALTIME_LOCK = threading.Lock()
_REALTIME_CACHE: Dict[str, Any] = {"fetched_at_ts": 0.0, "fetched_at": "", "codes_key": (), "quotes": None}
LHB_VOLATILITY_WARNING = "近期上龙虎榜，波动加剧"
LEGACY_LHB_WARNING_COPY = {"近期上龙虎榜(避雷)", "近期上龙虎榜，长线按避雷处理"}


def radar_payload() -> Dict[str, Any]:
    """游资雷达 ambush 结果（stock_hot_money_radar.py 落盘的 hot_money_ambush.json）。"""
    payload = load_json_file(AMBUSH_RESULT_FILE, {}) or {}
    for row in payload.get("stocks") or []:
        evidence = row.get("evidence") if isinstance(row, dict) else None
        if not isinstance(evidence, list):
            continue
        for index, item in enumerate(evidence):
            if isinstance(item, str) and item in LEGACY_LHB_WARNING_COPY:
                evidence[index] = LHB_VOLATILITY_WARNING
            elif isinstance(item, dict) and item.get("label") in LEGACY_LHB_WARNING_COPY:
                item["label"] = LHB_VOLATILITY_WARNING
    return payload


def radar_industry_heat_payload() -> Dict[str, Any]:
    """只读三级行业热度结构化报告；按钮点击不会联网或触发重算。"""
    if not RADAR_INDUSTRY_HEAT_FILE.exists():
        return {
            "available": False,
            "schema": SW3_INDUSTRY_HEAT_SCHEMA,
            "error": "尚未生成三级行业热度报告，请先刷新数据。",
        }
    payload = load_json_file(RADAR_INDUSTRY_HEAT_FILE, {}) or {}
    schema = payload.get("schema") if isinstance(payload, dict) else None
    industries = payload.get("industries") if isinstance(payload, dict) else None
    if schema not in {SW3_INDUSTRY_HEAT_SCHEMA, SW3_INDUSTRY_HEAT_LEGACY_SCHEMA}:
        industries = None

    valid_rows = isinstance(industries, list) and bool(industries)
    if valid_rows:
        industry_codes = []
        hottest_ranks = []
        rising_ranks = []
        for row in industries:
            if not isinstance(row, Mapping):
                valid_rows = False
                break
            code = str(row.get("segment_code") or "")
            hottest_rank = row.get("hottest_rank")
            if not code or hottest_rank is None:
                valid_rows = False
                break
            try:
                hottest_rank = int(hottest_rank)
                rising_rank = (
                    int(row["rising_rank"])
                    if row.get("rising_rank") is not None
                    else None
                )
            except (TypeError, ValueError):
                valid_rows = False
                break
            industry_codes.append(code)
            hottest_ranks.append(hottest_rank)
            if rising_rank is not None:
                rising_ranks.append(rising_rank)
        if (
            len(industry_codes) != len(set(industry_codes))
            or sorted(hottest_ranks) != list(range(1, len(industries) + 1))
            or sorted(rising_ranks) != list(range(1, len(rising_ranks) + 1))
        ):
            valid_rows = False

    if valid_rows and schema == SW3_INDUSTRY_HEAT_SCHEMA:
        rankings = payload.get("rankings")
        quality = payload.get("data_quality") or {}
        try:
            rising_candidate_count = int(quality.get("rising_candidate_count") or 0)
        except (TypeError, ValueError):
            rising_candidate_count = -1
        if (
            not isinstance(rankings, dict)
            or not isinstance(rankings.get("hottest"), list)
            or not isinstance(rankings.get("rising"), list)
            or len(rankings["hottest"]) != len(industries)
            or len(rankings["rising"]) != len(rising_ranks)
            or rising_candidate_count != len(rising_ranks)
        ):
            valid_rows = False

    if not valid_rows:
        return {
            "available": False,
            "schema": SW3_INDUSTRY_HEAT_SCHEMA,
            "error": "三级行业热度报告损坏或版本不兼容，请重新刷新数据。",
        }
    return {**payload, "available": True}


def radar_stock_context() -> Dict[str, Any]:
    """返回供其他页面关联展示的轻量雷达缓存摘要。"""
    payload = radar_payload()
    stocks: List[Dict[str, Any]] = []
    seen_codes = set()
    for rank, row in enumerate(payload.get("stocks") or [], start=1):
        code = str(row.get("code") or "").strip().zfill(6)
        if not code or code == "000000" or code in seen_codes:
            continue
        seen_codes.add(code)
        patterns = list(dict.fromkeys(
            str(pattern).upper()
            for pattern in (row.get("patterns") or [])
            if str(pattern).upper().startswith("P") and str(pattern)[1:].isdigit()
        ))
        stocks.append({
            "code": code,
            "rank": rank,
            "patterns": patterns,
            "pattern_phase": row.get("pattern_phase"),
            "opportunity_score": row.get("opportunity_score"),
            "reversal_score": row.get("reversal_score"),
        })
    return {
        "available": bool(stocks),
        "generated_at": payload.get("generated_at"),
        "pool": payload.get("pool"),
        "scored_count": payload.get("scored_count", len(stocks)),
        "stocks": stocks,
    }


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
    pool='leader' 细分龙头 / 'hotmoney' 游资小盘 / 'etf' ETF技术池。
    """
    pool = pool if pool in hot_money_radar.POOLS else "leader"
    cmd = [sys.executable, "-B", "stock_hot_money_radar.py", "ambush",
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
    """通过统一脚本刷新全部底层数据，不按雷达候选池拆分。"""
    return start_command_job(
        RADAR_DATA_JOB_ID,
        [sys.executable, "-B", RADAR_REFRESH_SCRIPT],
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


def _years_ago(date_text: str, years: int) -> str:
    value = datetime.strptime(date_text[:10], "%Y-%m-%d").date()
    try:
        return value.replace(year=value.year - years).isoformat()
    except ValueError:  # 2 月 29 日回看至非闰年
        return value.replace(year=value.year - years, day=28).isoformat()


def kline_bars(
    code: str,
    limit: int = 3600,
    period: str = "day",
    years: int | None = None,
) -> Dict[str, Any]:
    """取单只个股 K 线（日/周/月，升序）供前端画图。"""
    code = stock_storage._normalize_code(code)
    if not code:
        return {"code": "", "period": period, "bars": []}
    period = period if period in {"day", "week", "month"} else "day"
    requested_limit = int(limit or 0)
    all_history = requested_limit <= 0
    limit = max(60, min(requested_limit, 6000)) if not all_history else 0
    conn = stock_storage.connect()
    try:
        where = (
            "WHERE code = ? AND daily_open IS NOT NULL AND daily_high IS NOT NULL "
            "AND daily_low IS NOT NULL AND daily_close IS NOT NULL AND daily_volume IS NOT NULL "
        )
        params: tuple[Any, ...] = (code,)
        if years:
            latest = conn.execute(
                "SELECT MAX(date) FROM stock_history " + where,
                (code,),
            ).fetchone()[0]
            if not latest:
                rows = []
            else:
                cutoff = _years_ago(str(latest), max(1, int(years)))
                sql = (
                    "SELECT date, daily_open, daily_high, daily_low, daily_close, daily_volume "
                    "FROM stock_history " + where + "AND date >= ? ORDER BY date DESC"
                )
                query_params = params + (cutoff,)
                if not all_history:
                    sql += " LIMIT ?"
                    query_params += (limit,)
                rows = conn.execute(sql, query_params).fetchall()
        else:
            sql = (
                "SELECT date, daily_open, daily_high, daily_low, daily_close, daily_volume "
                "FROM stock_history " + where + "ORDER BY date DESC"
            )
            query_params = params
            if not all_history:
                sql += " LIMIT ?"
                query_params += (limit,)
            rows = conn.execute(sql, query_params).fetchall()
    finally:
        conn.close()
    daily_rows = list(reversed(rows))
    bars = [
        {"date": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5] or 0}
        for r in daily_rows
    ] if period == "day" else _aggregate_kline_rows(daily_rows, period)
    return {"code": code, "period": period, "bars": bars}


def _scan_effective_pattern_events(
    code: str,
    bars: List[Dict[str, Any]],
    display_start: int,
    pool: str,
    p1_market_gates: Dict[str, bool],
    capital_histories: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """逐日回放生产形态与疑似吸筹；买卖点复用统一形态集合。"""
    chip_by_index: Dict[int, Any] = {}
    events: List[Dict[str, Any]] = []
    buy_pattern_codes = hot_money_radar.PATTERN_BACKTEST_BUY
    sell_pattern_codes = hot_money_radar.DISTRIBUTION_WARNING_PATTERN_CODES

    for index in range(display_start, len(bars)):
        window_start = max(
            0,
            index - hot_money_radar.PATTERN_EVAL_BARS + 1,
        )
        window = bars[window_start:index + 1]
        if len(window) < hot_money_radar.MIN_BARS:
            continue
        event_date = str(bars[index].get("date") or "")
        p1_market_gate = p1_market_gates.get(event_date, False)
        cached_prior = chip_by_index.get(
            index - hot_money_radar.CHIP_WINNER_RISK_DAYS,
            hot_money_radar._PRIOR_CHIP_UNSET,
        )
        context = hot_money_radar._build_pattern_context(
            code,
            window,
            prior_chip=cached_prior,
            defer_chip=True,
        )
        p1_state = hot_money_radar._p1_rule_state(
            window, p1_market_gate if pool != "etf" else False,
        )
        # 必须扫描全部生产形态；疑似吸筹要求 fired 真正为空，不能只检查核心有效集。
        fired = hot_money_radar.match_patterns(
            code,
            window,
            pool=pool,
            p1_market_gate=p1_market_gate if pool != "etf" else False,
            p1_override=bool(p1_state.get("active")),
            ctx=context,
        )
        # defer_chip 只允许延迟到“形态匹配结束”；吸筹分始终需要当前筹码子分。
        # 显式补算可避免所有形态都在筹码判据前退出时把 chip 错当成 0。
        hot_money_radar._ensure_pattern_chip_context(context)
        if context.get("_chip_computed"):
            chip_by_index[index] = context.get("chip")

        score_base = hot_money_radar._score_bars(
            code, window[-hot_money_radar.LOOKBACK:], ctx=context,
        )
        if score_base is None:
            continue
        capital = hot_money_radar._accumulation_capital_at(
            capital_histories or {}, code, event_date,
        ) if pool != "etf" else {"holder_change": None, "repurchase_recent": False}
        score_row = {
            "sub_scores": score_base.get("sub_scores") or {},
            "patterns": [str(hit.get("code") or "") for hit in fired],
            **capital,
        }
        weights = hot_money_radar.accumulation_model_weights(pool)
        raw_features = hot_money_radar._accumulation_raw_features(score_row)
        accumulation_score = hot_money_radar._accumulation_model_score(
            {key: raw_features[key] for key in weights}, weights,
        )
        is_suspect_accumulation = bool(
            not fired
            and accumulation_score >= hot_money_radar.SUSPECT_ACCUM_SCORE
        )

        distribution_points = hot_money_radar._distribution_warning_points(fired)
        effective_sell_count = hot_money_radar._effective_sell_pattern_count(fired)
        distribution_warning = hot_money_radar._distribution_warning_triggered(fired)
        effective_hits = []
        for hit in fired:
            pattern_code = str(hit.get("code") or "")
            if pattern_code in buy_pattern_codes:
                # 生产买点集合统一使用绿色；不从原始 signal 字段猜测展示语义。
                effective_style = "bullish"
            elif pattern_code in sell_pattern_codes:
                if not distribution_warning:
                    continue
                # 只画统一预警集合内的形态；同日其他 risk 形态不跟随标红。
                effective_style = "risk"
            elif pattern_code in hot_money_radar.PATTERN_EFFECTIVE:
                effective_style = hot_money_radar.PATTERN_EFFECTIVE_STYLE.get(
                    pattern_code, "neutral",
                )
                if effective_style == "risk":
                    continue
            else:
                continue
            if effective_style not in {"bullish", "momentum", "risk"}:
                continue
            effective_hits.append({
                "code": pattern_code,
                "name": str(hit.get("name") or pattern_code),
                "phase": str(hit.get("phase") or ""),
                "signal": str(hit.get("signal") or ""),
                "effective_style": effective_style,
            })
        if is_suspect_accumulation:
            effective_hits.append({
                "code": hot_money_radar.SUSPECT_ACCUM_PATTERN_CODE,
                "name": "疑似吸筹（待确认）",
                "phase": "疑似吸筹",
                "signal": "buy",
                "effective_style": "bullish",
                "synthetic": True,
                "is_suspect_accumulation": True,
            })
        if effective_hits:
            events.append({
                "date": event_date,
                "close": bars[index].get("close"),
                "patterns": effective_hits,
                "distribution_warning_points": distribution_points,
                "effective_sell_pattern_count": effective_sell_count,
                "is_sell_point": distribution_warning,
                "is_buy_point": any(
                    pattern.get("code") in buy_pattern_codes
                    for pattern in effective_hits
                    if not pattern.get("synthetic")
                ),
                "is_suspect_accumulation": is_suspect_accumulation,
                "accumulation_score": accumulation_score,
                "accumulation_threshold": hot_money_radar.SUSPECT_ACCUM_SCORE,
                "matched_production_pattern_count": len(fired),
            })
    return events


def pattern_backtest_events(
    code: str,
    limit: int = 0,
    pool: str = "leader",
    years: int | None = None,
) -> Dict[str, Any]:
    """返回单标的历史产品买卖形态命中点，供 K 线按需标记。

    形态定义均为日线。即使前端正在看周/月 K，这里也始终按日线做 PIT
    回放，再由浏览器把命中日期归入对应周/月区间。
    """
    code = stock_storage._normalize_code(code)
    pool = pool if pool in hot_money_radar.POOLS else hot_money_radar.DEFAULT_POOL
    requested_limit = int(limit or 0)
    all_history = requested_limit <= 0
    limit = max(60, min(requested_limit, 6000)) if not all_history else 0
    years = (
        max(1, min(int(years), RADAR_PATTERN_BACKTEST_YEARS))
        if years is not None
        else None
    )
    warmup_bars = hot_money_radar.PATTERN_EVAL_BARS - 1
    base = {
        "code": code or "",
        "pool": pool,
        "warmup_bars": warmup_bars,
        "history_years": years,
        "history_scope": "all" if years is None and all_history else "bounded",
        "total_bars": 0,
        "evaluated_bars": 0,
        "skipped_bars": 0,
        "scanned_bars": 0,
        "range": {"start": None, "end": None},
        "hits": [],
        "counts": {
            "dates": 0,
            "patterns": 0,
            "bullish": 0,
            "momentum": 0,
            "risk": 0,
            "suspect_accumulation": 0,
        },
        "sell_rule": hot_money_radar.distribution_warning_rule_metadata(),
        "buy_rule": {
            "type": "any_of",
            "operator": "or",
            "pattern_codes": sorted(
                hot_money_radar.PATTERN_BACKTEST_BUY,
                key=lambda value: int(value[1:]),
            ),
            "pattern_count_threshold": 1,
            "trigger_on_any_pattern": True,
            "includes_suspect_accumulation": True,
        },
        "suspect_accumulation_rule": {
            "threshold": hot_money_radar.SUSPECT_ACCUM_SCORE,
            "requires_no_patterns": True,
            "marker_style": "bullish",
            "marker_code": hot_money_radar.SUSPECT_ACCUM_PATTERN_CODE,
            "minimum_history_bars": hot_money_radar.MIN_BARS,
        },
        "validation_scope": "stock_reference" if pool == "etf" else "production",
    }
    if not code:
        return base

    conn = stock_storage.connect()
    try:
        if years is not None:
            latest_row = conn.execute(
                "SELECT MAX(date) FROM stock_history WHERE code = ? "
                "AND daily_close IS NOT NULL AND daily_volume IS NOT NULL",
                (code,),
            ).fetchone()
            latest = str(latest_row[0]) if latest_row and latest_row[0] else ""
        else:
            latest = ""
        if years is not None and latest:
            cutoff = _years_ago(latest, years)
            sql = hot_money_radar._BAR_SQL + "AND date >= ? ORDER BY date DESC"
            query_params: tuple[Any, ...] = (code, cutoff)
        elif years is None:
            sql = hot_money_radar._BAR_SQL + "ORDER BY date DESC"
            query_params = (code,)
        else:
            sql = ""
            query_params = ()
        if sql:
            if not all_history:
                sql += " LIMIT ?"
                query_params += (limit,)
            recent_rows = conn.execute(sql, query_params).fetchall()
            display_rows = list(reversed(recent_rows))
        else:
            display_rows = []
        first_display_date = str(display_rows[0]["date"]) if display_rows else ""
        warmup_rows = (
            conn.execute(
                hot_money_radar._BAR_SQL + "AND date < ? ORDER BY date DESC LIMIT ?",
                (code, first_display_date, warmup_bars),
            ).fetchall()
            if first_display_date
            else []
        )
        ordered_rows = list(reversed(warmup_rows)) + list(display_rows)
        bars = [hot_money_radar._bar(row) for row in ordered_rows]
        display_start = len(warmup_rows)
        scan_dates = [str(bar.get("date") or "") for bar in bars[display_start:]]
        p1_market_gates = (
            {}
            if pool == "etf"
            else hot_money_radar._p1_market_gate_by_date(conn, scan_dates)
        )
        capital_histories = (
            {}
            if pool == "etf"
            else hot_money_radar._load_capital_histories(conn)
        )
    finally:
        conn.close()

    if not display_rows:
        return base

    first_evaluated_index = max(display_start, hot_money_radar.MIN_BARS - 1)
    evaluated_bars = max(0, len(bars) - first_evaluated_index)
    skipped_bars = len(display_rows) - evaluated_bars
    hits = _scan_effective_pattern_events(
        code, bars, display_start, pool, p1_market_gates, capital_histories,
    )
    counts = {
        "dates": len(hits),
        "patterns": 0,
        "bullish": 0,
        "momentum": 0,
        "risk": 0,
        "suspect_accumulation": 0,
    }
    for event in hits:
        counts["patterns"] += len(event["patterns"])
        if event.get("is_suspect_accumulation"):
            counts["suspect_accumulation"] += 1
        # 疑似吸筹用独立计数，避免 synthetic bullish 同时计入形态买点。
        for style in {
            pattern["effective_style"]
            for pattern in event["patterns"]
            if not pattern.get("synthetic")
        }:
            if style in counts:
                counts[style] += 1

    return {
        **base,
        "total_bars": len(display_rows),
        "evaluated_bars": evaluated_bars,
        "skipped_bars": skipped_bars,
        "scanned_bars": evaluated_bars,
        "range": {"start": bars[display_start]["date"], "end": bars[-1]["date"]},
        "hits": hits,
        "counts": counts,
    }
