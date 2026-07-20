"""申万三级行业近 20 个交易日热度报告。

该模块只负责行业层数据抓取、计算和结构化落盘，不改变龙头股票的行业内评分。
历史抓取使用申万宏源指数详情接口的 ``bargainsum``（亿元）；若申万只落后一个
已收盘交易日，生产流程可用 AkShare 新浪全 A 快照按当前成分汇总补齐。调用方可
注入各类 fetcher，让单元测试完全离线。
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import requests


SW3_INDUSTRY_HEAT_SCHEMA = "sw3_industry_heat.v3"
SW3_INDUSTRY_HEAT_V2_SCHEMA = "sw3_industry_heat.v2"
SW3_INDUSTRY_HEAT_LEGACY_SCHEMA = "sw3_industry_heat.v1"
SW3_INDUSTRY_HEAT_FILE = Path("data/capital/sw3_industry_heat.json")
SW3_INDUSTRY_HEAT_HISTORY_CACHE_SCHEMA = "sw3_industry_heat_history_cache.v1"
SW3_INDUSTRY_HEAT_HISTORY_CACHE_FILE = Path("data/capital/sw3_industry_heat_history_cache.json")
SW3_INDUSTRY_HEAT_LATEST_CACHE_FILE = Path("data/capital/sw3_akshare_latest_cache.json")
SWS_TREND_URL = "https://www.swsresearch.com/institute-sw/api/index_publish/trend/"
SWS_ANALYSIS_URL = "https://www.swsresearch.com/institute-sw/api/index_analysis/index_analysis_report/"
SWS_TREND_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.swsresearch.com/institute_sw/allIndex/releasedIndex",
}

DEFAULT_WINDOW_DAYS = 20
DEFAULT_MAX_WORKERS = 6
DEFAULT_MIN_ELIGIBLE_COVERAGE = 0.80
DEFAULT_HISTORY_CACHE_MAX_AGE_HOURS = 2.0


class IndustryHeatIncompleteError(RuntimeError):
    """本轮报告不满足完整性门槛；调用方应保留旧文件。"""


def _segment_code(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))[:6]


def _float_or_none(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round_or_none(value: Optional[float], digits: int) -> Optional[float]:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def _error_text(exc: Exception, limit: int = 220) -> str:
    text = " ".join(str(exc).split())
    if len(text) > limit:
        text = text[:limit - 1] + "…"
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _date_text(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    text = str(value or "").strip()[:10]
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return ""


def _latest_date_with_coverage(
    history_by_code: Mapping[str, Any],
    *,
    expected_count: int,
    minimum_coverage: float,
) -> str:
    """返回覆盖足够多行业的最近日期，避免被单个超前行业误导。"""
    required = max(1, math.ceil(max(0, int(expected_count)) * minimum_coverage))
    counts: Counter[str] = Counter()
    for raw_history in history_by_code.values():
        counts.update(normalize_amount_history(raw_history).keys())
    candidates = [trade_date for trade_date, count in counts.items() if count >= required]
    return max(candidates) if candidates else ""


def _batch_value(batch: Any, name: str, default: Any) -> Any:
    """同时兼容适配器 dataclass 和测试注入的映射。"""
    if isinstance(batch, Mapping):
        return batch.get(name, default)
    return getattr(batch, name, default)


def fetch_akshare_latest_amount_batch(
    segments: Iterable[Mapping[str, Any]],
    *,
    prior_as_of_date: str,
    cache_file: Path,
) -> Any:
    """延迟加载 AkShare 最新日适配器，避免离线计算路径导入外部包。"""
    from sw3_akshare_latest import build_akshare_latest_amount_batch

    return build_akshare_latest_amount_batch(
        segments,
        prior_as_of_date=prior_as_of_date,
        cache_file=cache_file,
    )


def fetch_sw3_daily_amount_history(segment_code: str, timeout: int = 20) -> List[Dict[str, Any]]:
    """读取一个申万三级指数的完整日频成交额序列。

    官方接口的 ``bargainsum`` 与指数详情页成交额口径一致，单位为亿元。这里只
    返回原始记录，日期去重和最近窗口裁剪由 ``normalize_amount_history`` 统一完成。
    """
    code = _segment_code(segment_code)
    if not code:
        raise ValueError(f"无效申万三级行业代码: {segment_code!r}")
    response = requests.get(
        SWS_TREND_URL,
        params={"swindexcode": code, "period": "DAY"},
        headers=SWS_TREND_HEADERS,
        timeout=timeout,
        verify=False,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"申万三级 {code} 成交额历史为空或结构异常")
    return rows


def fetch_sw3_daily_analysis(
    start_date: str,
    end_date: str,
    *,
    timeout: int = 30,
    page_size: int = 10000,
) -> List[Dict[str, Any]]:
    """批量读取三级行业日成交额份额，给官方趋势停更行业做透明估算兜底。"""
    params = {
        "page": "1",
        "page_size": str(page_size),
        "index_type": "三级行业",
        "start_date": start_date,
        "end_date": end_date,
        "type": "DAY",
        "swindexcode": "all",
    }
    rows: List[Dict[str, Any]] = []
    page = 1
    while True:
        params["page"] = str(page)
        response = requests.get(
            SWS_ANALYSIS_URL,
            params=params,
            headers=SWS_TREND_HEADERS,
            timeout=timeout,
            verify=False,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            raise RuntimeError("申万三级指数分析接口返回结构异常")
        rows.extend(row for row in results if isinstance(row, dict))
        total = int(_float_or_none(data.get("count")) or 0)
        if not results or len(rows) >= total:
            break
        page += 1
    if not rows:
        raise RuntimeError("申万三级指数分析接口为空")
    return rows


def normalize_amount_history(raw: Any, keep_latest: int = 80) -> Dict[str, float]:
    """把 DataFrame/接口列表规范为 ``{YYYY-MM-DD: 成交额亿元}``。"""
    if raw is None:
        return {}
    if hasattr(raw, "to_dict") and not isinstance(raw, (dict, list, tuple)):
        try:
            rows = raw.to_dict("records")
        except TypeError:
            rows = []
    elif isinstance(raw, Mapping):
        data = raw.get("data")
        if isinstance(data, list):
            rows = data
        elif raw and all(_date_text(key) for key in raw):
            rows = [{"date": key, "amount_yi": value} for key, value in raw.items()]
        else:
            rows = [raw]
    elif isinstance(raw, (list, tuple)):
        rows = raw
    else:
        rows = []

    by_date: Dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        trade_date = _date_text(row.get("bargaindate", row.get("日期", row.get("date"))))
        amount = _float_or_none(
            row.get("bargainsum", row.get("成交额", row.get("amount_yi")))
        )
        if trade_date and amount is not None and amount >= 0:
            by_date[trade_date] = amount
    dates = sorted(by_date)
    if keep_latest > 0:
        dates = dates[-keep_latest:]
    return {trade_date: by_date[trade_date] for trade_date in dates}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _trend_stats(values: Sequence[float]) -> tuple[float, float]:
    """返回 OLS 斜率与 Pearson 相关系数。"""
    if len(values) < 2:
        return 0.0, 0.0
    xs = list(range(len(values)))
    x_mean = _mean(xs)
    y_mean = _mean(values)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in values)
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    slope = covariance / x_var if x_var else 0.0
    correlation = covariance / math.sqrt(x_var * y_var) if x_var and y_var else 0.0
    return slope, max(-1.0, min(1.0, correlation))


def apply_analysis_amount_fallback(
    history_by_code: Mapping[str, Dict[str, float]],
    analysis_rows: Iterable[Mapping[str, Any]],
    segment_codes: Iterable[str],
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    minimum_coverage: float = DEFAULT_MIN_ELIGIBLE_COVERAGE,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, set[str]], Dict[str, Any]]:
    """用官方成交额份额估算趋势接口停更行业的日成交额。

    每日市场总成交额由仍有精确 ``bargainsum`` 的行业反推：
    ``sum(exact_amount) / sum(exact_bargainsumrate)``。份额字段虽有舍入，但使用
    数百个精确行业汇总后比逐行业用均价/流通市值反推更稳定。
    """
    allowed_codes = {_segment_code(code) for code in segment_codes}
    rate_by_code: Dict[str, Dict[str, float]] = {}
    date_counts: Counter[str] = Counter()
    for row in analysis_rows:
        code = _segment_code(row.get("swindexcode", row.get("指数代码")))
        trade_date = _date_text(row.get("bargaindate", row.get("发布日期")))
        rate = _float_or_none(row.get("bargainsumrate", row.get("成交额占比")))
        if code not in allowed_codes or not trade_date or rate is None or rate < 0:
            continue
        rate_by_code.setdefault(code, {})[trade_date] = rate
        date_counts[trade_date] += 1

    minimum_count = max(1, math.ceil(len(allowed_codes) * minimum_coverage))
    target_dates = sorted(
        trade_date for trade_date, count in date_counts.items() if count >= minimum_count
    )[-window_days:]
    merged = {code: dict(history) for code, history in history_by_code.items()}
    estimated_dates: Dict[str, set[str]] = {}
    daily_market_amount: Dict[str, float] = {}
    if len(target_dates) < window_days:
        return merged, estimated_dates, {
            "target_dates": target_dates,
            "estimated_industry_count": 0,
            "estimated_daily_points": 0,
            "previous_report_estimated_industry_count": 0,
            "previous_report_estimated_daily_points": 0,
        }

    for trade_date in target_dates:
        exact_amount = 0.0
        exact_rate = 0.0
        for code, rates in rate_by_code.items():
            amount = merged.get(code, {}).get(trade_date)
            rate = rates.get(trade_date)
            if amount is not None and rate is not None and rate > 0:
                exact_amount += amount
                exact_rate += rate
        if exact_amount > 0 and exact_rate > 0:
            daily_market_amount[trade_date] = exact_amount / (exact_rate / 100.0)

    for code in allowed_codes:
        history = merged.setdefault(code, {})
        rates = rate_by_code.get(code, {})
        for trade_date in target_dates:
            if trade_date in history:
                continue
            rate = rates.get(trade_date)
            market_amount = daily_market_amount.get(trade_date)
            if rate is None or market_amount is None:
                continue
            history[trade_date] = market_amount * rate / 100.0
            estimated_dates.setdefault(code, set()).add(trade_date)

    return merged, estimated_dates, {
        "target_dates": target_dates,
        "estimated_industry_count": len(estimated_dates),
        "estimated_daily_points": sum(len(dates) for dates in estimated_dates.values()),
        "market_amount_dates": len(daily_market_amount),
    }


def _average_rank_percentiles(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    """较大值较高分；并列项使用平均秩，映射到 0–100。"""
    ordered = sorted(rows, key=lambda row: (float(row[key]), row["segment_code"]))
    if not ordered:
        return {}
    if len(ordered) == 1:
        return {ordered[0]["segment_code"]: 100.0}
    scores: Dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and float(ordered[end][key]) == float(ordered[start][key]):
            end += 1
        average_zero_based_rank = (start + end - 1) / 2.0
        percentile = average_zero_based_rank / (len(ordered) - 1) * 100.0
        for row in ordered[start:end]:
            scores[row["segment_code"]] = percentile
        start = end
    return scores


def _market_cap_value(raw: Any) -> Optional[float]:
    if isinstance(raw, Mapping):
        raw = raw.get("market_cap_yi", raw.get("market_cap"))
    value = _float_or_none(raw)
    return value if value is not None and value > 0 else None


def _segment_market_cap(
    segment: Mapping[str, Any],
    market_cap_by_code: Mapping[str, Any],
) -> Dict[str, Any]:
    members_by_code: Dict[str, Mapping[str, Any]] = {}
    for member in segment.get("members") or []:
        if not isinstance(member, Mapping):
            continue
        code = re.sub(r"\D", "", str(member.get("code") or ""))[:6]
        if code:
            members_by_code[code] = member

    declared_count = int(_float_or_none(segment.get("member_count")) or 0)
    stored_count = len(members_by_code)
    expected_count = max(declared_count, stored_count)
    total_cap = 0.0
    known_count = 0
    live_count = 0
    for code, member in members_by_code.items():
        live_cap = _market_cap_value(market_cap_by_code.get(code))
        cached_cap = _market_cap_value(member.get("market_cap_yi"))
        cap = live_cap if live_cap is not None else cached_cap
        if cap is None:
            continue
        total_cap += cap
        known_count += 1
        if live_cap is not None:
            live_count += 1
    coverage = known_count / expected_count * 100.0 if expected_count else 0.0
    membership_coverage = stored_count / declared_count * 100.0 if declared_count else 100.0
    return {
        "member_count": declared_count,
        "stored_member_count": stored_count,
        "membership_coverage_pct": round(min(membership_coverage, 100.0), 2),
        "market_cap_expected_member_count": expected_count,
        "market_cap_known_member_count": known_count,
        "market_cap_live_member_count": live_count,
        "market_cap_coverage_pct": round(min(coverage, 100.0), 2),
        "market_cap_yi": round(total_cap, 4) if known_count else None,
        "market_cap_is_estimate": bool(known_count < expected_count),
    }


def _base_payload(
    *,
    generated_at: str,
    expected_count: int,
    history_success_count: int,
    errors: List[Dict[str, Any]],
    market_cap_source: str,
    window_days: int,
) -> Dict[str, Any]:
    return {
        "schema": SW3_INDUSTRY_HEAT_SCHEMA,
        "generated_at": generated_at,
        "as_of_date": None,
        "window": {"trading_days": window_days, "start_date": None, "end_date": None, "dates": []},
        "units": {
            "amount": "亿元（申万趋势为主；最新一日可由AkShare新浪全A成分汇总，派生值逐日标记）",
            "market_cap": "亿元（当前成分市值快照，结合覆盖率解读）",
            "market_share": "%（当日行业成交额 / 全部有效三级行业成交额）",
            "heat_index": "首5日平均成交额份额=100",
        },
        "methodology": {
            "common_as_of_date": "选择达到共同覆盖门槛的最近交易日，再统一向前取20日",
            "hottest_rank": "近20个共同交易日成交额合计降序（日均排序等价）",
            "rising_candidate": "末5日平均成交额份额高于首5日，且20日趋势相关系数>0",
            "rising_score": "60%×份额增幅横截面分位 + 40%×趋势相关系数横截面分位",
            "falling_candidate": "末5日平均成交额份额低于首5日，且20日趋势相关系数<0",
            "falling_score": "-份额增幅与-趋势相关系数分别取横截面分位，按60%/40%合成后记为负数",
            "trend_score": "升温沿用正上升分，降温使用负下降分；合并趋势榜按带符号分数降序",
            "ranking_storage": "rankings保存全量紧凑排名；完整指标与20日序列统一保存在industries",
            "amount_fallback": (
                "申万趋势停更行业可用官方成交额份额估算；若全市场只多出一个已收盘交易日，"
                "则用AkShare新浪全A快照按当前成分汇总，不跨多日补洞"
            ),
            "previous_estimate_reuse": (
                "官方份额接口暂时为空时，只续用上一份已通过完整性校验且明确标为估算的日度点；"
                "不续用旧AkShare派生点"
            ),
            "percentile_tie_method": "平均秩映射到0–100",
            "note": "行业热度是研究用相对活跃度，不是收益概率或买入信号。",
        },
        "data_quality": {
            "expected_segment_count": expected_count,
            "history_success_count": history_success_count,
            "history_coverage_pct": round(history_success_count / expected_count * 100.0, 2)
            if expected_count else 0.0,
            "eligible_segment_count": 0,
            "eligible_coverage_pct": 0.0,
            "rising_candidate_count": 0,
            "falling_candidate_count": 0,
            "trend_candidate_count": 0,
            "market_cap_source": market_cap_source,
            "market_cap_expected_memberships": 0,
            "market_cap_known_memberships": 0,
            "market_cap_live_memberships": 0,
            "market_cap_coverage_pct": 0.0,
            "common_date_minimum_segment_count": 0,
            "history_cache_reused_count": 0,
            "exact_current_history_count": 0,
            "estimated_industry_count": 0,
            "estimated_daily_points": 0,
            "expected_as_of_date": None,
            "latest_amount_source": None,
            "latest_amount_sources": [],
            "amount_source_counts": {},
            "derived_industry_count": 0,
            "derived_daily_points": 0,
            "latest_amount_added_industry_count": 0,
            "latest_amount_snapshot": {},
        },
        "rankings": {"hottest": [], "rising": [], "falling": [], "trend": []},
        "industries": [],
        "errors": errors,
    }


def compute_sw3_industry_heat(
    segments: Iterable[Mapping[str, Any]],
    history_by_code: Mapping[str, Any],
    *,
    market_cap_by_code: Optional[Mapping[str, Any]] = None,
    generated_at: Optional[str] = None,
    fetch_errors: Optional[List[Dict[str, Any]]] = None,
    market_cap_source: str = "membership_cache",
    window_days: int = DEFAULT_WINDOW_DAYS,
    common_date_coverage: float = DEFAULT_MIN_ELIGIBLE_COVERAGE,
    estimated_amount_dates_by_code: Optional[Mapping[str, set[str]]] = None,
    derived_amount_dates_by_code: Optional[Mapping[str, set[str]]] = None,
    amount_source_by_code_date: Optional[Mapping[str, Mapping[str, str]]] = None,
    latest_amount_coverage_pct_by_code: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """在统一交易日历上计算热门榜、上升榜及每行业具体数据。"""
    generated_at = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cap_map = market_cap_by_code or {}
    estimated_by_code = estimated_amount_dates_by_code or {}
    derived_by_code = derived_amount_dates_by_code or {}
    source_by_code_date = amount_source_by_code_date or {}
    latest_coverage_by_code = latest_amount_coverage_pct_by_code or {}
    segment_map: Dict[str, Mapping[str, Any]] = {}
    for segment in segments:
        code = _segment_code(segment.get("segment_code"))
        if code and code not in segment_map:
            segment_map[code] = segment

    normalized: Dict[str, Dict[str, float]] = {}
    errors = [dict(item) for item in (fetch_errors or []) if isinstance(item, Mapping)]
    error_codes = {_segment_code(item.get("segment_code")) for item in errors}
    for code in segment_map:
        history = normalize_amount_history(history_by_code.get(code))
        if len(history) >= window_days:
            normalized[code] = history
        elif code not in error_codes:
            errors.append({
                "stage": "history",
                "segment_code": code,
                "segment_name": segment_map[code].get("segment_name", ""),
                "error": f"有效日频成交额不足{window_days}条（{len(history)}条）",
            })

    payload = _base_payload(
        generated_at=generated_at,
        expected_count=len(segment_map),
        history_success_count=len(normalized),
        errors=errors,
        market_cap_source=market_cap_source,
        window_days=window_days,
    )
    if not normalized:
        return payload

    minimum_common = max(1, math.ceil(len(normalized) * common_date_coverage))
    payload["data_quality"]["common_date_minimum_segment_count"] = minimum_common
    date_counts: Counter[str] = Counter()
    for history in normalized.values():
        date_counts.update(history.keys())
    as_of_candidates = [
        trade_date for trade_date, count in date_counts.items()
        if count >= minimum_common
    ]
    if not as_of_candidates:
        payload["errors"].append({
            "stage": "calendar",
            "error": f"没有覆盖至少{minimum_common}个行业的共同截止日",
        })
        return payload
    as_of_date = max(as_of_candidates)
    common_dates = sorted(
        trade_date for trade_date, count in date_counts.items()
        if trade_date <= as_of_date and count >= minimum_common
    )[-window_days:]
    payload["as_of_date"] = as_of_date
    if len(common_dates) < window_days:
        payload["errors"].append({
            "stage": "calendar",
            "error": f"统一交易日不足{window_days}天（仅{len(common_dates)}天）",
        })
        return payload

    eligible_histories: Dict[str, Dict[str, float]] = {}
    for code, history in normalized.items():
        segment = segment_map[code]
        if max(history) < as_of_date:
            payload["errors"].append({
                "stage": "alignment",
                "segment_code": code,
                "segment_name": segment.get("segment_name", ""),
                "error": f"最新日期{max(history)}落后统一截止日{as_of_date}",
            })
            continue
        missing = [trade_date for trade_date in common_dates if trade_date not in history]
        if missing:
            payload["errors"].append({
                "stage": "alignment",
                "segment_code": code,
                "segment_name": segment.get("segment_name", ""),
                "error": f"缺少{len(missing)}个统一交易日",
            })
            continue
        eligible_histories[code] = history

    total_amount_by_date = {
        trade_date: sum(history[trade_date] for history in eligible_histories.values())
        for trade_date in common_dates
    }
    if not eligible_histories or any(value <= 0 for value in total_amount_by_date.values()):
        payload["errors"].append({"stage": "calculation", "error": "有效行业为空或市场成交额合计非正"})
        return payload

    industries: List[Dict[str, Any]] = []
    for code in sorted(eligible_histories):
        segment = segment_map[code]
        amounts = [eligible_histories[code][trade_date] for trade_date in common_dates]
        shares = [
            amount / total_amount_by_date[trade_date] * 100.0
            for trade_date, amount in zip(common_dates, amounts)
        ]
        first_amount = _mean(amounts[:5])
        last_amount = _mean(amounts[-5:])
        first_share = _mean(shares[:5])
        last_share = _mean(shares[-5:])
        amount_growth = (last_amount / first_amount - 1.0) * 100.0 if first_amount > 0 else None
        share_growth = (last_share / first_share - 1.0) * 100.0 if first_share > 0 else None
        slope, correlation = _trend_stats(shares)
        relative_slope = slope / _mean(shares) * 100.0 if _mean(shares) > 0 else None
        cap = _segment_market_cap(segment, cap_map)
        amount_20d = sum(amounts)
        market_cap_yi = cap["market_cap_yi"]
        turnover_intensity = amount_20d / market_cap_yi * 100.0 if market_cap_yi else None
        daily = []
        estimated_dates = estimated_by_code.get(code) or set()
        derived_dates = derived_by_code.get(code) or set()
        explicit_sources = source_by_code_date.get(code) or {}
        for trade_date, amount, share in zip(common_dates, amounts, shares):
            amount_source = str(explicit_sources.get(trade_date) or "")
            if not amount_source:
                amount_source = (
                    "sws_analysis_share_estimate"
                    if trade_date in estimated_dates
                    else "sws_trend"
                )
            daily.append({
                "date": trade_date,
                "amount_yi": round(amount, 4),
                "amount_is_estimate": trade_date in estimated_dates,
                "amount_is_derived": trade_date in derived_dates,
                "amount_source": amount_source,
                "market_share_pct": round(share, 6),
                "heat_index": round(share / first_share * 100.0, 2) if first_share > 0 else None,
            })
        industries.append({
            "segment_code": code,
            "segment_name": str(segment.get("segment_name") or ""),
            "parent_segment": str(segment.get("parent_segment") or ""),
            **cap,
            "amount_20d_yi": round(amount_20d, 4),
            "avg_daily_amount_yi": round(_mean(amounts), 4),
            "latest_amount_yi": round(amounts[-1], 4),
            "first5_avg_amount_yi": round(first_amount, 4),
            "last5_avg_amount_yi": round(last_amount, 4),
            "last5_vs_first5_amount_pct": _round_or_none(amount_growth, 2),
            "market_share_20d_pct": round(amount_20d / sum(total_amount_by_date.values()) * 100.0, 6),
            "avg_daily_market_share_pct": round(_mean(shares), 6),
            "first5_avg_market_share_pct": round(first_share, 6),
            "last5_avg_market_share_pct": round(last_share, 6),
            "last5_vs_first5_share_pct": _round_or_none(share_growth, 2),
            "trend_slope_share_pp_per_day": round(slope, 8),
            "trend_slope_pct_per_day": _round_or_none(relative_slope, 4),
            "trend_correlation": round(correlation, 6),
            "turnover_intensity_20d_pct": _round_or_none(turnover_intensity, 2),
            "amount_estimated_days": sum(1 for trade_date in common_dates if trade_date in estimated_dates),
            "amount_is_estimate": any(trade_date in estimated_dates for trade_date in common_dates),
            "amount_derived_days": sum(1 for trade_date in common_dates if trade_date in derived_dates),
            "amount_is_derived": any(trade_date in derived_dates for trade_date in common_dates),
            "latest_amount_member_coverage_pct": _round_or_none(
                _float_or_none(latest_coverage_by_code.get(code)), 2
            ),
            "amount_sources": sorted({point["amount_source"] for point in daily}),
            "rising_growth_percentile": None,
            "rising_trend_percentile": None,
            "rising_score": None,
            "is_rising_candidate": False,
            "falling_growth_percentile": None,
            "falling_trend_percentile": None,
            "falling_score": None,
            "is_falling_candidate": False,
            "trend_score": None,
            "daily": daily,
        })

    rising_candidates = [
        row for row in industries
        if row["last5_vs_first5_share_pct"] is not None
        and row["last5_vs_first5_share_pct"] > 0
        and row["trend_correlation"] > 0
    ]
    rising_growth_scores = _average_rank_percentiles(
        rising_candidates, "last5_vs_first5_share_pct"
    )
    rising_trend_scores = _average_rank_percentiles(
        rising_candidates, "trend_correlation"
    )
    for row in rising_candidates:
        code = row["segment_code"]
        row["is_rising_candidate"] = True
        row["rising_growth_percentile"] = round(rising_growth_scores[code], 2)
        row["rising_trend_percentile"] = round(rising_trend_scores[code], 2)
        row["rising_score"] = round(
            rising_growth_scores[code] * 0.60 + rising_trend_scores[code] * 0.40,
            2,
        )
        row["trend_score"] = row["rising_score"]

    falling_candidates = [
        row for row in industries
        if row["last5_vs_first5_share_pct"] is not None
        and row["last5_vs_first5_share_pct"] < 0
        and row["trend_correlation"] < 0
    ]
    falling_magnitudes = [
        {
            "segment_code": row["segment_code"],
            "share_decline": -row["last5_vs_first5_share_pct"],
            "negative_correlation": -row["trend_correlation"],
        }
        for row in falling_candidates
    ]
    falling_growth_scores = _average_rank_percentiles(
        falling_magnitudes, "share_decline"
    )
    falling_trend_scores = _average_rank_percentiles(
        falling_magnitudes, "negative_correlation"
    )
    for row in falling_candidates:
        code = row["segment_code"]
        row["is_falling_candidate"] = True
        row["falling_growth_percentile"] = round(falling_growth_scores[code], 2)
        row["falling_trend_percentile"] = round(falling_trend_scores[code], 2)
        row["falling_score"] = round(
            -(falling_growth_scores[code] * 0.60 + falling_trend_scores[code] * 0.40),
            2,
        )
        row["trend_score"] = row["falling_score"]

    hottest_all = sorted(
        industries,
        key=lambda row: (-row["amount_20d_yi"], row["segment_code"]),
    )
    rising_all = sorted(
        rising_candidates,
        key=lambda row: (
            -row["rising_score"],
            -row["last5_vs_first5_share_pct"],
            -row["trend_correlation"],
            row["segment_code"],
        ),
    )
    falling_all = sorted(
        falling_candidates,
        key=lambda row: (
            row["falling_score"],
            row["last5_vs_first5_share_pct"],
            row["trend_correlation"],
            row["segment_code"],
        ),
    )
    trend_all = sorted(
        rising_candidates + falling_candidates,
        key=lambda row: (
            -row["trend_score"],
            -row["last5_vs_first5_share_pct"],
            -row["trend_correlation"],
            row["segment_code"],
        ),
    )
    for rank, row in enumerate(hottest_all, 1):
        row["hottest_rank"] = rank
    for rank, row in enumerate(rising_all, 1):
        row["rising_rank"] = rank
    for rank, row in enumerate(falling_all, 1):
        row["falling_rank"] = rank
    for rank, row in enumerate(trend_all, 1):
        row["trend_rank"] = rank
    for row in industries:
        if "rising_rank" not in row:
            row["rising_rank"] = None
        if "falling_rank" not in row:
            row["falling_rank"] = None
        if "trend_rank" not in row:
            row["trend_rank"] = None

    quality = payload["data_quality"]
    quality["eligible_segment_count"] = len(industries)
    quality["eligible_coverage_pct"] = round(
        len(industries) / len(segment_map) * 100.0, 2
    ) if segment_map else 0.0
    quality["rising_candidate_count"] = len(rising_all)
    quality["falling_candidate_count"] = len(falling_all)
    quality["trend_candidate_count"] = len(trend_all)
    quality["market_cap_expected_memberships"] = sum(
        row["market_cap_expected_member_count"] for row in industries
    )
    quality["market_cap_known_memberships"] = sum(
        row["market_cap_known_member_count"] for row in industries
    )
    quality["market_cap_live_memberships"] = sum(
        row["market_cap_live_member_count"] for row in industries
    )
    quality["market_cap_coverage_pct"] = round(
        quality["market_cap_known_memberships"]
        / quality["market_cap_expected_memberships"] * 100.0,
        2,
    ) if quality["market_cap_expected_memberships"] else 0.0
    amount_source_counts: Counter[str] = Counter(
        point["amount_source"]
        for row in industries
        for point in row["daily"]
    )
    latest_sources = sorted({
        row["daily"][-1]["amount_source"]
        for row in industries
        if row.get("daily")
    })
    quality["amount_source_counts"] = dict(sorted(amount_source_counts.items()))
    quality["latest_amount_sources"] = latest_sources
    quality["latest_amount_source"] = (
        latest_sources[0] if len(latest_sources) == 1 else ("mixed" if latest_sources else None)
    )
    quality["derived_industry_count"] = sum(
        1 for row in industries if row["amount_is_derived"]
    )
    quality["derived_daily_points"] = sum(
        int(row["amount_derived_days"]) for row in industries
    )
    payload["window"] = {
        "trading_days": window_days,
        "start_date": common_dates[0],
        "end_date": common_dates[-1],
        "dates": common_dates,
    }
    payload["rankings"] = {
        "hottest": [
            {
                "rank": rank,
                "segment_code": row["segment_code"],
                "segment_name": row["segment_name"],
                "parent_segment": row["parent_segment"],
                "amount_20d_yi": row["amount_20d_yi"],
                "avg_daily_amount_yi": row["avg_daily_amount_yi"],
                "market_share_20d_pct": row["market_share_20d_pct"],
            }
            for rank, row in enumerate(hottest_all, 1)
        ],
        "rising": [
            {
                "rank": rank,
                "segment_code": row["segment_code"],
                "segment_name": row["segment_name"],
                "parent_segment": row["parent_segment"],
                "last5_vs_first5_share_pct": row["last5_vs_first5_share_pct"],
                "trend_correlation": row["trend_correlation"],
                "rising_score": row["rising_score"],
            }
            for rank, row in enumerate(rising_all, 1)
        ],
        "falling": [
            {
                "rank": rank,
                "segment_code": row["segment_code"],
                "segment_name": row["segment_name"],
                "parent_segment": row["parent_segment"],
                "last5_vs_first5_share_pct": row["last5_vs_first5_share_pct"],
                "trend_correlation": row["trend_correlation"],
                "falling_score": row["falling_score"],
            }
            for rank, row in enumerate(falling_all, 1)
        ],
        "trend": [
            {
                "rank": rank,
                "segment_code": row["segment_code"],
                "segment_name": row["segment_name"],
                "parent_segment": row["parent_segment"],
                "last5_vs_first5_share_pct": row["last5_vs_first5_share_pct"],
                "trend_correlation": row["trend_correlation"],
                "trend_score": row["trend_score"],
                "direction": "rising" if row["is_rising_candidate"] else "falling",
            }
            for rank, row in enumerate(trend_all, 1)
        ],
    }
    payload["industries"] = sorted(industries, key=lambda row: row["segment_code"])
    payload["errors"] = sorted(
        payload["errors"],
        key=lambda item: (str(item.get("stage") or ""), str(item.get("segment_code") or "")),
    )
    return payload


def validate_complete_report(
    payload: Mapping[str, Any],
    *,
    min_eligible_coverage: float = DEFAULT_MIN_ELIGIBLE_COVERAGE,
    required_as_of_date: Optional[str] = None,
) -> None:
    """校验全量行业、完整排名和统一窗口，失败时保留旧报告。"""
    def rank_value(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    schema = payload.get("schema")
    if schema not in {SW3_INDUSTRY_HEAT_SCHEMA, SW3_INDUSTRY_HEAT_V2_SCHEMA}:
        raise IndustryHeatIncompleteError("热度报告 schema 无效")
    is_v3 = schema == SW3_INDUSTRY_HEAT_SCHEMA
    required_date = _date_text(required_as_of_date)
    actual_date = _date_text(payload.get("as_of_date"))
    if required_as_of_date and not required_date:
        raise IndustryHeatIncompleteError(f"预期截止日无效：{required_as_of_date!r}")
    if required_date and actual_date != required_date:
        raise IndustryHeatIncompleteError(
            f"报告截止日未到最新已收盘交易日：实际{actual_date or '无'}，要求{required_date}"
        )
    quality = payload.get("data_quality") or {}
    expected = int(quality.get("expected_segment_count") or 0)
    eligible = int(quality.get("eligible_segment_count") or 0)
    required = math.ceil(expected * min_eligible_coverage)
    dates = ((payload.get("window") or {}).get("dates") or [])
    if expected <= 0 or eligible < required:
        raise IndustryHeatIncompleteError(
            f"有效行业覆盖不足：{eligible}/{expected}，要求至少{required}"
        )
    if len(dates) != int((payload.get("window") or {}).get("trading_days") or 0):
        raise IndustryHeatIncompleteError("统一交易日窗口不完整")

    industries = payload.get("industries")
    if not isinstance(industries, list) or len(industries) != eligible:
        raise IndustryHeatIncompleteError(
            f"全量行业明细不完整：{len(industries) if isinstance(industries, list) else 0}/{eligible}"
        )
    industry_by_code: Dict[str, Mapping[str, Any]] = {}
    for row in industries:
        if not isinstance(row, Mapping):
            raise IndustryHeatIncompleteError("全量行业明细包含非法记录")
        code = _segment_code(row.get("segment_code"))
        if not code or code in industry_by_code:
            raise IndustryHeatIncompleteError("全量行业明细代码为空或重复")
        daily = row.get("daily")
        if (
            not isinstance(daily, list)
            or not all(isinstance(point, Mapping) for point in daily)
            or [point.get("date") for point in daily] != dates
        ):
            raise IndustryHeatIncompleteError(f"{code}的20日序列与统一窗口不一致")
        industry_by_code[code] = row

    hottest_ranks = sorted(
        rank_value(row.get("hottest_rank")) for row in industries
    )
    if hottest_ranks != list(range(1, eligible + 1)):
        raise IndustryHeatIncompleteError("最热门全量排名不连续或不完整")

    rising_candidates = []
    for row in industries:
        growth = _float_or_none(row.get("last5_vs_first5_share_pct"))
        correlation = _float_or_none(row.get("trend_correlation"))
        if growth is not None and growth > 0 and correlation is not None and correlation > 0:
            if _float_or_none(row.get("rising_score")) is None:
                raise IndustryHeatIncompleteError("升温行业缺少有效上升分")
            rising_candidates.append(row)
    rising_count = len(rising_candidates)
    rising_candidate_codes = {
        _segment_code(row.get("segment_code")) for row in rising_candidates
    }
    if rank_value(quality.get("rising_candidate_count")) != rising_count:
        raise IndustryHeatIncompleteError("升温行业数量与数据质量摘要不一致")
    rising_ranks = sorted(rank_value(row.get("rising_rank")) for row in rising_candidates)
    if rising_ranks != list(range(1, rising_count + 1)):
        raise IndustryHeatIncompleteError("升温行业全量排名不连续或不完整")
    for row in industries:
        is_candidate = _segment_code(row.get("segment_code")) in rising_candidate_codes
        if bool(row.get("is_rising_candidate")) != is_candidate:
            raise IndustryHeatIncompleteError("升温候选标记与计算口径不一致")
        if not is_candidate and row.get("rising_rank") is not None:
            raise IndustryHeatIncompleteError("非升温行业不应包含升温排名")

    falling_candidates: List[Mapping[str, Any]] = []
    trend_candidates: List[Mapping[str, Any]] = []
    if is_v3:
        falling_candidates = [
            row for row in industries
            if (_float_or_none(row.get("last5_vs_first5_share_pct")) or 0.0) < 0
            and (_float_or_none(row.get("trend_correlation")) or 0.0) < 0
        ]
        falling_candidate_codes = {
            _segment_code(row.get("segment_code")) for row in falling_candidates
        }
        falling_count = len(falling_candidates)
        if rank_value(quality.get("falling_candidate_count")) != falling_count:
            raise IndustryHeatIncompleteError("降温行业数量与数据质量摘要不一致")
        falling_ranks = sorted(
            rank_value(row.get("falling_rank")) for row in falling_candidates
        )
        if falling_ranks != list(range(1, falling_count + 1)):
            raise IndustryHeatIncompleteError("降温行业全量排名不连续或不完整")

        trend_candidates = rising_candidates + falling_candidates
        trend_candidate_codes = rising_candidate_codes | falling_candidate_codes
        trend_count = len(trend_candidates)
        if rank_value(quality.get("trend_candidate_count")) != trend_count:
            raise IndustryHeatIncompleteError("趋势行业数量与数据质量摘要不一致")
        trend_ranks = sorted(
            rank_value(row.get("trend_rank")) for row in trend_candidates
        )
        if trend_ranks != list(range(1, trend_count + 1)):
            raise IndustryHeatIncompleteError("趋势行业全量排名不连续或不完整")

        for row in industries:
            code = _segment_code(row.get("segment_code"))
            is_rising = code in rising_candidate_codes
            is_falling = code in falling_candidate_codes
            is_trend = code in trend_candidate_codes
            falling_score = _float_or_none(row.get("falling_score"))
            rising_score = _float_or_none(row.get("rising_score"))
            trend_score = _float_or_none(row.get("trend_score"))

            if bool(row.get("is_falling_candidate")) != is_falling:
                raise IndustryHeatIncompleteError("降温候选标记与计算口径不一致")
            if is_rising:
                if rising_score is None or rising_score < 0:
                    raise IndustryHeatIncompleteError("升温行业缺少非负上升分")
                if trend_score != rising_score:
                    raise IndustryHeatIncompleteError("升温行业趋势分与上升分不一致")
            elif is_falling:
                if falling_score is None or falling_score > 0:
                    raise IndustryHeatIncompleteError("降温行业缺少非正下降分")
                if trend_score != falling_score:
                    raise IndustryHeatIncompleteError("降温行业趋势分与下降分不一致")
            elif trend_score is not None:
                raise IndustryHeatIncompleteError("方向冲突或缺失行业不应包含趋势分")

            if not is_rising and rising_score is not None:
                raise IndustryHeatIncompleteError("非升温行业不应包含上升分")
            if not is_falling:
                if falling_score is not None or row.get("falling_rank") is not None:
                    raise IndustryHeatIncompleteError("非降温行业不应包含下降分或降温排名")
            if not is_trend and row.get("trend_rank") is not None:
                raise IndustryHeatIncompleteError("非趋势行业不应包含趋势排名")

        ordered_trend_codes = [
            _segment_code(row.get("segment_code"))
            for row in sorted(
                trend_candidates,
                key=lambda row: (
                    -float(row["trend_score"]),
                    -float(row["last5_vs_first5_share_pct"]),
                    -float(row["trend_correlation"]),
                    _segment_code(row.get("segment_code")),
                ),
            )
        ]
        ranked_trend_codes = [
            _segment_code(row.get("segment_code"))
            for row in sorted(
                trend_candidates,
                key=lambda row: rank_value(row.get("trend_rank")),
            )
        ]
        if ranked_trend_codes != ordered_trend_codes:
            raise IndustryHeatIncompleteError("趋势榜未按带符号趋势分降序排列")

    rankings = payload.get("rankings") or {}
    hottest = rankings.get("hottest")
    rising = rankings.get("rising")
    if not isinstance(hottest, list) or len(hottest) != eligible:
        raise IndustryHeatIncompleteError("最热门全量榜单不完整")
    if not isinstance(rising, list) or len(rising) != rising_count:
        raise IndustryHeatIncompleteError("升温行业全量榜单不完整")
    falling = rankings.get("falling") if is_v3 else []
    trend = rankings.get("trend") if is_v3 else []
    if is_v3 and (
        not isinstance(falling, list) or len(falling) != len(falling_candidates)
    ):
        raise IndustryHeatIncompleteError("降温行业全量榜单不完整")
    if is_v3 and (
        not isinstance(trend, list) or len(trend) != len(trend_candidates)
    ):
        raise IndustryHeatIncompleteError("趋势行业全量榜单不完整")

    def validate_refs(
        refs: Sequence[Mapping[str, Any]],
        expected_rows: Sequence[Mapping[str, Any]],
        rank_field: str,
        label: str,
        score_field: Optional[str] = None,
    ) -> None:
        expected_codes = {
            _segment_code(row.get("segment_code"))
            for row in expected_rows
        }
        ref_codes = []
        ref_ranks = []
        for ref in refs:
            if not isinstance(ref, Mapping):
                raise IndustryHeatIncompleteError(f"{label}榜包含非法排名引用")
            code = _segment_code(ref.get("segment_code"))
            rank = rank_value(ref.get("rank"))
            if code not in industry_by_code:
                raise IndustryHeatIncompleteError(f"{label}榜引用了未知行业{code}")
            if rank_value(industry_by_code[code].get(rank_field)) != rank:
                raise IndustryHeatIncompleteError(f"{label}榜排名与行业明细不一致")
            if score_field is not None:
                ref_score = _float_or_none(ref.get(score_field))
                row_score = _float_or_none(industry_by_code[code].get(score_field))
                if ref_score is None or ref_score != row_score:
                    raise IndustryHeatIncompleteError(f"{label}榜分数与行业明细不一致")
            ref_codes.append(code)
            ref_ranks.append(rank)
        if set(ref_codes) != expected_codes or len(ref_codes) != len(set(ref_codes)):
            raise IndustryHeatIncompleteError(f"{label}榜行业集合不完整或重复")
        if sorted(ref_ranks) != list(range(1, len(refs) + 1)):
            raise IndustryHeatIncompleteError(f"{label}榜排名不连续")

    validate_refs(hottest, industries, "hottest_rank", "最热门")
    validate_refs(
        rising,
        rising_candidates,
        "rising_rank",
        "升温",
        "rising_score" if is_v3 else None,
    )
    if is_v3:
        validate_refs(
            falling,
            falling_candidates,
            "falling_rank",
            "降温",
            "falling_score",
        )
        validate_refs(
            trend,
            trend_candidates,
            "trend_rank",
            "趋势",
            "trend_score",
        )
        for ref in trend:
            code = _segment_code(ref.get("segment_code"))
            expected_direction = (
                "rising" if code in rising_candidate_codes else "falling"
            )
            if ref.get("direction") != expected_direction:
                raise IndustryHeatIncompleteError("趋势榜方向与行业明细不一致")


def atomic_write_report(output_file: Path, payload: Mapping[str, Any]) -> None:
    """在目标目录写临时文件并原子替换。"""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_file.name}.", suffix=".tmp", dir=str(output_file.parent)
    )
    os.close(fd)
    temp_file = Path(temp_name)
    try:
        with open(temp_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_file, output_file)
    finally:
        try:
            temp_file.unlink()
        except FileNotFoundError:
            pass


def _load_history_cache(cache_file: Path) -> Dict[str, Dict[str, Any]]:
    try:
        with open(cache_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if payload.get("schema") != SW3_INDUSTRY_HEAT_HISTORY_CACHE_SCHEMA:
        return {}
    entries = payload.get("histories")
    return entries if isinstance(entries, dict) else {}


def _cache_entry_is_fresh(entry: Mapping[str, Any], max_age_hours: float) -> bool:
    try:
        fetched_at = datetime.strptime(str(entry.get("fetched_at") or ""), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    return (datetime.now() - fetched_at).total_seconds() <= max_age_hours * 3600


def _write_history_cache(
    cache_file: Path,
    entries: Mapping[str, Mapping[str, Any]],
) -> None:
    payload = {
        "schema": SW3_INDUSTRY_HEAT_HISTORY_CACHE_SCHEMA,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "histories": dict(entries),
    }
    atomic_write_report(cache_file, payload)


def _load_previous_report_estimates(
    output_file: Path,
    *,
    allowed_codes: set[str],
    window_days: int,
    min_eligible_coverage: float,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, set[str]], str]:
    """读取上一份已通过完整性门的份额估算点，供官方分析接口暂时为空时续用。

    这里只复用明确标为 ``amount_is_estimate`` 的点；AkShare 成分汇总点不会在
    多个交易日间滚动累积，仍由最新日适配器逐次执行相邻一日门禁。
    """
    try:
        with open(output_file, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        validate_complete_report(
            previous,
            min_eligible_coverage=min_eligible_coverage,
        )
    except (OSError, json.JSONDecodeError, IndustryHeatIncompleteError):
        return {}, {}, ""

    dates = list((previous.get("window") or {}).get("dates") or [])
    if len(dates) != window_days:
        return {}, {}, ""
    expected_dates = set(dates)
    histories: Dict[str, Dict[str, float]] = {}
    estimated_dates: Dict[str, set[str]] = {}
    for row in previous.get("industries") or []:
        if not isinstance(row, Mapping):
            continue
        code = _segment_code(row.get("segment_code"))
        if code not in allowed_codes:
            continue
        for point in row.get("daily") or []:
            if not isinstance(point, Mapping) or not point.get("amount_is_estimate"):
                continue
            source = str(point.get("amount_source") or "").strip()
            if (
                point.get("amount_is_derived")
                or source.startswith("akshare_")
                or (source and source != "sws_analysis_share_estimate")
            ):
                continue
            trade_date = _date_text(point.get("date"))
            amount = _float_or_none(point.get("amount_yi"))
            if trade_date not in expected_dates or amount is None or amount < 0:
                continue
            histories.setdefault(code, {})[trade_date] = amount
            estimated_dates.setdefault(code, set()).add(trade_date)
    return histories, estimated_dates, _date_text(previous.get("as_of_date"))


def build_sw3_industry_heat_report(
    segments: Iterable[Mapping[str, Any]],
    *,
    history_fetcher: Optional[Callable[[str], Any]] = None,
    market_cap_fetcher: Optional[Callable[[], Mapping[str, Any]]] = None,
    output_file: Path = SW3_INDUSTRY_HEAT_FILE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    retries: int = 2,
    retry_sleep_sec: float = 0.5,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_eligible_coverage: float = DEFAULT_MIN_ELIGIBLE_COVERAGE,
    history_cache_file: Optional[Path] = None,
    history_cache_max_age_hours: float = DEFAULT_HISTORY_CACHE_MAX_AGE_HOURS,
    analysis_fetcher: Optional[Callable[[str, str], Any]] = None,
    latest_amount_fetcher: Optional[Callable[..., Any]] = None,
    latest_cache_file: Optional[Path] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """抓取全体三级行业、计算报告，通过质量门后原子落盘。"""
    segment_list = [dict(segment) for segment in segments if isinstance(segment, Mapping)]
    fetcher = history_fetcher or fetch_sw3_daily_amount_history
    cap_map: Mapping[str, Any] = {}
    cap_source = "membership_cache"
    initial_errors: List[Dict[str, Any]] = []
    if market_cap_fetcher is not None:
        try:
            cap_map = market_cap_fetcher() or {}
            if not isinstance(cap_map, Mapping) or not any(
                _market_cap_value(value) is not None for value in cap_map.values()
            ):
                raise RuntimeError("实时总市值快照为空")
            cap_source = "eastmoney_realtime+membership_cache_fallback"
        except Exception as exc:
            initial_errors.append({
                "stage": "market_cap",
                "error": f"实时总市值快照失败，改用归属缓存：{_error_text(exc)}",
            })

    cache_file = Path(history_cache_file) if history_cache_file is not None else Path(output_file).with_name(
        SW3_INDUSTRY_HEAT_HISTORY_CACHE_FILE.name
    )
    cache_entries = _load_history_cache(cache_file)
    cached_histories: Dict[str, Dict[str, float]] = {}
    histories: Dict[str, Dict[str, float]] = {}
    reusable_codes = set()
    for segment in segment_list:
        code = _segment_code(segment.get("segment_code"))
        entry = cache_entries.get(code)
        if not isinstance(entry, Mapping) or not _cache_entry_is_fresh(entry, history_cache_max_age_hours):
            continue
        history = normalize_amount_history(entry.get("daily"))
        if len(history) >= window_days:
            cached_histories[code] = history
    # 注入 fetcher 的离线测试/研究调用允许直接复用短时缓存；生产默认源则始终实际
    # 请求一次申万接口，缓存只在该行业本轮抓取失败时续传。这样即使收盘前两小时
    # 生成过缓存，收盘后的正式刷新也不会被旧截止日挡住。
    if history_fetcher is not None:
        histories.update(cached_histories)
        reusable_codes.update(cached_histories)
    history_errors: List[Dict[str, Any]] = []
    pending_segments = [
        segment for segment in segment_list
        if _segment_code(segment.get("segment_code")) not in reusable_codes
    ]
    if progress and reusable_codes:
        print(
            f"  [heat] 复用2小时内成功缓存 {len(reusable_codes)}/{len(segment_list)} 个行业",
            flush=True,
        )

    def fetch_one(segment: Mapping[str, Any]) -> tuple[str, Dict[str, float]]:
        code = _segment_code(segment.get("segment_code"))
        if not code:
            raise ValueError("行业代码为空")
        last_error: Optional[Exception] = None
        for attempt in range(max(1, retries)):
            try:
                history = normalize_amount_history(fetcher(code))
                if not history:
                    raise RuntimeError("成交额历史为空")
                return code, history
            except Exception as exc:
                last_error = exc
                if attempt + 1 < max(1, retries) and retry_sleep_sec > 0:
                    time.sleep(retry_sleep_sec * (attempt + 1))
        raise last_error or RuntimeError("成交额历史抓取失败")

    workers = max(1, min(int(max_workers or 1), 12))
    def persist_cache() -> None:
        now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        merged = dict(cache_entries)
        for code, history in histories.items():
            previous = cache_entries.get(code) if code in reusable_codes else None
            merged[code] = {
                "fetched_at": previous.get("fetched_at") if isinstance(previous, Mapping) else now_text,
                "daily": [{"date": trade_date, "amount_yi": amount} for trade_date, amount in history.items()],
            }
        _write_history_cache(cache_file, merged)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_one, segment): segment for segment in pending_segments}
        completed = 0
        for future in as_completed(futures):
            segment = futures[future]
            code = _segment_code(segment.get("segment_code"))
            completed += 1
            try:
                fetched_code, history = future.result()
                histories[fetched_code] = history
            except Exception as exc:
                cached_history = cached_histories.get(code)
                if cached_history:
                    histories[code] = cached_history
                    reusable_codes.add(code)
                    history_errors.append({
                        "stage": "history_cache_fallback",
                        "segment_code": code,
                        "segment_name": segment.get("segment_name", ""),
                        "error": f"本轮抓取失败，复用短时缓存：{_error_text(exc)}",
                    })
                else:
                    history_errors.append({
                        "stage": "history",
                        "segment_code": code,
                        "segment_name": segment.get("segment_name", ""),
                        "error": _error_text(exc),
                    })
            if completed % 20 == 0:
                persist_cache()
            if progress and (completed == len(pending_segments) or completed % 20 == 0):
                print(
                    f"  [heat] 成交额历史 {completed}/{len(pending_segments)}，"
                    f"成功 {len(histories)} / 失败 {len(history_errors)}",
                    flush=True,
                )
    persist_cache()

    calculation_histories = {code: dict(history) for code, history in histories.items()}
    estimated_amount_dates: Dict[str, set[str]] = {}
    fallback_meta = {"estimated_industry_count": 0, "estimated_daily_points": 0}
    if histories:
        latest_date = max(max(history) for history in histories.values() if history)
        exact_current_count = sum(1 for history in histories.values() if latest_date in history)
    else:
        latest_date = ""
        exact_current_count = 0
    required_current = math.ceil(len(segment_list) * min_eligible_coverage)
    if exact_current_count < required_current:
        analysis_loader = analysis_fetcher or fetch_sw3_daily_analysis
        end_day = datetime.now().date()
        start_day = end_day.fromordinal(end_day.toordinal() - 50)
        try:
            analysis_rows = analysis_loader(start_day.isoformat(), end_day.isoformat())
            calculation_histories, estimated_amount_dates, fallback_meta = apply_analysis_amount_fallback(
                histories,
                analysis_rows,
                [_segment_code(segment.get("segment_code")) for segment in segment_list],
                window_days=window_days,
                minimum_coverage=min_eligible_coverage,
            )
            if fallback_meta.get("estimated_industry_count"):
                initial_errors.append({
                    "stage": "amount_fallback",
                    "error": (
                        f"{fallback_meta['estimated_industry_count']}个趋势接口停更行业使用官方成交额份额估算，"
                        f"共{fallback_meta['estimated_daily_points']}个日度点"
                    ),
                })
        except Exception as exc:
            initial_errors.append({
                "stage": "amount_fallback",
                "error": f"申万指数分析成交额份额兜底失败：{_error_text(exc)}",
            })

    previous_estimated_industries: set[str] = set()
    previous_estimated_points = 0
    allowed_segment_codes = {
        _segment_code(segment.get("segment_code")) for segment in segment_list
    }
    previous_estimates, previous_estimated_dates, previous_as_of_date = (
        _load_previous_report_estimates(
            Path(output_file),
            allowed_codes=allowed_segment_codes,
            window_days=window_days,
            min_eligible_coverage=min_eligible_coverage,
        )
    )
    current_common_date = _latest_date_with_coverage(
        calculation_histories,
        expected_count=len(segment_list),
        minimum_coverage=min_eligible_coverage,
    )
    if previous_as_of_date and previous_as_of_date > current_common_date:
        for code, daily in previous_estimates.items():
            target_history = calculation_histories.setdefault(code, {})
            for trade_date, amount in daily.items():
                if trade_date in target_history:
                    continue
                target_history[trade_date] = amount
                estimated_amount_dates.setdefault(code, set()).add(trade_date)
                previous_estimated_industries.add(code)
                previous_estimated_points += 1
        if previous_estimated_points:
            initial_errors.append({
                "stage": "amount_previous_report_fallback",
                "error": (
                    f"申万指数分析接口本轮未恢复完整份额，复用上一份已校验报告中的"
                    f"{len(previous_estimated_industries)}个行业、{previous_estimated_points}个估算点"
                ),
            })

    prior_as_of_date = _latest_date_with_coverage(
        calculation_histories,
        expected_count=len(segment_list),
        minimum_coverage=min_eligible_coverage,
    )
    latest_enabled = history_fetcher is None or latest_amount_fetcher is not None
    expected_as_of_date = ""
    latest_source_by_code_date: Dict[str, Dict[str, str]] = {}
    derived_amount_dates: Dict[str, set[str]] = {}
    latest_coverage_by_code: Dict[str, float] = {}
    latest_quality: Dict[str, Any] = {}
    latest_added_codes: set[str] = set()

    if latest_enabled:
        if not prior_as_of_date:
            raise IndustryHeatIncompleteError("无法确定达到覆盖门槛的现有行业截止日")
        latest_loader = latest_amount_fetcher or fetch_akshare_latest_amount_batch
        resolved_latest_cache = (
            Path(latest_cache_file)
            if latest_cache_file is not None
            else Path(output_file).with_name(SW3_INDUSTRY_HEAT_LATEST_CACHE_FILE.name)
        )
        try:
            latest_batch = latest_loader(
                segment_list,
                prior_as_of_date=prior_as_of_date,
                cache_file=resolved_latest_cache,
            )
        except Exception as exc:
            raise IndustryHeatIncompleteError(
                f"AkShare最新交易日补齐失败：{_error_text(exc)}"
            ) from exc

        expected_as_of_date = _date_text(
            _batch_value(latest_batch, "expected_as_of_date", "")
        )
        latest_quality_raw = _batch_value(latest_batch, "quality", {})
        if isinstance(latest_quality_raw, Mapping):
            latest_quality = dict(latest_quality_raw)
        batch_errors = _batch_value(latest_batch, "errors", [])
        if isinstance(batch_errors, list):
            for raw_error in batch_errors:
                if not isinstance(raw_error, Mapping):
                    continue
                error = dict(raw_error)
                error["stage"] = f"latest_amount_{error.get('stage') or 'unknown'}"
                initial_errors.append(error)
        if not expected_as_of_date:
            raise IndustryHeatIncompleteError(
                "AkShare未能确认最新已收盘交易日，保留上一份完整报告"
            )

        batch_histories = _batch_value(latest_batch, "histories", {})
        batch_sources = _batch_value(latest_batch, "source_by_code_date", {})
        batch_derived = _batch_value(latest_batch, "derived_dates_by_code", {})
        batch_coverage = _batch_value(latest_batch, "coverage_pct_by_code", {})
        batch_default_source = str(
            _batch_value(latest_batch, "source", "akshare_stock_zh_a_spot_component")
            or "akshare_stock_zh_a_spot_component"
        )
        if isinstance(batch_histories, Mapping):
            for raw_code, raw_history in batch_histories.items():
                code = _segment_code(raw_code)
                if not code:
                    continue
                daily = normalize_amount_history(raw_history)
                for trade_date, amount in daily.items():
                    if trade_date != expected_as_of_date:
                        initial_errors.append({
                            "stage": "latest_amount_merge",
                            "segment_code": code,
                            "error": (
                                f"忽略非目标日期{trade_date}的最新日数据；"
                                f"目标为{expected_as_of_date}"
                            ),
                        })
                        continue
                    target_history = calculation_histories.setdefault(code, {})
                    estimated_dates = estimated_amount_dates.setdefault(code, set())
                    is_estimate = trade_date in estimated_dates
                    if trade_date in target_history and not is_estimate:
                        continue
                    target_history[trade_date] = amount
                    estimated_dates.discard(trade_date)
                    latest_added_codes.add(code)
                    raw_coverage = (
                        batch_coverage.get(code, batch_coverage.get(raw_code))
                        if isinstance(batch_coverage, Mapping)
                        else None
                    )
                    coverage = _float_or_none(raw_coverage)
                    if coverage is not None:
                        latest_coverage_by_code[code] = coverage
                    source_map = (
                        batch_sources.get(code, batch_sources.get(raw_code, {}))
                        if isinstance(batch_sources, Mapping)
                        else {}
                    )
                    if not isinstance(source_map, Mapping):
                        source_map = {}
                    latest_source_by_code_date.setdefault(code, {})[trade_date] = str(
                        source_map.get(trade_date) or batch_default_source
                    )
                    derived_dates = (
                        batch_derived.get(code, batch_derived.get(raw_code, set()))
                        if isinstance(batch_derived, Mapping)
                        else set()
                    )
                    if trade_date in set(derived_dates or []):
                        derived_amount_dates.setdefault(code, set()).add(trade_date)
        if progress:
            print(
                f"  [heat] 最新交易日 {expected_as_of_date} · "
                f"AkShare成分汇总补齐 {len(latest_added_codes)}/{len(segment_list)} 个行业",
                flush=True,
            )

    payload = compute_sw3_industry_heat(
        segment_list,
        calculation_histories,
        market_cap_by_code=cap_map,
        fetch_errors=initial_errors + history_errors,
        market_cap_source=cap_source,
        window_days=window_days,
        estimated_amount_dates_by_code=estimated_amount_dates,
        derived_amount_dates_by_code=derived_amount_dates,
        amount_source_by_code_date=latest_source_by_code_date,
        latest_amount_coverage_pct_by_code=latest_coverage_by_code,
    )
    payload["data_quality"]["history_cache_reused_count"] = len(reusable_codes)
    payload["data_quality"]["exact_current_history_count"] = exact_current_count
    output_industries = payload.get("industries") or []
    payload["data_quality"]["estimated_industry_count"] = sum(
        1 for row in output_industries
        if isinstance(row, Mapping) and row.get("amount_is_estimate")
    )
    payload["data_quality"]["estimated_daily_points"] = sum(
        int(row.get("amount_estimated_days") or 0)
        for row in output_industries if isinstance(row, Mapping)
    )
    payload["data_quality"]["previous_report_estimated_industry_count"] = len(
        previous_estimated_industries
    )
    payload["data_quality"]["previous_report_estimated_daily_points"] = (
        previous_estimated_points
    )
    payload["data_quality"]["expected_as_of_date"] = expected_as_of_date or None
    payload["data_quality"]["latest_amount_added_industry_count"] = len(latest_added_codes)
    payload["data_quality"]["latest_amount_snapshot"] = latest_quality
    if progress:
        quality = payload.get("data_quality") or {}
        print(
            f"  [heat] 统一截止 {payload.get('as_of_date')} · "
            f"有效行业 {quality.get('eligible_segment_count')}/{quality.get('expected_segment_count')}",
            flush=True,
        )
        if not quality.get("eligible_segment_count"):
            for item in (payload.get("errors") or [])[:5]:
                print(
                    f"    [heat-error] {item.get('segment_code', '-')}: {item.get('error')}",
                    flush=True,
                )
    validate_complete_report(
        payload,
        min_eligible_coverage=min_eligible_coverage,
        required_as_of_date=expected_as_of_date or None,
    )
    atomic_write_report(Path(output_file), payload)
    return payload
