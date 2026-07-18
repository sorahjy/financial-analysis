"""用 AkShare 新浪全 A 快照补齐申万三级行业的最新交易日成交额。

这个模块只提供独立、可注入的最新日适配器，不负责合并历史或写生产报告：

1. 用新浪单股行情确认最近一个已经收盘的交易日；
2. 只调用一次 ``akshare.stock_zh_a_spot()`` 获取全 A 成交额（元）；
3. 按调用方传入的当前 SW3 成分聚合为行业成交额（亿元）。

AkShare 在真正抓取快照时才导入。测试可注入 ``anchor_fetcher`` 和
``spot_fetcher``，不需要访问网络。
"""

from __future__ import annotations

import importlib
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import requests


DEFAULT_ANCHOR_SYMBOL = "sh600000"
DEFAULT_CLOSE_TIME = time(15, 0)
DEFAULT_MAX_ANCHOR_AGE_DAYS = 14
DEFAULT_MIN_MARKET_ROWS = 5000
DEFAULT_MIN_SEGMENT_COVERAGE = 0.90
DEFAULT_MIN_WEIGHT_COVERAGE = 0.95
SOURCE_NAME = "akshare_stock_zh_a_spot_component"
SINA_QUOTE_URL = "https://hq.sinajs.cn/list={symbol}"
AKSHARE_LATEST_CACHE_SCHEMA = "sw3_akshare_latest_cache.v1"
AKSHARE_LATEST_CACHE_FILE = Path("data/capital/sw3_akshare_latest_cache.json")


class AkShareLatestError(RuntimeError):
    """AkShare 最新交易日适配失败。"""


class AkShareDependencyError(AkShareLatestError):
    """AkShare 未安装或无法导入。"""


class AkShareDataError(AkShareLatestError):
    """上游数据不满足完整性要求。"""


@dataclass
class AkShareLatestAmountBatch:
    """可直接合并进历史序列的最新日批次。"""

    histories: Dict[str, Dict[str, float]] = field(default_factory=dict)
    trade_date: str = ""
    expected_as_of_date: str = ""
    source: str = SOURCE_NAME
    coverage_pct_by_code: Dict[str, float] = field(default_factory=dict)
    source_by_code_date: Dict[str, Dict[str, str]] = field(default_factory=dict)
    derived_dates_by_code: Dict[str, set[str]] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    quality: Dict[str, Any] = field(default_factory=dict)


def _error_text(exc: Exception, limit: int = 240) -> str:
    text = " ".join(str(exc).split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def normalize_stock_code(value: Any) -> str:
    """把 ``sh600000``、``600000.SH``、数字等规范为六位代码。"""
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value).zfill(6) if 0 <= value <= 999999 else ""
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return ""
        integer = int(value)
        return str(integer).zfill(6) if 0 <= integer <= 999999 else ""

    text = str(value).strip()
    if text.isdigit() and len(text) <= 6:
        return text.zfill(6)
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
    return match.group(1) if match else ""


def _float_or_none(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_present(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row:
            return row.get(name)
    return None


def _records(raw: Any) -> List[Mapping[str, Any]]:
    if raw is None:
        return []
    if hasattr(raw, "to_dict") and not isinstance(raw, Mapping):
        try:
            rows = raw.to_dict("records")
        except (TypeError, ValueError):
            rows = []
    elif isinstance(raw, Mapping):
        nested = raw.get("data")
        rows = nested if isinstance(nested, list) else [raw]
    elif isinstance(raw, (list, tuple)):
        rows = list(raw)
    else:
        rows = []
    return [row for row in rows if isinstance(row, Mapping)]


def fetch_akshare_a_spot() -> Any:
    """延迟导入 AkShare，并调用一次新浪全 A 快照。"""
    try:
        akshare = importlib.import_module("akshare")
    except Exception as exc:  # pragma: no cover - 具体导入错误依赖本机环境
        raise AkShareDependencyError(f"无法导入 akshare: {exc}") from exc
    fetcher = getattr(akshare, "stock_zh_a_spot", None)
    if not callable(fetcher):
        raise AkShareDependencyError("当前 akshare 不提供 stock_zh_a_spot")
    try:
        return fetcher()
    except Exception as exc:
        raise AkShareLatestError(f"stock_zh_a_spot 抓取失败: {exc}") from exc


def fetch_akshare_trade_calendar() -> Any:
    """延迟导入 AkShare，并读取新浪交易日历。"""
    try:
        akshare = importlib.import_module("akshare")
    except Exception as exc:  # pragma: no cover - 具体导入错误依赖本机环境
        raise AkShareDependencyError(f"无法导入 akshare: {exc}") from exc
    fetcher = getattr(akshare, "tool_trade_date_hist_sina", None)
    if not callable(fetcher):
        raise AkShareDependencyError("当前 akshare 不提供 tool_trade_date_hist_sina")
    try:
        return fetcher()
    except Exception as exc:
        raise AkShareLatestError(f"新浪交易日历抓取失败: {exc}") from exc


def parse_sina_quote_anchor(text: str, *, symbol: str = DEFAULT_ANCHOR_SYMBOL) -> Dict[str, str]:
    """解析新浪单股 quote 的日期与时间字段。"""
    pattern = rf'var\s+hq_str_{re.escape(symbol)}="([^"]*)"'
    match = re.search(pattern, text or "")
    if not match:
        raise AkShareDataError(f"新浪锚点 {symbol} 返回为空或结构异常")
    parts = match.group(1).split(",")
    if len(parts) < 32 or not parts[0].strip():
        raise AkShareDataError(f"新浪锚点 {symbol} 字段不完整")
    return {
        "symbol": symbol,
        "date": parts[30].strip(),
        "time": parts[31].strip(),
    }


def fetch_sina_quote_anchor(
    *,
    symbol: str = DEFAULT_ANCHOR_SYMBOL,
    timeout: int = 8,
    http_get: Optional[Callable[..., Any]] = None,
) -> Dict[str, str]:
    """用一只高流动性股票的新浪 quote 锚定快照交易日。"""
    getter = http_get or requests.get
    response = getter(
        SINA_QUOTE_URL.format(symbol=symbol),
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"},
        timeout=timeout,
    )
    response.raise_for_status()
    response.encoding = response.encoding or "GB18030"
    return parse_sina_quote_anchor(response.text, symbol=symbol)


def _normalize_anchor(raw: Any) -> tuple[str, str]:
    if not isinstance(raw, Mapping):
        raise AkShareDataError("新浪日期锚点必须是映射")
    date_text = str(
        _first_present(raw, ("date", "quote_date", "日期", "trade_date")) or ""
    ).strip()[:10]
    time_value = _first_present(raw, ("time", "时间", "quote_time"))
    time_text = str(time_value or "").strip()
    if " " in time_text:
        possible_date, possible_time = time_text.rsplit(" ", 1)
        date_text = date_text or possible_date[:10]
        time_text = possible_time
    return date_text, time_text


def validate_quote_anchor(
    raw: Any,
    *,
    now: Optional[datetime] = None,
    close_time: time = DEFAULT_CLOSE_TIME,
    max_age_days: int = DEFAULT_MAX_ANCHOR_AGE_DAYS,
) -> str:
    """确认锚点是最近且已经收盘的交易日，返回 ``YYYY-MM-DD``。"""
    date_text, time_text = _normalize_anchor(raw)
    try:
        quote_date = date.fromisoformat(date_text)
    except ValueError as exc:
        raise AkShareDataError(f"新浪锚点日期无效: {date_text!r}") from exc
    try:
        quote_time = time.fromisoformat(time_text)
    except ValueError as exc:
        raise AkShareDataError(f"新浪锚点时间无效: {time_text!r}") from exc

    current = now or datetime.now()
    if quote_date.weekday() >= 5:
        raise AkShareDataError(f"新浪锚点落在非交易日: {quote_date.isoformat()}")
    if quote_date > current.date():
        raise AkShareDataError(f"新浪锚点来自未来日期: {quote_date.isoformat()}")
    age_days = (current.date() - quote_date).days
    if max_age_days >= 0 and age_days > max_age_days:
        raise AkShareDataError(
            f"新浪锚点过旧: {quote_date.isoformat()}，距本地日期 {age_days} 天"
        )
    if quote_time < close_time:
        raise AkShareDataError(
            f"新浪锚点尚未收盘: {quote_date.isoformat()} {quote_time.isoformat()}"
        )
    if quote_date == current.date() and current.time() < close_time:
        raise AkShareDataError("本地时间尚未收盘，拒绝把盘中快照当作日终数据")
    return quote_date.isoformat()


def normalize_trade_dates(raw: Any) -> List[str]:
    """把 AkShare DataFrame 或注入列表规范为有序交易日。"""
    if raw is None:
        return []
    if hasattr(raw, "to_dict") and not isinstance(raw, Mapping):
        try:
            values: List[Any] = list(raw.to_dict("records"))
        except (TypeError, ValueError):
            values = []
    elif isinstance(raw, Mapping):
        nested = raw.get("data")
        values = list(nested) if isinstance(nested, list) else [raw]
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        values = []

    dates: set[str] = set()
    for value in values:
        if isinstance(value, Mapping):
            value = _first_present(value, ("trade_date", "日期", "date"))
        if isinstance(value, (date, datetime)):
            text = value.isoformat()[:10]
        else:
            text = str(value or "").strip()[:10]
        try:
            dates.add(date.fromisoformat(text).isoformat())
        except ValueError:
            continue
    return sorted(dates)


def validate_single_trade_day_gap(
    prior_as_of_date: str,
    snapshot_date: str,
    trade_dates: Iterable[str],
) -> int:
    """返回交易日差；只允许 0（已最新）或 1（可补一天）。"""
    try:
        prior = date.fromisoformat(str(prior_as_of_date or "").strip()[:10]).isoformat()
    except ValueError as exc:
        raise AkShareDataError(f"现有截止日无效: {prior_as_of_date!r}") from exc
    try:
        snapshot = date.fromisoformat(str(snapshot_date or "").strip()[:10]).isoformat()
    except ValueError as exc:
        raise AkShareDataError(f"快照日期无效: {snapshot_date!r}") from exc
    calendar = sorted(set(trade_dates))
    if prior not in calendar or snapshot not in calendar:
        raise AkShareDataError(
            f"交易日历未同时覆盖现有截止日 {prior} 和快照日 {snapshot}"
        )
    prior_index = calendar.index(prior)
    snapshot_index = calendar.index(snapshot)
    gap = snapshot_index - prior_index
    if gap < 0:
        raise AkShareDataError(f"快照日 {snapshot} 早于现有截止日 {prior}")
    if gap > 1:
        raise AkShareDataError(
            f"现有截止日 {prior} 到快照日 {snapshot} 相差 {gap} 个交易日，只允许补 1 日"
        )
    return gap


def normalize_spot_amounts(
    raw: Any,
    *,
    min_market_rows: int = DEFAULT_MIN_MARKET_ROWS,
) -> Dict[str, float]:
    """校验全市场快照，并返回 ``{股票代码: 成交额元}``。"""
    rows = _records(raw)
    if not rows:
        raise AkShareDataError("AkShare 全 A 快照为空或结构异常")

    amounts: Dict[str, float] = {}
    invalid_rows: List[int] = []
    duplicate_codes: List[str] = []
    for index, row in enumerate(rows):
        code = normalize_stock_code(
            _first_present(row, ("代码", "code", "证券代码", "symbol"))
        )
        amount = _float_or_none(
            _first_present(row, ("成交额", "amount", "成交金额", "成交额(元)"))
        )
        if not code or amount is None or amount < 0:
            invalid_rows.append(index)
            continue
        if code in amounts:
            duplicate_codes.append(code)
            continue
        amounts[code] = amount

    if invalid_rows:
        preview = ",".join(str(index) for index in invalid_rows[:5])
        raise AkShareDataError(
            f"全 A 快照有 {len(invalid_rows)} 行代码/成交额无效或成交额为负（行 {preview}）"
        )
    if duplicate_codes:
        preview = ",".join(sorted(set(duplicate_codes))[:5])
        raise AkShareDataError(
            f"全 A 快照股票代码不唯一，共 {len(duplicate_codes)} 个重复行（{preview}）"
        )
    minimum = max(1, int(min_market_rows))
    if len(amounts) < minimum:
        raise AkShareDataError(
            f"全 A 快照疑似截断，仅 {len(amounts)} 行，低于最低 {minimum} 行"
        )
    return amounts


def membership_fingerprint(segments: Iterable[Mapping[str, Any]]) -> str:
    """生成与行业代码、声明数量、成分代码和官方权重有关的稳定指纹。"""
    normalized: List[Dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        members: List[Dict[str, Any]] = []
        for member in segment.get("members") or []:
            if not isinstance(member, Mapping):
                continue
            code = normalize_stock_code(
                _first_present(member, ("code", "证券代码", "股票代码", "symbol"))
            )
            if code:
                official_weight = _valid_member_weight(member.get("official_market_cap_ratio"))
                index_weight = _valid_member_weight(member.get("index_weight"))
                members.append({
                    "code": code,
                    "official_market_cap_ratio": official_weight,
                    "index_weight": index_weight,
                })
        declared = _float_or_none(segment.get("member_count"))
        members.sort(key=lambda item: (
            item["code"],
            item["official_market_cap_ratio"] is None,
            item["official_market_cap_ratio"] or 0.0,
            item["index_weight"] is None,
            item["index_weight"] or 0.0,
        ))
        normalized.append({
            "segment_code": normalize_stock_code(segment.get("segment_code")),
            "member_count": max(0, int(declared or 0)),
            "members": members,
        })
    normalized.sort(key=lambda item: (
        item["segment_code"],
        json.dumps(item["members"], sort_keys=True, separators=(",", ":")),
    ))
    canonical = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _valid_member_weight(value: Any) -> Optional[float]:
    weight = _float_or_none(value)
    if weight is None or weight < 0 or weight > 100:
        return None
    return weight


def _effective_member_weight(member: Mapping[str, Any]) -> Optional[float]:
    official_weight = _valid_member_weight(member.get("official_market_cap_ratio"))
    if official_weight is not None:
        return official_weight
    return _valid_member_weight(member.get("index_weight"))


def _batch_to_cache_payload(batch: AkShareLatestAmountBatch) -> Dict[str, Any]:
    return {
        "histories": batch.histories,
        "trade_date": batch.trade_date,
        "expected_as_of_date": batch.expected_as_of_date,
        "source": batch.source,
        "coverage_pct_by_code": batch.coverage_pct_by_code,
        "source_by_code_date": batch.source_by_code_date,
        "derived_dates_by_code": {
            code: sorted(dates) for code, dates in batch.derived_dates_by_code.items()
        },
        "errors": batch.errors,
        "quality": batch.quality,
    }


def _batch_from_cache_payload(raw: Any) -> AkShareLatestAmountBatch:
    if not isinstance(raw, Mapping):
        raise AkShareDataError("AkShare 最新日缓存批次结构异常")
    histories_raw = raw.get("histories")
    coverage_raw = raw.get("coverage_pct_by_code")
    sources_raw = raw.get("source_by_code_date")
    derived_raw = raw.get("derived_dates_by_code")
    errors_raw = raw.get("errors")
    quality_raw = raw.get("quality")
    if not all(isinstance(value, Mapping) for value in (
        histories_raw, coverage_raw, sources_raw, derived_raw, quality_raw
    )) or not isinstance(errors_raw, list):
        raise AkShareDataError("AkShare 最新日缓存字段结构异常")
    return AkShareLatestAmountBatch(
        histories={
            str(code): {str(day): float(amount) for day, amount in daily.items()}
            for code, daily in histories_raw.items()
            if isinstance(daily, Mapping)
        },
        trade_date=str(raw.get("trade_date") or ""),
        expected_as_of_date=str(raw.get("expected_as_of_date") or ""),
        source=str(raw.get("source") or SOURCE_NAME),
        coverage_pct_by_code={
            str(code): float(value) for code, value in coverage_raw.items()
        },
        source_by_code_date={
            str(code): {str(day): str(source) for day, source in daily.items()}
            for code, daily in sources_raw.items()
            if isinstance(daily, Mapping)
        },
        derived_dates_by_code={
            str(code): {str(day) for day in dates}
            for code, dates in derived_raw.items()
            if isinstance(dates, list)
        },
        errors=[dict(error) for error in errors_raw if isinstance(error, Mapping)],
        quality=dict(quality_raw),
    )


def _load_cached_batch(
    cache_file: Path,
    *,
    snapshot_date: str,
    fingerprint: str,
    min_market_rows: int,
    min_segment_coverage: float,
    min_weight_coverage: float,
) -> Optional[AkShareLatestAmountBatch]:
    if not cache_file.exists():
        return None
    raw = json.loads(cache_file.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema") != AKSHARE_LATEST_CACHE_SCHEMA:
        return None
    if raw.get("snapshot_date") != snapshot_date:
        return None
    if raw.get("membership_fingerprint") != fingerprint:
        return None
    batch = _batch_from_cache_payload(raw.get("batch"))
    quality = batch.quality
    cached_rows = _float_or_none(quality.get("snapshot_row_count"))
    cached_segment_min = _float_or_none(quality.get("minimum_segment_coverage_pct"))
    cached_weight_min = _float_or_none(quality.get("minimum_weight_coverage_pct"))
    if cached_rows is None or cached_rows < max(1, int(min_market_rows)):
        return None
    if (
        cached_segment_min is None
        or cached_segment_min + 1e-9 < min(1.0, max(0.0, float(min_segment_coverage))) * 100.0
    ):
        return None
    if (
        cached_weight_min is None
        or cached_weight_min + 1e-9 < min(1.0, max(0.0, float(min_weight_coverage))) * 100.0
    ):
        return None
    return batch


def _atomic_write_cached_batch(
    cache_file: Path,
    *,
    snapshot_date: str,
    fingerprint: str,
    batch: AkShareLatestAmountBatch,
) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": AKSHARE_LATEST_CACHE_SCHEMA,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot_date": snapshot_date,
        "membership_fingerprint": fingerprint,
        "batch": _batch_to_cache_payload(batch),
    }
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_file.name}.", suffix=".tmp", dir=str(cache_file.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, cache_file)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        finally:
            raise


def aggregate_segment_amounts(
    segments: Iterable[Mapping[str, Any]],
    stock_amount_yuan: Mapping[str, float],
    trade_date: str,
    *,
    min_segment_coverage: float = DEFAULT_MIN_SEGMENT_COVERAGE,
    min_weight_coverage: float = DEFAULT_MIN_WEIGHT_COVERAGE,
) -> AkShareLatestAmountBatch:
    """按当前成分聚合最新日成交额，成分表或快照不完整时不输出数值。"""
    histories: Dict[str, Dict[str, float]] = {}
    coverage_by_code: Dict[str, float] = {}
    source_by_code_date: Dict[str, Dict[str, str]] = {}
    derived_dates_by_code: Dict[str, set[str]] = {}
    errors: List[Dict[str, Any]] = []
    seen_segments: set[str] = set()
    expected_memberships = 0
    stored_memberships = 0
    matched_memberships = 0
    membership_complete_segments = 0
    snapshot_complete_segments = 0
    segment_quality_by_code: Dict[str, Dict[str, Any]] = {}

    segment_list = [segment for segment in segments if isinstance(segment, Mapping)]
    minimum = min(1.0, max(0.0, float(min_segment_coverage)))
    weight_minimum = min(1.0, max(0.0, float(min_weight_coverage)))
    for segment in segment_list:
        segment_code = normalize_stock_code(segment.get("segment_code"))
        segment_name = str(segment.get("segment_name") or "").strip()
        if not segment_code:
            errors.append({
                "stage": "membership",
                "segment_code": "",
                "segment_name": segment_name,
                "error": "行业代码为空或无效",
            })
            continue
        if segment_code in seen_segments:
            errors.append({
                "stage": "membership",
                "segment_code": segment_code,
                "segment_name": segment_name,
                "error": "行业代码重复，已忽略后续记录",
            })
            continue
        seen_segments.add(segment_code)

        members: Dict[str, Optional[float]] = {}
        invalid_member_count = 0
        duplicate_member_count = 0
        for member in segment.get("members") or []:
            if not isinstance(member, Mapping):
                invalid_member_count += 1
                continue
            code = normalize_stock_code(
                _first_present(member, ("code", "证券代码", "股票代码", "symbol"))
            )
            if code:
                weight = _effective_member_weight(member)
                if code in members:
                    duplicate_member_count += 1
                    if members[code] is None and weight is not None:
                        members[code] = weight
                else:
                    members[code] = weight
            else:
                invalid_member_count += 1

        declared = _float_or_none(segment.get("member_count"))
        declared_count = max(0, int(declared or 0))
        stored_count = len(members)
        expected_count = max(declared_count, stored_count)
        if expected_count <= 0:
            coverage_by_code[segment_code] = 0.0
            segment_quality_by_code[segment_code] = {
                "declared_member_count": declared_count,
                "stored_valid_member_count": 0,
                "invalid_member_count": invalid_member_count,
                "duplicate_member_count": duplicate_member_count,
                "membership_gate": "count",
                "membership_count_coverage_pct": 0.0,
                "membership_weight_sum_pct": None,
                "membership_complete": False,
                "snapshot_member_count": 0,
                "snapshot_coverage_pct": 0.0,
                "snapshot_complete": False,
                "aggregated": False,
            }
            errors.append({
                "stage": "membership",
                "segment_code": segment_code,
                "segment_name": segment_name,
                "error": "行业没有可聚合的成分股",
            })
            continue

        matched = [code for code in members if code in stock_amount_yuan]
        count_coverage = min(stored_count / declared_count, 1.0) if declared_count else 1.0
        member_weights = list(members.values())
        all_members_weighted = bool(member_weights) and all(
            weight is not None for weight in member_weights
        )
        weight_sum = sum(weight for weight in member_weights if weight is not None)
        if all_members_weighted:
            membership_gate = "weight"
            membership_gate_coverage = weight_sum / 100.0
            membership_threshold = weight_minimum
        else:
            membership_gate = "count"
            membership_gate_coverage = count_coverage
            membership_threshold = minimum
        membership_complete = membership_gate_coverage + 1e-12 >= membership_threshold
        snapshot_coverage = len(matched) / stored_count if stored_count else 0.0
        snapshot_complete = len(matched) == stored_count and stored_count > 0
        effective_coverage = min(membership_gate_coverage, snapshot_coverage, 1.0)
        coverage_by_code[segment_code] = round(effective_coverage * 100.0, 2)
        expected_memberships += expected_count
        stored_memberships += stored_count
        matched_memberships += len(matched)
        membership_complete_segments += int(membership_complete)
        snapshot_complete_segments += int(snapshot_complete)
        segment_quality_by_code[segment_code] = {
            "declared_member_count": declared_count,
            "stored_valid_member_count": stored_count,
            "invalid_member_count": invalid_member_count,
            "duplicate_member_count": duplicate_member_count,
            "membership_gate": membership_gate,
            "membership_count_coverage_pct": round(count_coverage * 100.0, 2),
            "membership_weight_sum_pct": round(weight_sum, 6) if all_members_weighted else None,
            "membership_threshold_pct": round(membership_threshold * 100.0, 2),
            "membership_complete": membership_complete,
            "snapshot_member_count": len(matched),
            "snapshot_coverage_pct": round(snapshot_coverage * 100.0, 2),
            "snapshot_complete": snapshot_complete,
            "aggregated": False,
        }
        if invalid_member_count:
            errors.append({
                "stage": "membership",
                "segment_code": segment_code,
                "segment_name": segment_name,
                "error": f"忽略 {invalid_member_count} 条无效成分记录",
            })
        if not membership_complete:
            if membership_gate == "weight":
                reason = (
                    f"官方权重合计 {weight_sum:.2f}% 低于最低 {weight_minimum * 100.0:.2f}%"
                )
            else:
                reason = (
                    f"本地成分完整率 {count_coverage * 100.0:.2f}% 低于最低 "
                    f"{minimum * 100.0:.2f}%（{stored_count}/{declared_count}）"
                )
            errors.append({
                "stage": "segment_coverage",
                "segment_code": segment_code,
                "segment_name": segment_name,
                "error": reason,
            })
        if not snapshot_complete:
            missing_codes = sorted(set(members) - set(matched))
            errors.append({
                "stage": "segment_snapshot_coverage",
                "segment_code": segment_code,
                "segment_name": segment_name,
                "error": (
                    f"全 A 快照未覆盖全部本地有效成分（{len(matched)}/{stored_count}），"
                    f"缺失 {','.join(missing_codes[:5])}"
                ),
            })
        if not membership_complete or not snapshot_complete:
            continue

        # 快照完整后直接汇总全部本地成分，不按成分数量或权重覆盖率放大。
        total_yuan = sum(float(stock_amount_yuan[code]) for code in members)
        histories[segment_code] = {trade_date: round(total_yuan / 100_000_000.0, 8)}
        source_by_code_date[segment_code] = {trade_date: SOURCE_NAME}
        derived_dates_by_code[segment_code] = {trade_date}
        segment_quality_by_code[segment_code]["aggregated"] = True

    quality = {
        "source": SOURCE_NAME,
        "snapshot_valid": True,
        "trade_date": trade_date,
        "segment_count": len(segment_list),
        "aggregated_segment_count": len(histories),
        "minimum_segment_coverage_pct": round(minimum * 100.0, 2),
        "minimum_weight_coverage_pct": round(weight_minimum * 100.0, 2),
        "membership_complete_segment_count": membership_complete_segments,
        "snapshot_complete_segment_count": snapshot_complete_segments,
        "expected_membership_count": expected_memberships,
        "stored_valid_membership_count": stored_memberships,
        "matched_membership_count": matched_memberships,
        "segment_quality_by_code": segment_quality_by_code,
        "membership_coverage_pct": round(
            stored_memberships / expected_memberships * 100.0, 2
        ) if expected_memberships else 0.0,
        "snapshot_coverage_pct": round(
            matched_memberships / stored_memberships * 100.0, 2
        ) if stored_memberships else 0.0,
    }
    return AkShareLatestAmountBatch(
        histories=histories,
        trade_date=trade_date,
        expected_as_of_date=trade_date,
        source=SOURCE_NAME,
        coverage_pct_by_code=coverage_by_code,
        source_by_code_date=source_by_code_date,
        derived_dates_by_code=derived_dates_by_code,
        errors=errors,
        quality=quality,
    )


def _failed_batch(
    stage: str,
    exc: Exception,
    *,
    expected_as_of_date: str = "",
    **quality: Any,
) -> AkShareLatestAmountBatch:
    payload = {
        "source": SOURCE_NAME,
        "snapshot_valid": False,
        **quality,
    }
    return AkShareLatestAmountBatch(
        trade_date=expected_as_of_date,
        expected_as_of_date=expected_as_of_date,
        source=SOURCE_NAME,
        errors=[{"stage": stage, "error": _error_text(exc)}],
        quality=payload,
    )


def build_akshare_latest_amount_batch(
    segments: Iterable[Mapping[str, Any]],
    *,
    prior_as_of_date: str,
    spot_fetcher: Optional[Callable[[], Any]] = None,
    anchor_fetcher: Optional[Callable[[], Any]] = None,
    trade_calendar_fetcher: Optional[Callable[[], Any]] = None,
    now: Optional[datetime] = None,
    close_time: time = DEFAULT_CLOSE_TIME,
    max_anchor_age_days: int = DEFAULT_MAX_ANCHOR_AGE_DAYS,
    min_market_rows: int = DEFAULT_MIN_MARKET_ROWS,
    min_segment_coverage: float = DEFAULT_MIN_SEGMENT_COVERAGE,
    min_weight_coverage: float = DEFAULT_MIN_WEIGHT_COVERAGE,
    cache_file: Optional[Path] = AKSHARE_LATEST_CACHE_FILE,
) -> AkShareLatestAmountBatch:
    """抓一次全 A 快照并构造 SW3 最新日批次；失败时返回空批次与错误。"""
    segment_list = [dict(segment) for segment in segments if isinstance(segment, Mapping)]
    fingerprint = membership_fingerprint(segment_list)
    anchor_loader = anchor_fetcher or fetch_sina_quote_anchor
    snapshot_loader = spot_fetcher or fetch_akshare_a_spot
    calendar_loader = trade_calendar_fetcher or fetch_akshare_trade_calendar

    try:
        trade_date = validate_quote_anchor(
            anchor_loader(),
            now=now,
            close_time=close_time,
            max_age_days=max_anchor_age_days,
        )
    except Exception as exc:
        return _failed_batch("date_anchor", exc, segment_count=len(segment_list))

    try:
        trade_dates = normalize_trade_dates(calendar_loader())
        gap = validate_single_trade_day_gap(prior_as_of_date, trade_date, trade_dates)
    except Exception as exc:
        return _failed_batch(
            "trade_gap",
            exc,
            expected_as_of_date=trade_date,
            prior_as_of_date=prior_as_of_date,
            segment_count=len(segment_list),
        )

    common_quality = {
        "trade_date": trade_date,
        "expected_as_of_date": trade_date,
        "prior_as_of_date": prior_as_of_date,
        "trade_day_gap": gap,
        "segment_count": len(segment_list),
        "membership_fingerprint": fingerprint,
    }
    if gap == 0:
        return AkShareLatestAmountBatch(
            trade_date=trade_date,
            expected_as_of_date=trade_date,
            source=SOURCE_NAME,
            quality={
                "source": SOURCE_NAME,
                "snapshot_valid": True,
                "snapshot_required": False,
                "snapshot_called": False,
                "already_current": True,
                "cache_hit": False,
                **common_quality,
            },
        )

    cache_warning = ""
    resolved_cache_file = Path(cache_file) if cache_file is not None else None
    if resolved_cache_file is not None:
        try:
            cached = _load_cached_batch(
                resolved_cache_file,
                snapshot_date=trade_date,
                fingerprint=fingerprint,
                min_market_rows=min_market_rows,
                min_segment_coverage=min_segment_coverage,
                min_weight_coverage=min_weight_coverage,
            )
        except Exception as exc:
            cached = None
            cache_warning = _error_text(exc)
        if cached is not None:
            cached.trade_date = trade_date
            cached.expected_as_of_date = trade_date
            cached.quality.update({
                **common_quality,
                "snapshot_required": True,
                "snapshot_called": False,
                "cache_hit": True,
                "anchor_confirmed": True,
                "anchor_confirmation": "cache_snapshot_date_matches_current_anchor",
            })
            return cached

    try:
        stock_amounts = normalize_spot_amounts(
            snapshot_loader(),
            min_market_rows=min_market_rows,
        )
    except Exception as exc:
        return _failed_batch(
            "market_snapshot",
            exc,
            expected_as_of_date=trade_date,
            trade_date=trade_date,
            segment_count=len(segment_list),
        )

    try:
        confirmed_trade_date = validate_quote_anchor(
            anchor_loader(),
            now=now,
            close_time=close_time,
            max_age_days=max_anchor_age_days,
        )
        if confirmed_trade_date != trade_date:
            raise AkShareDataError(
                f"全 A 快照抓取前后交易日锚点变化：{trade_date} -> {confirmed_trade_date}"
            )
    except Exception as exc:
        return _failed_batch(
            "date_anchor_confirm",
            exc,
            expected_as_of_date=trade_date,
            trade_date=trade_date,
            snapshot_called=True,
            snapshot_row_count=len(stock_amounts),
            segment_count=len(segment_list),
        )

    batch = aggregate_segment_amounts(
        segment_list,
        stock_amounts,
        trade_date,
        min_segment_coverage=min_segment_coverage,
        min_weight_coverage=min_weight_coverage,
    )
    batch.quality["snapshot_row_count"] = len(stock_amounts)
    batch.quality["minimum_market_rows"] = max(1, int(min_market_rows))
    batch.quality.update({
        **common_quality,
        "snapshot_required": True,
        "snapshot_called": True,
        "cache_hit": False,
        "anchor_confirmed": True,
        "anchor_confirmation": "before_and_after_snapshot",
    })
    if resolved_cache_file is not None:
        try:
            _atomic_write_cached_batch(
                resolved_cache_file,
                snapshot_date=trade_date,
                fingerprint=fingerprint,
                batch=batch,
            )
        except Exception as exc:
            batch.errors.append({
                "stage": "cache_write",
                "error": f"AkShare 最新日缓存写入失败：{_error_text(exc)}",
            })
    if cache_warning:
        batch.errors.append({
            "stage": "cache_read",
            "error": f"忽略损坏的 AkShare 最新日缓存：{cache_warning}",
        })
    return batch
