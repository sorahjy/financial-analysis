"""Hot-money risk factors kept for the next buy-factor search.

This file intentionally extracts only the drawdown-control factors that have
worked in short-horizon tests: P17, P19 and P22.  Keep this layer small so the
next round of 2/5/10-trading-day event studies can add candidate buy factors
without inheriting all of ``stock_hot_money_radar.py``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


Bar = Dict[str, Any]
PatternFn = Callable[[List[Bar], Dict[str, Any]], bool]
PatternSpec = Tuple[str, str, str, str, PatternFn]

# Research target for this stripped-down factor layer.
BACKTEST_HORIZONS: Tuple[int, ...] = (2, 5, 10)

# Same market-data windows as stock_hot_money_radar.py.
MIN_BARS = 40
SHORT_WIN = 5
BASE_WIN = 20
HIGH_WIN = 60
TURNOVER_COVERAGE = 0.7
P17_POSITION_MIN = 0.80
P17_RECENT_DAYS = 10
P17_BASE_DAYS = 6
P17_RUNUP_MIN = 0.16
P17_PULLBACK_MIN = 0.08
P17_PEAK_RECENCY_BARS = 5


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _volume_series(bars: List[Bar]) -> Tuple[List[Optional[float]], str]:
    """Use turnover when coverage is good; otherwise fall back to volume."""
    turns = [b.get("turnover") for b in bars]
    coverage = sum(1 for t in turns if t is not None) / len(turns) if turns else 0.0
    if coverage >= TURNOVER_COVERAGE:
        return turns, "turnover"
    return [b.get("volume") for b in bars], "volume"


def _volume_ratio(vol: List[Optional[float]]) -> Optional[float]:
    """Near-term volume ratio: recent 5-day average / previous 20-day average."""
    if len(vol) < SHORT_WIN + BASE_WIN:
        return None
    short = [v for v in vol[-SHORT_WIN:] if v is not None]
    base = [v for v in vol[-(SHORT_WIN + BASE_WIN):-SHORT_WIN] if v is not None]
    short_avg, base_avg = _mean(short), _mean(base)
    if not short_avg or not base_avg or base_avg <= 0:
        return None
    return short_avg / base_avg


def _close_position_pctile(bars: List[Bar]) -> Optional[float]:
    """Current close percentile in the latest 60 trading days, 0 low to 1 high."""
    closes = [b.get("close") for b in bars[-HIGH_WIN:] if b.get("close") is not None]
    if len(closes) < 20:
        return None
    last = closes[-1]
    return sum(1 for c in closes if c <= last) / len(closes)


def build_risk_context(bars: List[Bar]) -> Dict[str, Any]:
    """Build the small context required by P17/P19/P22."""
    closes = [b.get("close") for b in bars]
    vol, vol_measure = _volume_series(bars)
    return {
        "closes": closes,
        "vol": vol,
        "vol_measure": vol_measure,
        "pos": _close_position_pctile(bars),
        "vol_ratio": _volume_ratio(vol),
    }


def _kl_pc(bars: List[Bar], i: int) -> Optional[float]:
    return bars[i - 1].get("close") if i > 0 else None


def _kl_body(bars: List[Bar], i: int) -> Optional[float]:
    pc = _kl_pc(bars, i)
    o, c = bars[i].get("open"), bars[i].get("close")
    return abs(c - o) / pc if pc and o is not None and c is not None else None


def _pat_inverted_v(bars: List[Bar], ctx: Dict[str, Any]) -> bool:
    """P17 新鲜倒V: recent high-position peak, sharp pullback, no signal-day bounce."""
    pos, closes = ctx["pos"], ctx["closes"]
    required = P17_RECENT_DAYS + P17_BASE_DAYS
    if (
        pos is None
        or pos < P17_POSITION_MIN
        or len(bars) < required
        or len(closes) < required
    ):
        return False
    recent_window = closes[-P17_RECENT_DAYS:]
    recent = [c for c in recent_window if c]
    base = [c for c in closes[-required:-P17_RECENT_DAYS] if c]
    if len(recent) < 8 or not base:
        return False
    peak = max(recent)
    last, previous = closes[-1], closes[-2]
    if not last or not previous or not peak:
        return False
    first_peak_index = next(
        index for index, close in enumerate(recent_window) if close == peak
    )
    return bool(
        peak / min(base) - 1 > P17_RUNUP_MIN
        and last / peak - 1 <= -P17_PULLBACK_MIN
        and first_peak_index >= P17_RECENT_DAYS - P17_PEAK_RECENCY_BARS
        and last <= previous
    )


def _pat_dump_bigbear(bars: List[Bar], ctx: Dict[str, Any]) -> bool:
    """P19 灌压巨量大阴: high-position heavy bearish candle."""
    pos, vr = ctx["pos"], ctx["vol_ratio"]
    if pos is None or pos < 0.70 or vr is None or vr <= 1.8:
        return False
    body = _kl_body(bars, len(bars) - 1)
    o = bars[-1].get("open")
    c = bars[-1].get("close")
    h = bars[-1].get("high")
    l = bars[-1].get("low")
    if None in (o, c, h, l) or h <= l:
        return False
    return body is not None and body > 0.06 and c < o and (c - l) / (h - l) < 0.25


def _pat_failed_breakout(bars: List[Bar], ctx: Dict[str, Any]) -> bool:
    """P22 放量假突破: intraday 40-day high breakout rejected by the close."""
    if len(bars) < 42:
        return False
    highs = [b.get("high") for b in bars if b.get("high") is not None]
    if len(highs) < 42:
        return False
    high_40_prior = max(highs[-41:-1])
    today_high, c = bars[-1].get("high"), ctx["closes"][-1]
    vol = ctx["vol"]
    base = _mean([v for v in vol[-21:-1] if v])
    today_v = vol[-1] if vol else None
    if not high_40_prior or today_high is None or not c or not base or not today_v:
        return False
    day_ratio = today_v / base
    return today_high > high_40_prior and c <= high_40_prior and day_ratio > 1.8


RISK_PATTERNS: List[PatternSpec] = [
    ("P17", "倒V反转", "出货", "sell", _pat_inverted_v),
    ("P19", "灌压巨量大阴", "出货", "sell", _pat_dump_bigbear),
    ("P22", "放量假突破", "出货", "sell", _pat_failed_breakout),
]

RISK_PATTERN_DESC: Dict[str, str] = {
    "P17": "位置>=0.80 + 较前期冲高>16%后从峰值回落<=-8% + 峰值首次出现于最近5根 + 信号日不反弹：新鲜倒V",
    "P19": "位置>=0.70 + 量比>1.8 + 实体跌>6%且收在当日价区下1/4：巨量灌压大阴",
    "P22": "今日盘中破前40日高但收盘没站上 + 当日放量>1.8x：放量假突破(高位拒绝)",
}


def risk_pattern_catalog() -> List[Dict[str, str]]:
    """Structured catalog for display or experiment metadata."""
    return [
        {
            "code": code,
            "name": name,
            "category": phase,
            "signal": signal,
            "desc": RISK_PATTERN_DESC.get(code, ""),
        }
        for code, name, phase, signal, _ in RISK_PATTERNS
    ]


def match_risk_patterns(
    code: str,
    bars: List[Bar],
    ctx: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Match P17/P19/P22 against a PIT-safe daily-bar window."""
    _ = code  # Kept for parity with stock_hot_money_radar.match_patterns().
    if len(bars) < MIN_BARS:
        return []
    ctx = ctx or build_risk_context(bars)
    fired: List[Dict[str, str]] = []
    for pcode, name, phase, signal, fn in RISK_PATTERNS:
        try:
            if fn(bars, ctx):
                fired.append({"code": pcode, "name": name, "phase": phase, "signal": signal})
        except Exception:
            continue
    return fired


__all__ = [
    "BACKTEST_HORIZONS",
    "RISK_PATTERNS",
    "RISK_PATTERN_DESC",
    "build_risk_context",
    "match_risk_patterns",
    "risk_pattern_catalog",
]
