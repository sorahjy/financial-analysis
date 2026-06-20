"""Hot-money pattern matcher for A-share theme candidates.

This script turns the playbook in stock_hot_money_pattern_playbook.md into a
first runnable scorer.  It reads only local project data:

  - data/stock_data.sqlite3: leader universe, stock OHLCV, turnover, market cap
  - data/plate_data.sqlite3: SW2 industry daily series
  - data/capital/theme_candidates.json: latest stock/theme mapping cache, if any

The output is not a trading signal by itself.  It is a structured diagnosis of
where each candidate sits in the "ambush -> ignite -> relay -> distribution"
state machine, with evidence fields that can later be verified historically.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text

from stock_hot_money_radar import _limit_pct


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CAPITAL_DIR = DATA_DIR / "capital"
STOCK_DB = DATA_DIR / "stock_data.sqlite3"
PLATE_DB = DATA_DIR / "plate_data.sqlite3"
THEME_CANDIDATES_JSON = CAPITAL_DIR / "theme_candidates.json"
OUT_JSON = CAPITAL_DIR / "hot_money_patterns.json"
OUT_CSV = CAPITAL_DIR / "hot_money_patterns.csv"

SCHEMA = "hot_money_patterns.v1"
PLATE_TYPE = "sw2"
DEFAULT_LOOKBACK_DAYS = 920
DEFAULT_MAX_CAP_YI = 300.0
DEFAULT_MIN_BARS = 160
DEFAULT_PRINT_TOP = 40


def safe_float(value: Any) -> Optional[float]:
    if value in (None, "", "-", "nan", "None"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return float(number)


def round_or_none(value: Any, digits: int = 4) -> Optional[float]:
    number = safe_float(value)
    return round(number, digits) if number is not None else None


def clamp(value: Optional[float], low: float = 0.0, high: float = 100.0) -> float:
    if value is None:
        return 0.0
    return max(low, min(high, float(value)))


def score_range(value: Optional[float], bad_low: float, good_low: float,
                good_high: float, bad_high: float) -> float:
    """0 outside bad bounds, 100 inside the good band, linear shoulders."""
    value = safe_float(value)
    if value is None:
        return 0.0
    if value <= bad_low or value >= bad_high:
        return 0.0
    if good_low <= value <= good_high:
        return 100.0
    if value < good_low:
        return clamp((value - bad_low) / (good_low - bad_low) * 100.0)
    return clamp((bad_high - value) / (bad_high - good_high) * 100.0)


def score_low_position(pctile: Optional[float], full: float = 0.25, zero: float = 0.78) -> float:
    pctile = safe_float(pctile)
    if pctile is None:
        return 0.0
    if pctile <= full:
        return 100.0
    if pctile >= zero:
        return 0.0
    return clamp((zero - pctile) / (zero - full) * 100.0)


def score_high_position(pctile: Optional[float], start: float = 0.70, full: float = 0.93) -> float:
    pctile = safe_float(pctile)
    if pctile is None:
        return 0.0
    return clamp((pctile - start) / (full - start) * 100.0)


def pctile_of_last(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 5:
        return None
    last = values.iloc[-1]
    return float((values <= last).mean())


def mean_last(series: pd.Series, n: int) -> Optional[float]:
    values = pd.to_numeric(series.tail(n), errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def pct_change_last(series: pd.Series, n: int) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= n:
        return None
    base = values.iloc[-1 - n]
    if not base or base <= 0:
        return None
    return float((values.iloc[-1] / base - 1.0) * 100.0)


def normalize_code(code: Any) -> str:
    text = str(code or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6) if digits else text


def market_cap_score(cap_yi: Optional[float]) -> float:
    cap_yi = safe_float(cap_yi)
    if cap_yi is None or cap_yi <= 0:
        return 45.0
    if 30 <= cap_yi <= 300:
        return 100.0
    if 15 <= cap_yi < 30:
        return 80.0
    if 300 < cap_yi <= 500:
        return 75.0
    if 500 < cap_yi <= 800:
        return 50.0
    if cap_yi < 15:
        return 45.0
    return 25.0


def liquidity_score(amount_ma20_yi: Optional[float]) -> float:
    amount_ma20_yi = safe_float(amount_ma20_yi)
    if amount_ma20_yi is None:
        return 40.0
    if 1.0 <= amount_ma20_yi <= 15.0:
        return 100.0
    if 0.3 <= amount_ma20_yi < 1.0:
        return 45.0 + (amount_ma20_yi - 0.3) / 0.7 * 55.0
    if 15.0 < amount_ma20_yi <= 35.0:
        return 100.0 - (amount_ma20_yi - 15.0) / 20.0 * 25.0
    if amount_ma20_yi < 0.3:
        return 25.0
    return 55.0


def load_candidates(max_cap_yi: Optional[float]) -> pd.DataFrame:
    query = """
        WITH latest_cap AS (
            SELECT code, market_cap
            FROM (
                SELECT code, market_cap,
                       ROW_NUMBER() OVER (PARTITION BY code ORDER BY date DESC) AS rn
                FROM stock_history
                WHERE market_cap IS NOT NULL
            )
            WHERE rn = 1
        )
        SELECT
            m.code,
            m.name,
            m.segment_code,
            COALESCE(m.market_cap_yi, latest_cap.market_cap) AS market_cap_yi,
            m.official_market_cap_ratio,
            s.parent_segment AS static_sw2,
            s.segment_name AS static_sw3
        FROM sw3_member m
        LEFT JOIN sw3_segment s ON s.segment_code = m.segment_code
        LEFT JOIN latest_cap ON latest_cap.code = m.code
        WHERE m.is_leader = 1
        ORDER BY m.code
    """
    with sqlite3.connect(STOCK_DB) as conn:
        df = pd.read_sql_query(query, conn)
    if df.empty:
        return df
    df["code"] = df["code"].map(normalize_code)
    df["market_cap_yi"] = pd.to_numeric(df["market_cap_yi"], errors="coerce")
    if max_cap_yi and max_cap_yi > 0:
        df = df[(df["market_cap_yi"].isna()) | (df["market_cap_yi"] <= max_cap_yi)].copy()
    return df.reset_index(drop=True)


def latest_stock_date(as_of: Optional[str] = None) -> str:
    with sqlite3.connect(STOCK_DB) as conn:
        if as_of:
            row = conn.execute(
                "SELECT MAX(date) FROM stock_history WHERE date <= ? AND daily_close IS NOT NULL",
                (as_of,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT MAX(date) FROM stock_history WHERE daily_close IS NOT NULL"
            ).fetchone()
    if not row or not row[0]:
        raise RuntimeError("stock_history 没有可用日线")
    return str(row[0])


def load_stock_history(codes: Iterable[str], as_of: str, lookback_days: int) -> pd.DataFrame:
    codes = [normalize_code(c) for c in codes]
    if not codes:
        return pd.DataFrame()
    start_date = (pd.to_datetime(as_of) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    placeholders = ",".join("?" for _ in codes)
    query = f"""
        SELECT
            code,
            date AS trade_date,
            daily_open,
            daily_high,
            daily_low,
            daily_close,
            daily_volume,
            daily_amount,
            daily_change_pct,
            daily_turnover_rate,
            market_cap
        FROM stock_history
        WHERE code IN ({placeholders})
          AND date >= ?
          AND date <= ?
          AND daily_close IS NOT NULL
        ORDER BY code, date
    """
    with sqlite3.connect(STOCK_DB) as conn:
        df = pd.read_sql_query(query, conn, params=[*codes, start_date, as_of])
    if df.empty:
        return df
    df["code"] = df["code"].map(normalize_code)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for col in [
        "daily_open", "daily_high", "daily_low", "daily_close", "daily_volume",
        "daily_amount", "daily_change_pct", "daily_turnover_rate", "market_cap",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_theme_cache(use_cache: bool) -> Dict[str, Dict[str, Any]]:
    if not use_cache or not THEME_CANDIDATES_JSON.exists():
        return {}
    with THEME_CANDIDATES_JSON.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    rows = payload.get("stock_themes") or []
    if not rows:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        code = normalize_code(row.get("code"))
        if code:
            out[code] = dict(row)
    return out


def pct_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average").fillna(0.0) * 100.0


def load_plate_metrics(as_of: str, lookback_days: int) -> Dict[str, Dict[str, Any]]:
    if not PLATE_DB.exists():
        return {}
    start_date = (pd.to_datetime(as_of) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    query = """
        SELECT
            plate_code,
            plate_name,
            trade_date,
            close_index,
            change_pct,
            turnover_rate,
            amount_share_pct
        FROM plate_daily
        WHERE plate_type = ?
          AND trade_date >= ?
          AND trade_date <= ?
          AND close_index IS NOT NULL
        ORDER BY plate_code, trade_date
    """
    with sqlite3.connect(PLATE_DB) as conn:
        df = pd.read_sql_query(query, conn, params=(PLATE_TYPE, start_date, as_of))
    if df.empty:
        return {}
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for col in ["close_index", "change_pct", "turnover_rate", "amount_share_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    frames = []
    for _, group in df.groupby("plate_code"):
        g = group.sort_values("trade_date").copy()
        g["plate_ret_5d"] = g["close_index"].pct_change(5) * 100.0
        g["plate_ret_20d"] = g["close_index"].pct_change(20) * 100.0
        g["plate_ret_60d"] = g["close_index"].pct_change(60) * 100.0
        g["plate_turn_ma20"] = g["turnover_rate"].rolling(20, min_periods=5).mean()
        g["plate_turn_ma120"] = g["turnover_rate"].rolling(120, min_periods=20).mean()
        g["plate_amount_ma20"] = g["amount_share_pct"].rolling(20, min_periods=5).mean()
        frames.append(g)
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if full.empty:
        return {}

    latest_date = full["trade_date"].max()
    latest = full[full["trade_date"] == latest_date].copy()
    latest["ret20_rank"] = pct_rank(latest["plate_ret_20d"])
    latest["ret60_rank"] = pct_rank(latest["plate_ret_60d"])
    latest["turn_rank"] = pct_rank(latest["plate_turn_ma20"])
    latest["amount_rank"] = pct_rank(latest["plate_amount_ma20"])
    accel = latest["plate_turn_ma20"] / latest["plate_turn_ma120"].replace(0, pd.NA)
    latest["turn_accel"] = accel.astype("float64")
    latest["turn_accel_rank"] = pct_rank(latest["turn_accel"])
    latest["theme_hot_score"] = (
        latest["ret20_rank"] * 0.25
        + latest["ret60_rank"] * 0.20
        + latest["turn_rank"] * 0.25
        + latest["amount_rank"] * 0.20
        + latest["turn_accel_rank"] * 0.10
    )
    latest = latest.sort_values("theme_hot_score", ascending=False).reset_index(drop=True)
    latest["theme_rank"] = latest.index + 1

    out: Dict[str, Dict[str, Any]] = {}
    for _, row in latest.iterrows():
        item = {
            "plate_code": row["plate_code"],
            "plate_name": row["plate_name"],
            "theme_rank": int(row["theme_rank"]),
            "theme_hot_score": round_or_none(row["theme_hot_score"], 4),
            "plate_ret_5d": round_or_none(row["plate_ret_5d"], 4),
            "plate_ret_20d": round_or_none(row["plate_ret_20d"], 4),
            "plate_ret_60d": round_or_none(row["plate_ret_60d"], 4),
            "plate_turn_ma20": round_or_none(row["plate_turn_ma20"], 4),
            "plate_turn_accel": round_or_none(row["turn_accel"], 4),
            "latest_plate_change": round_or_none(row["change_pct"], 4),
            "latest_amount_share": round_or_none(row["amount_share_pct"], 4),
            "plate_latest_date": row["trade_date"].strftime("%Y-%m-%d"),
        }
        out[str(row["plate_name"])] = item
    return out


def cmf_score(group: pd.DataFrame, window: int = 20) -> Tuple[Optional[float], Optional[float]]:
    num = 0.0
    den = 0.0
    for _, row in group.tail(window).iterrows():
        high = safe_float(row["daily_high"])
        low = safe_float(row["daily_low"])
        close = safe_float(row["daily_close"])
        weight = safe_float(row["daily_amount"]) or safe_float(row["daily_volume"])
        if high is None or low is None or close is None or high <= low or not weight:
            continue
        mfm = ((close - low) - (high - close)) / (high - low)
        num += mfm * weight
        den += weight
    if den <= 0:
        return None, None
    cmf = num / den
    return clamp(cmf / 0.20 * 100.0), cmf


def _avg_trade_price(row: pd.Series) -> Optional[float]:
    high = safe_float(row["daily_high"])
    low = safe_float(row["daily_low"])
    close = safe_float(row["daily_close"])
    amount = safe_float(row["daily_amount"])
    volume = safe_float(row["daily_volume"])
    if amount and volume and high is not None and low is not None:
        for avg in (amount / (volume * 100.0), amount / volume):
            if low <= avg <= high:
                return avg
    values = [v for v in (high, low, close) if v is not None]
    return sum(values) / len(values) if values else None


def _triangular_weights(prices: np.ndarray, low: float, high: float, peak: float) -> np.ndarray:
    if len(prices) == 1 or high <= low:
        return np.ones(len(prices))
    peak = min(high, max(low, peak))
    if peak <= low:
        weights = (high - prices) / (high - low)
    elif peak >= high:
        weights = (prices - low) / (high - low)
    else:
        weights = np.zeros(len(prices))
        left = prices <= peak
        weights[left] = (prices[left] - low) / (peak - low)
        weights[~left] = (high - prices[~left]) / (high - peak)
    weights = np.maximum(weights, 0.0)
    if weights.sum() <= 0:
        weights[np.argmin(np.abs(prices - peak))] = 1.0
    return weights


def chip_score(group: pd.DataFrame, window: int = 90) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    rows = []
    for _, row in group.tail(window).iterrows():
        low = safe_float(row["daily_low"])
        high = safe_float(row["daily_high"])
        turnover = safe_float(row["daily_turnover_rate"])
        peak = _avg_trade_price(row)
        if low is not None and high is not None and high > low and turnover is not None and peak is not None:
            rows.append((low, high, peak, turnover))
    if len(rows) < 30:
        return None, {}

    pmin = min(row[0] for row in rows)
    pmax = max(row[1] for row in rows)
    if pmax <= pmin:
        return None, {}

    bucket_count = 80
    grid = np.linspace(pmin, pmax, bucket_count)
    chips = np.zeros(bucket_count)
    span = pmax - pmin
    for low, high, peak, turnover in rows:
        frac = min(1.0, max(0.0, turnover / 100.0))
        chips *= 1.0 - frac
        i0 = max(0, int((low - pmin) / span * (bucket_count - 1)))
        i1 = min(bucket_count - 1, max(i0, int((high - pmin) / span * (bucket_count - 1))))
        weights = _triangular_weights(grid[i0:i1 + 1], low, high, peak)
        chips[i0:i1 + 1] += frac * weights / weights.sum()

    total = chips.sum()
    if total <= 0:
        return None, {}
    chips /= total

    close = safe_float(group.iloc[-1]["daily_close"])
    peak_price = float(grid[int(chips.argmax())])
    concentration = float(chips[np.abs(grid - peak_price) <= 0.07 * peak_price].sum())
    winner = float(chips[grid <= close].sum()) if close else 0.0
    peak_pctile = (peak_price - pmin) / span
    price_to_peak = close / peak_price - 1.0 if close and peak_price else None

    conc_score = clamp((concentration - 0.25) / (0.60 - 0.25) * 100.0)
    peak_low_score = clamp((0.70 - peak_pctile) / (0.70 - 0.35) * 100.0)
    if price_to_peak is None or price_to_peak <= -0.08 or price_to_peak >= 0.22:
        price_score = 0.0
    elif price_to_peak <= 0:
        price_score = clamp((price_to_peak + 0.08) / 0.08 * 100.0)
    elif price_to_peak <= 0.08:
        price_score = 100.0
    else:
        price_score = clamp((0.22 - price_to_peak) / (0.22 - 0.08) * 100.0)

    if winner <= 0.15 or winner >= 0.82:
        winner_score = 0.0
    elif winner < 0.25:
        winner_score = clamp((winner - 0.15) / (0.25 - 0.15) * 100.0)
    elif winner <= 0.60:
        winner_score = 100.0
    else:
        winner_score = clamp((0.82 - winner) / (0.82 - 0.60) * 100.0)

    score = conc_score * 0.45 + peak_low_score * 0.25 + price_score * 0.20 + winner_score * 0.10
    return score, {
        "chip_concentration": round_or_none(concentration, 4),
        "chip_winner": round_or_none(winner, 4),
        "chip_peak_pctile": round_or_none(peak_pctile, 4),
        "chip_price_to_peak": round_or_none(price_to_peak, 4),
    }


def latest_kline(group: pd.DataFrame, code: str) -> Dict[str, Any]:
    row = group.iloc[-1]
    prev_close = safe_float(group["daily_close"].iloc[-2]) if len(group) >= 2 else None
    open_ = safe_float(row["daily_open"])
    high = safe_float(row["daily_high"])
    low = safe_float(row["daily_low"])
    close = safe_float(row["daily_close"])
    chg = safe_float(row["daily_change_pct"])
    if chg is None and close is not None and prev_close:
        chg = (close / prev_close - 1.0) * 100.0

    day_range = (high - low) if high is not None and low is not None else None
    close_loc = None
    upper_ratio = None
    upper_pct = None
    body_pct = None
    amplitude = None
    if day_range and day_range > 0 and close is not None:
        close_loc = (close - low) / day_range
        if open_ is not None and prev_close:
            upper = high - max(open_, close)
            upper_ratio = upper / day_range
            upper_pct = upper / prev_close * 100.0
            body_pct = abs(close - open_) / prev_close * 100.0
        if prev_close:
            amplitude = day_range / prev_close * 100.0

    limit_pct = _limit_pct(code)
    near_limit = chg is not None and chg >= limit_pct - 0.4
    strong_yang = bool(chg is not None and chg >= min(7.0, limit_pct - 2.0)
                       and (close_loc is None or close_loc >= 0.72))
    return {
        "latest_date": row["trade_date"].strftime("%Y-%m-%d"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "prev_close": prev_close,
        "daily_change_pct": chg,
        "close_loc": close_loc,
        "upper_ratio": upper_ratio,
        "upper_pct": upper_pct,
        "body_pct": body_pct,
        "amplitude": amplitude,
        "near_limit": near_limit,
        "strong_yang": strong_yang,
    }


def stock_feature_pack(code: str, group: pd.DataFrame) -> Dict[str, Any]:
    group = group.sort_values("trade_date").copy()
    close = group["daily_close"]
    high = group["daily_high"]
    low = group["daily_low"]
    amount_yi = group["daily_amount"] / 1e8
    turnover = group["daily_turnover_rate"]

    returns = {
        "ret_5d": pct_change_last(close, 5),
        "ret_10d": pct_change_last(close, 10),
        "ret_20d": pct_change_last(close, 20),
        "ret_60d": pct_change_last(close, 60),
        "ret_120d": pct_change_last(close, 120),
    }

    latest = latest_kline(group, code)
    price_pos_60 = pctile_of_last(close.tail(60))
    price_pos_120 = pctile_of_last(close.tail(120))
    price_pos_250 = pctile_of_last(close.tail(250))
    high_20 = safe_float(high.tail(20).max())
    high_40_prior = safe_float(high.iloc[:-1].tail(40).max()) if len(group) > 41 else None
    high_60 = safe_float(high.tail(60).max())
    low_40 = safe_float(low.tail(40).min())
    low_120 = safe_float(low.tail(120).min())

    amount_ma5 = mean_last(amount_yi, 5)
    amount_ma20 = mean_last(amount_yi, 20)
    amount_ma60 = mean_last(amount_yi, 60)
    amount_ma120 = mean_last(amount_yi, 120)
    latest_amount_yi = safe_float(amount_yi.iloc[-1])
    amount_ratio_latest = (
        latest_amount_yi / amount_ma20 if latest_amount_yi is not None and amount_ma20 and amount_ma20 > 0 else None
    )
    amount_ratio_20_60 = amount_ma20 / amount_ma60 if amount_ma20 and amount_ma60 and amount_ma60 > 0 else None
    amount_ratio_20_120 = amount_ma20 / amount_ma120 if amount_ma20 and amount_ma120 and amount_ma120 > 0 else None
    amount_pctile_250 = pctile_of_last(amount_yi.tail(250))

    turn_ma5 = mean_last(turnover, 5)
    turn_ma20 = mean_last(turnover, 20)
    turn_ma60 = mean_last(turnover, 60)
    turn_ma120 = mean_last(turnover, 120)
    latest_turnover = safe_float(turnover.iloc[-1])
    turn_accel = turn_ma20 / turn_ma120 if turn_ma20 and turn_ma120 and turn_ma120 > 0 else None
    turnover_pctile_250 = pctile_of_last(turnover.tail(250))

    prev_close = close.shift(1)
    amp_series = (high - low) / prev_close.replace(0, np.nan) * 100.0
    amp20 = mean_last(amp_series, 20)
    amp120 = mean_last(amp_series, 120)
    amp_ratio_20_120 = amp20 / amp120 if amp20 and amp120 and amp120 > 0 else None
    box_range_40 = (safe_float(high.tail(40).max()) - safe_float(low.tail(40).min())) / latest["close"] * 100.0 if latest["close"] else None
    box_range_120 = (safe_float(high.tail(120).max()) - safe_float(low.tail(120).min())) / latest["close"] * 100.0 if latest["close"] else None
    box_ratio = box_range_40 / box_range_120 if box_range_40 is not None and box_range_120 and box_range_120 > 0 else None

    prior_big_days = 0
    if len(group) >= 21:
        prior_chg = pd.to_numeric(group["daily_change_pct"].iloc[-21:-1], errors="coerce")
        prior_big_days = int((prior_chg >= min(7.0, _limit_pct(code) - 2.0)).sum())

    breakout_40 = bool(
        latest["close"] is not None
        and high_40_prior is not None
        and latest["close"] > high_40_prior
    )
    high_pierced_40 = bool(
        latest["high"] is not None
        and high_40_prior is not None
        and latest["high"] > high_40_prior
    )
    failed_breakout_today = bool(
        high_pierced_40
        and not breakout_40
        and amount_ratio_latest is not None
        and amount_ratio_latest >= 1.8
    )

    spring_reclaim = False
    if len(group) >= 55 and latest["close"] is not None:
        box_low_prior = safe_float(low.iloc[:-5].tail(40).min())
        recent_low = safe_float(low.tail(5).min())
        if box_low_prior and recent_low:
            spring_reclaim = recent_low < box_low_prior * 0.985 and latest["close"] > box_low_prior

    ma20 = mean_last(close, 20)
    ma60 = mean_last(close, 60)
    close_to_ma20 = latest["close"] / ma20 - 1.0 if latest["close"] and ma20 else None
    close_to_high60 = latest["close"] / high_60 if latest["close"] and high_60 else None
    close_to_low120 = latest["close"] / low_120 - 1.0 if latest["close"] and low_120 else None

    cmf, cmf_raw = cmf_score(group)
    chip, chip_detail = chip_score(group)

    return {
        **latest,
        **returns,
        "bar_count": int(len(group)),
        "price_pos_60": price_pos_60,
        "price_pos_120": price_pos_120,
        "price_pos_250": price_pos_250,
        "high_20": high_20,
        "high_40_prior": high_40_prior,
        "high_60": high_60,
        "low_40": low_40,
        "amount_ma5_yi": amount_ma5,
        "amount_ma20_yi": amount_ma20,
        "amount_ma60_yi": amount_ma60,
        "amount_ma120_yi": amount_ma120,
        "latest_amount_yi": latest_amount_yi,
        "amount_ratio_latest": amount_ratio_latest,
        "amount_ratio_20_60": amount_ratio_20_60,
        "amount_ratio_20_120": amount_ratio_20_120,
        "amount_pctile_250": amount_pctile_250,
        "latest_turnover": latest_turnover,
        "turn_ma5": turn_ma5,
        "turn_ma20": turn_ma20,
        "turn_ma60": turn_ma60,
        "turn_ma120": turn_ma120,
        "turn_accel": turn_accel,
        "turnover_pctile_250": turnover_pctile_250,
        "amp20": amp20,
        "amp120": amp120,
        "amp_ratio_20_120": amp_ratio_20_120,
        "box_range_40": box_range_40,
        "box_range_120": box_range_120,
        "box_ratio_40_120": box_ratio,
        "prior_big_days_20": prior_big_days,
        "breakout_40": breakout_40,
        "failed_breakout_today": failed_breakout_today,
        "spring_reclaim": spring_reclaim,
        "close_to_ma20": close_to_ma20,
        "close_to_high60": close_to_high60,
        "close_to_low120": close_to_low120,
        "cmf_score": cmf,
        "cmf": cmf_raw,
        "chip_score": chip,
        **chip_detail,
    }


def theme_context(candidate: Mapping[str, Any], theme_cache: Mapping[str, Dict[str, Any]],
                  plate_metrics: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    code = normalize_code(candidate.get("code"))
    cached = dict(theme_cache.get(code, {}))
    static_sw2 = str(candidate.get("static_sw2") or cached.get("static_sw2") or "")
    trading_theme = str(cached.get("tracking_theme") or cached.get("trading_theme") or static_sw2 or "")
    plate = plate_metrics.get(trading_theme) or plate_metrics.get(static_sw2) or {}
    theme_hot = safe_float(cached.get("theme_hot_score"))
    if theme_hot is None:
        theme_hot = safe_float(plate.get("theme_hot_score"))

    ret_corr = safe_float(cached.get("ret_corr"))
    if ret_corr is None:
        ret_corr = safe_float(cached.get("return_corr"))
    turn_corr = safe_float(cached.get("turnover_corr"))
    ret_corr_score = clamp(((ret_corr or 0.0) + 0.05) / 0.70 * 100.0) if ret_corr is not None else 50.0
    turn_corr_score = clamp(((turn_corr or 0.0) + 0.05) / 0.70 * 100.0) if turn_corr is not None else 50.0
    theme_base = theme_hot if theme_hot is not None else 40.0
    theme_fit = clamp(theme_base * 0.70 + ret_corr_score * 0.18 + turn_corr_score * 0.12)

    return {
        "trading_theme": trading_theme,
        "trading_theme_code": cached.get("tracking_theme_code") or cached.get("trading_theme_code") or plate.get("plate_code"),
        "static_sw2": static_sw2,
        "static_sw3": candidate.get("static_sw3") or cached.get("static_sw3") or "",
        "theme_hot_score": round_or_none(theme_hot, 4),
        "theme_rank": plate.get("theme_rank"),
        "theme_fit_score": round(theme_fit, 2),
        "ret_corr": round_or_none(ret_corr if ret_corr is not None else cached.get("return_corr"), 4),
        "turnover_corr": round_or_none(turn_corr, 4),
        "relative_20d": round_or_none(cached.get("relative_20d"), 4),
        "relative_60d": round_or_none(cached.get("relative_60d"), 4),
        "plate_ret_5d": plate.get("plate_ret_5d"),
        "plate_ret_20d": plate.get("plate_ret_20d"),
        "plate_ret_60d": plate.get("plate_ret_60d"),
        "latest_plate_change": plate.get("latest_plate_change"),
    }


def compression_score(features: Mapping[str, Any]) -> float:
    amp_ratio = safe_float(features.get("amp_ratio_20_120"))
    box_ratio = safe_float(features.get("box_ratio_40_120"))
    amp_score = clamp((1.10 - (amp_ratio or 1.10)) / 0.60 * 100.0)
    box_score = clamp((0.78 - (box_ratio or 0.78)) / 0.45 * 100.0)
    absolute_box = score_range(features.get("box_range_40"), 8.0, 12.0, 32.0, 60.0)
    return clamp(amp_score * 0.35 + box_score * 0.35 + absolute_box * 0.30)


def absorption_score(features: Mapping[str, Any]) -> float:
    vol_warm = score_range(features.get("amount_ratio_20_60"), 0.75, 1.03, 1.85, 2.60)
    turn_warm = score_range(features.get("turn_accel"), 0.75, 0.98, 1.75, 2.60)
    stable = score_range(features.get("ret_20d"), -18.0, -5.0, 16.0, 28.0)
    spring_bonus = 16.0 if features.get("spring_reclaim") else 0.0
    return clamp(vol_warm * 0.40 + turn_warm * 0.25 + stable * 0.35 + spring_bonus)


def runup_score(features: Mapping[str, Any]) -> float:
    ret20 = safe_float(features.get("ret_20d")) or 0.0
    ret60 = safe_float(features.get("ret_60d")) or 0.0
    score20 = clamp((ret20 - 20.0) / 45.0 * 100.0)
    score60 = clamp((ret60 - 35.0) / 85.0 * 100.0)
    return clamp(score20 * 0.55 + score60 * 0.45)


def score_distribution(features: Mapping[str, Any], theme: Mapping[str, Any]) -> Tuple[float, List[str]]:
    evidence: List[str] = []
    high_pos = score_high_position(features.get("price_pos_120"))
    runup = runup_score(features)
    amount_hot = clamp(((safe_float(features.get("amount_pctile_250")) or 0.0) - 0.75) / 0.20 * 100.0)
    turn_hot = clamp(((safe_float(features.get("turnover_pctile_250")) or 0.0) - 0.75) / 0.20 * 100.0)
    hot_volume = max(amount_hot, turn_hot)
    upper_ratio = safe_float(features.get("upper_ratio")) or 0.0
    close_loc = safe_float(features.get("close_loc")) or 0.5
    upper = clamp((upper_ratio - 0.25) / 0.35 * 100.0)
    if upper > 30 and amount_hot > 55:
        evidence.append("高量长上影")

    daily_chg = safe_float(features.get("daily_change_pct")) or 0.0
    amount_ratio = safe_float(features.get("amount_ratio_latest")) or 0.0
    stall = 0.0
    if amount_ratio >= 1.8 and daily_chg <= 2.0:
        stall = clamp((amount_ratio - 1.5) / 3.0 * 100.0) * (1.0 if close_loc < 0.65 else 0.65)
        evidence.append("放量滞涨")
    if features.get("failed_breakout_today"):
        stall = max(stall, 85.0)
        evidence.append("放量假突破")

    near_high = clamp(((safe_float(features.get("close_to_high60")) or 0.0) - 0.92) / 0.08 * 100.0)
    theme_retreat = 0.0
    if (safe_float(theme.get("theme_hot_score")) or 0.0) >= 75:
        latest_plate_change = safe_float(theme.get("latest_plate_change"))
        plate_ret5 = safe_float(theme.get("plate_ret_5d"))
        if latest_plate_change is not None and latest_plate_change < -1.0:
            theme_retreat += 45.0
        if plate_ret5 is not None and plate_ret5 < -3.0:
            theme_retreat += 45.0
        if theme_retreat:
            evidence.append("题材高位回落")

    if runup >= 60:
        evidence.append("阶段涨幅过高")
    if hot_volume >= 70:
        evidence.append("成交/换手历史高分位")
    if high_pos >= 70:
        evidence.append("价格接近阶段高位")

    score = (
        runup * 0.24
        + hot_volume * 0.24
        + upper * 0.18
        + stall * 0.16
        + near_high * 0.10
        + clamp(theme_retreat) * 0.08
    )
    return clamp(score), evidence


def score_ambush(features: Mapping[str, Any], theme: Mapping[str, Any],
                 distribution_score: float) -> Tuple[float, List[str]]:
    evidence: List[str] = []
    position = score_low_position(features.get("price_pos_120"), full=0.32, zero=0.80)
    compress = compression_score(features)
    absorb = absorption_score(features)
    cmf = safe_float(features.get("cmf_score")) or 0.0
    chip = safe_float(features.get("chip_score")) or 0.0
    theme_hot = safe_float(theme.get("theme_hot_score")) or 40.0
    theme_early = score_range(theme_hot, 35.0, 55.0, 88.0, 99.5)

    if position >= 70:
        evidence.append("价格处于中低位")
    if compress >= 65:
        evidence.append("箱体/波动压缩")
    if absorb >= 65:
        evidence.append("温和放量但价格横住")
    if chip >= 60:
        evidence.append("低位筹码集中")
    if features.get("spring_reclaim"):
        evidence.append("假跌破后收回")

    raw = (
        position * 0.18
        + compress * 0.18
        + absorb * 0.20
        + cmf * 0.13
        + chip * 0.20
        + theme_early * 0.11
    )
    penalty = 0.0
    if (safe_float(features.get("ret_20d")) or 0.0) > 28:
        penalty += 18.0
        evidence.append("20日涨幅偏高")
    if (safe_float(features.get("turnover_pctile_250")) or 0.0) > 0.92:
        penalty += 12.0
        evidence.append("换手已偏拥挤")
    penalty += distribution_score * 0.22
    return clamp(raw - penalty), evidence


def score_ignite(features: Mapping[str, Any], theme: Mapping[str, Any],
                 distribution_score: float) -> Tuple[float, List[str]]:
    evidence: List[str] = []
    chg = safe_float(features.get("daily_change_pct")) or 0.0
    close_loc = safe_float(features.get("close_loc")) or 0.5
    strong_yang = 100.0 if features.get("strong_yang") else score_range(chg, 2.5, 5.5, 12.0, 23.0)
    strong_yang *= clamp((close_loc - 0.45) / 0.40 * 100.0) / 100.0

    breakout = 100.0 if features.get("breakout_40") else 0.0
    if features.get("breakout_40"):
        evidence.append("放量突破近40日高点")
    if features.get("near_limit"):
        evidence.append("低位首板/接近涨停")
    elif strong_yang >= 65:
        evidence.append("低位强阳点火")

    prior_big_days = int(features.get("prior_big_days_20") or 0)
    first_signal = clamp((3 - prior_big_days) / 3.0 * 100.0)
    volume_quality = score_range(features.get("amount_ratio_latest"), 1.05, 1.55, 3.80, 7.00)
    theme_fit = safe_float(theme.get("theme_fit_score")) or 40.0
    low_enough = score_low_position(features.get("price_pos_120"), full=0.45, zero=0.90)

    raw = (
        strong_yang * 0.22
        + breakout * 0.22
        + volume_quality * 0.16
        + theme_fit * 0.18
        + low_enough * 0.14
        + first_signal * 0.08
    )
    if prior_big_days == 0 and (features.get("near_limit") or strong_yang >= 75):
        evidence.append("近20日首次强K")
    if volume_quality >= 65:
        evidence.append("放量质量较好")

    return clamp(raw - distribution_score * 0.35), evidence


def score_relay(features: Mapping[str, Any], theme: Mapping[str, Any],
                distribution_score: float) -> Tuple[float, List[str]]:
    evidence: List[str] = []
    momentum = score_range(features.get("ret_20d"), 8.0, 18.0, 45.0, 78.0)
    close_to_ma20 = safe_float(features.get("close_to_ma20"))
    support = score_range(close_to_ma20, -0.12, -0.03, 0.09, 0.20)
    close_loc = safe_float(features.get("close_loc")) or 0.5
    turn_pct = safe_float(features.get("turnover_pctile_250")) or 0.0
    amount_ratio = safe_float(features.get("amount_ratio_latest")) or 0.0
    dispute = 0.0
    if turn_pct >= 0.65 and amount_ratio >= 1.2:
        dispute = clamp((turn_pct - 0.60) / 0.30 * 100.0) * clamp((close_loc - 0.45) / 0.35)
    two_wave = 0.0
    ret60 = safe_float(features.get("ret_60d")) or 0.0
    ret20 = safe_float(features.get("ret_20d")) or 0.0
    close_to_high60 = safe_float(features.get("close_to_high60")) or 0.0
    if ret60 > 25 and 0 <= ret20 <= 45 and close_to_high60 >= 0.82:
        two_wave = score_range(close_to_high60, 0.78, 0.86, 1.02, 1.12)

    theme_fit = safe_float(theme.get("theme_fit_score")) or 40.0
    if dispute >= 55:
        evidence.append("高换手分歧后收强")
    if two_wave >= 55:
        evidence.append("二波回踩再起")
    if support >= 70:
        evidence.append("回踩后仍守住短中期成本")

    raw = (
        momentum * 0.22
        + support * 0.18
        + dispute * 0.22
        + two_wave * 0.18
        + theme_fit * 0.20
    )
    return clamp(raw - distribution_score * 0.45), evidence


def final_stage(scores: Mapping[str, float]) -> str:
    if scores["distribution_score"] >= 70:
        return "dump_risk"
    if scores["ignite_score"] >= 68:
        return "ignite"
    if scores["relay_score"] >= 66:
        return "relay"
    if scores["distribution_score"] >= 55:
        return "distribution"
    if scores["ambush_score"] >= 58:
        return "ambush"
    return "observe"


def invalidations_for(stage: str) -> List[str]:
    if stage == "ambush":
        return ["跌破40日箱体下沿且3-5日不收回", "放量突破失败并回到箱体", "题材指数退潮"]
    if stage == "ignite":
        return ["次日低开低走", "跌回首阳实体中位", "突破位无法站稳"]
    if stage == "relay":
        return ["高换手后不能新高", "跌破分歧日低点", "二波不过前高且放量长上影"]
    if stage in {"distribution", "dump_risk"}:
        return ["缩量回踩后重新放量突破前高才可解除风险", "跌破高位平台下沿则风险确认"]
    return ["数据不足或题材映射失败时不做预测"]


def score_one(candidate: Mapping[str, Any], group: pd.DataFrame,
              theme_cache: Mapping[str, Dict[str, Any]],
              plate_metrics: Mapping[str, Dict[str, Any]],
              min_bars: int) -> Optional[Dict[str, Any]]:
    code = normalize_code(candidate.get("code"))
    if len(group) < min_bars:
        return None
    features = stock_feature_pack(code, group)
    theme = theme_context(candidate, theme_cache, plate_metrics)
    if theme.get("plate_ret_20d") is not None and features.get("ret_20d") is not None:
        theme["relative_20d_computed"] = round_or_none(features["ret_20d"] - theme["plate_ret_20d"], 4)
    if theme.get("plate_ret_60d") is not None and features.get("ret_60d") is not None:
        theme["relative_60d_computed"] = round_or_none(features["ret_60d"] - theme["plate_ret_60d"], 4)

    dist, dist_evidence = score_distribution(features, theme)
    ambush, ambush_evidence = score_ambush(features, theme, dist)
    ignite, ignite_evidence = score_ignite(features, theme, dist)
    relay, relay_evidence = score_relay(features, theme, dist)

    cap = safe_float(candidate.get("market_cap_yi"))
    cap_score = market_cap_score(cap)
    liq_score = liquidity_score(features.get("amount_ma20_yi"))
    data_score = clamp((len(group) - min_bars) / 80.0 * 100.0)
    universe_score = clamp(
        cap_score * 0.35
        + liq_score * 0.25
        + (safe_float(theme.get("theme_fit_score")) or 40.0) * 0.30
        + data_score * 0.10
    )
    scores = {
        "universe_score": universe_score,
        "ambush_score": ambush,
        "ignite_score": ignite,
        "relay_score": relay,
        "distribution_score": dist,
    }
    stage = final_stage(scores)
    opportunity = clamp(
        universe_score * 0.18
        + ambush * 0.40
        + ignite * 0.24
        + relay * 0.18
        - dist * 0.42
    )
    if stage == "dump_risk":
        opportunity = min(opportunity, 35.0)

    evidence = []
    if ambush >= 50:
        evidence.extend(ambush_evidence)
    if ignite >= 50:
        evidence.extend(ignite_evidence)
    if relay >= 50:
        evidence.extend(relay_evidence)
    if dist >= 45:
        evidence.extend(dist_evidence)
    evidence = list(dict.fromkeys(evidence))[:10]

    result = {
        "code": code,
        "name": candidate.get("name") or "",
        "latest_date": features["latest_date"],
        "pattern_stage": stage,
        "opportunity_score": round(opportunity, 2),
        "universe_score": round(universe_score, 2),
        "ambush_score": round(ambush, 2),
        "ignite_score": round(ignite, 2),
        "relay_score": round(relay, 2),
        "distribution_score": round(dist, 2),
        "market_cap_yi": round_or_none(cap, 4),
        "cap_score": round(cap_score, 2),
        "liquidity_score": round(liq_score, 2),
        "trading_theme": theme.get("trading_theme"),
        "trading_theme_code": theme.get("trading_theme_code"),
        "static_sw2": theme.get("static_sw2"),
        "static_sw3": theme.get("static_sw3"),
        "theme_hot_score": theme.get("theme_hot_score"),
        "theme_rank": theme.get("theme_rank"),
        "theme_fit_score": theme.get("theme_fit_score"),
        "ret_corr": theme.get("ret_corr"),
        "turnover_corr": theme.get("turnover_corr"),
        "stock_ret_5d": round_or_none(features.get("ret_5d"), 4),
        "stock_ret_20d": round_or_none(features.get("ret_20d"), 4),
        "stock_ret_60d": round_or_none(features.get("ret_60d"), 4),
        "plate_ret_20d": theme.get("plate_ret_20d"),
        "relative_20d": theme.get("relative_20d") if theme.get("relative_20d") is not None else theme.get("relative_20d_computed"),
        "relative_60d": theme.get("relative_60d") if theme.get("relative_60d") is not None else theme.get("relative_60d_computed"),
        "price_pos_120": round_or_none(features.get("price_pos_120"), 4),
        "close_to_high60": round_or_none(features.get("close_to_high60"), 4),
        "latest_turnover": round_or_none(features.get("latest_turnover"), 4),
        "turnover_pctile_250": round_or_none(features.get("turnover_pctile_250"), 4),
        "latest_amount_yi": round_or_none(features.get("latest_amount_yi"), 4),
        "amount_ma20_yi": round_or_none(features.get("amount_ma20_yi"), 4),
        "amount_ratio_latest": round_or_none(features.get("amount_ratio_latest"), 4),
        "amount_pctile_250": round_or_none(features.get("amount_pctile_250"), 4),
        "box_range_40": round_or_none(features.get("box_range_40"), 4),
        "amp_ratio_20_120": round_or_none(features.get("amp_ratio_20_120"), 4),
        "cmf": round_or_none(features.get("cmf"), 4),
        "cmf_score": round_or_none(features.get("cmf_score"), 4),
        "chip_score": round_or_none(features.get("chip_score"), 4),
        "chip_concentration": features.get("chip_concentration"),
        "chip_winner": features.get("chip_winner"),
        "chip_peak_pctile": features.get("chip_peak_pctile"),
        "chip_price_to_peak": features.get("chip_price_to_peak"),
        "breakout_40": bool(features.get("breakout_40")),
        "spring_reclaim": bool(features.get("spring_reclaim")),
        "failed_breakout_today": bool(features.get("failed_breakout_today")),
        "evidence": evidence,
        "invalidations": invalidations_for(stage),
    }
    return result


def build_patterns(
    *,
    as_of: Optional[str] = None,
    max_cap_yi: Optional[float] = DEFAULT_MAX_CAP_YI,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_bars: int = DEFAULT_MIN_BARS,
    use_theme_cache: bool = True,
) -> Dict[str, Any]:
    as_of_date = latest_stock_date(as_of)
    candidates = load_candidates(max_cap_yi)
    if candidates.empty:
        raise RuntimeError("sw3_member.is_leader 候选池为空")

    history = load_stock_history(candidates["code"].tolist(), as_of_date, lookback_days)
    if history.empty:
        raise RuntimeError("stock_history 没有候选股可用日线")

    # The theme candidate cache is a latest-date artifact.  Do not use it for a
    # historical as_of run, otherwise pattern verification would leak the future.
    cache_allowed = use_theme_cache and as_of is None
    theme_cache = load_theme_cache(cache_allowed)
    plate_metrics = load_plate_metrics(as_of_date, lookback_days)

    candidate_by_code = candidates.set_index("code").to_dict("index")
    rows: List[Dict[str, Any]] = []
    skipped = 0
    for code, group in history.groupby("code", sort=False):
        code = normalize_code(code)
        candidate = candidate_by_code.get(code)
        if not candidate:
            continue
        candidate = dict(candidate)
        candidate["code"] = code
        row = score_one(candidate, group, theme_cache, plate_metrics, min_bars)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    rows.sort(key=lambda item: item["opportunity_score"], reverse=True)
    stage_counts = dict(Counter(row["pattern_stage"] for row in rows))
    payload = {
        "schema": SCHEMA,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of": as_of_date,
        "source": {
            "stock_db": str(STOCK_DB.relative_to(ROOT)),
            "plate_db": str(PLATE_DB.relative_to(ROOT)),
            "theme_cache": str(THEME_CANDIDATES_JSON.relative_to(ROOT)) if cache_allowed and THEME_CANDIDATES_JSON.exists() else None,
            "candidate_pool": "sw3_member.is_leader=1",
        },
        "params": {
            "lookback_days": lookback_days,
            "min_bars": min_bars,
            "max_cap_yi": max_cap_yi,
            "use_theme_cache": cache_allowed,
        },
        "candidate_count": int(len(candidates)),
        "scored_count": int(len(rows)),
        "skipped_count": int(skipped),
        "stage_counts": stage_counts,
        "stocks": rows,
    }
    return payload


def write_outputs(payload: Mapping[str, Any]) -> None:
    CAPITAL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_json = OUT_JSON.with_suffix(".tmp")
    tmp_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_json.replace(OUT_JSON)

    df = pd.DataFrame(payload["stocks"])
    if not df.empty:
        df = df.copy()
        df["evidence"] = df["evidence"].map(lambda items: "；".join(items or []))
        df["invalidations"] = df["invalidations"].map(lambda items: "；".join(items or []))
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")


def fmt(value: Any, width: int = 6, digits: int = 1) -> str:
    value = safe_float(value)
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


STAGE_LABELS = {
    "ambush": "潜伏",
    "ignite": "点火",
    "relay": "接力",
    "observe": "观察",
    "distribution": "出货",
    "dump_risk": "砸盘风险",
}


EVIDENCE_LABELS = {
    "价格处于中低位": "中低位",
    "箱体/波动压缩": "波动压缩",
    "温和放量但价格横住": "放量横住",
    "低位筹码集中": "筹码",
    "假跌破后收回": "假跌破收回",
    "20日涨幅偏高": "20日涨高",
    "换手已偏拥挤": "换手拥挤",
    "放量突破近40日高点": "突破40日",
    "低位首板/接近涨停": "首板涨停",
    "低位强阳点火": "强阳",
    "近20日首次强K": "20日首K",
    "放量质量较好": "放量好",
    "高换手分歧后收强": "分歧收强",
    "二波回踩再起": "二波再起",
    "回踩后仍守住短中期成本": "回踩守成本",
    "高量长上影": "高量上影",
    "阶段涨幅过高": "涨幅高",
    "成交/换手历史高分位": "量换高位",
    "价格接近阶段高位": "近高位",
    "放量滞涨": "放量滞涨",
    "放量假突破": "假突破",
    "题材高位回落": "题材回落",
}


def _console(file: Any = None, width: Optional[int] = None) -> Console:
    width = width or min(max(shutil.get_terminal_size((180, 24)).columns, 120), 220)
    return Console(file=file, width=width, color_system=None, highlight=False, markup=False)


def _cell(value: Any) -> Text:
    return Text(str(value if value is not None else ""))


def _plain_table(columns: List[Tuple[str, str, Optional[int]]]) -> Table:
    table = Table(box=None, show_header=True, pad_edge=False, collapse_padding=True)
    for header, justify, max_width in columns:
        kwargs = {"max_width": max_width} if max_width else {}
        table.add_column(header, justify=justify, no_wrap=True, overflow="ellipsis", **kwargs)
    return table


def _brief_evidence(items: Any) -> str:
    evidence = [EVIDENCE_LABELS.get(str(item), str(item)) for item in (items or []) if item]
    if not evidence:
        return ""
    return "；".join(evidence)


def _display_theme(value: Any) -> str:
    text = str(value or "")
    for suffix in ("Ⅱ", "Ⅲ", "II", "III"):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break
    return text


def build_pattern_table(rows: List[Mapping[str, Any]], top: int) -> Table:
    table = _plain_table([
        ("#", "right", None),
        ("代码", "left", None),
        ("名称", "left", 8),
        ("阶段", "left", 8),
        ("机会", "right", None),
        ("潜伏", "right", None),
        ("点火", "right", None),
        ("接力", "right", None),
        ("风险", "right", None),
        ("市值", "right", None),
        ("题材", "left", 14),
        ("依据", "left", 80),
    ])
    for i, row in enumerate(rows[:top], 1):
        evidence = _brief_evidence(row.get("evidence"))
        table.add_row(
            _cell(i),
            _cell(row["code"]),
            _cell(row.get("name") or ""),
            _cell(STAGE_LABELS.get(str(row.get("pattern_stage") or ""), row.get("pattern_stage") or "")),
            _cell(fmt(row["opportunity_score"], digits=1)),
            _cell(fmt(row["ambush_score"], digits=0)),
            _cell(fmt(row["ignite_score"], digits=0)),
            _cell(fmt(row["relay_score"], digits=0)),
            _cell(fmt(row["distribution_score"], digits=0)),
            _cell(fmt(row["market_cap_yi"], digits=0)),
            _cell(_display_theme(row.get("trading_theme"))),
            _cell(evidence),
        )
    return table


def print_table(title: str, rows: List[Mapping[str, Any]], top: int) -> None:
    if top <= 0:
        return
    print(title, flush=True)
    _console().print(build_pattern_table(rows, top))


def print_summary(payload: Mapping[str, Any], top: int) -> None:
    print(
        f"[patterns] as_of={payload['as_of']} candidates={payload['candidate_count']} "
        f"scored={payload['scored_count']} skipped={payload['skipped_count']} "
        f"written={OUT_JSON.relative_to(ROOT)} / {OUT_CSV.relative_to(ROOT)}",
        flush=True,
    )
    print(f"[patterns] stage_counts={payload['stage_counts']}", flush=True)
    stocks = list(payload["stocks"])
    print_table("[patterns] 机会分 Top:", stocks, top)
    ambush_rows = sorted(
        [s for s in stocks if s["pattern_stage"] == "ambush"],
        key=lambda item: item["ambush_score"],
        reverse=True,
    )
    print_table("[patterns] ambush 潜伏 Top:", ambush_rows, top)
    risk_rows = sorted(stocks, key=lambda item: item["distribution_score"], reverse=True)
    print_table("[patterns] dump/distribution 风险 Top:", risk_rows, top)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="游资题材股形态匹配器")
    parser.add_argument("--as-of", help="只使用该日期及以前的数据，YYYY-MM-DD；默认使用最新交易日")
    parser.add_argument("--max-cap-yi", type=float, default=DEFAULT_MAX_CAP_YI,
                        help="候选池市值上限，单位亿元；设为 0 表示不过滤")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--min-bars", type=int, default=DEFAULT_MIN_BARS)
    parser.add_argument("--print-top", type=int, default=DEFAULT_PRINT_TOP)
    parser.add_argument("--no-theme-cache", action="store_true",
                        help="不读取 data/capital/theme_candidates.json，只用 DB 现算的 SW2 行业热度")
    return parser


def main() -> Dict[str, Any]:
    args = build_parser().parse_args()
    max_cap = None if args.max_cap_yi == 0 else args.max_cap_yi
    payload = build_patterns(
        as_of=args.as_of,
        max_cap_yi=max_cap,
        lookback_days=args.lookback_days,
        min_bars=args.min_bars,
        use_theme_cache=not args.no_theme_cache,
    )
    write_outputs(payload)
    print_summary(payload, args.print_top)
    return payload


if __name__ == "__main__":
    main()
