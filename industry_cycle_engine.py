"""Industry cycle extraction engine.

This module implements the first usable version of the industry-cycle plan in
``industry_cycle_extractor.py``.  It deliberately starts from local SW2 plate
history so the core signals can run without live network dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLATE_DB = DATA_DIR / "plate_data.sqlite3"
INDUSTRY_CYCLE_DIR = DATA_DIR / "industry_cycle"

CYCLE_POSITION_FILE = INDUSTRY_CYCLE_DIR / "cycle_position_latest.json"
SMART_MONEY_FILE = INDUSTRY_CYCLE_DIR / "smart_money_latest.json"
PROSPERITY_FILE = INDUSTRY_CYCLE_DIR / "industry_prosperity_latest.json"
INDUSTRY_STRENGTH_FILE = INDUSTRY_CYCLE_DIR / "industry_strength_latest.json"
RUN_REPORT_FILE = INDUSTRY_CYCLE_DIR / "industry_cycle_run_report.json"

SCHEMA_VERSION = "industry_cycle_engine.v1"
PLATE_TYPE_SW2 = "sw2"
DEFAULT_MIN_HISTORY_DAYS = 180
DEFAULT_FORECAST_HORIZONS = (20, 60, 120)
PRICE_POSITION_WINDOWS = (252, 756, 1260)


def safe_float(value: Any, digits: int = 6) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def pct_rank(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    return clean.rank(pct=True, method="average").fillna(0.0) * 100.0


def weighted_average(values: Iterable[Tuple[Optional[float], float]]) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for value, weight in values:
        number = safe_float(value)
        if number is None or weight <= 0:
            continue
        numerator += number * weight
        denominator += weight
    if denominator <= 0:
        return None
    return numerator / denominator


def weighted_columns(frame: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    numerator = pd.Series(0.0, index=frame.index)
    denominator = pd.Series(0.0, index=frame.index)
    for column, weight in weights.items():
        if column not in frame.columns or weight <= 0:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        mask = values.notna()
        numerator = numerator.add(values.fillna(0.0) * weight, fill_value=0.0)
        denominator = denominator.add(mask.astype("float64") * weight, fill_value=0.0)
    return numerator / denominator.where(denominator > 0)


def value_percentile(value: Any, series: pd.Series, *, window: int = 1260) -> Optional[float]:
    number = safe_float(value)
    if number is None:
        return None
    sample = pd.to_numeric(series, errors="coerce").dropna().tail(window)
    if sample.empty:
        return None
    minimum = float(sample.min())
    maximum = float(sample.max())
    if math.isclose(minimum, maximum):
        return 50.0
    below = float((sample < number).sum())
    equal = float((sample == number).sum())
    return clamp((below + 0.5 * equal) / len(sample) * 100.0)


def score_from_return(value: Any, *, low: float = -20.0, high: float = 20.0) -> float:
    number = safe_float(value)
    if number is None:
        return 50.0
    return clamp((number - low) / (high - low) * 100.0)


def accel_score(value: Any) -> float:
    number = safe_float(value)
    if number is None:
        return 50.0
    return clamp((number - 0.75) / 0.75 * 100.0)


def infer_cycle_phase(
    position: Optional[float],
    *,
    ret20: Optional[float],
    ret60: Optional[float],
    close: Optional[float],
    ma20: Optional[float],
    ma60: Optional[float],
) -> str:
    if position is None:
        return "unknown"
    ret20 = safe_float(ret20) or 0.0
    ret60 = safe_float(ret60) or 0.0
    close = safe_float(close)
    ma20 = safe_float(ma20)
    ma60 = safe_float(ma60)
    above_ma20 = close is not None and ma20 is not None and close >= ma20
    above_ma60 = close is not None and ma60 is not None and close >= ma60

    if position <= 25:
        if ret20 > 1.5 and ret60 > -8.0 and above_ma20:
            return "recovery"
        return "bottom"
    if position <= 45 and ret20 > 0 and ret60 > 0 and above_ma60:
        return "recovery"
    if position >= 82:
        if ret20 < -2.0 or not above_ma20:
            return "top"
        return "overheated"
    if position >= 68 and ret60 > 10.0:
        return "overheated"
    return "middle"


def infer_forecast_phase(position: Optional[float], expected_change: Optional[float]) -> str:
    if position is None:
        return "unknown"
    expected_change = safe_float(expected_change) or 0.0
    if position <= 25:
        return "recovery" if expected_change > 5.0 else "bottom"
    if position <= 45 and expected_change > 3.0:
        return "recovery"
    if position >= 82:
        return "top" if expected_change < -4.0 else "overheated"
    if position >= 68 and expected_change > 2.0:
        return "overheated"
    return "middle"


@dataclass
class ForecastPoint:
    horizon_days: int
    position: Optional[float]
    phase: str
    expected_change: Optional[float]
    confidence: Optional[float]
    model: str = "trend_mean_reversion_ensemble"


@dataclass
class CycleRecord:
    symbol: str
    name: str
    asset_class: str
    as_of: str
    percentile: Optional[float]
    phase: str
    score: Optional[float]
    source: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SmartMoneyRecord:
    industry: str
    symbol: str
    as_of: str
    active: bool
    score: Optional[float]
    reason: str
    only_valid_near_cycle_bottom: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProsperityRecord:
    industry: str
    symbol: str
    as_of: str
    score: Optional[float]
    trend: str
    source: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrengthRecord:
    industry: str
    symbol: str
    as_of: str
    score: Optional[float]
    quadrant: str
    meta: Dict[str, Any] = field(default_factory=dict)


class CycleForecastModel:
    """Lightweight time-series model for future cycle position.

    The model blends damped trend continuation with autoregressive mean
    reversion.  It has no external dependency and is intentionally conservative:
    confidence falls when the history is short, noisy, or when the trend and
    mean-reversion components disagree.
    """

    def __init__(self, horizons: Sequence[int] = DEFAULT_FORECAST_HORIZONS) -> None:
        self.horizons = tuple(int(h) for h in horizons if int(h) > 0)

    def predict(self, cycle_position: pd.Series) -> List[ForecastPoint]:
        series = pd.to_numeric(cycle_position, errors="coerce").dropna()
        if len(series) < 60:
            return [
                ForecastPoint(
                    horizon_days=horizon,
                    position=None,
                    phase="unknown",
                    expected_change=None,
                    confidence=0.0,
                )
                for horizon in self.horizons
            ]

        current = float(series.iloc[-1])
        smoothed = series.ewm(span=20, min_periods=5, adjust=False).mean()
        slope20 = self._slope(smoothed, 20)
        slope60 = self._slope(smoothed, 60)
        target = float(series.tail(min(len(series), 756)).median())
        phi = self._lag1_autocorr(series.tail(min(len(series), 252)))
        daily_noise = float(series.diff().dropna().tail(min(len(series), 252)).std() or 0.0)
        sample_bonus = clamp((len(series) - 60) / 420 * 25.0, 0.0, 25.0)

        forecasts: List[ForecastPoint] = []
        for horizon in self.horizons:
            short_trend = current + slope20 * horizon * math.exp(-horizon / 45.0)
            long_trend = current + slope60 * horizon * math.exp(-horizon / 120.0)
            reversion = target + (phi ** horizon) * (current - target)
            raw = 0.30 * short_trend + 0.30 * long_trend + 0.40 * reversion
            predicted = clamp(raw)
            expected_change = predicted - current

            disagreement = max(
                abs(predicted - clamp(short_trend)),
                abs(predicted - clamp(long_trend)),
                abs(predicted - clamp(reversion)),
            )
            uncertainty = daily_noise * math.sqrt(horizon)
            confidence = clamp(65.0 + sample_bonus - uncertainty * 1.8 - disagreement * 1.2, 5.0, 92.0)
            forecasts.append(
                ForecastPoint(
                    horizon_days=horizon,
                    position=safe_float(predicted),
                    phase=infer_forecast_phase(predicted, expected_change),
                    expected_change=safe_float(expected_change),
                    confidence=safe_float(confidence),
                )
            )
        return forecasts

    @staticmethod
    def _slope(series: pd.Series, periods: int) -> float:
        if len(series) <= periods:
            return 0.0
        now = safe_float(series.iloc[-1])
        before = safe_float(series.iloc[-periods])
        if now is None or before is None:
            return 0.0
        return (now - before) / periods

    @staticmethod
    def _lag1_autocorr(series: pd.Series) -> float:
        if len(series) < 30:
            return 0.85
        if safe_float(series.std()) in (None, 0.0):
            return 0.85
        value = series.autocorr(lag=1)
        if safe_float(value) is None:
            return 0.85
        return clamp(float(value), 0.45, 0.98)


class IndustryCycleFeatureBuilder:
    def build(self, daily: pd.DataFrame) -> pd.DataFrame:
        if daily.empty:
            return daily.copy()
        frames: List[pd.DataFrame] = []
        for _, group in daily.groupby("plate_code", sort=False):
            frames.append(self._build_one(group))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _build_one(self, group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("trade_date").copy()
        close = pd.to_numeric(g["close_index"], errors="coerce")
        g["ret_5d"] = close.pct_change(5) * 100.0
        g["ret_20d"] = close.pct_change(20) * 100.0
        g["ret_60d"] = close.pct_change(60) * 100.0
        g["ret_120d"] = close.pct_change(120) * 100.0
        g["ret_252d"] = close.pct_change(252) * 100.0

        for window in (20, 60, 120, 250):
            g[f"ma{window}"] = close.rolling(window, min_periods=max(5, window // 4)).mean()
            g[f"ma{window}_deviation"] = (close / g[f"ma{window}"] - 1.0) * 100.0

        g["ma20_slope_20d"] = (g["ma20"] / g["ma20"].shift(20) - 1.0) * 100.0
        daily_ret = close.pct_change()
        g["volatility_60d"] = daily_ret.rolling(60, min_periods=20).std() * math.sqrt(252.0) * 100.0
        g["volatility_252d"] = daily_ret.rolling(252, min_periods=80).std() * math.sqrt(252.0) * 100.0

        for window in (60, *PRICE_POSITION_WINDOWS):
            low = close.rolling(window, min_periods=min(max(60, window // 3), window)).min()
            high = close.rolling(window, min_periods=min(max(60, window // 3), window)).max()
            spread = high - low
            position = (close - low) / spread.where(spread != 0) * 100.0
            g[f"position_{window}d"] = position.fillna(50.0).where(spread.notna(), pd.NA)
            g[f"drawdown_{window}d"] = (close / high.where(high != 0) - 1.0) * 100.0

        g["cycle_position"] = weighted_columns(
            g,
            {
                "position_756d": 0.45,
                "position_252d": 0.35,
                "position_1260d": 0.20,
            },
        )
        g["turnover_ma20"] = pd.to_numeric(g["turnover_rate"], errors="coerce").rolling(20, min_periods=5).mean()
        g["turnover_ma120"] = pd.to_numeric(g["turnover_rate"], errors="coerce").rolling(120, min_periods=20).mean()
        g["turnover_accel"] = g["turnover_ma20"] / g["turnover_ma120"].where(g["turnover_ma120"] != 0)
        g["amount_ma20"] = pd.to_numeric(g["amount_share_pct"], errors="coerce").rolling(20, min_periods=5).mean()
        g["amount_ma120"] = pd.to_numeric(g["amount_share_pct"], errors="coerce").rolling(120, min_periods=20).mean()
        g["amount_accel"] = g["amount_ma20"] / g["amount_ma120"].where(g["amount_ma120"] != 0)
        return g


class IndustryCycleEngine:
    def __init__(
        self,
        *,
        db_file: Path | str = PLATE_DB,
        plate_type: str = PLATE_TYPE_SW2,
        min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
        forecast_horizons: Sequence[int] = DEFAULT_FORECAST_HORIZONS,
    ) -> None:
        self.db_file = Path(db_file)
        self.plate_type = plate_type
        self.min_history_days = int(min_history_days)
        self.feature_builder = IndustryCycleFeatureBuilder()
        self.forecast_model = CycleForecastModel(forecast_horizons)

    def load_local_daily(self, *, as_of: Optional[str] = None) -> pd.DataFrame:
        if not self.db_file.exists():
            raise FileNotFoundError(f"plate DB not found: {self.db_file}")
        query = """
            SELECT
                plate_code,
                plate_name,
                trade_date,
                close_index,
                change_pct,
                turnover_rate,
                pe,
                pb,
                amount_share_pct,
                dividend_yield,
                source
            FROM plate_daily
            WHERE plate_type = ?
              AND close_index IS NOT NULL
        """
        params: List[Any] = [self.plate_type]
        if as_of:
            query += " AND trade_date <= ?"
            params.append(as_of)
        query += " ORDER BY plate_code, trade_date"
        with sqlite3.connect(self.db_file) as conn:
            daily = pd.read_sql_query(query, conn, params=params)
        if daily.empty:
            return daily
        daily["trade_date"] = pd.to_datetime(daily["trade_date"])
        for column in (
            "close_index",
            "change_pct",
            "turnover_rate",
            "pe",
            "pb",
            "amount_share_pct",
            "dividend_yield",
        ):
            daily[column] = pd.to_numeric(daily[column], errors="coerce")
        return daily

    def build_payloads(self, *, as_of: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        daily = self.load_local_daily(as_of=as_of)
        return self.build_payloads_from_frame(daily, as_of=as_of)

    def build_payloads_from_frame(self, daily: pd.DataFrame, *, as_of: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        if daily.empty:
            raise RuntimeError("plate_daily has no usable rows")

        # ``as_of`` is a point-in-time boundary for the whole calculation, not
        # merely for selecting the final row.  Keeping future observations in
        # ``features`` would leak them into valuation percentiles and the
        # forecast series even though ``latest`` itself was historical.
        if as_of:
            requested = pd.to_datetime(as_of)
            trade_dates = pd.to_datetime(daily["trade_date"], errors="coerce")
            daily = daily.loc[trade_dates <= requested].copy()
            if daily.empty:
                raise RuntimeError(f"no data available up to as_of={as_of}")

        features = self.feature_builder.build(daily)
        resolved_as_of = self._resolve_as_of(features, as_of)
        latest = self._latest_by_plate(features, resolved_as_of)
        latest = latest[latest["history_days"] >= self.min_history_days].copy()
        if latest.empty:
            raise RuntimeError(f"no plate has enough history_days >= {self.min_history_days}")

        latest = self._add_cross_section_metrics(latest)
        groups = {str(code): group.sort_values("trade_date") for code, group in features.groupby("plate_code")}

        cycle_records: List[CycleRecord] = []
        smart_records: List[SmartMoneyRecord] = []
        prosperity_records: List[ProsperityRecord] = []
        strength_records: List[StrengthRecord] = []

        for _, row in latest.sort_values("plate_code").iterrows():
            code = str(row["plate_code"])
            group = groups[code]
            cycle = self._build_cycle_record(row, group, resolved_as_of)
            strength = self._build_strength_record(row, resolved_as_of)
            smart = self._build_smart_money_record(row, cycle, resolved_as_of)
            prosperity = self._build_prosperity_record(row, cycle, resolved_as_of)
            cycle_records.append(cycle)
            strength_records.append(strength)
            smart_records.append(smart)
            prosperity_records.append(prosperity)

        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source = {
            "db_file": str(self.db_file.relative_to(ROOT) if self.db_file.is_relative_to(ROOT) else self.db_file),
            "plate_type": self.plate_type,
            "source_table": "plate_daily",
            "mode": "local_cache",
        }
        params = {
            "min_history_days": self.min_history_days,
            "forecast_horizons": list(self.forecast_model.horizons),
            "price_position_windows": list(PRICE_POSITION_WINDOWS),
            "as_of": resolved_as_of,
        }

        cycle_payload = {
            "schema": SCHEMA_VERSION,
            "generated_at": generated_at,
            "as_of": resolved_as_of,
            "source": source,
            "params": params,
            "records": [asdict(record) for record in cycle_records],
        }
        smart_payload = {
            "schema": SCHEMA_VERSION,
            "generated_at": generated_at,
            "as_of": resolved_as_of,
            "source": source,
            "params": params,
            "records": [asdict(record) for record in smart_records],
        }
        prosperity_payload = {
            "schema": SCHEMA_VERSION,
            "generated_at": generated_at,
            "as_of": resolved_as_of,
            "source": {
                **source,
                "model_note": "price_valuation_activity_proxy; not a real industry fundamental nowcast",
            },
            "params": params,
            "records": [asdict(record) for record in prosperity_records],
        }
        strength_payload = {
            "schema": SCHEMA_VERSION,
            "generated_at": generated_at,
            "as_of": resolved_as_of,
            "source": source,
            "params": params,
            "records": [asdict(record) for record in strength_records],
        }
        report_payload = {
            "schema": SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "ok",
            "as_of": resolved_as_of,
            "source": source,
            "params": params,
            "input_rows": int(len(daily)),
            "feature_rows": int(len(features)),
            "plate_count": int(latest["plate_code"].nunique()),
            "latest_trade_date": resolved_as_of,
            "outputs": {
                "cycle_position": str(CYCLE_POSITION_FILE.relative_to(ROOT)),
                "smart_money": str(SMART_MONEY_FILE.relative_to(ROOT)),
                "prosperity": str(PROSPERITY_FILE.relative_to(ROOT)),
                "industry_strength": str(INDUSTRY_STRENGTH_FILE.relative_to(ROOT)),
                "run_report": str(RUN_REPORT_FILE.relative_to(ROOT)),
            },
            "limitations": [
                "First version uses local SW2 plate price, valuation, turnover, and amount-share data.",
                "Smart-money signals are activity proxies, not real net-inflow data.",
                "Prosperity is a price/valuation/activity proxy until real industry fundamental indicators are added.",
            ],
        }
        return {
            "cycle": cycle_payload,
            "smart_money": smart_payload,
            "prosperity": prosperity_payload,
            "strength": strength_payload,
            "report": report_payload,
        }

    @staticmethod
    def _resolve_as_of(features: pd.DataFrame, as_of: Optional[str]) -> str:
        if as_of:
            requested = pd.to_datetime(as_of)
            eligible = features[features["trade_date"] <= requested]
        else:
            eligible = features
        if eligible.empty:
            raise RuntimeError(f"no data available up to as_of={as_of}")
        return eligible["trade_date"].max().strftime("%Y-%m-%d")

    @staticmethod
    def _latest_by_plate(features: pd.DataFrame, as_of: str) -> pd.DataFrame:
        eligible = features[features["trade_date"] <= pd.to_datetime(as_of)].copy()
        eligible["history_days"] = eligible.groupby("plate_code")["trade_date"].transform("count")
        idx = eligible.groupby("plate_code")["trade_date"].idxmax()
        return eligible.loc[idx].copy()

    @staticmethod
    def _add_cross_section_metrics(latest: pd.DataFrame) -> pd.DataFrame:
        result = latest.copy()
        result["ret20_rank"] = pct_rank(result["ret_20d"])
        result["ret60_rank"] = pct_rank(result["ret_60d"])
        result["ret120_rank"] = pct_rank(result["ret_120d"])
        result["turnover_accel_rank"] = pct_rank(result["turnover_accel"])
        result["amount_accel_rank"] = pct_rank(result["amount_accel"])
        result["amount_rank"] = pct_rank(result["amount_ma20"])
        result["strength_score"] = (
            result["ret20_rank"] * 0.25
            + result["ret60_rank"] * 0.25
            + result["ret120_rank"] * 0.20
            + result["turnover_accel_rank"] * 0.15
            + result["amount_accel_rank"] * 0.15
        )
        result["relative_strength_60d"] = pd.to_numeric(result["ret_60d"], errors="coerce") - result["ret_60d"].median()
        result["relative_momentum_20d"] = pd.to_numeric(result["ret_20d"], errors="coerce") - result["ret_20d"].median()
        return result

    def _build_cycle_record(self, row: pd.Series, group: pd.DataFrame, as_of: str) -> CycleRecord:
        position = safe_float(row.get("cycle_position"))
        valuation_percentile = weighted_average(
            (
                (value_percentile(row.get("pe"), group["pe"]), 0.6),
                (value_percentile(row.get("pb"), group["pb"]), 0.4),
            )
        )
        phase = infer_cycle_phase(
            position,
            ret20=safe_float(row.get("ret_20d")),
            ret60=safe_float(row.get("ret_60d")),
            close=safe_float(row.get("close_index")),
            ma20=safe_float(row.get("ma20")),
            ma60=safe_float(row.get("ma60")),
        )
        recovery_score = weighted_average(
            (
                (score_from_return(row.get("ret_20d"), low=-8, high=12), 0.45),
                (score_from_return(row.get("ret_60d"), low=-15, high=18), 0.35),
                (accel_score(row.get("turnover_accel")), 0.20),
            )
        )
        low_position_score = None if position is None else 100.0 - position
        valuation_score = None if valuation_percentile is None else 100.0 - valuation_percentile
        opportunity_score = weighted_average(
            (
                (low_position_score, 0.35),
                (recovery_score, 0.30),
                (valuation_score, 0.20),
                (safe_float(row.get("strength_score")), 0.15),
            )
        )
        forecast = self.forecast_model.predict(group["cycle_position"])
        metrics = self._latest_metrics(row, valuation_percentile)
        explain = self._cycle_explain(position, phase, metrics, forecast)

        return CycleRecord(
            symbol=str(row["plate_code"]),
            name=str(row["plate_name"]),
            asset_class="sw2_industry",
            as_of=as_of,
            percentile=safe_float(position),
            phase=phase,
            score=safe_float(opportunity_score),
            source="data/plate_data.sqlite3:plate_daily",
            meta={
                "metrics": metrics,
                "forecast": [asdict(item) for item in forecast],
                "explain": explain,
                "history_days": int(row.get("history_days") or 0),
                "data_quality": self._data_quality(row),
            },
        )

    @staticmethod
    def _build_strength_record(row: pd.Series, as_of: str) -> StrengthRecord:
        rel_strength = safe_float(row.get("relative_strength_60d")) or 0.0
        rel_momentum = safe_float(row.get("relative_momentum_20d")) or 0.0
        if rel_strength >= 0 and rel_momentum >= 0:
            quadrant = "leading"
        elif rel_strength >= 0 and rel_momentum < 0:
            quadrant = "weakening"
        elif rel_strength < 0 and rel_momentum >= 0:
            quadrant = "improving"
        else:
            quadrant = "lagging"
        return StrengthRecord(
            industry=str(row["plate_name"]),
            symbol=str(row["plate_code"]),
            as_of=as_of,
            score=safe_float(row.get("strength_score")),
            quadrant=quadrant,
            meta={
                "relative_strength_60d": safe_float(row.get("relative_strength_60d")),
                "relative_momentum_20d": safe_float(row.get("relative_momentum_20d")),
                "ret20_rank": safe_float(row.get("ret20_rank")),
                "ret60_rank": safe_float(row.get("ret60_rank")),
                "ret120_rank": safe_float(row.get("ret120_rank")),
                "turnover_accel_rank": safe_float(row.get("turnover_accel_rank")),
                "amount_accel_rank": safe_float(row.get("amount_accel_rank")),
            },
        )

    @staticmethod
    def _build_smart_money_record(row: pd.Series, cycle: CycleRecord, as_of: str) -> SmartMoneyRecord:
        position = safe_float(row.get("cycle_position"))
        near_bottom = cycle.phase in {"bottom", "recovery"} and position is not None and position <= 40.0
        if not near_bottom:
            return SmartMoneyRecord(
                industry=str(row["plate_name"]),
                symbol=str(row["plate_code"]),
                as_of=as_of,
                active=False,
                score=0.0,
                reason="非周期底部或底部修复区，不公布进场动作",
                meta={
                    "cycle_phase": cycle.phase,
                    "cycle_position": position,
                    "gated": True,
                },
            )

        ret20 = safe_float(row.get("ret_20d")) or 0.0
        ret60 = safe_float(row.get("ret_60d")) or 0.0
        quiet_price_score = clamp(100.0 - max(ret20 - 10.0, 0.0) * 5.0 - max(-ret20 - 10.0, 0.0) * 3.0)
        support_score = clamp((safe_float(row.get("position_60d")) or 50.0) * 0.7 + score_from_return(ret60, low=-18, high=8) * 0.3)
        activity_score = weighted_average(
            (
                (safe_float(row.get("turnover_accel_rank")), 0.45),
                (safe_float(row.get("amount_accel_rank")), 0.35),
                (accel_score(row.get("turnover_accel")), 0.10),
                (accel_score(row.get("amount_accel")), 0.10),
            )
        )
        score = weighted_average(
            (
                (activity_score, 0.45),
                (quiet_price_score, 0.25),
                (support_score, 0.20),
                (100.0 - position, 0.10),
            )
        )
        active = bool(
            score is not None
            and score >= 62.0
            and ((safe_float(row.get("turnover_accel")) or 0.0) >= 1.03 or (safe_float(row.get("amount_accel")) or 0.0) >= 1.03)
        )
        reason = "低位区间出现成交/换手活跃度抬升，价格尚未明显透支" if active else "处于低位但活跃度或承接证据不足"
        return SmartMoneyRecord(
            industry=str(row["plate_name"]),
            symbol=str(row["plate_code"]),
            as_of=as_of,
            active=active,
            score=safe_float(score),
            reason=reason,
            meta={
                "cycle_phase": cycle.phase,
                "cycle_position": position,
                "turnover_accel": safe_float(row.get("turnover_accel")),
                "amount_accel": safe_float(row.get("amount_accel")),
                "quiet_price_score": safe_float(quiet_price_score),
                "support_score": safe_float(support_score),
                "activity_score": safe_float(activity_score),
                "proxy_fields": ["turnover_rate", "amount_share_pct", "close_index"],
            },
        )

    @staticmethod
    def _build_prosperity_record(row: pd.Series, cycle: CycleRecord, as_of: str) -> ProsperityRecord:
        forecast_60 = None
        for item in cycle.meta.get("forecast", []):
            if item.get("horizon_days") == 60:
                forecast_60 = item
                break
        forecast_change = safe_float((forecast_60 or {}).get("expected_change")) or 0.0
        trend_score = weighted_average(
            (
                (score_from_return(row.get("ret_60d"), low=-15, high=25), 0.45),
                (score_from_return(row.get("ret_120d"), low=-20, high=35), 0.35),
                (score_from_return(forecast_change, low=-10, high=12), 0.20),
            )
        )
        activity_score = weighted_average(
            (
                (safe_float(row.get("turnover_accel_rank")), 0.45),
                (safe_float(row.get("amount_accel_rank")), 0.45),
                (safe_float(row.get("amount_rank")), 0.10),
            )
        )
        valuation_balance = weighted_average(
            (
                (score_from_return(row.get("dividend_yield"), low=0, high=6), 0.45),
                (100.0 - (safe_float(cycle.meta.get("metrics", {}).get("valuation_percentile")) or 50.0), 0.35),
                (safe_float(row.get("strength_score")), 0.20),
            )
        )
        score = weighted_average(
            (
                (trend_score, 0.45),
                (activity_score, 0.30),
                (valuation_balance, 0.15),
                (safe_float(cycle.score), 0.10),
            )
        )
        if score is None:
            trend = "unknown"
        elif score >= 65 and (safe_float(row.get("ret_60d")) or 0.0) >= 0:
            trend = "rising"
        elif score <= 40 and (safe_float(row.get("ret_60d")) or 0.0) <= 0:
            trend = "falling"
        else:
            trend = "stable"
        return ProsperityRecord(
            industry=str(row["plate_name"]),
            symbol=str(row["plate_code"]),
            as_of=as_of,
            score=safe_float(score),
            trend=trend,
            source="price_valuation_activity_proxy",
            meta={
                "trend_score": safe_float(trend_score),
                "activity_score": safe_float(activity_score),
                "valuation_balance": safe_float(valuation_balance),
                "forecast_60d_expected_change": safe_float(forecast_change),
                "note": "代理景气度，不等同于订单/库存/开工率等真实产业景气指标",
            },
        )

    @staticmethod
    def _latest_metrics(row: pd.Series, valuation_percentile: Optional[float]) -> Dict[str, Any]:
        keys = [
            "close_index",
            "ret_20d",
            "ret_60d",
            "ret_120d",
            "ret_252d",
            "position_60d",
            "position_252d",
            "position_756d",
            "position_1260d",
            "drawdown_252d",
            "ma20_deviation",
            "ma60_deviation",
            "ma120_deviation",
            "volatility_60d",
            "turnover_ma20",
            "turnover_accel",
            "amount_ma20",
            "amount_accel",
            "pe",
            "pb",
            "dividend_yield",
            "strength_score",
        ]
        metrics = {key: safe_float(row.get(key)) for key in keys}
        metrics["valuation_percentile"] = safe_float(valuation_percentile)
        return metrics

    @staticmethod
    def _cycle_explain(
        position: Optional[float],
        phase: str,
        metrics: Mapping[str, Any],
        forecast: Sequence[ForecastPoint],
    ) -> List[str]:
        explain = [f"周期位置={safe_float(position)}，phase={phase}"]
        if metrics.get("position_756d") is not None:
            explain.append(f"3年价格区间位置={metrics['position_756d']}")
        if metrics.get("ret_60d") is not None:
            explain.append(f"60日涨跌幅={metrics['ret_60d']}%")
        if metrics.get("turnover_accel") is not None:
            explain.append(f"换手MA20/MA120={metrics['turnover_accel']}")
        if metrics.get("valuation_percentile") is not None:
            explain.append(f"估值历史分位={metrics['valuation_percentile']}")
        horizon_60 = next((item for item in forecast if item.horizon_days == 60), None)
        if horizon_60 and horizon_60.position is not None:
            explain.append(
                f"时间序列模型60日预测位置={horizon_60.position}，confidence={horizon_60.confidence}"
            )
        return explain

    @staticmethod
    def _data_quality(row: pd.Series) -> Dict[str, Any]:
        required = ["close_index", "ret_60d", "cycle_position", "turnover_accel", "amount_accel"]
        available = sum(1 for key in required if safe_float(row.get(key)) is not None)
        return {
            "required_fields": required,
            "available_required_fields": available,
            "score": safe_float(available / len(required) * 100.0),
        }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_payloads(payloads: Mapping[str, Mapping[str, Any]]) -> None:
    write_json(CYCLE_POSITION_FILE, payloads["cycle"])
    write_json(SMART_MONEY_FILE, payloads["smart_money"])
    write_json(PROSPERITY_FILE, payloads["prosperity"])
    write_json(INDUSTRY_STRENGTH_FILE, payloads["strength"])
    write_json(RUN_REPORT_FILE, payloads["report"])


def parse_horizons(value: str) -> Tuple[int, ...]:
    horizons: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        horizon = int(item)
        if horizon <= 0:
            raise ValueError("forecast horizon must be positive")
        horizons.append(horizon)
    if not horizons:
        raise ValueError("at least one forecast horizon is required")
    return tuple(horizons)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建行业周期、聪明资金、景气度和周期预测数据")
    parser.add_argument("--db-file", default=str(PLATE_DB), help="plate_data.sqlite3 路径")
    parser.add_argument("--as-of", help="历史截面日期 YYYY-MM-DD；默认使用库内最新交易日")
    parser.add_argument("--min-history-days", type=int, default=DEFAULT_MIN_HISTORY_DAYS, help="最少历史样本天数")
    parser.add_argument("--forecast-horizons", default="20,60,120", help="预测窗口，逗号分隔交易日")
    parser.add_argument("--write", action="store_true", help="写出 data/industry_cycle/*.json")
    parser.add_argument("--top", type=int, default=10, help="控制台展示机会分最高的前 N 个行业")
    return parser


def main() -> Dict[str, Dict[str, Any]]:
    args = build_parser().parse_args()
    engine = IndustryCycleEngine(
        db_file=args.db_file,
        min_history_days=args.min_history_days,
        forecast_horizons=parse_horizons(args.forecast_horizons),
    )
    payloads = engine.build_payloads(as_of=args.as_of)
    if args.write:
        write_payloads(payloads)

    records = payloads["cycle"]["records"]
    top_records = sorted(records, key=lambda item: item.get("score") or -1, reverse=True)[: max(args.top, 0)]
    print(
        f"[industry-cycle] as_of={payloads['cycle']['as_of']} plates={len(records)} "
        f"write={bool(args.write)}"
    )
    for index, record in enumerate(top_records, 1):
        forecast_60 = next(
            (item for item in record.get("meta", {}).get("forecast", []) if item.get("horizon_days") == 60),
            {},
        )
        print(
            f"{index:>2}. {record['symbol']} {record['name']} "
            f"phase={record['phase']} pos={record.get('percentile')} "
            f"score={record.get('score')} f60={forecast_60.get('position')}"
        )
    return payloads


if __name__ == "__main__":
    main()
