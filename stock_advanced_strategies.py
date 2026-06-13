
"""
Advanced A-share strategy engine.

This module is intentionally self-contained and read-only against existing
project files. It loads the JSON data already produced by the stock_* scripts,
scores two strategy families, and can be used by both CLI and the dashboard:

1. Long horizon: large-cap CSI300 quality compounders, intended for 2-5 years.
2. Short horizon: non-ST Dragon Tiger List / hot-money tracking, intended for
   1-5 trading days.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
STOCK_DATA_DIR = DATA_DIR / "stock_data"
CN_STOCK_DIR = DATA_DIR / "CN_stock"
CAPITAL_DIR = DATA_DIR / "capital"
OUTPUT_FILE = DATA_DIR / "stock_advanced_strategy_results.json"
OPTIMIZED_CONFIG_FILE = DATA_DIR / "stock_strategy_optimized_config.json"


# ---------------------------------------------------------------------------
# Factor registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorSpec:
    key: str
    label: str
    strategy: str
    group: str
    description: str
    default_weight: float
    score: str = "percentile"  # percentile, direct, boolean, bounded
    direction: str = "high"    # high or low
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    missing_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "strategy": self.strategy,
            "group": self.group,
            "description": self.description,
            "default_weight": self.default_weight,
            "score": self.score,
            "direction": self.direction,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "missing_score": self.missing_score,
        }


LONG_FACTORS: List[FactorSpec] = [
    FactorSpec("csi300_current", "当前沪深300", "long", "指数约束", "现仍在沪深300成分池。", 2.0, "boolean"),
    FactorSpec("csi300_persistence", "沪深300稳定代理", "long", "指数约束", "用当前沪深300、上市年限、中证大盘池近似长期稳定成分。", 1.6, "direct"),
    FactorSpec("market_cap", "总市值", "long", "规模流动性", "偏好大盘股。", 1.2, "percentile"),
    FactorSpec("size_reversal", "小市值因子", "long", "规模流动性", "SMB代理：总市值越小得分越高，给中小盘优质股入场机会，与总市值因子由权重搜索博弈。", 0.5, "percentile", "low", missing_score=40),
    FactorSpec("liquidity", "日均成交额", "long", "规模流动性", "偏好交易容量更高的股票。", 0.6, "percentile"),
    FactorSpec("low_volatility", "低波动", "long", "风险", "年化波动率越低越好。", 0.8, "percentile", "low", missing_score=40),
    FactorSpec("roe_mean", "ROE均值", "long", "质量", "近年ROE均值。", 1.2, "percentile", missing_score=20),
    FactorSpec("roe_stability", "ROE稳定性", "long", "质量", "ROE均值减标准差。", 1.1, "percentile", missing_score=20),
    FactorSpec("revenue_growth", "营收增速", "long", "成长", "TTM营收同比。", 0.8, "percentile", missing_score=35),
    FactorSpec("net_profit_growth", "利润增速", "long", "成长", "TTM净利润同比。", 0.9, "percentile", missing_score=35),
    FactorSpec("gross_margin", "毛利率", "long", "盈利", "销售毛利率。", 0.5, "percentile", missing_score=35),
    FactorSpec("net_margin", "净利率", "long", "盈利", "销售净利率。", 0.8, "percentile", missing_score=35),
    FactorSpec("cashflow_quality", "现金流质量", "long", "质量", "经营现金流 / 净利润，1附近及以上更好。", 0.9, "bounded", "high", 0.0, 2.0, 35),
    FactorSpec("dividend_yield_5y", "五年股息率", "long", "股东回报", "近五年平均每股分红 / 当前股价。", 0.7, "percentile", missing_score=20),
    FactorSpec("dividend_consistency", "连续分红", "long", "股东回报", "近三年连续现金分红。", 0.5, "boolean"),
    FactorSpec("debt_safety", "低负债率", "long", "风险", "资产负债率越低越好，银行等行业会由其他质量因子平衡。", 0.5, "percentile", "low", missing_score=45),
    FactorSpec("pledge_safety", "低质押", "long", "风险", "股权质押比例越低越好。", 0.4, "percentile", "low", missing_score=60),
    FactorSpec("industry_leadership", "行业规模地位", "long", "规模流动性", "行业内市值百分位。", 0.5, "direct", missing_score=40),
    FactorSpec("book_to_market", "账面市值比", "long", "估值", "Fama-French HML 代理，净资产 / 市值。", 0.6, "percentile", missing_score=35),
    FactorSpec("earnings_yield", "盈利收益率", "long", "估值", "E/P，净利润TTM / 市值。", 0.7, "percentile", missing_score=35),
    FactorSpec("cashflow_yield", "现金流收益率", "long", "估值", "CF/P，经营现金流TTM / 市值。", 0.7, "percentile", missing_score=35),
    FactorSpec("sales_to_price", "营收市值比", "long", "估值", "S/P，营业收入TTM / 市值。", 0.4, "percentile", missing_score=35),
    FactorSpec("roa", "ROA", "long", "质量", "净利润TTM / 总资产。", 0.8, "percentile", missing_score=25),
    FactorSpec("operating_profitability", "经营盈利能力", "long", "质量", "营业利润代理 / 总资产。", 0.8, "percentile", missing_score=25),
    FactorSpec("gross_profit_assets", "毛利资产比", "long", "质量", "毛利TTM / 总资产，Novy-Marx 质量代理。", 0.7, "percentile", missing_score=25),
    FactorSpec("asset_growth", "资产扩张", "long", "投资", "总资产同比增速，越保守越好（CMA投资因子）。", 0.7, "percentile", "low", missing_score=45),
    FactorSpec("accrual_quality", "低应计", "long", "质量", "净利润与经营现金流差额 / 总资产，越低越好。", 0.7, "percentile", "low", missing_score=45),
    FactorSpec("margin_stability", "利润率稳定", "long", "质量", "净利率波动越低越好。", 0.5, "percentile", "low", missing_score=45),
    FactorSpec("revenue_stability", "营收稳定", "long", "质量", "近年营收波动越低越好。", 0.4, "percentile", "low", missing_score=45),
    FactorSpec("leverage_trend", "杠杆改善", "long", "风险", "资产负债率趋势下降更好。", 0.4, "percentile", "low", missing_score=45),
    FactorSpec("reversal_1m", "一月反转", "long", "价格", "近20个交易日收益越低越好，A股短期反转效应（CH-4/CNE6）。", 0.6, "percentile", "low", missing_score=45),
    FactorSpec("momentum_12_1", "12-1月动量", "long", "价格", "近一年剔除最近一月的经典动量，A股动量偏弱故默认低权重。", 0.3, "percentile", missing_score=45),
    FactorSpec("dist_52w_high", "52周高点距离", "long", "价格", "现价/52周最高收盘价，George-Hwang 高点动量。", 0.4, "percentile", missing_score=45),
    FactorSpec("reversal_long_term", "长期反转", "long", "价格", "近3年收益越低越好，长期输家组合溢价（CNE6长期反转）。", 0.4, "percentile", "low", missing_score=45),
    FactorSpec("abnormal_turnover", "异常换手", "long", "规模流动性", "近20日均换手/近250日均换手，低换手溢价（CH-4 PMO情绪代理）。", 0.7, "percentile", "low", missing_score=45),
    FactorSpec("piotroski_f", "PiotroskiF分", "long", "质量", "九项财务体检通过数按可得分量折算到0-9，经典质量打分。", 0.8, "bounded", "high", 0.0, 9.0, 35),
]


SHORT_FACTORS: List[FactorSpec] = [
    FactorSpec("lhb_recent_count", "近期上榜次数", "short", "龙虎榜", "近期龙虎榜上榜次数。", 0.8, "percentile"),
    FactorSpec("lhb_net_buy", "龙虎榜净买", "short", "龙虎榜", "近期龙虎榜净买额，机构/主力净买入正向预测短期收益。", 1.0, "percentile"),
    FactorSpec("lhb_net_buy_pct", "净买占成交", "short", "龙虎榜", "净买额占成交比例峰值。", 0.9, "percentile"),
    FactorSpec("lhb_buy_ratio", "龙虎榜买方主导", "short", "龙虎榜", "榜内买入额占龙虎榜成交额峰值，买方主导度。", 0.7, "percentile", missing_score=40),
    FactorSpec("net_buy_to_float", "净买占流通比", "short", "龙虎榜", "窗口净买额/流通市值，资金推动力按体量归一。", 0.8, "percentile", missing_score=40),
    FactorSpec("float_cap_small", "小流通盘", "short", "龙虎榜", "流通市值越小越易拉升，短线弹性因子。", 0.5, "percentile", "low", missing_score=40),
    FactorSpec("institution_buy_count", "机构买入次数", "short", "席位", "机构席位买入次数。", 0.5, "percentile"),
    FactorSpec("institution_net_buy", "机构净买额", "short", "席位", "机构席位净买额，实证对短期收益有正向预测力。", 0.7, "percentile"),
    FactorSpec("stat_lhb_count", "近月上榜频率", "short", "龙虎榜", "统计窗口内上榜次数。", 0.5, "percentile"),
    FactorSpec("stat_net_buy", "近月净买", "short", "龙虎榜", "统计窗口内龙虎榜净买。", 0.5, "percentile"),
    FactorSpec("hot_money_concurrent", "游资共振数", "short", "游资追踪", "同一窗口共买席位数。", 1.2, "percentile"),
    FactorSpec("weighted_hot_money", "席位加权分", "short", "游资追踪", "席位质量、金额与共振的加权结果。", 1.0, "percentile"),
    FactorSpec("buy_amount_total", "游资买入额", "short", "游资追踪", "跟踪席位合计买入金额。", 0.8, "percentile"),
    FactorSpec("known_hot_money_ratio", "知名游资占比", "short", "游资追踪", "已识别知名游资席位占比。", 0.7, "bounded", "high", 0.0, 1.0),
    FactorSpec("seat_diversity", "席位多样性", "short", "游资追踪", "不同营业部数量。", 0.5, "percentile"),
    FactorSpec("recency", "上榜新鲜度", "short", "时效", "最近席位日期距快照日越近越好。", 0.7, "percentile", "low", missing_score=40),
    FactorSpec("best_window_tightness", "共振窗口紧凑", "short", "时效", "共振窗口跨度越短越好。", 0.5, "percentile", "low", missing_score=50),
    FactorSpec("resonance_score", "原共振分", "short", "资金评分", "现有龙虎榜共振评分。", 0.8, "direct", missing_score=40),
    FactorSpec("momentum_score", "短线动量", "short", "技术", "现有短线动量评分或技术形态。", 0.8, "direct", missing_score=40),
    FactorSpec("position_score", "短线位置", "short", "技术", "距高点/RSI等位置评分。", 0.6, "direct", missing_score=40),
    FactorSpec("risk_control", "风险控制", "short", "风控", "100 - 风险扣分。", 0.9, "direct", missing_score=45),
    FactorSpec("limit_up_control", "连板约束", "short", "风控", "连板越少越容易成交。", 0.5, "percentile", "low", missing_score=60),
    FactorSpec("tradability", "可交易性", "short", "风控", "非一字板、非T字板得分更高。", 0.6, "direct", missing_score=60),
    FactorSpec("turnover_heat", "换手热度", "short", "技术", "当日换手率或成交热度。", 0.4, "percentile", missing_score=35),
    FactorSpec("ma_bull", "均线多头", "short", "技术", "MA5 > MA10 > MA20 且站上MA5。", 0.4, "boolean"),
    FactorSpec("volume_ratio", "量比", "short", "技术", "量比越活跃越好，极端天量由风险项约束。", 0.4, "percentile", missing_score=35),
    FactorSpec("rsi_sweetspot", "RSI甜区", "short", "技术", "RSI处在50-75附近更适合1-5天进攻。", 0.4, "direct", missing_score=45),
    FactorSpec("lhb_reason_strength", "上榜原因强度", "short", "龙虎榜", "涨停/异常波动/换手等上榜原因强度。", 0.6, "direct", missing_score=40),
    FactorSpec("lhb_reason_diversity", "上榜原因多样性", "short", "龙虎榜", "多种上榜原因叠加。", 0.3, "percentile", missing_score=35),
    FactorSpec("institution_hotmoney_combo", "机构游资共振", "short", "席位", "机构净买和游资共振同时为正。", 0.8, "direct", missing_score=35),
    FactorSpec("institution_conflict", "机构分歧惩罚", "short", "席位", "机构净卖或龙虎榜净买背离时扣分。", 0.4, "direct", missing_score=45),
    FactorSpec("buy_concentration", "买额集中度", "short", "游资追踪", "单一席位买额占比，过散/过独都降权。", 0.5, "direct", missing_score=40),
    FactorSpec("known_hot_money_amount_ratio", "知名游资金额占比", "short", "游资追踪", "知名游资买入金额占跟踪买入额。", 0.7, "bounded", "high", 0.0, 1.0, 35),
    FactorSpec("hot_money_persistence", "游资持续性", "short", "游资追踪", "同席位多日重复出现。", 0.7, "direct", missing_score=35),
    FactorSpec("amount_per_buyer", "席位平均买额", "short", "游资追踪", "每个买方席位平均买入额。", 0.4, "percentile", missing_score=35),
    FactorSpec("event_recency_decay", "事件衰减分", "short", "时效", "按上榜距今自然日指数衰减。", 0.6, "direct", missing_score=40),
    FactorSpec("limit_up_heat", "涨停热度", "short", "技术", "近期涨停次数。", 0.4, "percentile", missing_score=35),
    FactorSpec("overheat_penalty", "过热惩罚", "short", "风控", "短期涨幅、连板、RSI过热综合扣分后得分。", 0.7, "direct", missing_score=45),
    FactorSpec("ma_distance", "均线乖离", "short", "技术", "距离MA20适中更好，过远降权。", 0.4, "direct", missing_score=45),
    FactorSpec("macd_strength", "MACD强度", "short", "技术", "DIF转强但不过热。", 0.3, "direct", missing_score=45),
    FactorSpec("alpha_price_volume_1", "价量Alpha1", "short", "价量Alpha", "涨幅与量比的短周期合成（WorldQuant式价量交互）。", 0.5, "direct", missing_score=40),
    FactorSpec("alpha_reversal_1", "短反Alpha", "short", "价量Alpha", "当日涨幅甜区与过热约束的反转/延续平衡。", 0.4, "direct", missing_score=40),
]


FACTOR_REGISTRY: Dict[str, FactorSpec] = {
    spec.key: spec for spec in LONG_FACTORS + SHORT_FACTORS
}


DEFAULT_CONFIG: Dict[str, Any] = {
    "long": {
        "enabled": True,
        "top_n": 12,
        "min_score": 60,
        "min_market_cap_yi": 100,
        "min_listing_years": 5,
        "require_csi300": False,
        "min_csi300_persistence": 45,
        "require_high_drawdown": False,
        "min_high_drawdown_pct": 40,
        "exclude_st": True,
        "hold_years_min": 2,
        "hold_years_max": 5,
        "weights": {spec.key: spec.default_weight for spec in LONG_FACTORS},
    },
    "short": {
        "enabled": True,
        "top_n": 8,
        "min_score": 55,
        "hold_days_min": 1,
        "hold_days_max": 5,
        "exclude_st": True,
        "min_lhb_count": 1,
        "min_hot_money_concurrent": 0,
        "max_consecutive_limit_up": 3,
        "weights": {spec.key: spec.default_weight for spec in SHORT_FACTORS},
    },
}


# ---------------------------------------------------------------------------
# Safe numerics and JSON helpers
# ---------------------------------------------------------------------------


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def first_not_none(*values: Any) -> Optional[float]:
    # Unlike `safe_float(a) or safe_float(b)`, a legitimate 0.0 is kept.
    for value in values:
        num = safe_float(value)
        if num is not None:
            return num
    return None


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def clean_round(value: Any, digits: int = 4) -> Any:
    num = safe_float(value)
    if num is None:
        return None
    return round(num, digits)


def parse_date(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value)[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None


def _file_signature(path: Path) -> Tuple[int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (0, 0)
    return (stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=256)
def _load_json_cached(path_text: str, signature: Tuple[int, int]) -> Any:
    with open(path_text, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_json_file(path_text: str) -> Any:
    # Keyed by (path, mtime, size) so externally refreshed files are re-read
    # without restarting the long-running dashboard process.
    return _load_json_cached(path_text, _file_signature(Path(path_text)))


_DIR_FINGERPRINT_TTL_SEC = 2.0
_dir_fingerprint_cache: Dict[Tuple[str, str], Tuple[float, int]] = {}


def _dir_fingerprint(directory: Path, pattern: str) -> int:
    key = (str(directory), pattern)
    now = time.monotonic()
    cached = _dir_fingerprint_cache.get(key)
    if cached is not None and now - cached[0] < _DIR_FINGERPRINT_TTL_SEC:
        return cached[1]
    fingerprint = 0
    if directory.exists():
        for fp in directory.glob(pattern):
            try:
                stat = fp.stat()
            except OSError:
                continue
            fingerprint ^= hash((fp.name, stat.st_mtime_ns, stat.st_size))
    _dir_fingerprint_cache[key] = (now, fingerprint)
    return fingerprint


def invalidate_dir_fingerprints() -> None:
    # Drops the TTL'd directory fingerprints so the next access re-scans
    # immediately; loaded data is still reused when fingerprints are unchanged.
    _dir_fingerprint_cache.clear()


def load_json_optional(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return load_json_file(str(path))
    except (OSError, json.JSONDecodeError):
        return default


def deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    if not isinstance(override, dict):
        return result
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Financial helpers
# ---------------------------------------------------------------------------


def compute_ttm(records: List[Dict[str, Any]], field: str) -> Optional[float]:
    if not records:
        return None
    latest = records[0]
    latest_val = safe_float(latest.get(field))
    if latest_val is None:
        return None
    date = latest.get("date", "")
    if len(date) < 7:
        return latest_val
    month = date[5:7]
    if month == "12":
        return latest_val

    try:
        latest_year = int(date[:4])
    except ValueError:
        return latest_val

    prev_fy = None
    prev_same = None
    for row in records[1:]:
        row_date = row.get("date", "")
        if len(row_date) < 7:
            continue
        value = safe_float(row.get(field))
        if value is None:
            continue
        try:
            year = int(row_date[:4])
        except ValueError:
            continue
        if year == latest_year - 1:
            if row_date[5:7] == "12":
                prev_fy = value
            if row_date[5:7] == month:
                prev_same = value
    if prev_fy is not None and prev_same is not None:
        return latest_val + prev_fy - prev_same
    return latest_val


def compute_yoy_growth(records: List[Dict[str, Any]], field: str) -> Optional[float]:
    now = compute_ttm(records, field)
    if now is None or not records:
        return None
    latest_date = records[0].get("date", "")
    if len(latest_date) < 7:
        return None
    try:
        latest_year = int(latest_date[:4])
    except ValueError:
        return None
    target_prefix = f"{latest_year - 1}-{latest_date[5:7]}"
    sub_records = [row for row in records if str(row.get("date", "")) <= target_prefix]
    prev = compute_ttm(sub_records, field)
    if prev is None or abs(prev) < 1e-9:
        return None
    return (now - prev) / abs(prev)


def mean_or_none(values: Iterable[Any]) -> Optional[float]:
    nums = [safe_float(v) for v in values]
    nums = [v for v in nums if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def stdev_or_zero(values: Iterable[Any]) -> float:
    nums = [safe_float(v) for v in values]
    nums = [v for v in nums if v is not None]
    if len(nums) < 2:
        return 0.0
    mu = sum(nums) / len(nums)
    return math.sqrt(sum((v - mu) ** 2 for v in nums) / len(nums))


def latest_record_with(records: List[Dict[str, Any]], field: str) -> Optional[float]:
    for row in reversed(records or []):
        value = safe_float(row.get(field))
        if value is not None:
            return value
    return None


def estimate_shares(stock: Dict[str, Any]) -> Optional[float]:
    indicators = stock.get("indicators", {}).get("records", [])
    balance = stock.get("financials", {}).get("balance", [])
    if not indicators or not balance:
        return None
    bvps = safe_float(indicators[0].get("bvps_adjusted"))
    equity = safe_float(balance[0].get("total_equity_parent"))
    if bvps and bvps > 0 and equity and equity > 0:
        return equity / bvps
    return None


def estimate_market_cap_cny(stock: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> Optional[float]:
    snap_cap = safe_float((snapshot or {}).get("market_cap_est"))
    if snap_cap and snap_cap > 0:
        return snap_cap

    shares = estimate_shares(stock)
    price = safe_float(stock.get("daily", {}).get("stats", {}).get("latest_close"))
    if shares and shares > 0 and price and price > 0:
        return shares * price
    return None


def dividend_yield(stock: Dict[str, Any], years: int = 5) -> Optional[float]:
    yearly = stock.get("dividends", {}).get("yearly_dividends", {})
    price = safe_float(stock.get("daily", {}).get("stats", {}).get("latest_close"))
    if not yearly or not price or price <= 0:
        return None
    current_year = datetime.now().year
    values = []
    for offset in range(1, years + 1):
        div = safe_float(yearly.get(str(current_year - offset)))
        if div is not None and div > 0:
            values.append(div / 10.0)
    if not values:
        return None
    return (sum(values) / len(values)) / price


def listing_age_years(snapshot: Optional[Dict[str, Any]]) -> Optional[float]:
    date = parse_date((snapshot or {}).get("listing_date"))
    if not date:
        return None
    return max(0.0, (datetime.now() - date).days / 365.25)


def high_to_latest_drawdown_pct(rows: List[Dict[str, Any]]) -> Optional[float]:
    closes = [safe_float(row.get("close")) for row in rows or []]
    closes = [c for c in closes if c is not None and c > 0]
    if len(closes) < 2:
        return None
    peak = max(closes)
    if peak <= 0:
        return None
    return (peak - closes[-1]) / peak * 100.0


def safe_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    num = safe_float(numerator)
    den = safe_float(denominator)
    if num is None or den is None or abs(den) < 1e-9:
        return None
    return num / den


def annual_values(records: List[Dict[str, Any]], field: str, limit: int = 5) -> List[float]:
    values = []
    for row in records or []:
        if not str(row.get("date", "")).endswith("-12-31"):
            continue
        value = safe_float(row.get(field))
        if value is not None:
            values.append(value)
            if len(values) >= limit:
                break
    return values


def latest_value(records: List[Dict[str, Any]], field: str) -> Optional[float]:
    if not records:
        return None
    return safe_float(records[0].get(field))


def yoy_from_latest_and_annual(records: List[Dict[str, Any]], field: str) -> Optional[float]:
    latest = latest_value(records, field)
    annual = annual_values(records, field, limit=2)
    base = annual[1] if len(annual) > 1 else (annual[0] if annual else None)
    if latest is None or base is None or abs(base) < 1e-9:
        return None
    return (latest - base) / abs(base)


def piotroski_f_score(
        income: List[Dict[str, Any]],
        indicators: List[Dict[str, Any]],
        net_profit_ttm: Optional[float],
        op_cash_ttm: Optional[float],
        total_assets: Optional[float],
) -> Optional[float]:
    """Piotroski F-Score 的本地可计算子集（折算到满分9）。

    九项中可用本地数据核验七项：ROA>0、经营现金流>0、现金流>净利润(低应计)、
    ROE同比改善、负债率同比下降、毛利率同比改善、资产周转率同比改善；
    发行新股与流动比率两项缺数据。可得分量不足5项返回 None。
    """
    checks: List[bool] = []
    roa = safe_ratio(net_profit_ttm, total_assets)
    if roa is not None:
        checks.append(roa > 0)
    if op_cash_ttm is not None:
        checks.append(op_cash_ttm > 0)
    if op_cash_ttm is not None and net_profit_ttm is not None:
        checks.append(op_cash_ttm > net_profit_ttm)
    roe_y = annual_values(indicators, "roe", limit=2)
    if len(roe_y) >= 2:
        checks.append(roe_y[0] > roe_y[1])
    debt_y = annual_values(indicators, "asset_liability_ratio", limit=2)
    if len(debt_y) >= 2:
        checks.append(debt_y[0] < debt_y[1])
    gm_y = annual_values(indicators, "gross_margin", limit=2)
    if len(gm_y) >= 2:
        checks.append(gm_y[0] > gm_y[1])
    rev_y = annual_values(income, "revenue", limit=2)
    assets_y = annual_values(indicators, "total_assets", limit=2)
    if len(rev_y) >= 2 and len(assets_y) >= 2 and assets_y[0] and assets_y[1]:
        checks.append(rev_y[0] / assets_y[0] > rev_y[1] / assets_y[1])
    if len(checks) < 5:
        return None
    return round(sum(checks) / len(checks) * 9.0, 3)


def pct_volatility(values: Iterable[Any]) -> Optional[float]:
    nums = [safe_float(v) for v in values]
    nums = [v for v in nums if v is not None]
    if len(nums) < 2:
        return None
    avg = sum(abs(v) for v in nums) / len(nums)
    if avg < 1e-9:
        return None
    return stdev_or_zero(nums) / avg


def bounded_linear(value: Optional[float], low: float, high: float, reverse: bool = False) -> Optional[float]:
    if value is None or abs(high - low) < 1e-9:
        return None
    score = (value - low) / (high - low) * 100.0
    if reverse:
        score = 100.0 - score
    return clamp(score)


def sweetspot_score(value: Optional[float], low: float, high: float, hard_low: float, hard_high: float) -> Optional[float]:
    v = safe_float(value)
    if v is None:
        return None
    if low <= v <= high:
        return 100.0
    if v < low:
        if v <= hard_low:
            return 0.0
        return clamp((v - hard_low) / (low - hard_low) * 100.0)
    if v >= hard_high:
        return 0.0
    return clamp((hard_high - v) / (hard_high - high) * 100.0)


def average_score(*values: Optional[float]) -> Optional[float]:
    nums = [safe_float(v) for v in values]
    nums = [v for v in nums if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def rsi_sweetspot_score(rsi: Optional[float]) -> Optional[float]:
    r = safe_float(rsi)
    if r is None:
        return None
    if 50 <= r <= 75:
        return 100.0
    if 40 <= r < 50:
        return 75.0 + (r - 40) * 2.5
    if 75 < r <= 85:
        return 100.0 - (r - 75) * 5.0
    if 30 <= r < 40:
        return 45.0 + (r - 30) * 3.0
    if 85 < r <= 100:
        return max(0.0, 50.0 - (r - 85) * 3.0)
    return 25.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_stock_universe() -> Dict[str, Any]:
    return load_json_optional(DATA_DIR / "stock_universe.json", {})


def load_market_snapshot() -> Dict[str, Any]:
    return load_json_optional(DATA_DIR / "market_snapshot.json", {})


def load_cn_stock_index() -> Dict[str, Dict[str, Any]]:
    return _load_cn_stock_index_cached(_dir_fingerprint(CN_STOCK_DIR, "CN_*.json"))


@lru_cache(maxsize=1)
def _load_cn_stock_index_cached(fingerprint: int) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    if not CN_STOCK_DIR.exists():
        return index
    for fp in CN_STOCK_DIR.glob("CN_*.json"):
        try:
            data = load_json_file(str(fp))
        except (OSError, json.JSONDecodeError):
            continue
        code = str(data.get("symbol") or fp.name.split("_")[1]).zfill(6)
        index[code] = data
    return index


def load_fundamental_stocks() -> Dict[str, Dict[str, Any]]:
    return _load_fundamental_stocks_cached(_dir_fingerprint(STOCK_DATA_DIR, "CN_*.json"))


@lru_cache(maxsize=1)
def _load_fundamental_stocks_cached(fingerprint: int) -> Dict[str, Dict[str, Any]]:
    stocks: Dict[str, Dict[str, Any]] = {}
    if STOCK_DATA_DIR.exists():
        for fp in STOCK_DATA_DIR.glob("CN_*.json"):
            try:
                data = load_json_file(str(fp))
            except (OSError, json.JSONDecodeError):
                continue
            code = str(data.get("symbol") or fp.name.split("_")[1]).zfill(6)
            stocks[code] = data
    return stocks


def latest_valuation_from_cn(code: str) -> Dict[str, Optional[float]]:
    cn = load_cn_stock_index().get(code) or {}
    records = cn.get("records", [])
    return {
        "market_cap": latest_record_with(records, "market_cap"),
        "pe_ttm": latest_record_with(records, "pe_ttm"),
        "pb": latest_record_with(records, "pb"),
        "pcf": latest_record_with(records, "pcf"),
    }


def combined_recent_rows(code: str, stock: Dict[str, Any]) -> List[Dict[str, Any]]:
    cn = load_cn_stock_index().get(code)
    if cn and cn.get("records"):
        return cn["records"][-80:]
    return stock.get("daily", {}).get("recent_daily", [])[-80:]


def price_history_rows(code: str, stock: Dict[str, Any]) -> List[Dict[str, Any]]:
    cn = load_cn_stock_index().get(code)
    if cn and cn.get("records"):
        return cn["records"]
    return stock.get("daily", {}).get("recent_daily", [])


def historical_high_drawdown_pct(code: str, stock: Dict[str, Any]) -> Optional[float]:
    return high_to_latest_drawdown_pct(price_history_rows(code, stock))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def percentile_factor_scores(
        items: List[Dict[str, Any]], spec: FactorSpec
) -> Dict[str, float]:
    raw_values = []
    for idx, item in enumerate(items):
        value = safe_float(item["raw_factors"].get(spec.key))
        if value is not None:
            raw_values.append((idx, value))
    if not raw_values:
        return {item["code"]: spec.missing_score for item in items}
    raw_values.sort(key=lambda pair: pair[1])
    n = len(raw_values)
    scores_by_idx: Dict[int, float] = {}
    if n == 1:
        scores_by_idx[raw_values[0][0]] = 100.0
    else:
        # Ties share the average rank so equal raw values get equal scores.
        i = 0
        while i < n:
            j = i
            while j + 1 < n and raw_values[j + 1][1] == raw_values[i][1]:
                j += 1
            rank = (i + j) / 2.0
            score = rank / (n - 1) * 100.0
            if spec.direction == "low":
                score = 100.0 - score
            for k in range(i, j + 1):
                scores_by_idx[raw_values[k][0]] = score
            i = j + 1
    return {
        item["code"]: round(scores_by_idx.get(idx, spec.missing_score), 4)
        for idx, item in enumerate(items)
    }


def direct_factor_score(raw: Any, spec: FactorSpec) -> float:
    value = safe_float(raw)
    if value is None:
        return spec.missing_score
    if spec.score == "boolean":
        return 100.0 if value else 0.0
    if spec.score == "direct":
        return clamp(value)
    if spec.score == "bounded":
        low = spec.min_value if spec.min_value is not None else 0.0
        high = spec.max_value if spec.max_value is not None else 1.0
        if abs(high - low) < 1e-12:
            return spec.missing_score
        score = (value - low) / (high - low) * 100.0
        if spec.direction == "low":
            score = 100.0 - score
        return clamp(score)
    return spec.missing_score


def apply_scores(
        items: List[Dict[str, Any]],
        specs: List[FactorSpec],
        weights: Dict[str, Any],
) -> List[Dict[str, Any]]:
    percentile_cache: Dict[str, Dict[str, float]] = {}
    for spec in specs:
        if spec.score == "percentile":
            percentile_cache[spec.key] = percentile_factor_scores(items, spec)

    scored = []
    for item in items:
        factor_scores = {}
        weighted_sum = 0.0
        weight_sum = 0.0
        positive_weight_count = 0
        for spec in specs:
            weight = safe_float(weights.get(spec.key))
            if weight is None:
                weight = spec.default_weight
            weight = max(0.0, weight)
            raw = item["raw_factors"].get(spec.key)
            if spec.score == "percentile":
                score = percentile_cache[spec.key].get(item["code"], spec.missing_score)
            else:
                score = direct_factor_score(raw, spec)
            if weight > 0:
                weighted_sum += score * weight
                weight_sum += weight
                positive_weight_count += 1
            factor_scores[spec.key] = {
                "label": spec.label,
                "raw": clean_round(raw, 6),
                "score": round(score, 2),
                "weight": round(weight, 4),
            }
        total = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        data_quality = item.get("data_quality", 0.0)
        result = dict(item)
        result["score"] = round(total, 2)
        result["factor_scores"] = factor_scores
        result["factor_count"] = positive_weight_count
        result["data_quality"] = round(data_quality, 3)
        scored.append(result)
    scored.sort(key=lambda row: row["score"], reverse=True)
    for rank, item in enumerate(scored, 1):
        item["rank"] = rank
    return scored


def rerank_scored(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for rank, item in enumerate(items, 1):
        item["rank"] = rank
    return items


def available_factor_ratio(raw_factors: Dict[str, Any], specs: List[FactorSpec]) -> float:
    if not specs:
        return 0.0
    present = 0
    for spec in specs:
        value = raw_factors.get(spec.key)
        if safe_float(value) is not None:
            present += 1
    return present / len(specs)


# ---------------------------------------------------------------------------
# Long horizon strategy
# ---------------------------------------------------------------------------


def build_long_candidates(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    stocks = load_fundamental_stocks()
    universe = load_stock_universe()
    snapshot = load_market_snapshot()
    csi300 = set(str(code).zfill(6) for code in universe.get("csi300", []))
    csi_all = set(str(code).zfill(6) for code in universe.get("all", []))
    notes = []
    if not csi300:
        notes.append("data/stock_universe.json lacks csi300; CSI300 filters are inactive.")

    min_cap_yi = safe_float(config.get("min_market_cap_yi")) or 0.0
    min_listing = safe_float(config.get("min_listing_years")) or 0.0
    require_csi300 = bool(config.get("require_csi300", True))
    min_persistence = safe_float(config.get("min_csi300_persistence")) or 0.0
    require_high_drawdown = bool(config.get("require_high_drawdown", False))
    min_high_drawdown_pct = safe_float(config.get("min_high_drawdown_pct")) or 0.0
    exclude_st = bool(config.get("exclude_st", True))

    candidates: List[Dict[str, Any]] = []
    industry_buckets: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    for code, stock in stocks.items():
        name = str(stock.get("name") or snapshot.get(code, {}).get("name") or "")
        if exclude_st and is_st_name(name):
            continue
        snap = snapshot.get(code, {})
        in_csi300 = code in csi300
        in_csi_all = bool(snap.get("in_csi_all")) or code in csi_all
        age = listing_age_years(snap)
        if min_listing and (age is None or age < min_listing):
            continue
        if require_csi300 and not in_csi300:
            continue

        market_cap_cny = estimate_market_cap_cny(stock, snap)
        cn_val = latest_valuation_from_cn(code)
        if cn_val.get("market_cap"):
            market_cap_yi = cn_val["market_cap"]
            market_cap_cny = market_cap_yi * 1e8
        else:
            market_cap_yi = market_cap_cny / 1e8 if market_cap_cny else None
        if min_cap_yi and (market_cap_yi is None or market_cap_yi < min_cap_yi):
            continue

        high_drawdown_pct = historical_high_drawdown_pct(code, stock)
        if require_high_drawdown and (
                high_drawdown_pct is None or high_drawdown_pct < min_high_drawdown_pct
        ):
            continue

        raw = compute_long_raw_factors(
            code, stock, snap, in_csi300, in_csi_all, age, market_cap_cny, high_drawdown_pct
        )
        if raw["csi300_persistence"] < min_persistence:
            continue

        industry = str(stock.get("pledge", {}).get("industry") or snap.get("pledge_industry") or "UNKNOWN")
        item = {
            "code": code,
            "name": name,
            "strategy": "long",
            "horizon": f"{config.get('hold_years_min', 2)}-{config.get('hold_years_max', 5)} years",
            "industry": industry,
            "raw_factors": raw,
            "reasons": [],
            "warnings": [],
        }
        candidates.append(item)
        if market_cap_cny:
            industry_buckets[industry].append((code, market_cap_cny))

    apply_industry_leadership(candidates, industry_buckets)
    for item in candidates:
        item["data_quality"] = available_factor_ratio(item["raw_factors"], LONG_FACTORS)
        item["reasons"] = long_reasons(item)
    return candidates, notes


def passes_long_hard_filters(item: Dict[str, Any], config: Dict[str, Any]) -> bool:
    raw = item.get("raw_factors", {})
    if bool(config.get("exclude_st", True)) and is_st_name(str(item.get("name") or "")):
        return False

    min_cap_yi = safe_float(config.get("min_market_cap_yi")) or 0.0
    market_cap_yi = safe_float(raw.get("market_cap"))
    if min_cap_yi and (market_cap_yi is None or market_cap_yi < min_cap_yi):
        return False

    min_listing = safe_float(config.get("min_listing_years")) or 0.0
    listing_age = safe_float(raw.get("listing_age"))
    if min_listing and (listing_age is None or listing_age < min_listing):
        return False

    min_persistence = safe_float(config.get("min_csi300_persistence")) or 0.0
    persistence = safe_float(raw.get("csi300_persistence"))
    if min_persistence and (persistence is None or persistence < min_persistence):
        return False

    if bool(config.get("require_csi300", True)) and not raw.get("csi300_current"):
        return False

    if bool(config.get("require_high_drawdown", False)):
        min_high_drawdown_pct = safe_float(config.get("min_high_drawdown_pct")) or 0.0
        high_drawdown = safe_float(raw.get("historical_high_drawdown"))
        if high_drawdown is None or high_drawdown < min_high_drawdown_pct:
            return False

    return True


def compute_long_raw_factors(
        code: str,
        stock: Dict[str, Any],
        snapshot: Dict[str, Any],
        in_csi300: bool,
        in_csi_all: bool,
        age: Optional[float],
        market_cap_cny: Optional[float],
        high_drawdown_pct: Optional[float] = None,
) -> Dict[str, Any]:
    daily_stats = stock.get("daily", {}).get("stats", {})
    financials = stock.get("financials", {})
    income = financials.get("income", [])
    balance = financials.get("balance", [])
    cashflow = financials.get("cashflow", [])
    indicators = stock.get("indicators", {}).get("records", [])
    roe_stats = stock.get("indicators", {}).get("roe_stats", {})
    latest_ind = indicators[0] if indicators else {}

    net_profit_ttm = compute_ttm(income, "net_profit")
    revenue_ttm = compute_ttm(income, "revenue")
    cost_ttm = compute_ttm(income, "cost_of_revenue")
    operating_cost_ttm = compute_ttm(income, "operating_cost")
    op_cash_ttm = compute_ttm(cashflow, "operating_cashflow_net")
    cash_quality = None
    if op_cash_ttm is not None and net_profit_ttm is not None and abs(net_profit_ttm) > 1e-9:
        cash_quality = op_cash_ttm / abs(net_profit_ttm)

    market_cap_yi = market_cap_cny / 1e8 if market_cap_cny else None

    # 价格行为因子用全量历史日线（CN_stock 最长约10年）
    hist_rows = price_history_rows(code, stock)
    hist_closes = [safe_float(row.get("close")) for row in hist_rows]
    hist_closes = [c for c in hist_closes if c is not None and c > 0]
    reversal_1m = None
    momentum_12_1 = None
    dist_52w_high = None
    reversal_long_term = None
    if len(hist_closes) >= 21:
        reversal_1m = hist_closes[-1] / hist_closes[-21] - 1
    if len(hist_closes) >= 250:
        momentum_12_1 = hist_closes[-21] / hist_closes[-250] - 1
    if len(hist_closes) >= 120:
        window = hist_closes[-250:]
        peak = max(window)
        dist_52w_high = hist_closes[-1] / peak if peak > 0 else None
    if len(hist_closes) >= 750:
        reversal_long_term = hist_closes[-1] / hist_closes[-750] - 1

    hist_turnovers = [safe_float(row.get("turnover_rate")) for row in hist_rows]
    recent_to = [t for t in hist_turnovers[-20:] if t is not None and t > 0]
    base_to = [t for t in hist_turnovers[-250:] if t is not None and t > 0]
    abnormal_turnover = None
    if len(recent_to) >= 10 and len(base_to) >= 60:
        base_avg = sum(base_to) / len(base_to)
        if base_avg > 1e-9:
            abnormal_turnover = (sum(recent_to) / len(recent_to)) / base_avg

    roe_values = [safe_float(row.get("roe")) for row in indicators]
    roe_values = [v for v in roe_values if v is not None]
    roe_mean = first_not_none(roe_stats.get("mean"), mean_or_none(roe_values))
    roe_std = safe_float(roe_stats.get("std"))
    if roe_std is None:
        roe_std = stdev_or_zero(roe_values)
    roe_stability = None
    if roe_mean is not None:
        roe_stability = roe_mean - (roe_std or 0.0)

    persistence = csi300_persistence_proxy(in_csi300, in_csi_all, age)
    total_assets = safe_float(latest_ind.get("total_assets"))
    if total_assets is None:
        total_assets = latest_value(balance, "total_assets_liabilities")
    equity = latest_value(balance, "total_equity_parent") or latest_value(balance, "total_equity")
    gross_profit_ttm = None
    if revenue_ttm is not None and cost_ttm is not None:
        gross_profit_ttm = revenue_ttm - cost_ttm
    elif revenue_ttm is not None and safe_float(latest_ind.get("gross_margin")) is not None:
        gross_profit_ttm = revenue_ttm * safe_float(latest_ind.get("gross_margin")) / 100.0

    operating_profit = None
    if revenue_ttm is not None and operating_cost_ttm is not None:
        operating_profit = revenue_ttm - operating_cost_ttm

    book_to_market = safe_ratio(equity, market_cap_cny)
    earnings_yield = safe_ratio(net_profit_ttm, market_cap_cny)
    cashflow_yield = safe_ratio(op_cash_ttm, market_cap_cny)
    sales_to_price = safe_ratio(revenue_ttm, market_cap_cny)
    roa = safe_ratio(net_profit_ttm, total_assets)
    operating_profitability = safe_ratio(operating_profit, total_assets)
    gross_profit_assets = safe_ratio(gross_profit_ttm, total_assets)
    asset_growth = yoy_from_latest_and_annual(indicators, "total_assets")
    accrual_quality = None
    if net_profit_ttm is not None and op_cash_ttm is not None and total_assets:
        accrual_quality = (net_profit_ttm - op_cash_ttm) / total_assets

    margin_values = [safe_float(row.get("net_margin")) for row in indicators[:8]]
    margin_stability = pct_volatility(margin_values)
    revenue_stability = pct_volatility(annual_values(income, "revenue", limit=5))
    debt_values = [safe_float(row.get("asset_liability_ratio")) for row in indicators[:8]]
    leverage_trend = None
    if len([v for v in debt_values if v is not None]) >= 2 and debt_values[0] is not None:
        older = next((v for v in reversed(debt_values) if v is not None), None)
        if older is not None:
            leverage_trend = debt_values[0] - older

    dividend_yield_5y = scale_ratio_to_pct(dividend_yield(stock, years=5))
    piotroski = piotroski_f_score(income, indicators, net_profit_ttm, op_cash_ttm, total_assets)

    return {
        "csi300_current": 1.0 if in_csi300 else 0.0,
        "csi300_persistence": persistence,
        "market_cap": market_cap_yi,
        "size_reversal": market_cap_yi,
        "liquidity": first_not_none(
            daily_stats.get("avg_daily_turnover_approx"), snapshot.get("turnover")
        ),
        "low_volatility": safe_float(daily_stats.get("volatility_annual")),
        "roe_mean": roe_mean,
        "roe_stability": roe_stability,
        "revenue_growth": scale_ratio_to_pct(compute_yoy_growth(income, "revenue")),
        "net_profit_growth": scale_ratio_to_pct(compute_yoy_growth(income, "net_profit")),
        "gross_margin": safe_float(latest_ind.get("gross_margin")),
        "net_margin": safe_float(latest_ind.get("net_margin")),
        "cashflow_quality": cash_quality,
        "dividend_yield_5y": dividend_yield_5y,
        "dividend_consistency": 1.0 if stock.get("dividends", {}).get("consecutive_3y_dividend") else 0.0,
        "debt_safety": safe_float(latest_ind.get("asset_liability_ratio")),
        "pledge_safety": first_not_none(
            stock.get("pledge", {}).get("pledge_ratio"), snapshot.get("pledge_ratio")
        ),
        "historical_high_drawdown": high_drawdown_pct,
        "industry_leadership": None,
        # listing_age 不再是打分因子，但 min_listing_years 硬过滤仍读取它
        "listing_age": age,
        "book_to_market": book_to_market,
        "earnings_yield": scale_ratio_to_pct(earnings_yield),
        "cashflow_yield": scale_ratio_to_pct(cashflow_yield),
        "sales_to_price": sales_to_price,
        "roa": scale_ratio_to_pct(roa),
        "operating_profitability": scale_ratio_to_pct(operating_profitability),
        "gross_profit_assets": scale_ratio_to_pct(gross_profit_assets),
        "asset_growth": scale_ratio_to_pct(asset_growth),
        "accrual_quality": accrual_quality,
        "margin_stability": margin_stability,
        "revenue_stability": revenue_stability,
        "leverage_trend": leverage_trend,
        "reversal_1m": scale_ratio_to_pct(reversal_1m),
        "momentum_12_1": scale_ratio_to_pct(momentum_12_1),
        "dist_52w_high": dist_52w_high,
        "reversal_long_term": scale_ratio_to_pct(reversal_long_term),
        "abnormal_turnover": abnormal_turnover,
        "piotroski_f": piotroski,
    }


def csi300_persistence_proxy(in_csi300: bool, in_csi_all: bool, age: Optional[float]) -> float:
    age_score = clamp(((age or 0.0) / 12.0) * 100.0)
    csi_score = 100.0 if in_csi300 else (45.0 if in_csi_all else 0.0)
    return round(csi_score * 0.72 + age_score * 0.28, 4)


def scale_ratio_to_pct(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value * 100.0


def apply_industry_leadership(
        candidates: List[Dict[str, Any]],
        buckets: Dict[str, List[Tuple[str, float]]],
) -> None:
    scores: Dict[str, float] = {}
    for _, pairs in buckets.items():
        if not pairs:
            continue
        pairs = sorted(pairs, key=lambda pair: pair[1])
        if len(pairs) == 1:
            scores[pairs[0][0]] = 100.0
            continue
        for rank, (code, _) in enumerate(pairs):
            scores[code] = rank / (len(pairs) - 1) * 100.0
    for item in candidates:
        item["raw_factors"]["industry_leadership"] = scores.get(item["code"])


def long_reasons(item: Dict[str, Any]) -> List[str]:
    raw = item["raw_factors"]
    reasons = []
    if raw.get("csi300_current"):
        reasons.append("当前沪深300")
    if safe_float(raw.get("market_cap")) is not None:
        reasons.append(f"市值约{raw['market_cap']:.0f}亿")
    if safe_float(raw.get("roe_stability")) is not None:
        reasons.append(f"ROE稳定因子{raw['roe_stability']:.1f}")
    if safe_float(raw.get("dividend_yield_5y")) is not None:
        reasons.append(f"五年股息率{raw['dividend_yield_5y']:.2f}%")
    if safe_float(raw.get("cashflow_quality")) is not None:
        reasons.append(f"现金流/利润{raw['cashflow_quality']:.2f}")
    return reasons[:5]


# ---------------------------------------------------------------------------
# Short horizon strategy
# ---------------------------------------------------------------------------


def load_short_pool() -> Dict[str, Dict[str, Any]]:
    pool: Dict[str, Dict[str, Any]] = {}
    scored = load_json_optional(CAPITAL_DIR / "scored_stocks.json", {})
    for stock in scored.get("stocks", []) if isinstance(scored, dict) else []:
        code = str(stock.get("code", "")).zfill(6)
        if not code:
            continue
        entry = pool.setdefault(code, {"code": code, "name": stock.get("name", ""), "sources": []})
        entry["sources"].append("capital_scored")
        entry["scored"] = stock
        entry["as_of_date"] = scored.get("as_of_date") or scored.get("generated_at")

    picks = load_json_optional(DATA_DIR / "main_capital_picks.json", {})
    for pick in picks.get("picks", []) if isinstance(picks, dict) else []:
        code = str(pick.get("code", "")).zfill(6)
        if not code:
            continue
        entry = pool.setdefault(code, {"code": code, "name": pick.get("name", ""), "sources": []})
        entry["sources"].append("main_capital_picks")
        entry["name"] = entry.get("name") or pick.get("name", "")
        entry["pick"] = pick
        entry["as_of_date"] = picks.get("generated_at")
    return pool


def build_short_candidates(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    pool = load_short_pool()
    notes = []
    if not pool:
        notes.append("No Dragon Tiger List pool found. Run stock_crawl_capital.py or main-capital script first.")

    exclude_st = bool(config.get("exclude_st", True))
    min_lhb_count = int(safe_float(config.get("min_lhb_count")) or 0)
    min_concurrent = int(safe_float(config.get("min_hot_money_concurrent")) or 0)
    max_consec = int(first_not_none(config.get("max_consecutive_limit_up"), 999))

    candidates = []
    for code, data in pool.items():
        name = str(data.get("name") or "")
        if exclude_st and is_st_name(name):
            continue
        raw = compute_short_raw_factors(data)
        if (safe_float(raw.get("lhb_recent_count")) or 0) < min_lhb_count:
            continue
        if (safe_float(raw.get("hot_money_concurrent")) or 0) < min_concurrent:
            continue
        if (safe_float(raw.get("limit_up_control")) or 0) > max_consec:
            continue

        item = {
            "code": code,
            "name": name,
            "strategy": "short",
            "horizon": f"{config.get('hold_days_min', 1)}-{config.get('hold_days_max', 5)} days",
            "industry": "",
            "raw_factors": raw,
            "reasons": short_reasons(data, raw),
            "warnings": short_warnings(data, raw),
            "followers": follower_sample(data),
            "sources": sorted(set(data.get("sources", []))),
        }
        item["data_quality"] = available_factor_ratio(raw, SHORT_FACTORS)
        candidates.append(item)
    return candidates, notes


def compute_short_raw_factors(data: Dict[str, Any]) -> Dict[str, Any]:
    pick = data.get("pick") or {}
    scored = data.get("scored") or {}
    signals = pick.get("signals") or scored.get("signals") or {}
    lhb = signals.get("lhb") or {}
    inst = signals.get("inst") or {}
    stat = signals.get("stat") or {}
    tech = signals.get("tech") or {}
    scores = scored.get("scores") or {}
    followers = normalize_followers(scored.get("followers") or [])

    seats = [str(f.get("seat", "")) for f in followers if f.get("seat")]
    seat_counts = Counter(seats)
    known_count = sum(1 for f in followers if str(f.get("category", "")).lower() == "knownhotmoney")
    buy_values = [safe_float(f.get("buy_est")) for f in followers]
    buy_values = [v for v in buy_values if v is not None and v > 0]
    total_buy_est = sum(buy_values) if buy_values else safe_float(scored.get("buy_amount_total"))
    known_buy_est = sum(
        safe_float(f.get("buy_est")) or 0.0
        for f in followers
        if str(f.get("category", "")).lower() == "knownhotmoney"
    )
    latest_date = latest_follower_date(followers)
    as_of = parse_date(scored.get("as_of_date") or data.get("as_of_date"))
    recency_days = None
    if latest_date and as_of:
        recency_days = max(0, (as_of - latest_date).days)

    best_window_span = None
    best_window = scored.get("best_window")
    if isinstance(best_window, list) and len(best_window) >= 2:
        d0 = parse_date(best_window[0])
        d1 = parse_date(best_window[1])
        if d0 and d1:
            best_window_span = abs((d1 - d0).days)

    risk_penalty = safe_float(scores.get("risk_penalty"))
    is_yizi = bool(scores.get("is_yizi_ban"))
    is_t_ban = bool(scores.get("is_t_ban"))
    tradability = 100.0
    if is_yizi:
        tradability -= 60.0
    if is_t_ban:
        tradability -= 25.0
    lhb_reasons = list(lhb.get("reasons") or [])
    reason_strength, reason_diversity = lhb_reason_features(lhb_reasons)
    inst_net = safe_float(inst.get("inst_net_buy"))
    lhb_net = safe_float(lhb.get("total_net_buy"))
    follower_count = len(followers)
    concurrent = first_not_none(
        scored.get("concurrent_count"), follower_count if follower_count else None
    )
    buy_concentration = buy_concentration_score(buy_values)
    known_amount_ratio = safe_ratio(known_buy_est, total_buy_est)
    persistence_score = hot_money_persistence_score(followers, seat_counts)
    event_decay = event_recency_decay_score(recency_days)
    institution_combo = institution_hotmoney_combo_score(inst_net, concurrent, lhb_net)
    institution_conflict = institution_conflict_score(inst_net, lhb_net)
    amount_per_buyer = safe_ratio(
        total_buy_est, first_not_none(scored.get("total_buyers"), len(seat_counts) or None)
    )
    limit_up_count = first_not_none(
        tech.get("recent_limit_up"), scores.get("consecutive_limit_up")
    )
    chg_5d = safe_float(tech.get("chg_5d"))
    chg_today = safe_float(tech.get("chg_today"))
    dist_ma20 = safe_float(tech.get("dist_from_ma20_pct"))
    rsi_score = rsi_sweetspot_score(safe_float(tech.get("rsi")))
    vol_ratio = safe_float(tech.get("vol_ratio"))
    turnover_today = safe_float(tech.get("turnover_today"))
    overheat = overheat_control_score(chg_5d, safe_float(scores.get("consecutive_limit_up")), safe_float(tech.get("rsi")))
    ma_distance = sweetspot_score(dist_ma20, 0.0, 18.0, -12.0, 45.0)
    macd_strength = sweetspot_score(safe_float(tech.get("macd_dif")), 0.0, 0.8, -0.6, 2.5)
    alpha_pv1 = average_score(
        sweetspot_score(chg_5d, 2.0, 12.0, -8.0, 28.0),
        sweetspot_score(vol_ratio, 1.1, 3.2, 0.4, 6.0),
    )
    alpha_rev1 = average_score(
        sweetspot_score(chg_today, -3.0, 7.5, -10.0, 15.0),
        overheat,
        bounded_linear(safe_float(scores.get("consecutive_limit_up")), 0.0, 4.0, reverse=True),
    )

    return {
        "lhb_recent_count": first_not_none(
            lhb.get("count"), float(follower_count) if follower_count else None
        ),
        "lhb_net_buy": safe_float(lhb.get("total_net_buy")),
        "lhb_net_buy_pct": safe_float(lhb.get("max_net_pct")),
        "lhb_buy_ratio": safe_float(lhb.get("max_buy_ratio")),
        "net_buy_to_float": safe_float(lhb.get("net_buy_to_float_pct")),
        "float_cap_small": safe_float(lhb.get("float_cap")),
        "institution_buy_count": safe_float(inst.get("inst_buy_count")),
        "institution_net_buy": safe_float(inst.get("inst_net_buy")),
        "stat_lhb_count": safe_float(stat.get("stat_count")),
        "stat_net_buy": safe_float(stat.get("stat_net_buy")),
        "hot_money_concurrent": concurrent,
        "weighted_hot_money": first_not_none(scored.get("weighted_score"), scores.get("total")),
        "buy_amount_total": safe_float(scored.get("buy_amount_total")),
        "known_hot_money_ratio": known_count / len(followers) if followers else None,
        "seat_diversity": len(seat_counts) if seat_counts else None,
        "recency": recency_days,
        "best_window_tightness": best_window_span,
        "resonance_score": safe_float(scores.get("resonance")),
        "momentum_score": first_not_none(scores.get("momentum"), tech_momentum_score(tech)),
        "position_score": safe_float(scores.get("position")),
        "risk_control": 100.0 - risk_penalty if risk_penalty is not None else None,
        "limit_up_control": safe_float(scores.get("consecutive_limit_up")),
        "tradability": clamp(tradability),
        "turnover_heat": safe_float(tech.get("turnover_today")),
        "ma_bull": 1.0 if tech.get("ma_bull") else 0.0,
        "volume_ratio": safe_float(tech.get("vol_ratio")),
        "rsi_sweetspot": rsi_sweetspot_score(safe_float(tech.get("rsi"))),
        "lhb_reason_strength": reason_strength,
        "lhb_reason_diversity": reason_diversity,
        "institution_hotmoney_combo": institution_combo,
        "institution_conflict": institution_conflict,
        "buy_concentration": buy_concentration,
        "known_hot_money_amount_ratio": known_amount_ratio,
        "hot_money_persistence": persistence_score,
        "amount_per_buyer": amount_per_buyer,
        "event_recency_decay": event_decay,
        "limit_up_heat": limit_up_count,
        "overheat_penalty": overheat,
        "ma_distance": ma_distance,
        "macd_strength": macd_strength,
        "alpha_price_volume_1": alpha_pv1,
        "alpha_reversal_1": alpha_rev1,
    }


def latest_follower_date(followers: List[Dict[str, Any]]) -> Optional[datetime]:
    dates = [parse_date(f.get("date")) for f in normalize_followers(followers)]
    dates = [d for d in dates if d is not None]
    if not dates:
        return None
    return max(dates)


def lhb_reason_features(reasons: List[str]) -> Tuple[float, int]:
    if not reasons:
        return 40.0, 0
    strength = 45.0
    tags = set()
    for reason in reasons:
        text = str(reason)
        if "涨幅" in text or "涨停" in text:
            strength += 16
            tags.add("momentum")
        if "换手" in text:
            strength += 12
            tags.add("turnover")
        if "振幅" in text:
            strength += 8
            tags.add("range")
        if "连续" in text or "异常" in text or "偏离值" in text:
            strength += 14
            tags.add("abnormal")
        if "跌幅" in text:
            strength -= 10
            tags.add("selloff")
        if "严重异常" in text:
            strength -= 8
            tags.add("extreme")
    return clamp(strength), len(tags)


def buy_concentration_score(values: List[float]) -> Optional[float]:
    if not values:
        return None
    total = sum(values)
    if total <= 0:
        return None
    top_ratio = max(values) / total
    return sweetspot_score(top_ratio, 0.22, 0.55, 0.02, 0.92)


def hot_money_persistence_score(followers: List[Dict[str, Any]], seat_counts: Counter) -> Optional[float]:
    if not followers or not seat_counts:
        return None
    normalized = normalize_followers(followers)
    dates = {str(f.get("date"))[:10] for f in normalized if f.get("date")}
    max_repeat = max(seat_counts.values())
    repeat_score = bounded_linear(max_repeat, 1.0, 5.0)
    date_score = bounded_linear(len(dates), 1.0, 5.0)
    return average_score(repeat_score, date_score)


def normalize_followers(followers: Any) -> List[Dict[str, Any]]:
    if not isinstance(followers, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for follower in followers:
        if isinstance(follower, dict):
            normalized.append(follower)
        elif isinstance(follower, str) and follower.strip():
            normalized.append({"seat": follower.strip(), "category": "top_active_seat"})
    return normalized


def event_recency_decay_score(days: Optional[float]) -> Optional[float]:
    d = safe_float(days)
    if d is None:
        return None
    return clamp(100.0 * math.exp(-d / 4.0))


def institution_hotmoney_combo_score(
        inst_net: Optional[float],
        concurrent: Optional[float],
        lhb_net: Optional[float],
) -> Optional[float]:
    if inst_net is None and concurrent is None and lhb_net is None:
        return None
    inst_score = bounded_linear(inst_net, 0.0, 5e8) if inst_net is not None else None
    conc_score = bounded_linear(concurrent, 1.0, 5.0) if concurrent is not None else None
    lhb_score = bounded_linear(lhb_net, 0.0, 8e8) if lhb_net is not None else None
    return average_score(inst_score, conc_score, lhb_score)


def institution_conflict_score(inst_net: Optional[float], lhb_net: Optional[float]) -> Optional[float]:
    if inst_net is None and lhb_net is None:
        return None
    if inst_net is not None and inst_net < 0:
        # 机构净卖但龙虎榜整体净买，分歧弱于双向卖出。
        if lhb_net is not None and lhb_net > 0:
            return 35.0
        return 20.0
    if inst_net is not None and lhb_net is not None and inst_net > 0 and lhb_net < 0:
        return 45.0
    return 85.0 if (inst_net or 0) > 0 else 60.0


def overheat_control_score(
        chg_5d: Optional[float],
        consecutive_limit_up: Optional[float],
        rsi: Optional[float],
) -> Optional[float]:
    scores = []
    if chg_5d is not None:
        scores.append(sweetspot_score(chg_5d, -2.0, 14.0, -18.0, 38.0))
    if consecutive_limit_up is not None:
        scores.append(bounded_linear(consecutive_limit_up, 0.0, 4.0, reverse=True))
    if rsi is not None:
        scores.append(rsi_sweetspot_score(rsi))
    return average_score(*scores)


def tech_momentum_score(tech: Dict[str, Any]) -> Optional[float]:
    if not tech:
        return None
    score = 50.0
    chg_5d = safe_float(tech.get("chg_5d"))
    vol_ratio = safe_float(tech.get("vol_ratio"))
    ma_bull = bool(tech.get("ma_bull"))
    if chg_5d is not None:
        if 2 <= chg_5d <= 12:
            score += 25
        elif 0 <= chg_5d < 2:
            score += 10
        elif chg_5d > 25:
            score -= 20
        elif chg_5d < -8:
            score -= 20
    if vol_ratio is not None:
        if 1.1 <= vol_ratio <= 3.5:
            score += 15
        elif vol_ratio > 5:
            score -= 10
    if ma_bull:
        score += 10
    return clamp(score)


def short_reasons(data: Dict[str, Any], raw: Dict[str, Any]) -> List[str]:
    pick = data.get("pick") or {}
    reasons = list(pick.get("reasons") or [])[:3]
    if safe_float(raw.get("hot_money_concurrent")):
        reasons.append(f"游资共振{int(raw['hot_money_concurrent'])}")
    if safe_float(raw.get("buy_amount_total")):
        reasons.append(f"跟踪买入{raw['buy_amount_total' ] /1e8:.2f}亿")
    if safe_float(raw.get("known_hot_money_ratio")):
        reasons.append(f"知名游资占比{raw['known_hot_money_ratio' ] *100:.0f}%")
    return reasons[:5]


def short_warnings(data: Dict[str, Any], raw: Dict[str, Any]) -> List[str]:
    warnings = []
    if first_not_none(raw.get("risk_control"), 100) < 60:
        warnings.append("风险扣分偏高")
    if (safe_float(raw.get("limit_up_control")) or 0) >= 3:
        warnings.append("连板较高，注意接力失败")
    if first_not_none(raw.get("tradability"), 100) < 80:
        warnings.append("可能一字/T字板，成交质量需复核")
    if data.get("pick") and not data.get("scored"):
        warnings.append("缺少席位级followers明细")
    return warnings


def follower_sample(data: Dict[str, Any], limit: int = 6) -> List[Dict[str, Any]]:
    followers = normalize_followers((data.get("scored") or {}).get("followers") or [])
    out = []
    for follower in followers[:limit]:
        out.append({
            "seat": follower.get("seat"),
            "date": follower.get("date"),
            "category": follower.get("category"),
            "buy_est": clean_round(follower.get("buy_est"), 2),
        })
    return out


def is_st_name(name: str) -> bool:
    upper = str(name).upper()
    return "ST" in upper or "*ST" in upper or upper.startswith("S ")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_long_strategy(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = deep_merge(get_default_config()["long"], config or {})
    scoring_config = copy.deepcopy(merged)
    # Keep percentile factor scores comparable when UI controls are used as
    # hard pool constraints; otherwise kept stocks can see their scores shift
    # merely because the normalization pool changed.
    scoring_config["exclude_st"] = False
    scoring_config["min_market_cap_yi"] = 0
    scoring_config["min_listing_years"] = 0
    scoring_config["min_csi300_persistence"] = 0
    scoring_config["require_csi300"] = False
    scoring_config["require_high_drawdown"] = False
    candidates, notes = build_long_candidates(scoring_config)
    scored = apply_scores(candidates, LONG_FACTORS, merged.get("weights", {}))
    scored = rerank_scored([item for item in scored if passes_long_hard_filters(item, merged)])
    candidate_codes = {item["code"] for item in scored}
    candidates = [item for item in candidates if item["code"] in candidate_codes]
    min_score = safe_float(merged.get("min_score")) or 0.0
    selected = [item for item in scored if item["score"] >= min_score]
    selected = selected[: max(0, int(first_not_none(merged.get("top_n"), 30)))]
    return {
        "strategy": "long",
        "title": "Long horizon CSI300 compounder strategy",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": merged,
        "factor_count": len(LONG_FACTORS),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "notes": notes + long_data_notes(candidates),
        "picks": strip_internal(selected),
        "diagnostics": diagnostics(scored, LONG_FACTORS),
    }


def run_short_strategy(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = deep_merge(get_default_config()["short"], config or {})
    candidates, notes = build_short_candidates(merged)
    scored = apply_scores(candidates, SHORT_FACTORS, merged.get("weights", {}))
    min_score = safe_float(merged.get("min_score")) or 0.0
    selected = [item for item in scored if item["score"] >= min_score]
    selected = selected[: max(0, int(first_not_none(merged.get("top_n"), 30)))]
    return {
        "strategy": "short",
        "title": "Short horizon Dragon Tiger List hot-money strategy",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": merged,
        "factor_count": len(SHORT_FACTORS),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "notes": notes + short_data_notes(candidates),
        "picks": strip_internal(selected),
        "diagnostics": diagnostics(scored, SHORT_FACTORS),
    }


def run_strategies(config: Optional[Dict[str, Any]] = None, persist: bool = False) -> Dict[str, Any]:
    merged = deep_merge(get_default_config(), config or {})
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "factor_total": len(LONG_FACTORS) + len(SHORT_FACTORS),
        "factor_counts": {
            "long": len(LONG_FACTORS),
            "short": len(SHORT_FACTORS),
            "total": len(LONG_FACTORS) + len(SHORT_FACTORS),
        },
        "factor_registry": [spec.to_dict() for spec in LONG_FACTORS + SHORT_FACTORS],
        "long": run_long_strategy(merged.get("long", {})) if merged.get("long", {}).get("enabled", True) else None,
        "short": run_short_strategy(merged.get("short", {})) if merged.get("short", {}).get("enabled", True) else None,
        "self_review": self_review_summary(),
    }
    optimized_meta = merged.get("_optimized_defaults")
    if isinstance(optimized_meta, dict):
        note = (
            f"默认参数来自 {optimized_meta.get('generated_at') or '未知时间'} 的本地参数搜索"
            f"（代理回测，存在过拟合/前视风险）：{optimized_meta.get('caveat') or ''}"
        ).rstrip("：")
        for key in ("long", "short"):
            if payload.get(key):
                payload[key]["notes"].append(note)
    if persist:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
    return payload


def get_default_config() -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    optimized = load_json_optional(OPTIMIZED_CONFIG_FILE, {})
    if isinstance(optimized, dict) and optimized.get("config"):
        config = deep_merge(config, optimized["config"])
        config["_optimized_defaults"] = {
            "source": str(OPTIMIZED_CONFIG_FILE),
            "generated_at": optimized.get("generated_at"),
            "iterations_per_strategy": optimized.get("iterations_per_strategy"),
            "seed": optimized.get("seed"),
            "caveat": optimized.get("caveat"),
        }
    return config


def get_factor_registry() -> Dict[str, Any]:
    return {
        "long": [spec.to_dict() for spec in LONG_FACTORS],
        "short": [spec.to_dict() for spec in SHORT_FACTORS],
        "total": len(LONG_FACTORS) + len(SHORT_FACTORS),
    }


def strip_internal(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for item in items:
        row = {
            "rank": item.get("rank"),
            "code": item.get("code"),
            "name": item.get("name"),
            "strategy": item.get("strategy"),
            "horizon": item.get("horizon"),
            "industry": item.get("industry"),
            "score": item.get("score"),
            "data_quality": item.get("data_quality"),
            "reasons": item.get("reasons", []),
            "warnings": item.get("warnings", []),
            "raw_factors": item.get("raw_factors", {}),
            "factor_scores": item.get("factor_scores", {}),
        }
        if item.get("followers"):
            row["followers"] = item["followers"]
        if item.get("sources"):
            row["sources"] = item["sources"]
        out.append(row)
    return out


def diagnostics(scored: List[Dict[str, Any]], specs: List[FactorSpec]) -> Dict[str, Any]:
    if not scored:
        return {"score_range": None, "top_factor_coverage": [], "groups": {}}
    scores = [item["score"] for item in scored]
    groups: Dict[str, float] = defaultdict(float)
    for spec in specs:
        groups[spec.group] += spec.default_weight
    return {
        "score_range": [round(min(scores), 2), round(max(scores), 2)],
        "avg_score": round(mean(scores), 2),
        "avg_data_quality": round(mean(item.get("data_quality", 0) for item in scored), 3),
        "top_factor_coverage": factor_coverage(scored[: min(20, len(scored))], specs),
        "groups": {k: round(v, 3) for k, v in sorted(groups.items())},
    }


def factor_coverage(items: List[Dict[str, Any]], specs: List[FactorSpec]) -> List[Dict[str, Any]]:
    rows = []
    if not items:
        return rows
    for spec in specs:
        present = sum(1 for item in items if safe_float(item["raw_factors"].get(spec.key)) is not None)
        rows.append({
            "key": spec.key,
            "label": spec.label,
            "coverage": round(present / len(items), 3),
        })
    rows.sort(key=lambda row: (row["coverage"], row["key"]))
    return rows


def long_data_notes(candidates: List[Dict[str, Any]]) -> List[str]:
    notes = []
    if candidates:
        avg_quality = mean(item.get("data_quality", 0) for item in candidates)
        if avg_quality < 0.7:
            notes.append("Long strategy data coverage is below 70%; enrich financial/price history for production use.")
    notes.append("CSI300 long-term membership is proxied unless historical constituent snapshots are added.")
    return notes


def short_data_notes(candidates: List[Dict[str, Any]]) -> List[str]:
    notes = []
    if candidates and all("缺少席位级followers明细" in item.get("warnings", []) for item in candidates[:5]):
        notes.append \
            ("Top short picks mainly come from aggregate capital picks; run capital snapshots for finer seat tracking.")
    return notes


def self_review_summary() -> Dict[str, Any]:
    return {
        "factor_count_ok": len(LONG_FACTORS) + len(SHORT_FACTORS) >= 20,
        "long_factor_count": len(LONG_FACTORS),
        "short_factor_count": len(SHORT_FACTORS),
        "old_files_modified": False,
        "network_required": False,
        "known_limits": [
            "Historical CSI300 membership is exact only if historical constituent files are later supplied.",
            "Long-horizon excess-return validation needs multi-year daily prices; current local fundamental files mostly contain recent daily samples.",
            "Short strategy uses the latest saved Dragon Tiger List snapshots; refresh those files before live trading.",
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_cli_summary(result: Dict[str, Any], top: int = 10) -> None:
    print(f"Generated: {result['generated_at']}")
    print(f"Total factors: {result['factor_total']}")
    for key in ("long", "short"):
        section = result.get(key)
        if not section:
            continue
        print()
        print("=" * 88)
        print(f"{section['title']} · candidates={section['candidate_count']} selected={section['selected_count']}")
        print("=" * 88)
        for note in section.get("notes", []):
            print(f"[NOTE] {note}")
        for item in section.get("picks", [])[:top]:
            reasons = " | ".join(item.get("reasons", [])[:3])
            print(f"{item['rank']:>3}. {item['code']} {item['name']:<10} score={item['score']:>6.2f}  {reasons}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced A-share long/short strategy engine")
    parser.add_argument("--strategy", choices=["all", "long", "short"], default="all")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--persist", action="store_true", help=f"write {OUTPUT_FILE}")
    parser.add_argument("--json", action="store_true", help="print full JSON")
    args = parser.parse_args()

    config = get_default_config()
    if args.strategy == "long":
        config["short"]["enabled"] = False
    elif args.strategy == "short":
        config["long"]["enabled"] = False

    result = run_strategies(config, persist=args.persist)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_cli_summary(result, top=args.top)
        if args.persist:
            print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

