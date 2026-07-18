
"""
Advanced A-share strategy engine.

This module is intentionally self-contained and read-only against existing
project data, scores three strategy families, and can be used by both CLI and
the dashboard:

1. Long horizon: quality compounders, intended for 2-5 years.
2. Small-cap: the short/radar universe scored with the long-factor framework.
3. Short horizon: the radar's stable hot-money small-cap universe enriched by
   Dragon Tiger List / seat signals, intended for 1-5 trading days.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math

import numpy as np
import threading
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import stock_storage
from stock_crawl_common import daily_stats_from_history_records


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
META_DATA_BACKUP_DIR = ROOT / "meta_data_backup"
CAPITAL_DIR = DATA_DIR / "capital"
OUTPUT_FILE = DATA_DIR / "stock_advanced_strategy_results.json"
OPTIMIZED_CONFIG_FILE = DATA_DIR / "stock_strategy_optimized_config.json"
OPTIMIZED_CONFIG_BACKUP_FILE = META_DATA_BACKUP_DIR / "stock_strategy_optimized_config.json"
SMALLCAP_OPTIMIZED_CONFIG_FILE = DATA_DIR / "stock_strategy_smallcap_optimized_config.json"
SMALLCAP_OPTIMIZED_CONFIG_BACKUP_FILE = META_DATA_BACKUP_DIR / "stock_strategy_smallcap_optimized_config.json"
LIVE_CANDIDATE_CACHE_FILE = DATA_DIR / "stock_strategy_candidate_cache.json"
LIVE_CANDIDATE_CACHE_VERSION = 5
SHORT_UNIVERSE_VERSION = "hotmoney_small_cap_v1"
SMALLCAP_UNIVERSE_VERSION = SHORT_UNIVERSE_VERSION
LONG_LIQUIDITY_FACTOR_VERSION = "avg_turnover_rate_v1"
LONG_DIVIDEND_FACTOR_VERSION = "rolling_realized_5y_v2"
LONG_CAPITAL_EVENT_DAYS = 90
HOLDER_TABLE = "shareholder_count"
REPURCHASE_TABLE = "repurchase"
LHB_ALL_TABLE = "lhb_all"


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
    FactorSpec("csi300_current", "当前沪深300", "long", "指数约束", "现仍在沪深300成分池。", 0.0, "boolean"),
    FactorSpec("csi300_persistence", "沪深300稳定代理", "long", "指数约束", "用当前沪深300、中证大盘池近似长期稳定成分。", 0.0, "direct"),
    FactorSpec("market_cap", "总市值", "long", "规模流动性", "偏好大盘股。", 0.0, "percentile"),
    FactorSpec("size_reversal", "小市值因子", "long", "规模流动性", "SMB代理：总市值越小得分越高，给中小盘优质股入场机会。", 0.0, "percentile", "low", missing_score=40),
    FactorSpec("liquidity", "日均换手率", "long", "规模流动性", "偏好换手更活跃、且不直接偏向大市值的股票。", 0.6, "percentile"),
    FactorSpec("low_volatility", "低波动", "long", "风险", "年化波动率越低越好。", 0.8, "percentile", "low", missing_score=40),
    FactorSpec("roe_mean", "ROE均值", "long", "质量", "近年ROE均值。", 1.2, "percentile", missing_score=20),
    FactorSpec("roe_stability", "ROE稳定性", "long", "质量", "ROE均值减标准差。", 1.1, "percentile", missing_score=20),
    FactorSpec("revenue_growth", "营收增速", "long", "成长", "TTM营收同比。", 0.8, "percentile", missing_score=35),
    FactorSpec("net_profit_growth", "利润增速", "long", "成长", "TTM净利润同比。", 0.9, "percentile", missing_score=35),
    FactorSpec("gross_margin", "毛利率", "long", "盈利", "销售毛利率。", 0.5, "percentile", missing_score=35),
    FactorSpec("net_margin", "净利率", "long", "盈利", "销售净利率。", 0.8, "percentile", missing_score=35),
    FactorSpec("cashflow_quality", "现金流质量", "long", "质量", "经营现金流 / 净利润，1附近及以上更好。", 0.9, "bounded", "high", 0.0, 2.0, 35),
    FactorSpec(
        "dividend_yield_5y",
        "滚动五年年均股息率",
        "long",
        "股东回报",
        "截至最新交易日，过去五年已实施的每股现金分红合计 / 5 / 当前股价；未分红期间按0计。",
        0.7,
        "percentile",
        missing_score=20,
    ),
    FactorSpec("dividend_consistency", "连续分红", "long", "股东回报", "近三年连续现金分红。", 0.5, "boolean"),
    FactorSpec("holder_count_change", "股东户数变化", "long", "资金面", "最近已公告股东户数增减比例，户数下降代表筹码集中。", 0.7, "percentile", "low", missing_score=50),
    FactorSpec("repurchase_recent", "公司回购", "long", "资金面", f"近{LONG_CAPITAL_EVENT_DAYS}日有回购公告。", 0.4, "bounded", "high", 0.0, 1.0, 50),
    FactorSpec("lhb_recent_avoid", "近期龙虎榜", "long", "资金面", f"近{LONG_CAPITAL_EVENT_DAYS}日上过龙虎榜按避雷处理；长线默认权重为0且不参与参数搜索。", 0.0, "bounded", "low", 0.0, 1.0, 50),
    FactorSpec("debt_safety", "低负债率", "long", "风险", "资产负债率越低越好，银行等行业会由其他质量因子平衡。", 0.5, "percentile", "low", missing_score=45),
    FactorSpec("pledge_safety", "低质押", "long", "风险", "股权质押比例越低越好。", 0.4, "percentile", "low", missing_score=60),
    FactorSpec("industry_leadership", "行业规模地位", "long", "规模流动性", "行业内市值百分位。", 0.0, "direct", missing_score=40),
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
    FactorSpec("resonance_score", "共振强度分", "short", "资金评分", "由共买席位数折算的资金共振强度。", 0.8, "direct", missing_score=40),
    FactorSpec("momentum_score", "短线动量", "short", "技术", "由近期涨幅、量比和均线形态生成。", 0.8, "direct", missing_score=40),
    FactorSpec("position_score", "短线位置", "short", "技术", "由MA20乖离和RSI甜区生成。", 0.6, "direct", missing_score=40),
    FactorSpec("risk_control", "风险控制", "short", "风控", "由过热、可交易性和RSI位置合成。", 0.9, "direct", missing_score=45),
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


# 小盘策略复用长线的财务/价量因子，但其股票池已经由游资小盘池限定，以下规模与
# 指数约束因子不再重复参与展示、评分或优化。
SMALLCAP_EXCLUDED_FACTOR_KEYS = {
    "csi300_current",
    "csi300_persistence",
    "market_cap",
    "size_reversal",
}
SMALLCAP_DEFAULT_WEIGHT_OVERRIDES = {
    # 本次只关闭长线页的自动搜索；小盘页维持原有默认权重与搜索口径。
    "lhb_recent_avoid": 0.5,
}
SMALLCAP_FACTORS: List[FactorSpec] = [
    replace(
        spec,
        strategy="smallcap",
        default_weight=SMALLCAP_DEFAULT_WEIGHT_OVERRIDES.get(
            spec.key, spec.default_weight
        ),
    )
    for spec in LONG_FACTORS
    if spec.key not in SMALLCAP_EXCLUDED_FACTOR_KEYS
]


LONG_PRICE_COVERAGE_KEYS = frozenset({
    "liquidity",
    "low_volatility",
    "reversal_1m",
    "momentum_12_1",
    "dist_52w_high",
    "reversal_long_term",
    "abnormal_turnover",
})
LONG_CAPITAL_COVERAGE_KEYS = frozenset({
    "holder_count_change",
    "repurchase_recent",
    "lhb_recent_avoid",
    "pledge_safety",
})


FACTOR_REGISTRY: Dict[str, FactorSpec] = {
    spec.key: spec for spec in LONG_FACTORS + SHORT_FACTORS
}

LONG_DEFAULT_TOP_N = 20
SMALLCAP_DEFAULT_TOP_N = 10
SHORT_DEFAULT_TOP_N = 10

LONG_FIXED_ZERO_WEIGHT_KEYS = {
    "csi300_current",
    "csi300_persistence",
    "market_cap",
    "size_reversal",
    "industry_leadership",
    "lhb_recent_avoid",
}


DEFAULT_CONFIG: Dict[str, Any] = {
    "long": {
        "enabled": True,
        "use_segment_leaders": True,
        "top_n": LONG_DEFAULT_TOP_N,
        "min_score": 60,
        "min_market_cap_yi": 0,
        "require_csi300": False,
        "require_high_drawdown": False,
        "min_high_drawdown_pct": 40,
        "exclude_st": True,
        "hold_years_min": 2,
        "hold_years_max": 5,
        "weights": {
            spec.key: (
                0.0
                if spec.key in LONG_FIXED_ZERO_WEIGHT_KEYS
                else spec.default_weight
            )
            for spec in LONG_FACTORS
        },
    },
    "short": {
        "enabled": True,
        "top_n": SHORT_DEFAULT_TOP_N,
        "min_score": 55,
        "hold_days_min": 1,
        "hold_days_max": 5,
        "exclude_st": True,
        "min_lhb_count": 0,
        "min_hot_money_concurrent": 0,
        "max_consecutive_limit_up": 3,
        "weights": {spec.key: spec.default_weight for spec in SHORT_FACTORS},
    },
    "smallcap": {
        "enabled": True,
        "top_n": SMALLCAP_DEFAULT_TOP_N,
        "min_score": 60,
        "require_high_drawdown": False,
        "min_high_drawdown_pct": 40,
        "exclude_st": True,
        "hold_years_min": 2,
        "hold_years_max": 5,
        "weights": {spec.key: spec.default_weight for spec in SMALLCAP_FACTORS},
    },
}


# ---------------------------------------------------------------------------
# Safe numerics and JSON helpers
# ---------------------------------------------------------------------------


_SF_INF = float("inf")


def safe_float(value: Any) -> Optional[float]:
    # fast path：DB REAL/已转值多为 float，内联 nan/inf 比较省去 float()+math.isnan()+math.isinf()
    # 三次函数调用开销（该函数在因子计算里被调上亿次，是 profile 头号热点）。
    if type(value) is float:
        return value if (value == value and value != _SF_INF and value != -_SF_INF) else None
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num or num == _SF_INF or num == -_SF_INF:
        return None
    return num


_PRICE_NUM_FIELDS = ("close", "open", "high", "low", "turnover_rate", "amount", "volume", "change_pct")
_PRICE_ROW_CACHE_MAXSIZE = 2048
_price_row_cache_lock = threading.RLock()
# Lists cannot be weak-referenced.  Keep the object itself beside its id so an
# id reused by CPython can never hit an entry that belongs to a released list.
# Both caches are bounded LRU maps because retaining the row object also keeps
# its full history alive.
_pretransformed_rows: "OrderedDict[int, List[Dict[str, Any]]]" = OrderedDict()
_price_arr_cache: "OrderedDict[int, Tuple[List[Dict[str, Any]], Dict[str, Any]]]" = OrderedDict()


def _bounded_row_cache_put(cache: OrderedDict, key: int, value: Any) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _PRICE_ROW_CACHE_MAXSIZE:
        cache.popitem(last=False)


def _clear_price_row_caches() -> None:
    with _price_row_cache_lock:
        _pretransformed_rows.clear()
        _price_arr_cache.clear()


def pretransform_price_rows(
    rows: List[Dict[str, Any]], *, cache: bool = False
) -> None:
    """全历史日线行的数值字段原地转 float/None 一次。

    因子计算里 `safe_float(row.get("close"))` 这类列表推导被调上亿次（profile 头号热点），
    且每个回测折(fold)对同一份全历史重复转换。rows 来自 cn_index/stock history 等长存活模块缓存、
    跨 fold/trial 复用同一批 row 对象，故只需预转一次，之后下游 safe_float 全走 float fast-path。
    ``cache=True`` 只用于模块内部从版本化数据快照加载、之后不再原地修改的列表。
    默认不缓存，因此通用调用者追加/修改行后再次调用仍会重新转换。缓存同时校验
    对象身份，不能只信任会被解释器复用的 ``id(rows)``。"""
    if not rows:
        return
    rid = id(rows)
    if cache:
        with _price_row_cache_lock:
            cached_rows = _pretransformed_rows.get(rid)
            if cached_rows is rows:
                _pretransformed_rows.move_to_end(rid)
                return
    for r in rows:
        for k in _PRICE_NUM_FIELDS:
            v = r.get(k)
            if v is not None and type(v) is not float:
                r[k] = safe_float(v)
    if cache:
        with _price_row_cache_lock:
            _bounded_row_cache_put(_pretransformed_rows, rid, rows)


def _build_price_arrays(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    close = np.empty(n, dtype=np.float64)
    turn = np.empty(n, dtype=np.float64)
    for i, r in enumerate(rows):
        c = r.get("close")
        cv = c if type(c) is float else safe_float(c)
        close[i] = np.nan if cv is None else cv
        t = first_not_none(r.get("turnover_rate"), r.get("daily_turnover_rate"))
        tv = t if type(t) is float else safe_float(t)
        turn[i] = np.nan if tv is None else tv
    return {"close": close, "turnover_rate": turn}


def price_arrays(
    rows: List[Dict[str, Any]], *, cache: bool = False
) -> Dict[str, Any]:
    """全历史 close/turnover_rate 的 numpy 数组(脏值→nan)，按对象身份缓存一次。

    供因子计算向量化(numpy 运算释放 GIL，解锁 optuna n_jobs 多线程)+各 fold 切前缀复用，
    替代每 fold 对全历史逐行 safe_float(profile 头号热点)。通用调用默认不缓存，确保
    同一列表原地变化后不会返回旧数组；模块内部只对不可变的数据快照显式启用缓存。"""
    rid = id(rows)
    if cache:
        with _price_row_cache_lock:
            entry = _price_arr_cache.get(rid)
            if entry is not None and entry[0] is rows:
                _price_arr_cache.move_to_end(rid)
                return entry[1]
    arrays = _build_price_arrays(rows)
    if cache:
        with _price_row_cache_lock:
            _bounded_row_cache_put(_price_arr_cache, rid, (rows, arrays))
    return arrays


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


@lru_cache(maxsize=131072)
def _parse_date_text(text: str) -> Optional[datetime]:
    if len(text) < 10:
        return None
    text = text[:10]
    try:
        # Hot path for YYYY-MM-DD avoids datetime.strptime's locale machinery.
        if text[4] == "-" and text[7] == "-":
            return datetime(int(text[:4]), int(text[5:7]), int(text[8:10]))
        return datetime.strptime(text, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def parse_date(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    return _parse_date_text(str(value)[:10])


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


_candidate_cache_lock = threading.Lock()


def invalidate_dir_fingerprints() -> None:
    # 清掉股票数据与派生候选的进程内 lru_cache，使下次访问从
    # data/stock_data.sqlite3 重新载入（数据刷新后由 app 调用）。
    # 名字沿用历史接口（app/services 依赖），实质是清 DB 数据缓存。
    for cached in (
        _load_cn_stock_index_cached,
        _load_fundamental_stocks_cached,
        _load_long_capital_signals_cached,
        _build_live_long_candidates_cached,
        _build_live_smallcap_candidates_cached,
        _build_live_short_candidates_cached,
    ):
        try:
            cached.cache_clear()
        except NameError:
            pass
    _clear_price_row_caches()


def load_json_optional(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return load_json_file(str(path))
    except (OSError, json.JSONDecodeError):
        return default


def load_optimized_config_payload() -> Tuple[Dict[str, Any], Optional[Path]]:
    for path in (OPTIMIZED_CONFIG_FILE, OPTIMIZED_CONFIG_BACKUP_FILE):
        payload = load_json_optional(path, {})
        if isinstance(payload, dict) and payload.get("config"):
            return payload, path
    return {}, None


def load_smallcap_optimized_config_payload() -> Tuple[Dict[str, Any], Optional[Path]]:
    """小盘权重独立回退：本地优化结果 → meta_data_backup → 代码默认值。"""
    for path in (SMALLCAP_OPTIMIZED_CONFIG_FILE, SMALLCAP_OPTIMIZED_CONFIG_BACKUP_FILE):
        payload = load_json_optional(path, {})
        if isinstance(payload, dict) and payload.get("config"):
            return payload, path
    return {}, None


def normalize_loaded_short_horizon(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize new optimized horizons and discard unrecoverable legacy ranges.

    Older optimizer files saved independently searched min/max labels but
    dropped the horizon used by the objective.  There is no honest way to
    reconstruct that selected value, so legacy labels fall back to the public
    1-5 day strategy default. New files carry ``hold_days`` and are forced to
    the same value in all three fields.
    """
    normalized = copy.deepcopy(config)
    selected_num = safe_float(normalized.get("hold_days"))
    selected = (
        int(selected_num)
        if selected_num is not None and selected_num.is_integer()
        else None
    )
    if selected is not None and 1 <= selected <= 5:
        normalized["hold_days"] = selected
        normalized["hold_days_min"] = selected
        normalized["hold_days_max"] = selected
    else:
        normalized.pop("hold_days", None)
        normalized.pop("hold_days_min", None)
        normalized.pop("hold_days_max", None)
    return normalized


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


def stable_config_key(config: Dict[str, Any], *, ignore: Iterable[str] = ()) -> str:
    trimmed = copy.deepcopy(config or {})
    for key in ignore:
        trimmed.pop(key, None)
    return json.dumps(trimmed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_from_key(key: str) -> Dict[str, Any]:
    return json.loads(key) if key else {}


def read_live_candidate_cache(section: str, meta: Dict[str, Any]) -> Optional[Tuple[List[Dict[str, Any]], List[str]]]:
    with _candidate_cache_lock:
        payload = load_json_optional(LIVE_CANDIDATE_CACHE_FILE, {})
        if not isinstance(payload, dict) or payload.get("version") != LIVE_CANDIDATE_CACHE_VERSION:
            return None
        entry = payload.get(section)
        if not isinstance(entry, dict) or entry.get("meta") != meta:
            return None
        candidates = entry.get("candidates")
        if not isinstance(candidates, list):
            return None
        notes = entry.get("notes")
        return copy.deepcopy(candidates), list(notes if isinstance(notes, list) else [])


def write_live_candidate_cache(
        section: str,
        meta: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        notes: List[str],
) -> None:
    with _candidate_cache_lock:
        payload = load_json_optional(LIVE_CANDIDATE_CACHE_FILE, {})
        if not isinstance(payload, dict) or payload.get("version") != LIVE_CANDIDATE_CACHE_VERSION:
            payload = {"version": LIVE_CANDIDATE_CACHE_VERSION}
        payload[section] = {
            "meta": meta,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "candidate_count": len(candidates),
            "candidates": candidates,
            "notes": notes,
        }
        LIVE_CANDIDATE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = LIVE_CANDIDATE_CACHE_FILE.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, separators=(",", ":"))
        tmp_path.replace(LIVE_CANDIDATE_CACHE_FILE)


def clear_live_candidate_cache() -> None:
    try:
        _build_live_long_candidates_cached.cache_clear()
        _build_live_short_candidates_cached.cache_clear()
        _load_sw3_segment_map_cached.cache_clear()
        _load_long_capital_signals_cached.cache_clear()
    except NameError:
        pass
    with _candidate_cache_lock:
        try:
            LIVE_CANDIDATE_CACHE_FILE.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Financial helpers
# ---------------------------------------------------------------------------


def _dated_financial_records(
        records: List[Dict[str, Any]],
) -> List[Tuple[datetime, Dict[str, Any]]]:
    """Parse and order report periods without relying on input order/string prefixes."""
    dated: List[Tuple[datetime, Dict[str, Any]]] = []
    seen = set()
    for row in records or []:
        period = parse_date(row.get("date"))
        if period is None or period in seen:
            continue
        seen.add(period)
        dated.append((period, row))
    dated.sort(key=lambda item: item[0], reverse=True)
    return dated


def _previous_year_period(period: datetime) -> Optional[datetime]:
    try:
        return period.replace(year=period.year - 1)
    except ValueError:
        # Financial periods are normally quarter ends; keep leap-day input
        # deterministic rather than silently comparing it with an annual row.
        if period.month == 2 and period.day == 29:
            return period.replace(year=period.year - 1, day=28)
        return None


def _ttm_value_at(
        dated: List[Tuple[datetime, Dict[str, Any]]],
        field: str,
        period: datetime,
        *,
        fallback_incomplete: bool,
) -> Optional[float]:
    by_period = {row_period: row for row_period, row in dated}
    current = safe_float((by_period.get(period) or {}).get(field))
    if current is None:
        return None
    if (period.month, period.day) == (12, 31):
        return current

    previous_same = _previous_year_period(period)
    previous_fy = datetime(period.year - 1, 12, 31)
    same_value = safe_float((by_period.get(previous_same) or {}).get(field)) if previous_same else None
    fy_value = safe_float((by_period.get(previous_fy) or {}).get(field))
    if same_value is not None and fy_value is not None:
        return current + fy_value - same_value
    return current if fallback_incomplete else None


def compute_ttm(records: List[Dict[str, Any]], field: str) -> Optional[float]:
    if not records:
        return None
    dated = _dated_financial_records(records)
    if not dated:
        # Preserve the legacy fallback for non-standard source rows that have
        # no parseable report period at all.
        return safe_float(records[0].get(field))
    return _ttm_value_at(dated, field, dated[0][0], fallback_incomplete=True)


def compute_yoy_growth(records: List[Dict[str, Any]], field: str) -> Optional[float]:
    dated = _dated_financial_records(records)
    if not dated:
        return None
    latest_period = dated[0][0]
    previous_period = _previous_year_period(latest_period)
    if previous_period is None or not any(period == previous_period for period, _ in dated):
        return None
    # Growth must compare complete like-for-like TTM periods. Falling back to
    # a cumulative YTD value would quietly turn this into a different metric.
    now = _ttm_value_at(dated, field, latest_period, fallback_incomplete=False)
    prev = _ttm_value_at(dated, field, previous_period, fallback_incomplete=False)
    if prev is None or abs(prev) < 1e-9:
        return None
    if now is None:
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


def stock_history_records(stock: Dict[str, Any]) -> List[Dict[str, Any]]:
    history = stock.get("history") or {}
    records = history.get("records") if isinstance(history, dict) else None
    return records if isinstance(records, list) else []


def derived_daily_stats(stock: Dict[str, Any]) -> Dict[str, Any]:
    stats = dict((stock.get("daily") or {}).get("stats") or {})
    if stats.get("latest_daily_close") is not None or stats.get("latest_close") is not None:
        if stats.get("history_window_avg_daily_turnover_rate") is None:
            computed = daily_stats_from_history_records(stock_history_records(stock))
            turnover = computed.get("history_window_avg_daily_turnover_rate")
            if turnover is not None:
                stats["history_window_avg_daily_turnover_rate"] = turnover
        return stats
    return daily_stats_from_history_records(stock_history_records(stock))


def estimate_market_cap_cny(stock: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> Optional[float]:
    snap_cap = safe_float((snapshot or {}).get("market_cap_est"))
    if snap_cap and snap_cap > 0:
        return snap_cap

    shares = estimate_shares(stock)
    stats = derived_daily_stats(stock)
    price = first_not_none(stats.get("latest_daily_close"), stats.get("latest_close"))
    if shares and shares > 0 and price and price > 0:
        return shares * price
    return None


def dividend_yield(
        stock: Dict[str, Any],
        years: int = 5,
        as_of: Optional[str] = None,
        price: Optional[float] = None,
) -> Optional[float]:
    """滚动 years 年的年均现金股息率。

    只统计截至参考日已经实施的现金分红，优先以除权除息日判断是否落入窗口；
    老数据缺少除权除息日时，仅对“实施”记录回退到公告日。窗口内没有分红时
    返回 0，只有分红明细本身不可用时才返回 None。
    """
    if years <= 0:
        return None
    dividends = stock.get("dividends") or {}
    records = dividends.get("records") if isinstance(dividends, dict) else None
    if not isinstance(records, list) or not records:
        return None

    if as_of is None:
        stats = derived_daily_stats(stock)
        ref_price = price if price is not None else first_not_none(
            stats.get("latest_daily_close"), stats.get("latest_close")
        )
        ref_date = parse_date(stats.get("latest_trade_date")) or datetime.now()
    else:
        ref_price = price
        ref_date = parse_date(as_of)
    if ref_date is None or not ref_price or ref_price <= 0:
        return None

    try:
        cutoff = ref_date.replace(year=ref_date.year - years)
    except ValueError:
        # 2 月 29 日回看非闰年时落到 2 月 28 日。
        cutoff = ref_date.replace(year=ref_date.year - years, day=28)

    has_visible_history = False
    dividend_per_share_sum = 0.0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        announce_date = parse_date(rec.get("announce_date"))
        ex_date = parse_date(rec.get("ex_date"))
        visible_date = announce_date or ex_date
        if visible_date is None or visible_date > ref_date:
            continue
        has_visible_history = True

        dividend_per_10 = safe_float(rec.get("dividend_per_10"))
        if dividend_per_10 is None or dividend_per_10 <= 0:
            continue
        if ex_date is not None:
            event_date = ex_date
        elif "实施" in str(rec.get("progress") or ""):
            event_date = announce_date
        else:
            # 预案或尚未实施的现金分红不计入已实现股东回报。
            continue
        if event_date is not None and cutoff <= event_date <= ref_date:
            dividend_per_share_sum += dividend_per_10 / 10.0

    if not has_visible_history:
        return None
    return (dividend_per_share_sum / years) / ref_price


def consecutive_dividend_asof(stock: Dict[str, Any], as_of: str, years: int = 3) -> bool:
    """PIT：截止 as_of，近 years 个公告年是否都有现金分红。"""
    paid_years = set()
    for rec in stock.get("dividends", {}).get("records", []) or []:
        ann = str(rec.get("announce_date") or "")[:10]
        div = safe_float(rec.get("dividend_per_10"))
        if len(ann) == 10 and ann <= as_of and div and div > 0:
            paid_years.add(int(ann[:4]))
    base = int(as_of[:4])
    return all((base - i) in paid_years for i in range(1, years + 1))


def high_to_latest_drawdown_pct(rows: List[Dict[str, Any]]) -> Optional[float]:
    closes = [safe_float(row.get("close")) for row in rows or []]
    closes = [c for c in closes if c is not None and c > 0]
    if len(closes) < 2:
        return None
    peak = max(closes)
    if peak <= 0:
        return None
    return (peak - closes[-1]) / peak * 100.0


# ---------------------------------------------------------------------------
# Point-in-time (PIT) helpers — 回测时只用某时点已可见的数据，消除前视偏差。
# 所有 as_of=None 分支必须与改动前逐字等价，保护实盘/dashboard 路径。
# ---------------------------------------------------------------------------


@lru_cache(maxsize=131072)
def report_available_date(period: Optional[str]) -> Optional[str]:
    """报告期 → 法定披露截止日(保守取，避免前视；实际公告通常更早)。
    年报(12-31)→次年4-30、一季报(03-31)→4-30、半年报(06-30)→8-31、三季报(09-30)→10-31。"""
    d = parse_date(period)
    if not d:
        return None
    md = f"{d.month:02d}-{d.day:02d}"
    if md == "12-31":
        return f"{d.year + 1}-04-30"
    if md == "03-31":
        return f"{d.year}-04-30"
    if md == "06-30":
        return f"{d.year}-08-31"
    if md == "09-30":
        return f"{d.year}-10-31"
    return (d + timedelta(days=120)).strftime("%Y-%m-%d")  # 非标准报告期兜底


def reports_available_asof(
        records: List[Dict[str, Any]], as_of: Optional[str]
) -> List[Dict[str, Any]]:
    """财报记录按可见日 <= as_of 过滤；as_of=None 原样返回。"""
    if as_of is None:
        return records
    return [
        r for r in records or []
        if (report_available_date(str(r.get("date", ""))) or "9999-99-99") <= as_of
    ]


def price_rows_asof(
        rows: List[Dict[str, Any]], as_of: Optional[str]
) -> List[Dict[str, Any]]:
    """日线切到 date <= as_of；as_of=None 原样返回。"""
    if as_of is None:
        return rows
    if not rows:
        return []
    first_date = str(rows[0].get("date", ""))
    last_date = str(rows[-1].get("date", ""))
    if first_date <= last_date:
        if last_date <= as_of:
            return rows
        lo = 0
        hi = len(rows)
        while lo < hi:
            mid = (lo + hi) // 2
            if str(rows[mid].get("date", "")) <= as_of:
                lo = mid + 1
            else:
                hi = mid
        return rows[:lo]
    return [r for r in rows if str(r.get("date", "")) <= as_of]


def annualized_vol_from_rows(
        rows: List[Dict[str, Any]], window: int = 250
) -> Optional[float]:
    """日线收盘 → 年化波动率(日收益 std × sqrt(252))。"""
    closes = [safe_float(r.get("close")) for r in (rows or [])[-(window + 1):]]
    closes = [c for c in closes if c is not None and c > 0]
    if len(closes) < 20:
        return None
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    return stdev_or_zero(rets) * math.sqrt(252)


def avg_turnover_rate_from_rows(
        rows: List[Dict[str, Any]], window: int = 60
) -> Optional[float]:
    """PIT 日均换手率(%) = 近窗平均换手率，不再乘总市值。"""
    tos = [
        safe_float(first_not_none(r.get("turnover_rate"), r.get("daily_turnover_rate")))
        for r in (rows or [])[-window:]
    ]
    tos = [t for t in tos if t is not None and t > 0]
    if not tos:
        return None
    return sum(tos) / len(tos)


# ── numpy 向量化版（与上面 rows 版逐字等价，供 optimizer 走缓存数组路径，释放 GIL）──
def _drawdown_np(close_full: "np.ndarray", asof_len: int) -> Optional[float]:
    """= high_to_latest_drawdown_pct：全历史(切到asof)过滤>0后 (peak-last)/peak*100。"""
    ca = close_full[:asof_len]
    ca = ca[~np.isnan(ca) & (ca > 0.0)]
    if ca.size < 2:
        return None
    peak = float(ca.max())
    if peak <= 0:
        return None
    return (peak - float(ca[-1])) / peak * 100.0


def _vol_np(close_full: "np.ndarray", asof_len: int, window: int = 250) -> Optional[float]:
    """= annualized_vol_from_rows：近 window+1 收盘(先切窗再过滤>0)日收益 std×sqrt(252)。"""
    w = close_full[:asof_len][-(window + 1):]
    w = w[~np.isnan(w) & (w > 0.0)]
    if w.size < 20:
        return None
    rets = (w[1:] / w[:-1] - 1.0)
    return stdev_or_zero(rets.tolist()) * math.sqrt(252)   # 复用原 std 函数保证口径一致


def _avg_turnover_rate_np(turn_full: "np.ndarray", asof_len: int, window: int = 60) -> Optional[float]:
    """= avg_turnover_rate_from_rows：近 window 换手率(先切窗再过滤>0)均值。"""
    w = turn_full[:asof_len][-window:]
    w = w[~np.isnan(w) & (w > 0.0)]
    if w.size == 0:
        return None
    return float(w.mean())


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


def latest_available_value(records: List[Dict[str, Any]], field: str) -> Optional[float]:
    """Return the newest non-null reported value, independent of input ordering."""
    dated = _dated_financial_records(records)
    rows = [row for _period, row in dated] if dated else list(records or [])
    for row in rows:
        value = safe_float(row.get(field))
        if value is not None:
            return value
    return None


def yoy_from_latest_and_annual(records: List[Dict[str, Any]], field: str) -> Optional[float]:
    """Point-in-time YoY for balance-sheet fields, matched to the same period."""
    dated = _dated_financial_records(records)
    if not dated:
        return None
    latest_period, latest_row = dated[0]
    previous_period = _previous_year_period(latest_period)
    by_period = {period: row for period, row in dated}
    latest = safe_float(latest_row.get(field))
    base = safe_float((by_period.get(previous_period) or {}).get(field)) if previous_period else None
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


def _signature_key(signature: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((str(k), v) for k, v in (signature or {}).items()))


def load_sw3_segment_map() -> Dict[str, Dict[str, Any]]:
    """一次性加载个股到申万二级/三级行业映射；按 sw3 表签名缓存，避免逐股查库。"""
    signature = _signature_key(stock_storage.sw3_signature())
    with _sw3_segment_load_lock:
        cached = _load_sw3_segment_map_cached(signature)
    return {code: dict(info) for code, info in cached.items()}


@lru_cache(maxsize=4)
def _load_sw3_segment_map_cached(
        signature: Tuple[Tuple[str, Any], ...],
) -> Dict[str, Dict[str, Any]]:
    _ = signature
    conn = stock_storage.connect()
    try:
        return stock_storage.segment_map(conn)
    finally:
        conn.close()


def load_segment_leader_codes() -> set:
    """当前 `stock_crawl_segment_leaders.py` 选出的 SW3 细分龙头 code 集。"""
    conn = stock_storage.connect()
    try:
        codes = set()
        for item in stock_storage.leader_members(conn):
            code = str(item.get("code") or "").strip()
            if code:
                codes.add(code.zfill(6))
        return codes
    finally:
        conn.close()


def load_hot_money_codes() -> set:
    """与短线策略、游资雷达共用的固定游资小盘池。"""
    conn = stock_storage.connect()
    try:
        return {
            str(item.get("code") or "").zfill(6)
            for item in stock_storage.pool_members(conn, "hotmoney")
            if item.get("code")
        }
    finally:
        conn.close()


def segment_leader_code_signature() -> Dict[str, Any]:
    """细分龙头集合签名，供候选池磁盘缓存感知 is_leader 变化。"""
    codes = sorted(load_segment_leader_codes())
    digest = hashlib.sha256(",".join(codes).encode("utf-8")).hexdigest()[:16]
    return {"count": len(codes), "digest": digest}


def _db_table_exists(conn, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _db_table_columns(conn, table: str) -> set:
    if not _db_table_exists(conn, table):
        return set()
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _latest_stock_history_date(conn) -> Optional[str]:
    if not _db_table_exists(conn, "stock_history"):
        return None
    row = conn.execute(
        "SELECT MAX(date) AS latest_date FROM stock_history WHERE daily_close IS NOT NULL"
    ).fetchone()
    return str(row["latest_date"]) if row and row["latest_date"] else None


def long_capital_signal_signature() -> Dict[str, Any]:
    tables = {
        "holder": (HOLDER_TABLE, "disclose_date"),
        "repurchase": (REPURCHASE_TABLE, "disclose_date"),
        "lhb": (LHB_ALL_TABLE, "date"),
    }
    conn = stock_storage.connect()
    try:
        sig: Dict[str, Any] = {}
        for key, (table, date_col) in tables.items():
            if not _db_table_exists(conn, table):
                sig[f"{key}_rows"] = 0
                sig[f"{key}_min_date"] = None
                sig[f"{key}_max_date"] = None
                sig[f"{key}_max_updated_at"] = None
                continue
            columns = _db_table_columns(conn, table)
            min_expr = f"MIN({date_col})" if date_col in columns else "NULL"
            max_expr = f"MAX({date_col})" if date_col in columns else "NULL"
            updated_expr = "MAX(updated_at)" if "updated_at" in columns else "NULL"
            row = conn.execute(
                f"SELECT COUNT(*) AS c, {min_expr} AS min_date, {max_expr} AS max_date, "
                f"{updated_expr} AS max_updated FROM {table}"
            ).fetchone()
            sig[f"{key}_rows"] = int(row["c"] or 0) if row else 0
            sig[f"{key}_min_date"] = row["min_date"] if row else None
            sig[f"{key}_max_date"] = row["max_date"] if row else None
            sig[f"{key}_max_updated_at"] = row["max_updated"] if row else None
        return sig
    finally:
        conn.close()


def load_long_capital_signals(as_of: Optional[str] = None) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, bool]]:
    signature = _signature_key(long_capital_signal_signature())
    signals, availability = _load_long_capital_signals_cached(as_of or "", signature)
    return {code: dict(info) for code, info in signals.items()}, dict(availability)


@lru_cache(maxsize=256)
def _load_long_capital_signals_cached(
        as_of_key: str,
        signature: Tuple[Tuple[str, Any], ...],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, bool]]:
    _ = signature
    conn = stock_storage.connect()
    try:
        asof = as_of_key or _latest_stock_history_date(conn) or datetime.now().strftime("%Y-%m-%d")
        asof = str(asof)[:10]
        try:
            cutoff = (
                datetime.strptime(asof, "%Y-%m-%d") - timedelta(days=LONG_CAPITAL_EVENT_DAYS)
            ).strftime("%Y-%m-%d")
        except ValueError:
            cutoff = (datetime.now() - timedelta(days=LONG_CAPITAL_EVENT_DAYS)).strftime("%Y-%m-%d")

        availability = {
            "holder": _db_table_exists(conn, HOLDER_TABLE) and stock_storage.table_count(conn, HOLDER_TABLE) > 0,
            "repurchase": _db_table_exists(conn, REPURCHASE_TABLE) and stock_storage.table_count(conn, REPURCHASE_TABLE) > 0,
            "lhb": _db_table_exists(conn, LHB_ALL_TABLE) and stock_storage.table_count(conn, LHB_ALL_TABLE) > 0,
        }
        signals: Dict[str, Dict[str, Any]] = {}

        def slot(code: str) -> Dict[str, Any]:
            return signals.setdefault(code, {
                "holder_change": None,
                "repurchase_recent": False,
                "lhb_recent": False,
            })

        if availability["holder"]:
            for row in conn.execute(
                f"SELECT code, change_pct FROM {HOLDER_TABLE} "
                "WHERE disclose_date IS NOT NULL AND change_pct IS NOT NULL AND disclose_date <= ? "
                "ORDER BY code, disclose_date",
                (asof,),
            ).fetchall():
                code = stock_storage._normalize_code(row["code"])
                if code:
                    slot(code)["holder_change"] = safe_float(row["change_pct"])
        if availability["repurchase"]:
            for row in conn.execute(
                f"SELECT DISTINCT code FROM {REPURCHASE_TABLE} "
                "WHERE disclose_date > ? AND disclose_date <= ?",
                (cutoff, asof),
            ).fetchall():
                code = stock_storage._normalize_code(row["code"])
                if code:
                    slot(code)["repurchase_recent"] = True
        if availability["lhb"]:
            for row in conn.execute(
                f"SELECT DISTINCT code FROM {LHB_ALL_TABLE} WHERE date > ? AND date <= ?",
                (cutoff, asof),
            ).fetchall():
                code = stock_storage._normalize_code(row["code"])
                if code:
                    slot(code)["lhb_recent"] = True
        return signals, availability
    finally:
        conn.close()


_UNKNOWN_INDUSTRY_NAMES = {"", "UNKNOWN", "unknown", "None", "none", "NULL", "null", "nan", "NaN"}


def clean_industry_name(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text in _UNKNOWN_INDUSTRY_NAMES else text


def industry_label_from_sw3(
        segment_info: Optional[Dict[str, Any]],
        fallback: Any = "",
) -> Tuple[str, str, str, str, str]:
    """返回 (展示行业, 分桶行业, 二级行业, 三级行业, 三级代码)。"""
    segment_info = segment_info or {}
    sw2 = clean_industry_name(segment_info.get("parent_segment"))
    sw3 = clean_industry_name(segment_info.get("segment_name"))
    segment_code = clean_industry_name(segment_info.get("segment_code"))
    legacy = clean_industry_name(fallback)
    if sw2 and sw3 and sw2 != sw3:
        display = f"{sw2} / {sw3}"
    else:
        display = sw3 or sw2 or legacy
    bucket = sw3 or sw2 or legacy or "UNKNOWN"
    return display, bucket, sw2, sw3, segment_code


# DB history 行(daily_* + 估值列) → 紧凑 analysis 记录的直接重命名。
# DB 存的已是 safe_float 后的干净值，无需再过 safe_float，等价于
# analysis_records_from_history_records 但省去对百万行的逐字段转换(实测 ~24s→~3s)。
_DB_HISTORY_TO_ANALYSIS = (
    ("open", "daily_open"), ("high", "daily_high"), ("low", "daily_low"), ("close", "daily_close"),
    ("volume", "daily_volume"), ("amount", "daily_amount"),
    ("change_pct", "daily_change_pct"), ("turnover_rate", "daily_turnover_rate"),
    ("market_cap", "market_cap"), ("pe_ttm", "pe_ttm"), ("pe_static", "pe_static"),
    ("pb", "pb"), ("pcf", "pcf"),
)


def _db_history_to_analysis_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"date": row["date"], **{ak: row[dk] for ak, dk in _DB_HISTORY_TO_ANALYSIS}}
        for row in records
    ]


_cn_index_load_lock = threading.Lock()
_fundamental_load_lock = threading.Lock()
_sw3_segment_load_lock = threading.Lock()


def load_cn_stock_index() -> Dict[str, Dict[str, Any]]:
    # 锁串行化首次加载：预热线程与首个请求并发时只算一次（lru 命中后锁开销可忽略）。
    with _cn_index_load_lock:
        return _load_cn_stock_index_cached()


@lru_cache(maxsize=1)
def _load_cn_stock_index_cached() -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    conn = stock_storage.connect()
    try:
        for meta, records in stock_storage.iter_history(conn):
            code = str(meta["code"]).zfill(6)
            item: Dict[str, Any] = {
                "symbol": meta["code"],
                "name": meta["name"] or code,
                "source": meta["history_source"],
                "price_adjust": meta["price_adjust"] or "qfq",
                "change_pct_basis": "close_to_close",
                "records": _db_history_to_analysis_records(records),
                "_source": "stock_data.history",
            }
            if meta["history_start_date"]:
                item["start_date"] = meta["history_start_date"]
            if meta["history_end_date"]:
                item["end_date"] = meta["history_end_date"]
            index[code] = item
    finally:
        conn.close()
    return index


def load_fundamental_stocks() -> Dict[str, Dict[str, Any]]:
    with _fundamental_load_lock:
        return _load_fundamental_stocks_cached()


def warm_caches() -> None:
    """预热数据 loader 缓存（供 server 启动时后台调用），把首个策略请求的 ~6s 冷载移出请求路径。"""
    load_fundamental_stocks()
    load_cn_stock_index()
    load_sw3_segment_map()


@lru_cache(maxsize=1)
def _load_fundamental_stocks_cached() -> Dict[str, Dict[str, Any]]:
    # 不读日线：价格因子一律经 price_history_rows/combined_recent_rows 走 cn_index，
    # 基本面 dict 里的 history 是冗余回退（无 cn_index 的票本就无日线），省去重建百万行。
    conn = stock_storage.connect()
    try:
        return {code: data for code, data in stock_storage.iter_stocks(conn, include_history=False)}
    finally:
        conn.close()


def latest_valuation_from_records(
        records: List[Dict[str, Any]],
        as_of: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    fields = {
        "market_cap": "market_cap",
        "pe_ttm": "pe_ttm",
        "pb": "pb",
        "pcf": "pcf",
    }
    values: Dict[str, Optional[float]] = {key: None for key in fields}
    missing = set(fields)
    if not records:
        return values

    sorted_ascending = str(records[0].get("date", "")) <= str(records[-1].get("date", ""))
    end = len(records)
    if as_of is not None and sorted_ascending:
        lo = 0
        hi = len(records)
        while lo < hi:
            mid = (lo + hi) // 2
            if str(records[mid].get("date", "")) <= as_of:
                lo = mid + 1
            else:
                hi = mid
        end = lo

    for idx in range(end - 1, -1, -1):
        row = records[idx]
        if as_of is not None and not sorted_ascending and str(row.get("date", "")) > as_of:
            continue
        for key in list(missing):
            value = safe_float(row.get(fields[key]))
            if value is not None:
                values[key] = value
                missing.remove(key)
        if not missing:
            break
    return values


def latest_valuation_from_cn(code: str, as_of: Optional[str] = None) -> Dict[str, Optional[float]]:
    cn = load_cn_stock_index().get(code) or {}
    return latest_valuation_from_records(cn.get("records", []), as_of=as_of)


def combined_recent_rows(code: str, stock: Dict[str, Any]) -> List[Dict[str, Any]]:
    cn = load_cn_stock_index().get(code)
    if cn and cn.get("records"):
        return cn["records"][-80:]
    return stock_history_records(stock)[-80:]


def price_history_rows(code: str, stock: Dict[str, Any]) -> List[Dict[str, Any]]:
    cn = load_cn_stock_index().get(code)
    if cn and cn.get("records"):
        return cn["records"]
    return stock_history_records(stock)


def historical_high_drawdown_pct(
        code: str,
        stock: Dict[str, Any],
        as_of: Optional[str] = None,
        hist_rows: Optional[List[Dict[str, Any]]] = None,
) -> Optional[float]:
    rows = hist_rows if hist_rows is not None else price_rows_asof(price_history_rows(code, stock), as_of)
    return high_to_latest_drawdown_pct(rows)


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
        available_weight_sum = 0.0
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
                if safe_float(raw) is not None:
                    available_weight_sum += weight
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
        result["weighted_data_quality"] = round(
            available_weight_sum / weight_sum if weight_sum > 0 else 0.0,
            3,
        )
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


def pit_market_cap_cny(
        code: str,
        stock: Dict[str, Any],
        as_of: str,
        hist_rows: Optional[List[Dict[str, Any]]] = None,
        cn_records: Optional[List[Dict[str, Any]]] = None,
) -> Optional[float]:
    """PIT 总市值(元)：优先 stock_data.history ≤as_of 的 market_cap；否则用截止 as_of 收盘 ×
    当时可见财报推算的股本(归母权益 / 每股净资产)。不碰 market_snapshot 的当前市值。"""
    cn_val = (
        latest_valuation_from_records(cn_records, as_of=as_of)
        if cn_records is not None
        else latest_valuation_from_cn(code, as_of=as_of)
    )
    if cn_val.get("market_cap"):
        return cn_val["market_cap"] * 1e8
    prows = hist_rows if hist_rows is not None else price_rows_asof(price_history_rows(code, stock), as_of)
    price_asof = next(
        (c for c in (safe_float(r.get("close")) for r in reversed(prows)) if c and c > 0),
        None,
    )
    ind = reports_available_asof(stock.get("indicators", {}).get("records", []), as_of)
    bal = reports_available_asof(stock.get("financials", {}).get("balance", []), as_of)
    bvps = safe_float(ind[0].get("bvps_adjusted")) if ind else None
    equity = safe_float(bal[0].get("total_equity_parent")) if bal else None
    shares = equity / bvps if (bvps and bvps > 0 and equity and equity > 0) else None
    return shares * price_asof if (shares and price_asof) else None


def build_long_candidates(
        config: Dict[str, Any],
        as_of: Optional[str] = None,
        universe: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    stocks = load_fundamental_stocks()
    notes = []
    target_universe = {str(code).zfill(6) for code in universe} if universe is not None else None
    if bool(config.get("use_segment_leaders", True)):
        leader_codes = load_segment_leader_codes()
        if leader_codes:
            target_universe = (
                leader_codes if target_universe is None else target_universe & leader_codes
            )
            notes.append(
                "长线候选池限定为 stock_crawl_segment_leaders.py 生成的 "
                f"SW3 细分龙头池（is_leader=1，{len(leader_codes)} 只）。"
            )
        else:
            notes.append(
                "sw3_member.is_leader 为空；长线候选暂回退到全量基本面股票池。"
                "请先运行 python stock_crawl_segment_leaders.py crawl 生成 SW3 细分龙头池。"
            )
    if target_universe is not None:
        stocks = {c: s for c, s in stocks.items() if c in target_universe}

    stock_universe = load_stock_universe()
    snapshot = load_market_snapshot()
    cn_index = load_cn_stock_index()
    sw3_segments = load_sw3_segment_map()
    capital_signals, capital_availability = load_long_capital_signals(as_of)
    csi300 = set(str(code).zfill(6) for code in stock_universe.get("csi300", []))
    csi_all = set(str(code).zfill(6) for code in stock_universe.get("all", []))
    if not csi300:
        notes.append("data/stock_universe.json lacks csi300; CSI300 filters are inactive.")
    if as_of is not None:
        notes.append(
            f"PIT 时点 {as_of}：财报/价格/估值/分红已按当时可见切片；"
            "资金面按公告/上榜日切片；但沪深300成分、申万行业与质押用的是当前快照(无历史数据)，这几维仍有轻微前视。"
        )
    if any(capital_availability.values()):
        notes.append(
            "长线资金面已挂载：股东户数变化、公司回购、近期龙虎榜避雷。"
        )
    else:
        notes.append(
            "长线资金面数据表为空；运行 stock_crawl_holders.py 与 stock_crawl_capital.py 后生效。"
        )

    min_cap_yi = safe_float(config.get("min_market_cap_yi")) or 0.0
    require_csi300 = bool(config.get("require_csi300", True))
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
        if require_csi300 and not in_csi300:
            continue

        cn_records = (cn_index.get(code) or {}).get("records", [])
        base_price_rows = cn_records or stock_history_records(stock)
        # load_* caches return a versioned snapshot that stays immutable until
        # invalidate_dir_fingerprints(); only this internal path enables LRU reuse.
        pretransform_price_rows(base_price_rows, cache=True)
        hist_rows = price_rows_asof(base_price_rows, as_of)
        price_arr = price_arrays(base_price_rows, cache=True)

        if as_of is None:
            market_cap_cny = estimate_market_cap_cny(stock, snap)
            cn_val = latest_valuation_from_records(cn_records)
            if cn_val.get("market_cap"):
                market_cap_yi = cn_val["market_cap"]
                market_cap_cny = market_cap_yi * 1e8
            else:
                market_cap_yi = market_cap_cny / 1e8 if market_cap_cny else None
        else:
            market_cap_cny = pit_market_cap_cny(
                code, stock, as_of, hist_rows=hist_rows, cn_records=cn_records
            )
            market_cap_yi = market_cap_cny / 1e8 if market_cap_cny else None
        if min_cap_yi and (market_cap_yi is None or market_cap_yi < min_cap_yi):
            continue

        high_drawdown_pct = _drawdown_np(price_arr["close"], len(hist_rows))
        if require_high_drawdown:
            if high_drawdown_pct is None or high_drawdown_pct < min_high_drawdown_pct:
                continue

        raw = compute_long_raw_factors(
            code, stock, snap, in_csi300, in_csi_all, market_cap_cny,
            high_drawdown_pct, as_of=as_of, hist_rows=hist_rows,
            capital_signal=capital_signals.get(code),
            capital_available=capital_availability,
            price_arr=price_arr, asof_len=len(hist_rows),
        )

        legacy_industry = (
            clean_industry_name(stock.get("pledge", {}).get("industry"))
            or clean_industry_name(snap.get("pledge_industry"))
        )
        industry, industry_bucket, sw2_industry, sw3_industry, sw3_segment_code = industry_label_from_sw3(
            sw3_segments.get(code), legacy_industry
        )
        item = {
            "code": code,
            "name": name,
            "strategy": "long",
            "horizon": f"{config.get('hold_years_min', 2)}-{config.get('hold_years_max', 5)} years",
            "industry": industry,
            "industry_bucket": industry_bucket,
            "sw2_industry": sw2_industry,
            "sw3_industry": sw3_industry,
            "sw3_segment_code": sw3_segment_code,
            "raw_factors": raw,
            "reasons": [],
            "warnings": long_warnings(raw),
        }
        candidates.append(item)
        if market_cap_cny:
            industry_buckets[industry_bucket].append((code, market_cap_cny))

    apply_industry_leadership(candidates, industry_buckets)
    for item in candidates:
        item["data_quality"] = available_factor_ratio(item["raw_factors"], LONG_FACTORS)
        item["reasons"] = long_reasons(item)
    return candidates, notes


def smallcap_reasons(item: Dict[str, Any]) -> List[str]:
    """沿用长线解释口径，但不展示小盘策略已明确排除的指数/市值信号。"""
    raw = item["raw_factors"]
    reasons = []
    if safe_float(raw.get("roe_stability")) is not None:
        reasons.append(f"ROE稳定因子{raw['roe_stability']:.1f}")
    if safe_float(raw.get("dividend_yield_5y")) is not None:
        reasons.append(f"滚动五年年均股息率{raw['dividend_yield_5y']:.2f}%")
    if safe_float(raw.get("cashflow_quality")) is not None:
        reasons.append(f"现金流/利润{raw['cashflow_quality']:.2f}")
    holder_change = safe_float(raw.get("holder_count_change"))
    if holder_change is not None and holder_change < 0:
        reasons.append(f"股东户数下降{abs(holder_change):.1f}%")
    if safe_float(raw.get("repurchase_recent")):
        reasons.append("近期回购")
    return reasons[:5]


def build_smallcap_candidates(
        config: Dict[str, Any],
        as_of: Optional[str] = None,
        universe: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """用游资小盘成员作为候选集合，复用长线的 PIT 基本面/价量因子计算。"""
    hot_money_codes = load_hot_money_codes()
    target = set(hot_money_codes)
    if universe is not None:
        target &= {str(code).zfill(6) for code in universe}

    long_style_config = deep_merge(DEFAULT_CONFIG["long"], config)
    long_style_config.update({
        "use_segment_leaders": False,
        "min_market_cap_yi": 0,
        "require_csi300": False,
    })
    candidates, notes = build_long_candidates(
        long_style_config,
        as_of=as_of,
        universe=target,
    )
    for item in candidates:
        item["strategy"] = "smallcap"
        item["horizon"] = (
            f"{config.get('hold_years_min', 2)}-"
            f"{config.get('hold_years_max', 5)} years"
        )
        item["data_quality"] = available_factor_ratio(item["raw_factors"], SMALLCAP_FACTORS)
        item["reasons"] = smallcap_reasons(item)
    notes.insert(
        0,
        "小盘候选池与短线策略/游资雷达共用 is_hot_money=1 成员"
        f"（当前 {len(hot_money_codes)} 只）；评分复用长线因子。",
    )
    return candidates, notes


def passes_long_hard_filters(item: Dict[str, Any], config: Dict[str, Any]) -> bool:
    raw = item.get("raw_factors", {})
    if bool(config.get("exclude_st", True)) and is_st_name(str(item.get("name") or "")):
        return False

    min_cap_yi = safe_float(config.get("min_market_cap_yi")) or 0.0
    market_cap_yi = safe_float(raw.get("market_cap"))
    if min_cap_yi and (market_cap_yi is None or market_cap_yi < min_cap_yi):
        return False

    if bool(config.get("require_csi300", True)) and not raw.get("csi300_current"):
        return False

    if bool(config.get("require_high_drawdown", False)):
        min_high_drawdown_pct = safe_float(config.get("min_high_drawdown_pct")) or 0.0
        high_drawdown = safe_float(raw.get("historical_high_drawdown"))
        if high_drawdown is None or high_drawdown < min_high_drawdown_pct:
            return False

    return True


def passes_smallcap_hard_filters(item: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """小盘不使用沪深300或总市值约束，只保留通用风控过滤。"""
    if bool(config.get("exclude_st", True)) and is_st_name(str(item.get("name") or "")):
        return False
    if bool(config.get("require_high_drawdown", False)):
        min_high_drawdown_pct = safe_float(config.get("min_high_drawdown_pct")) or 0.0
        high_drawdown = safe_float((item.get("raw_factors") or {}).get("historical_high_drawdown"))
        if high_drawdown is None or high_drawdown < min_high_drawdown_pct:
            return False
    return True


def compute_long_raw_factors(
        code: str,
        stock: Dict[str, Any],
        snapshot: Dict[str, Any],
        in_csi300: bool,
        in_csi_all: bool,
        market_cap_cny: Optional[float],
        high_drawdown_pct: Optional[float] = None,
        as_of: Optional[str] = None,
        hist_rows: Optional[List[Dict[str, Any]]] = None,
        capital_signal: Optional[Dict[str, Any]] = None,
        capital_available: Optional[Dict[str, bool]] = None,
        price_arr: Optional[Dict[str, Any]] = None,
        asof_len: Optional[int] = None,
) -> Dict[str, Any]:
    # PIT(as_of 给定)：财报只取截止 as_of 已公告可见的；as_of=None 时原样(实盘)。
    financials = stock.get("financials", {})
    income = reports_available_asof(financials.get("income", []), as_of)
    balance = reports_available_asof(financials.get("balance", []), as_of)
    cashflow = reports_available_asof(financials.get("cashflow", []), as_of)
    indicators = reports_available_asof(stock.get("indicators", {}).get("records", []), as_of)
    # 实盘信任预算的 roe_stats；PIT 下置空，强制从过滤后 indicators 现算
    roe_stats = stock.get("indicators", {}).get("roe_stats", {}) if as_of is None else {}
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

    # 价格行为因子用历史日线（stock_data.history 最长约10年）；PIT 下切到 as_of 之前
    if hist_rows is None:
        hist_rows = price_rows_asof(price_history_rows(code, stock), as_of)
    # 价格/换手序列：优先用缓存的全历史 numpy 数组切前缀(向量化释放 GIL、各折复用免重转)，
    # 无缓存则回退逐行 safe_float。两路数值等价(None/脏值→nan，过滤 >0 同语义)。
    if price_arr is not None and asof_len is not None:
        ca = price_arr["close"][:asof_len]
        ca = ca[~np.isnan(ca) & (ca > 0.0)]                 # 等价 [c for c if c is not None and c>0]
        ta = price_arr["turnover_rate"][:asof_len]
    else:
        _hc = [safe_float(row.get("close")) for row in hist_rows]
        ca = np.array([c for c in _hc if c is not None and c > 0], dtype=np.float64)
        turn_values = []
        for row in hist_rows:
            turn = first_not_none(row.get("turnover_rate"), row.get("daily_turnover_rate"))
            turn = safe_float(turn)
            turn_values.append(np.nan if turn is None else turn)
        ta = np.array(turn_values, dtype=np.float64) if turn_values else np.empty(0)
    n_close = ca.size

    if high_drawdown_pct is None and n_close >= 2:
        peak_close = float(ca.max())
        if peak_close > 0:
            high_drawdown_pct = (peak_close - float(ca[-1])) / peak_close * 100.0
    reversal_1m = None
    momentum_12_1 = None
    dist_52w_high = None
    reversal_long_term = None
    if n_close >= 21:
        reversal_1m = float(ca[-1] / ca[-21] - 1)
    if n_close >= 250:
        momentum_12_1 = float(ca[-21] / ca[-250] - 1)
    if n_close >= 120:
        peak = float(ca[-250:].max())
        dist_52w_high = float(ca[-1] / peak) if peak > 0 else None
    if n_close >= 750:
        reversal_long_term = float(ca[-1] / ca[-750] - 1)

    recent_arr = ta[-20:]
    recent_arr = recent_arr[~np.isnan(recent_arr) & (recent_arr > 0.0)]
    base_arr = ta[-250:]
    base_arr = base_arr[~np.isnan(base_arr) & (base_arr > 0.0)]
    abnormal_turnover = None
    if recent_arr.size >= 10 and base_arr.size >= 60:
        base_avg = float(base_arr.mean())
        if base_avg > 1e-9:
            abnormal_turnover = float(recent_arr.mean() / base_avg)

    roe_values = [safe_float(row.get("roe")) for row in indicators]
    roe_values = [v for v in roe_values if v is not None]
    roe_mean = first_not_none(roe_stats.get("mean"), mean_or_none(roe_values))
    roe_std = safe_float(roe_stats.get("std"))
    if roe_std is None:
        roe_std = stdev_or_zero(roe_values)
    roe_stability = None
    if roe_mean is not None:
        roe_stability = roe_mean - (roe_std or 0.0)

    persistence = csi300_persistence_proxy(in_csi300, in_csi_all)
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

    piotroski = piotroski_f_score(income, indicators, net_profit_ttm, op_cash_ttm, total_assets)

    reported_gross_margin = safe_float(latest_ind.get("gross_margin"))
    if reported_gross_margin is None:
        # 财务指标接口的最新季度经常不披露毛利率；优先用同一 PIT 切片的
        # TTM 营收和营业成本重算，再回退最近一期非空的已披露值。
        reported_gross_margin = first_not_none(
            scale_ratio_to_pct(safe_ratio(gross_profit_ttm, revenue_ttm)),
            latest_available_value(indicators, "gross_margin"),
        )

    dividends = stock.get("dividends")
    dividend_data_available = (
        isinstance(dividends, dict) and isinstance(dividends.get("records"), list)
    )

    # 波动率/流动性/分红：实盘读文件快照；PIT 从切片日线 + 截止 as_of 的分红现算
    if as_of is None:
        daily_stats = derived_daily_stats(stock)
        low_volatility = first_not_none(
            daily_stats.get("history_window_annualized_volatility"),
            daily_stats.get("volatility_annual"),
        )
        liquidity = first_not_none(
            daily_stats.get("history_window_avg_daily_turnover_rate"),
            avg_turnover_rate_from_rows(hist_rows),
        )
        dividend_yield_5y = scale_ratio_to_pct(dividend_yield(stock, years=5))
        dividend_consistency = (
            1.0 if dividends.get("consecutive_3y_dividend") else 0.0
        ) if dividend_data_available else None
    else:
        if price_arr is not None and asof_len is not None:
            low_volatility = _vol_np(price_arr["close"], asof_len)
            liquidity = _avg_turnover_rate_np(price_arr["turnover_rate"], asof_len)
        else:
            low_volatility = annualized_vol_from_rows(hist_rows)
            liquidity = avg_turnover_rate_from_rows(hist_rows)
        price_asof = float(ca[-1]) if n_close else None
        dividend_yield_5y = scale_ratio_to_pct(
            dividend_yield(stock, years=5, as_of=as_of, price=price_asof)
        )
        dividend_consistency = (
            1.0 if consecutive_dividend_asof(stock, as_of) else 0.0
        ) if dividend_data_available else None

    capital_signal = capital_signal or {}
    capital_available = capital_available or {}
    holder_count_change = (
        safe_float(capital_signal.get("holder_change"))
        if capital_available.get("holder")
        else None
    )
    repurchase_recent = (
        1.0 if capital_signal.get("repurchase_recent") else 0.0
    ) if capital_available.get("repurchase") else None
    lhb_recent_avoid = (
        1.0 if capital_signal.get("lhb_recent") else 0.0
    ) if capital_available.get("lhb") else None

    return {
        "csi300_current": 1.0 if in_csi300 else 0.0,
        "csi300_persistence": persistence,
        "market_cap": market_cap_yi,
        "size_reversal": market_cap_yi,
        "liquidity": liquidity,
        "low_volatility": low_volatility,
        "roe_mean": roe_mean,
        "roe_stability": roe_stability,
        "revenue_growth": scale_ratio_to_pct(compute_yoy_growth(income, "revenue")),
        "net_profit_growth": scale_ratio_to_pct(compute_yoy_growth(income, "net_profit")),
        "gross_margin": reported_gross_margin,
        "net_margin": safe_float(latest_ind.get("net_margin")),
        "cashflow_quality": cash_quality,
        "dividend_yield_5y": dividend_yield_5y,
        "dividend_consistency": dividend_consistency,
        "holder_count_change": holder_count_change,
        "repurchase_recent": repurchase_recent,
        "lhb_recent_avoid": lhb_recent_avoid,
        "debt_safety": safe_float(latest_ind.get("asset_liability_ratio")),
        "pledge_safety": first_not_none(
            stock.get("pledge", {}).get("pledge_ratio"), snapshot.get("pledge_ratio")
        ),
        "historical_high_drawdown": high_drawdown_pct,
        "industry_leadership": None,
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


def csi300_persistence_proxy(in_csi300: bool, in_csi_all: bool) -> float:
    return 100.0 if in_csi300 else (45.0 if in_csi_all else 0.0)


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
        reasons.append(f"滚动五年年均股息率{raw['dividend_yield_5y']:.2f}%")
    if safe_float(raw.get("cashflow_quality")) is not None:
        reasons.append(f"现金流/利润{raw['cashflow_quality']:.2f}")
    holder_change = safe_float(raw.get("holder_count_change"))
    if holder_change is not None and holder_change < 0:
        reasons.append(f"股东户数下降{abs(holder_change):.1f}%")
    if safe_float(raw.get("repurchase_recent")):
        reasons.append("近期回购")
    return reasons[:5]


def long_warnings(raw: Dict[str, Any]) -> List[str]:
    warnings = []
    if safe_float(raw.get("lhb_recent_avoid")):
        warnings.append("近期上龙虎榜，长线按避雷处理")
    holder_change = safe_float(raw.get("holder_count_change"))
    if holder_change is not None and holder_change > 15:
        warnings.append("股东户数明显增加，筹码集中度走弱")
    return warnings


# ---------------------------------------------------------------------------
# Short horizon strategy
# ---------------------------------------------------------------------------


def load_short_universe() -> Dict[str, Dict[str, Any]]:
    """加载与游资雷达完全相同的固定小盘池，信号缺失不影响成员资格。"""
    conn = stock_storage.connect()
    try:
        members = stock_storage.pool_members(conn, "hotmoney")
        signals = stock_storage.load_short_signals(
            conn, [str(member.get("code") or "") for member in members]
        )
    finally:
        conn.close()

    pool: Dict[str, Dict[str, Any]] = {}
    for member in members:
        code = str(member.get("code") or "").zfill(6)
        if not code:
            continue
        signal = signals.get(code) or {}
        sources = ["hotmoney_universe"]
        if signal:
            sources.append("short_signal_snapshot")
        pool[code] = {
            "code": code,
            "name": str(member.get("name") or signal.get("name") or code),
            "sources": sources,
            "capital": signal,
            "as_of_date": signal.get("as_of_date") or signal.get("generated_at"),
        }
    return pool


def build_short_candidates(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    pool = load_short_universe()
    sw3_segments = load_sw3_segment_map()
    notes = []
    if not pool:
        notes.append("Hot-money small-cap universe is empty. Run stock_crawl_hot_money_universe.py first.")
    else:
        signal_count = sum(
            1 for data in pool.values() if "short_signal_snapshot" in data.get("sources", [])
        )
        if signal_count < len(pool):
            notes.append(
                f"Short signal snapshot coverage: {signal_count}/{len(pool)}; "
                "missing signals remain eligible and score as unavailable."
            )

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
        industry, _, sw2_industry, sw3_industry, sw3_segment_code = industry_label_from_sw3(
            sw3_segments.get(code)
        )

        item = {
            "code": code,
            "name": name,
            "strategy": "short",
            "horizon": f"{config.get('hold_days_min', 1)}-{config.get('hold_days_max', 5)} days",
            # The optimizer must not infer this from the truncated display
            # sample below. It is computed from the complete signal payload.
            "event_date": short_signal_event_date(data),
            "industry": industry,
            "sw2_industry": sw2_industry,
            "sw3_industry": sw3_industry,
            "sw3_segment_code": sw3_segment_code,
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
    capital = data.get("capital") or data.get("scored") or pick or {}
    signals = pick.get("signals") or capital.get("signals") or {}
    lhb = signals.get("lhb") or {}
    inst = signals.get("inst") or {}
    stat = signals.get("stat") or {}
    tech = signals.get("tech") or {}
    followers = normalize_followers(capital.get("followers") or pick.get("followers") or [])

    seats = [str(f.get("seat", "")) for f in followers if f.get("seat")]
    seat_counts = Counter(seats)
    known_count = sum(1 for f in followers if str(f.get("category", "")).lower() == "knownhotmoney")
    buy_values = [safe_float(f.get("buy_est")) for f in followers]
    buy_values = [v for v in buy_values if v is not None and v > 0]
    total_buy_est = sum(buy_values) if buy_values else first_not_none(
        safe_float(capital.get("buy_amount_total")),
        safe_float(pick.get("buy_amount_total")),
    )
    known_buy_est = sum(
        safe_float(f.get("buy_est")) or 0.0
        for f in followers
        if str(f.get("category", "")).lower() == "knownhotmoney"
    )
    latest_date = latest_follower_date(followers)
    as_of = parse_date(capital.get("as_of_date") or data.get("as_of_date"))
    recency_days = None
    if latest_date and as_of:
        recency_days = max(0, (as_of - latest_date).days)

    best_window_span = None
    best_window = capital.get("best_window") or pick.get("best_window")
    if isinstance(best_window, list) and len(best_window) >= 2:
        d0 = parse_date(best_window[0])
        d1 = parse_date(best_window[1])
        if d0 and d1:
            best_window_span = abs((d1 - d0).days)

    is_yizi = bool(tech.get("is_yizi_ban"))
    is_t_ban = bool(tech.get("is_t_ban"))
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
        capital.get("concurrent_count"),
        pick.get("concurrent_count"),
        follower_count if follower_count else None,
    )
    buy_concentration = buy_concentration_score(buy_values)
    known_amount_ratio = safe_ratio(known_buy_est, total_buy_est)
    persistence_score = hot_money_persistence_score(followers, seat_counts)
    event_decay = event_recency_decay_score(recency_days)
    institution_combo = institution_hotmoney_combo_score(inst_net, concurrent, lhb_net)
    institution_conflict = institution_conflict_score(inst_net, lhb_net)
    amount_per_buyer = safe_ratio(
        total_buy_est,
        first_not_none(capital.get("total_buyers"), pick.get("total_buyers"), len(seat_counts) or None),
    )
    limit_up_count = first_not_none(
        tech.get("consecutive_limit_up"), tech.get("recent_limit_up")
    )
    chg_5d = safe_float(tech.get("chg_5d"))
    chg_today = safe_float(tech.get("chg_today"))
    dist_ma20 = safe_float(tech.get("dist_from_ma20_pct"))
    rsi_score = rsi_sweetspot_score(safe_float(tech.get("rsi")))
    vol_ratio = safe_float(tech.get("vol_ratio"))
    turnover_today = safe_float(tech.get("turnover_today"))
    overheat = overheat_control_score(chg_5d, safe_float(limit_up_count), safe_float(tech.get("rsi")))
    risk_control = average_score(overheat, tradability, rsi_score)
    ma_distance = sweetspot_score(dist_ma20, 0.0, 18.0, -12.0, 45.0)
    macd_strength = sweetspot_score(safe_float(tech.get("macd_dif")), 0.0, 0.8, -0.6, 2.5)
    alpha_pv1 = average_score(
        sweetspot_score(chg_5d, 2.0, 12.0, -8.0, 28.0),
        sweetspot_score(vol_ratio, 1.1, 3.2, 0.4, 6.0),
    )
    alpha_rev1 = average_score(
        sweetspot_score(chg_today, -3.0, 7.5, -10.0, 15.0),
        overheat,
        bounded_linear(safe_float(limit_up_count), 0.0, 4.0, reverse=True),
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
        "weighted_hot_money": first_not_none(capital.get("weighted_score"), pick.get("weighted_score")),
        "buy_amount_total": first_not_none(
            safe_float(capital.get("buy_amount_total")),
            safe_float(pick.get("buy_amount_total")),
        ),
        "known_hot_money_ratio": known_count / len(followers) if followers else None,
        "seat_diversity": len(seat_counts) if seat_counts else None,
        "recency": recency_days,
        "best_window_tightness": best_window_span,
        "resonance_score": resonance_strength_score(concurrent),
        "momentum_score": tech_momentum_score(tech),
        "position_score": tech_position_score(tech),
        "risk_control": risk_control,
        "limit_up_control": safe_float(limit_up_count),
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


def short_signal_event_date(data: Dict[str, Any]) -> Optional[str]:
    """Latest date whose information contributes to a short candidate.

    Followers drive most event factors, while technical fields are calculated
    at the signal/kline snapshot date. Taking the maximum of both prevents a
    backtest from entering before every factor in the candidate was knowable.
    """
    pick = data.get("pick") or {}
    capital = data.get("capital") or data.get("scored") or pick or {}
    followers = normalize_followers(capital.get("followers") or pick.get("followers") or [])
    dates = [parse_date(follower.get("date")) for follower in followers]
    dates.extend(parse_date(value) for value in (
        capital.get("kline_as_of_date"),
        capital.get("as_of_date"),
        data.get("as_of_date"),
    ))
    valid_dates = [date for date in dates if date is not None]
    return max(valid_dates).strftime("%Y-%m-%d") if valid_dates else None


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


def tech_position_score(tech: Dict[str, Any]) -> Optional[float]:
    if not tech:
        return None
    dist_score = sweetspot_score(
        safe_float(tech.get("dist_from_ma20_pct")),
        -3.0,
        14.0,
        -20.0,
        42.0,
    )
    return average_score(dist_score, rsi_sweetspot_score(safe_float(tech.get("rsi"))))


def resonance_strength_score(concurrent: Optional[float]) -> Optional[float]:
    value = safe_float(concurrent)
    if value is None:
        return None
    return bounded_linear(value, 1.0, 4.0)


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
    if data.get("pick") and not data.get("capital"):
        warnings.append("缺少席位级followers明细")
    return warnings


def follower_sample(data: Dict[str, Any], limit: int = 6) -> List[Dict[str, Any]]:
    pick = data.get("pick") or {}
    capital = data.get("capital") or data.get("scored") or pick or {}
    followers = normalize_followers(capital.get("followers") or pick.get("followers") or [])

    def sample_order(follower: Dict[str, Any]) -> Tuple[int, str, str, float]:
        date = parse_date(follower.get("date"))
        return (
            -(date.toordinal() if date else 0),
            str(follower.get("seat") or ""),
            str(follower.get("category") or ""),
            -(safe_float(follower.get("buy_est")) or 0.0),
        )

    out = []
    for follower in sorted(followers, key=sample_order)[:max(0, limit)]:
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


def live_long_candidate_pool(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    key = stable_config_key(config, ignore=("weights", "min_score", "top_n", "enabled"))
    universe_signature = _file_signature(DATA_DIR / "stock_universe.json")
    snapshot_signature = _file_signature(DATA_DIR / "market_snapshot.json")
    stock_data_signature = stock_storage.db_signature()
    leader_signature = segment_leader_code_signature()
    capital_signature = long_capital_signal_signature()
    meta = {
        "config_key": key,
        "long_liquidity_factor": LONG_LIQUIDITY_FACTOR_VERSION,
        "long_dividend_factor": LONG_DIVIDEND_FACTOR_VERSION,
        "stock_data": stock_data_signature,
        "stock_universe": list(universe_signature),
        "market_snapshot": list(snapshot_signature),
        "segment_leaders": leader_signature,
        "capital_signals": capital_signature,
    }
    cached = read_live_candidate_cache("long", meta)
    if cached is not None:
        return cached

    candidates, notes = _build_live_long_candidates_cached(
        universe_signature,
        snapshot_signature,
        _signature_key(stock_data_signature),
        _signature_key(leader_signature),
        _signature_key(capital_signature),
        LONG_LIQUIDITY_FACTOR_VERSION,
        LONG_DIVIDEND_FACTOR_VERSION,
        key,
    )
    write_live_candidate_cache("long", meta, candidates, notes)
    return copy.deepcopy(candidates), list(notes)


@lru_cache(maxsize=8)
def _build_live_long_candidates_cached(
        universe_signature: Tuple[int, int],
        snapshot_signature: Tuple[int, int],
        stock_data_signature: Tuple[Tuple[str, Any], ...],
        leader_signature: Tuple[Tuple[str, Any], ...],
        capital_signature: Tuple[Tuple[str, Any], ...],
        liquidity_factor_version: str,
        dividend_factor_version: str,
        config_key: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    # 进程内缓存：stock_data 变更由 invalidate_dir_fingerprints() 显式清缓存覆盖，
    # 磁盘候选缓存另用 db_signature()(内容指纹) 校验，无需易变的文件指纹做 key。
    _ = (
        universe_signature, snapshot_signature, stock_data_signature,
        leader_signature, capital_signature, liquidity_factor_version,
        dividend_factor_version,
    )
    return build_long_candidates(config_from_key(config_key))


def live_smallcap_candidate_pool(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    key = stable_config_key(config, ignore=("weights", "min_score", "top_n", "enabled"))
    conn = stock_storage.connect()
    try:
        hot_money_pool_signature = stock_storage.pool_signature(conn, "hotmoney")
    finally:
        conn.close()
    stock_data_signature = stock_storage.db_signature()
    capital_signature = long_capital_signal_signature()
    meta = {
        "config_key": key,
        "universe_version": SMALLCAP_UNIVERSE_VERSION,
        "hot_money_pool": hot_money_pool_signature,
        "stock_data": stock_data_signature,
        "capital_signals": capital_signature,
        "long_liquidity_factor": LONG_LIQUIDITY_FACTOR_VERSION,
        "long_dividend_factor": LONG_DIVIDEND_FACTOR_VERSION,
    }
    cached = read_live_candidate_cache("smallcap", meta)
    if cached is not None:
        return cached

    candidates, notes = _build_live_smallcap_candidates_cached(
        _signature_key(hot_money_pool_signature),
        _signature_key(stock_data_signature),
        _signature_key(capital_signature),
        SMALLCAP_UNIVERSE_VERSION,
        LONG_LIQUIDITY_FACTOR_VERSION,
        LONG_DIVIDEND_FACTOR_VERSION,
        key,
    )
    write_live_candidate_cache("smallcap", meta, candidates, notes)
    return copy.deepcopy(candidates), list(notes)


@lru_cache(maxsize=8)
def _build_live_smallcap_candidates_cached(
        hot_money_pool_signature: Tuple[Tuple[str, Any], ...],
        stock_data_signature: Tuple[Tuple[str, Any], ...],
        capital_signature: Tuple[Tuple[str, Any], ...],
        universe_version: str,
        liquidity_factor_version: str,
        dividend_factor_version: str,
        config_key: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    _ = (
        hot_money_pool_signature, stock_data_signature, capital_signature,
        universe_version, liquidity_factor_version, dividend_factor_version,
    )
    return build_smallcap_candidates(config_from_key(config_key))


def live_short_candidate_pool(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    key = stable_config_key(config, ignore=("weights", "min_score", "top_n", "enabled"))
    conn = stock_storage.connect()
    try:
        hot_money_pool_signature = stock_storage.pool_signature(conn, "hotmoney")
    finally:
        conn.close()
    signal_signature = stock_storage.short_signal_signature()
    meta = {
        "config_key": key,
        "hot_money_pool": hot_money_pool_signature,
        "short_signals": signal_signature,
    }
    cached = read_live_candidate_cache("short", meta)
    if cached is not None:
        return cached

    candidates, notes = _build_live_short_candidates_cached(
        _signature_key(hot_money_pool_signature),
        _signature_key(signal_signature),
        key,
    )
    write_live_candidate_cache("short", meta, candidates, notes)
    return copy.deepcopy(candidates), list(notes)


@lru_cache(maxsize=8)
def _build_live_short_candidates_cached(
        hot_money_pool_signature: Tuple[Tuple[str, Any], ...],
        signal_signature: Tuple[Tuple[str, Any], ...],
        config_key: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    _ = (hot_money_pool_signature, signal_signature)
    return build_short_candidates(config_from_key(config_key))


def passes_short_hard_filters(item: Dict[str, Any], config: Dict[str, Any]) -> bool:
    name = str(item.get("name") or "")
    if bool(config.get("exclude_st", True)) and is_st_name(name):
        return False

    raw = item.get("raw_factors", {})
    min_lhb_count = int(safe_float(config.get("min_lhb_count")) or 0)
    if (safe_float(raw.get("lhb_recent_count")) or 0) < min_lhb_count:
        return False

    min_concurrent = int(safe_float(config.get("min_hot_money_concurrent")) or 0)
    if (safe_float(raw.get("hot_money_concurrent")) or 0) < min_concurrent:
        return False

    max_consec = int(first_not_none(config.get("max_consecutive_limit_up"), 999))
    if (safe_float(raw.get("limit_up_control")) or 0) > max_consec:
        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_long_strategy(
    config: Optional[Dict[str, Any]] = None,
    *,
    include_search_pool: bool = False,
) -> Dict[str, Any]:
    merged = deep_merge(get_default_config()["long"], config or {})
    scoring_config = copy.deepcopy(merged)
    # Keep percentile factor scores comparable when UI controls are used as
    # hard pool constraints; otherwise kept stocks can see their scores shift
    # merely because the normalization pool changed.
    scoring_config["exclude_st"] = False
    scoring_config["min_market_cap_yi"] = 0
    scoring_config["require_csi300"] = False
    scoring_config["require_high_drawdown"] = False
    candidates, notes = live_long_candidate_pool(scoring_config)
    scored = apply_scores(candidates, LONG_FACTORS, merged.get("weights", {}))
    scored = rerank_scored([item for item in scored if passes_long_hard_filters(item, merged)])
    candidate_codes = {item["code"] for item in scored}
    candidates = [item for item in candidates if item["code"] in candidate_codes]
    min_score = safe_float(merged.get("min_score")) or 0.0
    selected = [item for item in scored if item["score"] >= min_score]
    selected = selected[: max(0, int(first_not_none(merged.get("top_n"), 30)))]
    payload = {
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
    if include_search_pool:
        payload["search_pool"] = strip_internal(scored)
    return payload


def run_smallcap_strategy(
    config: Optional[Dict[str, Any]] = None,
    *,
    include_search_pool: bool = False,
) -> Dict[str, Any]:
    merged = deep_merge(get_default_config()["smallcap"], config or {})
    scoring_config = copy.deepcopy(merged)
    scoring_config["exclude_st"] = False
    scoring_config["require_high_drawdown"] = False
    candidates, notes = live_smallcap_candidate_pool(scoring_config)
    scored = apply_scores(candidates, SMALLCAP_FACTORS, merged.get("weights", {}))
    scored = rerank_scored([
        item for item in scored if passes_smallcap_hard_filters(item, merged)
    ])
    candidate_codes = {item["code"] for item in scored}
    candidates = [item for item in candidates if item["code"] in candidate_codes]
    min_score = safe_float(merged.get("min_score")) or 0.0
    selected = [item for item in scored if item["score"] >= min_score]
    selected = selected[: max(0, int(first_not_none(merged.get("top_n"), 30)))]
    payload = {
        "strategy": "smallcap",
        "title": "Small-cap long-factor strategy on hot-money universe",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": merged,
        "factor_count": len(SMALLCAP_FACTORS),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "notes": notes + long_data_notes(candidates, include_csi_note=False),
        "picks": strip_internal(selected),
        "diagnostics": diagnostics(scored, SMALLCAP_FACTORS),
    }
    if include_search_pool:
        payload["search_pool"] = strip_internal(scored)
    return payload


def run_short_strategy(
    config: Optional[Dict[str, Any]] = None,
    *,
    include_search_pool: bool = False,
) -> Dict[str, Any]:
    merged = deep_merge(get_default_config()["short"], config or {})
    scoring_config = copy.deepcopy(merged)
    scoring_config["exclude_st"] = False
    scoring_config["min_lhb_count"] = 0
    scoring_config["min_hot_money_concurrent"] = 0
    scoring_config["max_consecutive_limit_up"] = 999
    candidates, notes = live_short_candidate_pool(scoring_config)
    scored = apply_scores(candidates, SHORT_FACTORS, merged.get("weights", {}))
    scored = rerank_scored([item for item in scored if passes_short_hard_filters(item, merged)])
    candidate_codes = {item["code"] for item in scored}
    candidates = [item for item in candidates if item["code"] in candidate_codes]
    min_score = safe_float(merged.get("min_score")) or 0.0
    selected = [item for item in scored if item["score"] >= min_score]
    selected = selected[: max(0, int(first_not_none(merged.get("top_n"), 30)))]
    payload = {
        "strategy": "short",
        "title": "Short horizon hot-money small-cap universe strategy",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": merged,
        "factor_count": len(SHORT_FACTORS),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "notes": notes + short_data_notes(candidates),
        "picks": strip_internal(selected),
        "diagnostics": diagnostics(scored, SHORT_FACTORS),
    }
    if include_search_pool:
        payload["search_pool"] = strip_internal(scored)
    return payload


def run_strategies(
    config: Optional[Dict[str, Any]] = None,
    persist: bool = False,
    *,
    include_search_pool: bool = False,
) -> Dict[str, Any]:
    merged = deep_merge(get_default_config(), config or {})
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "factor_total": len(LONG_FACTORS) + len(SMALLCAP_FACTORS) + len(SHORT_FACTORS),
        "factor_counts": {
            "long": len(LONG_FACTORS),
            "smallcap": len(SMALLCAP_FACTORS),
            "short": len(SHORT_FACTORS),
            "total": len(LONG_FACTORS) + len(SMALLCAP_FACTORS) + len(SHORT_FACTORS),
        },
        "factor_registry": [
            spec.to_dict() for spec in LONG_FACTORS + SMALLCAP_FACTORS + SHORT_FACTORS
        ],
        "long": (
            run_long_strategy(merged.get("long", {}), include_search_pool=include_search_pool)
            if merged.get("long", {}).get("enabled", True)
            else None
        ),
        "short": (
            run_short_strategy(merged.get("short", {}), include_search_pool=include_search_pool)
            if merged.get("short", {}).get("enabled", True)
            else None
        ),
        "smallcap": (
            run_smallcap_strategy(merged.get("smallcap", {}), include_search_pool=include_search_pool)
            if merged.get("smallcap", {}).get("enabled", True)
            else None
        ),
        "self_review": self_review_summary(),
    }
    optimized_meta = merged.get("_optimized_defaults")
    if isinstance(optimized_meta, dict):
        optimized_note = (
            f"默认参数来自 {optimized_meta.get('generated_at') or '未知时间'} 的本地参数搜索"
            f"（代理回测，存在过拟合/前视风险）：{optimized_meta.get('caveat') or ''}"
        ).rstrip("：")
        for key in ("long", "short"):
            if payload.get(key):
                if (
                    key == "short"
                    and optimized_meta.get("short_universe_version") != SHORT_UNIVERSE_VERSION
                ):
                    payload[key]["notes"].append(
                        "旧短线优化参数基于已停用的候选池，已忽略并回退游资小盘池默认参数。"
                    )
                else:
                    payload[key]["notes"].append(optimized_note)
    smallcap_optimized_meta = merged.get("_smallcap_optimized_defaults")
    if isinstance(smallcap_optimized_meta, dict) and payload.get("smallcap"):
        smallcap_note = (
            f"小盘默认参数来自 {smallcap_optimized_meta.get('source') or '未知文件'} 的独立优化配置"
            f"（{smallcap_optimized_meta.get('generated_at') or '未知时间'}）："
            f"{smallcap_optimized_meta.get('caveat') or ''}"
        ).rstrip("：")
        payload["smallcap"]["notes"].append(smallcap_note)
    if persist:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
    return payload


def get_default_config() -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    optimized, optimized_source = load_optimized_config_payload()
    if optimized.get("config"):
        optimized_config = copy.deepcopy(optimized["config"])
        if isinstance(optimized_config.get("short"), dict):
            optimized_config["short"] = normalize_loaded_short_horizon(
                optimized_config["short"]
            )
        # 小盘权重只允许从独立文件加载，避免与长/短配置互相覆盖。
        optimized_config.pop("smallcap", None)
        # 旧优化结果基于“席位共振 JSON 候选池”，不能跨池沿用短线硬过滤与权重；
        # 长线配置仍可安全复用。新版 optimizer 会写入明确的 universe 版本。
        if optimized.get("short_universe_version") != SHORT_UNIVERSE_VERSION:
            optimized_config.pop("short", None)
        config = deep_merge(config, optimized_config)
        config["_optimized_defaults"] = {
            "source": str(optimized_source),
            "generated_at": optimized.get("generated_at"),
            "iterations_per_strategy": optimized.get("iterations_per_strategy"),
            "seed": optimized.get("seed"),
            "caveat": optimized.get("caveat"),
            "short_universe_version": optimized.get("short_universe_version"),
        }
    smallcap_optimized, smallcap_source = load_smallcap_optimized_config_payload()
    if (
        smallcap_optimized.get("config")
        and smallcap_optimized.get("smallcap_universe_version", SMALLCAP_UNIVERSE_VERSION)
        == SMALLCAP_UNIVERSE_VERSION
    ):
        source_config = smallcap_optimized["config"]
        smallcap_override = (
            source_config.get("smallcap")
            if isinstance(source_config.get("smallcap"), dict)
            else source_config
        )
        config["smallcap"] = deep_merge(config["smallcap"], smallcap_override)
        # 小盘优化器固定不搜索、不启用历史高点回撤过滤；旧优化文件中的 True
        # 不得继续把实盘候选池从完整 is_hot_money 成员压缩成少量样本。
        config["smallcap"]["require_high_drawdown"] = False
        config["smallcap"]["min_high_drawdown_pct"] = DEFAULT_CONFIG["smallcap"]["min_high_drawdown_pct"]
        config["smallcap"]["top_n"] = min(
            SMALLCAP_DEFAULT_TOP_N,
            max(1, int(config["smallcap"].get("top_n") or SMALLCAP_DEFAULT_TOP_N)),
        )
        # 旧文件或手工编辑不得把已明确剔除的因子重新带回前端。
        config["smallcap"]["weights"] = {
            spec.key: config["smallcap"].get("weights", {}).get(spec.key, spec.default_weight)
            for spec in SMALLCAP_FACTORS
        }
        config["_smallcap_optimized_defaults"] = {
            "source": str(smallcap_source),
            "generated_at": smallcap_optimized.get("generated_at"),
            "iterations": smallcap_optimized.get("iterations"),
            "seed": smallcap_optimized.get("seed"),
            "caveat": smallcap_optimized.get("caveat"),
            "smallcap_universe_version": smallcap_optimized.get("smallcap_universe_version"),
        }
    # 历史优化文件可能仍保存这些长线因子的非零权重。它们可以在页面上
    # 手动调整，但不得继续作为自动加载的默认值。
    long_weights = config["long"].setdefault("weights", {})
    for key in LONG_FIXED_ZERO_WEIGHT_KEYS:
        long_weights[key] = 0.0
    return config


def get_factor_registry() -> Dict[str, Any]:
    return {
        "long": [spec.to_dict() for spec in LONG_FACTORS],
        "smallcap": [spec.to_dict() for spec in SMALLCAP_FACTORS],
        "short": [spec.to_dict() for spec in SHORT_FACTORS],
        "total": len(LONG_FACTORS) + len(SMALLCAP_FACTORS) + len(SHORT_FACTORS),
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
            "event_date": item.get("event_date"),
            "industry": item.get("industry"),
            "sw2_industry": item.get("sw2_industry"),
            "sw3_industry": item.get("sw3_industry"),
            "sw3_segment_code": item.get("sw3_segment_code"),
            "score": item.get("score"),
            "data_quality": item.get("data_quality"),
            "weighted_data_quality": item.get("weighted_data_quality"),
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
        return {
            "score_range": None,
            "top_factor_coverage": [],
            "coverage_sections": [],
            "groups": {},
        }
    scores = [item["score"] for item in scored]
    groups: Dict[str, float] = defaultdict(float)
    for spec in specs:
        groups[spec.group] += spec.default_weight
    return {
        "score_range": [round(min(scores), 2), round(max(scores), 2)],
        "avg_score": round(mean(scores), 2),
        "avg_data_quality": round(mean(item.get("data_quality", 0) for item in scored), 3),
        "avg_weighted_data_quality": round(
            mean(
                first_not_none(
                    item.get("weighted_data_quality"), item.get("data_quality"), 0.0
                ) or 0.0
                for item in scored
            ),
            3,
        ),
        "coverage_sections": factor_source_coverage(scored, specs),
        "top_factor_coverage": factor_coverage(scored[: min(20, len(scored))], specs),
        "groups": {k: round(v, 3) for k, v in sorted(groups.items())},
    }


def factor_coverage_bucket(spec: FactorSpec) -> Tuple[str, str]:
    if spec.strategy in {"long", "smallcap"}:
        if spec.key in LONG_PRICE_COVERAGE_KEYS:
            return "price", "价量"
        if spec.key in LONG_CAPITAL_COVERAGE_KEYS:
            return "capital", "资金面"
        return "fundamental", "财务/估值"
    if spec.group in {"技术", "价量Alpha"}:
        return "technical", "价量技术"
    if spec.group == "风控":
        return "risk", "风控"
    return "seats", "龙虎榜/席位"


def factor_source_coverage(
        items: List[Dict[str, Any]], specs: List[FactorSpec]
) -> List[Dict[str, Any]]:
    if not items:
        return []
    buckets: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for spec in specs:
        key, label = factor_coverage_bucket(spec)
        bucket = buckets.setdefault(key, {
            "key": key,
            "label": label,
            "factor_count": 0,
            "present": 0,
        })
        bucket["factor_count"] += 1
        bucket["present"] += sum(
            1
            for item in items
            if safe_float((item.get("raw_factors") or {}).get(spec.key)) is not None
        )
    rows = []
    for bucket in buckets.values():
        total = len(items) * bucket["factor_count"]
        rows.append({
            "key": bucket["key"],
            "label": bucket["label"],
            "factor_count": bucket["factor_count"],
            "coverage": round(bucket["present"] / total if total else 0.0, 3),
        })
    if specs and specs[0].strategy in {"long", "smallcap"}:
        order = {"fundamental": 0, "price": 1, "capital": 2}
        rows.sort(key=lambda row: order.get(row["key"], 99))
    return rows


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


def long_data_notes(
        candidates: List[Dict[str, Any]], *, include_csi_note: bool = True
) -> List[str]:
    notes = []
    if candidates:
        avg_quality = mean(item.get("data_quality", 0) for item in candidates)
        if avg_quality < 0.7:
            notes.append("Long strategy data coverage is below 70%; enrich financial/price history for production use.")
    if include_csi_note:
        notes.append("CSI300 long-term membership is proxied unless historical constituent snapshots are added.")
    return notes


def short_data_notes(candidates: List[Dict[str, Any]]) -> List[str]:
    notes = []
    if candidates and all("缺少席位级followers明细" in item.get("warnings", []) for item in candidates[:5]):
        notes.append \
            ("Top short picks lack seat-level detail; run stock_crawl_hot_money.py to refresh short signals.")
    return notes


def self_review_summary() -> Dict[str, Any]:
    return {
        "factor_count_ok": len(LONG_FACTORS) + len(SMALLCAP_FACTORS) + len(SHORT_FACTORS) >= 20,
        "long_factor_count": len(LONG_FACTORS),
        "smallcap_factor_count": len(SMALLCAP_FACTORS),
        "short_factor_count": len(SHORT_FACTORS),
        "old_files_modified": False,
        "network_required": False,
        "known_limits": [
            "Historical CSI300 membership is exact only if historical constituent files are later supplied.",
            "Long-horizon excess-return validation needs multi-year daily prices; current local fundamental files mostly contain recent daily samples.",
            "Short strategy uses the latest SQLite short-signal snapshot; refresh it before live trading.",
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_cli_summary(result: Dict[str, Any], top: int = 10) -> None:
    print(f"Generated: {result['generated_at']}")
    print(f"Total factors: {result['factor_total']}")
    for key in ("long", "smallcap", "short"):
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
    parser = argparse.ArgumentParser(description="Advanced A-share long/small-cap/short strategy engine")
    parser.add_argument("--strategy", choices=["all", "long", "smallcap", "short"], default="all")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--persist", action="store_true", help=f"write {OUTPUT_FILE}")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help=f"rebuild {LIVE_CANDIDATE_CACHE_FILE} before scoring")
    parser.add_argument("--json", action="store_true", help="print full JSON")
    args = parser.parse_args()

    if args.rebuild_cache:
        clear_live_candidate_cache()

    config = get_default_config()
    if args.strategy == "long":
        config["smallcap"]["enabled"] = False
        config["short"]["enabled"] = False
    elif args.strategy == "smallcap":
        config["long"]["enabled"] = False
        config["short"]["enabled"] = False
    elif args.strategy == "short":
        config["long"]["enabled"] = False
        config["smallcap"]["enabled"] = False

    result = run_strategies(config, persist=args.persist)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_cli_summary(result, top=args.top)
        if args.persist:
            print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
