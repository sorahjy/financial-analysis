
"""
Parameter search and proxy backtesting for stock_advanced_strategies.py.

The optimizer deliberately separates two layers:

1. price_backtest: uses only locally available price rows. This is the most
   honest metric, but current local files do not contain enough multi-year
   history for the requested 2-5 year horizon.
2. proxy_objective: used when the local price sample is too short. It combines
   excess return on the available recent window with risk, hit rate, and for
   short-term Dragon Tiger List picks, event-quality statistics.

Run 1500 Optuna/TPE trials per strategy (long, smallcap and short use separate worker processes):
    python3 -B stock_strategy_optimizer.py --iterations 1500

Only re-optimize the short strategy after its universe changes:
    python3 -B stock_strategy_optimizer.py --strategy short --iterations 1500

Only re-optimize the small-cap long-factor strategy:
    python3 -B stock_strategy_optimizer.py --strategy smallcap --iterations 1500
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
import os
import random
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from html import escape as html_escape
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import optuna
    import warnings as _warnings
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # multivariate TPE 在本版本仍标记 experimental，但更适合相关的权重维度；静音其提示。
    _warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
except ImportError:  # 未安装时回退随机搜索
    optuna = None

try:
    from tqdm import tqdm
except ImportError:  # tqdm 只影响 CLI 进度展示，不影响优化逻辑
    tqdm = None

import numpy as np

import stock_storage
from stock_advanced_strategies import (
    DATA_DIR,
    DEFAULT_CONFIG,
    LONG_FACTORS,
    LONG_FIXED_ZERO_WEIGHT_KEYS,
    SMALLCAP_FACTORS,
    SMALLCAP_UNIVERSE_VERSION,
    SHORT_FACTORS,
    SHORT_UNIVERSE_VERSION,
    apply_scores,
    build_long_candidates,
    build_smallcap_candidates,
    build_short_candidates,
    clean_round,
    combined_recent_rows,
    deep_merge,
    direct_factor_score,
    first_not_none,
    load_segment_leader_codes,
    load_hot_money_codes,
    load_cn_stock_index,
    load_fundamental_stocks,
    passes_long_hard_filters,
    passes_smallcap_hard_filters,
    percentile_factor_scores,
    price_history_rows,
    rerank_scored,
    safe_float,
)


OUTPUT_FILE = DATA_DIR / "stock_strategy_optimization.json"
OPTIMIZED_CONFIG_FILE = DATA_DIR / "stock_strategy_optimized_config.json"
SMALLCAP_OUTPUT_FILE = DATA_DIR / "stock_strategy_smallcap_optimization.json"
SMALLCAP_OPTIMIZED_CONFIG_FILE = DATA_DIR / "stock_strategy_smallcap_optimized_config.json"
# 长线超额基准：沪深300 与 中证500 按日等权再平衡的混合指数（各占50%）。
# 不直接用中证800：中证800按市值加权、前300只占权重大头，走势≈沪深300，体现不出中盘；
# 50/50 等权混合让大盘与中盘平权，更贴合本策略选股域。
# 换基准只改 BENCHMARK_COMPONENTS（增删成分即可，单成分时退化为该指数本身）；
# csi300_current 等仍是"指数成分"选股因子；当前优化器将其权重固定为0。
BENCHMARK_NAME = "沪深300+中证500等权"
BENCHMARK_COMPONENTS = [
    ("510310", DATA_DIR / "csi300_etf_nav.json"),  # 沪深300 ETF
    ("510580", DATA_DIR / "csi500_etf_nav.json"),  # 中证500 ETF
]
LONG_FOLD_PATH_CHART_FILE = DATA_DIR / "stock_strategy_best_fold_paths.svg"
SMALLCAP_FOLD_PATH_CHART_FILE = DATA_DIR / "stock_strategy_smallcap_fold_paths.svg"
DEFAULT_OPTIMIZATION_ITERATIONS = 1500

# 长线 walk-forward 回测参数：利用 data/stock_data/*.history 的多年日线，
# 每 40 个交易日取一折，固定持有 40 个交易日；相邻完整折首尾衔接。
LONG_HOLD_CHOICES = [60]   # 长线持有期固定为40交易日
LONG_FOLD_STEP_TD = 60     # 折锚点间隔(交易日)，与持有期一致，避免窗口重叠
LONG_MAX_LOOKBACK_TD = 2400
# 小盘独立回测合同：近5年、每20日一折、持有20日，折之间首尾衔接。
SMALLCAP_HOLD_CHOICES = [20]
SMALLCAP_FOLD_STEP_TD = 20
SMALLCAP_MAX_LOOKBACK_TD = 5 * 250
LONG_COST = 0.004          # 单折买卖往返成本（佣金+冲击的粗略值）
LONG_MIN_VALID_PICKS = 5   # 一折内至少几只持仓有价格数据才计入
LONG_MIN_FOLDS = 16        # 有效折数下限，低于此的配置不参与选优(防少数折彩票配置)
LONG_SOFT_TARGET_FOLDS = 40
LONG_FOLD_COUNT_PENALTY = 1.25
LONG_RECENCY_WEIGHT_MIN = 1.0
LONG_RECENCY_WEIGHT_MAX = 1.0
LONG_FIXED_ZERO_WEIGHTS = set(LONG_FIXED_ZERO_WEIGHT_KEYS)
LONG_SEARCHABLE_FACTORS = [f for f in LONG_FACTORS if f.key not in LONG_FIXED_ZERO_WEIGHTS]
SMALLCAP_FIXED_ZERO_WEIGHTS = {"industry_leadership"}
SMALLCAP_SEARCHABLE_FACTORS = [
    f for f in SMALLCAP_FACTORS if f.key not in SMALLCAP_FIXED_ZERO_WEIGHTS
]
LONG_FIXED_MIN_MARKET_CAP_YI = 0
LONG_FIXED_MIN_SCORE = 50
# 去重叠/尾部稳健参数（针对"最差折超额差"专门加的口径）：
LONG_MIN_INDEP_FOLDS = 4        # 去相关后折数下限，不足则尾部统计回退到稠密折
LONG_TAIL_FLOOR = 0.15          # 独立折最差超额跌破 -15% 的部分按硬惩罚计，直击最差折
LONG_DOWNSIDE_DEV_FLOOR = 0.04  # Sortino 下行偏差下限：样本内几乎无下行折时防止比率爆炸
LONG_SORTINO_CAP = 4.0          # Sortino 贡献封顶(±)，使其只作配平项、不主导选优
# ③ 防御因子权重下限：不允许搜索把低波动/低负债/保守投资(CMA)直接关到0，强制保留风险预算。
LONG_WEIGHT_FLOORS = {
    "low_volatility": 0.30,
    "debt_safety": 0.20,
    "asset_growth": 0.20,
}
# 高点回撤(抄底)过滤开关参与搜索；开启时搜索"高点至今跌幅下限"40~70%。
LONG_HIGH_DD_PCT_CHOICES = list(range(40, 71))
LONG_MAX_ABS_DAILY_RETURN = 0.45
OPTUNA_MIN_STARTUP_TRIALS = 40
OPTUNA_STARTUP_TRIAL_FRACTION = 0.35
OPTUNA_EI_CANDIDATES = 64
_RETURN_PREFIX_CACHE: Dict[Tuple[int, int, str, str], Tuple[List[int], List[float]]] = {}
RESEARCH_BASIS = [
    "Fama-French: size, value, profitability, investment; plus momentum/reversal sorts from the Kenneth French data library.",
    "Barra/MSCI-style families: size, value, momentum, volatility, liquidity, growth, leverage and quality.",
    "Conservative formula: low volatility + momentum + payout yield.",
    "WorldQuant-style short-horizon price-volume alphas.",
    "Long-horizon capital signals: shareholder-count change and buyback announcements; recent Dragon Tiger List avoidance is retained as a disabled diagnostic.",
    "A-share Dragon Tiger List event factors: net buy, reason strength, hot-money seat network, institution/hot-money resonance and tradability risk.",
]


def clone_config() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_CONFIG)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pct_return_from_rows(rows: List[Dict[str, Any]], entry_idx: int, hold_days: int) -> Optional[float]:
    if not rows:
        return None
    if entry_idx < 0:
        entry_idx = len(rows) + entry_idx
    exit_idx = entry_idx + hold_days
    if entry_idx < 0 or exit_idx >= len(rows):
        return None
    entry = safe_float(rows[entry_idx].get("close"))
    exit_ = safe_float(rows[exit_idx].get("close"))
    if entry is None or exit_ is None or entry <= 0:
        return None
    return exit_ / entry - 1.0


def recent_series_map() -> Dict[str, List[Dict[str, Any]]]:
    stocks = load_fundamental_stocks()
    return {code: combined_recent_rows(code, stock) for code, stock in stocks.items()}


def full_series_map() -> Dict[str, List[Dict[str, Any]]]:
    """长线回测用全量历史日线（stock_data.history 优先，最长约10年；缺失回退近段数据）。"""
    stocks = load_fundamental_stocks()
    cn_index = load_cn_stock_index()
    return {
        code: (cn_index.get(code) or {}).get("records") or price_history_rows(code, stock)
        for code, stock in stocks.items()
    }


def long_price_history_health(series: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    min_rows = max(LONG_HOLD_CHOICES) + 1
    price_rows = 0
    usable_codes = 0
    for rows in series.values():
        closes = [safe_float(row.get("close")) for row in rows or []]
        valid_closes = [close for close in closes if close is not None and close > 0]
        price_rows += len(valid_closes)
        if len(valid_closes) >= min_rows:
            usable_codes += 1
    return {
        "series_codes": len(series),
        "price_rows": price_rows,
        "usable_price_codes": usable_codes,
        "min_required_rows_per_code": min_rows,
    }


def ensure_long_price_history(series: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    health = long_price_history_health(series)
    if health["usable_price_codes"] < LONG_MIN_VALID_PICKS:
        raise RuntimeError(
            "长线优化缺少可用于前推收益的 OHLCV 收盘价："
            f"可用价格股票 {health['usable_price_codes']}/{health['series_codes']}，"
            f"非空 close 行 {health['price_rows']}，"
            f"每只至少需要 {health['min_required_rows_per_code']} 行。"
            "请先刷新 stock_history 的 daily_close/OHLCV，再运行 stock_strategy_optimizer.py。"
        )
    return health


def long_optimizer_universe_codes(
    series: Dict[str, List[Dict[str, Any]]],
    config: Dict[str, Any],
) -> Tuple[set, List[str]]:
    """长线优化股票池：足够长日线 ∩ SW3 细分龙头池。"""
    min_hist = max(LONG_HOLD_CHOICES) + 260
    codes = {code for code, rows in series.items() if len(rows) >= min_hist}
    notes: List[str] = []
    if bool(config.get("use_segment_leaders", True)):
        leader_codes = load_segment_leader_codes()
        if leader_codes:
            codes &= leader_codes
            notes.append(
                "长线优化候选池显式限制为 stock_crawl_segment_leaders.py 生成的 "
                f"SW3 细分龙头池：{len(leader_codes)} 只龙头，其中 {len(codes)} 只具备至少 {min_hist} 行历史。"
            )
        else:
            notes.append(
                "sw3_member.is_leader 为空；长线优化暂回退到具备多年日线的全量股票池。"
                "请先运行 python stock_crawl_segment_leaders.py crawl。"
            )
    return codes, notes


def smallcap_optimizer_universe_codes(
    series: Dict[str, List[Dict[str, Any]]],
    config: Dict[str, Any],
) -> Tuple[set, List[str]]:
    """小盘优化股票池：足够长日线 ∩ 当前游资小盘池。"""
    _ = config
    min_hist = max(SMALLCAP_HOLD_CHOICES) + 260
    deep_codes = {code for code, rows in series.items() if len(rows) >= min_hist}
    hot_money_codes = load_hot_money_codes()
    codes = deep_codes & hot_money_codes
    return codes, [
        "小盘优化候选池复用短线/游资雷达的 is_hot_money=1 成员："
        f"{len(hot_money_codes)} 只，其中 {len(codes)} 只具备至少 {min_hist} 行历史。"
    ]


def long_anchor_offsets(hold_td: int) -> List[int]:
    """折锚点：距最新交易日的偏移（交易日数），保证持有期之后仍有出场价。"""
    return list(range(hold_td + 1, LONG_MAX_LOOKBACK_TD, LONG_FOLD_STEP_TD))


def long_partial_anchor_offsets(hold_td: int) -> List[int]:
    """走势图专用：最新端尚未走满持有期的锚点，不参与优化选优。"""
    first = max(1, hold_td + 1 - LONG_FOLD_STEP_TD)
    return list(range(first, 0, -LONG_FOLD_STEP_TD))


def smallcap_anchor_offsets(hold_td: int) -> List[int]:
    return list(range(hold_td + 1, SMALLCAP_MAX_LOOKBACK_TD, SMALLCAP_FOLD_STEP_TD))


def smallcap_partial_anchor_offsets(hold_td: int) -> List[int]:
    first = max(1, hold_td + 1 - SMALLCAP_FOLD_STEP_TD)
    return list(range(first, 0, -SMALLCAP_FOLD_STEP_TD))


def fold_calendar(series: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """全市场交易日历(各股日期并集，升序)，把折偏移映射成 PIT 时点日期。"""
    dates: set = set()
    for rows in series.values():
        for r in rows:
            d = str(r.get("date", ""))
            if len(d) == 10:
                dates.add(d)
    return sorted(dates)


def _load_etf_nav_levels(code: str, path: Path) -> Dict[str, float]:
    """读单个 ETF 累计净值 -> {date: 累计净值}。优先 DB(index_nav)，回退 JSON 文件。"""
    levels: Dict[str, float] = {}
    records: List[Dict[str, Any]] = []
    try:
        conn = stock_storage.connect()
        try:
            records = stock_storage.load_index_nav(conn, code).get("records", [])
        finally:
            conn.close()
    except Exception:
        records = []
    if not records:
        try:
            with open(path, "r", encoding="utf-8") as fp:
                records = json.load(fp).get("records", [])
        except (OSError, json.JSONDecodeError):
            records = []
    for row in records:
        date = str(row.get("date", ""))
        close = first_not_none(row.get("nav_acc"), row.get("nav"), row.get("close"))
        if len(date) == 10 and close is not None and close > 0:
            levels[date] = float(close)
    return levels


def load_benchmark_series() -> Tuple[List[Dict[str, Any]], List[str]]:
    """加载基准 = BENCHMARK_COMPONENTS 按日等权再平衡的混合累计净值序列。

    单成分时退化为该指数本身；多成分时取各成分交易日交集，用每日各成分等权
    毛收益滚动出混合净值——即每天收益取各成分简单平均(沪深300+中证500)/2，
    保持恒定 50/50，消除买入持有下的权重漂移。
    """
    comps: List[Tuple[str, Dict[str, float]]] = []
    for code, path in BENCHMARK_COMPONENTS:
        levels = _load_etf_nav_levels(code, path)
        if not levels:
            return [], [f"基准成分 {code}（{path.name} / index_nav）没有可用累计净值。"]
        comps.append((code, levels))

    common = sorted(set.intersection(*[set(lv.keys()) for _, lv in comps]))
    if len(common) < 2:
        return [], [f"{BENCHMARK_NAME}基准成分交易日交集不足，无法构建。"]

    weight = 1.0 / len(comps)
    rows = [{"date": common[0], "close": 1.0}]
    for prev_date, date in zip(common, common[1:]):
        gross = sum(weight * (lv[date] / lv[prev_date]) for _, lv in comps)
        rows.append({"date": date, "close": rows[-1]["close"] * gross})

    codes = "+".join(code for code, _ in comps)
    return rows, [
        f"{BENCHMARK_NAME}基准 = {codes} 按日等权再平衡混合累计净值："
        f"{rows[0]['date']} ~ {rows[-1]['date']}（{len(rows)}个交易日）。"
    ]

def decorrelate_pairs(pairs: List[Dict[str, Any]], min_gap_td: int) -> List[Dict[str, Any]]:
    """按交易日间隔贪心抽稀重叠折，降低自相关，用于尾部/IR等风险统计。

    当前默认 hold=60/step=60，完整折基本首尾衔接、不再有旧版 125/30 的窗口重叠。
    这个函数保留给未来若持有期重新大于步长时使用，尾部统计会优先看去相关子样本。
    """
    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda p: p.get("cal_idx", 0))
    kept = [ordered[0]]
    for pair in ordered[1:]:
        if pair.get("cal_idx", 0) - kept[-1].get("cal_idx", 0) >= min_gap_td:
            kept.append(pair)
    return kept


def random_fold_split(
    pairs: List[Dict[str, Any]],
    train_frac: float = 0.6,
    salt: str = "longsplit",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按折随机打乱后切分训练/验证（约 train_frac / 1-train_frac），不做时间隔离。

    用户指定口径：取消 embargo 与时间分块，训练/验证折在时间上混合、各覆盖全部 regime。
    当前默认 hold=60/step=60，完整折首尾衔接，不再有旧版长持有窗口造成的重叠泄漏。
    每折按其 as_of 的确定性随机键排序后切分，保证可复现、且各配置用同一划分。
    """
    if len(pairs) < 2:
        return list(pairs), []
    keyed = sorted(pairs, key=lambda p: random.Random(f"{salt}-{p['as_of']}").random())
    k = max(1, min(int(len(keyed) * train_frac), len(keyed) - 1))
    return keyed[:k], keyed[k:]


def apply_recency_weights(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """给每个折设置优化统计权重；当前口径为所有折等权 1.0x。"""
    if not pairs:
        return []
    idxs = [safe_float(pair.get("cal_idx")) for pair in pairs]
    valid = [idx for idx in idxs if idx is not None]
    if not valid:
        return [{**pair, "fold_weight": 1.0} for pair in pairs]
    lo = min(valid)
    hi = max(valid)
    span = hi - lo
    weighted = []
    for pair, idx in zip(pairs, idxs):
        if idx is None or span <= 0:
            weight = 1.0
        else:
            pos = (idx - lo) / span
            weight = LONG_RECENCY_WEIGHT_MIN + pos * (LONG_RECENCY_WEIGHT_MAX - LONG_RECENCY_WEIGHT_MIN)
        weighted.append({**pair, "fold_weight": round(weight, 6)})
    return weighted


def fold_weight(pair: Dict[str, Any]) -> float:
    weight = safe_float(pair.get("fold_weight"))
    return weight if weight is not None and weight > 0 else 1.0


def weighted_mean(values: List[float], weights: List[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return mean(values) if values else 0.0
    return sum(value * weight for value, weight in zip(values, weights)) / total


def weighted_std(values: List[float], weights: List[float], center: float) -> float:
    total = sum(weights)
    if total <= 0:
        return pstdev(values) if len(values) > 1 else 0.0
    return math.sqrt(sum(weight * (value - center) ** 2 for value, weight in zip(values, weights)) / total)


def weighted_median(values: List[float], weights: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    total = sum(max(0.0, weight) for _, weight in ordered)
    if total <= 0:
        mid = len(ordered) // 2
        return ordered[mid][0] if len(ordered) % 2 else (ordered[mid - 1][0] + ordered[mid][0]) / 2.0
    acc = 0.0
    cutoff = total / 2.0
    for value, weight in ordered:
        acc += max(0.0, weight)
        if acc >= cutoff:
            return value
    return ordered[-1][0]


def weighted_tail_mean(values: List[float], weights: List[float], fraction: float = 0.2) -> float:
    if not values:
        return 0.0
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    total = sum(max(0.0, weight) for _, weight in ordered)
    if total <= 0:
        tail_k = max(1, math.ceil(len(values) * fraction))
        return mean(value for value, _ in ordered[:tail_k])
    target = total * fraction
    used = 0.0
    accum = 0.0
    for value, weight in ordered:
        weight = max(0.0, weight)
        take = min(weight, target - used)
        if take <= 0:
            break
        accum += value * take
        used += take
        if used >= target:
            break
    return accum / used if used > 0 else ordered[0][0]


def excess_risk_stats(pairs: List[Dict[str, Any]], decorr_gap_td: int) -> Dict[str, float]:
    """尾部/风险统计：最差折、CVaR、下行偏差在去相关子样本上算；回撤用全样本均值。"""
    indep = decorrelate_pairs(pairs, decorr_gap_td)
    use = indep if len(indep) >= LONG_MIN_INDEP_FOLDS else pairs
    excess = [p["excess_return"] for p in use]
    weights = [fold_weight(p) for p in use]
    worst_pair = min(use, key=lambda p: p["excess_return"])
    worst = worst_pair["excess_return"]
    cvar = weighted_tail_mean(excess, weights)        # 最差20%折的加权平均超额
    total_weight = sum(weights)
    downside_dev = (
        math.sqrt(sum(w * e * e for e, w in zip(excess, weights) if e < 0) / total_weight)
        if total_weight > 0 and any(e < 0 for e in excess)
        else 0.0
    )
    mdd_values = [p.get("portfolio_max_drawdown", 0.0) for p in pairs]
    mdd_weights = [fold_weight(p) for p in pairs]
    mean_fold_mdd = weighted_mean(mdd_values, mdd_weights) if pairs else 0.0
    return {
        "worst": worst,
        "worst_weight": fold_weight(worst_pair),
        "cvar": cvar,
        "downside_dev": downside_dev,
        "mean_fold_mdd": mean_fold_mdd,
        "indep_folds": len(indep),
    }


def path_max_drawdown(values: List[float]) -> Optional[float]:
    if not values:
        return None
    peak = values[0]
    max_dd = 0.0
    for value in values:
        if value is None or value <= 0:
            continue
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak)
    return max_dd


def row_index_on_or_before(rows: List[Dict[str, Any]], target_date: str) -> Optional[int]:
    if not rows:
        return None
    lo = 0
    hi = len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if str(rows[mid].get("date", "")) <= target_date:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1 if lo > 0 else None


def row_index_on_date(rows: List[Dict[str, Any]], target_date: str) -> Optional[int]:
    idx = row_index_on_or_before(rows, target_date)
    if idx is None or str(rows[idx].get("date", "")) != target_date:
        return None
    return idx


def has_trade_on_date(
    series: Optional[Dict[str, List[Dict[str, Any]]]],
    code: str,
    trade_date: Optional[str],
) -> bool:
    """Whether code has an actual daily bar on trade_date."""
    if not series or not trade_date:
        return True
    return row_index_on_date(series.get(code) or [], trade_date) is not None


def price_path_on_dates(rows: List[Dict[str, Any]], dates: List[str]) -> List[Optional[float]]:
    start_idx = row_index_on_or_before(rows, dates[0]) if dates else None
    if start_idx is None:
        return [None for _ in dates]
    idx = start_idx
    values = []
    for date in dates:
        while idx + 1 < len(rows) and str(rows[idx + 1].get("date", "")) <= date:
            idx += 1
        values.append(safe_float(rows[idx].get("close")))
    return values


def _daily_return_from_row(rows: List[Dict[str, Any]], idx: int) -> Optional[float]:
    if idx <= 0 or idx >= len(rows):
        return None
    change_pct = safe_float(rows[idx].get("change_pct"))
    if change_pct is not None:
        daily_return = change_pct / 100.0
    else:
        prev_close = safe_float(rows[idx - 1].get("close"))
        close = safe_float(rows[idx].get("close"))
        if prev_close is None or close is None or prev_close <= 0 or close <= 0:
            return None
        daily_return = close / prev_close - 1.0
    if (
        not math.isfinite(daily_return)
        or daily_return <= -1.0
        or abs(daily_return) > LONG_MAX_ABS_DAILY_RETURN
    ):
        return None
    return daily_return


def return_prefix_for_rows(rows: List[Dict[str, Any]]) -> Tuple[List[int], List[float]]:
    if not rows:
        return [], []
    key = (
        id(rows),
        len(rows),
        str(rows[0].get("date", "")),
        str(rows[-1].get("date", "")),
    )
    cached = _RETURN_PREFIX_CACHE.get(key)
    if cached is not None:
        return cached

    segment_ids = [0] * len(rows)
    cum_logs = [0.0] * len(rows)
    segment_id = 0
    cum_log = 0.0
    for idx in range(1, len(rows)):
        daily_return = _daily_return_from_row(rows, idx)
        if daily_return is None:
            segment_id += 1
            cum_log = 0.0
        else:
            cum_log += math.log1p(daily_return)
        segment_ids[idx] = segment_id
        cum_logs[idx] = cum_log

    value = (segment_ids, cum_logs)
    _RETURN_PREFIX_CACHE[key] = value
    return value


def return_path_on_dates(rows: List[Dict[str, Any]], dates: List[str]) -> Optional[List[float]]:
    if not dates:
        return None
    start_idx = row_index_on_date(rows, dates[0])
    if start_idx is None:
        return None
    if row_index_on_date(rows, dates[-1]) is None:
        return None
    segment_ids, cum_logs = return_prefix_for_rows(rows)
    if not segment_ids:
        return None
    base_segment = segment_ids[start_idx]
    base_log = cum_logs[start_idx]
    idx = start_idx
    values = []
    for date in dates:
        while idx + 1 < len(rows) and str(rows[idx + 1].get("date", "")) <= date:
            idx += 1
        if segment_ids[idx] != base_segment:
            return None
        values.append(math.exp(cum_logs[idx] - base_log))
    return values


def benchmark_window_from_date(
    benchmark_series: List[Dict[str, Any]],
    as_of: str,
    hold_td: int,
    *,
    allow_partial: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    entry_idx = row_index_on_or_before(benchmark_series, as_of)
    if entry_idx is None:
        return None
    exit_idx = entry_idx + hold_td
    if exit_idx >= len(benchmark_series):
        if not allow_partial:
            return None
        exit_idx = len(benchmark_series) - 1
    if exit_idx <= entry_idx:
        return None
    return benchmark_series[entry_idx:exit_idx + 1]


def portfolio_fold_path(
    picks: List[Dict[str, Any]],
    series: Dict[str, List[Dict[str, Any]]],
    benchmark_series: List[Dict[str, Any]],
    as_of: str,
    hold_td: int,
    *,
    allow_partial: bool = False,
) -> Optional[Dict[str, Any]]:
    benchmark_window = benchmark_window_from_date(
        benchmark_series, as_of, hold_td, allow_partial=allow_partial
    )
    if not benchmark_window:
        return None
    dates = [row["date"] for row in benchmark_window]
    actual_hold_td = len(dates) - 1
    benchmark_prices = [safe_float(row.get("close")) for row in benchmark_window]
    benchmark_entry = benchmark_prices[0]
    benchmark_exit = benchmark_prices[-1]
    if benchmark_entry is None or benchmark_exit is None or benchmark_entry <= 0:
        return None
    benchmark_path = [price / benchmark_entry for price in benchmark_prices if price is not None]
    if len(benchmark_path) != len(dates):
        return None

    stock_paths = []
    for pick in picks:
        path = return_path_on_dates(series.get(pick["code"]) or [], dates)
        if path is None:
            continue
        stock_paths.append(path)
    if len(stock_paths) < LONG_MIN_VALID_PICKS:
        return None

    portfolio_path = []
    for idx in range(len(dates)):
        values = [path[idx] for path in stock_paths if path[idx] is not None]
        if len(values) < LONG_MIN_VALID_PICKS:
            return None
        portfolio_path.append(mean(values))

    portfolio_return = portfolio_path[-1] - 1.0 - LONG_COST
    benchmark_return = benchmark_exit / benchmark_entry - 1.0
    return {
        "as_of": as_of,
        "dates": dates,
        "target_hold_td": hold_td,
        "actual_hold_td": actual_hold_td,
        "partial": actual_hold_td < hold_td,
        "portfolio_path": portfolio_path,
        "benchmark_path": benchmark_path,
        "stock_count": len(stock_paths),
        "portfolio_return": portfolio_return,
        "benchmark_return": benchmark_return,
        "excess_return": portfolio_return - benchmark_return,
        "portfolio_max_drawdown": path_max_drawdown(portfolio_path) or 0.0,
        "benchmark_max_drawdown": path_max_drawdown(benchmark_path) or 0.0,
    }


def portfolio_fold_stats(
    picks: List[Dict[str, Any]],
    series: Dict[str, List[Dict[str, Any]]],
    benchmark_series: List[Dict[str, Any]],
    as_of: str,
    hold_td: int,
) -> Optional[Dict[str, float]]:
    path = portfolio_fold_path(picks, series, benchmark_series, as_of, hold_td)
    if path is None:
        return None
    return {
        "portfolio_return": path["portfolio_return"],
        "benchmark_return": path["benchmark_return"],
        "excess_return": path["excess_return"],
        "portfolio_max_drawdown": path["portfolio_max_drawdown"],
        "benchmark_max_drawdown": path["benchmark_max_drawdown"],
    }


def long_fold_summary(pairs: List[Dict[str, Any]], hold_td: int) -> Dict[str, Any]:
    if not pairs:
        return {"folds": 0}
    ordered = sorted(pairs, key=lambda row: row["as_of"])
    excess = [row["excess_return"] for row in ordered]
    portfolio = [row["portfolio_return"] for row in ordered]
    benchmark = [row["benchmark_return"] for row in ordered]
    weights = [fold_weight(row) for row in ordered]
    ann = 250.0 / hold_td
    mean_e = weighted_mean(excess, weights)
    vol = weighted_std(excess, weights, mean_e) if len(excess) > 1 else 0.0
    max_dd = max(row.get("portfolio_max_drawdown", 0.0) for row in ordered)
    benchmark_max_dd = max(row.get("benchmark_max_drawdown", 0.0) for row in ordered)
    risk = excess_risk_stats(ordered, max(1, hold_td // 2))
    sortino = mean_e / max(risk["downside_dev"], LONG_DOWNSIDE_DEV_FLOOR)
    return {
        "folds": len(excess),
        "indep_folds": risk["indep_folds"],
        "hold_td": hold_td,
        "avg_fold_weight": round(mean(weights), 4),
        "min_fold_weight": round(min(weights), 4),
        "max_fold_weight": round(max(weights), 4),
        "avg_portfolio_return_pct": round(weighted_mean(portfolio, weights) * 100, 3),
        "avg_benchmark_return_pct": round(weighted_mean(benchmark, weights) * 100, 3),
        "avg_excess_pct": round(mean_e * 100, 3),
        "avg_excess_ann_pct": round(mean_e * ann * 100, 3),
        "hit_rate": round(sum(w for e, w in zip(excess, weights) if e > 0) / sum(weights) * 100, 2),
        "ir": round(mean_e / vol, 3) if vol > 1e-9 else None,
        "downside_ir": round(sortino, 3) if sortino is not None else None,
        "worst_fold_excess_pct": round(min(excess) * 100, 3),
        "tail_cvar_excess_pct": round(risk["cvar"] * 100, 3),
        "avg_fold_max_drawdown_pct": round(risk["mean_fold_mdd"] * 100, 3),
        "max_drawdown_pct": round(max_dd * 100, 3) if max_dd is not None else None,
        "benchmark_max_drawdown_pct": round(benchmark_max_dd * 100, 3) if benchmark_max_dd is not None else None,
    }


def long_fold_objective(
    pairs: List[Dict[str, Any]],
    hold_td: int,
    decorr_gap_td: int,
) -> Optional[float]:
    """稳健型目标：中心趋势为主，下行风险(CVaR/回撤/Sortino)显式入目标，重罚最差折。

    ② 重写要点（针对"最差折超额差"）：
    - 命中率权重从 15 降到 8——重叠折会系统性虚高胜率，过去它主导了选优。
    - 用 Sortino(下行调整IR)替代普通IR，只罚下行波动，不罚上行弹性。
    - 新增 CVaR(最差20%折均值) 与 持有期内组合回撤 两项下行惩罚。
    - 独立折最差超额跌破 -LONG_TAIL_FLOOR 的部分按硬惩罚，直击最差折。
    风险项都在去相关子样本上计算，避免重叠折把尾部统计冲淡。
    """
    if not pairs:
        return None
    excess = [p["excess_return"] for p in pairs]
    weights = [fold_weight(p) for p in pairs]
    ann = 250.0 / hold_td
    median = weighted_median(excess, weights)
    mean_e = weighted_mean(excess, weights)
    hit = sum(w for e, w in zip(excess, weights) if e > 0) / sum(weights)
    risk = excess_risk_stats(pairs, decorr_gap_td)
    # 下行偏差设地板并对 Sortino 封顶：样本内几乎无下行折时比率会趋于无穷，
    # 不封顶会让"在训练折恰好没有亏损折"的退化配置靠巨大Sortino霸榜(过拟合)。
    downside_dev = max(risk["downside_dev"], LONG_DOWNSIDE_DEV_FLOOR)
    sortino = clamp(mean_e / downside_dev, -LONG_SORTINO_CAP, LONG_SORTINO_CAP)
    tail_breach = max(0.0, -risk["worst"] - LONG_TAIL_FLOOR)
    return (
        (median * 0.55 + mean_e * 0.45) * ann * 100   # 中心趋势(年化超额)
        + hit * 8.0                                    # 命中率(重叠会虚高,已降权)
        + sortino * 2.0                                # 下行调整信息比(Sortino)
        + risk["cvar"] * ann * 100 * 0.5               # 最差20%折均值,尾部差则扣分
        - risk["mean_fold_mdd"] * 100 * 0.15           # 持有期内组合回撤惩罚(新增)
        - tail_breach * risk["worst_weight"] * 100 * 0.8  # 最差折击穿地板的加权惩罚
    )


def long_validation_adjusted_objective(
    train_pairs: List[Dict[str, Any]],
    val_pairs: List[Dict[str, Any]],
    all_pairs: List[Dict[str, Any]],
    hold_td: int,
    decorr_gap_td: int,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """长线最终选优目标：训练/验证共同决定，并惩罚不稳健的漂亮结果。

    验证为随机打乱后的折切分(见 random_fold_split)，按用户口径不做时间隔离；
    当前默认 60 日持有 / 60 日起点间隔，完整折不再像旧版 125 日持有那样高度重叠。
    """
    train_obj = long_fold_objective(train_pairs, hold_td, decorr_gap_td)
    val_obj = long_fold_objective(val_pairs, hold_td, decorr_gap_td)
    if train_obj is None or val_obj is None:
        return None, {"reason": "missing_train_or_validation_folds"}

    train_summary = long_fold_summary(train_pairs, hold_td)
    val_summary = long_fold_summary(val_pairs, hold_td)
    full_summary = long_fold_summary(all_pairs, hold_td)
    train_ann = safe_float(train_summary.get("avg_excess_ann_pct")) or 0.0
    val_ann = safe_float(val_summary.get("avg_excess_ann_pct")) or 0.0
    full_max_mdd = safe_float(full_summary.get("max_drawdown_pct")) or 0.0

    blended = train_obj * 0.55 + val_obj * 0.45
    train_val_gap_penalty = abs(train_ann - val_ann) * 0.35
    fold_count_penalty = max(0, LONG_SOFT_TARGET_FOLDS - len(all_pairs)) * LONG_FOLD_COUNT_PENALTY
    negative_validation_penalty = max(0.0, -val_ann) * 0.8
    # 全样本最深折内回撤超过45%再线性惩罚(基准约37%)，压制高回撤的"漂亮"配置。
    worst_mdd_penalty = max(0.0, full_max_mdd - 45.0) * 0.10
    fold_weights = [fold_weight(pair) for pair in all_pairs]
    objective = (
        blended
        - train_val_gap_penalty
        - fold_count_penalty
        - negative_validation_penalty
        - worst_mdd_penalty
    )
    return objective, {
        "train_objective": round(train_obj, 5),
        "validation_objective": round(val_obj, 5),
        "blended_objective": round(blended, 5),
        "train_folds": len(train_pairs),
        "validation_folds": len(val_pairs),
        "fold_weight_min": round(min(fold_weights), 5),
        "fold_weight_max": round(max(fold_weights), 5),
        "train_val_gap_ann_pct": round(abs(train_ann - val_ann), 3),
        "train_val_gap_penalty": round(train_val_gap_penalty, 5),
        "fold_count_penalty": round(fold_count_penalty, 5),
        "negative_validation_penalty": round(negative_validation_penalty, 5),
        "worst_mdd_penalty": round(worst_mdd_penalty, 5),
    }


def convergence_summary(trace: List[Dict[str, Any]], iterations: int) -> Dict[str, Any]:
    if not trace:
        return {}

    best_obj = -math.inf
    improvements = []
    for row in trace:
        objective = safe_float(row.get("objective"))
        if objective is None:
            continue
        if objective > best_obj:
            best_obj = objective
            improvements.append({
                "iteration": row.get("iteration"),
                "objective": round(objective, 5),
            })

    ranked = sorted(trace, key=lambda row: row.get("objective", -math.inf), reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    top10_floor = ranked[min(9, len(ranked) - 1)]
    best_iteration = int(best.get("iteration", 0) or 0)
    last_improvement = int(improvements[-1]["iteration"]) if improvements else best_iteration
    top_objective = safe_float(best.get("objective")) or 0.0
    second_objective = safe_float(second.get("objective")) if second else None
    top10_floor_objective = safe_float(top10_floor.get("objective")) or 0.0

    return {
        "best_iteration": best_iteration,
        "last_improvement_iteration": last_improvement,
        "iterations_after_last_improvement": max(0, iterations - last_improvement),
        "best_in_last_10pct": best_iteration > iterations * 0.9,
        "improvement_count": len(improvements),
        "recent_improvements": improvements[-10:],
        "top_objective": round(top_objective, 5),
        "second_objective": round(second_objective, 5) if second_objective is not None else None,
        "top2_gap": round(top_objective - second_objective, 5) if second_objective is not None else None,
        "top10_floor_objective": round(top10_floor_objective, 5),
        "top10_gap": round(top_objective - top10_floor_objective, 5),
    }


def select_best_long_trace(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [
        row for row in trace
        if int((row.get("summary") or {}).get("folds") or 0) >= LONG_MIN_FOLDS
        and safe_float(row.get("objective")) is not None
        and safe_float(row.get("objective")) > -999.0
    ]
    if not valid:
        max_selected = max((int(row.get("selected_count") or 0) for row in trace), default=0)
        max_folds = max((int((row.get("summary") or {}).get("folds") or 0) for row in trace), default=0)
        raise RuntimeError(
            "长线优化没有产生有效 trial："
            f"共 {len(trace)} 次，最大入选 {max_selected} 只，最大有效折数 {max_folds}。"
            "请检查 OHLCV 价格数据、min_score/min_market_cap_yi/min_high_drawdown_pct 等过滤条件。"
        )
    return max(valid, key=lambda r: r["objective"])


_prepared_matrix_cache: Dict[Tuple[int, Tuple[str, ...]], Any] = {}
_prepared_short_arrays_cache: Dict[int, Any] = {}
_prepared_matrix_lock = threading.Lock()


def _prepared_matrix(prepared: List[Dict[str, Any]], specs: List[Any] = LONG_FACTORS):
    """把 prepared 的因子分预转成 (score_matrix[n股×n因子], items) numpy 矩阵，按 id+因子集缓存。

    供 score_prepared 向量化加权(矩阵×权重向量,释放 GIL)，各 trial 复用同一 pit 的矩阵。
    预热阶段构建一次后只读;n_jobs 多线程下 setdefault 幂等(CPython dict 原子)。"""
    factor_keys = tuple(spec.key for spec in specs)
    key = (id(prepared), factor_keys)
    cached = _prepared_matrix_cache.get(key)
    if cached is not None:
        return cached
    n = len(prepared)
    nf = len(specs)
    matrix = np.empty((n, nf), dtype=np.float64)
    items = []
    for i, row in enumerate(prepared):
        s = row["scores"]
        for j, spec in enumerate(specs):
            matrix[i, j] = s.get(spec.key, spec.missing_score)
        items.append(row["item"])
    cached = (matrix, items)
    _prepared_matrix_cache[key] = cached
    return cached


def _prepared_short_filter_arrays(prepared: List[Dict[str, Any]]):
    """短线硬过滤所需 raw factor 数组，trial 期复用，避免每轮逐股解析。"""
    key = id(prepared)
    cached = _prepared_short_arrays_cache.get(key)
    if cached is not None:
        return cached
    _, items = _prepared_matrix(prepared, SHORT_FACTORS)

    def raw_array(name: str) -> np.ndarray:
        out = np.empty(len(items), dtype=np.float64)
        for i, item in enumerate(items):
            value = safe_float((item.get("raw_factors") or {}).get(name))
            out[i] = value if value is not None else 0.0
        return out

    cached = {
        "lhb_recent_count": raw_array("lhb_recent_count"),
        "hot_money_concurrent": raw_array("hot_money_concurrent"),
        "limit_up_control": raw_array("limit_up_control"),
    }
    _prepared_short_arrays_cache[key] = cached
    return cached


def prepare_candidate_factor_scores(
    items: List[Dict[str, Any]],
    specs: List[Any],
) -> List[Dict[str, Any]]:
    percentile_cache: Dict[str, Dict[str, float]] = {}
    for spec in specs:
        if spec.score == "percentile":
            percentile_cache[spec.key] = percentile_factor_scores(items, spec)

    prepared = []
    for item in items:
        raw = item.get("raw_factors", {})
        scores = {}
        for spec in specs:
            if spec.score == "percentile":
                score = percentile_cache[spec.key].get(item["code"], spec.missing_score)
            else:
                score = direct_factor_score(raw.get(spec.key), spec)
            scores[spec.key] = score
        prepared.append({"item": item, "scores": scores})
    return prepared


def _score_prepared_long_style_candidates(
    prepared: List[Dict[str, Any]],
    config: Dict[str, Any],
    specs: List[Any],
    hard_filter,
    include_details: bool = False,
    *,
    entry_series: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    as_of: Optional[str] = None,
) -> List[Dict[str, Any]]:
    weights = config.get("weights", {})
    min_score = safe_float(config.get("min_score")) or 0.0
    top_n = max(0, int(first_not_none(config.get("top_n"), 30)))

    # 权重向量(config 级，所有股票相同)：缺失/脏值→默认权重，负→0。等价原逐因子 weight 逻辑。
    w = np.empty(len(specs), dtype=np.float64)
    for j, spec in enumerate(specs):
        v = safe_float(weights.get(spec.key))
        w[j] = max(0.0, v if v is not None else spec.default_weight)
    weight_sum = float(w.sum())
    positive_weight_count = int((w > 0.0).sum())

    scored = []
    if not include_details:
        # 主路径(评估)：totals = (因子分矩阵 @ 权重) / weight_sum，numpy 向量化释放 GIL；过滤逐股保留。
        matrix, items = _prepared_matrix(prepared, specs)
        with np.errstate(all="ignore"):   # 被硬过滤股票的 nan/inf/overflow 运算噪声;它们随后过滤掉,结果同原标量版
            totals = (matrix @ w) / weight_sum if weight_sum > 0 else np.zeros(len(items))
        for i, item in enumerate(items):
            if not hard_filter(item, config):
                continue
            if not has_trade_on_date(entry_series, item.get("code"), as_of):
                continue
            total = round(float(totals[i]), 2)
            if total < min_score:
                continue
            scored.append((total, item, positive_weight_count, None))
    else:
        # 明细路径(repr 选股，少量调用)：逐因子保留 factor_scores 明细。
        for row in prepared:
            item = row["item"]
            if not hard_filter(item, config):
                continue
            if not has_trade_on_date(entry_series, item.get("code"), as_of):
                continue
            raw = item.get("raw_factors", {})
            prepared_scores = row["scores"]
            factor_scores = {}
            weighted_sum = 0.0
            for j, spec in enumerate(specs):
                weight = float(w[j])
                score = prepared_scores.get(spec.key, spec.missing_score)
                if weight > 0:
                    weighted_sum += score * weight
                factor_scores[spec.key] = {
                    "label": spec.label,
                    "raw": clean_round(raw.get(spec.key), 6),
                    "score": round(score, 2),
                    "weight": round(weight, 4),
                }
            total = round(weighted_sum / weight_sum if weight_sum > 0 else 0.0, 2)
            if total < min_score:
                continue
            scored.append((total, item, positive_weight_count, factor_scores))

    scored.sort(key=lambda entry: entry[0], reverse=True)
    result = []
    for rank, (score, item, factor_count, factor_scores) in enumerate(scored[:top_n], 1):
        row = dict(item)
        row["score"] = score
        row["factor_count"] = factor_count
        row["data_quality"] = round(item.get("data_quality", 0.0), 3)
        row["rank"] = rank
        if factor_scores is not None:
            row["factor_scores"] = factor_scores
        result.append(row)
    return result


def score_prepared_long_candidates(
    prepared: List[Dict[str, Any]],
    config: Dict[str, Any],
    include_details: bool = False,
    *,
    entry_series: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    as_of: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return _score_prepared_long_style_candidates(
        prepared, config, LONG_FACTORS, passes_long_hard_filters, include_details,
        entry_series=entry_series, as_of=as_of,
    )


def score_prepared_smallcap_candidates(
    prepared: List[Dict[str, Any]],
    config: Dict[str, Any],
    include_details: bool = False,
    *,
    entry_series: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    as_of: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return _score_prepared_long_style_candidates(
        prepared, config, SMALLCAP_FACTORS, passes_smallcap_hard_filters, include_details,
        entry_series=entry_series, as_of=as_of,
    )


def score_long_candidates(
    broad_candidates: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    scored = apply_scores(broad_candidates, LONG_FACTORS, config.get("weights", {}))
    scored = rerank_scored([row for row in scored if passes_long_hard_filters(row, config)])
    min_score = safe_float(config.get("min_score")) or 0.0
    top_n = max(0, int(first_not_none(config.get("top_n"), 30)))
    return [row for row in scored if row["score"] >= min_score][:top_n]


def score_short_candidates(
    broad_candidates: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    filtered = []
    min_lhb = safe_float(config.get("min_lhb_count")) or 0.0
    min_concurrent = safe_float(config.get("min_hot_money_concurrent")) or 0.0
    max_consec = first_not_none(config.get("max_consecutive_limit_up"), 999.0)
    for item in broad_candidates:
        raw = item["raw_factors"]
        if (safe_float(raw.get("lhb_recent_count")) or 0.0) < min_lhb:
            continue
        if (safe_float(raw.get("hot_money_concurrent")) or 0.0) < min_concurrent:
            continue
        if (safe_float(raw.get("limit_up_control")) or 0.0) > max_consec:
            continue
        filtered.append(item)
    scored = apply_scores(filtered, SHORT_FACTORS, config.get("weights", {}))
    min_score = safe_float(config.get("min_score")) or 0.0
    top_n = max(0, int(first_not_none(config.get("top_n"), 30)))
    return [row for row in scored if row["score"] >= min_score][:top_n]


def score_prepared_short_candidates(
    prepared: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """短线 optimizer 主路径：预计算因子分后用 NumPy 向量化打分、过滤、取 topN。"""
    weights = config.get("weights", {})
    min_score = safe_float(config.get("min_score")) or 0.0
    min_lhb = safe_float(config.get("min_lhb_count")) or 0.0
    min_concurrent = safe_float(config.get("min_hot_money_concurrent")) or 0.0
    max_consec = first_not_none(config.get("max_consecutive_limit_up"), 999.0)
    max_consec = safe_float(max_consec)
    max_consec = max_consec if max_consec is not None else 999.0
    top_n = max(0, int(first_not_none(config.get("top_n"), 30)))
    if top_n <= 0 or not prepared:
        return []

    matrix, items = _prepared_matrix(prepared, SHORT_FACTORS)
    filters = _prepared_short_filter_arrays(prepared)
    w = np.empty(len(SHORT_FACTORS), dtype=np.float64)
    for j, spec in enumerate(SHORT_FACTORS):
        value = safe_float(weights.get(spec.key))
        w[j] = max(0.0, value if value is not None else spec.default_weight)
    weight_sum = float(w.sum())
    positive_weight_count = int((w > 0.0).sum())
    with np.errstate(all="ignore"):
        totals = (matrix @ w) / weight_sum if weight_sum > 0 else np.zeros(len(items))
    rounded_totals = np.round(totals, 2)

    mask = (
        (filters["lhb_recent_count"] >= min_lhb)
        & (filters["hot_money_concurrent"] >= min_concurrent)
        & (filters["limit_up_control"] <= max_consec)
        & (rounded_totals >= min_score)
    )
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return []
    order = np.lexsort((idxs, -rounded_totals[idxs]))
    selected = idxs[order[:top_n]]

    result = []
    for rank, idx in enumerate(selected, 1):
        item = items[int(idx)]
        row = dict(item)
        row["score"] = round(float(rounded_totals[int(idx)]), 2)
        row["factor_count"] = positive_weight_count
        row["data_quality"] = round(item.get("data_quality", 0.0), 3)
        row["rank"] = rank
        result.append(row)
    return result


def summarize_returns(returns: List[float], benchmark: List[float]) -> Dict[str, Any]:
    if not returns:
        return {
            "samples": 0,
            "avg_return": None,
            "avg_benchmark": mean(benchmark) if benchmark else None,
            "avg_excess": None,
            "hit_rate": None,
            "sharpe_like": None,
            "max_drawdown": None,
        }
    avg_ret = mean(returns)
    avg_bm = mean(benchmark) if benchmark else 0.0
    excess = [r - avg_bm for r in returns]
    vol = pstdev(returns) if len(returns) > 1 else 0.0
    curve = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        curve *= 1.0 + r
        peak = max(peak, curve)
        max_dd = max(max_dd, (peak - curve) / peak if peak else 0.0)
    return {
        "samples": len(returns),
        "avg_return": round(avg_ret * 100, 4),
        "avg_benchmark": round(avg_bm * 100, 4),
        "avg_excess": round(mean(excess) * 100, 4),
        "hit_rate": round(sum(1 for r in returns if r > avg_bm) / len(returns) * 100, 2),
        "sharpe_like": round(avg_ret / vol, 4) if vol > 1e-9 else None,
        "max_drawdown": round(max_dd * 100, 4),
    }


def short_actual_backtest(
    picks: List[Dict[str, Any]],
    series: Dict[str, List[Dict[str, Any]]],
    hold_days: int,
) -> Dict[str, Any]:
    returns = []
    for pick in picks:
        rows = series.get(pick["code"]) or []
        event_date = latest_pick_event_date(pick)
        if not event_date or len(rows) < hold_days + 2:
            continue
        rows = sorted(rows, key=lambda row: str(row.get("date", "")))
        entry_idx = None
        for idx, row in enumerate(rows):
            if str(row.get("date", "")) > event_date:
                entry_idx = idx
                break
        if entry_idx is None or entry_idx + hold_days >= len(rows):
            continue
        ret = pct_return_from_rows(rows, entry_idx, hold_days)
        if ret is not None:
            returns.append(ret - 0.004)
    return summarize_returns(returns, [])


def latest_pick_event_date(pick: Dict[str, Any]) -> Optional[str]:
    explicit = str(pick.get("event_date") or "")[:10]
    try:
        if explicit:
            return datetime.strptime(explicit, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        pass

    # Compatibility fallback for candidate artifacts produced before
    # event_date was added. New artifacts must use the explicit date computed
    # from the full signal payload, not this truncated display sample.
    dates = []
    for follower in pick.get("followers", []) or []:
        date = str(follower.get("date", ""))[:10]
        try:
            dates.append(datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d"))
        except ValueError:
            continue
    if not dates:
        return None
    return max(dates)


def short_proxy_return(pick: Dict[str, Any]) -> float:
    raw = pick.get("raw_factors", {})
    # dt/network 原引用 dragon_tiger_composite / seat_network_score，二者在因子精简时已删除，
    # 改用现存的等价 0-100 直接分：机构游资共振 / 游资席位持续性。
    dt = first_not_none(raw.get("institution_hotmoney_combo"), 45.0)
    net_pct = safe_float(raw.get("lhb_net_buy_pct")) or 0.0
    concurrent = safe_float(raw.get("hot_money_concurrent")) or 0.0
    network = first_not_none(raw.get("hot_money_persistence"), 40.0)
    overheat = first_not_none(raw.get("overheat_penalty"), 45.0)
    tradability = first_not_none(raw.get("tradability"), 80.0)
    reason = first_not_none(raw.get("lhb_reason_strength"), 40.0)
    proxy_pct = (
        (dt - 50.0) * 0.055
        + min(net_pct, 30.0) * 0.035
        + concurrent * 0.22
        + (network - 50.0) * 0.025
        + (reason - 50.0) * 0.018
        + (overheat - 50.0) * 0.018
        + (tradability - 80.0) * 0.015
    )
    return proxy_pct / 100.0


def short_proxy_backtest(picks: List[Dict[str, Any]]) -> Dict[str, Any]:
    returns = [short_proxy_return(pick) for pick in picks]
    return summarize_returns(returns, [])


def objective_from_summary(summary: Dict[str, Any], fallback_quality: float = 0.0) -> float:
    if not summary.get("samples"):
        return fallback_quality
    excess = first_not_none(summary.get("avg_excess"), summary.get("avg_return"))
    if excess is None:
        excess = 0.0
    hit = (safe_float(summary.get("hit_rate")) or 0.0) / 100.0
    dd = safe_float(summary.get("max_drawdown")) or 0.0
    sharpe = safe_float(summary.get("sharpe_like")) or 0.0
    return excess + hit * 2.0 + sharpe * 0.3 - dd * 0.08


def mutate_weights(
    rng: random.Random,
    base_weights: Dict[str, float],
    factor_keys: List[str],
    iteration: int,
    floors: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    weights = {}
    group_scale = rng.choice([0.55, 0.75, 1.0, 1.25, 1.55])
    for key in factor_keys:
        base = safe_float(base_weights.get(key)) or 0.0
        if iteration == 0:
            value = base
        else:
            jitter = rng.lognormvariate(0.0, 0.42)
            if rng.random() < 0.10:
                jitter *= rng.choice([0.0, 0.25, 2.0])
            value = base * jitter * group_scale
        # ③ 防御因子保底：不让搜索把风险预算清零（low_volatility/debt_safety/asset_growth）。
        if floors and key in floors:
            value = max(value, floors[key])
        weights[key] = round(clamp(value, 0.0, 3.0), 3)
    return weights


def constrain_long_search_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    weights = cfg.setdefault("weights", {})
    for key in LONG_FIXED_ZERO_WEIGHTS:
        weights[key] = 0.0
    cfg["min_market_cap_yi"] = LONG_FIXED_MIN_MARKET_CAP_YI
    cfg["min_score"] = LONG_FIXED_MIN_SCORE
    # 当前沪深300成分没有历史快照，优化器不把它作为硬过滤；实盘页面仍可手动开启。
    cfg["require_csi300"] = False
    return cfg


def constrain_smallcap_search_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    weights = cfg.setdefault("weights", {})
    for key in SMALLCAP_FIXED_ZERO_WEIGHTS:
        weights[key] = 0.0
    # 防止外部旧配置把小盘明确排除的指数/市值因子重新写入。
    allowed = {factor.key for factor in SMALLCAP_FACTORS}
    cfg["weights"] = {key: value for key, value in weights.items() if key in allowed}
    cfg["min_score"] = LONG_FIXED_MIN_SCORE
    cfg["top_n"] = DEFAULT_CONFIG["smallcap"]["top_n"]
    # 小盘池本身已经限定为游资小盘成员；历史高点回撤会把成员池再次压缩并引入
    # regime 选择偏差，因此优化器固定关闭，不作为随机/Optuna 搜索维度。
    cfg["require_high_drawdown"] = False
    cfg["min_high_drawdown_pct"] = DEFAULT_CONFIG["smallcap"]["min_high_drawdown_pct"]
    return cfg


def set_long_high_drawdown_filter(cfg: Dict[str, Any], threshold_pct: Any) -> None:
    """设置历史高点回撤过滤阈值；开关由 require_high_drawdown 独立搜索。"""
    threshold = safe_float(threshold_pct)
    if threshold is None:
        threshold = 40.0
    threshold = int(round(clamp(threshold, 40.0, 70.0)))
    cfg["min_high_drawdown_pct"] = threshold


def random_long_config(rng: random.Random, iteration: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG["long"])
    cfg["weights"] = mutate_weights(
        rng, cfg["weights"], [f.key for f in LONG_SEARCHABLE_FACTORS], iteration, floors=LONG_WEIGHT_FLOORS
    )
    # 高点回撤过滤开关参与搜索；阈值仅在开启时作为历史高点至今跌幅下限。
    set_long_high_drawdown_filter(cfg, cfg.get("min_high_drawdown_pct"))
    if iteration > 0:
        # top_n 与 min_score 不参与搜索，分别固定为 DEFAULT_CONFIG 与 50。
        cfg["require_high_drawdown"] = rng.choice([False, True])
        set_long_high_drawdown_filter(cfg, rng.choice(LONG_HIGH_DD_PCT_CHOICES))
    return constrain_long_search_config(cfg)


def random_smallcap_config(rng: random.Random, iteration: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG["smallcap"])
    floors = {key: value for key, value in LONG_WEIGHT_FLOORS.items() if key in cfg["weights"]}
    cfg["weights"] = mutate_weights(
        rng,
        cfg["weights"],
        [f.key for f in SMALLCAP_SEARCHABLE_FACTORS],
        iteration,
        floors=floors,
    )
    return constrain_smallcap_search_config(cfg)


def random_short_config(rng: random.Random, iteration: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG["short"])
    cfg["weights"] = mutate_weights(rng, cfg["weights"], [f.key for f in SHORT_FACTORS], iteration)
    if iteration > 0:
        # top_n 不参与搜索，固定用 DEFAULT_CONFIG 的 10
        cfg["min_score"] = rng.choice([35, 40, 45, 50, 55, 60, 65, 70])
        cfg["max_consecutive_limit_up"] = rng.choice([1, 2, 3, 4])
    return cfg


def broad_long_candidate_config() -> Dict[str, Any]:
    broad_cfg = copy.deepcopy(DEFAULT_CONFIG["long"])
    broad_cfg.update({
        "exclude_st": False,
        "min_market_cap_yi": 0,
        "require_csi300": False,
        # broad 候选池不提前按高点回撤筛选；raw factor 由候选构建阶段保留，供 trial 硬过滤使用。
        "require_high_drawdown": False,
        "min_high_drawdown_pct": 0,
        "min_score": 0,
        "top_n": 80,
    })
    return broad_cfg


def broad_smallcap_candidate_config() -> Dict[str, Any]:
    broad_cfg = copy.deepcopy(DEFAULT_CONFIG["smallcap"])
    broad_cfg.update({
        "exclude_st": False,
        "require_high_drawdown": False,
        "min_high_drawdown_pct": 0,
        "min_score": 0,
        "top_n": 80,
    })
    return broad_cfg


# ─── Optuna(TPE 贝叶斯) 搜索空间与运行器 ────────────────────────
# 只替换"配置生成"：探索从默认配置基线起步、按代理目标的后验智能采样，
# 比无记忆随机搜索样本效率高得多。目标函数/折回测/输出结构完全不变。
# 未安装 optuna 时自动回退到原随机搜索（random_long_config / random_short_config）。

SHORT_MIN_SCORE_CHOICES = [35, 40, 45, 50, 55, 60, 65, 70]
SHORT_HOLD_DAYS_CHOICES = [1, 2, 3, 4, 5]


def _nearest_choice(value: Any, choices: List[Any]) -> Any:
    """把默认值映射到 choices 内最接近的合法项（enqueue_trial 要求枚举值必须在候选中）。"""
    num = safe_float(value)
    if num is not None and all(isinstance(c, (int, float)) for c in choices):
        return min(choices, key=lambda c: abs(c - num))
    return value if value in choices else choices[0]


def _short_config_with_hold_days(config: Dict[str, Any], hold_days: Any) -> Dict[str, Any]:
    """Keep displayed, evaluated and persisted short horizons identical."""
    selected = int(_nearest_choice(hold_days, SHORT_HOLD_DAYS_CHOICES))
    cfg = copy.deepcopy(config)
    cfg["hold_days"] = selected
    cfg["hold_days_min"] = selected
    cfg["hold_days_max"] = selected
    return cfg


def _long_weight_bounds(key: str) -> Tuple[float, float]:
    low = LONG_WEIGHT_FLOORS.get(key, 0.0)
    high = 3.0
    return low, max(high, low)


def _suggest_long_config(trial) -> Tuple[Dict[str, Any], int]:
    cfg = copy.deepcopy(DEFAULT_CONFIG["long"])
    cfg["weights"] = {
        f.key: round(trial.suggest_float(f"w_{f.key}", *_long_weight_bounds(f.key)), 3)
        for f in LONG_SEARCHABLE_FACTORS
    }
    cfg["require_high_drawdown"] = trial.suggest_categorical("require_high_drawdown", [False, True])
    set_long_high_drawdown_filter(
        cfg,
        trial.suggest_int("min_high_drawdown_pct", 40, 70),
    )
    hold_td = trial.suggest_categorical("hold_td", LONG_HOLD_CHOICES)
    return constrain_long_search_config(cfg), hold_td


def _suggest_smallcap_config(trial) -> Tuple[Dict[str, Any], int]:
    cfg = copy.deepcopy(DEFAULT_CONFIG["smallcap"])
    cfg["weights"] = {
        f.key: round(trial.suggest_float(f"w_{f.key}", *_long_weight_bounds(f.key)), 3)
        for f in SMALLCAP_SEARCHABLE_FACTORS
    }
    hold_td = trial.suggest_categorical("hold_td", SMALLCAP_HOLD_CHOICES)
    return constrain_smallcap_search_config(cfg), hold_td


def _default_long_params() -> Dict[str, Any]:
    base = DEFAULT_CONFIG["long"]
    base_weights = base.get("weights") or {}
    params: Dict[str, Any] = {}
    for f in LONG_SEARCHABLE_FACTORS:
        low, high = _long_weight_bounds(f.key)
        params[f"w_{f.key}"] = round(min(max(safe_float(base_weights.get(f.key)) or 0.0, low), high), 3)
    params["require_high_drawdown"] = bool(base.get("require_high_drawdown", False))
    params["min_high_drawdown_pct"] = _nearest_choice(base.get("min_high_drawdown_pct"), LONG_HIGH_DD_PCT_CHOICES)
    params["hold_td"] = LONG_HOLD_CHOICES[0]
    return params


def _default_smallcap_params() -> Dict[str, Any]:
    base = DEFAULT_CONFIG["smallcap"]
    base_weights = base.get("weights") or {}
    params: Dict[str, Any] = {}
    for f in SMALLCAP_SEARCHABLE_FACTORS:
        low, high = _long_weight_bounds(f.key)
        params[f"w_{f.key}"] = round(
            min(max(safe_float(base_weights.get(f.key)) or 0.0, low), high), 3
        )
    params["hold_td"] = SMALLCAP_HOLD_CHOICES[0]
    return params


def _suggest_short_config(trial) -> Tuple[Dict[str, Any], int]:
    cfg = copy.deepcopy(DEFAULT_CONFIG["short"])
    cfg["weights"] = {
        f.key: round(trial.suggest_float(f"w_{f.key}", 0.0, 3.0), 3) for f in SHORT_FACTORS
    }
    cfg["min_score"] = trial.suggest_categorical("min_score", SHORT_MIN_SCORE_CHOICES)
    cfg["max_consecutive_limit_up"] = trial.suggest_categorical("max_consecutive_limit_up", [1, 2, 3, 4])
    hold_days = trial.suggest_categorical("hold_days", SHORT_HOLD_DAYS_CHOICES)
    return _short_config_with_hold_days(cfg, hold_days), hold_days


def _default_short_params() -> Dict[str, Any]:
    base = DEFAULT_CONFIG["short"]
    base_weights = base.get("weights") or {}
    params: Dict[str, Any] = {
        f"w_{f.key}": round(min(max(safe_float(base_weights.get(f.key)) or 0.0, 0.0), 3.0), 3)
        for f in SHORT_FACTORS
    }
    params["min_score"] = _nearest_choice(base.get("min_score"), SHORT_MIN_SCORE_CHOICES)
    params["max_consecutive_limit_up"] = _nearest_choice(base.get("max_consecutive_limit_up"), [1, 2, 3, 4])
    params["hold_days"] = _nearest_choice(
        base.get("hold_days", base.get("hold_days_min", 1)), SHORT_HOLD_DAYS_CHOICES
    )
    return params


def _short_backtest_for_horizon(
        picks: List[Dict[str, Any]],
        series: Dict[str, List[Dict[str, Any]]],
        suggested_hold_days: Any,
) -> Tuple[int, Dict[str, Any]]:
    """Evaluate exactly the horizon attributed to this search trial."""
    selected = int(_nearest_choice(suggested_hold_days, SHORT_HOLD_DAYS_CHOICES))
    return selected, short_actual_backtest(picks, series, hold_days=selected)


def _finalize_short_best(row: Dict[str, Any]) -> Dict[str, Any]:
    """Do not persist an arbitrary horizon selected only by a timeless proxy."""
    best = copy.deepcopy(row)
    if best.get("hold_days_validated"):
        return best
    baseline = int(_default_short_params()["hold_days"])
    best["hold_days"] = baseline
    best["config"] = _short_config_with_hold_days(best.get("config") or {}, baseline)
    best["hold_days_fallback_reason"] = "no_forward_price_samples"
    return best


def optuna_startup_trials(iterations: int) -> int:
    """高维权重空间先保留足够随机探索，避免 TPE 十来轮后过早追逐噪声冠军。"""
    if iterations <= 0:
        return 0
    warmup = max(OPTUNA_MIN_STARTUP_TRIALS, math.ceil(iterations * OPTUNA_STARTUP_TRIAL_FRACTION))
    return min(iterations, warmup)


def make_progress_bar(total: int, desc: str, position: int = 0):
    if tqdm is None:
        return nullcontext()
    return tqdm(total=total, desc=desc, unit="trial", dynamic_ncols=True, disable=None, position=position)


def advance_progress(progress) -> None:
    if progress is not None:
        progress.update(1)


# 单个 Optuna study 内部 trial 并行度默认 1。外层长线/小盘/短线用独立子进程并行；
# study 内部实测多线程是负收益，故只向量化(单核提速)、默认不再拆 trial：
#   ① 预热以财报计算(纯 python)为主=GIL-bound，ThreadPool 并行因 GIL 争用反而更慢(53.7s vs 串行23.2s)，故预热串行；
#   ② optuna n_jobs>1 多线程并发退化 TPE 序贯采样、降搜索质量(实测 8 trials best -7.69→-13.6)。
# 要试单 study 内部 trial 并行可 env OPTUNA_TRIAL_JOBS=N，但默认保持 1 以保护 TPE 搜索质量。
OPTUNA_TRIAL_JOBS = max(1, int(os.environ.get("OPTUNA_TRIAL_JOBS", "1")))


def _run_config_search(
    iterations,
    rng,
    *,
    suggest,
    default_params,
    random_config,
    record,
    progress_label: str,
    n_jobs: int = 1,
    progress_position: int = 0,
) -> str:
    """跑 iterations 次配置搜索。有 optuna → TPE 贝叶斯（第1次试默认基线 + 随机预热后后验采样）；
    否则回退随机搜索。suggest(trial)->(cfg,extra)；random_config(i)->(cfg,extra)；
    record(cfg,extra)->objective(交给搜索器最大化)。返回一行描述搜索方法的 note。"""
    if optuna is not None:
        startup_trials = optuna_startup_trials(iterations)
        sampler = optuna.samplers.TPESampler(
            seed=rng.randrange(2 ** 31),
            multivariate=True,
            group=True,
            n_startup_trials=startup_trials,
            n_ei_candidates=OPTUNA_EI_CANDIDATES,
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        if default_params:
            study.enqueue_trial(default_params)  # 第1次评估默认配置作基线
        with make_progress_bar(iterations, progress_label, progress_position) as progress:
            def objective(trial) -> float:
                try:
                    return record(*suggest(trial))
                finally:
                    advance_progress(progress)

            study.optimize(objective, n_trials=iterations, n_jobs=n_jobs)
        return (f"参数搜索：Optuna TPE 贝叶斯优化（{iterations} trials，第1次为默认配置基线，"
                f"前 {startup_trials} 次保留随机/默认预热，之后按代理目标的后验采样选下一组参数；"
                "代理目标本身存在过拟合/前视风险，"
                "更强的搜索会更彻底地拟合该目标，best 仍需样本外复核）。")
    with make_progress_bar(iterations, progress_label, progress_position) as progress:
        for i in range(iterations):
            try:
                record(*random_config(i))
            finally:
                advance_progress(progress)
    return f"参数搜索：随机搜索回退（{iterations} 次，未检测到 optuna）。"


def _optimize_long_style(
    strategy: str,
    iterations: int,
    rng: random.Random,
    progress_position: int,
    *,
    label: str,
    factors: List[Any],
    broad_cfg: Dict[str, Any],
    universe_codes_fn,
    build_candidates_fn,
    score_prepared_fn,
    suggest_fn,
    default_params: Dict[str, Any],
    random_config_fn,
    strategy_notes: List[str],
) -> Dict[str, Any]:
    """长线与小盘共用的 PIT 多折、风险目标与 Optuna 搜索流程。"""
    series = full_series_map()
    price_health = ensure_long_price_history(series)
    cal = fold_calendar(series)
    cal_pos = {date: idx for idx, date in enumerate(cal)}  # as_of -> 交易日序号，供去相关/隔离带用
    benchmark_series, benchmark_notes = load_benchmark_series()
    deep_codes, universe_notes = universe_codes_fn(series, broad_cfg)

    # 折偏移 → PIT 时点日期（全市场日历倒数第 offset 个交易日）。[0] 最近、[-1] 最旧。
    hold_choices = SMALLCAP_HOLD_CHOICES if strategy == "smallcap" else LONG_HOLD_CHOICES
    anchor_offsets_fn = smallcap_anchor_offsets if strategy == "smallcap" else long_anchor_offsets
    fold_step_td = SMALLCAP_FOLD_STEP_TD if strategy == "smallcap" else LONG_FOLD_STEP_TD
    max_lookback_td = SMALLCAP_MAX_LOOKBACK_TD if strategy == "smallcap" else LONG_MAX_LOOKBACK_TD
    fold_dates_by_hold = {
        h: [cal[-off] for off in anchor_offsets_fn(h) if off <= len(cal)]
        for h in hold_choices
    }

    pit_cache: Dict[str, List[Dict[str, Any]]] = {}       # as_of -> PIT broad 候选预评分
    fold_stats_cache: Dict[Tuple[str, int, Tuple[str, ...]], Optional[Dict[str, float]]] = {}
    notes: List[str] = list(universe_notes)
    candidate_notes_loaded = False
    _notes_lock = threading.Lock()
    _fold_stats_lock = threading.Lock()

    def get_pit(as_of: str) -> List[Dict[str, Any]]:
        nonlocal candidate_notes_loaded
        if as_of not in pit_cache:
            cands, nts = build_candidates_fn(broad_cfg, as_of=as_of, universe=deep_codes)
            pit = prepare_candidate_factor_scores(cands, factors)
            _prepared_matrix(pit, factors)     # 预构因子分矩阵(预热期建好，trial 期只读)
            pit_cache[as_of] = pit             # 末尾赋值；并发同 as_of 最多重复算(幂等)
            with _notes_lock:
                if not candidate_notes_loaded:
                    notes.extend(nts)
                    candidate_notes_loaded = True
        return pit_cache[as_of]

    def get_fold_stats(
        picks: List[Dict[str, Any]],
        as_of: str,
        hold_td: int,
    ) -> Optional[Dict[str, float]]:
        key = (as_of, hold_td, tuple(p["code"] for p in picks))
        with _fold_stats_lock:
            if key in fold_stats_cache:
                return fold_stats_cache[key]
        val = portfolio_fold_stats(picks, series, benchmark_series, as_of, hold_td)  # 锁外算(慢)
        with _fold_stats_lock:
            fold_stats_cache[key] = val
        return val

    # 串行预热所有折的 PIT(填 pit_cache + 因子分矩阵)，使 trial 期缓存只读、n_jobs 多线程安全。
    # 不用 ThreadPool：预热是 GIL-bound 的财报计算为主，多线程并行因 GIL 争用反而更慢(见上方常量说明)。
    for _as_of in sorted({d for dates in fold_dates_by_hold.values() for d in dates}):
        get_pit(_as_of)

    def evaluate(cfg: Dict[str, Any], hold_td: int) -> Tuple[float, Dict[str, Any]]:
        fold_dates = fold_dates_by_hold[hold_td]
        pairs: List[Dict[str, Any]] = []
        for as_of in fold_dates:
            picks = score_prepared_fn(
                get_pit(as_of), cfg, entry_series=series, as_of=as_of
            )  # 该折用 PIT 因子重新选股，并要求入场日真实可交易
            fold_stats = get_fold_stats(picks, as_of, hold_td)
            if fold_stats is None:
                continue
            pairs.append({"as_of": as_of, "cal_idx": cal_pos.get(as_of, 0), **fold_stats})
        # 风险/尾部统计的去相关间隔：相邻保留折至少隔半个持有期(重叠≤50%)
        decorr_gap_td = max(1, hold_td // 2)
        pairs = apply_recency_weights(pairs)
        train_pairs, val_pairs = random_fold_split(pairs)
        adjusted_obj, objective_detail = long_validation_adjusted_objective(
            train_pairs, val_pairs, pairs, hold_td, decorr_gap_td
        )
        # 折数太少(严过滤把多数折剔空)不可信，直接出局，避免1~3折彩票配置夺魁
        if len(pairs) < LONG_MIN_FOLDS or adjusted_obj is None:
            objective = -999.0
        else:
            objective = adjusted_obj
        # 报告用最近一折的选股（最接近实盘当下）
        repr_picks = (
            score_prepared_fn(
                get_pit(fold_dates[0]), cfg, entry_series=series, as_of=fold_dates[0]
            )
            if fold_dates else []
        )
        row = {
            "objective": round(objective, 5),
            "hold_td": hold_td,
            "selected_count": len(repr_picks),
            "summary": long_fold_summary(pairs, hold_td),
            "train_summary": long_fold_summary(train_pairs, hold_td),
            "val_summary": long_fold_summary(val_pairs, hold_td),
            "objective_detail": objective_detail,
            "top_codes": [p["code"] for p in repr_picks[:5]],
            "config": cfg,
        }
        return objective, row

    trace: List[Dict[str, Any]] = []
    trace_lock = threading.Lock()

    def _record(cfg: Dict[str, Any], hold_td: int) -> float:
        objective, row = evaluate(cfg, hold_td)
        with trace_lock:
            row["iteration"] = len(trace) + 1
            trace.append(row)
        return objective

    search_note = _run_config_search(
        iterations, rng,
        suggest=suggest_fn,
        default_params=default_params,
        random_config=lambda i: (
            random_config_fn(rng, i),
            rng.choice(hold_choices) if i else hold_choices[0],
        ),
        record=_record,
        progress_label=f"{label}参数搜索",
        n_jobs=OPTUNA_TRIAL_JOBS,
        progress_position=progress_position,
    )
    best = select_best_long_trace(trace)
    return {
        "strategy": strategy,
        "iterations": iterations,
        "candidate_count": len(deep_codes),
        "price_history_health": price_health,
        "notes": [search_note] + benchmark_notes + notes + [
            f"PIT 滚动前推回测：每折以全市场日历倒数第N个交易日为时点，财报(按法定披露截止日)/价格/估值/分红全部按当时可见重算因子后再选股，且入选前要求 as_of 当天有真实交易行；固定持有{hold_choices[0]}交易日，组合等权收益-成本-沪深300与中证500按日等权混合(各50%)累计净值基准。",
            f"训练/验证按折随机打乱后切分(~60/40)；当前持有期={hold_choices[0]}、起点间隔={fold_step_td}个交易日，完整折首尾衔接；最多回看约{max_lookback_td}个交易日。",
            "折样本采用等权口径：最旧折到最新折均为 1.0x，所有历史折在均值、命中率、CVaR、Sortino 和回撤惩罚中权重一致。",
            f"完整折每{fold_step_td}交易日取一个起点、持有{hold_choices[0]}交易日；去相关子样本(indep_folds)逻辑保留。",
            f"{label}选优目标：训练目标分×0.55+验证×0.45；目标显式纳入 CVaR、持有期组合回撤和 Sortino 下行信息比，并对尾部风险、训练/验证差异和有效折数不足实施惩罚。",
            f"风险预算保底：低波动/低负债/保守投资因子权重下限为0.30/0.20/0.20；min_score 固定为 {LONG_FIXED_MIN_SCORE}；" + ("小盘高点回撤过滤固定关闭且不参与搜索。" if strategy == "smallcap" else "高点回撤过滤开关与40~70%阈值参与搜索。"),
            "max_drawdown_pct 为各折持有期内按基准交易日历逐日估值的组合路径最大回撤(取最深折)；avg_fold_max_drawdown_pct 为各折回撤均值；worst_fold_excess_pct 为全部折最差单折超额，tail_cvar_excess_pct 为尾部最差20%折的平均超额。",
        ] + strategy_notes,
        "best": best,
        "top_iterations": sorted(trace, key=lambda x: x["objective"], reverse=True)[:10],
        "convergence": convergence_summary(trace, iterations),
    }


def optimize_long(iterations: int, rng: random.Random, progress_position: int = 0) -> Dict[str, Any]:
    return _optimize_long_style(
        "long", iterations, rng, progress_position,
        label="长线",
        factors=LONG_FACTORS,
        broad_cfg=broad_long_candidate_config(),
        universe_codes_fn=long_optimizer_universe_codes,
        build_candidates_fn=build_long_candidates,
        score_prepared_fn=score_prepared_long_candidates,
        suggest_fn=_suggest_long_config,
        default_params=_default_long_params(),
        random_config_fn=random_long_config,
        strategy_notes=[
            "长线候选池限定为 SW3 细分龙头池；市值/指数成分相关信号、行业规模地位及近期龙虎榜在优化中固定为0。",
            "残留前视：沪深300成分与质押使用当前快照。",
        ],
    )


def optimize_smallcap(iterations: int, rng: random.Random, progress_position: int = 1) -> Dict[str, Any]:
    return _optimize_long_style(
        "smallcap", iterations, rng, progress_position,
        label="小盘",
        factors=SMALLCAP_FACTORS,
        broad_cfg=broad_smallcap_candidate_config(),
        universe_codes_fn=smallcap_optimizer_universe_codes,
        build_candidates_fn=build_smallcap_candidates,
        score_prepared_fn=score_prepared_smallcap_candidates,
        suggest_fn=_suggest_smallcap_config,
        default_params=_default_smallcap_params(),
        random_config_fn=random_smallcap_config,
        strategy_notes=[
            "小盘股票池严格复用短线/游资雷达当前 is_hot_money=1 成员；当前成员回看历史存在幸存者偏差。",
            "小盘不展示、不评分、不搜索 csi300_current、csi300_persistence、market_cap、size_reversal，也不使用沪深300或总市值硬约束。",
            "行业规模地位沿用长线 optimizer 口径，权重固定为0。",
        ],
    )


def optimize_short(iterations: int, rng: random.Random, progress_position: int = 2) -> Dict[str, Any]:
    broad_cfg = copy.deepcopy(DEFAULT_CONFIG["short"])
    broad_cfg.update({
        "min_lhb_count": 0,
        "min_hot_money_concurrent": 0,
        "max_consecutive_limit_up": 99,
        "min_score": 0,
        "top_n": 80,
    })
    broad, notes = build_short_candidates(broad_cfg)
    prepared = prepare_candidate_factor_scores(broad, SHORT_FACTORS)
    _prepared_matrix(prepared, SHORT_FACTORS)
    _prepared_short_filter_arrays(prepared)
    series = recent_series_map()
    trace: List[Dict[str, Any]] = []
    trace_lock = threading.Lock()

    def evaluate_short(cfg: Dict[str, Any], hold_days: int) -> Tuple[float, Dict[str, Any]]:
        picks = score_prepared_short_candidates(prepared, cfg)
        suggested_hold_days = int(_nearest_choice(hold_days, SHORT_HOLD_DAYS_CHOICES))
        # Never substitute another horizon's return: Optuna must attribute the
        # objective to the exact categorical value it suggested.
        effective_hold_days, actual = _short_backtest_for_horizon(
            picks, series, suggested_hold_days
        )
        cfg = _short_config_with_hold_days(cfg, effective_hold_days)
        proxy = short_proxy_backtest(picks)
        actual_objective = objective_from_summary(actual, fallback_quality=-999.0)
        proxy_objective = objective_from_summary(proxy, fallback_quality=0.0)
        if actual.get("samples"):
            objective = actual_objective * 0.7 + proxy_objective * 0.3
            objective_mode = "actual_plus_proxy"
        else:
            objective = proxy_objective
            objective_mode = "proxy_only_no_forward_prices"
        row = {
            "objective": round(objective, 5),
            "objective_mode": objective_mode,
            "hold_days": effective_hold_days,
            "suggested_hold_days": suggested_hold_days,
            "hold_days_validated": bool(actual.get("samples")),
            "selected_count": len(picks),
            "actual_backtest": actual,
            "proxy_backtest": proxy,
            "top_codes": [p["code"] for p in picks[:5]],
            "config": cfg,
        }
        return objective, row

    def _record(cfg: Dict[str, Any], hold_days: int) -> float:
        objective, row = evaluate_short(cfg, hold_days)
        with trace_lock:
            row["iteration"] = len(trace) + 1
            trace.append(row)
        return objective

    def random_short_trial(iteration: int) -> Tuple[Dict[str, Any], int]:
        hold_days = rng.choice(SHORT_HOLD_DAYS_CHOICES) if iteration else int(
            _default_short_params()["hold_days"]
        )
        return _short_config_with_hold_days(
            random_short_config(rng, iteration), hold_days
        ), hold_days

    search_note = _run_config_search(
        iterations, rng,
        suggest=_suggest_short_config,
        default_params=_default_short_params(),
        random_config=random_short_trial,
        record=_record,
        progress_label="短线参数搜索",
        n_jobs=OPTUNA_TRIAL_JOBS,
        progress_position=progress_position,
    )
    if not trace:
        raise RuntimeError("短线优化没有产生任何 trial。")
    best = _finalize_short_best(max(trace, key=lambda x: x["objective"]))
    return {
        "strategy": "short",
        "iterations": iterations,
        "candidate_count": len(broad),
        "notes": [search_note] + notes + [
            "短线候选池因子分在 trial 前预转为 NumPy 矩阵，trial 阶段向量化完成硬过滤、打分和 topN 选择。",
            "最少上榜与最少共振默认固定为0，不参与优化；页面仍可手动调高做二次筛选。",
            "短线真实前推收益回测仅在事件日之后存在本地价格数据时使用。",
            "持有期只由真实前推收益选择；无对应价格样本时回退默认持有期，代理分不冒充持有期依据。",
            "当前本地快照中，龙虎榜事件多数晚于缓存价格行，因此大多需要使用代理评分。",
        ],
        "best": best,
        "top_iterations": sorted(trace, key=lambda x: x["objective"], reverse=True)[:10],
        "convergence": convergence_summary(trace, iterations),
    }


def average_paths(paths: List[Dict[str, Any]], key: str) -> List[float]:
    if not paths:
        return []
    length = min(len(path.get(key, [])) for path in paths)
    averaged = []
    for idx in range(length):
        values = [path[key][idx] for path in paths if idx < len(path.get(key, []))]
        averaged.append(mean(values))
    return averaged


def stitched_fold_path(paths: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """把按时间排序的折内日度净值拼成一条完整历史曲线。"""
    dates: List[str] = []
    portfolio_path: List[float] = []
    benchmark_path: List[float] = []
    portfolio_nav = 1.0
    benchmark_nav = 1.0
    source_folds = 0
    last_date: Optional[str] = None

    for path in sorted(paths, key=lambda row: row["as_of"]):
        raw_dates = [str(date) for date in path.get("dates", [])]
        raw_portfolio = path.get("portfolio_path", [])
        raw_benchmark = path.get("benchmark_path", [])
        length = min(len(raw_dates), len(raw_portfolio), len(raw_benchmark))
        if length < 2:
            continue

        if not dates:
            first_date = raw_dates[0]
            if len(first_date) != 10:
                continue
            dates.append(first_date)
            portfolio_path.append(portfolio_nav)
            benchmark_path.append(benchmark_nav)
            last_date = first_date

        added = False
        for idx in range(1, length):
            date = raw_dates[idx]
            if len(date) != 10 or (last_date is not None and date <= last_date):
                continue
            prev_portfolio = safe_float(raw_portfolio[idx - 1])
            curr_portfolio = safe_float(raw_portfolio[idx])
            prev_benchmark = safe_float(raw_benchmark[idx - 1])
            curr_benchmark = safe_float(raw_benchmark[idx])
            if (
                prev_portfolio is None or curr_portfolio is None
                or prev_benchmark is None or curr_benchmark is None
                or prev_portfolio <= 0 or prev_benchmark <= 0
            ):
                continue
            portfolio_nav *= curr_portfolio / prev_portfolio
            benchmark_nav *= curr_benchmark / prev_benchmark
            dates.append(date)
            portfolio_path.append(portfolio_nav)
            benchmark_path.append(benchmark_nav)
            last_date = date
            added = True
        if added:
            source_folds += 1

    if len(dates) < 2:
        return None
    return {
        "dates": dates,
        "portfolio_path": portfolio_path,
        "benchmark_path": benchmark_path,
        "source_folds": source_folds,
        "portfolio_return": portfolio_path[-1] - 1.0,
        "benchmark_return": benchmark_path[-1] - 1.0,
        "excess_return": portfolio_path[-1] - benchmark_path[-1],
        "portfolio_max_drawdown": path_max_drawdown(portfolio_path) or 0.0,
        "benchmark_max_drawdown": path_max_drawdown(benchmark_path) or 0.0,
    }


def svg_points(values: List[float], x_scale, y_scale) -> str:
    return " ".join(
        f"{x_scale(idx):.2f},{y_scale(value):.2f}"
        for idx, value in enumerate(values)
        if value is not None
    )


def write_long_fold_paths_svg(
    paths: List[Dict[str, Any]],
    output_file: Path,
    hold_td: int,
    holding_top_n: Optional[int] = None,
    title: str = "长线最优参数各折走势小图矩阵",
    fold_interval_td: int = LONG_FOLD_STEP_TD,
) -> Dict[str, Any]:
    if not paths:
        raise ValueError("没有可用于画图的有效长线回测折。")

    paths = sorted(paths, key=lambda row: row["as_of"])
    complete_paths = [path for path in paths if not path.get("partial")]
    partial_paths = [path for path in paths if path.get("partial")]
    avg_source = complete_paths or paths
    avg_portfolio = average_paths(avg_source, "portfolio_path")
    avg_benchmark = average_paths(avg_source, "benchmark_path")
    full_path = stitched_fold_path(paths)
    try:
        replacement_top_n = int(holding_top_n if holding_top_n is not None else 10)
    except (TypeError, ValueError):
        replacement_top_n = 10
    replacement_top_n = max(1, replacement_top_n)
    replacement_prefix = f"Top{replacement_top_n}"
    visible_holding_count = min(replacement_top_n, 20)
    stock_names_per_line = 5
    stock_line_gap = 15
    stock_line_count = max(2, math.ceil(visible_holding_count / stock_names_per_line))
    stock_extra_h = max(0, stock_line_count - 2) * stock_line_gap

    cols = 4
    rows = math.ceil(len(paths) / cols)
    width = 1600
    header_h = 122
    footer_h = 42
    full_chart_gap = 28 if full_path else 0
    full_chart_h = 330 if full_path else 0
    left = 54
    right = 42
    gap_x = 24
    gap_y = 22
    cell_w = (width - left - right - gap_x * (cols - 1)) / cols
    cell_h = 230 + stock_extra_h
    grid_h = rows * cell_h + (rows - 1) * gap_y
    height = int(header_h + grid_h + full_chart_gap + full_chart_h + footer_h)

    def text(
        x: float,
        y: float,
        body: str,
        size: int = 13,
        weight: str = "400",
        anchor: str = "start",
        color: str = "#0f172a",
    ) -> str:
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" fill="{color}">'
            f'{html_escape(body)}</text>'
        )

    def path_scale(path: Dict[str, Any], plot_x: float, plot_y: float, plot_w: float, plot_h: float):
        values = [1.0] + path["portfolio_path"] + path["benchmark_path"]
        y_min = min(values)
        y_max = max(values)
        if y_max - y_min < 0.08:
            y_min -= 0.04
            y_max += 0.04
        else:
            pad = (y_max - y_min) * 0.12
            y_min = max(0.0, y_min - pad)
            y_max += pad
        max_idx = max(
            1,
            int(path.get("target_hold_td") or min(len(path["portfolio_path"]), len(path["benchmark_path"])) - 1),
        )

        def x_scale(idx: int) -> float:
            return plot_x + plot_w * idx / max_idx

        def y_scale(value: float) -> float:
            return plot_y + plot_h * (y_max - value) / (y_max - y_min)

        return x_scale, y_scale, y_min, y_max

    def holding_names(path: Dict[str, Any], limit: int) -> List[str]:
        holdings = path.get("holdings", []) or []
        return [
            str(item.get("name") or item.get("code") or "").strip()
            for item in holdings[:limit]
            if isinstance(item, dict) and (item.get("name") or item.get("code"))
        ]

    def holding_name_lines(path: Dict[str, Any]) -> List[str]:
        names = holding_names(path, visible_holding_count)
        if not names:
            return []
        return [
            " · ".join(names[idx:idx + stock_names_per_line])
            for idx in range(0, len(names), stock_names_per_line)
        ]

    def holding_topn_keys(path: Dict[str, Any]) -> List[str]:
        keys = []
        seen = set()
        for item in (path.get("holdings", []) or [])[:replacement_top_n]:
            if isinstance(item, dict):
                key = str(item.get("code") or item.get("name") or "").strip()
            else:
                key = str(item or "").strip()
            if key and key not in seen:
                keys.append(key)
                seen.add(key)
        return keys

    topn_replacement_counts = []
    prev_topn_keys: Optional[List[str]] = None
    for path in paths:
        topn_keys = holding_topn_keys(path)
        replaced_count = None
        if prev_topn_keys is not None and topn_keys:
            replaced_count = len(set(topn_keys) - set(prev_topn_keys))
        path["topn_replaced_count"] = replaced_count
        topn_replacement_counts.append({
            "as_of": path.get("as_of"),
            "replaced_count": replaced_count,
        })
        if topn_keys:
            prev_topn_keys = topn_keys

    def topn_replacement_label(path: Dict[str, Any]) -> str:
        replaced_count = path.get("topn_replaced_count")
        if replaced_count is None:
            return f"{replacement_prefix}替换 --"
        return f"{replacement_prefix}替换 {int(replaced_count)}只"

    elements: List[str] = []
    elements.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>')
    elements.append(text(36, 42, title, 25, "700"))
    fold_count_text = f"{len(complete_paths)} 个完整回测窗口"
    if partial_paths:
        fold_count_text += f" + {len(partial_paths)} 个未满持有期展示窗口"
    elements.append(
        text(
            36,
            72,
            (
                f"{fold_count_text} · 持有 {hold_td} 个交易日 · "
                f"每 {fold_interval_td} 个交易日取一个起点 · "
                "每个小图独立纵轴 · "
                f"生成 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ),
            14,
            "400",
        )
    )
    elements.append(f'<line x1="36" y1="94" x2="74" y2="94" stroke="#b91c1c" stroke-width="4"/>')
    avg_note = "完整折" if complete_paths else "展示折"
    elements.append(text(84, 99, f"组合等权平均期末 {avg_portfolio[-1]:.2f}x（{avg_note}）", 13, "600"))
    elements.append(f'<line x1="286" y1="94" x2="324" y2="94" stroke="#1d4ed8" stroke-width="4"/>')
    elements.append(text(334, 99, f"{BENCHMARK_NAME}平均期末 {avg_benchmark[-1]:.2f}x", 13, "600"))
    elements.append(text(640, 99, f"红线=组合等权 · 蓝线=基准({BENCHMARK_NAME}) · 橙色虚线=未满持有期当前进度", 13, "400", color="#475569"))

    for idx, path in enumerate(paths):
        row = idx // cols
        col = idx % cols
        cell_x = left + col * (cell_w + gap_x)
        cell_y = header_h + row * (cell_h + gap_y)
        plot_x = cell_x + 40
        plot_y = cell_y + 42
        plot_w = cell_w - 52
        plot_h = cell_h - 104 - stock_extra_h
        x_scale, y_scale, y_min, y_max = path_scale(path, plot_x, plot_y, plot_w, plot_h)
        excess_pct = path["excess_return"] * 100
        is_partial = bool(path.get("partial"))
        actual_hold_td = int(path.get("actual_hold_td") or max(0, len(path.get("dates", [])) - 1))
        title_color = "#b45309" if is_partial else ("#b91c1c" if excess_pct >= 0 else "#475569")
        window_label = f"未满 {actual_hold_td}/{hold_td}日" if is_partial else f"完整 {hold_td}日"
        stock_lines = holding_name_lines(path)
        title_names = holding_names(path, replacement_top_n)
        stock_title = " · 股票：" + "、".join(title_names) if title_names else ""
        replacement_label = topn_replacement_label(path)
        replacement_title = (
            f" · 首期{replacement_prefix}无上期可比"
            if path.get("topn_replaced_count") is None
            else f" · 较上期{replacement_prefix}替换 {int(path['topn_replaced_count'])} 只"
        )
        fold_title = (
            f"起点 {path['as_of']} · {window_label} · 有效持仓 {path['stock_count']} 只 · "
            f"组合收益 {path['portfolio_return'] * 100:.2f}% · "
            f"基准收益 {path['benchmark_return'] * 100:.2f}% · "
            f"超额 {path['excess_return'] * 100:.2f}%"
            f"{replacement_title}"
            f"{stock_title}"
        )
        elements.append(
            f'<rect x="{cell_x:.2f}" y="{cell_y:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" '
            'rx="4" fill="#ffffff" stroke="#dbe3ee"/>'
        )
        elements.append(text(cell_x + 12, cell_y + 21, f"{idx + 1:02d}. {path['as_of']} · {window_label} · 超额 {excess_pct:+.1f}%", 12, "700", color=title_color))
        elements.append(
            text(
                cell_x + 12,
                cell_y + 38,
                (
                    f"组合 {path['portfolio_return'] * 100:+.1f}% / "
                    f"基准 {path['benchmark_return'] * 100:+.1f}% / "
                    f"{path['stock_count']}只"
                ),
                10,
                "400",
                color="#475569",
            )
        )
        elements.append(text(cell_x + cell_w - 12, cell_y + 38, replacement_label, 10, "600", "end", "#475569"))
        elements.append(
            f'<rect x="{plot_x:.2f}" y="{plot_y:.2f}" width="{plot_w:.2f}" height="{plot_h:.2f}" '
            'fill="#fbfdff" stroke="#edf2f7"/>'
        )
        for tick_value in (y_min, 1.0, y_max):
            if tick_value < y_min or tick_value > y_max:
                continue
            y = y_scale(tick_value)
            dash = ' stroke-dasharray="4 4"' if abs(tick_value - 1.0) < 1e-9 else ""
            elements.append(
                f'<line x1="{plot_x:.2f}" y1="{y:.2f}" x2="{plot_x + plot_w:.2f}" '
                f'y2="{y:.2f}" stroke="#e6ebf2"{dash}/>'
            )
        elements.append(text(plot_x - 7, y_scale(y_max) + 4, f"{y_max:.1f}x", 9, "400", "end", "#64748b"))
        elements.append(text(plot_x - 7, y_scale(1.0) + 4, "1.0x", 9, "400", "end", "#64748b"))
        elements.append(text(plot_x - 7, y_scale(y_min) + 4, f"{y_min:.1f}x", 9, "400", "end", "#64748b"))
        elements.append(text(plot_x, plot_y + plot_h + 14, "0", 9, "400", "middle", "#64748b"))
        elements.append(text(plot_x + plot_w, plot_y + plot_h + 14, str(hold_td), 9, "400", "middle", "#64748b"))
        if is_partial:
            progress_x = x_scale(actual_hold_td)
            elements.append(
                f'<line x1="{progress_x:.2f}" y1="{plot_y:.2f}" x2="{progress_x:.2f}" '
                f'y2="{plot_y + plot_h:.2f}" stroke="#f59e0b" stroke-width="1.4" '
                'stroke-dasharray="4 4"/>'
            )
            elements.append(text(progress_x, plot_y + plot_h + 28, f"{actual_hold_td}日", 9, "600", "middle", "#b45309"))
        if stock_lines:
            for line_idx, line in enumerate(stock_lines):
                prefix = "股票：" if line_idx == 0 else ""
                elements.append(
                    text(
                        cell_x + 12,
                        plot_y + plot_h + 34 + line_idx * stock_line_gap,
                        prefix + line,
                        9,
                        "500",
                        color="#334155",
                    )
                )
        elements.append(
            '<polyline fill="none" stroke="#1d4ed8" stroke-width="1.8" stroke-opacity="0.9" '
            f'points="{svg_points(path["benchmark_path"], x_scale, y_scale)}"><title>{html_escape(BENCHMARK_NAME + " " + fold_title)}</title></polyline>'
        )
        elements.append(
            '<polyline fill="none" stroke="#b91c1c" stroke-width="1.9" stroke-opacity="0.95" '
            f'points="{svg_points(path["portfolio_path"], x_scale, y_scale)}"><title>{html_escape("组合 " + fold_title)}</title></polyline>'
        )
        end_x = x_scale(actual_hold_td)
        end_anchor = "end" if not is_partial else "start"
        label_dx = -2 if not is_partial else 4
        elements.append(text(end_x + label_dx, y_scale(path["portfolio_path"][-1]) - 4, f"{path['portfolio_path'][-1]:.2f}", 9, "700", end_anchor, "#b91c1c"))
        elements.append(text(end_x + label_dx, y_scale(path["benchmark_path"][-1]) + 10, f"{path['benchmark_path'][-1]:.2f}", 9, "700", end_anchor, "#1d4ed8"))

    if full_path:
        full_x = left
        full_y = header_h + grid_h + full_chart_gap
        full_w = width - left - right
        full_h = full_chart_h
        plot_x = full_x + 58
        plot_y = full_y + 58
        plot_w = full_w - 92
        plot_h = full_h - 112
        values = [1.0] + full_path["portfolio_path"] + full_path["benchmark_path"]
        y_min = min(values)
        y_max = max(values)
        if y_max - y_min < 0.08:
            y_min -= 0.04
            y_max += 0.04
        else:
            pad = (y_max - y_min) * 0.10
            y_min = max(0.0, y_min - pad)
            y_max += pad
        max_idx = max(1, len(full_path["dates"]) - 1)

        def full_x_scale(idx: int) -> float:
            return plot_x + plot_w * idx / max_idx

        def full_y_scale(value: float) -> float:
            return plot_y + plot_h * (y_max - value) / (y_max - y_min)

        full_excess_pct = full_path["excess_return"] * 100
        elements.append(
            f'<rect x="{full_x:.2f}" y="{full_y:.2f}" width="{full_w:.2f}" height="{full_h:.2f}" '
            'rx="4" fill="#ffffff" stroke="#dbe3ee"/>'
        )
        elements.append(text(full_x + 18, full_y + 26, "完整历史拼接收益图", 17, "700"))
        elements.append(
            text(
                full_x + 18,
                full_y + 49,
                (
                    f"{full_path['dates'][0]} ~ {full_path['dates'][-1]} · "
                    f"{full_path['source_folds']} 个窗口 · "
                    f"组合 {full_path['portfolio_path'][-1]:.2f}x / "
                    f"基准 {full_path['benchmark_path'][-1]:.2f}x / "
                    f"超额 {full_excess_pct:+.1f}%"
                ),
                12,
                "500",
                color="#475569",
            )
        )
        elements.append(
            f'<rect x="{plot_x:.2f}" y="{plot_y:.2f}" width="{plot_w:.2f}" height="{plot_h:.2f}" '
            'fill="#fbfdff" stroke="#edf2f7"/>'
        )
        tick_values = sorted({round(y_min, 4), 1.0, round((y_min + y_max) / 2.0, 4), round(y_max, 4)})
        for tick_value in tick_values:
            if tick_value < y_min or tick_value > y_max:
                continue
            y = full_y_scale(tick_value)
            dash = ' stroke-dasharray="4 4"' if abs(tick_value - 1.0) < 1e-9 else ""
            elements.append(
                f'<line x1="{plot_x:.2f}" y1="{y:.2f}" x2="{plot_x + plot_w:.2f}" '
                f'y2="{y:.2f}" stroke="#e6ebf2"{dash}/>'
            )
            elements.append(text(plot_x - 8, y + 4, f"{tick_value:.2f}x", 9, "400", "end", "#64748b"))
        for idx in (0, max_idx // 2, max_idx):
            x = full_x_scale(idx)
            elements.append(
                f'<line x1="{x:.2f}" y1="{plot_y:.2f}" x2="{x:.2f}" '
                f'y2="{plot_y + plot_h:.2f}" stroke="#edf2f7"/>'
            )
            elements.append(text(x, plot_y + plot_h + 18, full_path["dates"][idx], 10, "500", "middle", "#64748b"))
        elements.append(
            '<polyline fill="none" stroke="#1d4ed8" stroke-width="2.2" stroke-opacity="0.9" '
            f'points="{svg_points(full_path["benchmark_path"], full_x_scale, full_y_scale)}">'
            f'<title>{html_escape(BENCHMARK_NAME)} 完整拼接曲线</title></polyline>'
        )
        elements.append(
            '<polyline fill="none" stroke="#b91c1c" stroke-width="2.4" stroke-opacity="0.95" '
            f'points="{svg_points(full_path["portfolio_path"], full_x_scale, full_y_scale)}">'
            '<title>组合完整拼接曲线</title></polyline>'
        )
        end_x = full_x_scale(max_idx)
        elements.append(text(end_x - 4, full_y_scale(full_path["portfolio_path"][-1]) - 6, f"{full_path['portfolio_path'][-1]:.2f}x", 10, "700", "end", "#b91c1c"))
        elements.append(text(end_x - 4, full_y_scale(full_path["benchmark_path"][-1]) + 13, f"{full_path['benchmark_path'][-1]:.2f}x", 10, "700", "end", "#1d4ed8"))
        elements.append(f'<line x1="{full_x + full_w - 318:.2f}" y1="{full_y + 28:.2f}" x2="{full_x + full_w - 280:.2f}" y2="{full_y + 28:.2f}" stroke="#b91c1c" stroke-width="4"/>')
        elements.append(text(full_x + full_w - 270, full_y + 33, "组合", 12, "600", color="#334155"))
        elements.append(f'<line x1="{full_x + full_w - 218:.2f}" y1="{full_y + 28:.2f}" x2="{full_x + full_w - 180:.2f}" y2="{full_y + 28:.2f}" stroke="#1d4ed8" stroke-width="4"/>')
        elements.append(text(full_x + full_w - 170, full_y + 33, BENCHMARK_NAME, 12, "600", color="#334155"))
        elements.append(
            text(
                full_x + 18,
                full_y + full_h - 22,
                (
                    f"完整曲线最大回撤：组合 {full_path['portfolio_max_drawdown'] * 100:.1f}% · "
                    f"基准 {full_path['benchmark_max_drawdown'] * 100:.1f}%"
                ),
                12,
                "500",
                color="#475569",
            )
        )

    elements.append(text(width / 2, height - 18, "每个小图横轴为买入后交易日，纵轴为当折净值倍数；底部完整图按各折日度收益从旧到新拼接。", 12, "400", "middle", "#475569"))

    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",Arial,sans-serif;}</style>\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(svg, encoding="utf-8")
    return {
        "file": str(output_file),
        "folds": len(complete_paths),
        "partial_folds": len(partial_paths),
        "displayed_folds": len(paths),
        "hold_td": hold_td,
        "fold_interval_td": fold_interval_td,
        "replacement_top_n": replacement_top_n,
        "latest_topn_replaced_count": topn_replacement_counts[-1]["replaced_count"] if topn_replacement_counts else None,
        "topn_replacement_counts": topn_replacement_counts,
        "chart_type": "small_multiples_with_full_path" if full_path else "small_multiples",
        "columns": cols,
        "rows": rows,
        "oldest_as_of": paths[0]["as_of"],
        "latest_as_of": paths[-1]["as_of"],
        "avg_portfolio_final_nav": round(avg_portfolio[-1], 4),
        "avg_benchmark_final_nav": round(avg_benchmark[-1], 4),
        "full_path_points": len(full_path["dates"]) if full_path else 0,
        "full_path_start_date": full_path["dates"][0] if full_path else None,
        "full_path_end_date": full_path["dates"][-1] if full_path else None,
        "full_portfolio_final_nav": round(full_path["portfolio_path"][-1], 4) if full_path else None,
        "full_benchmark_final_nav": round(full_path["benchmark_path"][-1], 4) if full_path else None,
        "full_portfolio_max_drawdown_pct": round(full_path["portfolio_max_drawdown"] * 100, 3) if full_path else None,
        "full_benchmark_max_drawdown_pct": round(full_path["benchmark_max_drawdown"] * 100, 3) if full_path else None,
    }


def _create_long_style_fold_path_chart(
    cfg: Dict[str, Any],
    output_file: Path,
    hold_td: Optional[int] = None,
    *,
    title: str = "长线当前参数各折走势小图矩阵",
    strategy: str = "long",
) -> Dict[str, Any]:
    if strategy not in {"long", "smallcap"}:
        raise ValueError(f"不支持的长线风格策略: {strategy}")
    is_smallcap = strategy == "smallcap"
    hold_choices = SMALLCAP_HOLD_CHOICES if is_smallcap else LONG_HOLD_CHOICES
    hold_td = int(hold_td or hold_choices[0])
    series = full_series_map()
    cal = fold_calendar(series)
    benchmark_series, _ = load_benchmark_series()
    if not benchmark_series:
        raise RuntimeError(f"缺少{BENCHMARK_NAME}ETF基准序列，无法生成走势图。")

    broad_cfg = broad_smallcap_candidate_config() if is_smallcap else broad_long_candidate_config()
    universe_fn = smallcap_optimizer_universe_codes if is_smallcap else long_optimizer_universe_codes
    build_fn = build_smallcap_candidates if is_smallcap else build_long_candidates
    factors = SMALLCAP_FACTORS if is_smallcap else LONG_FACTORS
    score_fn = score_prepared_smallcap_candidates if is_smallcap else score_prepared_long_candidates
    deep_codes, _ = universe_fn(series, broad_cfg)
    anchor_fn = smallcap_anchor_offsets if is_smallcap else long_anchor_offsets
    partial_anchor_fn = smallcap_partial_anchor_offsets if is_smallcap else long_partial_anchor_offsets
    fold_interval_td = SMALLCAP_FOLD_STEP_TD if is_smallcap else LONG_FOLD_STEP_TD
    fold_dates = [cal[-off] for off in anchor_fn(hold_td) if off <= len(cal)]
    partial_fold_dates = [cal[-off] for off in partial_anchor_fn(hold_td) if off <= len(cal)]
    paths: List[Dict[str, Any]] = []

    def attach_holdings(path: Dict[str, Any], picks: List[Dict[str, Any]]) -> Dict[str, Any]:
        path["top_codes"] = [pick["code"] for pick in picks[:5]]
        path["holdings"] = [
            {
                "code": pick.get("code"),
                "name": pick.get("name") or pick.get("code"),
            }
            for pick in picks
        ]
        return path

    for as_of in fold_dates:
        cands, _ = build_fn(broad_cfg, as_of=as_of, universe=deep_codes)
        prepared = prepare_candidate_factor_scores(cands, factors)
        picks = score_fn(prepared, cfg, entry_series=series, as_of=as_of)
        path = portfolio_fold_path(picks, series, benchmark_series, as_of, hold_td)
        if path is None:
            continue
        paths.append(attach_holdings(path, picks))

    for as_of in partial_fold_dates:
        cands, _ = build_fn(broad_cfg, as_of=as_of, universe=deep_codes)
        prepared = prepare_candidate_factor_scores(cands, factors)
        picks = score_fn(prepared, cfg, entry_series=series, as_of=as_of)
        path = portfolio_fold_path(
            picks, series, benchmark_series, as_of, hold_td, allow_partial=True
        )
        if path is None:
            continue
        path["partial"] = True
        paths.append(attach_holdings(path, picks))

    chart = write_long_fold_paths_svg(
        paths,
        output_file,
        hold_td,
        holding_top_n=cfg.get("top_n"),
        title=title,
        fold_interval_td=fold_interval_td,
    )
    holding_codes = {
        str(holding.get("code") or "").zfill(6)
        for path in paths
        for holding in path.get("holdings", [])
        if holding.get("code")
    }
    invalid_holdings = holding_codes - deep_codes
    if invalid_holdings:
        raise RuntimeError(
            f"{strategy} 回测持仓越出候选池: {sorted(invalid_holdings)[:10]}"
        )
    chart.update({
        "strategy": strategy,
        "universe": "is_hot_money=1" if is_smallcap else "is_leader=1",
        "eligible_universe_count": len(deep_codes),
        "holding_code_count": len(holding_codes),
        "holding_pool_audit_passed": True,
    })
    return chart


def create_long_fold_path_chart(
    cfg: Dict[str, Any], output_file: Path, hold_td: Optional[int] = None, *,
    title: str = "长线当前参数各折走势小图矩阵",
) -> Dict[str, Any]:
    return _create_long_style_fold_path_chart(
        cfg, output_file, hold_td, title=title, strategy="long"
    )


def create_smallcap_fold_path_chart(
    cfg: Dict[str, Any], output_file: Path, hold_td: Optional[int] = None, *,
    title: str = "小盘当前参数各折走势小图矩阵",
) -> Dict[str, Any]:
    return _create_long_style_fold_path_chart(
        cfg, output_file, hold_td, title=title, strategy="smallcap"
    )


def create_best_long_fold_path_chart(long_result: Dict[str, Any]) -> Dict[str, Any]:
    best = long_result["best"]
    return create_long_fold_path_chart(
        best["config"],
        LONG_FOLD_PATH_CHART_FILE,
        int(best["hold_td"]),
        title="长线最优参数各折走势小图矩阵",
    )


def create_best_smallcap_fold_path_chart(smallcap_result: Dict[str, Any]) -> Dict[str, Any]:
    best = smallcap_result["best"]
    return create_smallcap_fold_path_chart(
        best["config"],
        SMALLCAP_FOLD_PATH_CHART_FILE,
        int(best["hold_td"]),
        title="小盘最优参数各折走势小图矩阵",
    )


def _init_process_progress_lock(lock) -> None:
    """让两个子进程共享 tqdm 写锁，避免双进度条互相插入输出。"""
    if tqdm is not None and lock is not None:
        tqdm.set_lock(lock)


def _process_pool_kwargs() -> Dict[str, Any]:
    if tqdm is None:
        return {}
    try:
        return {
            "initializer": _init_process_progress_lock,
            "initargs": (multiprocessing.RLock(),),
        }
    except Exception:
        return {}


def _run_strategy_optimizer(
    strategy: str,
    iterations: int,
    seed: int,
    progress_position: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    if strategy == "long":
        return optimize_long(iterations, rng, progress_position)
    if strategy == "smallcap":
        return optimize_smallcap(iterations, rng, progress_position)
    if strategy == "short":
        return optimize_short(iterations, rng, progress_position)
    raise ValueError(f"unknown strategy: {strategy}")


def run_optimization(
    iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
    seed: int = 20260611,
    persist: bool = True,
    strategies: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    strategy_names = tuple(dict.fromkeys(strategies or ("long", "smallcap", "short")))
    invalid = [name for name in strategy_names if name not in {"long", "smallcap", "short"}]
    if invalid or not strategy_names:
        raise ValueError(f"unsupported strategies: {invalid or strategy_names}")
    master_rng = random.Random(seed)
    strategy_seeds = {
        "long": master_rng.randrange(2 ** 31),
        "smallcap": master_rng.randrange(2 ** 31),
        "short": master_rng.randrange(2 ** 31),
    }
    started = datetime.now()
    strategy_results: Dict[str, Dict[str, Any]] = {}
    all_jobs = {
        "long": (strategy_seeds["long"], 0),
        "smallcap": (strategy_seeds["smallcap"], 1),
        "short": (strategy_seeds["short"], 2),
    }
    jobs = {name: all_jobs[name] for name in strategy_names}
    if len(jobs) == 1:
        key, (child_seed, position) = next(iter(jobs.items()))
        try:
            strategy_results[key] = _run_strategy_optimizer(
                key, iterations, child_seed, position
            )
        except Exception as exc:
            raise RuntimeError(f"{key} 策略参数搜索失败：{exc}") from exc
    else:
        with ProcessPoolExecutor(max_workers=len(jobs), **_process_pool_kwargs()) as executor:
            futures = {
                executor.submit(_run_strategy_optimizer, key, iterations, child_seed, position): key
                for key, (child_seed, position) in jobs.items()
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    strategy_results[key] = future.result()
                except Exception as exc:
                    raise RuntimeError(f"{key} 策略参数搜索失败：{exc}") from exc

    result: Dict[str, Any] = {
        "generated_at": started.strftime("%Y-%m-%d %H:%M:%S"),
        "iterations_per_strategy": iterations,
        "optimized_strategies": list(strategy_names),
        "seed": seed,
        "strategy_seeds": strategy_seeds,
        "research_basis": RESEARCH_BASIS,
    }
    result.update(strategy_results)
    if persist and "long" in strategy_results:
        try:
            chart = create_best_long_fold_path_chart(result["long"])
            result["long"]["best_fold_path_chart"] = chart
            result["long"].setdefault("notes", []).append(
                f"最优参数每折走势图已生成：{chart['file']}。"
            )
        except Exception as exc:
            result["long"].setdefault("notes", []).append(
                f"最优参数每折走势图生成失败：{exc}"
            )
    if persist and "smallcap" in strategy_results:
        try:
            chart = create_best_smallcap_fold_path_chart(result["smallcap"])
            result["smallcap"]["best_fold_path_chart"] = chart
            result["smallcap"].setdefault("notes", []).append(
                f"最优参数每折走势图已生成：{chart['file']}。"
            )
        except Exception as exc:
            result["smallcap"].setdefault("notes", []).append(
                f"最优参数每折走势图生成失败：{exc}"
            )
    result["elapsed_seconds"] = round((datetime.now() - started).total_seconds(), 3)
    if persist:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        standard_names = [name for name in strategy_names if name in {"long", "short"}]
        if standard_names:
            persisted_result: Dict[str, Any] = {}
            if len(standard_names) == 1 and OUTPUT_FILE.exists():
                try:
                    previous = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
                    if isinstance(previous, dict):
                        persisted_result.update(previous)
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            standard_payload = {
                key: value for key, value in result.items() if key != "smallcap"
            }
            standard_payload["optimized_strategies"] = standard_names
            standard_payload["strategy_seeds"] = {
                name: strategy_seeds[name] for name in standard_names
            }
            persisted_result.update(standard_payload)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
                json.dump(persisted_result, fp, ensure_ascii=False, indent=2)

            optimized_config: Dict[str, Any] = {}
            if len(standard_names) == 1 and OPTIMIZED_CONFIG_FILE.exists():
                try:
                    previous = json.loads(OPTIMIZED_CONFIG_FILE.read_text(encoding="utf-8"))
                    if isinstance(previous, dict):
                        optimized_config.update(previous)
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            optimized_config.update({
                "generated_at": result["generated_at"],
                "iterations_per_strategy": iterations,
                "optimized_strategies": standard_names,
                "seed": seed,
                "strategy_seeds": {
                    name: strategy_seeds[name] for name in standard_names
                },
                "caveat": "基于当前可用的本地/代理数据优化；刷新历史价格或短线信号快照后建议重新运行。",
            })
            if "short" in strategy_results:
                optimized_config["short_universe_version"] = SHORT_UNIVERSE_VERSION
            saved_config = optimized_config.setdefault("config", {})
            saved_scores = optimized_config.setdefault("scores", {})
            for name in standard_names:
                section = strategy_results[name]
                best_config = section["best"]["config"]
                if name == "short":
                    selected_hold_days = section["best"].get(
                        "hold_days", best_config.get("hold_days")
                    )
                    if selected_hold_days is not None:
                        best_config = _short_config_with_hold_days(
                            best_config, selected_hold_days
                        )
                saved_config[name] = best_config
                saved_scores[f"{name}_objective"] = section["best"]["objective"]
            with open(OPTIMIZED_CONFIG_FILE, "w", encoding="utf-8") as fp:
                json.dump(optimized_config, fp, ensure_ascii=False, indent=2)

        if "smallcap" in strategy_results:
            smallcap_section = strategy_results["smallcap"]
            smallcap_report = {
                key: value for key, value in result.items()
                if key not in {"long", "short"}
            }
            smallcap_report["optimized_strategies"] = ["smallcap"]
            smallcap_report["strategy_seeds"] = {
                "smallcap": strategy_seeds["smallcap"]
            }
            with open(SMALLCAP_OUTPUT_FILE, "w", encoding="utf-8") as fp:
                json.dump(smallcap_report, fp, ensure_ascii=False, indent=2)
            smallcap_config = {
                "generated_at": result["generated_at"],
                "iterations": iterations,
                "optimized_strategies": ["smallcap"],
                "seed": seed,
                "strategy_seed": strategy_seeds["smallcap"],
                "smallcap_universe_version": SMALLCAP_UNIVERSE_VERSION,
                "caveat": "复用长线PIT回测框架；股票池为当前游资小盘成员，存在当前成员回看历史的幸存者偏差。",
                "config": {"smallcap": smallcap_section["best"]["config"]},
                "scores": {"smallcap_objective": smallcap_section["best"]["objective"]},
            }
            with open(SMALLCAP_OPTIMIZED_CONFIG_FILE, "w", encoding="utf-8") as fp:
                json.dump(smallcap_config, fp, ensure_ascii=False, indent=2)
    return result


LONG_SUMMARY_FIELDS = [
    ("folds", "有效回测窗口", ""),
    ("indep_folds", "去重叠独立折", ""),
    ("hold_td", "持有交易日", ""),
    ("avg_fold_weight", "平均折权重", ""),
    ("min_fold_weight", "最小折权重", ""),
    ("max_fold_weight", "最大折权重", ""),
    ("avg_portfolio_return_pct", "平均组合收益", "%"),
    ("avg_benchmark_return_pct", "平均基准收益", "%"),
    ("avg_excess_pct", "平均超额收益", "%"),
    ("avg_excess_ann_pct", "年化超额收益", "%"),
    ("hit_rate", "跑赢比例", "%"),
    ("ir", "信息比率", ""),
    ("downside_ir", "下行信息比(Sortino)", ""),
    ("worst_fold_excess_pct", "最差单窗口超额", "%"),
    ("tail_cvar_excess_pct", "最差20%折均值超额", "%"),
    ("avg_fold_max_drawdown_pct", "平均折内回撤", "%"),
    ("max_drawdown_pct", "组合最大回撤", "%"),
    ("benchmark_max_drawdown_pct", "基准最大回撤", "%"),
]


SHORT_SUMMARY_FIELDS = [
    ("samples", "样本数", ""),
    ("avg_return", "平均收益", "%"),
    ("avg_benchmark", "平均基准收益", "%"),
    ("avg_excess", "平均超额收益", "%"),
    ("hit_rate", "胜率", "%"),
    ("sharpe_like", "类夏普", ""),
    ("max_drawdown", "最大回撤", "%"),
]


def format_summary_value(value: Any, suffix: str = "") -> str:
    if value is None:
        return "无"
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return f"{text}{suffix}" if suffix else text


def print_metric_block(title: str, summary: Dict[str, Any], fields: List[Tuple[str, str, str]]) -> None:
    print(f"{title}:")
    for key, label, suffix in fields:
        if key in summary:
            print(f"  {label}: {format_summary_value(summary.get(key), suffix)}")


def print_summary(result: Dict[str, Any]) -> None:
    print(
        f"生成时间: {result['generated_at']} · "
        f"每个策略迭代次数={result['iterations_per_strategy']} · "
        f"随机种子={result['seed']}"
    )
    strategy_labels = {
        "long": "长线策略",
        "smallcap": "小盘策略",
        "short": "短线策略",
    }
    strategy_names = result.get("optimized_strategies") or [
        key for key in ("long", "smallcap", "short") if result.get(key)
    ]
    for key in strategy_names:
        section = result.get(key)
        if not section:
            continue
        best = section["best"]
        label = strategy_labels.get(key, key)
        print()
        print("=" * 96)
        print(
            f"{label}最优结果: 第 {best['iteration']} 次迭代 · "
            f"目标分={best['objective']} · 入选={best['selected_count']} 只"
        )
        print("=" * 96)
        for note in section.get("notes", []):
            print(f"[说明] {note}")
        print(f"代表代码: {' '.join(best.get('top_codes', []))}")
        if key in {"long", "smallcap"}:
            print_metric_block("回测摘要", best["summary"], LONG_SUMMARY_FIELDS)
            chart = section.get("best_fold_path_chart")
            if chart:
                fold_text = f"{chart.get('folds')} 完整折"
                if chart.get("partial_folds"):
                    fold_text += f" + {chart.get('partial_folds')} 个未满展示折"
                print(
                    f"每折走势图: {chart.get('file')} · "
                    f"{fold_text} · 平均组合期末 {chart.get('avg_portfolio_final_nav')}x · "
                    f"平均基准期末 {chart.get('avg_benchmark_final_nav')}x"
                )
                if chart.get("full_path_points"):
                    print(
                        f"完整拼接收益图: {chart.get('full_path_start_date')} ~ {chart.get('full_path_end_date')} · "
                        f"组合 {chart.get('full_portfolio_final_nav')}x · "
                        f"基准 {chart.get('full_benchmark_final_nav')}x · "
                        f"组合最大回撤 {chart.get('full_portfolio_max_drawdown_pct')}%"
                    )
        else:
            print_metric_block("真实前推回测", best["actual_backtest"], SHORT_SUMMARY_FIELDS)
            print_metric_block("代理评分回测", best["proxy_backtest"], SHORT_SUMMARY_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="搜索股票策略因子权重和参数")
    parser.add_argument("--strategy", choices=["all", "long", "smallcap", "short"], default="all",
                        help="选择要重跑的策略；单策略模式保留另一策略已有优化配置")
    parser.add_argument("--iterations", type=int, default=DEFAULT_OPTIMIZATION_ITERATIONS, help="每个策略的搜索迭代次数")
    parser.add_argument("--seed", type=int, default=2333, help="随机种子")
    parser.add_argument("--no-persist", action="store_true", help="只打印结果，不写入结果文件")
    args = parser.parse_args()
    strategies = ("long", "smallcap", "short") if args.strategy == "all" else (args.strategy,)
    try:
        result = run_optimization(
            args.iterations,
            args.seed,
            persist=not args.no_persist,
            strategies=strategies,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1) from None
    print_summary(result)
    if not args.no_persist:
        print(f"\n已保存优化结果: {OUTPUT_FILE}")
        print(f"已保存优化默认配置: {OPTIMIZED_CONFIG_FILE}")
        if "smallcap" in strategies:
            print(f"已保存小盘优化结果: {SMALLCAP_OUTPUT_FILE}")
            print(f"已保存小盘独立优化配置: {SMALLCAP_OPTIMIZED_CONFIG_FILE}")


if __name__ == "__main__":
    main()
