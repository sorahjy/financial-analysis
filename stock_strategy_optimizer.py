

"""
Parameter search and proxy backtesting for stock_advanced_strategies.py.

The optimizer deliberately separates two layers:

1. price_backtest: uses only locally available price rows. This is the most
   honest metric, but current local files do not contain enough multi-year
   history for the requested 2-5 year horizon.
2. proxy_objective: used when the local price sample is too short. It combines
   excess return on the available recent window with risk, hit rate, and for
   short-term Dragon Tiger List picks, event-quality statistics.

Run 200 iterations per strategy:
    python3 -B stock_strategy_optimizer.py --iterations 200
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from html import escape as html_escape
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

from stock_advanced_strategies import (
    DATA_DIR,
    DEFAULT_CONFIG,
    LONG_FACTORS,
    SHORT_FACTORS,
    apply_scores,
    build_long_candidates,
    build_short_candidates,
    clean_round,
    combined_recent_rows,
    deep_merge,
    direct_factor_score,
    first_not_none,
    load_cn_stock_index,
    load_fundamental_stocks,
    passes_long_hard_filters,
    percentile_factor_scores,
    price_history_rows,
    rerank_scored,
    safe_float,
)


OUTPUT_FILE = DATA_DIR / "stock_strategy_optimization.json"
OPTIMIZED_CONFIG_FILE = DATA_DIR / "stock_strategy_optimized_config.json"
# 长线超额基准：沪深300 与 中证500 按日等权再平衡的混合指数（各占50%）。
# 不直接用中证800：中证800按市值加权、前300只占权重大头，走势≈沪深300，体现不出中盘；
# 50/50 等权混合让大盘与中盘平权，更贴合本策略 large-mid 选股域。
# 换基准只改 BENCHMARK_COMPONENTS（增删成分即可，单成分时退化为该指数本身）；
# csi300_current 等仍是"指数成分"选股因子，与收益基准无关、不受影响。
BENCHMARK_NAME = "沪深300+中证500等权"
BENCHMARK_COMPONENTS = [
    ("510310", DATA_DIR / "csi300_etf_nav.json"),  # 沪深300 ETF
    ("510580", DATA_DIR / "csi500_etf_nav.json"),  # 中证500 ETF
]
LONG_FOLD_PATH_CHART_FILE = DATA_DIR / "stock_strategy_best_fold_paths.svg"
DEFAULT_OPTIMIZATION_ITERATIONS = 200

# 长线 walk-forward 回测参数：利用 data/stock_data/*.history 的多年日线，
# 每隔约1.5个月取一折，固定持有 250 个交易日（约 1 年）。
LONG_HOLD_CHOICES = [250]  # 长线持有期固定为250交易日(约1年)
LONG_FOLD_STEP_TD = 30     # 折锚点间隔(交易日)；约每1.5个月一折，增加验证密度
LONG_MAX_LOOKBACK_TD = 2400
LONG_COST = 0.004          # 单折买卖往返成本（佣金+冲击的粗略值）
LONG_MIN_VALID_PICKS = 5   # 一折内至少几只持仓有价格数据才计入
LONG_MIN_FOLDS = 8         # 有效折数下限，低于此的配置不参与选优(防1~3折彩票配置)
LONG_SOFT_TARGET_FOLDS = 20
LONG_UNTRUSTED_WEIGHT_CAPS = {
    # 指数约束两因子权重上限（按用户要求压低，避免它们主导选参）；
    # csi300_current 另因无历史成分快照(前视)只保留极轻偏好。
    "csi300_current": 0.15,
    "csi300_persistence": 0.8,
}
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
# 高点回撤(抄底)过滤的"高点至今跌幅下限"候选值；搜索时强制开启该过滤(见 random_long_config)。
# 仅在深档 [50,60,70] 里搜：本策略定位深反转，但让模型在三档里挑、不写死。
# 尾部由防御因子保底 + 下行目标(CVaR/回撤/最差折惩罚)控制，而不是靠关闭抄底。
LONG_HIGH_DD_PCT_CHOICES = [50, 60, 70]
RESEARCH_BASIS = [
    "Fama-French: size, value, profitability, investment; plus momentum/reversal sorts from the Kenneth French data library.",
    "Barra/MSCI-style families: size, value, momentum, volatility, liquidity, growth, leverage and quality.",
    "Conservative formula: low volatility + momentum + payout yield.",
    "WorldQuant-style short-horizon price-volume alphas.",
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


def long_anchor_offsets(hold_td: int) -> List[int]:
    """折锚点：距最新交易日的偏移（交易日数），保证持有期之后仍有出场价。"""
    return list(range(hold_td + 1, LONG_MAX_LOOKBACK_TD, LONG_FOLD_STEP_TD))


def fold_calendar(series: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """全市场交易日历(各股日期并集，升序)，把折偏移映射成 PIT 时点日期。"""
    dates: set = set()
    for rows in series.values():
        for r in rows:
            d = str(r.get("date", ""))
            if len(d) == 10:
                dates.add(d)
    return sorted(dates)


def _load_etf_nav_levels(path: Path) -> Dict[str, float]:
    """读单个 ETF 累计净值文件 -> {date: 累计净值}。"""
    levels: Dict[str, float] = {}
    try:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return levels
    for row in payload.get("records", []):
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
        if not path.exists():
            return [], [f"未找到基准成分 {code}（{path.name}），无法构建{BENCHMARK_NAME}基准。"]
        levels = _load_etf_nav_levels(path)
        if not levels:
            return [], [f"基准成分 {code}（{path.name}）没有可用累计净值。"]
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

    稠密折每 LONG_FOLD_STEP_TD 个交易日取一折，持有期远大于步长时相邻折高度重叠
    （hold=250/step=30 时相邻折重叠约88%），会让胜率/IR虚高、把几十折当成几十个独立样本。
    抽稀到相邻保留折至少间隔 min_gap_td 个交易日后，再算风险统计才可信。
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
    ⚠ 注意泄漏：稠密折相邻重叠很高（hold=250/step=30 约 88%），随机切分会让训练折与
    验证折在时间上重叠 → 含样本内泄漏，验证不再是严格样本外（gap 被人为收窄、val 被抬高）。
    每折按其 as_of 的确定性随机键排序后切分，保证可复现、且各配置用同一划分。
    """
    if len(pairs) < 2:
        return list(pairs), []
    keyed = sorted(pairs, key=lambda p: random.Random(f"{salt}-{p['as_of']}").random())
    k = max(1, min(int(len(keyed) * train_frac), len(keyed) - 1))
    return keyed[:k], keyed[k:]


def excess_risk_stats(pairs: List[Dict[str, Any]], decorr_gap_td: int) -> Dict[str, float]:
    """尾部/风险统计：最差折、CVaR、下行偏差在去相关子样本上算；回撤用全样本均值。"""
    indep = decorrelate_pairs(pairs, decorr_gap_td)
    use = indep if len(indep) >= LONG_MIN_INDEP_FOLDS else pairs
    excess = sorted(p["excess_return"] for p in use)
    n = len(excess)
    worst = excess[0]
    tail_k = max(1, math.ceil(n * 0.2))
    cvar = mean(excess[:tail_k])                      # 最差20%折的平均超额
    negatives = [e for e in excess if e < 0]
    downside_dev = math.sqrt(sum(e * e for e in negatives) / n) if negatives else 0.0
    mean_fold_mdd = mean(p.get("portfolio_max_drawdown", 0.0) for p in pairs) if pairs else 0.0
    return {
        "worst": worst,
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


def benchmark_window_from_date(
    benchmark_series: List[Dict[str, Any]],
    as_of: str,
    hold_td: int,
) -> Optional[List[Dict[str, Any]]]:
    entry_idx = row_index_on_or_before(benchmark_series, as_of)
    if entry_idx is None:
        return None
    exit_idx = entry_idx + hold_td
    if exit_idx >= len(benchmark_series):
        return None
    return benchmark_series[entry_idx:exit_idx + 1]


def portfolio_fold_path(
    picks: List[Dict[str, Any]],
    series: Dict[str, List[Dict[str, Any]]],
    benchmark_series: List[Dict[str, Any]],
    as_of: str,
    hold_td: int,
) -> Optional[Dict[str, Any]]:
    benchmark_window = benchmark_window_from_date(benchmark_series, as_of, hold_td)
    if not benchmark_window:
        return None
    dates = [row["date"] for row in benchmark_window]
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
        prices = price_path_on_dates(series.get(pick["code"]) or [], dates)
        entry = prices[0]
        if entry is None or entry <= 0 or prices[-1] is None:
            continue
        stock_paths.append([
            price / entry if price is not None and price > 0 else None
            for price in prices
        ])
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
    ann = 250.0 / hold_td
    vol = pstdev(excess) if len(excess) > 1 else 0.0
    max_dd = max(row.get("portfolio_max_drawdown", 0.0) for row in ordered)
    benchmark_max_dd = max(row.get("benchmark_max_drawdown", 0.0) for row in ordered)
    mean_e = mean(excess)
    risk = excess_risk_stats(ordered, max(1, hold_td // 2))
    sortino = mean_e / max(risk["downside_dev"], LONG_DOWNSIDE_DEV_FLOOR)
    return {
        "folds": len(excess),
        "indep_folds": risk["indep_folds"],
        "hold_td": hold_td,
        "avg_portfolio_return_pct": round(mean(portfolio) * 100, 3),
        "avg_benchmark_return_pct": round(mean(benchmark) * 100, 3),
        "avg_excess_pct": round(mean_e * 100, 3),
        "avg_excess_ann_pct": round(mean_e * ann * 100, 3),
        "hit_rate": round(sum(1 for e in excess if e > 0) / len(excess) * 100, 2),
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
    excess = sorted(p["excess_return"] for p in pairs)
    n = len(excess)
    ann = 250.0 / hold_td
    median = excess[n // 2] if n % 2 else (excess[n // 2 - 1] + excess[n // 2]) / 2.0
    mean_e = mean(excess)
    hit = sum(1 for e in excess if e > 0) / n
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
        - tail_breach * 100 * 0.8                      # 最差折击穿地板的硬惩罚
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
    ⚠ 因稠密折重叠，此验证含样本内泄漏、非严格样本外(val 会贴近 train、gap 收窄)。
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
    fold_count_penalty = max(0, LONG_SOFT_TARGET_FOLDS - len(all_pairs)) * 0.75
    negative_validation_penalty = max(0.0, -val_ann) * 0.8
    # 全样本最深折内回撤超过45%再线性惩罚(基准约37%)，压制高回撤的"漂亮"配置。
    worst_mdd_penalty = max(0.0, full_max_mdd - 45.0) * 0.10
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


def score_prepared_long_candidates(
    prepared: List[Dict[str, Any]],
    config: Dict[str, Any],
    include_details: bool = False,
) -> List[Dict[str, Any]]:
    weights = config.get("weights", {})
    min_score = safe_float(config.get("min_score")) or 0.0
    top_n = max(0, int(first_not_none(config.get("top_n"), 30)))
    scored = []

    for row in prepared:
        item = row["item"]
        if not passes_long_hard_filters(item, config):
            continue
        raw = item.get("raw_factors", {})
        prepared_scores = row["scores"]
        factor_scores = {} if include_details else None
        weighted_sum = 0.0
        weight_sum = 0.0
        positive_weight_count = 0

        for spec in LONG_FACTORS:
            weight = safe_float(weights.get(spec.key))
            if weight is None:
                weight = spec.default_weight
            weight = max(0.0, weight)
            score = prepared_scores.get(spec.key, spec.missing_score)
            if weight > 0:
                weighted_sum += score * weight
                weight_sum += weight
                positive_weight_count += 1
            if factor_scores is not None:
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
    dates = []
    for follower in pick.get("followers", []) or []:
        date = str(follower.get("date", ""))[:10]
        if len(date) == 10:
            dates.append(date)
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
    for key, cap in LONG_UNTRUSTED_WEIGHT_CAPS.items():
        current = safe_float(weights.get(key))
        if current is not None:
            weights[key] = round(min(current, cap), 3)
    # 当前沪深300成分没有历史快照，优化器不把它作为硬过滤；实盘页面仍可手动开启。
    cfg["require_csi300"] = False
    return cfg


def random_long_config(rng: random.Random, iteration: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG["long"])
    cfg["weights"] = mutate_weights(
        rng, cfg["weights"], [f.key for f in LONG_FACTORS], iteration, floors=LONG_WEIGHT_FLOORS
    )
    # 高点回撤过滤：按用户设定全程强制开启；跌幅下限在 [50,60,70] 搜，i==0 基线取中间档60。
    cfg["require_high_drawdown"] = True
    cfg["min_high_drawdown_pct"] = LONG_HIGH_DD_PCT_CHOICES[1]
    if iteration > 0:
        # top_n 不参与搜索，固定用 DEFAULT_CONFIG 的 10
        cfg["min_score"] = rng.choice([45, 50, 55, 58, 60, 63, 66, 70])
        # 市值下限放开到100亿，让中盘股（中证500档）有入池机会
        cfg["min_market_cap_yi"] = rng.choice([100, 200, 300, 500, 800])
        cfg["min_listing_years"] = rng.choice([2, 3, 5, 8, 10])
        cfg["min_high_drawdown_pct"] = rng.choice(LONG_HIGH_DD_PCT_CHOICES)
    return constrain_long_search_config(cfg)


def random_short_config(rng: random.Random, iteration: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG["short"])
    cfg["weights"] = mutate_weights(rng, cfg["weights"], [f.key for f in SHORT_FACTORS], iteration)
    if iteration > 0:
        # top_n 不参与搜索，固定用 DEFAULT_CONFIG 的 8
        cfg["min_score"] = rng.choice([35, 40, 45, 50, 55, 60, 65, 70])
        cfg["min_lhb_count"] = rng.choice([1, 2, 3, 4])
        cfg["min_hot_money_concurrent"] = rng.choice([0, 1, 2, 3])
        cfg["max_consecutive_limit_up"] = rng.choice([1, 2, 3, 4])
        cfg["hold_days_min"] = rng.choice([1, 2])
        cfg["hold_days_max"] = rng.choice([3, 4, 5])
    return cfg


def broad_long_candidate_config() -> Dict[str, Any]:
    broad_cfg = copy.deepcopy(DEFAULT_CONFIG["long"])
    broad_cfg.update({
        "exclude_st": False,
        "min_market_cap_yi": 0,
        "min_listing_years": 0,
        "require_csi300": False,
        "require_high_drawdown": False,
        "min_score": 0,
        "top_n": 80,
    })
    return broad_cfg


def optimize_long(iterations: int, rng: random.Random) -> Dict[str, Any]:
    broad_cfg = broad_long_candidate_config()
    series = full_series_map()
    cal = fold_calendar(series)
    cal_pos = {date: idx for idx, date in enumerate(cal)}  # as_of -> 交易日序号，供去相关/隔离带用
    benchmark_series, benchmark_notes = load_benchmark_series()
    # PIT 候选只在有足够多年日线的票上建（快且干净）；最长持有 + 1年回看
    min_hist = max(LONG_HOLD_CHOICES) + 260
    deep_codes = {code for code, rows in series.items() if len(rows) >= min_hist}

    # 折偏移 → PIT 时点日期（全市场日历倒数第 offset 个交易日）。[0] 最近、[-1] 最旧。
    fold_dates_by_hold = {
        h: [cal[-off] for off in long_anchor_offsets(h) if off <= len(cal)]
        for h in LONG_HOLD_CHOICES
    }

    pit_cache: Dict[str, List[Dict[str, Any]]] = {}       # as_of -> PIT broad 候选预评分
    fold_stats_cache: Dict[Tuple[str, int, Tuple[str, ...]], Optional[Dict[str, float]]] = {}
    notes: List[str] = []

    def get_pit(as_of: str) -> List[Dict[str, Any]]:
        if as_of not in pit_cache:
            cands, nts = build_long_candidates(broad_cfg, as_of=as_of, universe=deep_codes)
            pit_cache[as_of] = prepare_candidate_factor_scores(cands, LONG_FACTORS)
            if not notes:
                notes.extend(nts)
        return pit_cache[as_of]

    def get_fold_stats(
        picks: List[Dict[str, Any]],
        as_of: str,
        hold_td: int,
    ) -> Optional[Dict[str, float]]:
        key = (as_of, hold_td, tuple(p["code"] for p in picks))
        if key not in fold_stats_cache:
            fold_stats_cache[key] = portfolio_fold_stats(
                picks, series, benchmark_series, as_of, hold_td
            )
        return fold_stats_cache[key]

    best = None
    trace = []
    for i in range(iterations):
        cfg = random_long_config(rng, i)
        hold_td = rng.choice(LONG_HOLD_CHOICES) if i else LONG_HOLD_CHOICES[0]
        fold_dates = fold_dates_by_hold[hold_td]
        pairs: List[Dict[str, Any]] = []
        for as_of in fold_dates:
            picks = score_prepared_long_candidates(get_pit(as_of), cfg)  # 该折用 PIT 因子重新选股
            fold_stats = get_fold_stats(picks, as_of, hold_td)
            if fold_stats is None:
                continue
            pairs.append({"as_of": as_of, "cal_idx": cal_pos.get(as_of, 0), **fold_stats})
        # 风险/尾部统计的去相关间隔：相邻保留折至少隔半个持有期(重叠≤50%)
        decorr_gap_td = max(1, hold_td // 2)
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
        repr_picks = score_prepared_long_candidates(get_pit(fold_dates[0]), cfg) if fold_dates else []
        row = {
            "iteration": i + 1,
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
        trace.append(row)
        if best is None or row["objective"] > best["objective"]:
            best = row
    assert best is not None
    return {
        "strategy": "long",
        "iterations": iterations,
        "candidate_count": len(deep_codes),
        "notes": benchmark_notes + notes + [
            "PIT 滚动前推回测：每折以全市场日历倒数第N个交易日为时点，财报(按法定披露截止日)/价格/估值/分红全部按当时可见重算因子后再选股，消除前视偏差；固定持有250交易日(约1年)，组合等权收益-成本-沪深300与中证500按日等权混合(各50%)累计净值基准。",
            "训练/验证按折随机打乱后切分(~60/40)，按用户口径取消时间分块与隔离带——训练/验证折时间混合、各覆盖全部regime。⚠注意：稠密折相邻重叠~88%，随机切分下训练折与验证折在时间上重叠，含样本内泄漏，验证不再是严格样本外(gap人为收窄、验证超额被抬高)，系有意为之的口径。",
            "稠密折(每30交易日一折)与持有期高度重叠，会把胜率/IR当成独立样本而虚高；故 worst/CVaR/Sortino 等风险统计在'相邻保留折至少隔半个持有期'的去相关子样本(indep_folds)上计算。",
            "长线选优目标：训练目标分×0.55+验证×0.45；目标显式纳入 CVaR(最差20%折均值)、持有期组合回撤、Sortino下行信息比，并对最差折跌破-15%、最深折回撤超45%、训练/验证年化超额差、有效折数不足、验证折为负分别惩罚；命中率权重已下调。",
            "③ 风险预算保底：搜索不再把低波动/低负债/保守投资因子权重清零(下限0.30/0.20/0.20)。高点回撤(抄底)过滤按设定强制开启，高点至今跌幅下限在20~70%搜优；接飞刀尾部改由防御因子保底 + 下行目标(CVaR/回撤/最差折惩罚)约束，而非靠关闭抄底。",
            "max_drawdown_pct 为各折持有期内按基准交易日历逐日估值的组合路径最大回撤(取最深折)；avg_fold_max_drawdown_pct 为各折回撤均值；worst_fold_excess_pct 为稠密折最差单折超额，tail_cvar_excess_pct 为去重叠后最差20%折的平均超额。",
            "残留前视：沪深300成分与质押用当前快照(无历史数据)；优化器已将 csi300_current/csi300_persistence 权重分别限制到 0.15/0.8，且不再搜索 require_csi300 或成分稳定硬过滤，候选仍限于本地有多年日线的票(约中证800)。",
        ],
        "best": best,
        "top_iterations": sorted(trace, key=lambda x: x["objective"], reverse=True)[:10],
        "convergence": convergence_summary(trace, iterations),
    }


def optimize_short(iterations: int, rng: random.Random) -> Dict[str, Any]:
    broad_cfg = copy.deepcopy(DEFAULT_CONFIG["short"])
    broad_cfg.update({
        "min_lhb_count": 0,
        "min_hot_money_concurrent": 0,
        "max_consecutive_limit_up": 99,
        "min_score": 0,
        "top_n": 80,
    })
    broad, notes = build_short_candidates(broad_cfg)
    series = recent_series_map()
    best = None
    trace = []
    for i in range(iterations):
        cfg = random_short_config(rng, i)
        picks = score_short_candidates(broad, cfg)
        hold_days = rng.choice([1, 2, 3, 4, 5]) if i else 1
        actual = short_actual_backtest(picks, series, hold_days=hold_days)
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
            "iteration": i + 1,
            "objective": round(objective, 5),
            "objective_mode": objective_mode,
            "hold_days": hold_days,
            "selected_count": len(picks),
            "actual_backtest": actual,
            "proxy_backtest": proxy,
            "top_codes": [p["code"] for p in picks[:5]],
            "config": cfg,
        }
        trace.append(row)
        if best is None or row["objective"] > best["objective"]:
            best = row
    assert best is not None
    return {
        "strategy": "short",
        "iterations": iterations,
        "candidate_count": len(broad),
        "notes": notes + [
            "短线真实前推收益回测仅在事件日之后存在本地价格数据时使用。",
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
) -> Dict[str, Any]:
    if not paths:
        raise ValueError("没有可用于画图的有效长线回测折。")

    paths = sorted(paths, key=lambda row: row["as_of"])
    avg_portfolio = average_paths(paths, "portfolio_path")
    avg_benchmark = average_paths(paths, "benchmark_path")

    cols = 4
    rows = math.ceil(len(paths) / cols)
    width = 1600
    header_h = 122
    footer_h = 42
    left = 54
    right = 42
    gap_x = 24
    gap_y = 22
    cell_w = (width - left - right - gap_x * (cols - 1)) / cols
    cell_h = 230
    height = int(header_h + rows * cell_h + (rows - 1) * gap_y + footer_h)

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
        max_idx = max(1, min(len(path["portfolio_path"]), len(path["benchmark_path"])) - 1)

        def x_scale(idx: int) -> float:
            return plot_x + plot_w * idx / max_idx

        def y_scale(value: float) -> float:
            return plot_y + plot_h * (y_max - value) / (y_max - y_min)

        return x_scale, y_scale, y_min, y_max

    def holding_name_lines(path: Dict[str, Any]) -> List[str]:
        holdings = path.get("holdings", []) or []
        names = [
            str(item.get("name") or item.get("code") or "").strip()
            for item in holdings[:10]
            if isinstance(item, dict) and (item.get("name") or item.get("code"))
        ]
        if not names:
            return []
        first = " · ".join(names[:5])
        second = " · ".join(names[5:10])
        return [line for line in (first, second) if line]

    elements: List[str] = []
    elements.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>')
    elements.append(text(36, 42, "长线最优参数各折走势小图矩阵", 25, "700"))
    elements.append(
        text(
            36,
            72,
            (
                f"{len(paths)} 个有效回测窗口 · 持有 {hold_td} 个交易日 · "
                f"每 {LONG_FOLD_STEP_TD} 个交易日取一个起点 · "
                "每个小图独立纵轴 · "
                f"生成 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ),
            14,
            "400",
        )
    )
    elements.append(f'<line x1="36" y1="94" x2="74" y2="94" stroke="#b91c1c" stroke-width="4"/>')
    elements.append(text(84, 99, f"组合等权平均期末 {avg_portfolio[-1]:.2f}x", 13, "600"))
    elements.append(f'<line x1="286" y1="94" x2="324" y2="94" stroke="#1d4ed8" stroke-width="4"/>')
    elements.append(text(334, 99, f"{BENCHMARK_NAME}平均期末 {avg_benchmark[-1]:.2f}x", 13, "600"))
    elements.append(text(640, 99, f"红线=组合等权 · 蓝线=基准({BENCHMARK_NAME})", 13, "400", color="#475569"))

    for idx, path in enumerate(paths):
        row = idx // cols
        col = idx % cols
        cell_x = left + col * (cell_w + gap_x)
        cell_y = header_h + row * (cell_h + gap_y)
        plot_x = cell_x + 40
        plot_y = cell_y + 42
        plot_w = cell_w - 52
        plot_h = cell_h - 104
        x_scale, y_scale, y_min, y_max = path_scale(path, plot_x, plot_y, plot_w, plot_h)
        excess_pct = path["excess_return"] * 100
        title_color = "#b91c1c" if excess_pct >= 0 else "#475569"
        stock_lines = holding_name_lines(path)
        stock_title = " · 股票：" + "、".join(stock_lines).replace(" · ", "、") if stock_lines else ""
        fold_title = (
            f"起点 {path['as_of']} · 有效持仓 {path['stock_count']} 只 · "
            f"组合收益 {path['portfolio_return'] * 100:.2f}% · "
            f"基准收益 {path['benchmark_return'] * 100:.2f}% · "
            f"超额 {path['excess_return'] * 100:.2f}%"
            f"{stock_title}"
        )
        elements.append(
            f'<rect x="{cell_x:.2f}" y="{cell_y:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" '
            'rx="4" fill="#ffffff" stroke="#dbe3ee"/>'
        )
        elements.append(text(cell_x + 12, cell_y + 21, f"{idx + 1:02d}. {path['as_of']} · 超额 {excess_pct:+.1f}%", 12, "700", color=title_color))
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
        if stock_lines:
            elements.append(text(cell_x + 12, plot_y + plot_h + 34, "股票：" + stock_lines[0], 9, "500", color="#334155"))
            if len(stock_lines) > 1:
                elements.append(text(cell_x + 12, plot_y + plot_h + 49, stock_lines[1], 9, "500", color="#334155"))
        elements.append(
            '<polyline fill="none" stroke="#1d4ed8" stroke-width="1.8" stroke-opacity="0.9" '
            f'points="{svg_points(path["benchmark_path"], x_scale, y_scale)}"><title>{html_escape(BENCHMARK_NAME + " " + fold_title)}</title></polyline>'
        )
        elements.append(
            '<polyline fill="none" stroke="#b91c1c" stroke-width="1.9" stroke-opacity="0.95" '
            f'points="{svg_points(path["portfolio_path"], x_scale, y_scale)}"><title>{html_escape("组合 " + fold_title)}</title></polyline>'
        )
        elements.append(text(plot_x + plot_w - 2, y_scale(path["portfolio_path"][-1]) - 4, f"{path['portfolio_path'][-1]:.2f}", 9, "700", "end", "#b91c1c"))
        elements.append(text(plot_x + plot_w - 2, y_scale(path["benchmark_path"][-1]) + 10, f"{path['benchmark_path'][-1]:.2f}", 9, "700", "end", "#1d4ed8"))

    elements.append(text(width / 2, height - 18, "每个小图横轴为买入后交易日，纵轴为当折净值倍数；纵轴为独立缩放，方便看清单折形态。", 12, "400", "middle", "#475569"))

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
        "folds": len(paths),
        "hold_td": hold_td,
        "fold_interval_td": LONG_FOLD_STEP_TD,
        "chart_type": "small_multiples",
        "columns": cols,
        "rows": rows,
        "oldest_as_of": paths[0]["as_of"],
        "latest_as_of": paths[-1]["as_of"],
        "avg_portfolio_final_nav": round(avg_portfolio[-1], 4),
        "avg_benchmark_final_nav": round(avg_benchmark[-1], 4),
    }


def create_best_long_fold_path_chart(long_result: Dict[str, Any]) -> Dict[str, Any]:
    best = long_result["best"]
    cfg = best["config"]
    hold_td = int(best["hold_td"])
    series = full_series_map()
    cal = fold_calendar(series)
    benchmark_series, _ = load_benchmark_series()
    if not benchmark_series:
        raise RuntimeError(f"缺少{BENCHMARK_NAME}ETF基准序列，无法生成走势图。")

    min_hist = max(LONG_HOLD_CHOICES) + 260
    deep_codes = {code for code, rows in series.items() if len(rows) >= min_hist}
    broad_cfg = broad_long_candidate_config()
    fold_dates = [cal[-off] for off in long_anchor_offsets(hold_td) if off <= len(cal)]
    paths: List[Dict[str, Any]] = []
    for as_of in fold_dates:
        cands, _ = build_long_candidates(broad_cfg, as_of=as_of, universe=deep_codes)
        prepared = prepare_candidate_factor_scores(cands, LONG_FACTORS)
        picks = score_prepared_long_candidates(prepared, cfg)
        path = portfolio_fold_path(picks, series, benchmark_series, as_of, hold_td)
        if path is None:
            continue
        path["top_codes"] = [pick["code"] for pick in picks[:5]]
        path["holdings"] = [
            {
                "code": pick.get("code"),
                "name": pick.get("name") or pick.get("code"),
            }
            for pick in picks
        ]
        paths.append(path)

    return write_long_fold_paths_svg(paths, LONG_FOLD_PATH_CHART_FILE, hold_td)


def run_optimization(
    iterations: int = DEFAULT_OPTIMIZATION_ITERATIONS,
    seed: int = 20260611,
    persist: bool = True,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    started = datetime.now()
    result = {
        "generated_at": started.strftime("%Y-%m-%d %H:%M:%S"),
        "iterations_per_strategy": iterations,
        "seed": seed,
        "research_basis": RESEARCH_BASIS,
        "long": optimize_long(iterations, rng),
        "short": optimize_short(iterations, rng),
    }
    if persist:
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
    result["elapsed_seconds"] = round((datetime.now() - started).total_seconds(), 3)
    if persist:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
            json.dump(result, fp, ensure_ascii=False, indent=2)
        optimized_config = {
            "generated_at": result["generated_at"],
            "iterations_per_strategy": iterations,
            "seed": seed,
            "config": {
                "long": result["long"]["best"]["config"],
                "short": result["short"]["best"]["config"],
            },
            "scores": {
                "long_objective": result["long"]["best"]["objective"],
                "short_objective": result["short"]["best"]["objective"],
            },
            "caveat": "基于当前可用的本地/代理数据优化；刷新历史价格与龙虎榜快照后建议重新运行。",
        }
        with open(OPTIMIZED_CONFIG_FILE, "w", encoding="utf-8") as fp:
            json.dump(optimized_config, fp, ensure_ascii=False, indent=2)
    return result


LONG_SUMMARY_FIELDS = [
    ("folds", "有效回测窗口", ""),
    ("indep_folds", "去重叠独立折", ""),
    ("hold_td", "持有交易日", ""),
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
        "short": "短线策略",
    }
    for key in ("long", "short"):
        section = result[key]
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
        if key == "long":
            print_metric_block("回测摘要", best["summary"], LONG_SUMMARY_FIELDS)
            chart = section.get("best_fold_path_chart")
            if chart:
                print(
                    f"每折走势图: {chart.get('file')} · "
                    f"{chart.get('folds')} 折 · 平均组合期末 {chart.get('avg_portfolio_final_nav')}x · "
                    f"平均基准期末 {chart.get('avg_benchmark_final_nav')}x"
                )
        else:
            print_metric_block("真实前推回测", best["actual_backtest"], SHORT_SUMMARY_FIELDS)
            print_metric_block("代理评分回测", best["proxy_backtest"], SHORT_SUMMARY_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="搜索股票策略因子权重和参数")
    parser.add_argument("--iterations", type=int, default=DEFAULT_OPTIMIZATION_ITERATIONS, help="每个策略的搜索迭代次数")
    parser.add_argument("--seed", type=int, default=20260611, help="随机种子")
    parser.add_argument("--no-persist", action="store_true", help="只打印结果，不写入结果文件")
    args = parser.parse_args()
    result = run_optimization(args.iterations, args.seed, persist=not args.no_persist)
    print_summary(result)
    if not args.no_persist:
        print(f"\n已保存优化结果: {OUTPUT_FILE}")
        print(f"已保存优化默认配置: {OPTIMIZED_CONFIG_FILE}")


if __name__ == "__main__":
    main()
