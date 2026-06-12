

"""
Parameter search and proxy backtesting for stock_advanced_strategies.py.

The optimizer deliberately separates two layers:

1. price_backtest: uses only locally available price rows. This is the most
   honest metric, but current local files do not contain enough multi-year
   history for the requested 2-5 year horizon.
2. proxy_objective: used when the local price sample is too short. It combines
   excess return on the available recent window with risk, hit rate, and for
   short-term Dragon Tiger List picks, event-quality statistics.

Run at least 100 iterations per strategy:
    python3 -B stock_strategy_optimizer.py --iterations 100
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
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
    combined_recent_rows,
    deep_merge,
    first_not_none,
    load_fundamental_stocks,
    load_stock_universe,
    passes_long_hard_filters,
    rerank_scored,
    safe_float,
)


OUTPUT_FILE = DATA_DIR / "stock_strategy_optimization.json"
OPTIMIZED_CONFIG_FILE = DATA_DIR / "stock_strategy_optimized_config.json"
LONG_BACKTEST_ANCHORS = [20, 15, 10, 7, 5]
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


def benchmark_returns(
    series: Dict[str, List[Dict[str, Any]]],
    hold_days: int,
    anchors: Iterable[int],
) -> List[float]:
    csi300 = set(str(code).zfill(6) for code in load_stock_universe().get("csi300", []))
    out = []
    for anchor in anchors:
        fold_returns = []
        for code in csi300:
            rows = series.get(code) or []
            ret = pct_return_from_rows(rows, -anchor, hold_days)
            if ret is not None:
                fold_returns.append(ret)
        if fold_returns:
            out.append(mean(fold_returns))
    return out


def long_price_backtest(
    picks: List[Dict[str, Any]],
    series: Dict[str, List[Dict[str, Any]]],
    hold_days: int,
    fold_bm: Optional[List[float]] = None,
) -> Dict[str, Any]:
    anchors = LONG_BACKTEST_ANCHORS
    returns = []
    if fold_bm is None:
        fold_bm = benchmark_returns(series, hold_days, anchors)
    for anchor in anchors:
        fold_returns = []
        for pick in picks:
            rows = series.get(pick["code"]) or []
            ret = pct_return_from_rows(rows, -anchor, hold_days)
            if ret is not None:
                fold_returns.append(ret)
        if fold_returns:
            returns.append(mean(fold_returns) - 0.003)
    return summarize_returns(returns, fold_bm)


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
    dt = first_not_none(raw.get("dragon_tiger_composite"), 45.0)
    net_pct = safe_float(raw.get("lhb_net_buy_pct")) or 0.0
    concurrent = safe_float(raw.get("hot_money_concurrent")) or 0.0
    network = first_not_none(raw.get("seat_network_score"), 40.0)
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
        weights[key] = round(clamp(value, 0.0, 3.0), 3)
    return weights


def random_long_config(rng: random.Random, iteration: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG["long"])
    cfg["weights"] = mutate_weights(rng, cfg["weights"], [f.key for f in LONG_FACTORS], iteration)
    if iteration > 0:
        cfg["top_n"] = rng.choice([8, 10, 15, 20, 25, 30, 40])
        cfg["min_score"] = rng.choice([45, 50, 55, 58, 60, 63, 66, 70])
        cfg["min_market_cap_yi"] = rng.choice([300, 500, 800, 1000, 1500, 2000])
        cfg["min_listing_years"] = rng.choice([3, 5, 8, 10, 12])
        cfg["min_csi300_persistence"] = rng.choice([55, 60, 65, 70, 75, 80])
        cfg["require_csi300"] = rng.random() < 0.82
        cfg["require_high_drawdown"] = rng.random() < 0.25
        cfg["min_high_drawdown_pct"] = rng.choice([20, 25, 30, 35, 40, 50])
    return cfg


def random_short_config(rng: random.Random, iteration: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG["short"])
    cfg["weights"] = mutate_weights(rng, cfg["weights"], [f.key for f in SHORT_FACTORS], iteration)
    if iteration > 0:
        cfg["top_n"] = rng.choice([2, 3, 5, 8, 10, 15, 20, 30])
        cfg["min_score"] = rng.choice([35, 40, 45, 50, 55, 60, 65, 70])
        cfg["min_lhb_count"] = rng.choice([1, 2, 3, 4])
        cfg["min_hot_money_concurrent"] = rng.choice([0, 1, 2, 3])
        cfg["max_consecutive_limit_up"] = rng.choice([1, 2, 3, 4])
        cfg["hold_days_min"] = rng.choice([1, 2])
        cfg["hold_days_max"] = rng.choice([3, 4, 5])
    return cfg


def optimize_long(iterations: int, rng: random.Random) -> Dict[str, Any]:
    broad_cfg = copy.deepcopy(DEFAULT_CONFIG["long"])
    broad_cfg.update({
        "exclude_st": False,
        "min_market_cap_yi": 0,
        "min_listing_years": 0,
        "require_csi300": False,
        "min_csi300_persistence": 0,
        "require_high_drawdown": False,
        "min_score": 0,
        "top_n": 80,
    })
    broad, notes = build_long_candidates(broad_cfg)
    series = recent_series_map()
    anchors = LONG_BACKTEST_ANCHORS
    bm_cache: Dict[int, List[float]] = {}
    best = None
    trace = []
    for i in range(iterations):
        cfg = random_long_config(rng, i)
        picks = score_long_candidates(broad, cfg)
        hold_days = rng.choice([5, 7, 10, 12]) if i else 10
        if hold_days not in bm_cache:
            bm_cache[hold_days] = benchmark_returns(series, hold_days, anchors)
        summary = long_price_backtest(picks, series, hold_days=hold_days, fold_bm=bm_cache[hold_days])
        fallback = (mean([p["score"] for p in picks]) if picks else -999.0) * 0.03
        objective = objective_from_summary(summary, fallback_quality=fallback)
        if not picks:
            objective = -999.0
        row = {
            "iteration": i + 1,
            "objective": round(objective, 5),
            "hold_days_proxy": hold_days,
            "selected_count": len(picks),
            "summary": summary,
            "top_codes": [p["code"] for p in picks[:5]],
            "config": cfg,
        }
        trace.append(row)
        if best is None or row["objective"] > best["objective"]:
            best = row
    assert best is not None
    return {
        "strategy": "long",
        "iterations": iterations,
        "candidate_count": len(broad),
        "notes": notes + [
            "Long price backtest uses available recent local daily samples, not a true 2-5 year history.",
        ],
        "best": best,
        "top_iterations": sorted(trace, key=lambda x: x["objective"], reverse=True)[:10],
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
            "Short actual forward-return backtest is used only when local prices after the event date exist.",
            "Current local snapshots mostly require proxy scoring because Dragon Tiger List events are later than cached price rows.",
        ],
        "best": best,
        "top_iterations": sorted(trace, key=lambda x: x["objective"], reverse=True)[:10],
    }


def run_optimization(iterations: int = 100, seed: int = 20260611, persist: bool = True) -> Dict[str, Any]:
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
            "caveat": "Optimized on currently available local/proxy data; rerun after refreshing historical prices and Dragon Tiger List snapshots.",
        }
        with open(OPTIMIZED_CONFIG_FILE, "w", encoding="utf-8") as fp:
            json.dump(optimized_config, fp, ensure_ascii=False, indent=2)
    return result


def print_summary(result: Dict[str, Any]) -> None:
    print(f"Generated: {result['generated_at']} · iterations/strategy={result['iterations_per_strategy']} · seed={result['seed']}")
    for key in ("long", "short"):
        section = result[key]
        best = section["best"]
        print()
        print("=" * 96)
        print(f"{key.upper()} best iteration {best['iteration']} objective={best['objective']} selected={best['selected_count']}")
        print("=" * 96)
        for note in section.get("notes", []):
            print(f"[NOTE] {note}")
        print(f"Top codes: {' '.join(best.get('top_codes', []))}")
        if key == "long":
            print(f"Backtest: {best['summary']}")
        else:
            print(f"Actual: {best['actual_backtest']}")
            print(f"Proxy:  {best['proxy_backtest']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search factor weights and parameters for stock strategies")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args()
    result = run_optimization(args.iterations, args.seed, persist=not args.no_persist)
    print_summary(result)
    if not args.no_persist:
        print(f"\nSaved: {OUTPUT_FILE}")
        print(f"Optimized defaults: {OPTIMIZED_CONFIG_FILE}")


if __name__ == "__main__":
    main()


