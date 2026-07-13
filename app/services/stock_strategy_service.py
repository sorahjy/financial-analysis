from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import DATA_DIR, ROOT_DIR
from app.services.job_service import get_job_state, start_command_job


_RUN_LOCK = threading.Lock()
LONG_DEFAULT_CHART_FILENAME = "stock_strategy_best_fold_paths.svg"
LONG_CUSTOM_CHART_PREFIX = "stock_strategy_fold_paths_custom_"
SMALLCAP_DEFAULT_CHART_FILENAME = "stock_strategy_smallcap_fold_paths.svg"
SMALLCAP_CUSTOM_CHART_PREFIX = "stock_strategy_smallcap_fold_paths_custom_"
STOCK_REFRESH_JOB_ID = "stock-refresh"
STOCK_OPTIMIZER_JOB_ID = "stock-optimizer"
STOCK_REFRESH_SCRIPT = "stock_radar_fresh_data.sh"


def run_stock_strategies(
    config: Optional[Dict[str, Any]] = None,
    *,
    persist: bool = False,
    invalidate: bool = False,
    include_search_pool: bool = False,
) -> Dict[str, Any]:
    from stock_advanced_strategies import invalidate_dir_fingerprints, run_strategies

    with _RUN_LOCK:
        if invalidate:
            invalidate_dir_fingerprints()
        return run_strategies(config or {}, persist=persist, include_search_pool=include_search_pool)


def stable_long_config(config: Optional[Dict[str, Any]]) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    from stock_advanced_strategies import deep_merge, get_default_config

    defaults = get_default_config()
    default_long = defaults["long"]
    source = config or {}
    override = source.get("long") if isinstance(source.get("long"), dict) else source
    merged = deep_merge(default_long, override or {})
    key = json.dumps(merged, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return merged, default_long, key


def build_long_backtest_chart(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return build_strategy_backtest_chart(config, strategy="long")


def build_strategy_backtest_chart(
    config: Optional[Dict[str, Any]] = None, *, strategy: str = "long"
) -> Dict[str, Any]:
    from stock_advanced_strategies import deep_merge, get_default_config
    from stock_strategy_optimizer import (
        LONG_FOLD_PATH_CHART_FILE,
        SMALLCAP_FOLD_PATH_CHART_FILE,
        create_long_fold_path_chart,
        create_smallcap_fold_path_chart,
    )

    if strategy not in {"long", "smallcap"}:
        raise ValueError("策略走势仅支持长线和小盘")

    defaults = get_default_config()
    default_strategy = defaults[strategy]
    source = config or {}
    override = source.get(strategy) if isinstance(source.get(strategy), dict) else source
    merged = deep_merge(default_strategy, override or {})
    key = json.dumps(merged, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    default_key = json.dumps(default_strategy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    is_default = key == default_key
    prefix = LONG_CUSTOM_CHART_PREFIX if strategy == "long" else SMALLCAP_CUSTOM_CHART_PREFIX
    if is_default:
        output_file = LONG_FOLD_PATH_CHART_FILE if strategy == "long" else SMALLCAP_FOLD_PATH_CHART_FILE
    else:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        output_file = DATA_DIR / f"{prefix}{digest}.svg"
    label = "长线" if strategy == "long" else "小盘"
    title = f"{label}{'默认' if is_default else '当前'}参数各折走势小图矩阵"
    chart_fn = create_long_fold_path_chart if strategy == "long" else create_smallcap_fold_path_chart

    with _RUN_LOCK:
        chart = chart_fn(merged, output_file, title=title)

    return {
        "strategy": strategy,
        "is_default": is_default,
        "file": str(output_file),
        "filename": output_file.name,
        "url": f"/api/stock/long-backtest-chart/file/{output_file.name}",
        "chart": chart,
    }


def resolve_long_backtest_chart_file(filename: str) -> Path:
    name = Path(filename).name
    if name != filename:
        raise FileNotFoundError(filename)
    allowed = (
        name == LONG_DEFAULT_CHART_FILENAME
        or (name.startswith(LONG_CUSTOM_CHART_PREFIX) and name.endswith(".svg"))
        or name == SMALLCAP_DEFAULT_CHART_FILENAME
        or (name.startswith(SMALLCAP_CUSTOM_CHART_PREFIX) and name.endswith(".svg"))
    )
    if not allowed:
        raise FileNotFoundError(filename)
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(filename)
    return path


def start_stock_data_refresh() -> bool:
    return start_command_job(
        STOCK_REFRESH_JOB_ID,
        ["bash", STOCK_REFRESH_SCRIPT],
        cwd=ROOT_DIR,
        timeout=10800,
        on_success=_after_stock_data_refresh,
        resource_key="stock-data-refresh",
    )


def stock_data_refresh_state() -> Dict[str, Any]:
    return get_job_state(STOCK_REFRESH_JOB_ID)


def start_optimizer_job() -> bool:
    from stock_data_refresh import resolve_python
    from stock_strategy_optimizer import DEFAULT_OPTIMIZATION_ITERATIONS

    return start_command_job(
        STOCK_OPTIMIZER_JOB_ID,
        [
            resolve_python(),
            "-B",
            "stock_strategy_optimizer.py",
            "--iterations",
            str(DEFAULT_OPTIMIZATION_ITERATIONS),
        ],
        cwd=ROOT_DIR,
        timeout=1800,
        on_success=_after_optimizer,
        # Optimization reads the same DB/artifacts that refresh rewrites.
        resource_key="stock-data-refresh",
    )


def optimizer_state_snapshot() -> Dict[str, Any]:
    return get_job_state(STOCK_OPTIMIZER_JOB_ID)


def _after_optimizer() -> None:
    from stock_advanced_strategies import invalidate_dir_fingerprints
    invalidate_dir_fingerprints()


def _after_stock_data_refresh() -> None:
    from stock_advanced_strategies import invalidate_dir_fingerprints

    invalidate_dir_fingerprints()
