from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import DATA_DIR, ROOT_DIR
from app.services.job_service import get_job_state, start_command_job


_RUN_LOCK = threading.Lock()
_OPTIMIZE_LOCK = threading.Lock()
LONG_DEFAULT_CHART_FILENAME = "stock_strategy_best_fold_paths.svg"
LONG_CUSTOM_CHART_PREFIX = "stock_strategy_fold_paths_custom_"
_OPTIMIZE_STATE: Dict[str, Any] = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "ok": None,
    "error": "",
    "elapsed_sec": None,
}
STOCK_REFRESH_JOB_ID = "stock-refresh"


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
    from stock_strategy_optimizer import LONG_FOLD_PATH_CHART_FILE, create_long_fold_path_chart

    merged, default_long, key = stable_long_config(config)
    default_key = json.dumps(default_long, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    is_default = key == default_key
    if is_default:
        output_file = LONG_FOLD_PATH_CHART_FILE
        title = "长线默认参数各折走势小图矩阵"
    else:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        output_file = DATA_DIR / f"{LONG_CUSTOM_CHART_PREFIX}{digest}.svg"
        title = "长线当前参数各折走势小图矩阵"

    with _RUN_LOCK:
        if not is_default or not output_file.exists():
            chart = create_long_fold_path_chart(merged, output_file, title=title)
        else:
            chart = {"file": str(output_file), "chart_type": "existing_default"}

    return {
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
        [
            "python",
            "stock_data_refresh.py",
            "--mode",
            "full",
            "--no-proxy",
        ],
        cwd=ROOT_DIR,
        timeout=1800,
        on_success=_after_stock_data_refresh,
    )


def stock_data_refresh_state() -> Dict[str, Any]:
    return get_job_state(STOCK_REFRESH_JOB_ID)


def start_optimizer_job() -> bool:
    with _OPTIMIZE_LOCK:
        if _OPTIMIZE_STATE["running"]:
            return False
        _OPTIMIZE_STATE.update(
            running=True,
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            finished_at=None,
            ok=None,
            error="",
            elapsed_sec=None,
        )
    threading.Thread(target=_run_optimizer_job, daemon=True).start()
    return True


def optimizer_state_snapshot() -> Dict[str, Any]:
    with _OPTIMIZE_LOCK:
        return dict(_OPTIMIZE_STATE)


def _run_optimizer_job() -> None:
    from stock_advanced_strategies import invalidate_dir_fingerprints
    from stock_data_refresh import resolve_python

    started = time.time()
    cmd = [resolve_python(), "-B", "stock_strategy_optimizer.py", "--iterations", "300"]
    ok = False
    error = ""
    try:
        completed = subprocess.run(
            cmd, cwd=str(ROOT_DIR), capture_output=True, text=True, timeout=1800
        )
        ok = completed.returncode == 0
        if not ok:
            tail = (completed.stderr or completed.stdout or "").strip().splitlines()
            error = " | ".join(tail[-3:]) if tail else f"exit={completed.returncode}"
    except subprocess.TimeoutExpired:
        error = "参数搜索超时(1800秒)"
    except OSError as exc:
        error = str(exc)
    if ok:
        invalidate_dir_fingerprints()
    with _OPTIMIZE_LOCK:
        _OPTIMIZE_STATE.update(
            running=False,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            ok=ok,
            error=error,
            elapsed_sec=round(time.time() - started, 1),
        )


def _after_stock_data_refresh() -> None:
    from stock_advanced_strategies import invalidate_dir_fingerprints

    invalidate_dir_fingerprints()
