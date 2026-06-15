from __future__ import annotations

import subprocess
import threading
import time
from typing import Any, Dict, Optional

from app.config import ROOT_DIR
from app.services.job_service import get_job_state, start_command_job


_RUN_LOCK = threading.Lock()
_OPTIMIZE_LOCK = threading.Lock()
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
) -> Dict[str, Any]:
    from stock_advanced_strategies import invalidate_dir_fingerprints, run_strategies

    with _RUN_LOCK:
        if invalidate:
            invalidate_dir_fingerprints()
        return run_strategies(config or {}, persist=persist)


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
