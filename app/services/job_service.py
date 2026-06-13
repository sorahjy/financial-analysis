from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO

from app.config import ROOT_DIR


JobCallback = Optional[Callable[[], None]]

_LOCK = threading.Lock()
_STATES: Dict[str, Dict[str, Any]] = {}
_MAX_LOG_LINES = 500


def _empty_state(job_id: str) -> Dict[str, Any]:
    return {
        "id": job_id,
        "running": False,
        "started_at": None,
        "finished_at": None,
        "ok": None,
        "error": "",
        "elapsed_sec": None,
        "command": [],
        "command_text": "",
        "log_lines": [],
    }


def get_job_state(job_id: str) -> Dict[str, Any]:
    with _LOCK:
        state = dict(_STATES.get(job_id, _empty_state(job_id)))
        state["command"] = list(state.get("command", []))
        state["log_lines"] = list(state.get("log_lines", []))
        return state


def start_command_job(
    job_id: str,
    command: List[str],
    *,
    cwd: Path = ROOT_DIR,
    timeout: int = 1800,
    on_success: JobCallback = None,
) -> bool:
    with _LOCK:
        current = _STATES.get(job_id)
        if current and current.get("running"):
            return False
        _STATES[job_id] = {
            **_empty_state(job_id),
            "running": True,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "command": list(command),
            "command_text": _format_command(command),
            "log_lines": [f"$ {_format_command(command)}"],
        }

    thread = threading.Thread(
        target=_run_command,
        args=(job_id, command, cwd, timeout, on_success),
        daemon=True,
    )
    thread.start()
    return True


def _run_command(
    job_id: str,
    command: List[str],
    cwd: Path,
    timeout: int,
    on_success: JobCallback,
) -> None:
    started = time.time()
    ok = False
    error = ""
    return_code: Optional[int] = None
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        reader = threading.Thread(
            target=_stream_process_output,
            args=(job_id, process.stdout),
            daemon=True,
        )
        reader.start()
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            error = f"任务超时({timeout}秒)"
            _append_job_log(job_id, error)
            process.kill()
            return_code = process.wait()
        finally:
            reader.join(timeout=1)
            if process.stdout:
                process.stdout.close()

        ok = return_code == 0 and not error
        if not ok and not error:
            recent_lines = _recent_job_logs(job_id, limit=5)
            error = " | ".join(recent_lines) if recent_lines else f"exit={return_code}"
            _append_job_log(job_id, f"任务失败: {error}")
    except OSError as exc:
        error = str(exc)
        _append_job_log(job_id, f"任务启动失败: {error}")

    if ok and on_success:
        try:
            on_success()
        except Exception as exc:  # Keep callback failures visible in status.
            ok = False
            error = str(exc)
            _append_job_log(job_id, f"任务回调失败: {error}")

    if ok:
        _append_job_log(job_id, "任务完成")

    with _LOCK:
        _STATES[job_id] = {
            **_STATES.get(job_id, _empty_state(job_id)),
            "running": False,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ok": ok,
            "error": error,
            "elapsed_sec": round(time.time() - started, 1),
        }


def _format_command(command: List[str]) -> str:
    return shlex.join(command)


def _stream_process_output(job_id: str, stream: Optional[TextIO]) -> None:
    if stream is None:
        return
    for line in stream:
        _append_job_log(job_id, line.rstrip())


def _append_job_log(job_id: str, line: str) -> None:
    if not line:
        return
    with _LOCK:
        state = _STATES.setdefault(job_id, _empty_state(job_id))
        lines = list(state.get("log_lines", []))
        lines.append(line)
        state["log_lines"] = lines[-_MAX_LOG_LINES:]


def _recent_job_logs(job_id: str, *, limit: int) -> List[str]:
    with _LOCK:
        lines = list(_STATES.get(job_id, _empty_state(job_id)).get("log_lines", []))
    return lines[-limit:]
