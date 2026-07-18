from __future__ import annotations

import os
import signal
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, TextIO

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
        "resource_key": None,
        "log_lines": [],
    }


def get_job_state(job_id: str) -> Dict[str, Any]:
    with _LOCK:
        state = dict(_STATES.get(job_id, _empty_state(job_id)))
        state["command"] = list(state.get("command", []))
        state["log_lines"] = list(state.get("log_lines", []))
        return state


def is_resource_running(resource_key: str) -> bool:
    if not resource_key:
        return False
    with _LOCK:
        return any(
            state.get("running") and state.get("resource_key") == resource_key
            for state in _STATES.values()
        )


def start_command_job(
    job_id: str,
    command: List[str],
    *,
    cwd: Path = ROOT_DIR,
    timeout: int = 1800,
    on_success: JobCallback = None,
    resource_key: Optional[str] = None,
) -> bool:
    with _LOCK:
        current = _STATES.get(job_id)
        if current and current.get("running"):
            return False
        if resource_key and any(
            state.get("running") and state.get("resource_key") == resource_key
            for state in _STATES.values()
        ):
            return False
        _STATES[job_id] = {
            **_empty_state(job_id),
            "running": True,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "command": list(command),
            "command_text": _format_command(command),
            "resource_key": resource_key,
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
        env = _job_environment()
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
            **_popen_group_kwargs(),
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
            terminate_process_tree(process)
            return_code = process.returncode
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


def _job_environment(base_env: Mapping[str, str] | None = None) -> Dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _popen_group_kwargs(platform_name: str | None = None) -> Dict[str, Any]:
    if (platform_name or os.name) == "nt":
        create_new_process_group = getattr(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
            0x00000200,
        )
        return {"creationflags": create_new_process_group}
    return {"start_new_session": True}


def _format_command(command: List[str], *, platform_name: str | None = None) -> str:
    if (platform_name or os.name) == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def terminate_process_tree(
    process: subprocess.Popen[Any],
    *,
    grace_seconds: float = 2.0,
    platform_name: str | None = None,
) -> None:
    """Stop a subprocess and every descendant in its process group.

    Refresh commands often launch shell/Python grandchildren.  Killing only the
    direct process lets those children keep writing SQLite/JSON after the UI has
    already reported a timeout.
    """
    if process.poll() is not None:
        return

    if (platform_name or os.name) == "nt":
        _terminate_windows_process_tree(process, grace_seconds=grace_seconds)
        return

    def send(sig: int) -> None:
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    send(signal.SIGTERM)
    try:
        process.wait(timeout=max(0.0, grace_seconds))
        return
    except subprocess.TimeoutExpired:
        pass
    send(getattr(signal, "SIGKILL", signal.SIGTERM))
    try:
        process.wait(timeout=max(1.0, grace_seconds))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _terminate_windows_process_tree(
    process: subprocess.Popen[Any],
    *,
    grace_seconds: float,
) -> None:
    """Use Windows' built-in taskkill so grandchildren cannot outlive a timeout."""
    try:
        completed = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=max(1.0, grace_seconds),
        )
    except (OSError, subprocess.TimeoutExpired):
        completed = None

    try:
        process.wait(timeout=max(1.0, grace_seconds))
        return
    except subprocess.TimeoutExpired:
        pass

    # taskkill can fail under a restricted account. At least stop the direct
    # child rather than leaving the task running indefinitely.
    try:
        if completed is None or completed.returncode != 0:
            process.terminate()
            process.wait(timeout=max(1.0, grace_seconds))
            return
    except (OSError, subprocess.TimeoutExpired):
        pass
    process.kill()
    process.wait()


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
