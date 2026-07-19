from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence


Command = Sequence[str]
Step = tuple[str, Command]
Runner = Callable[..., subprocess.CompletedProcess[object]]

_PROXY_KEYS = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
_TRUE_VALUES = {"1", "true", "yes", "on"}


def is_no_proxy_enabled(
    marker: str,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return whether a no-proxy marker is truthy, ignoring key case."""
    source = os.environ if env is None else env
    marker_lower = marker.lower()
    return any(
        key.lower() == marker_lower
        and str(value).strip().lower() in _TRUE_VALUES
        for key, value in source.items()
    )


def strip_proxy_environment(
    env: MutableMapping[str, str] | None = None,
    *,
    no_proxy_marker: str | None = None,
) -> MutableMapping[str, str]:
    """Disable environment proxies in-place, including mixed-case variants."""
    target = os.environ if env is None else env
    for key in list(target):
        if key.lower() in _PROXY_KEYS:
            target.pop(key, None)
    target["NO_PROXY"] = "*"
    target["no_proxy"] = "*"
    if no_proxy_marker:
        marker_lower = no_proxy_marker.lower()
        for key in list(target):
            if key.lower() == marker_lower:
                target.pop(key, None)
        target[no_proxy_marker] = "1"
    return target


def build_child_environment(
    base_env: Mapping[str, str] | None = None,
    *,
    no_proxy: bool = False,
    no_proxy_marker: str | None = None,
) -> dict[str, str]:
    """Build a deterministic UTF-8 environment for refresh subprocesses."""
    env = dict(os.environ if base_env is None else base_env)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    if no_proxy:
        strip_proxy_environment(env, no_proxy_marker=no_proxy_marker)
    return env


def format_command(command: Command, *, platform_name: str | None = None) -> str:
    if (platform_name or os.name) == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(list(command))


def run_steps(
    steps: Iterable[Step],
    *,
    cwd: Path,
    env: Mapping[str, str],
    runner: Runner | None = None,
) -> int:
    """Run commands without a shell and stop at the first failed step."""
    run = runner or subprocess.run
    for name, command_parts in steps:
        command = list(command_parts)
        print(f"\n==> {name}")
        print(f"$ {format_command(command)}")
        try:
            completed = run(
                command,
                cwd=str(cwd),
                env=dict(env),
                check=False,
                shell=False,
            )
        except OSError as exc:
            print(f"步骤启动失败: {name}: {exc}")
            return 127

        returncode = int(completed.returncode)
        if returncode != 0:
            print(f"步骤失败: {name} (exit={returncode})")
            return returncode
        print(f"步骤完成: {name}")
    return 0
