from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


Command = Sequence[str]
Step = tuple[str, Command]
Runner = Callable[..., subprocess.CompletedProcess[object]]

_PROXY_KEYS = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}


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
        for key in list(env):
            if key.lower() in _PROXY_KEYS:
                env.pop(key, None)
        env["NO_PROXY"] = "*"
        env["no_proxy"] = "*"
        if no_proxy_marker:
            env[no_proxy_marker] = "1"
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
