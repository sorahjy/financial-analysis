from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

from refresh_workflow import Runner, Step, build_child_environment, run_steps


ROOT = Path(__file__).resolve().parent


def build_refresh_steps(python_executable: str | None = None) -> list[Step]:
    python = python_executable or sys.executable
    return [
        ("校验并刷新 ETF 配置池行情", [python, "-B", "stock_crawl_etf_pool.py"]),
        (
            "刷新股票全量数据",
            [python, "-B", "stock_data_refresh.py", "--mode", "full", "--no-proxy"],
        ),
        (
            "刷新申万二级行业历史",
            [python, "-B", "plate_crawl_history.py", "--no-proxy"],
        ),
        ("重建题材候选", [python, "-B", "stock_theme_candidates.py"]),
        (
            "重建默认细分龙头雷达",
            [
                python,
                "-B",
                "stock_hot_money_radar.py",
                "ambush",
                "--no-exclude-large-cap",
                "--pool",
                "leader",
            ],
        ),
    ]


def refresh_stock_data(
    *,
    python_executable: str | None = None,
    runner: Runner | None = None,
    base_env: Mapping[str, str] | None = None,
) -> int:
    env = build_child_environment(
        base_env,
        no_proxy=True,
        no_proxy_marker="STOCK_CRAWL_NO_PROXY",
    )
    print("已绕过代理直连境内数据源")
    return run_steps(
        build_refresh_steps(python_executable),
        cwd=ROOT,
        env=env,
        runner=runner,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跨平台刷新股票、ETF、板块和雷达数据")
    parser.parse_args(argv)
    try:
        return refresh_stock_data()
    except Exception as exc:
        print(f"股票刷新失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
