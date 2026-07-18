from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Mapping

from refresh_workflow import Runner, Step, build_child_environment, run_steps


ROOT = Path(__file__).resolve().parent
SCRAPY_CACHE_DIR = ROOT / ".scrapy" / "httpcache" / "jijin"


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def build_refresh_steps(python_executable: str | None = None) -> list[Step]:
    python = python_executable or sys.executable
    return [
        ("抓取基金概况", [python, "-B", "-m", "scrapy", "crawl", "jijin"]),
        ("抓取基金净值与实时估算", [python, "-B", "fund_fetch_data.py"]),
        ("计算基金技术指标", [python, "-B", "fund_technical_analysis.py"]),
        ("生成基金报告数据", [python, "-B", "fund_generate_output.py"]),
    ]


def sync_validated_fund_codes() -> None:
    # Import lazily so command construction and unit tests have no Flask side effects.
    from app.routes.fund import _load_validated_fund_config, _sync_fund_codes

    _sync_fund_codes(_load_validated_fund_config())


def prepare_scrapy_cache(
    cache_dir: Path = SCRAPY_CACHE_DIR,
    *,
    today: date | None = None,
) -> bool:
    """Remove the Scrapy cache when its directory was not updated today."""
    if not cache_dir.is_dir():
        return False
    cache_date = datetime.fromtimestamp(cache_dir.stat().st_mtime).date()
    current_date = today or date.today()
    if cache_date == current_date:
        print(f"今日缓存有效，复用: {cache_dir}")
        return False
    print(f"缓存非今日，清除: {cache_dir}")
    shutil.rmtree(cache_dir)
    return True


def refresh_fund_data(
    *,
    no_proxy: bool = False,
    python_executable: str | None = None,
    runner: Runner | None = None,
    syncer: Callable[[], None] = sync_validated_fund_codes,
    cache_dir: Path = SCRAPY_CACHE_DIR,
    base_env: Mapping[str, str] | None = None,
) -> int:
    print("校验基金配置并同步基金代码")
    syncer()
    prepare_scrapy_cache(cache_dir)
    env = build_child_environment(
        base_env,
        no_proxy=no_proxy,
        no_proxy_marker="FUND_CRAWL_NO_PROXY",
    )
    if no_proxy:
        print("已绕过代理直连境内数据源")
    return run_steps(
        build_refresh_steps(python_executable),
        cwd=ROOT,
        env=env,
        runner=runner,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跨平台刷新基金数据并生成报告")
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="绕过系统代理直连境内数据源",
    )
    args = parser.parse_args(argv)
    try:
        return refresh_fund_data(
            no_proxy=args.no_proxy or _env_flag("FUND_CRAWL_NO_PROXY")
        )
    except Exception as exc:
        print(f"基金刷新失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
