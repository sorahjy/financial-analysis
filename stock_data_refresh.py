"""
Data refresh orchestration for the local stock strategy server.

The crawlers in this repository already know how to update incrementally. This
module only wires them into one startup preflight so the dashboard never serves
stale strategy output by accident.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from stock_crawl_common import (
    daily_payload_from_history_records,
    history_payload_from_records,
    load_json_file,
    prune_snapshot_only_history_records,
    write_json_file,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CAPITAL_DIR = DATA_DIR / "capital"
HOT_MONEY_CANDIDATES_FILE = CAPITAL_DIR / "hot_money_candidates.json"
REFRESH_REPORT_FILE = DATA_DIR / "stock_data_refresh_report.json"


@dataclass
class StepResult:
    name: str
    command: str
    ok: bool
    returncode: int
    elapsed_sec: float
    skipped: bool = False
    error: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "ok": self.ok,
            "returncode": self.returncode,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "skipped": self.skipped,
            "error": self.error,
            "meta": self.meta,
        }


def resolve_python(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    env_python = os.getenv("STOCK_REFRESH_PYTHON")
    if env_python:
        return env_python
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def command_text(cmd: Iterable[str]) -> str:
    return " ".join(str(part) for part in cmd)


def env_text(names, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def env_int(names, default: int) -> int:
    if isinstance(names, str):
        names = (names,)
    try:
        return int(env_text(names, str(default)) or str(default))
    except ValueError:
        return default


def run_step(
    name: str,
    cmd: List[str],
    *,
    timeout: Optional[int],
    env: Optional[Dict[str, str]] = None,
    skip: bool = False,
) -> StepResult:
    start = time.time()
    text = command_text(cmd)
    if skip:
        print(f"[refresh] skip {name}: {text}")
        return StepResult(name, text, True, 0, 0.0, skipped=True)

    print(f"[refresh] start {name}: {text}", flush=True)
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            timeout=timeout,
            text=True,
        )
        elapsed = time.time() - start
        ok = completed.returncode == 0
        state = "ok" if ok else f"failed({completed.returncode})"
        print(f"[refresh] {state} {name} in {elapsed:.1f}s", flush=True)
        return StepResult(name, text, ok, completed.returncode, elapsed)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - start
        print(f"[refresh] timeout {name} after {elapsed:.1f}s", flush=True)
        return StepResult(name, text, False, 124, elapsed, error=str(exc))
    except OSError as exc:
        elapsed = time.time() - start
        print(f"[refresh] error {name}: {exc}", flush=True)
        return StepResult(name, text, False, 127, elapsed, error=str(exc))


def _history_payload(code: str, name: str, records: List[Dict[str, Any]], source: str) -> Dict[str, Any]:
    return history_payload_from_records(code, name, records, source)


def fill_stock_history_from_local_data() -> Dict[str, Any]:
    """Normalize stock_data.history and remove snapshot-only daily points.

    The old fallback used daily.recent_daily and market_snapshot to synthesize
    history rows. New stock_data.daily intentionally stores only stats, so this
    step must not create close-only rows that later poison OHLCV checks.
    """
    stock_files = sorted(glob.glob(str(DATA_DIR / "stock_data" / "CN_*.json")))
    updated = 0
    skipped = 0
    removed_records = 0

    for path_text in stock_files:
        stock_path = Path(path_text)
        stock = load_json_file(stock_path, {})
        if not isinstance(stock, dict):
            skipped += 1
            continue
        code = str(stock.get("symbol") or stock_path.name.split("_")[1]).zfill(6)
        name = str(stock.get("name") or code)
        existing = (stock.get("history") or {}).get("records", [])
        cleaned, removed = prune_snapshot_only_history_records(existing)
        if not removed:
            skipped += 1
            continue
        stock["history"] = _history_payload(code, name, cleaned, "stock_data.history")
        stock["daily"] = daily_payload_from_history_records(cleaned)
        stock.pop("history_fallback_at", None)
        stock["history_sanitized_at"] = datetime.now().isoformat()
        write_json_file(stock_path, stock)
        updated += 1
        removed_records += removed

    stock_history_files = 0
    for fp in (DATA_DIR / "stock_data").glob("CN_*.json"):
        payload = load_json_file(fp, {})
        if isinstance(payload, dict) and (payload.get("history") or {}).get("records"):
            stock_history_files += 1

    return {
        "updated": updated,
        "skipped": skipped,
        "removed_snapshot_only_records": removed_records,
        "source_files": len(stock_files),
        "stock_history_files": stock_history_files,
    }


def local_step_result(name: str, command: str, func) -> StepResult:
    start = time.time()
    print(f"[refresh] start {name}: {command}", flush=True)
    try:
        meta = func()
        elapsed = time.time() - start
        print(f"[refresh] ok {name} in {elapsed:.1f}s: {meta}", flush=True)
        result = StepResult(name, command, True, 0, elapsed)
        result.meta = meta if isinstance(meta, dict) else {"result": meta}
        return result
    except Exception as exc:
        elapsed = time.time() - start
        print(f"[refresh] error {name}: {exc}", flush=True)
        return StepResult(name, command, False, 1, elapsed, error=str(exc))


def mirror_capital_outputs() -> Dict[str, Any]:
    candidates = load_json_file(HOT_MONEY_CANDIDATES_FILE, {})
    stocks = candidates.get("stocks", []) if isinstance(candidates, dict) else []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today = datetime.now().strftime("%Y-%m-%d")

    snapshot_path = CAPITAL_DIR / "snapshots" / f"hot_money_candidates_{today}.json"
    if HOT_MONEY_CANDIDATES_FILE.exists():
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HOT_MONEY_CANDIDATES_FILE, snapshot_path)

    picks = []
    for item in stocks:
        picks.append(
            {
                "code": str(item.get("code", "")).zfill(6),
                "name": item.get("name", ""),
                "followers": item.get("followers", []),
                "concurrent_count": item.get("concurrent_count"),
                "total_buyers": item.get("total_buyers"),
                "buy_amount_total": item.get("buy_amount_total"),
                "weighted_score": item.get("weighted_score"),
                "best_window": item.get("best_window"),
                "known_seat_count": item.get("known_seat_count"),
                "signals": item.get("signals"),
            }
        )
    write_json_file(
        DATA_DIR / "main_capital_picks.json",
        {
            "generated_at": now,
            "source": "data/capital/hot_money_candidates.json",
            "count": len(picks),
            "picks": picks,
        },
    )
    return {
        "capital_candidate_count": len(stocks),
        "capital_snapshot": str(snapshot_path.relative_to(ROOT)),
        "main_capital_picks_count": len(picks),
    }


def refresh_csi300_benchmark() -> Dict[str, Any]:
    from stock_crawl_price_valuation import (
        CSI300_ETF_CODE,
        CSI300_ETF_YEARS,
        CSI300_FILE,
        fetch_csi300_etf,
    )

    records = fetch_csi300_etf(years=CSI300_ETF_YEARS)
    path = Path(CSI300_FILE)
    try:
        display_path = str(path.resolve().relative_to(ROOT))
    except ValueError:
        display_path = str(path)
    return {
        "code": CSI300_ETF_CODE,
        "target_years": CSI300_ETF_YEARS,
        "records": len(records),
        "start_date": records[0]["date"] if records else None,
        "end_date": records[-1]["date"] if records else None,
        "file": display_path,
    }


def collect_data_health() -> Dict[str, Any]:
    stock_data_count = len(list((DATA_DIR / "stock_data").glob("CN_*.json")))
    stock_history_count = 0
    for fp in (DATA_DIR / "stock_data").glob("CN_*.json"):
        stock = load_json_file(fp, {})
        if isinstance(stock, dict) and (stock.get("history") or {}).get("records"):
            stock_history_count += 1
    capital = load_json_file(HOT_MONEY_CANDIDATES_FILE, {})
    strategy = load_json_file(DATA_DIR / "stock_advanced_strategy_results.json", {})
    csi300 = load_json_file(DATA_DIR / "csi300_etf_nav.json", {})
    csi300_records = csi300.get("records", []) if isinstance(csi300, dict) else []
    candidate_cache = load_json_file(DATA_DIR / "stock_strategy_candidate_cache.json", {})
    long_cache = candidate_cache.get("long") if isinstance(candidate_cache, dict) else {}
    short_cache = candidate_cache.get("short") if isinstance(candidate_cache, dict) else {}
    return {
        "stock_data_files": stock_data_count,
        "stock_history_files": stock_history_count,
        "capital_candidate_count": capital.get("count") if isinstance(capital, dict) else None,
        "capital_generated_at": capital.get("generated_at") if isinstance(capital, dict) else None,
        "strategy_generated_at": strategy.get("generated_at") if isinstance(strategy, dict) else None,
        "strategy_candidate_cache_version": candidate_cache.get("version") if isinstance(candidate_cache, dict) else None,
        "strategy_long_cache_generated_at": long_cache.get("generated_at") if isinstance(long_cache, dict) else None,
        "strategy_long_cache_candidates": long_cache.get("candidate_count") if isinstance(long_cache, dict) else None,
        "strategy_short_cache_generated_at": short_cache.get("generated_at") if isinstance(short_cache, dict) else None,
        "strategy_short_cache_candidates": short_cache.get("candidate_count") if isinstance(short_cache, dict) else None,
        "csi300_benchmark_records": len(csi300_records),
        "csi300_benchmark_start": csi300.get("start_date") if isinstance(csi300, dict) else None,
        "csi300_benchmark_end": csi300.get("end_date") if isinstance(csi300, dict) else None,
    }


def refresh_before_server(
    *,
    mode: str = "full",
    strict: bool = False,
    timeout: Optional[int] = None,
    python: Optional[str] = None,
    index_workers: int = 40,
    index_limit: int = 0,
    capital_days: int = 14,
    capital_top_yyb: int = 30,
    capital_min_followers: int = 1,
    capital_score_top: int = 100,
    no_proxy: bool = False,
    daily_process_workers: int = 32,
    daily_process_sources: str = "腾讯,新浪",
) -> Dict[str, Any]:
    python_bin = resolve_python(python)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TQDM_DISABLE", "1")
    env.setdefault("STOCK_THREAD_COUNT", "16")
    env["STOCK_DAILY_PROCESS_WORKERS"] = str(max(daily_process_workers, 0))
    env["STOCK_DAILY_PROCESS_SOURCES"] = daily_process_sources
    env["STOCK_DAILY_FALLBACK_PROCESS_WORKERS"] = env["STOCK_DAILY_PROCESS_WORKERS"]
    env["STOCK_DAILY_FALLBACK_PROCESS_SOURCES"] = env["STOCK_DAILY_PROCESS_SOURCES"]
    if no_proxy:
        # 数据源均为境内接口，绕过本地代理直连；NO_PROXY=* 同时屏蔽系统代理
        for var in ("http_proxy", "https_proxy", "all_proxy",
                    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            env.pop(var, None)
        env["NO_PROXY"] = "*"
        env["no_proxy"] = "*"
        env["STOCK_CRAWL_NO_PROXY"] = "1"

    full = mode == "full"
    if mode not in {"full", "quick", "capital-only"}:
        raise ValueError(f"unsupported refresh mode: {mode}")

    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    steps: List[StepResult] = []

    index_cmd = [
        python_bin,
        "-B",
        "stock_crawl_fundamentals.py",
        "--mode",
        "staged",
        "--workers",
        str(index_workers),
        "--limit",
        str(index_limit if full else max(index_limit, 80)),
    ]
    steps.append(
        run_step(
            "index_candidate_data",
            index_cmd,
            timeout=timeout,
            env=env,
            skip=mode == "capital-only",
        )
    )

    steps.append(
        run_step(
            "csi800_history",
            [python_bin, "-B", "stock_crawl_price_valuation.py"],
            timeout=timeout,
            env=env,
            skip=not full,
        )
    )
    if mode != "capital-only":
        steps.append(
            local_step_result(
                "csi300_benchmark",
                "fetch 510310 CSI300 ETF accumulated NAV for 12 years",
                refresh_csi300_benchmark,
            )
        )
    if full:
        steps.append(
            local_step_result(
                "stock_history_fallback",
                "normalize stock_data.history and remove snapshot-only points",
                fill_stock_history_from_local_data,
            )
        )

    steps.append(
        run_step(
            "dragon_tiger_capital",
            [
                python_bin,
                "-B",
                "stock_crawl_hot_money.py",
                "--days",
                str(capital_days),
                "--top-yyb",
                str(capital_top_yyb),
                "--min-followers",
                str(capital_min_followers),
                "--score-top",
                str(capital_score_top),
            ],
            timeout=timeout,
            env=env,
        )
    )

    mirror_step = local_step_result(
        "mirror_capital_outputs",
        "mirror data/capital/hot_money_candidates.json -> data/main_capital_picks.json",
        mirror_capital_outputs,
    )
    steps.append(mirror_step)
    mirror_meta = mirror_step.meta

    steps.append(
        run_step(
            "strategy_results",
            [python_bin, "-B", "stock_advanced_strategies.py", "--persist", "--rebuild-cache"],
            timeout=timeout,
            env=env,
        )
    )

    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    health = collect_data_health()
    ok = all(step.ok for step in steps)
    report = {
        "started_at": started_at,
        "finished_at": finished_at,
        "mode": mode,
        "python": python_bin,
        "ok": ok,
        "steps": [step.to_dict() for step in steps],
        "mirror": mirror_meta,
        "health": health,
    }
    write_json_file(REFRESH_REPORT_FILE, report)

    if strict and not ok:
        failed = [step.name for step in steps if not step.ok]
        raise RuntimeError(f"data refresh failed before server startup: {', '.join(failed)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh local stock data before serving the dashboard")
    parser.add_argument("--mode", choices=["full", "quick", "capital-only"], default="full")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800, help="per-step timeout seconds, 0=disabled")
    parser.add_argument("--python", default=None, help="python executable used for crawler subprocesses")
    parser.add_argument("--index-workers", type=int, default=40)
    parser.add_argument("--index-limit", type=int, default=0)
    parser.add_argument("--capital-days", type=int, default=14)
    parser.add_argument("--capital-top-yyb", type=int, default=30)
    parser.add_argument("--capital-min-followers", type=int, default=1)
    parser.add_argument("--capital-score-top", type=int, default=100)
    parser.add_argument("--daily-process-workers", type=int,
                        default=env_int(("STOCK_DAILY_PROCESS_WORKERS",
                                         "STOCK_DAILY_FALLBACK_PROCESS_WORKERS"), 32),
                        help="日线源进程池大小；<=1 表示关闭，默认 32")
    parser.add_argument("--daily-process-sources",
                        default=env_text(("STOCK_DAILY_PROCESS_SOURCES",
                                          "STOCK_DAILY_FALLBACK_PROCESS_SOURCES"), "腾讯,新浪"),
                        help="需要进程池隔离的日线源，逗号分隔，默认 腾讯,新浪")
    parser.add_argument("--fallback-process-workers", dest="daily_process_workers", type=int,
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--fallback-process-sources", dest="daily_process_sources",
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--no-proxy", action="store_true",
                        help="绕过系统代理直连境内数据源（东财/腾讯/新浪/百度）")
    args = parser.parse_args()

    report = refresh_before_server(
        mode=args.mode,
        strict=args.strict,
        timeout=args.timeout or None,
        python=args.python,
        index_workers=args.index_workers,
        index_limit=args.index_limit,
        capital_days=args.capital_days,
        capital_top_yyb=args.capital_top_yyb,
        capital_min_followers=args.capital_min_followers,
        capital_score_top=args.capital_score_top,
        no_proxy=args.no_proxy,
        daily_process_workers=args.daily_process_workers,
        daily_process_sources=args.daily_process_sources,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
