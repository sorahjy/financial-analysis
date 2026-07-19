"""
Data refresh orchestration for the local stock strategy server.

The crawlers in this repository already know how to update incrementally. This
module only wires them into one startup preflight so the dashboard never serves
stale strategy output by accident.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List, Optional

import stock_storage as ss
from app.services.job_service import terminate_process_tree
from refresh_workflow import (
    build_child_environment,
    is_no_proxy_enabled,
    strip_proxy_environment,
)
from stock_crawl_common import (
    daily_payload_from_history_records,
    history_payload_from_records,
    load_json_file,
    prune_snapshot_only_history_records,
    write_json_file,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REFRESH_REPORT_FILE = DATA_DIR / "stock_data_refresh_report.json"
UNIVERSE_CACHE_MIN_RATIO = 0.98
HOT_MONEY_MAX_HISTORY_LAG_DAYS = 7


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


def resolve_python(
    explicit: Optional[str] = None,
) -> str:
    if explicit:
        return explicit
    env_python = os.getenv("STOCK_REFRESH_PYTHON")
    if env_python:
        return env_python
    return sys.executable


def command_text(cmd: Iterable[str]) -> str:
    return " ".join(str(part) for part in cmd)


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
        popen_group_kwargs: Dict[str, Any] = {}
        # When the web job owns the timeout (the normal full-refresh path),
        # inherit its process group so an outer timeout reaches this child too.
        # A standalone per-step timeout still needs its own POSIX group because
        # terminate_process_tree() sends signals to process.pid's group.
        if os.name != "nt" and timeout is not None:
            popen_group_kwargs["start_new_session"] = True
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            text=True,
            **popen_group_kwargs,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            terminate_process_tree(process)
            elapsed = time.time() - start
            print(f"[refresh] timeout {name} after {elapsed:.1f}s", flush=True)
            return StepResult(name, text, False, 124, elapsed, error=str(exc))
        elapsed = time.time() - start
        ok = returncode == 0
        state = "ok" if ok else f"failed({returncode})"
        print(f"[refresh] {state} {name} in {elapsed:.1f}s", flush=True)
        return StepResult(name, text, ok, returncode, elapsed)
    except OSError as exc:
        elapsed = time.time() - start
        print(f"[refresh] error {name}: {exc}", flush=True)
        return StepResult(name, text, False, 127, elapsed, error=str(exc))


def _attach_failure_sidecar(step: StepResult, path: Path) -> None:
    """Attach structured child failures to ``StepResult.meta`` after it exits."""
    payload = load_json_file(path, None)
    if isinstance(payload, dict):
        for key in (
            "requested",
            "processed",
            "write_enqueued",
            "write_completed",
            "failed",
            "failure_details",
        ):
            if key in payload:
                step.meta[key] = payload[key]
        details = payload.get("failure_details")
        if isinstance(details, list) and details and not step.error:
            first = details[0] if isinstance(details[0], dict) else {}
            code = first.get("code") or "全局"
            stage = first.get("stage") or step.name
            error = first.get("error") or f"exit={step.returncode}"
            step.error = f"{code} [{stage}] {error}"
        elif not step.ok and not step.error:
            step.error = f"exit={step.returncode}; 子进程失败报告没有异常明细"
        return

    if not step.ok and not step.error:
        step.error = f"exit={step.returncode}; 子进程未生成逐股失败报告"


def _collect_report_failures(steps: Iterable[StepResult]) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for step in steps:
        details = step.meta.get("failure_details") if isinstance(step.meta, dict) else None
        if isinstance(details, list) and details:
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                failures.append({"step": step.name, **detail})
            continue
        if not step.ok and not step.skipped:
            failures.append({
                "step": step.name,
                "code": None,
                "name": "",
                "stage": step.name,
                "error": step.error or f"exit={step.returncode}",
            })
    return failures


def _collect_health_failures(health: Dict[str, Any]) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    reference = health.get("hot_money_history_reference_date")
    cutoff = health.get("hot_money_history_cutoff")
    details = health.get("hot_money_history_stale_details")
    if not isinstance(details, list):
        return failures
    for detail in details:
        if not isinstance(detail, dict):
            continue
        code = str(detail.get("code") or "").zfill(6)
        latest_date = detail.get("latest_date") or "无日线"
        failures.append({
            "step": "data_health",
            "code": code,
            "name": "",
            "stage": "hot_money_history_freshness",
            "error": (
                f"最新日线={latest_date}，早于健康门槛={cutoff}"
                f"（池内参考日={reference}）"
            ),
        })
    return failures


def _history_payload(code: str, name: str, records: List[Dict[str, Any]], source: str) -> Dict[str, Any]:
    return history_payload_from_records(code, name, records, source)


def fill_stock_history_from_local_data() -> Dict[str, Any]:
    """Normalize stock_history: drop snapshot-only / stub rows (now in SQLite).

    只挑库里实际含 snapshot/空行(daily_open 为空)的 code 来 prune，避免每次全库逐只 load 扫描
    (4900+ 只)；干净库直接返回 candidates=0。
    """
    conn = ss.connect()
    try:
        codes = ss.codes_needing_history_cleanup(conn)
        updated = 0
        skipped = 0
        removed_records = 0
        for code in codes:
            existing = ss.load_history_records(conn, code)
            cleaned, removed = prune_snapshot_only_history_records(existing)
            if not removed:
                skipped += 1
                continue
            stock = ss.load_stock(conn, code)
            name = str(stock.get("name") or code)
            stock["history"] = _history_payload(code, name, cleaned, "stock_data.history")
            stock["daily"] = daily_payload_from_history_records(cleaned)
            ss.save_stock(conn, stock)
            updated += 1
            removed_records += removed

        return {
            "candidates": len(codes),
            "updated": updated,
            "skipped": skipped,
            "removed_snapshot_only_records": removed_records,
            "source_files": ss.table_count(conn, "stock_meta"),
        }
    finally:
        conn.close()


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


def _relative_display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def refresh_benchmark_etfs() -> Dict[str, Any]:
    from stock_crawl_price_valuation import BENCHMARK_ETFS, fetch_benchmark_etfs

    results = fetch_benchmark_etfs()
    benchmarks = []
    for item in BENCHMARK_ETFS:
        records = results.get(item["code"], [])
        benchmarks.append({
            "code": item["code"],
            "label": item["label"],
            "target_start_date": item.get("start_date"),
            "records": len(records),
            "start_date": records[0]["date"] if records else None,
            "end_date": records[-1]["date"] if records else None,
            "file": _relative_display_path(Path(item["file"])),
        })
    return {
        "codes": [item["code"] for item in BENCHMARK_ETFS],
        "benchmarks": benchmarks,
    }


def _protect_index_universe(
    label: str,
    fetched_map: Dict[str, str],
    cached_codes: Any,
) -> tuple[Dict[str, str], bool]:
    if not isinstance(cached_codes, list) or not cached_codes:
        return fetched_map, False

    cached = [str(code).zfill(6) for code in cached_codes if str(code).strip()]
    if not cached:
        return fetched_map, False

    if len(fetched_map) >= len(cached) * UNIVERSE_CACHE_MIN_RATIO:
        return fetched_map, False

    print(
        f"  [WARN] {label} 成分本次仅 {len(fetched_map)} 个，旧缓存 {len(cached)} 个；"
        "保留旧缓存避免部分数据覆盖全量股票池",
        flush=True,
    )
    return {code: fetched_map.get(code, "") for code in cached}, True


def refresh_stock_universe() -> Dict[str, Any]:
    """刷新沪深300(000300)与中证全指(000985)成分到 data/stock_universe.json。

    供 stock_advanced_strategies / stock_strategy_optimizer 的 csi300_current /
    csi300_persistence 因子与 require_csi300 过滤使用。成分变动慢，但需随刷新定期更新——
    旧版只有手动跑 stock_crawl_fundamentals 才会写它，导致 csi300 一直停在旧快照。
    """
    from stock_crawl_common import fetch_index_constituents
    from stock_crawl_fundamentals import save_stock_universe

    existing = load_json_file(DATA_DIR / "stock_universe.json", {})
    csi300 = fetch_index_constituents("000300")
    csi_all = fetch_index_constituents("000985")

    used_cache = []
    if isinstance(existing, dict):
        csi300, cached = _protect_index_universe("沪深300(000300)", csi300, existing.get("csi300"))
        if cached:
            used_cache.append("csi300")
        csi_all, cached = _protect_index_universe("中证全指(000985)", csi_all, existing.get("all"))
        if cached:
            used_cache.append("all")

    source = "akshare index_stock_cons_csindex/index_stock_cons"
    if used_cache:
        source += " + cached stock_universe guard"
    save_stock_universe(csi300, csi_all, source=source)
    return {"csi300": len(csi300), "all": len(csi_all), "used_cache": used_cache}


def sync_sw3_market_caps() -> Dict[str, Any]:
    """整轮爬完后批量把各股 stock_history 最新总市值同步进 sw3_member.market_cap_yi。

    取代旧的「每次 save_stock 都查 stock_history 大表同步单只」热路开销；读端 pool_members
    仍对残留 NULL 做兜底，故本步骤纯属把雷达/龙头池要用的市值列一次性刷新到位。
    """
    conn = ss.connect()
    try:
        updated = ss.sync_sw3_member_market_caps(conn)
    finally:
        conn.close()
    return {"sw3_members_market_cap_synced": updated}


def collect_data_health() -> Dict[str, Any]:
    conn = ss.connect()
    try:
        stock_data_count = ss.table_count(conn, "stock_meta")
        stock_history_count = len(ss.codes_with_history(conn))
        benchmark_nav = {}
        for code in ("510310", "510580"):
            entry = ss.load_index_nav(conn, code)
            records = entry.get("records", []) if isinstance(entry, dict) else []
            benchmark_nav[code] = {
                "records": len(records),
                "start_date": entry.get("start_date") if isinstance(entry, dict) else None,
                "end_date": entry.get("end_date") if isinstance(entry, dict) else None,
            }
        hot_money_pool = ss.pool_signature(conn, "hotmoney")
        short_signals = ss.short_signal_status(conn)
        history_rows = conn.execute(
            "SELECT m.code, MAX(h.date) AS latest_date FROM sw3_member m "
            "LEFT JOIN stock_history h ON h.code=m.code "
            "WHERE m.is_hot_money=1 GROUP BY m.code ORDER BY m.code"
        ).fetchall()
        history_reference_date = max(
            (str(row["latest_date"]) for row in history_rows if row["latest_date"]),
            default=None,
        )
        history_cutoff = (
            datetime.strptime(history_reference_date, "%Y-%m-%d")
            - timedelta(days=HOT_MONEY_MAX_HISTORY_LAG_DAYS)
        ).strftime("%Y-%m-%d") if history_reference_date else None
        stale_history_codes = [
            str(row["code"]).zfill(6)
            for row in history_rows
            if not row["latest_date"] or not history_cutoff or str(row["latest_date"]) < history_cutoff
        ]
        stale_history_details = [
            {
                "code": str(row["code"]).zfill(6),
                "latest_date": str(row["latest_date"]) if row["latest_date"] else None,
            }
            for row in history_rows
            if not row["latest_date"] or not history_cutoff or str(row["latest_date"]) < history_cutoff
        ]
    finally:
        conn.close()
    strategy = load_json_file(DATA_DIR / "stock_advanced_strategy_results.json", {})
    candidate_cache = load_json_file(DATA_DIR / "stock_strategy_candidate_cache.json", {})
    long_cache = candidate_cache.get("long") if isinstance(candidate_cache, dict) else {}
    smallcap_cache = candidate_cache.get("smallcap") if isinstance(candidate_cache, dict) else {}
    short_cache = candidate_cache.get("short") if isinstance(candidate_cache, dict) else {}
    return {
        "stock_data_files": stock_data_count,
        "stock_history_files": stock_history_count,
        "hot_money_pool_count": hot_money_pool.get("count"),
        "hot_money_pool_digest": hot_money_pool.get("code_digest"),
        "hot_money_history_reference_date": history_reference_date,
        "hot_money_history_cutoff": history_cutoff,
        "hot_money_history_fresh_count": len(history_rows) - len(stale_history_codes),
        "hot_money_history_stale_count": len(stale_history_codes),
        "hot_money_history_stale_sample": stale_history_codes[:20],
        "hot_money_history_stale_details": stale_history_details,
        "short_signal_count": short_signals.get("count"),
        "short_signal_generated_at": short_signals.get("generated_at"),
        "short_signal_as_of_date": short_signals.get("as_of_date"),
        "strategy_generated_at": strategy.get("generated_at") if isinstance(strategy, dict) else None,
        "strategy_candidate_cache_version": candidate_cache.get("version") if isinstance(candidate_cache, dict) else None,
        "strategy_long_cache_generated_at": long_cache.get("generated_at") if isinstance(long_cache, dict) else None,
        "strategy_long_cache_candidates": long_cache.get("candidate_count") if isinstance(long_cache, dict) else None,
        "strategy_smallcap_cache_generated_at": smallcap_cache.get("generated_at") if isinstance(smallcap_cache, dict) else None,
        "strategy_smallcap_cache_candidates": smallcap_cache.get("candidate_count") if isinstance(smallcap_cache, dict) else None,
        "strategy_short_cache_generated_at": short_cache.get("generated_at") if isinstance(short_cache, dict) else None,
        "strategy_short_cache_candidates": short_cache.get("candidate_count") if isinstance(short_cache, dict) else None,
        "benchmark_nav": benchmark_nav,
        "csi300_benchmark_records": benchmark_nav["510310"]["records"],
        "csi300_benchmark_start": benchmark_nav["510310"]["start_date"],
        "csi300_benchmark_end": benchmark_nav["510310"]["end_date"],
        "csi500_benchmark_records": benchmark_nav["510580"]["records"],
        "csi500_benchmark_start": benchmark_nav["510580"]["start_date"],
        "csi500_benchmark_end": benchmark_nav["510580"]["end_date"],
    }


def refresh_before_server(
    *,
    mode: str = "full",
    timeout: Optional[int] = None,
    no_proxy: bool = False,
) -> Dict[str, Any]:
    python_bin = resolve_python()
    effective_no_proxy = no_proxy or is_no_proxy_enabled("STOCK_CRAWL_NO_PROXY")
    if effective_no_proxy:
        # Local refresh steps run inside this process, so clearing only the
        # child environment would still leave them routed through the proxy.
        strip_proxy_environment(
            os.environ,
            no_proxy_marker="STOCK_CRAWL_NO_PROXY",
        )
    env = build_child_environment(
        os.environ,
        no_proxy=effective_no_proxy,
        no_proxy_marker="STOCK_CRAWL_NO_PROXY",
    )
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TQDM_DISABLE", "1")
    # 并发按本机 CPU 核数自适应（旧硬编码 48 线程 + 32 进程在 ~10 核机器上是数倍过订阅，
    # 把 CPU 烧满/磁盘压垮而不会更快）。stock 线程偏网络+GIL 取 ~2× 核数；日线进程是 CPU 绑定取 ~核数。
    # 显式设对应环境变量仍优先生效（大机器可上调）。
    cpu = os.cpu_count() or 8
    env.setdefault("STOCK_THREAD_COUNT", str(min(32, cpu * 2)))
    # 日线源进程池：腾讯,新浪；用 STOCK_DAILY_PROCESS_WORKERS / _SOURCES 环境变量可覆盖
    env.setdefault("STOCK_DAILY_PROCESS_WORKERS", str(min(32, cpu)))
    env.setdefault("STOCK_DAILY_PROCESS_SOURCES", "腾讯,新浪")
    full = mode == "full"
    if mode not in {"full", "quick", "capital-only"}:
        raise ValueError(f"unsupported refresh mode: {mode}")

    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    steps: List[StepResult] = []

    with TemporaryDirectory(prefix="stock-segment-refresh-") as tmpdir:
        failure_sidecar = Path(tmpdir) / "failure-report.json"
        history_step = run_step(
            "segment_leader_history",
            [
                python_bin,
                "-B",
                "stock_crawl_price_valuation.py",
                "--failure-report",
                str(failure_sidecar),
            ],
            timeout=timeout,
            env=env,
            skip=not full,
        )
        if full:
            _attach_failure_sidecar(history_step, failure_sidecar)
    steps.append(history_step)
    steps.append(
        run_step(
            "segment_leader_fundamentals",
            [python_bin, "-B", "stock_crawl_fundamentals.py", "--mode", "full", "--segment-refresh-slice", "0"],
            timeout=timeout,
            env=env,
            skip=(not full) or (not history_step.ok),
        )
    )
    if mode != "capital-only":
        steps.append(
            local_step_result(
                "benchmark_etfs",
                "fetch 510310 CSI300 and 510580 CSI500 ETF accumulated NAV since 2012-01",
                refresh_benchmark_etfs,
            )
        )
        steps.append(
            local_step_result(
                "stock_universe_indices",
                "refresh 沪深300(000300) + 中证全指(000985) 成分 -> data/stock_universe.json",
                refresh_stock_universe,
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
        local_step_result(
            "sync_sw3_market_caps",
            "batch sync latest market cap from stock_history -> sw3_member.market_cap_yi",
            sync_sw3_market_caps,
        )
    )

    steps.append(
        run_step(
            "shareholder_count_history",
            [python_bin, "-B", "stock_crawl_holders.py", "--no-proxy"],
            timeout=timeout,
            env=env,
            skip=(mode == "quick"),
        )
    )
    steps.append(
        run_step(
            "long_capital_events",
            [python_bin, "-B", "stock_crawl_capital.py", "--no-proxy"],
            timeout=timeout,
            env=env,
            skip=(mode == "quick"),
        )
    )

    hot_money_universe_step = run_step(
        "hot_money_small_cap_universe",
        [
            python_bin,
            "-B",
            "stock_crawl_hot_money_universe.py",
            *(["--enrich-long-factors"] if full else []),
        ],
        timeout=timeout,
        env=env,
    )
    steps.append(hot_money_universe_step)

    steps.append(
        run_step(
            "short_signal_enrichment",
            [python_bin, "-B", "stock_crawl_hot_money.py",
             "--days", "14", "--top-yyb", "30"],
            timeout=timeout,
            env=env,
            skip=not hot_money_universe_step.ok,
        )
    )

    # 策略结果只能建立在本轮所有应执行的数据步骤都成功的前提下。StepResult 对主动
    # skip 的步骤保持 ok=True，因此 quick/capital-only 的模式跳过不会误阻断策略；真正
    # 失败的抓取或本地整理则会留下旧策略产物，而不是写出一份新旧数据混合的快照。
    strategy_dependencies_ok = all(step.ok for step in steps)
    steps.append(
        run_step(
            "strategy_results",
            [python_bin, "-B", "stock_advanced_strategies.py", "--persist", "--rebuild-cache"],
            timeout=timeout,
            env=env,
            skip=not strategy_dependencies_ok,
        )
    )

    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    health = collect_data_health()
    ok = all(step.ok for step in steps) and not health.get("hot_money_history_stale_count")
    failures = _collect_report_failures(steps)
    failures.extend(_collect_health_failures(health))
    report = {
        "started_at": started_at,
        "finished_at": finished_at,
        "mode": mode,
        "python": python_bin,
        "ok": ok,
        "steps": [step.to_dict() for step in steps],
        "failures": failures,
        "health": health,
    }
    write_json_file(REFRESH_REPORT_FILE, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh local stock data before serving the dashboard")
    parser.add_argument("--mode", choices=["full", "quick", "capital-only"], default="full")
    parser.add_argument("--timeout", type=int, default=0, help="per-step timeout seconds, 0=disabled")
    parser.add_argument("--no-proxy", action="store_true",
                        help="绕过系统代理直连境内数据源（东财/腾讯/新浪/百度）")
    args = parser.parse_args()

    report = refresh_before_server(
        mode=args.mode,
        timeout=args.timeout or None,
        no_proxy=(
            args.no_proxy
            or is_no_proxy_enabled("STOCK_CRAWL_NO_PROXY")
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report.get("ok"):
        # Print concise diagnostics after the large JSON payload so the web
        # job's retained log tail shows the actual stock/stage/error instead
        # of ending on unrelated health-report braces.
        failures = [
            detail for detail in report.get("failures", [])
            if isinstance(detail, dict)
        ]
        if failures:
            print(
                f"[refresh] 失败明细共 {len(failures)} 条：",
                file=sys.stderr,
                flush=True,
            )
        for detail in failures:
            code = detail.get("code") or "全局"
            name = str(detail.get("name") or "").strip()
            stage = detail.get("stage") or detail.get("step") or "stock_refresh"
            error = detail.get("error") or "unknown error"
            identity = f"{code} {name}" if name else str(code)
            print(
                f"[refresh failure] {identity} [{stage}] {error}",
                file=sys.stderr,
                flush=True,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
