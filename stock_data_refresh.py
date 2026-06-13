
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
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CAPITAL_DIR = DATA_DIR / "capital"
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


def load_json(path: Path, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def safe_name_component(name: Any) -> str:
    text = str(name).strip()
    text = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", text)
    return text.strip(" ._") or "UNKNOWN"


def cn_stock_path(code: str, name: str) -> Path:
    return DATA_DIR / "CN_stock" / f"CN_{code}_{safe_name_component(name)}.json"


def find_cn_stock_file(code: str) -> Optional[Path]:
    matches = sorted((DATA_DIR / "CN_stock").glob(f"CN_{code}_*.json"))
    return matches[0] if matches else None


def merge_records(existing: List[Dict[str, Any]], new_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_date: Dict[str, Dict[str, Any]] = {}
    for row in existing:
        date = row.get("date")
        if date:
            by_date[str(date)] = dict(row)
    for row in new_records:
        date = row.get("date")
        if not date:
            continue
        merged = by_date.setdefault(str(date), {})
        merged.update({key: value for key, value in row.items() if value is not None or key not in merged})
    return sorted(by_date.values(), key=lambda item: str(item.get("date", "")))


def cn_records_from_stock_data(stock: Dict[str, Any], snapshot_row: Dict[str, Any], today: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in (stock.get("daily", {}) or {}).get("recent_daily", []) or []:
        if not isinstance(row, dict) or not row.get("date"):
            continue
        records.append(
            {
                "date": str(row.get("date"))[:10],
                "close": row.get("close"),
                "change_pct": row.get("change_pct"),
                "turnover_rate": row.get("turnover_rate"),
                "market_cap": None,
                "pe_ttm": None,
                "pe_static": None,
                "pb": None,
                "pcf": None,
            }
        )

    price = snapshot_row.get("price")
    if price is not None:
        market_cap_est = snapshot_row.get("market_cap_est")
        records.append(
            {
                "date": today,
                "close": price,
                "change_pct": None,
                "turnover_rate": None,
                "market_cap": market_cap_est / 100000000 if isinstance(market_cap_est, (int, float)) else None,
                "pe_ttm": None,
                "pe_static": None,
                "pb": None,
                "pcf": None,
            }
        )
    return records


def fill_cn_stock_from_local_data() -> Dict[str, Any]:
    stock_files = sorted(glob.glob(str(DATA_DIR / "stock_data" / "CN_*.json")))
    snapshot = load_json(DATA_DIR / "market_snapshot.json", {})
    today = datetime.now().strftime("%Y-%m-%d")
    created = 0
    updated = 0
    skipped = 0

    for path_text in stock_files:
        stock_path = Path(path_text)
        stock = load_json(stock_path, {})
        code = str(stock.get("symbol") or stock_path.name.split("_")[1]).zfill(6)
        name = str(stock.get("name") or (snapshot.get(code, {}) or {}).get("name") or code)
        current_path = find_cn_stock_file(code)
        current = load_json(current_path, {}) if current_path else {}
        existing = current.get("records", []) if isinstance(current, dict) else []
        new_records = cn_records_from_stock_data(stock, snapshot.get(code, {}) if isinstance(snapshot, dict) else {}, today)
        merged = merge_records(existing, new_records)
        if not merged:
            skipped += 1
            continue
        payload = {
            "symbol": code,
            "name": name,
            "start_date": merged[0]["date"],
            "end_date": merged[-1]["date"],
            "records": merged,
            "fallback_source": "stock_data+market_snapshot",
        }
        out_path = current_path or cn_stock_path(code, name)
        write_json(out_path, payload)
        if current_path:
            updated += 1
        else:
            created += 1

    return {
        "created": created,
        "updated": updated,
        "skipped": skipped,
        "source_files": len(stock_files),
        "cn_stock_files": len(list((DATA_DIR / "CN_stock").glob("CN_*.json"))),
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
    scored_path = CAPITAL_DIR / "scored_stocks.json"
    scored = load_json(scored_path, {})
    stocks = scored.get("stocks", []) if isinstance(scored, dict) else []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today = datetime.now().strftime("%Y-%m-%d")

    snapshot_path = CAPITAL_DIR / "snapshots" / f"scored_{today}.json"
    if scored_path.exists():
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scored_path, snapshot_path)

    picks = []
    for item in stocks:
        scores = item.get("scores", {}) if isinstance(item, dict) else {}
        picks.append(
            {
                "code": str(item.get("code", "")).zfill(6),
                "name": item.get("name", ""),
                "score": scores.get("total"),
                "scores": scores,
                "followers": item.get("followers", []),
                "signals": item.get("signals"),
            }
        )
    write_json(
        DATA_DIR / "main_capital_picks.json",
        {
            "generated_at": now,
            "source": "data/capital/scored_stocks.json",
            "count": len(picks),
            "picks": picks,
        },
    )
    return {
        "capital_scored_count": len(stocks),
        "capital_snapshot": str(snapshot_path.relative_to(ROOT)),
        "main_capital_picks_count": len(picks),
    }


def collect_data_health() -> Dict[str, Any]:
    stock_data_count = len(list((DATA_DIR / "stock_data").glob("CN_*.json")))
    cn_stock_count = len(list((DATA_DIR / "CN_stock").glob("CN_*.json")))
    capital = load_json(CAPITAL_DIR / "scored_stocks.json", {})
    strategy = load_json(DATA_DIR / "stock_advanced_strategy_results.json", {})
    return {
        "stock_data_files": stock_data_count,
        "cn_stock_files": cn_stock_count,
        "capital_scored_count": capital.get("count") if isinstance(capital, dict) else None,
        "capital_generated_at": capital.get("generated_at") if isinstance(capital, dict) else None,
        "strategy_generated_at": strategy.get("generated_at") if isinstance(strategy, dict) else None,
    }


def refresh_before_server(
    *,
    mode: str = "full",
    strict: bool = False,
    timeout: Optional[int] = None,
    python: Optional[str] = None,
    index_workers: int = 20,
    index_limit: int = 0,
    capital_days: int = 14,
    capital_top_yyb: int = 30,
    capital_min_followers: int = 1,
    capital_score_top: int = 100,
    no_proxy: bool = False,
) -> Dict[str, Any]:
    python_bin = resolve_python(python)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TQDM_DISABLE", "1")
    env.setdefault("STOCK_THREAD_COUNT", "6")
    if no_proxy:
        # 数据源均为境内接口，绕过本地代理直连；NO_PROXY=* 同时屏蔽系统代理
        for var in ("http_proxy", "https_proxy", "all_proxy",
                    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            env.pop(var, None)
        env["NO_PROXY"] = "*"
        env["no_proxy"] = "*"
        env["STOCK_CRAWL_NO_PROXY"] = "1"

    quick = mode == "quick"
    full = mode == "full"
    if mode not in {"full", "quick", "capital-only"}:
        raise ValueError(f"unsupported refresh mode: {mode}")

    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    steps: List[StepResult] = []

    index_cmd = [
        python_bin,
        "-B",
        "stock_crawl_index_all_stock_data.py",
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
            [python_bin, "-B", "stock_crawl_top_800_data.py"],
            timeout=timeout,
            env=env,
            skip=not full,
        )
    )
    if full:
        steps.append(
            local_step_result(
                "cn_stock_fallback",
                "local fallback from data/stock_data and data/market_snapshot",
                fill_cn_stock_from_local_data,
            )
        )

    steps.append(
        run_step(
            "dragon_tiger_capital",
            [
                python_bin,
                "-B",
                "stock_crawl_capital.py",
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
        "mirror data/capital/scored_stocks.json -> data/main_capital_picks.json",
        mirror_capital_outputs,
    )
    steps.append(mirror_step)
    mirror_meta = mirror_step.meta

    steps.append(
        run_step(
            "strategy_results",
            [python_bin, "-B", "stock_advanced_strategies.py", "--persist"],
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
    write_json(REFRESH_REPORT_FILE, report)

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
    parser.add_argument("--index-workers", type=int, default=20)
    parser.add_argument("--index-limit", type=int, default=0)
    parser.add_argument("--capital-days", type=int, default=14)
    parser.add_argument("--capital-top-yyb", type=int, default=30)
    parser.add_argument("--capital-min-followers", type=int, default=1)
    parser.add_argument("--capital-score-top", type=int, default=100)
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
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
