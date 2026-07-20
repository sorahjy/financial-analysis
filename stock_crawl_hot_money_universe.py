"""游资小盘 universe 选股 + 建池。

目标：产出一个【稳定的小盘题材/游资活跃股票池】，同时供 stock_hot_money_radar.py
和 stock_advanced_strategies.py 做截面打分。stock_crawl_hot_money.py 只为本池刷新
龙虎榜席位与技术面补充信号，不再另建候选池。

口径（用户确定）：
  · 游资活跃度主筛 = 近 1 年龙虎榜上榜 ≥ N 次（N 默认 5；龙虎榜高频 = 游资活跃 + 题材热门二合一）。
  · 标准 A 股前缀（沪 60 / 深 00 / 创业 30 / 科创 68）；排除 B 股(9*)、北交所、ST。
  · 爬 K 线（≥3.5 年，供 verify）后按【流通市值 ≤ 100 亿】过滤（流通市值=成交额/换手率反推，近 20 日中位）。
  · 选出标记 sw3_member.is_hot_money=1（复刻 is_leader 模式，radar 用 --pool hotmoney 取）。

MVP：先用「在 sw3_member 的票」（有 name/行业归属、能落 is_hot_money）；不在 sw3_member 的票
记到日志作为后补 TODO（需补行业归属 + sw3_segment 外键）。
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("STOCK_CRAWL_NO_PROXY", "1")

import stock_storage
from stock_crawl_common import load_json_file, strip_proxy_env, write_json_file
from stock_crawl_price_valuation import (
    plan_fundamentals_refresh,
    refresh_stock_histories,
)

strip_proxy_env()

VALID_PREFIXES = ("60", "00", "30", "68")
DEFAULT_MIN_LHB = 5
DEFAULT_MAX_CAP_YI = 100.0
DEFAULT_WORKERS = 6          # 别调高：>6 易把龙虎榜/K线接口跑挂(限流)
DEFAULT_YEARS = 4.0         # 覆盖 verify 的 LOOKBACK(90)+WINDOW(750)+前向(40) ≈ 880 交易日
MIN_BARS = 250              # 上市/数据不足(次新)门槛：有效日线 < 250 视为次新剔除
MAX_HISTORY_LAG_DAYS = 7    # 停牌容忍；超过则不进入需要即时交易的游资小盘池


def _failure_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:4000]


def _failure_detail(code, name, stage, error) -> Dict[str, Any]:
    return {
        "code": str(code).zfill(6) if code else None,
        "name": str(name or ""),
        "stage": str(stage or "hot_money_small_cap_universe"),
        "error": str(error or "unknown error")[:4000],
    }


def _refresh_failure_payload(
    refresh_result: Dict[str, Any],
    selected: List[str],
    names: Dict[str, str],
    *,
    extra_details: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    failed = [str(code).zfill(6) for code in refresh_result.get("failed", []) if code]
    details = []
    for item in refresh_result.get("failure_details", []) or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").zfill(6) if item.get("code") else None
        details.append(_failure_detail(
            code,
            item.get("name") or names.get(code or "", ""),
            item.get("stage") or "hot_money_long_factor_refresh",
            item.get("error") or "长线因子刷新失败",
        ))
    details.extend(extra_details or [])
    detailed_codes = {item.get("code") for item in details if item.get("code")}
    for code in failed:
        if code not in detailed_codes:
            details.append(_failure_detail(
                code,
                names.get(code, ""),
                "hot_money_long_factor_refresh",
                "长线因子数据刷新失败",
            ))
    for item in details:
        if item.get("code") and item["code"] not in failed:
            failed.append(item["code"])
    return {
        "requested": int(refresh_result.get("requested", len(selected)) or 0),
        "processed": int(refresh_result.get("processed", 0) or 0),
        "write_enqueued": int(refresh_result.get("write_enqueued", 0) or 0),
        "write_completed": int(refresh_result.get("write_completed", 0) or 0),
        "failed": failed,
        "failure_details": details,
        "selected": len(selected),
    }


def incomplete_selected_fundamentals(codes: List[str]) -> Dict[str, List[str]]:
    """Final atomic-switch guard: every selected stock must have complete factor families."""
    from stock_crawl_fundamentals import fundamentals_missing_families

    conn = stock_storage.connect()
    try:
        states = stock_storage.fundamentals_state_map(conn, codes)
    finally:
        conn.close()
    incomplete = {}
    for code in codes:
        missing = fundamentals_missing_families(states.get(code) or {})
        if missing:
            incomplete[code] = missing
    return incomplete


def select_seed(conn, min_lhb: int, since: str) -> Dict[str, List[str]]:
    """近 since 起龙虎榜上榜 ≥ min_lhb 次的 code；拆成 在册(in sw3_member) / 不在册(TODO)。"""
    rows = conn.execute(
        "SELECT code FROM lhb_all WHERE date >= ? GROUP BY code HAVING COUNT(*) >= ?",
        (since, min_lhb),
    ).fetchall()
    codes = [str(r["code"]).zfill(6) for r in rows]
    codes = [c for c in codes if c.startswith(VALID_PREFIXES)]
    members = {str(r["code"]).zfill(6): (r["name"] or "")
               for r in conn.execute("SELECT code, name FROM sw3_member").fetchall()}
    in_member, not_member = [], []
    for c in codes:
        if c in members:
            if "ST" in members[c].upper():
                continue
            in_member.append(c)
        else:
            not_member.append(c)
    return {"in_member": in_member, "not_member": not_member, "names": members}


def _bars_count(conn, code: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM stock_history WHERE code=? AND daily_close IS NOT NULL", (code,)
    ).fetchone()[0]


def latest_history_dates(conn, codes: List[str]) -> Dict[str, str]:
    """批量读取每只候选的实际最新K线日。"""
    out: Dict[str, str] = {}
    for start in range(0, len(codes), 500):
        chunk = codes[start:start + 500]
        if not chunk:
            continue
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT code, MAX(date) AS latest_date FROM stock_history "
            f"WHERE code IN ({placeholders}) GROUP BY code",
            chunk,
        ).fetchall()
        out.update({
            str(row["code"]).zfill(6): str(row["latest_date"])
            for row in rows if row["latest_date"]
        })
    return out


def history_is_fresh(latest_date: Optional[str], reference_date: Optional[str],
                     max_lag_days: int = MAX_HISTORY_LAG_DAYS) -> bool:
    if not latest_date or not reference_date:
        return False
    try:
        lag = datetime.strptime(reference_date, "%Y-%m-%d") - datetime.strptime(latest_date, "%Y-%m-%d")
    except ValueError:
        return False
    return lag.days <= max_lag_days


def refresh_seed_histories(codes: List[str], names: Dict[str, str], *, years: float, workers: int):
    """初筛候选统一复用龙头历史更新器；最终市值过滤必须在它完成后执行。"""
    return refresh_stock_histories(
        {code: names.get(code, "") for code in codes},
        max_years=years,
        workers=workers,
        refresh_valuation=False,
        label="游资初筛",
    )


def refresh_selected_factor_data(
        codes: List[str], names: Dict[str, str], *, years: float, workers: int):
    """补齐最终游资小盘池的估值、财报、指标、分红和质押数据。"""
    stocks = {code: names.get(code, "") for code in codes}
    fundamentals_plan, pledge_map = plan_fundamentals_refresh(set(stocks))
    return refresh_stock_histories(
        stocks,
        max_years=years,
        workers=workers,
        refresh_valuation=True,
        fundamentals_plan=fundamentals_plan,
        pledge_map=pledge_map,
        label="游资小盘长线因子",
    )


def float_cap_yi(conn, code: str) -> Optional[float]:
    """流通市值(亿) = 成交额 / (换手率/100)，取近 20 个有效日中位数(抗一字板/停牌噪声)。"""
    rows = conn.execute(
        "SELECT daily_amount, daily_turnover_rate FROM stock_history "
        "WHERE code=? AND daily_amount>0 AND daily_turnover_rate>0 "
        "ORDER BY date DESC LIMIT 20", (code,)
    ).fetchall()
    caps = [r["daily_amount"] / (r["daily_turnover_rate"] / 100.0) / 1e8
            for r in rows if r["daily_amount"] and r["daily_turnover_rate"]]
    return statistics.median(caps) if caps else None


def main(argv=None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description="游资小盘 universe 选股建池")
    ap.add_argument("--min-lhb", type=int, default=DEFAULT_MIN_LHB, help="近1年龙虎榜最少上榜次数")
    ap.add_argument("--max-cap", type=float, default=DEFAULT_MAX_CAP_YI, help="流通市值上限(亿)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="爬取线程数(别>6)")
    ap.add_argument("--years", type=float, default=DEFAULT_YEARS, help="爬取历史年数")
    ap.add_argument("--since", default=None, help="龙虎榜起始日(默认今天往前365天)")
    ap.add_argument(
        "--enrich-long-factors",
        action="store_true",
        help="建池后补齐最终成员的估值与基本面；生产 full 刷新启用",
    )
    ap.add_argument(
        "--failure-report",
        type=Path,
        help="把致命的逐股失败代码、阶段和异常写入该 JSON sidecar",
    )
    ap.add_argument("--dry-run", action="store_true", help="只选股不爬K线不标记")
    args = ap.parse_args(argv)

    since = args.since or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    conn = stock_storage.connect()
    seed = select_seed(conn, args.min_lhb, since)
    in_member, not_member, names = seed["in_member"], seed["not_member"], seed["names"]
    print(f"近1年(>= {since})龙虎榜≥{args.min_lhb}次 ∩ 标准前缀 ∩ 非ST：")
    print(f"  在 sw3_member(可建池): {len(in_member)}  ·  不在 sw3_member(TODO后补): {len(not_member)}")

    if args.dry_run:
        latest_map = latest_history_dates(conn, in_member)
        reference = conn.execute("SELECT MAX(date) FROM stock_history").fetchone()[0]
        need = [
            c for c in in_member
            if _bars_count(conn, c) < MIN_BARS or not history_is_fresh(latest_map.get(c), reference)
        ]
        print(f"  [dry-run] 需补/刷新 K 线: {len(need)}  ·  已足量且新鲜: {len(in_member)-len(need)}")
        if not_member[:10]:
            print(f"  [dry-run] 不在册样本: {' '.join(not_member[:10])}")
        conn.close()
        result = {
            "requested": len(in_member),
            "processed": len(in_member),
            "write_enqueued": 0,
            "write_completed": 0,
            "failed": [],
            "failure_details": [],
            "selected": None,
        }
        if args.failure_report:
            write_json_file(args.failure_report, result)
        return result

    # 1) 加 is_hot_money 列
    stock_storage._ensure_table_columns(conn, "sw3_member", {"is_hot_money": "INTEGER NOT NULL DEFAULT 0"})
    conn.commit()

    # 2) 初筛完成后，复用龙头池的统一增量更新器。必须先刷新再算流通市值，避免旧行情误筛。
    refresh_result = refresh_seed_histories(
        in_member, names, years=args.years, workers=args.workers
    )
    if refresh_result["failed"]:
        print(f"  [WARN] 历史刷新失败 {len(refresh_result['failed'])} 只；新鲜度门槛会阻止旧数据入池")

    latest_map = latest_history_dates(conn, in_member)
    reference_date = conn.execute("SELECT MAX(date) FROM stock_history").fetchone()[0]

    # 3) 流通市值过滤 + 次新过滤。先只计算新池，因子数据补齐成功后再原子换池。
    selected, skip_cap, skip_new, skip_nocap, skip_stale = [], 0, 0, 0, 0
    for c in in_member:
        bars = _bars_count(conn, c)
        if bars < MIN_BARS:
            skip_new += 1
            continue
        if not history_is_fresh(latest_map.get(c), reference_date):
            skip_stale += 1
            continue
        cap = float_cap_yi(conn, c)
        if cap is None:
            skip_nocap += 1
            continue
        if cap > args.max_cap:
            skip_cap += 1
            continue
        selected.append(c)
    conn.close()

    if not selected:
        detail = _failure_detail(
            None,
            "",
            "hot_money_pool_selection",
            "RuntimeError: 游资小盘筛选结果为空，保留上一份有效股票池",
        )
        result = {
            "requested": len(in_member),
            "processed": len(in_member),
            "write_enqueued": int(refresh_result.get("write_enqueued", 0) or 0),
            "write_completed": int(refresh_result.get("write_completed", 0) or 0),
            "failed": [],
            "failure_details": [detail],
            "selected": 0,
        }
        if args.failure_report:
            write_json_file(args.failure_report, result)
        raise RuntimeError("游资小盘筛选结果为空，保留上一份有效股票池")

    # 4) 生产 full 刷新会启用因子补齐。任一股票失败时不切换池标记，让上层
    # 保留旧策略结果，避免新池配旧数据；独立建池/quick 模式维持轻量行为。
    factor_result = {
        "requested": len(selected),
        "processed": len(selected),
        "write_enqueued": 0,
        "write_completed": 0,
        "failed": [],
        "failure_details": [],
    }
    if args.enrich_long_factors:
        try:
            factor_result = refresh_selected_factor_data(
                selected, names, years=args.years, workers=args.workers
            )
        except Exception as exc:
            detail = _failure_detail(
                None,
                "",
                "hot_money_long_factor_refresh",
                _failure_error(exc),
            )
            result = _refresh_failure_payload(
                {}, selected, names, extra_details=[detail]
            )
            if args.failure_report:
                write_json_file(args.failure_report, result)
            raise
        if factor_result.get("failed") or factor_result.get("failure_details"):
            result = _refresh_failure_payload(factor_result, selected, names)
            if args.failure_report:
                write_json_file(args.failure_report, result)
            samples = ", ".join(str(code) for code in result["failed"][:8])
            raise RuntimeError(
                f"游资小盘长线因子数据刷新失败 {len(result['failed'])} 只"
                f"（{samples}），保留上一份有效股票池"
            )
        incomplete = incomplete_selected_fundamentals(selected)
        if incomplete:
            details = [
                _failure_detail(
                    code,
                    names.get(code, ""),
                    "fundamentals_validation",
                    f"基本面数据不完整: {', '.join(missing)}",
                )
                for code, missing in incomplete.items()
            ]
            result = _refresh_failure_payload(
                factor_result, selected, names, extra_details=details
            )
            if args.failure_report:
                write_json_file(args.failure_report, result)
            samples = ", ".join(str(code) for code in result["failed"][:8])
            raise RuntimeError(
                f"游资小盘长线因子完整性复核失败 {len(incomplete)} 只"
                f"（{samples}），保留上一份有效股票池"
            )

    conn = stock_storage.connect()
    try:
        conn.execute("UPDATE sw3_member SET is_hot_money = 0 WHERE is_hot_money = 1")
        conn.executemany(
            "UPDATE sw3_member SET is_hot_money = 1 WHERE code = ?",
            [(c,) for c in selected],
        )
        conn.commit()
    finally:
        conn.close()
    print(f"\n=== 建池完成 ===")
    print(f"  入池(is_hot_money=1): {len(selected)}")
    print(
        f"  剔除: 流通市值>{args.max_cap}亿 {skip_cap} · 次新(<{MIN_BARS}日) {skip_new} "
        f"· K线落后>{MAX_HISTORY_LAG_DAYS}天 {skip_stale} · 无市值数据 {skip_nocap}"
    )
    print(f"  不在 sw3_member 未处理(TODO): {len(not_member)}")
    result = _refresh_failure_payload(factor_result, selected, names)
    if args.failure_report:
        write_json_file(args.failure_report, result)
    return result


def _failure_report_from_argv(argv) -> Optional[Path]:
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        index = values.index("--failure-report")
        return Path(values[index + 1])
    except (ValueError, IndexError):
        return None


def cli(argv=None) -> int:
    failure_report = _failure_report_from_argv(argv)
    if failure_report:
        write_json_file(failure_report, {"failure_details": []})
    try:
        main(argv)
        return 0
    except Exception as exc:
        payload = load_json_file(failure_report, None) if failure_report else None
        details = payload.get("failure_details") if isinstance(payload, dict) else None
        if failure_report and not details:
            detail = _failure_detail(
                None,
                "",
                "hot_money_small_cap_universe",
                _failure_error(exc),
            )
            write_json_file(failure_report, {
                "requested": 0,
                "processed": 0,
                "write_enqueued": 0,
                "write_completed": 0,
                "failed": [],
                "failure_details": [detail],
            })
        print(f"[ERROR] hot_money_small_cap_universe: {_failure_error(exc)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
