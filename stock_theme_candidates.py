"""Map leader stocks to their closest trading themes from local data.

Candidate pool:
  - sw3_member.is_leader = 1 from stock_data.sqlite3.

Theme signal:
  - Full SW2 plate heat ranking from plate_data.sqlite3.
  - Stock/plate tracking relationship from return and turnover correlations.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text


DATA_DIR = Path("data")
CAPITAL_DIR = DATA_DIR / "capital"
STOCK_DB = DATA_DIR / "stock_data.sqlite3"
PLATE_DB = DATA_DIR / "plate_data.sqlite3"
OUT_JSON = CAPITAL_DIR / "theme_candidates.json"
LEGACY_OUTPUTS = (CAPITAL_DIR / "theme_candidates.csv", CAPITAL_DIR / "theme_rankings.csv")
PLATE_TYPE = "sw2"
DEFAULT_LOOKBACK_DAYS = 760


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, 6)


def pct_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average").fillna(0.0) * 100.0


def load_candidate_pool() -> pd.DataFrame:
    with sqlite3.connect(STOCK_DB) as conn:
        query = """
            WITH latest_cap AS (
                SELECT code, market_cap
                FROM (
                    SELECT code, market_cap,
                           ROW_NUMBER() OVER (PARTITION BY code ORDER BY date DESC) AS rn
                    FROM stock_history
                    WHERE market_cap IS NOT NULL
                )
                WHERE rn = 1
            )
            SELECT
                m.code,
                m.name,
                s.parent_segment AS sw2_industry,
                s.segment_name AS sw3_industry,
                s.segment_code AS sw3_segment_code,
                COALESCE(m.market_cap_yi, latest_cap.market_cap) AS market_cap_yi
            FROM sw3_member m
            LEFT JOIN sw3_segment s ON s.segment_code = m.segment_code
            LEFT JOIN latest_cap ON latest_cap.code = m.code
            WHERE m.is_leader = 1
            ORDER BY m.code
        """
        return pd.read_sql_query(query, conn)


def load_stock_history(codes: Iterable[str], start_date: str) -> pd.DataFrame:
    codes = [str(c).zfill(6) for c in codes]
    if not codes:
        return pd.DataFrame()
    placeholders = ",".join("?" for _ in codes)
    with sqlite3.connect(STOCK_DB) as conn:
        query = f"""
            SELECT
                code,
                date AS trade_date,
                daily_close,
                daily_amount,
                daily_turnover_rate
            FROM stock_history
            WHERE code IN ({placeholders})
              AND date >= ?
              AND daily_close IS NOT NULL
            ORDER BY code, date
        """
        df = pd.read_sql_query(query, conn, params=[*codes, start_date])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def load_plate_daily(start_date: str) -> pd.DataFrame:
    with sqlite3.connect(PLATE_DB) as conn:
        query = """
            SELECT
                plate_code,
                plate_name,
                trade_date,
                close_index,
                change_pct,
                turnover_rate,
                amount_share_pct
            FROM plate_daily
            WHERE plate_type = ?
              AND trade_date >= ?
              AND close_index IS NOT NULL
            ORDER BY plate_code, trade_date
        """
        df = pd.read_sql_query(query, conn, params=(PLATE_TYPE, start_date))
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def add_plate_metrics(plates: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for _, group in plates.groupby("plate_code"):
        g = group.sort_values("trade_date").copy()
        g["plate_ret_20d"] = g["close_index"].pct_change(20) * 100
        g["plate_ret_60d"] = g["close_index"].pct_change(60) * 100
        g["plate_turn_ma20"] = g["turnover_rate"].rolling(20, min_periods=5).mean()
        g["plate_turn_ma120"] = g["turnover_rate"].rolling(120, min_periods=20).mean()
        g["plate_amount_ma20"] = g["amount_share_pct"].rolling(20, min_periods=5).mean()
        frames.append(g)
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    latest_date = full["trade_date"].max()
    latest = full[full["trade_date"] == latest_date].copy()
    latest["ret20_rank"] = pct_rank(latest["plate_ret_20d"])
    latest["ret60_rank"] = pct_rank(latest["plate_ret_60d"])
    latest["turn_rank"] = pct_rank(latest["plate_turn_ma20"])
    latest["amount_rank"] = pct_rank(latest["plate_amount_ma20"])
    accel = latest["plate_turn_ma20"] / latest["plate_turn_ma120"].replace(0, pd.NA)
    latest["turn_accel"] = accel.astype("float64")
    latest["turn_accel_rank"] = pct_rank(accel.astype("float64"))
    latest["theme_hot_score"] = (
        latest["ret20_rank"] * 0.25
        + latest["ret60_rank"] * 0.20
        + latest["turn_rank"] * 0.25
        + latest["amount_rank"] * 0.20
        + latest["turn_accel_rank"] * 0.10
    )
    return full, latest.sort_values("theme_hot_score", ascending=False)


def add_stock_metrics(history: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in history.groupby("code"):
        g = group.sort_values("trade_date").copy()
        g["stock_ret_1d"] = g["daily_close"].pct_change() * 100
        g["stock_ret_60d"] = g["daily_close"].pct_change(60) * 100
        frames.append(g)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def tracking_matrices(plate_full: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "change": plate_full.pivot(index="trade_date", columns="plate_code", values="change_pct"),
        "turnover": plate_full.pivot(index="trade_date", columns="plate_code", values="turnover_rate"),
        "ret60": plate_full.pivot(index="trade_date", columns="plate_code", values="plate_ret_60d"),
    }


def best_tracking_theme(
    stock: pd.DataFrame,
    matrices: Dict[str, pd.DataFrame],
    plate_meta: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    g = stock.sort_values("trade_date").set_index("trade_date")
    if len(g) < 180:
        return None

    stock_ret = g["stock_ret_1d"]
    stock_turn = g["daily_turnover_rate"]
    stock_ret60 = g["stock_ret_60d"]
    plate_change = matrices["change"].reindex(g.index)
    plate_turn = matrices["turnover"].reindex(g.index)
    plate_ret60 = matrices["ret60"].reindex(g.index)

    matched_days = plate_change.notna().mul(stock_ret.notna(), axis=0).sum()
    return_corr = plate_change.corrwith(stock_ret)
    turnover_corr = plate_turn.corrwith(stock_turn)
    trend_corr_60d = plate_ret60.corrwith(stock_ret60)

    tracking_corr = (
        return_corr.fillna(0.0) * 0.55
        + trend_corr_60d.fillna(0.0) * 0.25
        + turnover_corr.fillna(0.0) * 0.20
    )
    tracking_corr = tracking_corr.where(matched_days >= 120).dropna()
    if tracking_corr.empty:
        return None

    plate_code = str(tracking_corr.idxmax())
    meta = plate_meta.get(plate_code, {})
    return {
        "tracking_theme_code": plate_code,
        "tracking_theme": meta.get("plate_name") or plate_code,
        "tracking_corr": safe_float(tracking_corr.get(plate_code)),
        "return_corr": safe_float(return_corr.get(plate_code)),
        "turnover_corr": safe_float(turnover_corr.get(plate_code)),
        "trend_corr_60d": safe_float(trend_corr_60d.get(plate_code)),
        "matched_days": int(matched_days.get(plate_code, 0)),
    }


def build_theme_candidates(
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Dict[str, Any]:
    pool = load_candidate_pool()
    if pool.empty:
        raise RuntimeError("sw3_member.is_leader 候选池为空")
    start_date = (datetime.now() - pd.Timedelta(days=int(lookback_days * 1.55))).strftime("%Y-%m-%d")
    stock_hist = add_stock_metrics(load_stock_history(pool["code"], start_date))
    plate_full, plate_latest = add_plate_metrics(load_plate_daily(start_date))
    matrices = tracking_matrices(plate_full)
    plate_meta = plate_latest.set_index("plate_code").to_dict("index")
    pool_meta = pool.set_index("code").to_dict("index")

    stock_themes: List[Dict[str, Any]] = []
    for code, group in stock_hist.groupby("code"):
        code = str(code).zfill(6)
        if code not in pool_meta:
            continue
        tracking = best_tracking_theme(group, matrices, plate_meta)
        if not tracking:
            continue
        meta = pool_meta[code]
        latest_stock = group.sort_values("trade_date").iloc[-1]
        stock_themes.append({
            "code": code,
            "name": meta.get("name"),
            "tracking_theme": tracking["tracking_theme"],
            "tracking_theme_code": tracking["tracking_theme_code"],
            "static_sw2": meta.get("sw2_industry"),
            "static_sw3": meta.get("sw3_industry"),
            "tracking_corr": tracking["tracking_corr"],
            "return_corr": tracking["return_corr"],
            "turnover_corr": tracking["turnover_corr"],
            "trend_corr_60d": tracking["trend_corr_60d"],
            "matched_days": tracking["matched_days"],
            "latest_date": latest_stock["trade_date"].strftime("%Y-%m-%d"),
        })

    theme_rankings = [
        {
            "rank": int(rank),
            "plate_code": row["plate_code"],
            "plate_name": row["plate_name"],
            "theme_hot_score": safe_float(row["theme_hot_score"]),
            "ret20": safe_float(row["plate_ret_20d"]),
            "ret20_rank": safe_float(row["ret20_rank"]),
            "ret60": safe_float(row["plate_ret_60d"]),
            "ret60_rank": safe_float(row["ret60_rank"]),
            "turn_ma20": safe_float(row["plate_turn_ma20"]),
            "turn_ma120": safe_float(row["plate_turn_ma120"]),
            "turn_rank": safe_float(row["turn_rank"]),
            "turn_accel": safe_float(row["turn_accel"]),
            "turn_accel_rank": safe_float(row["turn_accel_rank"]),
            "amount_ma20": safe_float(row["plate_amount_ma20"]),
            "amount_rank": safe_float(row["amount_rank"]),
            "latest_turnover": safe_float(row["turnover_rate"]),
            "latest_amount_share": safe_float(row["amount_share_pct"]),
            "latest_date": row["trade_date"].strftime("%Y-%m-%d"),
        }
        for rank, (_, row) in enumerate(plate_latest.iterrows(), 1)
    ]
    payload = {
        "schema": "theme_tracking.v1",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": {
            "stock_db": str(STOCK_DB),
            "plate_db": str(PLATE_DB),
            "candidate_pool": "sw3_member.is_leader=1",
            "lookback_days": lookback_days,
            "tracking_method": "max(0.55*return_corr + 0.25*trend_corr_60d + 0.20*turnover_corr) across all SW2 plates",
        },
        "candidate_pool_size": int(len(pool)),
        "mapped_stock_count": int(len(stock_themes)),
        "theme_count": int(len(theme_rankings)),
        "theme_rankings": theme_rankings,
        "stock_themes": stock_themes,
    }
    return payload


def write_outputs(payload: Dict[str, Any]) -> None:
    CAPITAL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in LEGACY_OUTPUTS:
        if path.exists():
            path.unlink()


def fmt_num(value: Any, width: int = 6, decimals: int = 1, suffix: str = "") -> str:
    number = safe_float(value)
    if number is None:
        return "-"
    return f"{number:.{decimals}f}{suffix}"


def _console(file: Any = None, width: Optional[int] = None) -> Console:
    width = width or max(shutil.get_terminal_size((220, 24)).columns, 220)
    return Console(file=file, width=width, color_system=None, highlight=False, markup=False)


def _cell(value: Any) -> Text:
    return Text(str(value if value is not None else ""))


def _plain_table(columns: List[Tuple[str, str]]) -> Table:
    table = Table(box=None, show_header=True, pad_edge=False, collapse_padding=True)
    for header, justify in columns:
        table.add_column(header, justify=justify, no_wrap=True)
    return table


def build_plate_table(plates: List[Dict[str, Any]]) -> Table:
    table = _plain_table([
        ("#", "right"),
        ("代码", "left"),
        ("题材", "left"),
        ("热度分", "right"),
        ("20涨", "right"),
        ("20排名", "right"),
        ("60涨", "right"),
        ("60排名", "right"),
        ("20换手", "right"),
        ("换手排名", "right"),
        ("20占比", "right"),
        ("占比排名", "right"),
        ("换手加速", "right"),
        ("加速排名", "right"),
        ("最新换手", "right"),
        ("最新占比", "right"),
        ("日期", "left"),
    ])
    for plate in plates:
        table.add_row(
            _cell(plate["rank"]),
            _cell(plate["plate_code"]),
            _cell(plate["plate_name"]),
            _cell(fmt_num(plate["theme_hot_score"], decimals=1)),
            _cell(fmt_num(plate["ret20"], decimals=1)),
            _cell(fmt_num(plate["ret20_rank"], decimals=0)),
            _cell(fmt_num(plate["ret60"], decimals=1)),
            _cell(fmt_num(plate["ret60_rank"], decimals=0)),
            _cell(fmt_num(plate["turn_ma20"], decimals=2)),
            _cell(fmt_num(plate["turn_rank"], decimals=0)),
            _cell(fmt_num(plate["amount_ma20"], decimals=2)),
            _cell(fmt_num(plate["amount_rank"], decimals=0)),
            _cell(fmt_num(plate["turn_accel"], decimals=2)),
            _cell(fmt_num(plate["turn_accel_rank"], decimals=0)),
            _cell(fmt_num(plate["latest_turnover"], decimals=2)),
            _cell(fmt_num(plate["latest_amount_share"], decimals=2)),
            _cell(plate["latest_date"]),
        )
    return table


def build_stock_table(stocks: List[Dict[str, Any]]) -> Table:
    table = _plain_table([
        ("#", "right"),
        ("代码", "left"),
        ("名称", "left"),
        ("跟踪题材", "left"),
        ("静态SW2", "left"),
        ("静态SW3", "left"),
        ("跟踪相关", "right"),
        ("收益相关", "right"),
        ("60日相关", "right"),
        ("换手相关", "right"),
        ("匹配天数", "right"),
        ("日期", "left"),
    ])
    for i, stock in enumerate(stocks, 1):
        table.add_row(
            _cell(i),
            _cell(stock["code"]),
            _cell(stock.get("name") or ""),
            _cell(stock.get("tracking_theme") or ""),
            _cell(stock.get("static_sw2") or ""),
            _cell(stock.get("static_sw3") or ""),
            _cell(fmt_num(stock["tracking_corr"], decimals=2)),
            _cell(fmt_num(stock["return_corr"], decimals=2)),
            _cell(fmt_num(stock["trend_corr_60d"], decimals=2)),
            _cell(fmt_num(stock["turnover_corr"], decimals=2)),
            _cell(stock.get("matched_days")),
            _cell(stock["latest_date"]),
        )
    return table


def print_summary(payload: Dict[str, Any]) -> None:
    print(
        f"[theme] pool={payload['candidate_pool_size']} mapped={payload['mapped_stock_count']} "
        f"themes={payload['theme_count']} file={OUT_JSON.name}",
        flush=True,
    )
    print("[theme] 题材完整排名:", flush=True)
    console = _console()
    console.print(build_plate_table(payload["theme_rankings"]))
    print("[theme] 股票真实跟踪题材:", flush=True)
    console.print(build_stock_table(payload["stock_themes"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于本地数据生成股票真实跟踪题材映射")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    return parser


def main() -> Dict[str, Any]:
    args = build_parser().parse_args()
    payload = build_theme_candidates(
        lookback_days=args.lookback_days,
    )
    write_outputs(payload)
    print_summary(payload)
    return payload


if __name__ == "__main__":
    main()
