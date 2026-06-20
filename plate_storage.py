"""Plate/industry level SQLite storage.

This module stores board/industry daily series separately from stock_data.sqlite3.
Current producer: plate_crawl_history.py crawling SW second-level industry data.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DATA_DIR = Path("data")
DEFAULT_DB_FILE = DATA_DIR / "plate_data.sqlite3"
SCHEMA_VERSION = 1
SQLITE_BUSY_TIMEOUT_MS = 120000
PLATE_TYPE_SW2 = "sw2"
SOURCE_SW2_DAILY = "swsresearch.index_analysis_daily_sw(symbol='二级行业')"

_CONNECT_LOCK = threading.Lock()


def connect(db_file: Path | str = DEFAULT_DB_FILE) -> sqlite3.Connection:
    if str(db_file) != ":memory:":
        db_target = Path(db_file)
        db_target.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_target), timeout=SQLITE_BUSY_TIMEOUT_MS / 1000.0)
    else:
        conn = sqlite3.connect(":memory:", timeout=SQLITE_BUSY_TIMEOUT_MS / 1000.0)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    if str(db_file) == ":memory:":
        conn.execute("PRAGMA foreign_keys = ON")
        ensure_schema(conn)
    else:
        with _CONNECT_LOCK:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS plate_meta (
            plate_type   TEXT NOT NULL,
            plate_code   TEXT NOT NULL,
            plate_name   TEXT NOT NULL,
            source       TEXT NOT NULL,
            first_date   TEXT,
            last_date    TEXT,
            record_count INTEGER NOT NULL DEFAULT 0,
            updated_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (plate_type, plate_code)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS plate_daily (
            plate_type           TEXT NOT NULL,
            plate_code           TEXT NOT NULL,
            trade_date           TEXT NOT NULL,
            plate_name           TEXT NOT NULL,
            close_index          REAL,
            volume               REAL,
            change_pct           REAL,
            turnover_rate        REAL,
            pe                   REAL,
            pb                   REAL,
            mean_price           REAL,
            amount_share_pct     REAL,
            float_market_cap     REAL,
            avg_float_market_cap REAL,
            dividend_yield       REAL,
            source               TEXT NOT NULL,
            raw_json             TEXT,
            updated_at           TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (plate_type, plate_code, trade_date),
            FOREIGN KEY (plate_type, plate_code)
                REFERENCES plate_meta (plate_type, plate_code)
                ON DELETE CASCADE
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_plate_daily_date_type
        ON plate_daily (trade_date, plate_type);

        CREATE INDEX IF NOT EXISTS idx_plate_daily_type_name_date
        ON plate_daily (plate_type, plate_name, trade_date);

        CREATE TABLE IF NOT EXISTS plate_column_comments (
            table_name  TEXT NOT NULL,
            column_name TEXT NOT NULL,
            comment     TEXT NOT NULL,
            PRIMARY KEY (table_name, column_name)
        ) WITHOUT ROWID;
        """
    )
    _seed_column_comments(conn)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


_COLUMN_COMMENTS: Tuple[Tuple[str, str, str], ...] = (
    ("plate_meta", "plate_type", "板块类型；当前 sw2 表示申万二级行业。"),
    ("plate_meta", "plate_code", "板块/行业代码，例如 801125。"),
    ("plate_meta", "plate_name", "板块/行业名称，例如 白酒Ⅱ。"),
    ("plate_meta", "source", "数据来源标识。"),
    ("plate_meta", "first_date", "该板块已入库的最早交易日期，YYYY-MM-DD。"),
    ("plate_meta", "last_date", "该板块已入库的最新交易日期，YYYY-MM-DD。"),
    ("plate_meta", "record_count", "该板块 plate_daily 已入库记录数。"),
    ("plate_meta", "updated_at", "该元数据行最近更新时间。"),
    ("plate_daily", "plate_type", "板块类型；当前 sw2 表示申万二级行业。"),
    ("plate_daily", "plate_code", "板块/行业代码，例如 801125。"),
    ("plate_daily", "trade_date", "交易日期，YYYY-MM-DD。"),
    ("plate_daily", "plate_name", "板块/行业名称快照。"),
    ("plate_daily", "close_index", "申万指数分析接口的收盘指数。"),
    ("plate_daily", "volume", "申万指数分析接口的成交量原始口径。"),
    ("plate_daily", "change_pct", "当日涨跌幅，百分比数值。"),
    ("plate_daily", "turnover_rate", "当日换手率，百分比数值。"),
    ("plate_daily", "pe", "市盈率。"),
    ("plate_daily", "pb", "市净率。"),
    ("plate_daily", "mean_price", "均价。"),
    ("plate_daily", "amount_share_pct", "成交额占比，申万接口原始口径。"),
    ("plate_daily", "float_market_cap", "申万接口流通市值口径；不等于行业成分股真实总市值。"),
    ("plate_daily", "avg_float_market_cap", "申万接口平均流通市值口径。"),
    ("plate_daily", "dividend_yield", "股息率。"),
    ("plate_daily", "source", "数据来源标识。"),
    ("plate_daily", "raw_json", "原始接口行 JSON，保留用于追溯字段口径。"),
    ("plate_daily", "updated_at", "该日线记录最近写入/更新时间。"),
    ("plate_column_comments", "table_name", "被说明字段所属表名。"),
    ("plate_column_comments", "column_name", "被说明字段名。"),
    ("plate_column_comments", "comment", "字段中文说明。"),
)


def _seed_column_comments(conn: sqlite3.Connection) -> None:
    conn.executemany(
        """
        INSERT INTO plate_column_comments (table_name, column_name, comment)
        VALUES (?, ?, ?)
        ON CONFLICT(table_name, column_name) DO UPDATE SET
            comment = excluded.comment
        """,
        _COLUMN_COMMENTS,
    )


def first_present(row: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return round(number, 6)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (datetime,)):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def normalize_date(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value)[:10]
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text


def normalize_sw2_row(row: Mapping[str, Any], *, source: str = SOURCE_SW2_DAILY) -> Optional[Dict[str, Any]]:
    plate_code = first_present(row, ("指数代码", "swindexcode", "plate_code"))
    plate_name = first_present(row, ("指数名称", "swindexname", "plate_name"))
    trade_date = normalize_date(first_present(row, ("发布日期", "bargaindate", "trade_date", "date")))
    if not plate_code or not plate_name or not trade_date:
        return None
    raw = {str(k): _json_safe(v) for k, v in row.items()}
    return {
        "plate_type": PLATE_TYPE_SW2,
        "plate_code": str(plate_code),
        "trade_date": trade_date,
        "plate_name": str(plate_name),
        "close_index": safe_float(first_present(row, ("收盘指数", "closeindex", "close_index"))),
        "volume": safe_float(first_present(row, ("成交量", "bargainamount", "volume"))),
        "change_pct": safe_float(first_present(row, ("涨跌幅", "markup", "change_pct"))),
        "turnover_rate": safe_float(first_present(row, ("换手率", "turnoverrate", "turnover_rate"))),
        "pe": safe_float(first_present(row, ("市盈率", "pe"))),
        "pb": safe_float(first_present(row, ("市净率", "pb"))),
        "mean_price": safe_float(first_present(row, ("均价", "meanprice", "mean_price"))),
        "amount_share_pct": safe_float(first_present(row, ("成交额占比", "bargainsumrate", "amount_share_pct"))),
        "float_market_cap": safe_float(first_present(row, ("流通市值", "negotiablessharesum1", "float_market_cap"))),
        "avg_float_market_cap": safe_float(first_present(row, ("平均流通市值", "negotiablessharesum2", "avg_float_market_cap"))),
        "dividend_yield": safe_float(first_present(row, ("股息率", "dp", "dividend_yield"))),
        "source": source,
        "raw_json": json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
    }


def _existing_daily_keys(
    conn: sqlite3.Connection,
    keys: Sequence[Tuple[str, str, str]],
) -> set[Tuple[str, str, str]]:
    existing: set[Tuple[str, str, str]] = set()
    for i in range(0, len(keys), 300):
        chunk = keys[i:i + 300]
        conditions = " OR ".join(["(plate_type = ? AND plate_code = ? AND trade_date = ?)"] * len(chunk))
        params: List[str] = []
        for key in chunk:
            params.extend(key)
        rows = conn.execute(
            f"SELECT plate_type, plate_code, trade_date FROM plate_daily WHERE {conditions}",
            params,
        ).fetchall()
        existing.update((row["plate_type"], row["plate_code"], row["trade_date"]) for row in rows)
    return existing


def save_sw2_daily_rows(
    conn: sqlite3.Connection,
    rows: Iterable[Mapping[str, Any]],
    *,
    source: str = SOURCE_SW2_DAILY,
) -> Dict[str, int]:
    normalized = [item for row in rows if (item := normalize_sw2_row(row, source=source))]
    if not normalized:
        return {"fetched": 0, "inserted": 0, "updated": 0}

    keys = [(r["plate_type"], r["plate_code"], r["trade_date"]) for r in normalized]
    existing = _existing_daily_keys(conn, keys)
    inserted = sum(1 for key in keys if key not in existing)
    updated = len(keys) - inserted

    with conn:
        conn.executemany(
            """
            INSERT INTO plate_meta
                (plate_type, plate_code, plate_name, source, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(plate_type, plate_code) DO UPDATE SET
                plate_name = excluded.plate_name,
                source = excluded.source,
                updated_at = CURRENT_TIMESTAMP
            """,
            [
                (r["plate_type"], r["plate_code"], r["plate_name"], r["source"])
                for r in normalized
            ],
        )
        conn.executemany(
            """
            INSERT INTO plate_daily (
                plate_type, plate_code, trade_date, plate_name,
                close_index, volume, change_pct, turnover_rate, pe, pb,
                mean_price, amount_share_pct, float_market_cap,
                avg_float_market_cap, dividend_yield, source, raw_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(plate_type, plate_code, trade_date) DO UPDATE SET
                plate_name = excluded.plate_name,
                close_index = excluded.close_index,
                volume = excluded.volume,
                change_pct = excluded.change_pct,
                turnover_rate = excluded.turnover_rate,
                pe = excluded.pe,
                pb = excluded.pb,
                mean_price = excluded.mean_price,
                amount_share_pct = excluded.amount_share_pct,
                float_market_cap = excluded.float_market_cap,
                avg_float_market_cap = excluded.avg_float_market_cap,
                dividend_yield = excluded.dividend_yield,
                source = excluded.source,
                raw_json = excluded.raw_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            [
                (
                    r["plate_type"], r["plate_code"], r["trade_date"], r["plate_name"],
                    r["close_index"], r["volume"], r["change_pct"], r["turnover_rate"],
                    r["pe"], r["pb"], r["mean_price"], r["amount_share_pct"],
                    r["float_market_cap"], r["avg_float_market_cap"], r["dividend_yield"],
                    r["source"], r["raw_json"],
                )
                for r in normalized
            ],
        )
        _refresh_plate_meta(conn, PLATE_TYPE_SW2)

    return {"fetched": len(normalized), "inserted": inserted, "updated": updated}


def _refresh_plate_meta(conn: sqlite3.Connection, plate_type: str) -> None:
    conn.execute(
        """
        UPDATE plate_meta
        SET
            first_date = (
                SELECT MIN(d.trade_date)
                FROM plate_daily d
                WHERE d.plate_type = plate_meta.plate_type
                  AND d.plate_code = plate_meta.plate_code
            ),
            last_date = (
                SELECT MAX(d.trade_date)
                FROM plate_daily d
                WHERE d.plate_type = plate_meta.plate_type
                  AND d.plate_code = plate_meta.plate_code
            ),
            record_count = (
                SELECT COUNT(*)
                FROM plate_daily d
                WHERE d.plate_type = plate_meta.plate_type
                  AND d.plate_code = plate_meta.plate_code
            ),
            updated_at = CURRENT_TIMESTAMP
        WHERE plate_type = ?
        """,
        (plate_type,),
    )


def latest_trade_date(conn: sqlite3.Connection, plate_type: str = PLATE_TYPE_SW2) -> Optional[str]:
    row = conn.execute(
        "SELECT MAX(trade_date) AS trade_date FROM plate_daily WHERE plate_type = ?",
        (plate_type,),
    ).fetchone()
    return row["trade_date"] if row and row["trade_date"] else None


def table_count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
    return int(row["n"] if row else 0)
