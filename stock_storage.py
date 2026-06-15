"""个股数据的 SQLite 持久化层（对照 fund_storage.py 的范式）。

设计要点（详见 schema 讨论）：
  - 混合存储：日线+估值时间序列(history)正规化进 stock_history 大表(~12M 行)，
    财务/指标/分红等"整只读取、体量小"的中等结构以 JSON blob 列存进 stock_meta。
  - 主键用 6 位 code，彻底丢掉 CN_{code}_{name}.json 的文件名方案；改名只 UPDATE name。
  - upsert 语义：ON CONFLICT(code,date) DO UPDATE 同时支持增量补日与 qfq 全量重算。
  - 基准 ETF NAV(510310/510580)结构同基金 NAV，单独存进 index_nav/index_nav_meta。

load_stock() 把行重新拼回原 JSON dict，消费方(stock_advanced_strategies 等)零改造，
拿到的仍是 stock.get("history").records / stock.get("financials").balance 那一套。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from stock_crawl_common import (
    HISTORY_PRICE_FIELDS,
    HISTORY_VALUATION_FIELDS,
    load_json_file,
    safe_float,
)


DATA_DIR = Path("data")
DEFAULT_DB_FILE = DATA_DIR / "stock_data.sqlite3"
STOCK_DATA_DIR = DATA_DIR / "stock_data"
SCHEMA_VERSION = 1

# 大表列：8 个日频行情字段 + 5 个估值字段，与 stock_crawl_common 的 canonical schema 同源。
HISTORY_COLUMNS: tuple = tuple(HISTORY_PRICE_FIELDS) + tuple(HISTORY_VALUATION_FIELDS)

# 从 JSON 保留的新鲜度时间戳（一次性迁移疤痕 *_migrated_at/*_sanitized_at/*_cleaned_at 丢弃）。
META_TIMESTAMP_FIELDS = (
    "financials_refetched_at",
    "daily_refetched_at",
    "history_refetched_at",
)

# 基准 ETF 的友好名（文件本身不带 name）。
KNOWN_INDEX_NAMES = {"510310": "沪深300ETF", "510580": "中证500ETF"}


def connect(db_file: Path | str = DEFAULT_DB_FILE) -> sqlite3.Connection:
    """打开个股缓存库并确保 schema 存在。busy_timeout 取 30s 以容纳多 worker 并发写。"""
    if str(db_file) != ":memory:":
        db_target = Path(db_file)
        db_target.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_target))
    else:
        conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    # schema 已是目标版本时直接返回：避免每次 connect 都写 PRAGMA user_version，
    # 那会在 WAL 里产生帧、扰动文件 mtime（曾导致消费方按文件指纹缓存被反复打穿）。
    row = conn.execute("PRAGMA user_version").fetchone()
    if row is not None and row[0] == SCHEMA_VERSION:
        return
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS stock_meta (
            code                    TEXT PRIMARY KEY,
            name                    TEXT NOT NULL,
            fetch_time              TEXT,
            financials_refetched_at TEXT,
            daily_refetched_at      TEXT,
            history_refetched_at    TEXT,
            daily_stats_json        TEXT,
            financials_json         TEXT,
            indicators_json         TEXT,
            dividends_json          TEXT,
            pledge_ratio            REAL,
            pledge_count            REAL,
            pledge_trade_date       TEXT,
            industry                TEXT,
            candidate_for_json      TEXT,
            history_source          TEXT,
            history_start_date      TEXT,
            history_end_date        TEXT,
            price_adjust            TEXT DEFAULT 'qfq',
            updated_at              TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_stock_meta_industry
        ON stock_meta (industry);

        CREATE TABLE IF NOT EXISTS stock_history (
            code                TEXT NOT NULL,
            date                TEXT NOT NULL,
            daily_open          REAL,
            daily_high          REAL,
            daily_low           REAL,
            daily_close         REAL,
            daily_volume        REAL,
            daily_amount        REAL,
            daily_change_pct    REAL,
            daily_turnover_rate REAL,
            market_cap          REAL,
            pe_ttm              REAL,
            pe_static           REAL,
            pb                  REAL,
            pcf                 REAL,
            PRIMARY KEY (code, date)
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS index_nav (
            code              TEXT NOT NULL,
            date              TEXT NOT NULL,
            nav               TEXT,
            nav_acc           TEXT,
            daily_growth_rate TEXT,
            updated_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (code, date)
        );

        CREATE TABLE IF NOT EXISTS index_nav_meta (
            code         TEXT PRIMARY KEY,
            name         TEXT,
            target_years INTEGER,
            start_date   TEXT,
            end_date     TEXT,
            record_count INTEGER NOT NULL DEFAULT 0,
            updated_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


# ─── 动态 SQL（列集中定义，避免手抖漏列）────────────────────────

_META_COLUMNS = (
    "code", "name", "fetch_time",
    "financials_refetched_at", "daily_refetched_at", "history_refetched_at",
    "daily_stats_json", "financials_json", "indicators_json", "dividends_json",
    "pledge_ratio", "pledge_count", "pledge_trade_date", "industry",
    "candidate_for_json", "history_source", "history_start_date", "history_end_date",
    "price_adjust",
)
_META_UPDATE_SET = ", ".join(f"{c} = excluded.{c}" for c in _META_COLUMNS if c != "code")
_META_INSERT_SQL = (
    f"INSERT INTO stock_meta ({', '.join(_META_COLUMNS)}, updated_at) "
    f"VALUES ({', '.join('?' for _ in _META_COLUMNS)}, CURRENT_TIMESTAMP) "
    f"ON CONFLICT(code) DO UPDATE SET {_META_UPDATE_SET}, updated_at = CURRENT_TIMESTAMP"
)

_HISTORY_INSERT_COLUMNS = ("code", "date") + HISTORY_COLUMNS
_HISTORY_UPDATE_SET = ", ".join(f"{c} = excluded.{c}" for c in HISTORY_COLUMNS)
_HISTORY_INSERT_SQL = (
    f"INSERT INTO stock_history ({', '.join(_HISTORY_INSERT_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in _HISTORY_INSERT_COLUMNS)}) "
    f"ON CONFLICT(code, date) DO UPDATE SET {_HISTORY_UPDATE_SET}"
)


# ─── 写入 ──────────────────────────────────────────────────────

def save_stock(
    conn: sqlite3.Connection,
    data: Mapping[str, Any],
    *,
    replace_history: bool = True,
    write_history: bool = True,
) -> Optional[str]:
    """把一只股票的完整 JSON dict 落库（stock_meta 一行 + stock_history 多行）。

    replace_history=True（默认）：先删该 code 全部日线再整段写入，匹配爬虫"内存合并后
    写全集"的语义。置 False 时改走纯 upsert，仅补/改传入的日期，适合增量补日。
    write_history=False 时完全不动 stock_history（只 upsert meta），用于只改财报/指标的场景。
    返回写入的 code；symbol 缺失则跳过返回 None。
    """
    code = _normalize_code(data.get("symbol"))
    if not code:
        return None

    history = data.get("history") if isinstance(data.get("history"), Mapping) else {}
    records = _sorted_records(history.get("records"))
    daily = data.get("daily") if isinstance(data.get("daily"), Mapping) else {}
    pledge = data.get("pledge") if isinstance(data.get("pledge"), Mapping) else {}

    start_date = _optional_text(history.get("start_date")) or (records[0]["date"] if records else None)
    end_date = _optional_text(history.get("end_date")) or (records[-1]["date"] if records else None)

    meta_row = (
        code,
        str(data.get("name") or code),
        _optional_text(data.get("fetch_time")),
        _optional_text(data.get("financials_refetched_at")),
        _optional_text(data.get("daily_refetched_at")),
        _optional_text(data.get("history_refetched_at")),
        _json_or_none(daily.get("stats")),
        _json_or_none(data.get("financials")),
        _json_or_none(data.get("indicators")),
        _json_or_none(data.get("dividends")),
        safe_float(pledge.get("pledge_ratio")),
        safe_float(pledge.get("pledge_count")),
        _optional_text(pledge.get("trade_date")),
        _optional_text(pledge.get("industry")),
        _json_or_none(data.get("candidate_for")),
        _optional_text(history.get("source")),
        start_date,
        end_date,
        _optional_text(history.get("price_adjust")) or "qfq",
    )

    with conn:
        conn.execute(_META_INSERT_SQL, meta_row)
        if write_history:
            if replace_history:
                conn.execute("DELETE FROM stock_history WHERE code = ?", (code,))
            if records:
                conn.executemany(
                    _HISTORY_INSERT_SQL,
                    [_history_value_tuple(code, record) for record in records],
                )
    return code


def save_index_nav(conn: sqlite3.Connection, payload: Mapping[str, Any], *, name: Optional[str] = None) -> int:
    """把基准 ETF NAV 文件 dict（code/records/start_date/...）落库。返回写入记录数。"""
    code = _optional_text(payload.get("code"))
    if not code:
        return 0
    records = _sorted_records(payload.get("records"))
    name = name or payload.get("name") or KNOWN_INDEX_NAMES.get(code)
    start_date = _optional_text(payload.get("start_date")) or (records[0]["date"] if records else None)
    end_date = _optional_text(payload.get("end_date")) or (records[-1]["date"] if records else None)
    target_years = payload.get("target_years")

    with conn:
        conn.execute("DELETE FROM index_nav WHERE code = ?", (code,))
        if records:
            conn.executemany(
                """
                INSERT INTO index_nav (code, date, nav, nav_acc, daily_growth_rate, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [
                    (
                        code,
                        str(record["date"]),
                        _optional_text(record.get("nav")),
                        _optional_text(record.get("nav_acc")),
                        _optional_text(record.get("daily_growth_rate")),
                    )
                    for record in records
                ],
            )
        conn.execute(
            """
            INSERT INTO index_nav_meta (code, name, target_years, start_date, end_date, record_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(code) DO UPDATE SET
                name = excluded.name,
                target_years = excluded.target_years,
                start_date = excluded.start_date,
                end_date = excluded.end_date,
                record_count = excluded.record_count,
                updated_at = CURRENT_TIMESTAMP
            """,
            (code, name, target_years, start_date, end_date, len(records)),
        )
    return len(records)


def import_stock_data_dir(
    conn: sqlite3.Connection,
    source_dir: Path | str = STOCK_DATA_DIR,
    *,
    limit: int = 0,
    progress=None,
) -> Dict[str, Any]:
    """把 data/stock_data/CN_*.json 全量灌进 DB（每只一个事务，断点可续）。"""
    files = sorted(Path(source_dir).glob("CN_*.json"))
    if limit:
        files = files[:limit]
    imported = 0
    skipped = 0
    for idx, fp in enumerate(files, 1):
        data = load_json_file(fp, None)
        if not isinstance(data, dict):
            skipped += 1
            continue
        if not data.get("symbol"):
            # 老文件偶尔缺 symbol，从文件名 CN_{code}_{name}.json 兜底
            parts = fp.name.split("_")
            data = dict(data)
            data["symbol"] = parts[1] if len(parts) > 1 else ""
        if save_stock(conn, data) is None:
            skipped += 1
            continue
        imported += 1
        if progress and idx % 200 == 0:
            progress(idx, len(files))
    return {"files": len(files), "imported": imported, "skipped": skipped}


# ─── 读取 ──────────────────────────────────────────────────────

def load_stock(conn: sqlite3.Connection, code: str, *, include_history: bool = True) -> Dict[str, Any]:
    """按 code 取回完整 JSON dict（与原磁盘文件结构等价）。不存在返回 {}。

    include_history=False 时不读 stock_history（history.records 为空），用于只需
    财报/指标/分红/估值统计、日线另有来源(如 cn_index)的消费方，省去重建百万行记录。
    """
    code = _normalize_code(code)
    meta = conn.execute("SELECT * FROM stock_meta WHERE code = ?", (code,)).fetchone()
    if meta is None:
        return {}
    records = _load_history_records(conn, code) if include_history else []
    return _rebuild_stock(meta, records)


def iter_stocks(
    conn: sqlite3.Connection,
    codes: Optional[Sequence[str]] = None,
    *,
    include_history: bool = True,
) -> Iterator[tuple[str, Dict[str, Any]]]:
    """逐只产出 (code, dict)，避免一次性把全 universe(~1GB)读进内存。"""
    if codes is None:
        codes = list_codes(conn)
    for code in codes:
        entry = load_stock(conn, code, include_history=include_history)
        if entry:
            yield code, entry


def load_all_stocks(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    """取回全 universe {code: dict}。注意：会把全部日线读进内存，约 1GB 量级。"""
    return {code: entry for code, entry in iter_stocks(conn)}


def load_index_nav(conn: sqlite3.Connection, code: str) -> Dict[str, Any]:
    """取回基准 ETF NAV，结构同 csi300_etf_nav.json。不存在返回 {}。"""
    code = str(code)
    rows = conn.execute(
        "SELECT date, nav, nav_acc, daily_growth_rate FROM index_nav WHERE code = ? ORDER BY date",
        (code,),
    ).fetchall()
    if not rows:
        return {}
    records = [
        {
            "date": row["date"],
            "nav": row["nav"],
            "nav_acc": row["nav_acc"],
            "daily_growth_rate": row["daily_growth_rate"],
        }
        for row in rows
    ]
    payload: Dict[str, Any] = {
        "code": code,
        "records": records,
        "start_date": records[0]["date"],
        "end_date": records[-1]["date"],
    }
    meta = conn.execute(
        "SELECT name, target_years, start_date, end_date FROM index_nav_meta WHERE code = ?",
        (code,),
    ).fetchone()
    if meta:
        if meta["name"]:
            payload["name"] = meta["name"]
        if meta["target_years"] is not None:
            payload["target_years"] = meta["target_years"]
        if meta["start_date"]:
            payload["start_date"] = meta["start_date"]
        if meta["end_date"]:
            payload["end_date"] = meta["end_date"]
    return payload


def list_codes(conn: sqlite3.Connection) -> List[str]:
    return [row["code"] for row in conn.execute("SELECT code FROM stock_meta ORDER BY code")]


def stock_history_end_date(conn: sqlite3.Connection, code: str) -> Optional[str]:
    """该 code 已有日线的最新日期，供增量补日定起点。无数据返回 None。"""
    code = _normalize_code(code)
    row = conn.execute(
        "SELECT MAX(date) AS end_date FROM stock_history WHERE code = ?", (code,)
    ).fetchone()
    return row["end_date"] if row else None


def table_count(conn: sqlite3.Connection, table_name: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
    return int(row["count"]) if row else 0


# ─── 外部集成辅助（爬虫/消费方切库用）──────────────────────────

_thread_local = threading.local()


def thread_conn(db_file: Path | str = DEFAULT_DB_FILE) -> sqlite3.Connection:
    """每线程一个连接（sqlite3 连接非线程安全）。供爬虫多 worker 并发写：
    WAL + busy_timeout 让写锁排队，连接随线程复用，进程退出由 sqlite 回收。"""
    key = str(db_file)
    cache = getattr(_thread_local, "conns", None)
    if cache is None:
        cache = _thread_local.conns = {}
    conn = cache.get(key)
    if conn is None:
        conn = cache[key] = connect(db_file)
    return conn


def existing_codes(conn: sqlite3.Connection) -> set:
    """已爬到财报三件套的 code 集合（对应旧 load_existing()：要求 financials/indicators/dividends 齐全）。"""
    rows = conn.execute(
        """
        SELECT code FROM stock_meta
        WHERE financials_json IS NOT NULL
          AND indicators_json IS NOT NULL
          AND dividends_json IS NOT NULL
        """
    ).fetchall()
    return {row["code"] for row in rows}


def codes_with_history(conn: sqlite3.Connection) -> List[str]:
    """有日线记录的 code（PIT 回测股票池，对应旧"带 history.records 的文件"）。"""
    return [row["code"] for row in conn.execute("SELECT DISTINCT code FROM stock_history ORDER BY code")]


def stock_exists(conn: sqlite3.Connection, code: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM stock_meta WHERE code = ? LIMIT 1", (_normalize_code(code),)
    ).fetchone() is not None


def load_history_records(conn: sqlite3.Connection, code: str) -> List[Dict[str, Any]]:
    """只取该 code 的日线记录（daily_* + 估值列），不解析财报 blob。"""
    return _load_history_records(conn, _normalize_code(code))


def history_refetched_at(conn: sqlite3.Connection, code: str) -> Optional[str]:
    row = conn.execute(
        "SELECT history_refetched_at FROM stock_meta WHERE code = ?", (_normalize_code(code),)
    ).fetchone()
    return row["history_refetched_at"] if row else None


def iter_history(conn: sqlite3.Connection):
    """轻量产出 (meta_row, records[])：只读 stock_history + 必要 meta 列，跳过财报 blob 解析。

    meta_row 含 code/name/history_source/history_start_date/history_end_date/price_adjust。
    用单条按 (code,date) 排序的 bulk 查询流式分组（PK WITHOUT ROWID 已物理有序），
    避免对每个 code 单独发查询（4900+ 次往返）。
    """
    meta_by_code = {
        row["code"]: row
        for row in conn.execute(
            """
            SELECT code, name, history_source, history_start_date, history_end_date, price_adjust
            FROM stock_meta
            """
        )
    }
    columns = ", ".join(HISTORY_COLUMNS)
    cursor = conn.execute(
        f"SELECT code, date, {columns} FROM stock_history ORDER BY code, date"
    )
    current_code = None
    records: List[Dict[str, Any]] = []
    for row in cursor:
        code = row["code"]
        if code != current_code:
            if records and current_code in meta_by_code:
                yield meta_by_code[current_code], records
            current_code = code
            records = []
        records.append({"date": row["date"], **{col: row[col] for col in HISTORY_COLUMNS}})
    if records and current_code in meta_by_code:
        yield meta_by_code[current_code], records


def db_signature(db_file: Path | str = DEFAULT_DB_FILE) -> Dict[str, Any]:
    """缓存版本签名：股票数/最新 updated_at/日线行数/库文件大小。"""
    path = Path(str(db_file))
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    conn = connect(db_file)
    try:
        meta = conn.execute(
            "SELECT COUNT(*) AS c, MAX(updated_at) AS m FROM stock_meta"
        ).fetchone()
        history_rows = table_count(conn, "stock_history")
    finally:
        conn.close()
    return {
        "count": int(meta["c"]) if meta else 0,
        "max_updated_at": meta["m"] if meta else None,
        "history_rows": history_rows,
        "db_size": size,
    }


# ─── 内部工具 ──────────────────────────────────────────────────

def _load_history_records(conn: sqlite3.Connection, code: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        f"SELECT date, {', '.join(HISTORY_COLUMNS)} FROM stock_history WHERE code = ? ORDER BY date",
        (code,),
    ).fetchall()
    return [
        {"date": row["date"], **{col: row[col] for col in HISTORY_COLUMNS}}
        for row in rows
    ]


def _rebuild_stock(meta: sqlite3.Row, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    code = meta["code"]
    name = meta["name"]
    data: Dict[str, Any] = {"symbol": code, "name": name}
    if meta["fetch_time"]:
        data["fetch_time"] = meta["fetch_time"]

    stats = _loads(meta["daily_stats_json"])
    data["daily"] = {"stats": stats} if stats is not None else {}

    for key, column in (
        ("financials", "financials_json"),
        ("indicators", "indicators_json"),
        ("dividends", "dividends_json"),
    ):
        value = _loads(meta[column])
        if value is not None:
            data[key] = value

    data["pledge"] = {
        "pledge_ratio": meta["pledge_ratio"],
        "pledge_count": meta["pledge_count"],
        "trade_date": meta["pledge_trade_date"],
        "industry": meta["industry"],
    }

    candidate_for = _loads(meta["candidate_for_json"])
    if candidate_for is not None:
        data["candidate_for"] = candidate_for

    for field in META_TIMESTAMP_FIELDS:
        if meta[field]:
            data[field] = meta[field]

    history: Dict[str, Any] = {
        "symbol": code,
        "name": name,
        "records": records,
        "source": meta["history_source"],
        "price_adjust": meta["price_adjust"] or "qfq",
        "change_pct_basis": "close_to_close",
    }
    if meta["history_start_date"]:
        history["start_date"] = meta["history_start_date"]
    if meta["history_end_date"]:
        history["end_date"] = meta["history_end_date"]
    data["history"] = history
    return data


def _history_value_tuple(code: str, record: Mapping[str, Any]) -> tuple:
    return (code, str(record["date"])) + tuple(safe_float(record.get(col)) for col in HISTORY_COLUMNS)


def _sorted_records(records: Any) -> List[Mapping[str, Any]]:
    rows = [r for r in (records or []) if isinstance(r, Mapping) and r.get("date")]
    return sorted(rows, key=lambda r: str(r["date"]))


def _normalize_code(value: Any) -> str:
    text = str(value or "").strip()
    return text.zfill(6) if text else ""


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _json_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(text: Optional[str]) -> Any:
    if text is None:
        return None
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
