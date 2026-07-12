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

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from stock_crawl_common import (
    HISTORY_DAILY_OHLCV_FIELDS,
    HISTORY_PRICE_FIELDS,
    HISTORY_VALUATION_FIELDS,
    load_json_file,
    normalize_history_records,
    safe_float,
)


DATA_DIR = Path("data")
DEFAULT_DB_FILE = DATA_DIR / "stock_data.sqlite3"
STOCK_DATA_DIR = DATA_DIR / "stock_data"
SCHEMA_VERSION = 5
SQLITE_BUSY_TIMEOUT_MS = 120000
# 批量刷新友好的连接级调优（均为连接局部或不写库文件，不扰动 user_version/mtime）：
#   cache_size 负值=KB，每连接一块页缓存减少大表反复读盘（多线程下 N 连接各占一份，故默认偏保守 16MB）；
#   wal_autocheckpoint 调大，批量写时少做 checkpoint 抖动（连接关闭/末尾自然收尾）。env 可覆盖。
SQLITE_CACHE_SIZE_KB = -int(os.getenv("STOCK_SQLITE_CACHE_MB", "16")) * 1024
SQLITE_WAL_AUTOCHECKPOINT = int(os.getenv("STOCK_SQLITE_WAL_AUTOCHECKPOINT", "4000"))

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

_CONNECT_LOCK = threading.Lock()
_WRITE_LOCK = threading.Lock()


def connect(db_file: Path | str = DEFAULT_DB_FILE) -> sqlite3.Connection:
    """打开个股缓存库并确保 schema 存在。

    SQLite 只能单写。爬虫多线程会把整只股票的多年 history 作为一个事务写入，
    所以这里把 busy_timeout 拉长，并串行化文件库初始化，避免大量线程同时设置 WAL。
    """
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
            # 让 strategies 那步对 ~300 万行的 ORDER BY/GROUP BY 临时结果走内存而非落盘临时文件。
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute(f"PRAGMA cache_size = {SQLITE_CACHE_SIZE_KB}")
            conn.execute(f"PRAGMA wal_autocheckpoint = {SQLITE_WAL_AUTOCHECKPOINT}")
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

        -- 申万三级"行业→成分股"归属缓存（对应旧 data/capital/sw3_membership.json）。
        -- 1股=1三级行业，故 sw3_member.code 全表唯一可做主键，反查个股→行业为 PK 命中。
        CREATE TABLE IF NOT EXISTS sw3_segment (
            segment_code   TEXT PRIMARY KEY,
            segment_name   TEXT,
            parent_segment TEXT,
            member_count   INTEGER,
            refreshed_at   TEXT,
            error          TEXT,
            updated_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sw3_member (
            code               TEXT PRIMARY KEY,
            segment_code       TEXT NOT NULL,
            name               TEXT,
            price              REAL,
            market_cap_yi      REAL,
            official_market_cap_ratio REAL,
            roe_pct            REAL,
            profit_growth_pct  REAL,
            revenue_growth_pct REAL,
            is_leader          INTEGER NOT NULL DEFAULT 0,
            is_hot_money       INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (segment_code) REFERENCES sw3_segment (segment_code) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_sw3_member_segment
        ON sw3_member (segment_code);

        -- A股短线策略的龙虎榜/席位/技术面信号快照。池成员资格只由
        -- sw3_member.is_hot_money 决定，本表仅提供可缺省的打分补充数据。
        CREATE TABLE IF NOT EXISTS short_signal_snapshot (
            code         TEXT PRIMARY KEY,
            generated_at TEXT NOT NULL,
            as_of_date   TEXT,
            payload_json TEXT NOT NULL,
            updated_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (code) REFERENCES sw3_member (code) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_short_signal_generated_at
        ON short_signal_snapshot (generated_at);
        """
    )
    _ensure_table_columns(conn, "sw3_member", {
        "official_market_cap_ratio": "REAL",
        # 细分龙头标记：build_segment_leader_pool 选出龙头后回写(见 mark_sw3_leaders)，
        # 供主力雷达等模块直接 WHERE is_leader=1 取候选池，无需再解析 segment_leader_pool.json。
        "is_leader": "INTEGER NOT NULL DEFAULT 0",
        # 游资小盘池是游资雷达与 A 股短线策略共用的唯一成员口径。
        "is_hot_money": "INTEGER NOT NULL DEFAULT 0",
    })
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def _ensure_table_columns(conn: sqlite3.Connection, table: str, columns: Mapping[str, str]) -> None:
    """Add nullable columns for older SQLite files without rebuilding hot tables."""
    existing = {
        row["name"] if isinstance(row, sqlite3.Row) else row[1]
        for row in conn.execute(f"PRAGMA table_info({table})")
    }
    for name, declaration in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


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

    with _WRITE_LOCK:
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


def upsert_history_records(
    conn: sqlite3.Connection,
    code: str,
    name: str,
    records: Sequence[Mapping[str, Any]],
    *,
    source: str = "stock_history_upsert",
    price_adjust: str = "qfq",
    daily_stats: Optional[Mapping[str, Any]] = None,
) -> int:
    """只增量写入 stock_history，并保留 stock_meta 里的财报/指标等 JSON blob。

    适合短线/雷达这类临时补 K 线场景：不需要构造整只股票 payload，也不能因为补日线
    把已有 financials_json / indicators_json / dividends_json 覆盖成空。
    daily_stats 给定时一并刷新 daily_stats_json（价格爬虫增量快路用，避免读回整只再 dump 财报 blob）。
    market_cap 不再逐只同步到 sw3_member——见 sync_sw3_member_market_caps（爬完批量跑一次）。
    """
    code = _normalize_code(code)
    if not code:
        return 0
    rows = _sorted_records(normalize_history_records(records, include_valuation=True))
    if not rows:
        return 0

    start_date = rows[0]["date"]
    end_date = rows[-1]["date"]
    now = datetime.now().isoformat()
    display_name = str(name or code)

    with _WRITE_LOCK:
        with conn:
            existing = conn.execute(
                "SELECT history_start_date, history_end_date FROM stock_meta WHERE code = ?",
                (code,),
            ).fetchone()
            if existing:
                old_start = _optional_text(existing["history_start_date"])
                old_end = _optional_text(existing["history_end_date"])
                merged_start = min([d for d in (old_start, start_date) if d])
                merged_end = max([d for d in (old_end, end_date) if d])
                conn.execute(
                    """
                    UPDATE stock_meta
                    SET name = ?,
                        history_refetched_at = ?,
                        history_source = ?,
                        history_start_date = ?,
                        history_end_date = ?,
                        price_adjust = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE code = ?
                    """,
                    (display_name, now, source, merged_start, merged_end, price_adjust, code),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO stock_meta
                    (code, name, fetch_time, history_refetched_at, history_source,
                     history_start_date, history_end_date, price_adjust, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (code, display_name, now, now, source, start_date, end_date, price_adjust),
                )
            conn.executemany(
                _HISTORY_INSERT_SQL,
                [_history_value_tuple(code, row) for row in rows],
            )
            if daily_stats:
                conn.execute(
                    "UPDATE stock_meta SET daily_stats_json = ? WHERE code = ?",
                    (_json_or_none(daily_stats), code),
                )
    return len(rows)


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

    with _WRITE_LOCK:
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


# ─── 申万三级归属缓存 (sw3 membership) ─────────────────────────

_SW3_MEMBER_METRIC_COLS = (
    "price",
    "market_cap_yi",
    "official_market_cap_ratio",
    "roe_pct",
    "profit_growth_pct",
    "revenue_growth_pct",
)


def _sw3_member_metric(mem: Mapping[str, Any], col: str) -> Optional[float]:
    value = mem.get(col)
    if col == "official_market_cap_ratio" and value is None:
        value = mem.get("index_weight")
    return safe_float(value)


def save_sw3_membership(conn: sqlite3.Connection, payload: Mapping[str, Any]) -> Dict[str, int]:
    """把 sw3_membership payload 全量落库（sw3_segment + sw3_member 两表替换）。

    一个赛道可同时出现在 segments(带 members) 与 errors(上次刷新失败)——后者只在对应
    segment 行的 error 列打标、不丢 members；纯失败赛道补一行无 member 的 segment。
    全量 DELETE + 重插，匹配旧 json"整文件重写"语义(自然丢弃消失赛道)。返回写入计数。
    """
    segments = payload.get("segments") or []
    errors = payload.get("errors") or []
    error_by_code = {e.get("segment_code"): e.get("error")
                     for e in errors if e.get("segment_code")}

    seg_rows: List[tuple] = []
    member_rows: List[tuple] = []
    seen: set = set()
    seen_members: set = set()
    for seg in segments:
        sc = _optional_text(seg.get("segment_code"))
        if not sc:
            continue
        seen.add(sc)
        seg_rows.append((
            sc,
            _optional_text(seg.get("segment_name")),
            _optional_text(seg.get("parent_segment")),
            seg.get("member_count"),
            _optional_text(seg.get("refreshed_at")),
            error_by_code.get(sc),
        ))
        for mem in seg.get("members") or []:
            code = _normalize_code(mem.get("code"))
            # 1股=1三级行业(code 唯一)；防御 legulegu 偶发重复，保第一次出现、不让整批崩
            if not code or code in seen_members:
                continue
            seen_members.add(code)
            member_rows.append(
                (code, sc, _optional_text(mem.get("name")))
                + tuple(_sw3_member_metric(mem, col) for col in _SW3_MEMBER_METRIC_COLS)
            )
    # 纯失败赛道(不在 segments 里)：补一行无 member 的 segment 以保留失败标记
    for err in errors:
        sc = _optional_text(err.get("segment_code"))
        if sc and sc not in seen:
            seen.add(sc)
            seg_rows.append((sc, _optional_text(err.get("segment_name")), None, None, None, err.get("error")))

    member_cols = ("code", "segment_code", "name") + _SW3_MEMBER_METRIC_COLS
    with conn:
        conn.execute("DELETE FROM sw3_member")
        conn.execute("DELETE FROM sw3_segment")
        conn.executemany(
            "INSERT INTO sw3_segment "
            "(segment_code, segment_name, parent_segment, member_count, refreshed_at, error, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            seg_rows,
        )
        if member_rows:
            conn.executemany(
                f"INSERT INTO sw3_member ({', '.join(member_cols)}) "
                f"VALUES ({', '.join('?' for _ in member_cols)})",
                member_rows,
            )
    return {"segments": len(seg_rows), "members": len(member_rows)}


def load_sw3_membership(conn: sqlite3.Connection, max_age_days: Optional[int] = 30) -> Optional[Dict[str, Any]]:
    """从 DB 拼回 sw3_membership dict（结构与旧 json 等价，下游零改动）。

    generated_at 取所有赛道 refreshed_at 的最大值；max_age_days>0 时按它做 TTL，
    过期返回 None（触发上层全量重爬）。库里无任何赛道返回 None。
    """
    seg_rows = conn.execute(
        "SELECT segment_code, segment_name, parent_segment, member_count, refreshed_at, error "
        "FROM sw3_segment ORDER BY segment_code"
    ).fetchall()
    if not seg_rows:
        return None

    members_by_seg: Dict[str, List[Dict[str, Any]]] = {}
    for row in conn.execute(
        "SELECT code, segment_code, name, price, market_cap_yi, official_market_cap_ratio, "
        "roe_pct, profit_growth_pct, revenue_growth_pct FROM sw3_member ORDER BY segment_code, code"
    ):
        official_ratio = row["official_market_cap_ratio"]
        members_by_seg.setdefault(row["segment_code"], []).append({
            "code": row["code"],
            "name": row["name"],
            "price": row["price"],
            "market_cap_yi": row["market_cap_yi"],
            "official_market_cap_ratio": official_ratio,
            "index_weight": official_ratio,
            "roe_pct": row["roe_pct"],
            "profit_growth_pct": row["profit_growth_pct"],
            "revenue_growth_pct": row["revenue_growth_pct"],
        })

    segments: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    refreshed_times: List[str] = []
    for row in seg_rows:
        sc = row["segment_code"]
        members = members_by_seg.get(sc, [])
        if members:
            segments.append({
                "segment_code": sc,
                "segment_name": row["segment_name"],
                "parent_segment": row["parent_segment"],
                "member_count": row["member_count"],
                "members": members,
                "refreshed_at": row["refreshed_at"],
            })
            if row["refreshed_at"]:
                refreshed_times.append(row["refreshed_at"])
        if row["error"]:
            errors.append({"segment_code": sc, "segment_name": row["segment_name"], "error": row["error"]})

    if not segments:
        return None
    generated_at = max(refreshed_times) if refreshed_times else None
    if max_age_days and max_age_days > 0 and generated_at:
        try:
            gen = datetime.strptime(generated_at, "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - gen).days > max_age_days:
                return None
        except ValueError:
            pass
    return {
        "schema": "sw3_membership.v1",
        "generated_at": generated_at,
        "source": "申万三级行业成分(Legulegu, SQLite缓存)",
        "segment_count": len(segments),
        "segments": segments,
        "errors": errors,
    }


def stock_segment(conn: sqlite3.Connection, code: str) -> Optional[Dict[str, Any]]:
    """个股 → 申万三级行业(+二级父行业)反查。命中一行(1股=1赛道)，无则 None。"""
    row = conn.execute(
        "SELECT m.segment_code, s.segment_name, s.parent_segment "
        "FROM sw3_member m JOIN sw3_segment s ON s.segment_code = m.segment_code "
        "WHERE m.code = ?",
        (_normalize_code(code),),
    ).fetchone()
    if row is None:
        return None
    return {
        "segment_code": row["segment_code"],
        "segment_name": row["segment_name"],
        "parent_segment": row["parent_segment"],
    }


def segment_map(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    """全量 {code: {segment_code, segment_name, parent_segment}}，供批量打标/反查。"""
    out: Dict[str, Dict[str, Any]] = {}
    for row in conn.execute(
        "SELECT m.code, m.segment_code, s.segment_name, s.parent_segment "
        "FROM sw3_member m JOIN sw3_segment s ON s.segment_code = m.segment_code"
    ):
        out[row["code"]] = {
            "segment_code": row["segment_code"],
            "segment_name": row["segment_name"],
            "parent_segment": row["parent_segment"],
        }
    return out


def mark_sw3_leaders(conn: sqlite3.Connection, codes: Sequence[str]) -> int:
    """把给定 code 标记为细分龙头(is_leader=1)，其余全部清零。

    供 build_segment_leader_pool 选出龙头后回写：save_sw3_membership 会 DELETE 重插
    sw3_member（is_leader 落回默认 0），故每次重建龙头池后调用本函数重新打标，
    让 is_leader 反映最新一轮选股结果。返回实际打标命中的行数。
    """
    norm = sorted({_normalize_code(c) for c in codes if _normalize_code(c)})
    with _WRITE_LOCK:
        with conn:
            conn.execute("UPDATE sw3_member SET is_leader = 0 WHERE is_leader = 1")
            updated = 0
            for i in range(0, len(norm), 500):
                chunk = norm[i:i + 500]
                placeholders = ",".join("?" for _ in chunk)
                cur = conn.execute(
                    f"UPDATE sw3_member SET is_leader = 1 WHERE code IN ({placeholders})", chunk
                )
                updated += cur.rowcount
    return updated


def _latest_history_market_caps(conn: sqlite3.Connection, codes: Sequence[str]) -> Dict[str, float]:
    """取指定股票在 stock_history 中最新一条非空总市值(亿元)。"""
    norm = sorted({_normalize_code(c) for c in codes if _normalize_code(c)})
    out: Dict[str, float] = {}
    for i in range(0, len(norm), 500):
        chunk = norm[i:i + 500]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT h.code, h.market_cap
            FROM stock_history h
            JOIN (
                SELECT code, MAX(date) AS date
                FROM stock_history
                WHERE market_cap IS NOT NULL AND code IN ({placeholders})
                GROUP BY code
            ) latest ON latest.code = h.code AND latest.date = h.date
            """,
            chunk,
        ).fetchall()
        for row in rows:
            cap = safe_float(row["market_cap"])
            if cap is not None:
                out[row["code"]] = cap
    return out


def sync_sw3_member_market_caps(
    conn: sqlite3.Connection, codes: Optional[Sequence[str]] = None
) -> int:
    """批量把 sw3_member.market_cap_yi 刷成各股 stock_history 最新非空总市值(亿元)。

    取代旧的「每次 save_stock/upsert 都查 stock_history 大表同步单只」热路开销：整轮爬完后
    调一次即可（_latest_history_market_caps 已按 500 分批，全量同步只需 ~N/500 条查询）。
    codes=None 同步全部 sw3_member；读端 pool_members 仍对残留 NULL 做兜底。返回写入的行数。
    """
    if codes is None:
        codes = [row["code"] for row in conn.execute("SELECT code FROM sw3_member")]
    norm = sorted({_normalize_code(c) for c in codes if _normalize_code(c)})
    if not norm:
        return 0
    caps = _latest_history_market_caps(conn, norm)
    if not caps:
        return 0
    with _WRITE_LOCK:
        with conn:
            conn.executemany(
                "UPDATE sw3_member SET market_cap_yi = ? WHERE code = ?",
                [(cap, code) for code, cap in caps.items()],
            )
    return len(caps)


_POOL_FLAG_COL = {"leader": "is_leader", "hotmoney": "is_hot_money"}


def _sw3_has_column(conn: sqlite3.Connection, col: str) -> bool:
    return any(r[1] == col for r in conn.execute("PRAGMA table_info(sw3_member)").fetchall())


def pool_members(conn: sqlite3.Connection, pool: str = "leader") -> List[Dict[str, Any]]:
    """指定池成分：pool='leader' 细分龙头(is_leader) / 'hotmoney' 游资小盘universe(is_hot_money)。

    市值优先取 sw3_member，缺失时用 stock_history 最新非空 market_cap 回补。
    若池标记列尚未建(如 is_hot_money 未跑建池脚本)→ 返回空池。
    """
    flag = _POOL_FLAG_COL.get(pool, "is_leader")
    if not _sw3_has_column(conn, flag):
        return []
    rows = conn.execute(
        f"SELECT m.code, m.name, m.market_cap_yi, m.segment_code, s.segment_name, s.parent_segment "
        f"FROM sw3_member m LEFT JOIN sw3_segment s ON s.segment_code = m.segment_code "
        f"WHERE m.{flag} = 1 ORDER BY m.segment_code, m.code"
    ).fetchall()
    leaders = [
        {
            "code": row["code"],
            "name": row["name"],
            "market_cap_yi": row["market_cap_yi"],
            "segment_code": row["segment_code"],
            "segment_name": row["segment_name"],
            "parent_segment": row["parent_segment"],
        }
        for row in rows
    ]
    missing = [item["code"] for item in leaders if item["market_cap_yi"] is None]
    if missing:
        history_caps = _latest_history_market_caps(conn, missing)
        for item in leaders:
            if item["market_cap_yi"] is None:
                item["market_cap_yi"] = history_caps.get(item["code"])
    return leaders


def leader_members(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """细分龙头(is_leader=1)成分——pool_members(pool='leader') 的兼容别名。"""
    return pool_members(conn, "leader")


def pool_signature(conn: sqlite3.Connection, pool: str = "leader") -> Dict[str, Any]:
    """返回池成员的稳定签名；标记变化会使候选缓存立即失效。"""
    flag = _POOL_FLAG_COL.get(pool, "is_leader")
    if not _sw3_has_column(conn, flag):
        return {"pool": pool, "count": 0, "code_digest": None}
    codes = [
        str(row["code"]).zfill(6)
        for row in conn.execute(
            f"SELECT code FROM sw3_member WHERE {flag} = 1 ORDER BY code"
        )
    ]
    digest = hashlib.sha256("\n".join(codes).encode("utf-8")).hexdigest()[:20]
    return {"pool": pool, "count": len(codes), "code_digest": digest}


def replace_short_signals(
    conn: sqlite3.Connection,
    signals: Mapping[str, Mapping[str, Any]],
    *,
    generated_at: Optional[str] = None,
    as_of_date: Optional[str] = None,
) -> int:
    """原子替换短线信号快照；调用方必须先确认本轮数据源不是整体失败。

    ``signals`` 只承载打分补充数据，不会写入或改变 ``is_hot_money`` 成员标记。
    """
    generated_at = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for raw_code, raw_payload in signals.items():
        code = _normalize_code(raw_code)
        if not code or not isinstance(raw_payload, Mapping):
            continue
        payload = dict(raw_payload)
        payload["code"] = code
        rows.append(
            (
                code,
                generated_at,
                as_of_date,
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            )
        )
    rows.sort(key=lambda row: row[0])
    with _WRITE_LOCK:
        with conn:
            conn.execute("DELETE FROM short_signal_snapshot")
            conn.executemany(
                """
                INSERT INTO short_signal_snapshot
                    (code, generated_at, as_of_date, payload_json, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                rows,
            )
    return len(rows)


def load_short_signals(
    conn: sqlite3.Connection,
    codes: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """读取短线补充信号，返回 ``{code: payload}``；坏行按缺失信号处理。"""
    normalized = sorted({_normalize_code(code) for code in (codes or []) if _normalize_code(code)})
    if codes is not None and not normalized:
        return {}
    if codes is None:
        rows = conn.execute(
            "SELECT code, generated_at, as_of_date, payload_json FROM short_signal_snapshot"
        ).fetchall()
    else:
        rows = []
        for start in range(0, len(normalized), 500):
            chunk = normalized[start:start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(
                conn.execute(
                    f"SELECT code, generated_at, as_of_date, payload_json "
                    f"FROM short_signal_snapshot WHERE code IN ({placeholders})",
                    chunk,
                ).fetchall()
            )
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        code = str(row["code"]).zfill(6)
        payload["code"] = code
        payload.setdefault("generated_at", row["generated_at"])
        payload.setdefault("as_of_date", row["as_of_date"])
        out[code] = payload
    return out


def _short_signal_signature_from_conn(conn: sqlite3.Connection) -> Dict[str, Any]:
    rows = conn.execute(
        "SELECT code, generated_at, as_of_date, payload_json "
        "FROM short_signal_snapshot ORDER BY code"
    ).fetchall()
    digest_source = "\n".join(
        f"{row['code']}|{row['generated_at']}|{row['as_of_date'] or ''}|{row['payload_json']}"
        for row in rows
    )
    return {
        "count": len(rows),
        "generated_at": max((row["generated_at"] for row in rows), default=None),
        "as_of_date": max((row["as_of_date"] or "" for row in rows), default=None) or None,
        "payload_digest": hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:20],
    }


def short_signal_status(conn: sqlite3.Connection) -> Dict[str, Any]:
    """复用现有连接读取短线信号覆盖与新鲜度。"""
    return _short_signal_signature_from_conn(conn)


def short_signal_signature(db_file: Path | str = DEFAULT_DB_FILE) -> Dict[str, Any]:
    """短线补充信号的稳定内容签名，供策略候选缓存校验。"""
    conn = connect(db_file)
    try:
        return _short_signal_signature_from_conn(conn)
    finally:
        conn.close()


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


def _sw3_signature_from_conn(conn: sqlite3.Connection) -> Dict[str, Any]:
    seg = conn.execute(
        "SELECT COUNT(*) AS c, MAX(updated_at) AS m FROM sw3_segment"
    ).fetchone()
    return {
        "sw3_segments": int(seg["c"]) if seg else 0,
        "sw3_members": table_count(conn, "sw3_member"),
        "sw3_max_updated_at": seg["m"] if seg else None,
    }


def sw3_signature(db_file: Path | str = DEFAULT_DB_FILE) -> Dict[str, Any]:
    """申万行业归属缓存签名；供消费方高效判断 sw3 映射是否需要重读。"""
    conn = connect(db_file)
    try:
        return _sw3_signature_from_conn(conn)
    finally:
        conn.close()


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


def codes_needing_history_cleanup(conn: sqlite3.Connection) -> List[str]:
    """有 snapshot-only/空行(daily_open 为空)需要 prune 清洗的 code。

    正常日线 bar 的 daily_open 必非空；snapshot-only 与空行都满足 daily_open IS NULL，
    故这是 prunable 行的安全超集——用它替代'全库逐只 load+prune'的扫描(多数情况返回空)。
    """
    return [
        row["code"]
        for row in conn.execute(
            "SELECT DISTINCT code FROM stock_history WHERE daily_open IS NULL ORDER BY code"
        )
    ]


def stock_exists(conn: sqlite3.Connection, code: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM stock_meta WHERE code = ? LIMIT 1", (_normalize_code(code),)
    ).fetchone() is not None


def load_history_records(conn: sqlite3.Connection, code: str) -> List[Dict[str, Any]]:
    """只取该 code 的日线记录（daily_* + 估值列），不解析财报 blob。"""
    return _load_history_records(conn, _normalize_code(code))


def load_recent_history_records(
    conn: sqlite3.Connection,
    code: str,
    *,
    limit: int,
    start_date: Optional[str] = None,
    require_ohlcv: bool = False,
) -> List[Dict[str, Any]]:
    """按 (code,date) 主键直接取最近 N 根日线，避免先读全历史再切片。"""
    code = _normalize_code(code)
    if not code or limit <= 0:
        return []
    where = ["code = ?"]
    params: List[Any] = [code]
    if start_date:
        where.append("date >= ?")
        params.append(str(start_date))
    if require_ohlcv:
        where.extend(f"{field} IS NOT NULL" for field in HISTORY_DAILY_OHLCV_FIELDS)
    params.append(int(limit))
    rows = conn.execute(
        f"""
        SELECT date, {', '.join(HISTORY_COLUMNS)}
        FROM stock_history
        WHERE {' AND '.join(where)}
        ORDER BY date DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        {"date": row["date"], **{col: row[col] for col in HISTORY_COLUMNS}}
        for row in reversed(rows)
    ]


def load_recent_news(conn: sqlite3.Connection, code: str, *,
                     since: Optional[str] = None, until: Optional[str] = None,
                     limit: int = 50) -> List[Dict[str, Any]]:
    """取该 code 近端新闻(降序)；stock_news 表由 stock_crawl_news.py 建，缺表返回 []。

    since/until 按 pub_time 过滤（until 供 as-of PIT 防泄漏，只取该时点前的新闻）。
    """
    code = _normalize_code(code)
    if not code:
        return []
    where = ["code = ?"]
    params: List[Any] = [code]
    if since:
        where.append("pub_time >= ?")
        params.append(str(since))
    if until:
        where.append("pub_time <= ?")
        params.append(str(until))
    params.append(int(limit))
    try:
        rows = conn.execute(
            f"SELECT pub_time, title, source, url, keyword, themes FROM stock_news "
            f"WHERE {' AND '.join(where)} ORDER BY pub_time DESC LIMIT ?", params,
        ).fetchall()
    except sqlite3.OperationalError:
        return []   # 新闻表尚未建立（未爬过新闻）
    return [dict(r) for r in rows]


def history_refetched_at(conn: sqlite3.Connection, code: str) -> Optional[str]:
    row = conn.execute(
        "SELECT history_refetched_at FROM stock_meta WHERE code = ?", (_normalize_code(code),)
    ).fetchone()
    return row["history_refetched_at"] if row else None


def financials_refetched_map(
    conn: sqlite3.Connection, codes: Optional[Sequence[str]] = None
) -> Dict[str, Optional[str]]:
    """{code: financials_refetched_at}（含值为 None 的，便于判断"从没爬过财报"）。

    codes 给定时只查这些(分批 IN 查询，避免 SQL 变量上限)，库里没有的 code 也回填 None；
    不给则全表。供龙头池财报增量刷新在进线程池前一次性读取新鲜度。
    """
    if codes is None:
        return {
            row["code"]: row["financials_refetched_at"]
            for row in conn.execute("SELECT code, financials_refetched_at FROM stock_meta")
        }
    norm = [_normalize_code(c) for c in codes if c]
    out: Dict[str, Optional[str]] = {}
    for i in range(0, len(norm), 500):
        chunk = norm[i:i + 500]
        placeholders = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"SELECT code, financials_refetched_at FROM stock_meta WHERE code IN ({placeholders})",
            chunk,
        ):
            out[row["code"]] = row["financials_refetched_at"]
    for code in norm:
        out.setdefault(code, None)
    return out


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
    """缓存版本签名：股票数/最新 updated_at/日线行数/申万行业归属/库文件大小。"""
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
        sw3 = _sw3_signature_from_conn(conn)
    finally:
        conn.close()
    return {
        "count": int(meta["c"]) if meta else 0,
        "max_updated_at": meta["m"] if meta else None,
        "history_rows": history_rows,
        **sw3,
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
