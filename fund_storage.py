from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


DATA_DIR = Path("data")
DEFAULT_DB_FILE = DATA_DIR / "fund_data.sqlite3"
SCHEMA_VERSION = 1


def connect(db_file: Path | str = DEFAULT_DB_FILE) -> sqlite3.Connection:
    """Open the fund cache database and ensure the core schema exists."""
    db_target = Path(db_file)
    if str(db_file) != ":memory:":
        db_target.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_target))
    else:
        conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS fund_nav_records (
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            nav TEXT,
            nav_acc TEXT,
            daily_growth_rate TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (code, date)
        );

        CREATE INDEX IF NOT EXISTS idx_fund_nav_records_code_date
        ON fund_nav_records (code, date);

        CREATE TABLE IF NOT EXISTS fund_nav_meta (
            code TEXT PRIMARY KEY,
            start_date TEXT,
            end_date TEXT,
            record_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS fund_realtime_estimates (
            code TEXT PRIMARY KEY,
            gsz TEXT,
            gszzl TEXT,
            gztime TEXT,
            dwjz TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS fund_profile_snapshots (
            code TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            manager_trigger TEXT,
            manager_total_asset TEXT,
            raw_json TEXT NOT NULL,
            source_updated_at TEXT,
            imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def save_nav_entry(conn: sqlite3.Connection, code: str, entry: Mapping[str, Any]) -> None:
    records = sorted(
        [record for record in entry.get("records", []) if isinstance(record, Mapping) and record.get("date")],
        key=lambda item: str(item["date"]),
    )
    with conn:
        conn.execute("DELETE FROM fund_nav_records WHERE code = ?", (code,))
        if records:
            conn.executemany(
                """
                INSERT INTO fund_nav_records
                    (code, date, nav, nav_acc, daily_growth_rate, updated_at)
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

        start_date = _optional_text(entry.get("start_date")) or (str(records[0]["date"]) if records else None)
        end_date = _optional_text(entry.get("end_date")) or (str(records[-1]["date"]) if records else None)
        conn.execute(
            """
            INSERT INTO fund_nav_meta (code, start_date, end_date, record_count, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(code) DO UPDATE SET
                start_date = excluded.start_date,
                end_date = excluded.end_date,
                record_count = excluded.record_count,
                updated_at = CURRENT_TIMESTAMP
            """,
            (code, start_date, end_date, len(records)),
        )


def load_nav_entry(conn: sqlite3.Connection, code: str) -> Dict[str, Any]:
    rows = conn.execute(
        """
        SELECT date, nav, nav_acc, daily_growth_rate
        FROM fund_nav_records
        WHERE code = ?
        ORDER BY date
        """,
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
    meta = conn.execute(
        "SELECT start_date, end_date FROM fund_nav_meta WHERE code = ?",
        (code,),
    ).fetchone()
    return {
        "start_date": meta["start_date"] if meta else records[0]["date"],
        "end_date": meta["end_date"] if meta else records[-1]["date"],
        "records": records,
    }


def load_nav_store(conn: sqlite3.Connection) -> Dict[str, Any]:
    codes = [
        row["code"]
        for row in conn.execute(
            """
            SELECT code FROM fund_nav_meta
            UNION
            SELECT DISTINCT code FROM fund_nav_records
            ORDER BY code
            """
        )
    ]
    return {code: entry for code in codes if (entry := load_nav_entry(conn, code))}


def load_nav_history(conn: sqlite3.Connection) -> Dict[str, List[Dict[str, Any]]]:
    return {code: entry["records"] for code, entry in load_nav_store(conn).items()}


def import_nav_store_payload(conn: sqlite3.Connection, store: Mapping[str, Any]) -> int:
    count = 0
    for code, entry in store.items():
        if not isinstance(entry, Mapping):
            continue
        save_nav_entry(conn, str(code), entry)
        count += 1
    return count


def save_realtime_estimates(
    conn: sqlite3.Connection,
    estimates: Mapping[str, Mapping[str, Any]],
    *,
    replace: bool = False,
    expected_codes: Optional[Iterable[str]] = None,
) -> None:
    existing = {
        row["code"]: dict(row)
        for row in conn.execute(
            "SELECT code, gsz, gszzl, gztime, dwjz FROM fund_realtime_estimates"
        )
    }
    rows = []
    for code, estimate in estimates.items():
        if not isinstance(estimate, Mapping):
            continue
        code = str(code)
        incoming = {
            field: _nonempty_text(estimate.get(field))
            for field in ("gsz", "gszzl", "gztime", "dwjz")
        }
        # A syntactically valid JSONP response can contain only empty strings.
        # It is not a successful observation and must not create/replace a row.
        if not any(value is not None for value in incoming.values()):
            continue
        previous = existing.get(code, {})
        rows.append(
            (
                code,
                incoming["gsz"] or previous.get("gsz"),
                incoming["gszzl"] or previous.get("gszzl"),
                incoming["gztime"] or previous.get("gztime"),
                incoming["dwjz"] or previous.get("dwjz"),
            )
        )
    if replace:
        _validate_complete_snapshot(
            {row[0] for row in rows},
            expected_codes,
            snapshot_name="基金实时估值",
        )

    with conn:
        if replace:
            conn.execute("DELETE FROM fund_realtime_estimates")
        conn.executemany(
            """
            INSERT INTO fund_realtime_estimates
                (code, gsz, gszzl, gztime, dwjz, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(code) DO UPDATE SET
                gsz = COALESCE(excluded.gsz, fund_realtime_estimates.gsz),
                gszzl = COALESCE(excluded.gszzl, fund_realtime_estimates.gszzl),
                gztime = COALESCE(excluded.gztime, fund_realtime_estimates.gztime),
                dwjz = COALESCE(excluded.dwjz, fund_realtime_estimates.dwjz),
                updated_at = CURRENT_TIMESTAMP
            """,
            rows,
        )


def load_realtime_estimates(conn: sqlite3.Connection) -> Dict[str, Dict[str, str]]:
    rows = conn.execute(
        """
        SELECT code, gsz, gszzl, gztime, dwjz
        FROM fund_realtime_estimates
        ORDER BY code
        """
    ).fetchall()
    return {
        row["code"]: {
            "gsz": row["gsz"] or "",
            "gszzl": row["gszzl"] or "",
            "gztime": row["gztime"] or "",
            "dwjz": row["dwjz"] or "",
        }
        for row in rows
    }


def save_profile_snapshots(
    conn: sqlite3.Connection,
    items: Iterable[Mapping[str, Any]],
    *,
    replace: bool = False,
    expected_codes: Optional[Iterable[str]] = None,
    source_updated_at: Optional[str] = None,
) -> int:
    source_updated_at = source_updated_at or _now_text()
    existing = {}
    for row in conn.execute("SELECT code, raw_json FROM fund_profile_snapshots"):
        try:
            payload = json.loads(row["raw_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            existing[row["code"]] = dict(payload)

    rows = []
    for item in items:
        if not isinstance(item, Mapping) or not item.get("fundCode"):
            continue
        code = str(item["fundCode"])
        merged = _merge_nonblank_snapshot(existing.get(code, {}), dict(item))
        merged["fundCode"] = code
        rows.append(
            (
                code,
                str(merged.get("name") or ""),
                _nonempty_text(merged.get("managerTrigger")),
                _nonempty_text(merged.get("fund_manager_total_asset")),
                json.dumps(merged, ensure_ascii=False, separators=(",", ":")),
                source_updated_at,
            )
        )

    if replace:
        _validate_complete_snapshot(
            {row[0] for row in rows},
            expected_codes,
            snapshot_name="基金概况",
        )

    with conn:
        if replace:
            conn.execute("DELETE FROM fund_profile_snapshots")
        conn.executemany(
            """
            INSERT INTO fund_profile_snapshots
                (code, name, manager_trigger, manager_total_asset, raw_json, source_updated_at, imported_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(code) DO UPDATE SET
                name = excluded.name,
                manager_trigger = excluded.manager_trigger,
                manager_total_asset = excluded.manager_total_asset,
                raw_json = excluded.raw_json,
                source_updated_at = excluded.source_updated_at,
                imported_at = CURRENT_TIMESTAMP
            """,
            rows,
        )
    return len(rows)


def load_profile_snapshots(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT code, raw_json
        FROM fund_profile_snapshots
        ORDER BY code
        """
    ).fetchall()
    return {row["code"]: json.loads(row["raw_json"]) for row in rows}


def _table_count(conn: sqlite3.Connection, table_name: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
    return int(row["count"]) if row else 0


def _validate_complete_snapshot(
    actual_codes: set[str],
    expected_codes: Optional[Iterable[str]],
    *,
    snapshot_name: str,
) -> None:
    if expected_codes is None:
        raise ValueError(f"{snapshot_name}全量替换必须提供 expected_codes")

    expected = {str(code) for code in expected_codes}
    if actual_codes == expected:
        return

    missing = sorted(expected - actual_codes)
    unexpected = sorted(actual_codes - expected)
    details = []
    if missing:
        details.append(f"缺少 {len(missing)} 只: {', '.join(missing[:10])}")
    if unexpected:
        details.append(f"多出 {len(unexpected)} 只: {', '.join(unexpected[:10])}")
    raise ValueError(f"{snapshot_name}数据不完整，拒绝全量替换（{'；'.join(details)}）")


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _nonempty_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _merge_nonblank_snapshot(
    previous: Mapping[str, Any], incoming: Mapping[str, Any]
) -> Dict[str, Any]:
    """Merge a partial crawl result without erasing earlier successful fields."""
    merged = dict(incoming)
    for key, old_value in previous.items():
        if key not in merged or _is_blank_snapshot_value(merged[key]):
            merged[key] = old_value
        elif isinstance(old_value, Mapping) and isinstance(merged[key], Mapping):
            merged[key] = _merge_nonblank_snapshot(old_value, merged[key])
    return merged


def _is_blank_snapshot_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return not value
    return False


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
