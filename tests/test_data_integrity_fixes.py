import sys
import tempfile
import time
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import plate_crawl_history
import plate_storage
import stock_crawl_fundamentals
import stock_crawl_common
import stock_crawl_price_valuation
import stock_crawl_segment_leaders
import stock_data_refresh
import stock_storage
from stock_crawl_common import history_payload_from_records, latest_weekday_date


def sw2_row(code, name, trade_date, close=100.0):
    return {
        "swindexcode": code,
        "swindexname": name,
        "bargaindate": trade_date,
        "closeindex": close,
        "bargainamount": 1.0,
    }


def complete_daily_rows(start, count):
    return [
        {
            "date": (start + timedelta(days=i)).strftime("%Y-%m-%d"),
            "daily_open": 10 + i,
            "daily_high": 11 + i,
            "daily_low": 9 + i,
            "daily_close": 10.5 + i,
            "daily_volume": 1000 + i,
            "daily_amount": 10000 + i,
        }
        for i in range(count)
    ]


def valid_fundamentals():
    return {
        "financials": {
            "income": [{"date": "2025-12-31", "revenue": 1}],
            "balance": [{"date": "2025-12-31", "total_equity": 1}],
            "cashflow": [{"date": "2025-12-31", "operating_cashflow_net": 1}],
        },
        "indicators": {"records": [{"date": "2025-12-31", "roe": 10}]},
        "dividends": {"records": []},
    }


class Sw3MembershipIntegrityTest(unittest.TestCase):
    def test_taxonomy_truncation_guard_ignores_tiny_fixture_but_protects_real_table(self):
        self.assertFalse(
            stock_crawl_segment_leaders._taxonomy_response_looks_truncated(2, 3)
        )
        self.assertTrue(
            stock_crawl_segment_leaders._taxonomy_response_looks_truncated(80, 100)
        )
        self.assertFalse(
            stock_crawl_segment_leaders._taxonomy_response_looks_truncated(81, 100)
        )

    def test_refresh_preserves_flags_and_signals_for_retained_members(self):
        conn = stock_storage.connect(":memory:")
        try:
            stock_storage.save_sw3_membership(conn, {
                "segments": [{
                    "segment_code": "850111",
                    "segment_name": "旧行业名",
                    "members": [
                        {
                            "code": "000001", "name": "保留成员",
                            "price": 10, "market_cap_yi": 100,
                        },
                        {"code": "000002", "name": "退出成员", "price": 20},
                    ],
                }],
                "errors": [],
            })
            conn.execute(
                "UPDATE sw3_member SET is_leader=1, is_hot_money=1 WHERE code='000001'"
            )
            conn.commit()
            stock_storage.replace_short_signals(
                conn,
                {
                    "000001": {"signals": {"lhb": {"count": 1}}},
                    "000002": {"signals": {"lhb": {"count": 2}}},
                },
                generated_at="2026-07-10 10:00:00",
                as_of_date="2026-07-10",
            )

            stock_storage.save_sw3_membership(conn, {
                "segments": [{
                    "segment_code": "850111",
                    "segment_name": "新行业名",
                    "members": [
                        {"code": "000001", "name": "保留成员", "price": 11},
                        {"code": "000003", "name": "新增成员", "price": 30},
                    ],
                }],
                "errors": [],
            })

            retained = conn.execute(
                "SELECT price, market_cap_yi, is_leader, is_hot_money "
                "FROM sw3_member WHERE code='000001'"
            ).fetchone()
            self.assertEqual(
                (
                    retained["price"], retained["market_cap_yi"],
                    retained["is_leader"], retained["is_hot_money"],
                ),
                (11.0, 100.0, 1, 1),
            )
            self.assertEqual(set(stock_storage.load_short_signals(conn)), {"000001"})
            self.assertIsNone(
                conn.execute("SELECT 1 FROM sw3_member WHERE code='000002'").fetchone()
            )
            new_member = conn.execute(
                "SELECT is_leader, is_hot_money FROM sw3_member WHERE code='000003'"
            ).fetchone()
            self.assertEqual(tuple(new_member), (0, 0))
        finally:
            conn.close()

    def test_failed_segment_and_empty_payload_preserve_existing_members_and_signals(self):
        conn = stock_storage.connect(":memory:")
        try:
            stock_storage.save_sw3_membership(conn, {
                "segments": [{
                    "segment_code": "850111",
                    "segment_name": "测试行业",
                    "parent_segment": "测试父行业",
                    "member_count": 1,
                    "refreshed_at": "2026-07-09 10:00:00",
                    "members": [{"code": "000001", "name": "成员"}],
                }],
                "errors": [],
            })
            conn.execute(
                "UPDATE sw3_member SET is_leader=1, is_hot_money=1 WHERE code='000001'"
            )
            conn.commit()
            stock_storage.replace_short_signals(
                conn,
                {"000001": {"signals": {"lhb": {"count": 1}}}},
                generated_at="2026-07-10 10:00:00",
                as_of_date="2026-07-10",
            )

            stock_storage.save_sw3_membership(conn, {
                "segments": [],
                "errors": [{
                    "segment_code": "850111",
                    "error": "upstream unavailable",
                }],
            })
            segment = conn.execute(
                "SELECT segment_name, parent_segment, member_count, refreshed_at, error "
                "FROM sw3_segment WHERE segment_code='850111'"
            ).fetchone()
            self.assertEqual(
                tuple(segment),
                ("测试行业", "测试父行业", 1, "2026-07-09 10:00:00", "upstream unavailable"),
            )
            member = conn.execute(
                "SELECT is_leader, is_hot_money FROM sw3_member WHERE code='000001'"
            ).fetchone()
            self.assertEqual(tuple(member), (1, 1))
            self.assertEqual(set(stock_storage.load_short_signals(conn)), {"000001"})

            with self.assertRaisesRegex(ValueError, "empty payload"):
                stock_storage.save_sw3_membership(conn, {"segments": [], "errors": []})
            self.assertIsNotNone(
                conn.execute("SELECT 1 FROM sw3_member WHERE code='000001'").fetchone()
            )
            self.assertEqual(set(stock_storage.load_short_signals(conn)), {"000001"})
        finally:
            conn.close()

    def test_all_failed_segments_are_protected_without_error_truncation(self):
        conn = stock_storage.connect(":memory:")
        try:
            segments = [
                {
                    "segment_code": f"85{idx:04d}",
                    "segment_name": f"行业{idx}",
                    "members": [{"code": f"{idx:06d}", "name": f"成员{idx}"}],
                }
                for idx in range(100)
            ]
            stock_storage.save_sw3_membership(conn, {"segments": segments, "errors": []})
            errors = [
                {
                    "segment_code": segment["segment_code"],
                    "segment_name": segment["segment_name"],
                    "error": "upstream unavailable",
                }
                for segment in segments
            ]

            stock_storage.save_sw3_membership(conn, {"segments": [], "errors": errors})

            self.assertEqual(stock_storage.table_count(conn, "sw3_member"), 100)
            self.assertEqual(stock_storage.table_count(conn, "sw3_segment"), 100)
        finally:
            conn.close()


class RefreshFailurePropagationTest(unittest.TestCase):
    def test_failed_dependency_skips_strategy_rebuild_and_marks_report_failed(self):
        calls = []

        def fake_run_step(name, cmd, *, timeout=None, env=None, skip=False):
            calls.append((name, skip))
            if skip:
                return stock_data_refresh.StepResult(
                    name, " ".join(cmd), True, 0, 0.0, skipped=True
                )
            ok = name != "hot_money_small_cap_universe"
            return stock_data_refresh.StepResult(
                name, " ".join(cmd), ok, 0 if ok else 2, 0.0
            )

        def fake_local_step(name, command, func):
            return stock_data_refresh.StepResult(name, command, True, 0, 0.0)

        with patch.object(stock_data_refresh, "run_step", side_effect=fake_run_step), \
             patch.object(stock_data_refresh, "local_step_result", side_effect=fake_local_step), \
             patch.object(stock_data_refresh, "collect_data_health", return_value={}), \
             patch.object(stock_data_refresh, "write_json_file"):
            report = stock_data_refresh.refresh_before_server(mode="full", timeout=1)

        self.assertFalse(report["ok"])
        self.assertIn(("short_signal_enrichment", True), calls)
        self.assertIn(("strategy_results", True), calls)

    def test_cli_exits_nonzero_for_failed_report(self):
        with patch.object(
            stock_data_refresh, "refresh_before_server", return_value={"ok": False}
        ), patch.object(sys, "argv", ["stock_data_refresh.py"]), \
             patch("builtins.print"):
            with self.assertRaises(SystemExit) as raised:
                stock_data_refresh.main()
        self.assertEqual(raised.exception.code, 1)

    @unittest.skipIf(sys.platform == "win32", "process-group assertion is POSIX-specific")
    def test_timed_out_step_terminates_descendant_processes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            marker = Path(tmpdir) / "child-survived"
            child_code = (
                "import pathlib,time; time.sleep(0.7); "
                f"pathlib.Path({str(marker)!r}).write_text('alive')"
            )
            parent_code = (
                "import subprocess,sys,time; "
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                "time.sleep(10)"
            )

            result = stock_data_refresh.run_step(
                "timeout-tree-test",
                [sys.executable, "-c", parent_code],
                timeout=0.1,
            )
            time.sleep(0.9)
            child_survived = marker.exists()

        self.assertFalse(result.ok)
        self.assertEqual(result.returncode, 124)
        self.assertFalse(child_survived)


class DailySourceFallbackTest(unittest.TestCase):
    def test_empty_primary_source_falls_back_to_next_source(self):
        expected = [{"date": "2026-07-10", "close": 10.0}]
        warnings = []
        with patch.object(
            stock_crawl_common,
            "_enabled_daily_source_groups",
            return_value=[("primary",), ("fallback",)],
        ), patch.object(
            stock_crawl_common,
            "_fetch_daily_source",
            side_effect=[[], expected],
        ) as fetch_mock, patch.object(
            stock_crawl_common, "_record_daily_source_success"
        ) as success_mock:
            records = stock_crawl_common.fetch_qfq_daily_records(
                "000001", "20260710", "20260710", warn=warnings.append
            )

        self.assertEqual(records, expected)
        self.assertEqual(fetch_mock.call_count, 2)
        success_mock.assert_called_once_with("fallback")
        self.assertTrue(any("空数据" in message for message in warnings))

    def test_empty_primary_does_not_hide_fallback_transport_failure(self):
        def passthrough(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch.object(
            stock_crawl_common,
            "_enabled_daily_source_groups",
            return_value=[("primary",), ("fallback",)],
        ), patch.object(
            stock_crawl_common,
            "_fetch_daily_source",
            side_effect=[[], RuntimeError("fallback unavailable")],
        ), patch.object(
            stock_crawl_common, "retry_fetch", side_effect=passthrough
        ):
            with self.assertRaisesRegex(RuntimeError, "fallback unavailable"):
                stock_crawl_common.fetch_qfq_daily_records(
                    "000001", "20260710", "20260710"
                )


class PlateCoverageIntegrityTest(unittest.TestCase):
    def test_incremental_refresh_starts_from_oldest_per_plate_frontier(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "plate.sqlite3"
            conn = plate_storage.connect(db_path)
            try:
                plate_storage.save_sw2_daily_rows(conn, [
                    sw2_row("801001", "行业甲", "2026-07-07"),
                    sw2_row("801002", "行业乙", "2026-07-07"),
                    sw2_row("801001", "行业甲", "2026-07-08"),
                ])
                self.assertEqual(
                    plate_storage.latest_trade_date(conn), "2026-07-08"
                )
                self.assertEqual(
                    plate_storage.oldest_latest_trade_date(conn), "2026-07-07"
                )
            finally:
                conn.close()

            requested = []

            def fake_fetch(start, end, *, page_size):
                requested.append((start, end))
                return [
                    sw2_row("801001", "行业甲", "2026-07-08", 101),
                    sw2_row("801002", "行业乙", "2026-07-08", 102),
                ]

            with patch.object(
                plate_crawl_history, "fetch_sw2_daily_rows", side_effect=fake_fetch
            ):
                result = plate_crawl_history.fetch_sw2_daily_analysis(
                    date(2026, 7, 1),
                    date(2026, 7, 8),
                    chunk_days=30,
                    sleep_sec=0,
                    db_file=str(db_path),
                )

            self.assertEqual(requested[0][0], date(2026, 7, 8))
            self.assertEqual(result["coverage_trade_date"], "2026-07-08")

    def test_incremental_refresh_repairs_internal_cross_section_gap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "plate.sqlite3"
            conn = plate_storage.connect(db_path)
            try:
                plate_storage.save_sw2_daily_rows(conn, [
                    sw2_row("801001", "行业甲", "2026-07-07"),
                    sw2_row("801002", "行业乙", "2026-07-07"),
                    sw2_row("801001", "行业甲", "2026-07-08"),
                    sw2_row("801001", "行业甲", "2026-07-09"),
                    sw2_row("801002", "行业乙", "2026-07-09"),
                ])
                self.assertEqual(
                    plate_storage.oldest_latest_trade_date(conn), "2026-07-09"
                )
                self.assertEqual(
                    plate_storage.recent_incomplete_trade_date(conn), "2026-07-08"
                )
            finally:
                conn.close()

            requested = []

            def fake_fetch(start, end, *, page_size):
                requested.append((start, end))
                return [
                    sw2_row("801001", "行业甲", "2026-07-08"),
                    sw2_row("801002", "行业乙", "2026-07-08"),
                    sw2_row("801001", "行业甲", "2026-07-09"),
                    sw2_row("801002", "行业乙", "2026-07-09"),
                ]

            with patch.object(
                plate_crawl_history, "fetch_sw2_daily_rows", side_effect=fake_fetch
            ):
                result = plate_crawl_history.fetch_sw2_daily_analysis(
                    date(2026, 7, 1),
                    date(2026, 7, 9),
                    chunk_days=30,
                    sleep_sec=0,
                    db_file=str(db_path),
                )

            self.assertEqual(requested[0][0], date(2026, 7, 8))
            self.assertIsNone(result["coverage_gap_date"])

    def test_first_failed_chunk_stops_later_chunks_from_advancing_watermark(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            plate_crawl_history,
            "fetch_sw2_daily_rows",
            side_effect=RuntimeError("source down"),
        ) as fetch_mock:
            result = plate_crawl_history.fetch_sw2_daily_analysis(
                date(2026, 7, 1),
                date(2026, 7, 10),
                chunk_days=3,
                sleep_sec=0,
                db_file=str(Path(tmpdir) / "plate.sqlite3"),
            )

        self.assertEqual(fetch_mock.call_count, 1)
        self.assertEqual(len(result["errors"]), 1)

    def test_retired_plate_does_not_permanently_pin_coverage_frontier(self):
        conn = plate_storage.connect(":memory:")
        try:
            start = date(2026, 4, 1)
            rows = [sw2_row("809999", "已退役", start.strftime("%Y-%m-%d"))]
            for offset in range(45):
                trade_date = (start + timedelta(days=offset + 1)).strftime("%Y-%m-%d")
                rows.extend([
                    sw2_row("801001", "行业甲", trade_date),
                    sw2_row("801002", "行业乙", trade_date),
                ])
            plate_storage.save_sw2_daily_rows(conn, rows)

            self.assertEqual(
                plate_storage.oldest_latest_trade_date(conn),
                (start + timedelta(days=45)).strftime("%Y-%m-%d"),
            )
        finally:
            conn.close()


class FundamentalsIntegrityTest(unittest.TestCase):
    def test_empty_required_record_family_is_incomplete(self):
        payload = valid_fundamentals()
        payload["indicators"] = {"records": []}

        self.assertEqual(
            stock_crawl_fundamentals.fundamentals_missing_families(payload),
            ["indicators.records"],
        )
        with self.assertRaisesRegex(RuntimeError, "indicators.records"):
            stock_crawl_fundamentals.ensure_fundamentals_complete(payload, "1")

    def test_date_only_rows_are_incomplete_after_upstream_schema_drift(self):
        payload = {
            "financials": {
                family: [{"date": "2025-12-31"}]
                for family in ("income", "balance", "cashflow")
            },
            "indicators": {"records": [{"date": "2025-12-31"}]},
            "dividends": {"records": []},
        }

        self.assertEqual(
            stock_crawl_fundamentals.fundamentals_missing_families(payload),
            [
                "financials.income", "financials.balance",
                "financials.cashflow", "indicators.records",
            ],
        )

    def test_main_exits_nonzero_when_any_target_fundamental_fails(self):
        with patch.object(
            stock_crawl_fundamentals,
            "get_segment_leader_universe",
            return_value={"600001": {"name": "测试"}},
        ), patch.object(
            stock_crawl_fundamentals, "fetch_pledge_data_bulk", return_value={}
        ), patch.object(
            stock_crawl_fundamentals,
            "crawl_stocks",
            return_value={"errors": {"600001": {"error": "empty"}}},
        ), patch.object(
            stock_crawl_fundamentals, "strip_proxy_env"
        ), patch.object(
            sys,
            "argv",
            ["stock_crawl_fundamentals.py", "--mode", "full"],
        ):
            with self.assertRaises(SystemExit) as raised:
                stock_crawl_fundamentals.main()

        self.assertEqual(raised.exception.code, 1)

    def test_fetch_fundamentals_does_not_timestamp_empty_results(self):
        with patch.object(
            stock_crawl_fundamentals,
            "fetch_financial_reports",
            return_value={"income": [], "balance": [], "cashflow": []},
        ), patch.object(
            stock_crawl_fundamentals,
            "fetch_financial_indicators",
            return_value={"records": [], "roe_stats": {}},
        ), patch.object(
            stock_crawl_fundamentals,
            "fetch_dividend_history",
            return_value={"records": []},
        ):
            with self.assertRaisesRegex(RuntimeError, "基本面数据不完整"):
                stock_crawl_fundamentals.fetch_fundamentals("000001")

    def test_load_existing_rejects_empty_indicators_even_with_fresh_daily_rows(self):
        conn = stock_storage.connect(":memory:")
        try:
            payload = {
                "symbol": "600001",
                "name": "测试",
                **valid_fundamentals(),
                "history": history_payload_from_records(
                    "600001",
                    "测试",
                    complete_daily_rows(
                        date(2025, 7, 15), stock_crawl_fundamentals.MIN_COMPLETE_DAILY_ROWS
                    ),
                    "test",
                ),
            }
            payload["indicators"] = {"records": []}
            stock_storage.save_stock(conn, payload)
            with patch.object(stock_crawl_fundamentals.ss, "thread_conn", return_value=conn), \
                 patch.object(
                     stock_crawl_fundamentals,
                     "latest_weekday_date",
                     return_value="2026-02-13",
                 ):
                existing = stock_crawl_fundamentals.load_existing()
        finally:
            conn.close()

        self.assertNotIn("600001", existing)


class CompletedDailyBarTest(unittest.TestCase):
    def test_latest_weekday_date_excludes_current_session_before_close(self):
        self.assertEqual(
            latest_weekday_date(datetime(2026, 7, 10, 14, 59)),
            "2026-07-09",
        )
        self.assertEqual(
            latest_weekday_date(datetime(2026, 7, 10, 15, 10)),
            "2026-07-10",
        )
        self.assertEqual(latest_weekday_date("2026-07-10"), "2026-07-10")

    def test_overlap_upserts_final_same_date_ohlcv_without_full_rewrite(self):
        existing = [{
            "date": "2026-07-01",
            "daily_open": 28.1,
            "daily_high": 28.9,
            "daily_low": 28.0,
            "daily_close": 28.90,
            "daily_volume": 100.0,
            "daily_amount": 1000.0,
            "daily_change_pct": 1.3,
            "daily_turnover_rate": 0.1,
        }]
        final_row = {
            "date": "2026-07-01",
            "open": 28.1,
            "high": 29.6,
            "low": 28.0,
            "close": 28.92,
            "volume": 108508.84,
            "amount": 446790220,
            "change_pct": 1.402525,
            "turnover_rate": 2.9065,
        }
        saved = []

        with patch.object(
            stock_crawl_price_valuation,
            "load_stock_file",
            return_value={
                "records": existing,
                "start_date": "2026-07-01",
                "end_date": "2026-07-01",
            },
        ), patch.object(
            stock_crawl_price_valuation,
            "latest_weekday_date",
            return_value="2026-07-01",
        ), patch.object(
            stock_crawl_price_valuation,
            "fetch_daily_range",
            return_value=[final_row],
        ):
            stock_crawl_price_valuation.process_stock(
                "603281",
                "江瀚新材",
                1,
                1,
                save_callback=lambda code, name, data: saved.append(data),
                max_years=0.0,
                refresh_valuation=False,
            )

        result = saved[0]
        self.assertFalse(result["history_replace"])
        self.assertEqual(
            [row["date"] for row in result["history_write_records"]],
            ["2026-07-01"],
        )
        self.assertEqual(result["history_write_records"][0]["daily_volume"], 108508.84)
        self.assertEqual(result["history_write_records"][0]["daily_high"], 29.6)


if __name__ == "__main__":
    unittest.main()
