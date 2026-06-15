import unittest
import tempfile
from pathlib import Path

from fund_generate_output import esc, parse_percent
from fund_technical_analysis import analyze_fund, calc_percentile
from fund_storage import (
    connect as connect_fund_db,
    load_nav_entry,
    load_profile_snapshots,
    load_realtime_estimates,
    save_nav_entry,
    save_profile_snapshots,
    save_realtime_estimates,
)
from stock_crawl_common import history_payload_from_records, merge_records_by_date
from stock_crawl_price_valuation import (
    _decide_valuation_period,
    load_stock_file,
    records_need_ohlcv_backfill,
    save_stock_file,
)


class ParsePercentTest(unittest.TestCase):
    def test_parse_percent_accepts_percent_suffix(self):
        self.assertEqual(parse_percent("12.34%"), 12.34)

    def test_parse_percent_accepts_plain_number_string(self):
        self.assertEqual(parse_percent("-1.5"), -1.5)

    def test_parse_percent_rejects_missing_value(self):
        with self.assertRaises(ValueError):
            parse_percent("--")

    def test_html_escape_helper_escapes_external_text(self):
        self.assertEqual(esc('<script>"x"</script>'), '&lt;script&gt;&quot;x&quot;&lt;/script&gt;')


class TechnicalAnalysisTest(unittest.TestCase):
    def test_calc_percentile_flat_series_defaults_to_midpoint(self):
        self.assertEqual(calc_percentile(1.0, [1.0, 1.0, 1.0]), 50.0)

    def test_analyze_fund_handles_flat_nav_percentile(self):
        records = [
            {"date": f"2024-01-{day:02d}", "nav_acc": "1.0"}
            for day in range(1, 31)
        ] + [
            {"date": f"2024-02-{day:02d}", "nav_acc": "1.0"}
            for day in range(1, 29)
        ] + [
            {"date": f"2024-03-{day:02d}", "nav_acc": "1.0"}
            for day in range(1, 32)
        ] + [
            {"date": f"2024-04-{day:02d}", "nav_acc": "1.0"}
            for day in range(1, 32)
        ]

        result = analyze_fund(records)

        self.assertIsNotNone(result)
        self.assertEqual(result["nav_percentile"], 50.0)


class FundSQLiteStorageTest(unittest.TestCase):
    def test_schema_keeps_only_core_fund_cache_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect_fund_db(Path(tmp) / "fund.sqlite3")
            try:
                tables = {
                    row["name"]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            finally:
                conn.close()

        self.assertIn("fund_nav_records", tables)
        self.assertIn("fund_nav_meta", tables)
        self.assertIn("fund_realtime_estimates", tables)
        self.assertIn("fund_profile_snapshots", tables)
        self.assertNotIn("fund_reports", tables)
        self.assertNotIn("fund_backtest_results", tables)
        self.assertNotIn("fund_signals", tables)

    def test_nav_entry_round_trips_with_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect_fund_db(Path(tmp) / "fund.sqlite3")
            try:
                save_nav_entry(conn, "000001", {
                    "start_date": "2026-01-01",
                    "end_date": "2026-01-02",
                    "records": [
                        {"date": "2026-01-02", "nav": "1.1", "nav_acc": "1.2", "daily_growth_rate": "1.00"},
                        {"date": "2026-01-01", "nav": "1.0", "nav_acc": "1.1", "daily_growth_rate": "0.00"},
                    ],
                })

                entry = load_nav_entry(conn, "000001")
            finally:
                conn.close()

        self.assertEqual(entry["start_date"], "2026-01-01")
        self.assertEqual(entry["end_date"], "2026-01-02")
        self.assertEqual([row["date"] for row in entry["records"]], ["2026-01-01", "2026-01-02"])
        self.assertEqual(entry["records"][1]["nav_acc"], "1.2")

    def test_realtime_estimates_replace_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect_fund_db(Path(tmp) / "fund.sqlite3")
            try:
                save_realtime_estimates(conn, {
                    "000001": {"gsz": "1.1", "gszzl": "1.0", "gztime": "2026-01-01 15:00", "dwjz": "1.0"},
                    "000002": {"gsz": "2.1", "gszzl": "2.0", "gztime": "2026-01-01 15:00", "dwjz": "2.0"},
                })
                save_realtime_estimates(conn, {
                    "000002": {"gsz": "2.2", "gszzl": "2.1", "gztime": "2026-01-02 15:00", "dwjz": "2.0"},
                })

                estimates = load_realtime_estimates(conn)
            finally:
                conn.close()

        self.assertEqual(list(estimates.keys()), ["000002"])
        self.assertEqual(estimates["000002"]["gsz"], "2.2")

    def test_profile_snapshot_save_replaces_absent_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_file = Path(tmp) / "fund.sqlite3"

            conn = connect_fund_db(db_file)
            try:
                self.assertEqual(save_profile_snapshots(conn, [
                    {"fundCode": "000001", "name": "基金A"},
                    {"fundCode": "000002", "name": "基金B"},
                ]), 2)

                self.assertEqual(save_profile_snapshots(conn, [
                    {"fundCode": "000002", "name": "基金B2"},
                ]), 1)
                profiles = load_profile_snapshots(conn)
            finally:
                conn.close()

        self.assertEqual(list(profiles.keys()), ["000002"])
        self.assertEqual(profiles["000002"]["name"], "基金B2")


class StockValuationDecisionTest(unittest.TestCase):
    def test_skips_when_latest_valuation_fields_are_complete(self):
        records = [{
            "date": "2026-05-29",
            "market_cap": 1000,
            "pe_ttm": 12,
            "pe_static": 15,
            "pb": 1.5,
            "pcf": 8,
        }]

        self.assertIsNone(_decide_valuation_period(records, "2026-05-30"))

    def test_fetches_when_latest_pb_or_pe_is_missing(self):
        records = [
            {
                "date": "2026-05-20",
                "market_cap": 1000,
                "pe_ttm": 12,
                "pe_static": 15,
                "pb": 1.5,
                "pcf": 8,
            },
            {
                "date": "2026-05-29",
                "market_cap": 1100,
                "pe_ttm": None,
                "pe_static": 16,
                "pb": None,
                "pcf": 9,
            },
        ]

        self.assertEqual(_decide_valuation_period(records, "2026-05-30"), "近一年")

    def test_fetches_five_years_when_complete_valuation_is_stale(self):
        records = [
            {
                "date": "2024-01-01",
                "market_cap": 1000,
                "pe_ttm": 12,
                "pe_static": 15,
                "pb": 1.5,
                "pcf": 8,
            },
            {"date": "2026-05-29", "close": 10},
        ]

        self.assertEqual(_decide_valuation_period(records, "2026-05-30"), "近五年")


class StockFilePathTest(unittest.TestCase):
    def test_save_stock_file_round_trips_through_sqlite(self):
        import stock_storage as ss
        with tempfile.TemporaryDirectory() as tmp:
            conn = ss.connect(Path(tmp) / "stock.sqlite3")
            ss._thread_local.conns = {str(ss.DEFAULT_DB_FILE): conn}
            try:
                save_stock_file("000001", "坏/名字:测试", {"records": [{
                    "date": "2026-01-01",
                    "daily_open": 10,
                    "daily_high": 11,
                    "daily_low": 9,
                    "daily_close": 10.5,
                    "daily_volume": 100,
                    "daily_amount": 1050,
                }]})

                # 主键是 6 位 code：名字异常或改名都不影响按 code 取回
                loaded = load_stock_file("000001", "新名字")
                self.assertEqual(loaded["records"][0]["date"], "2026-01-01")
                self.assertEqual(loaded["records"][0]["daily_close"], 10.5)
            finally:
                ss._thread_local.conns = {}
                conn.close()


class StockRecordMergeTest(unittest.TestCase):
    def test_merge_records_by_date_can_overwrite_with_none(self):
        merged = merge_records_by_date(
            [{"date": "2026-01-01", "pb": 1.2, "close": 10}],
            [{"date": "2026-01-01", "pb": None, "close": 11}],
            overwrite_none=True,
        )

        self.assertIsNone(merged[0]["pb"])
        self.assertEqual(merged[0]["close"], 11)

    def test_merge_records_by_date_can_preserve_existing_when_new_value_is_none(self):
        merged = merge_records_by_date(
            [{"date": "2026-01-01", "pb": 1.2, "close": 10}],
            [{"date": "2026-01-01", "pb": None, "close": 11}],
            overwrite_none=False,
        )

        self.assertEqual(merged[0]["pb"], 1.2)
        self.assertEqual(merged[0]["close"], 11)


class StockHistorySchemaTest(unittest.TestCase):
    def complete_row(self, date):
        return {
            "date": date,
            "daily_open": 10,
            "daily_high": 11,
            "daily_low": 9,
            "daily_close": 10.5,
            "daily_volume": 10000,
            "daily_amount": 105000,
            "daily_change_pct": 1.2,
            "daily_turnover_rate": 2.3,
        }

    def test_history_payload_prunes_snapshot_only_rows(self):
        payload = history_payload_from_records(
            "000001",
            "测试股份",
            [
                self.complete_row("2026-06-11"),
                {"date": "2026-06-11"},
                {"date": "2026-06-12", "daily_close": 10.8, "market_cap": 100},
            ],
            "test",
        )

        self.assertEqual([row["date"] for row in payload["records"]], ["2026-06-11"])

    def test_single_snapshot_row_does_not_trigger_full_ohlcv_backfill(self):
        records = [
            self.complete_row("2026-06-10"),
            self.complete_row("2026-06-11"),
            {"date": "2026-06-12", "daily_close": 10.8, "market_cap": 100},
        ]

        self.assertFalse(records_need_ohlcv_backfill(records))

    def test_majority_missing_ohlcv_still_triggers_backfill(self):
        records = [
            self.complete_row("2026-06-10"),
            {"date": "2026-06-11", "daily_close": 10.6, "daily_change_pct": 0.9},
            {"date": "2026-06-12", "daily_close": 10.8, "daily_change_pct": 1.1},
        ]

        self.assertTrue(records_need_ohlcv_backfill(records))


class StockStrategyOptimizerTest(unittest.TestCase):
    def test_optuna_startup_trials_keeps_small_runs_exploratory(self):
        from stock_strategy_optimizer import optuna_startup_trials

        self.assertEqual(optuna_startup_trials(1), 1)
        self.assertEqual(optuna_startup_trials(20), 20)
        self.assertEqual(optuna_startup_trials(200), 70)
        self.assertEqual(optuna_startup_trials(300), 105)

    def test_long_objective_penalizes_sparse_fold_coverage(self):
        from stock_strategy_optimizer import (
            LONG_FOLD_COUNT_PENALTY,
            LONG_SOFT_TARGET_FOLDS,
            long_validation_adjusted_objective,
        )

        pairs = [
            {
                "as_of": f"2024-{(idx % 12) + 1:02d}-01",
                "cal_idx": idx * 30,
                "portfolio_return": 0.12,
                "benchmark_return": 0.02,
                "excess_return": 0.10,
                "portfolio_max_drawdown": 0.10,
                "benchmark_max_drawdown": 0.12,
            }
            for idx in range(LONG_SOFT_TARGET_FOLDS)
        ]
        sparse_pairs = pairs[:20]

        _, sparse_detail = long_validation_adjusted_objective(
            sparse_pairs[:12], sparse_pairs[12:], sparse_pairs, 250, 125
        )
        _, full_detail = long_validation_adjusted_objective(
            pairs[:24], pairs[24:], pairs, 250, 125
        )

        expected_penalty = (LONG_SOFT_TARGET_FOLDS - len(sparse_pairs)) * LONG_FOLD_COUNT_PENALTY
        self.assertEqual(sparse_detail["fold_count_penalty"], round(expected_penalty, 5))
        self.assertEqual(full_detail["fold_count_penalty"], 0)


if __name__ == "__main__":
    unittest.main()
