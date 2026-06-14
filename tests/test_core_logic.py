import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from fund_backtest import backtest_fund
from fund_generate_output import esc, parse_percent
from fund_technical_analysis import analyze_fund, calc_percentile
from stock_crawl_common import history_payload_from_records, merge_records_by_date
from stock_crawl_price_valuation import (
    _decide_valuation_period,
    find_stock_file,
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
    def test_save_stock_file_sanitizes_name_and_load_finds_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("stock_crawl_price_valuation.DATA_DIR", Path(tmp)):
                save_stock_file("000001", "坏/名字:测试", {"records": [{
                    "date": "2026-01-01",
                    "daily_open": 10,
                    "daily_high": 11,
                    "daily_low": 9,
                    "daily_close": 10.5,
                    "daily_volume": 100,
                    "daily_amount": 1050,
                }]})

                files = list(Path(tmp).glob("CN_000001_*.json"))
                self.assertEqual(len(files), 1)
                self.assertNotIn("/", files[0].name)
                self.assertEqual(load_stock_file("000001", "坏/名字:测试")["records"][0]["date"], "2026-01-01")

    def test_find_stock_file_falls_back_to_code_when_name_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = Path(tmp) / "CN_000002_旧名.json"
            fp.write_text(json.dumps({"records": []}), encoding="utf-8")

            with patch("stock_crawl_price_valuation.DATA_DIR", Path(tmp)):
                self.assertEqual(find_stock_file("000002", "新名"), fp)


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


class FundBacktestTest(unittest.TestCase):
    def test_buy_signal_adds_capital_and_computes_excess_return(self):
        sig = {
            "recent_navs": [1.0, 1.0, 2.0],
            "recent_dates": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "buy_markers": [(1, 1.0)],
            "sell_markers": [],
            "force_sell_markers": [],
        }

        with patch("fund_backtest.INIT_CAPITAL", 100), \
             patch("fund_backtest.TRADE_AMOUNT", 50), \
             patch("fund_backtest.MAX_CAPITAL", 150), \
             patch("fund_backtest.MIN_CAPITAL", 0):
            result = backtest_fund(sig)

        self.assertEqual(result["buy_count"], 1)
        self.assertEqual(result["strategy_value"], 250.0)
        self.assertEqual(result["benchmark_value"], 200.0)
        self.assertEqual(result["excess_return"], 50.0)


if __name__ == "__main__":
    unittest.main()
