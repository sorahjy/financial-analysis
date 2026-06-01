import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from fund_backtest import backtest_fund
from fund_generate_output import esc, parse_percent
from fund_technical_analysis import analyze_fund, calc_percentile
from stock_crawl_top_800_data import _decide_valuation_period, find_stock_file, load_stock_file, save_stock_file


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
            with patch("stock_crawl_top_800_data.DATA_DIR", Path(tmp)):
                save_stock_file("000001", "坏/名字:测试", {"records": [{"date": "2026-01-01"}]})

                files = list(Path(tmp).glob("CN_000001_*.json"))
                self.assertEqual(len(files), 1)
                self.assertNotIn("/", files[0].name)
                self.assertEqual(load_stock_file("000001", "坏/名字:测试")["records"][0]["date"], "2026-01-01")

    def test_find_stock_file_falls_back_to_code_when_name_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = Path(tmp) / "CN_000002_旧名.json"
            fp.write_text(json.dumps({"records": []}), encoding="utf-8")

            with patch("stock_crawl_top_800_data.DATA_DIR", Path(tmp)):
                self.assertEqual(find_stock_file("000002", "新名"), fp)


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
