import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import stock_short_term_radar as radar


class StockShortTermRadarTest(unittest.TestCase):
    def test_collect_factor_samples_truncates_all_bars_at_as_of(self):
        start = datetime(2024, 1, 1)
        count = radar.LOOKBACK + max(radar.SHORT_HORIZONS) + 40
        bars = [
            {
                "date": (start + timedelta(days=idx)).strftime("%Y-%m-%d"),
                "open": 10.0 + idx,
                "close": 10.0 + idx,
            }
            for idx in range(count)
        ]
        cutoff_idx = radar.LOOKBACK + max(radar.SHORT_HORIZONS) + 10
        as_of = bars[cutoff_idx]["date"]

        with patch.object(radar.hmr, "_all_bars", return_value=bars), \
             patch.object(radar, "_candidate_factor_values", return_value={}), \
             patch.object(radar, "VERIFY_STEP", 1), \
             patch.object(radar, "VERIFY_WINDOW_DAYS", count):
            result = radar._collect_factor_samples(
                object(), [{"code": "000001"}], as_of=as_of
            )

        self.assertTrue(result["samples"])
        self.assertLessEqual(
            max(sample["date"] for sample in result["samples"]),
            bars[cutoff_idx - max(radar.SHORT_HORIZONS)]["date"],
        )
        self.assertTrue(all(sample["date"] <= as_of for sample in result["samples"]))

    def test_verify_passes_as_of_to_sample_collection(self):
        conn = MagicMock()
        empty = {"samples": [], "dates": [], "codes": []}
        with patch.object(radar.stock_storage, "connect", return_value=conn), \
             patch.object(radar.hmr, "load_candidates", return_value=[]), \
             patch.object(radar, "_collect_factor_samples", return_value=empty) as collect, \
             patch.object(radar.hmr, "write_payload"), \
             patch.object(radar, "_print_verify_summary"):
            result = radar.run_verify(as_of="2025-06-30")

        collect.assert_called_once_with(conn, [], as_of="2025-06-30")
        self.assertEqual(result["as_of"], "2025-06-30")

    def test_backtest_passes_as_of_to_sample_collection(self):
        conn = MagicMock()
        empty = {"samples": [], "dates": [], "codes": []}
        benchmark = {
            "name": "test",
            "nav_field": "nav",
            "n_jumps": 0,
            "components": [],
        }
        with patch.object(radar.stock_storage, "connect", return_value=conn), \
             patch.object(radar.hmr, "load_candidates", return_value=[]), \
             patch.object(radar, "_collect_factor_samples", return_value=empty) as collect, \
             patch.object(radar, "_load_benchmark", return_value=benchmark), \
             patch.object(radar.hmr, "write_payload"), \
             patch.object(radar, "_print_backtest_summary"):
            result = radar.run_backtest(as_of="2025-06-30")

        collect.assert_called_once_with(conn, [], as_of="2025-06-30")
        self.assertEqual(result["as_of"], "2025-06-30")


if __name__ == "__main__":
    unittest.main()
