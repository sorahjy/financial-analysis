import unittest

import research_p25_p26_horizon_backtest as backtest


class P25P26HorizonBacktestTest(unittest.TestCase):
    def test_period_split_matches_research_contract(self):
        self.assertEqual(backtest._period("2023-12-31"), "train")
        self.assertEqual(backtest._period("2024-06-30"), "validation")
        self.assertEqual(backtest._period("2025-06-30"), "test")
        self.assertEqual(backtest._period("2026-01-02"), "holdout")

    def test_excess_is_equal_weighted_by_date_section(self):
        signals = {
            "2024-01-02": [0.10],
            "2024-01-03": [0.00, 0.00, 0.00],
        }
        benchmark = {
            "2024-01-02": (0.0, 10),
            "2024-01-03": (0.0, 10),
        }
        result = backtest._stats(signals, benchmark, horizon=2, period="all")

        self.assertEqual(result["signals"], 4)
        self.assertAlmostEqual(result["pooled_excess"], 0.025)
        self.assertAlmostEqual(result["section_excess_mean"], 0.05)
        self.assertAlmostEqual(result["absolute_win_rate"], 0.25)

    def test_unified_p25_uses_production_h_rule_in_both_pools(self):
        leader = backtest._p25_production_factor("leader")
        hotmoney = backtest._p25_production_factor("hotmoney")

        self.assertEqual(leader["rule"], hotmoney["rule"])
        self.assertEqual(leader["pool"], "leader")
        self.assertEqual(hotmoney["pool"], "hotmoney")
        self.assertIn("position_120 <= 0.45", leader["expression"])
        self.assertNotIn("market_above_ma20", leader["expression"])
        self.assertNotIn("cmf_5", leader["expression"])


if __name__ == "__main__":
    unittest.main()
