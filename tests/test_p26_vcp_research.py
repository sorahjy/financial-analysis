import unittest

import numpy as np

import research_p25_buy_factor as p25
import research_p26_vcp_factor as p26


class P26VcpResearchTest(unittest.TestCase):
    def test_standard_vcp_has_moderate_volume_band_and_high_location(self):
        predicates = {
            (p25.FEATURE_NAMES[item.feature], item.op, item.threshold)
            for item in p26._standard_vcp_rule()
        }
        self.assertIn(("volume_last_vs_prev20", ">=", 1.2), predicates)
        self.assertIn(("volume_last_vs_prev20", "<=", 2.5), predicates)
        self.assertIn(("distance_high_180", ">=", -0.25), predicates)
        self.assertIn(("volume_mavd_21_120", "<=", 0.0), predicates)

    def test_preholdout_selection_score_does_not_read_holdout(self):
        def scope(win, excess, lift=2.0, signals=100):
            return {
                "signals": signals,
                "forward_positive_rate": win,
                "avg_excess_forward_5d": excess,
                "lift": lift,
                "episode_recall": 0.01,
            }

        metrics = {
            "train": scope(0.60, 0.01),
            "validation": scope(0.58, 0.02),
            "test": scope(0.57, 0.01),
            "holdout": scope(0.10, -0.20),
            "all": scope(0.59, 0.01),
        }
        row = {"search_universe": "combined", "metrics": metrics}
        before = p26._preholdout_score(row)
        metrics["holdout"] = scope(0.95, 0.30)
        self.assertEqual(before, p26._preholdout_score(row))

    def test_shared_feature_matrix_exposes_vcp_features(self):
        bars = []
        for i in range(220):
            close = 10.0 + i * 0.02
            previous = 10.0 + max(i - 1, 0) * 0.02
            volume = 2000.0 - min(i, 180) * 4.0
            bars.append({
                "date": f"2025-{1 + i // 28:02d}-{1 + i % 28:02d}",
                "open": previous,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": volume,
                "turnover": volume / 1000.0,
            })
        matrix = p25._feature_matrix(bars)
        self.assertIsNotNone(matrix)
        for name in (
            "ret_120",
            "range_ratio_10_60",
            "distance_high_180",
            "volume_mavd_21_120",
            "close_ma60_gap",
            "ma20_ma60_gap",
        ):
            values = matrix["x"][:, p25.FEATURE_NAMES.index(name)]
            self.assertTrue(np.isfinite(values).all(), name)


if __name__ == "__main__":
    unittest.main()
