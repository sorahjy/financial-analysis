import random
import unittest
from unittest.mock import patch

import stock_strategy_optimizer as optimizer
from stock_advanced_strategies import DEFAULT_CONFIG


class RecordingTrial:
    def __init__(self):
        self.float_names = []
        self.categorical_names = []
        self.int_names = []

    def suggest_float(self, name, low, high):
        self.float_names.append(name)
        return low

    def suggest_categorical(self, name, choices):
        self.categorical_names.append(name)
        return choices[0]

    def suggest_int(self, name, low, high):
        self.int_names.append(name)
        return low


class StockStrategyOptimizerTest(unittest.TestCase):
    def test_long_optimizer_universe_intersects_segment_leaders(self):
        series = {
            "600000": [{"date": "2020-01-01", "close": 1.0}] * 320,
            "600001": [{"date": "2020-01-01", "close": 1.0}] * 320,
            "600002": [{"date": "2020-01-01", "close": 1.0}] * 10,
        }

        with patch.object(optimizer, "load_segment_leader_codes", return_value={"600000", "600002"}):
            codes, notes = optimizer.long_optimizer_universe_codes(
                series, {"use_segment_leaders": True}
            )

        self.assertEqual(codes, {"600000"})
        self.assertTrue(any("SW3 细分龙头池" in note for note in notes))

    def test_long_market_cap_signals_are_fixed_zero_and_not_searched(self):
        fixed_keys = optimizer.LONG_FIXED_ZERO_WEIGHTS
        base = DEFAULT_CONFIG["long"]

        self.assertEqual(base["min_market_cap_yi"], 0)
        for key in fixed_keys:
            self.assertEqual(base["weights"][key], 0.0)

        random_cfg = optimizer.random_long_config(random.Random(7), iteration=5)
        self.assertEqual(random_cfg["min_market_cap_yi"], 0)
        for key in fixed_keys:
            self.assertEqual(random_cfg["weights"][key], 0.0)

        default_params = optimizer._default_long_params()
        self.assertNotIn("min_market_cap_yi", default_params)
        for key in fixed_keys:
            self.assertNotIn(f"w_{key}", default_params)

        trial = RecordingTrial()
        optuna_cfg, _ = optimizer._suggest_long_config(trial)

        self.assertNotIn("min_market_cap_yi", trial.categorical_names)
        for key in fixed_keys:
            self.assertNotIn(f"w_{key}", trial.float_names)
            self.assertEqual(optuna_cfg["weights"][key], 0.0)
        self.assertEqual(optuna_cfg["min_market_cap_yi"], 0)


if __name__ == "__main__":
    unittest.main()
