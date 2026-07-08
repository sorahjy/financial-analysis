import random
import unittest
from concurrent.futures import Future
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
        self.assertEqual(random_cfg["min_score"], optimizer.LONG_FIXED_MIN_SCORE)
        for key in fixed_keys:
            self.assertEqual(random_cfg["weights"][key], 0.0)

        default_params = optimizer._default_long_params()
        self.assertNotIn("min_market_cap_yi", default_params)
        self.assertNotIn("min_score", default_params)
        for key in fixed_keys:
            self.assertNotIn(f"w_{key}", default_params)

        trial = RecordingTrial()
        optuna_cfg, _ = optimizer._suggest_long_config(trial)

        self.assertNotIn("min_market_cap_yi", trial.categorical_names)
        self.assertNotIn("min_score", trial.categorical_names)
        self.assertEqual(optuna_cfg["min_score"], optimizer.LONG_FIXED_MIN_SCORE)
        for key in fixed_keys:
            self.assertNotIn(f"w_{key}", trial.float_names)
            self.assertEqual(optuna_cfg["weights"][key], 0.0)
        self.assertEqual(optuna_cfg["min_market_cap_yi"], 0)

    def test_long_capital_factors_are_searchable(self):
        keys = {"holder_count_change", "repurchase_recent", "lhb_recent_avoid"}
        searchable = {factor.key for factor in optimizer.LONG_SEARCHABLE_FACTORS}

        self.assertTrue(keys <= searchable)

        random_cfg = optimizer.random_long_config(random.Random(7), iteration=0)
        for key in keys:
            self.assertIn(key, random_cfg["weights"])
            self.assertGreater(random_cfg["weights"][key], 0)

        trial = RecordingTrial()
        optimizer._suggest_long_config(trial)
        for key in keys:
            self.assertIn(f"w_{key}", trial.float_names)

    def test_short_optuna_search_space_matches_random_search_knobs(self):
        trial = RecordingTrial()
        cfg, hold_days = optimizer._suggest_short_config(trial)

        for factor in optimizer.SHORT_FACTORS:
            self.assertIn(f"w_{factor.key}", trial.float_names)
        for name in (
            "min_score",
            "min_lhb_count",
            "min_hot_money_concurrent",
            "max_consecutive_limit_up",
            "hold_days_min",
            "hold_days_max",
            "hold_days",
        ):
            self.assertIn(name, trial.categorical_names)
        self.assertIn(hold_days, optimizer.SHORT_HOLD_DAYS_CHOICES)
        self.assertIn("weights", cfg)

    def test_prepared_short_scoring_matches_scalar_scoring(self):
        broad = []
        for idx, code in enumerate(("600000", "600001", "600002", "600003"), 1):
            raw = {factor.key: idx * 10 + pos for pos, factor in enumerate(optimizer.SHORT_FACTORS)}
            raw["known_hot_money_ratio"] = min(1.0, idx / 4)
            raw["limit_up_control"] = idx - 1
            broad.append({
                "code": code,
                "name": f"测试{idx}",
                "raw_factors": raw,
                "followers": [],
                "data_quality": 0.8,
            })
        cfg = {
            **DEFAULT_CONFIG["short"],
            "top_n": 3,
            "min_score": 0,
            "min_lhb_count": 0,
            "min_hot_money_concurrent": 0,
            "max_consecutive_limit_up": 99,
        }
        prepared = optimizer.prepare_candidate_factor_scores(broad, optimizer.SHORT_FACTORS)

        scalar = optimizer.score_short_candidates(broad, cfg)
        vector = optimizer.score_prepared_short_candidates(prepared, cfg)

        self.assertEqual(
            [(row["code"], row["score"], row["rank"]) for row in vector],
            [(row["code"], row["score"], row["rank"]) for row in scalar],
        )

    def test_run_optimization_launches_long_and_short_in_process_pool(self):
        submitted = []
        executor_kwargs = {}

        class FakeProcessPoolExecutor:
            def __init__(self, max_workers=None, **kwargs):
                executor_kwargs["max_workers"] = max_workers
                executor_kwargs.update(kwargs)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args):
                submitted.append((fn, args))
                future = Future()
                future.set_result(fn(*args))
                return future

        def fake_strategy(strategy, iterations, seed, progress_position):
            return {
                "strategy": strategy,
                "iterations": iterations,
                "best": {"config": {"kind": strategy}, "objective": 1.0},
                "progress_position": progress_position,
                "seed": seed,
            }

        with patch.object(optimizer, "ProcessPoolExecutor", FakeProcessPoolExecutor), \
             patch.object(optimizer, "_process_pool_kwargs", return_value={"initializer": "init", "initargs": ("lock",)}), \
             patch.object(optimizer, "_run_strategy_optimizer", side_effect=fake_strategy):
            result = optimizer.run_optimization(iterations=3, seed=123, persist=False)

        self.assertEqual(result["long"]["best"]["config"], {"kind": "long"})
        self.assertEqual(result["short"]["best"]["config"], {"kind": "short"})
        self.assertEqual(executor_kwargs["max_workers"], 2)
        self.assertEqual(executor_kwargs["initializer"], "init")
        submitted_by_strategy = {args[0]: args for _, args in submitted}
        self.assertEqual(set(submitted_by_strategy), {"long", "short"})
        self.assertEqual(submitted_by_strategy["long"][1], 3)
        self.assertEqual(submitted_by_strategy["short"][1], 3)
        self.assertEqual(submitted_by_strategy["long"][3], 0)
        self.assertEqual(submitted_by_strategy["short"][3], 1)
        self.assertNotEqual(submitted_by_strategy["long"][2], submitted_by_strategy["short"][2])
        self.assertIn("strategy_seeds", result)

    def test_pit_long_scoring_requires_entry_day_trade(self):
        weights = {factor.key: 0.0 for factor in optimizer.LONG_FACTORS}
        weights["roe_mean"] = 1.0
        config = {
            "weights": weights,
            "min_score": 50,
            "top_n": 10,
            "exclude_st": False,
            "require_csi300": False,
            "min_market_cap_yi": 0,
        }
        prepared = [
            {
                "item": {"code": "TRADE", "name": "可交易", "raw_factors": {}},
                "scores": {"roe_mean": 80.0},
            },
            {
                "item": {"code": "HALT", "name": "停牌", "raw_factors": {}},
                "scores": {"roe_mean": 95.0},
            },
            {
                "item": {"code": "NEW", "name": "未上市", "raw_factors": {}},
                "scores": {"roe_mean": 99.0},
            },
        ]
        series = {
            "TRADE": [{"date": "2024-01-02", "close": 10.0}],
            "HALT": [{"date": "2024-01-01", "close": 10.0}],
            "NEW": [{"date": "2024-01-03", "close": 10.0}],
        }

        picks = optimizer.score_prepared_long_candidates(
            prepared, config, entry_series=series, as_of="2024-01-02"
        )

        self.assertEqual([row["code"] for row in picks], ["TRADE"])

    def test_portfolio_fold_path_uses_daily_returns_when_qfq_close_is_negative(self):
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        series = {
            "BADQFQ": [
                {"date": dates[0], "close": -0.10, "change_pct": None},
                {"date": dates[1], "close": -0.11, "change_pct": 10.0},
                {"date": dates[2], "close": -0.12, "change_pct": 10.0},
            ],
        }
        benchmark = [{"date": date, "close": 1.0} for date in dates]

        with patch.object(optimizer, "LONG_MIN_VALID_PICKS", 1):
            path = optimizer.portfolio_fold_path(
                [{"code": "BADQFQ"}], series, benchmark, dates[0], 2
            )

        self.assertIsNotNone(path)
        self.assertEqual(path["stock_count"], 1)
        self.assertAlmostEqual(path["portfolio_path"][-1], 1.21, places=6)
        self.assertAlmostEqual(path["portfolio_return"], 0.21 - optimizer.LONG_COST, places=6)

    def test_portfolio_fold_path_rejects_unrealistic_daily_return(self):
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        series = {
            "DIRTY": [
                {"date": dates[0], "close": 0.10, "change_pct": None},
                {"date": dates[1], "close": -0.01, "change_pct": -110.0},
                {"date": dates[2], "close": 1.00, "change_pct": 1000.0},
            ],
            "CLEAN": [
                {"date": dates[0], "close": 10.0, "change_pct": None},
                {"date": dates[1], "close": 11.0, "change_pct": 10.0},
                {"date": dates[2], "close": 12.0, "change_pct": 9.090909},
            ],
        }
        benchmark = [{"date": date, "close": 1.0} for date in dates]

        with patch.object(optimizer, "LONG_MIN_VALID_PICKS", 1):
            path = optimizer.portfolio_fold_path(
                [{"code": "DIRTY"}, {"code": "CLEAN"}], series, benchmark, dates[0], 2
            )

        self.assertIsNotNone(path)
        self.assertEqual(path["stock_count"], 1)
        self.assertAlmostEqual(path["portfolio_path"][-1], 1.2, places=6)
        self.assertAlmostEqual(path["portfolio_return"], 0.2 - optimizer.LONG_COST, places=6)

    def test_portfolio_fold_path_requires_entry_and_exit_trading_days(self):
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        series = {
            "ENTRY_SUSP": [
                {"date": "2023-12-29", "close": 10.0, "change_pct": None},
                {"date": "2024-01-03", "close": 11.0, "change_pct": 10.0},
                {"date": "2024-01-04", "close": 12.0, "change_pct": 9.090909},
            ],
            "EXIT_SUSP": [
                {"date": "2024-01-02", "close": 10.0, "change_pct": None},
                {"date": "2024-01-03", "close": 11.0, "change_pct": 10.0},
            ],
            "CLEAN": [
                {"date": "2024-01-02", "close": 10.0, "change_pct": None},
                {"date": "2024-01-03", "close": 11.0, "change_pct": 10.0},
                {"date": "2024-01-04", "close": 12.0, "change_pct": 9.090909},
            ],
        }
        benchmark = [{"date": date, "close": 1.0} for date in dates]

        with patch.object(optimizer, "LONG_MIN_VALID_PICKS", 1):
            path = optimizer.portfolio_fold_path(
                [{"code": "ENTRY_SUSP"}, {"code": "EXIT_SUSP"}, {"code": "CLEAN"}],
                series,
                benchmark,
                dates[0],
                2,
            )

        self.assertIsNotNone(path)
        self.assertEqual(path["stock_count"], 1)
        self.assertAlmostEqual(path["portfolio_path"][-1], 1.2, places=6)


if __name__ == "__main__":
    unittest.main()
