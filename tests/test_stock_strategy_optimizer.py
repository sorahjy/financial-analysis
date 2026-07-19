import random
import json
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
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
    def test_smallcap_optimizer_universe_intersects_hotmoney_pool(self):
        series = {
            "600000": [{"date": "2020-01-01", "close": 1.0}] * 320,
            "600001": [{"date": "2020-01-01", "close": 1.0}] * 320,
            "600002": [{"date": "2020-01-01", "close": 1.0}] * 10,
        }
        with patch.object(optimizer, "load_hot_money_codes", return_value={"600000", "600002"}):
            codes, notes = optimizer.smallcap_optimizer_universe_codes(series, {})

        self.assertEqual(codes, {"600000"})
        self.assertTrue(any("is_hot_money=1" in note for note in notes))

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

    def test_short_optuna_search_space_matches_random_search_knobs(self):
        trial = RecordingTrial()
        cfg, hold_days = optimizer._suggest_short_config(trial)

        for factor in optimizer.SHORT_FACTORS:
            self.assertIn(f"w_{factor.key}", trial.float_names)
        for name in (
            "min_score",
            "max_consecutive_limit_up",
            "hold_days",
        ):
            self.assertIn(name, trial.categorical_names)
        self.assertNotIn("hold_days_min", trial.categorical_names)
        self.assertNotIn("hold_days_max", trial.categorical_names)
        self.assertIn(hold_days, optimizer.SHORT_HOLD_DAYS_CHOICES)
        self.assertEqual(cfg["hold_days"], hold_days)
        self.assertEqual(cfg["hold_days_min"], hold_days)
        self.assertEqual(cfg["hold_days_max"], hold_days)
        self.assertIn("weights", cfg)
        self.assertEqual(cfg["min_lhb_count"], 0)
        self.assertEqual(cfg["min_hot_money_concurrent"], 0)
        self.assertNotIn("min_lhb_count", trial.categorical_names)
        self.assertNotIn("min_hot_money_concurrent", trial.categorical_names)
        self.assertNotIn("hold_days_min", optimizer._default_short_params())
        self.assertNotIn("hold_days_max", optimizer._default_short_params())

    def test_short_backtest_prefers_explicit_full_signal_event_date(self):
        pick = {
            "code": "000001",
            "event_date": "2026-07-10",
            # Legacy/truncated sample points to an earlier date and must not win.
            "followers": [{"date": "2026-07-03"}],
        }
        series = {
            "000001": [
                {"date": "2026-07-04", "close": 5.0},
                {"date": "2026-07-11", "close": 10.0},
                {"date": "2026-07-12", "close": 11.0},
            ]
        }

        self.assertEqual(optimizer.latest_pick_event_date(pick), "2026-07-10")
        result = optimizer.short_actual_backtest([pick], series, hold_days=1)

        self.assertEqual(result["samples"], 1)
        self.assertAlmostEqual(result["avg_return"], 9.6, places=6)

    def test_short_horizon_without_samples_does_not_borrow_another_horizon(self):
        picks = [{"code": "000001", "event_date": "2026-07-01"}]
        series = {
            "000001": [
                {"date": "2026-07-02", "close": 10.0},
                {"date": "2026-07-03", "close": 11.0},
                {"date": "2026-07-04", "close": 12.0},
            ]
        }

        hold_days, actual = optimizer._short_backtest_for_horizon(
            picks, series, suggested_hold_days=5
        )

        self.assertEqual(hold_days, 5)
        self.assertEqual(actual["samples"], 0)

        finalized = optimizer._finalize_short_best({
            "hold_days": 5,
            "hold_days_validated": False,
            "config": optimizer._short_config_with_hold_days(DEFAULT_CONFIG["short"], 5),
        })
        self.assertEqual(finalized["hold_days"], optimizer._default_short_params()["hold_days"])
        self.assertEqual(finalized["config"]["hold_days"], finalized["hold_days"])
        self.assertEqual(finalized["hold_days_fallback_reason"], "no_forward_price_samples")

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

    def test_run_optimization_launches_all_three_strategies_in_process_pool(self):
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
        self.assertEqual(result["smallcap"]["best"]["config"], {"kind": "smallcap"})
        self.assertEqual(result["short"]["best"]["config"], {"kind": "short"})
        self.assertEqual(executor_kwargs["max_workers"], 3)
        self.assertEqual(executor_kwargs["initializer"], "init")
        submitted_by_strategy = {args[0]: args for _, args in submitted}
        self.assertEqual(set(submitted_by_strategy), {"long", "smallcap", "short"})
        self.assertEqual(submitted_by_strategy["long"][1], 3)
        self.assertEqual(submitted_by_strategy["smallcap"][1], 3)
        self.assertEqual(submitted_by_strategy["short"][1], 3)
        self.assertEqual(submitted_by_strategy["long"][3], 0)
        self.assertEqual(submitted_by_strategy["smallcap"][3], 1)
        self.assertEqual(submitted_by_strategy["short"][3], 2)
        self.assertNotEqual(submitted_by_strategy["long"][2], submitted_by_strategy["short"][2])
        self.assertIn("strategy_seeds", result)

    def test_smallcap_only_optimization_generates_best_fold_chart(self):
        optimized = {
            "strategy": "smallcap",
            "iterations": 1,
            "best": {
                "config": {"top_n": 20},
                "hold_td": 60,
                "objective": 1.0,
            },
            "notes": [],
        }
        chart = {"file": "data/stock_strategy_smallcap_fold_paths.svg", "folds": 3}
        with patch.object(optimizer, "_run_strategy_optimizer", return_value=optimized), \
             patch.object(optimizer, "create_best_smallcap_fold_path_chart", return_value=chart) as create_chart, \
             tempfile.TemporaryDirectory() as tmpdir, \
             patch.object(optimizer, "SMALLCAP_OUTPUT_FILE", Path(tmpdir) / "smallcap.json"), \
             patch.object(optimizer, "SMALLCAP_OPTIMIZED_CONFIG_FILE", Path(tmpdir) / "smallcap_config.json"):
            result = optimizer.run_optimization(
                iterations=1, seed=7, persist=True, strategies=("smallcap",)
            )

        create_chart.assert_called_once_with(result["smallcap"])
        self.assertEqual(result["smallcap"]["best_fold_path_chart"], chart)
        self.assertTrue(any("走势图已生成" in note for note in result["smallcap"]["notes"]))

    def test_smallcap_optimizer_excludes_index_and_size_factors(self):
        excluded = {"csi300_current", "csi300_persistence", "market_cap", "size_reversal"}
        keys = {factor.key for factor in optimizer.SMALLCAP_FACTORS}
        self.assertFalse(keys & excluded)

        trial = RecordingTrial()
        cfg, hold_td = optimizer._suggest_smallcap_config(trial)
        self.assertIn(hold_td, optimizer.SMALLCAP_HOLD_CHOICES)
        self.assertFalse(any(name.removeprefix("w_") in excluded for name in trial.float_names))
        self.assertFalse(set(cfg["weights"]) & excluded)
        self.assertEqual(cfg["top_n"], 10)

    def test_smallcap_optimizer_never_searches_or_enables_high_drawdown_filter(self):
        random_cfg = optimizer.random_smallcap_config(random.Random(7), iteration=5)
        self.assertFalse(random_cfg["require_high_drawdown"])
        self.assertEqual(
            random_cfg["min_high_drawdown_pct"],
            DEFAULT_CONFIG["smallcap"]["min_high_drawdown_pct"],
        )

        trial = RecordingTrial()
        optuna_cfg, _ = optimizer._suggest_smallcap_config(trial)
        self.assertNotIn("require_high_drawdown", trial.categorical_names)
        self.assertNotIn("min_high_drawdown_pct", trial.int_names)
        self.assertFalse(optuna_cfg["require_high_drawdown"])

        default_params = optimizer._default_smallcap_params()
        self.assertNotIn("require_high_drawdown", default_params)
        self.assertNotIn("min_high_drawdown_pct", default_params)

    def test_smallcap_optimization_saves_independent_config_file(self):
        def fake_strategy(strategy, iterations, seed, progress_position):
            return {
                "strategy": strategy,
                "iterations": iterations,
                "best": {
                    "config": {"kind": strategy, "weights": {"roe_mean": 1.25}},
                    "objective": 2.5,
                },
                "progress_position": progress_position,
                "seed": seed,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            standard_output = Path(tmpdir) / "standard_optimization.json"
            standard_config = Path(tmpdir) / "standard_config.json"
            smallcap_output = Path(tmpdir) / "smallcap_optimization.json"
            smallcap_config = Path(tmpdir) / "smallcap_config.json"
            standard_config.write_text(json.dumps({"config": {"long": {"kept": True}}}), encoding="utf-8")

            with patch.object(optimizer, "_run_strategy_optimizer", side_effect=fake_strategy), \
                 patch.object(optimizer, "OUTPUT_FILE", standard_output), \
                 patch.object(optimizer, "OPTIMIZED_CONFIG_FILE", standard_config), \
                 patch.object(optimizer, "SMALLCAP_OUTPUT_FILE", smallcap_output), \
                 patch.object(optimizer, "SMALLCAP_OPTIMIZED_CONFIG_FILE", smallcap_config):
                result = optimizer.run_optimization(
                    iterations=3, seed=123, persist=True, strategies=("smallcap",)
                )

            saved = json.loads(smallcap_config.read_text(encoding="utf-8"))
            report = json.loads(smallcap_output.read_text(encoding="utf-8"))
            standard_was_written = standard_output.exists()
            preserved_standard = json.loads(standard_config.read_text(encoding="utf-8"))

        self.assertEqual(result["optimized_strategies"], ["smallcap"])
        self.assertFalse(standard_was_written)
        self.assertEqual(preserved_standard["config"]["long"], {"kept": True})
        self.assertEqual(saved["config"]["smallcap"]["weights"]["roe_mean"], 1.25)
        self.assertEqual(saved["smallcap_universe_version"], optimizer.SMALLCAP_UNIVERSE_VERSION)
        self.assertIn("smallcap", report)

    def test_short_only_optimization_preserves_existing_long_config(self):
        executor_workers = []

        class FakeProcessPoolExecutor:
            def __init__(self, max_workers=None, **_kwargs):
                executor_workers.append(max_workers)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args):
                future = Future()
                future.set_result(fn(*args))
                return future

        def fake_strategy(strategy, iterations, seed, progress_position):
            return {
                "strategy": strategy,
                "iterations": iterations,
                "best": {
                    "config": {"kind": strategy},
                    "hold_days": 3,
                    "objective": 2.0,
                },
                "progress_position": progress_position,
                "seed": seed,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "optimization.json"
            config_file = Path(tmpdir) / "optimized_config.json"
            config_file.write_text(json.dumps({
                "config": {"long": {"kept": True}, "short": {"legacy": True}},
                "scores": {"long_objective": 9.0, "short_objective": -1.0},
                "short_universe_version": "legacy",
            }), encoding="utf-8")

            with patch.object(optimizer, "ProcessPoolExecutor", FakeProcessPoolExecutor), \
                 patch.object(optimizer, "_process_pool_kwargs", return_value={}), \
                 patch.object(optimizer, "_run_strategy_optimizer", side_effect=fake_strategy), \
                 patch.object(optimizer, "OUTPUT_FILE", output_file), \
                 patch.object(optimizer, "OPTIMIZED_CONFIG_FILE", config_file):
                result = optimizer.run_optimization(
                    iterations=3,
                    seed=123,
                    persist=True,
                    strategies=("short",),
                )

            saved = json.loads(config_file.read_text(encoding="utf-8"))

        self.assertEqual(result["optimized_strategies"], ["short"])
        self.assertNotIn("long", result)
        self.assertEqual(executor_workers, [])
        self.assertEqual(saved["config"]["long"], {"kept": True})
        self.assertEqual(saved["config"]["short"]["kind"], "short")
        self.assertEqual(saved["config"]["short"]["hold_days"], 3)
        self.assertEqual(saved["config"]["short"]["hold_days_min"], 3)
        self.assertEqual(saved["config"]["short"]["hold_days_max"], 3)
        self.assertEqual(saved["scores"]["long_objective"], 9.0)
        self.assertEqual(saved["short_universe_version"], optimizer.SHORT_UNIVERSE_VERSION)

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
