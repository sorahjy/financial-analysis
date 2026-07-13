import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import stock_advanced_strategies as strategies
from stock_advanced_strategies import (
    FactorSpec,
    SMALLCAP_EXCLUDED_FACTOR_KEYS,
    apply_scores,
    build_long_candidates,
    build_smallcap_candidates,
    compute_ttm,
    compute_yoy_growth,
    compute_long_raw_factors,
    csi300_persistence_proxy,
    first_not_none,
    get_factor_registry,
    get_default_config,
    high_to_latest_drawdown_pct,
    industry_label_from_sw3,
    institution_conflict_score,
    passes_long_hard_filters,
    report_available_date,
    reports_available_asof,
    rsi_sweetspot_score,
    run_long_strategy,
    run_smallcap_strategy,
    run_strategies,
    strip_internal,
    yoy_from_latest_and_annual,
)


PIT_SYNTH_STOCK = {
    "indicators": {
        "records": [
            {"date": "2025-12-31", "roe": 30.0, "total_assets": 1e9,
             "asset_liability_ratio": 40, "gross_margin": 50, "net_margin": 20,
             "bvps_adjusted": 10.0},
            {"date": "2024-12-31", "roe": 10.0, "total_assets": 9e8,
             "asset_liability_ratio": 42, "gross_margin": 48, "net_margin": 18,
             "bvps_adjusted": 9.0},
        ],
        "roe_stats": {"mean": 20.0, "std": 10.0, "count": 2},
    },
    "financials": {
        "income": [
            {"date": "2025-12-31", "revenue": 200, "net_profit": 60,
             "cost_of_revenue": 100, "operating_cost": 120},
            {"date": "2024-12-31", "revenue": 150, "net_profit": 40,
             "cost_of_revenue": 80, "operating_cost": 95},
        ],
        "balance": [
            {"date": "2025-12-31", "total_equity_parent": 500, "total_assets_liabilities": 1000},
            {"date": "2024-12-31", "total_equity_parent": 450, "total_assets_liabilities": 900},
        ],
        "cashflow": [
            {"date": "2025-12-31", "operating_cashflow_net": 70},
            {"date": "2024-12-31", "operating_cashflow_net": 45},
        ],
    },
    "dividends": {"records": [], "yearly_dividends": {}, "consecutive_3y_dividend": False},
    "daily": {"stats": {"history_window_annualized_volatility": 0.3,
                        "history_window_avg_daily_amount": 1e8,
                        "history_window_avg_daily_turnover_rate": 2.5,
                        "latest_daily_close": 15}},
    "history": {"records": [
        {"date": "2025-05-28", "close": 10.0, "turnover_rate": 1.0},
        {"date": "2025-05-29", "close": 10.2, "turnover_rate": 3.0},
        {"date": "2025-06-02", "close": 10.4, "turnover_rate": 9.0},
    ]},
}


class StockAdvancedStrategyTest(unittest.TestCase):
    def test_price_row_caches_verify_identity_and_clear_on_invalidation(self):
        strategies.invalidate_dir_fingerprints()
        old_rows = [{"close": "1", "turnover_rate": "2"}]
        fresh_rows = [{"close": "99", "turnover_rate": "88"}]
        stale_arrays = strategies.price_arrays(old_rows, cache=True)

        # Simulate an entry left under a newly reused integer id. The retained
        # object identity must reject it rather than return old stock data.
        strategies._price_arr_cache[id(fresh_rows)] = (old_rows, stale_arrays)
        strategies._pretransformed_rows[id(fresh_rows)] = old_rows

        arrays = strategies.price_arrays(fresh_rows, cache=True)
        strategies.pretransform_price_rows(fresh_rows, cache=True)

        self.assertEqual(arrays["close"].tolist(), [99.0])
        self.assertEqual(arrays["turnover_rate"].tolist(), [88.0])
        self.assertIs(type(fresh_rows[0]["close"]), float)
        self.assertIs(strategies._price_arr_cache[id(fresh_rows)][0], fresh_rows)
        self.assertIs(strategies._pretransformed_rows[id(fresh_rows)], fresh_rows)

        strategies.invalidate_dir_fingerprints()
        self.assertFalse(strategies._price_arr_cache)
        self.assertFalse(strategies._pretransformed_rows)

    def test_price_row_caches_are_bounded(self):
        strategies.invalidate_dir_fingerprints()
        rows_by_stock = [
            [{"close": str(idx), "turnover_rate": "1"}] for idx in range(4)
        ]
        with patch.object(strategies, "_PRICE_ROW_CACHE_MAXSIZE", 2):
            for rows in rows_by_stock:
                strategies.pretransform_price_rows(rows, cache=True)
                strategies.price_arrays(rows, cache=True)

        self.assertEqual(len(strategies._pretransformed_rows), 2)
        self.assertEqual(len(strategies._price_arr_cache), 2)
        strategies.invalidate_dir_fingerprints()

    def test_uncached_price_helpers_follow_in_place_mutation(self):
        rows = [{"close": "1", "turnover_rate": "2"}]
        first = strategies.price_arrays(rows)
        rows[0]["close"] = "9"
        rows.append({"close": "10", "turnover_rate": "3"})
        strategies.pretransform_price_rows(rows)
        second = strategies.price_arrays(rows)

        self.assertEqual(first["close"].tolist(), [1.0])
        self.assertEqual(second["close"].tolist(), [9.0, 10.0])
        self.assertTrue(all(type(row["close"]) is float for row in rows))

    def test_quarterly_ttm_yoy_matches_same_period_with_real_dates(self):
        records = [
            {"date": "2024-03-31", "revenue": 100},
            {"date": "2023-12-31", "revenue": 300},
            {"date": "2025-03-31", "revenue": 120},
            {"date": "2023-03-31", "revenue": 80},
            {"date": "2024-12-31", "revenue": 400},
        ]

        self.assertEqual(compute_ttm(records, "revenue"), 420.0)
        self.assertAlmostEqual(compute_yoy_growth(records, "revenue"), 0.3125)

    def test_balance_sheet_yoy_uses_previous_same_quarter_not_annual(self):
        records = [
            {"date": "2024-12-31", "total_assets": 130},
            {"date": "2025-03-31", "total_assets": 140},
            {"date": "2024-03-31", "total_assets": 100},
            {"date": "2023-12-31", "total_assets": 90},
        ]

        self.assertAlmostEqual(
            yoy_from_latest_and_annual(records, "total_assets"), 0.4
        )

    def test_short_event_date_and_follower_sample_use_full_sorted_signal(self):
        data = {
            "as_of_date": "2026-07-09",
            "capital": {
                "kline_as_of_date": "2026-07-10",
                "followers": [
                    {"seat": "B", "date": "2026-07-03", "buy_est": 2},
                    {"seat": "A", "date": "2026-07-10", "buy_est": 1},
                    {"seat": "C", "date": "2026-07-08", "buy_est": 3},
                ],
            },
        }

        self.assertEqual(strategies.short_signal_event_date(data), "2026-07-10")
        self.assertEqual(
            [row["date"] for row in strategies.follower_sample(data, limit=2)],
            ["2026-07-10", "2026-07-08"],
        )

    def test_factor_registry_has_required_depth(self):
        registry = get_factor_registry()

        self.assertGreaterEqual(registry["total"], 20)
        self.assertGreaterEqual(len(registry["long"]), 20)
        self.assertGreaterEqual(len(registry["smallcap"]), 20)
        self.assertGreaterEqual(len(registry["short"]), 20)

    def test_smallcap_registry_reuses_long_without_index_or_size_factors(self):
        registry = get_factor_registry()
        long_keys = {item["key"] for item in registry["long"]}
        smallcap_keys = {item["key"] for item in registry["smallcap"]}

        self.assertEqual(smallcap_keys, long_keys - SMALLCAP_EXCLUDED_FACTOR_KEYS)
        self.assertFalse(smallcap_keys & SMALLCAP_EXCLUDED_FACTOR_KEYS)
        self.assertFalse(
            set(get_default_config()["smallcap"]["weights"]) & SMALLCAP_EXCLUDED_FACTOR_KEYS
        )

    def test_long_factor_registry_exposes_capital_group(self):
        registry = get_factor_registry()
        long_by_key = {item["key"]: item for item in registry["long"]}
        defaults = get_default_config()["long"]["weights"]

        for key in ("holder_count_change", "repurchase_recent", "lhb_recent_avoid"):
            self.assertIn(key, long_by_key)
            self.assertEqual(long_by_key[key]["group"], "资金面")
            self.assertIn(key, defaults)
            self.assertGreater(defaults[key], 0)

    def test_default_config_falls_back_to_backup_optimized_config(self):
        payload = {
            "generated_at": "2026-06-25 14:00:00",
            "iterations_per_strategy": 1500,
            "seed": 42,
            "short_universe_version": "hotmoney_small_cap_v1",
            "config": {
                "long": {"top_n": 33},
                "short": {"top_n": 6},
            },
            "caveat": "test",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            primary = Path(tmpdir) / "data" / "stock_strategy_optimized_config.json"
            backup = Path(tmpdir) / "meta_data_backup" / "stock_strategy_optimized_config.json"
            backup.parent.mkdir(parents=True)
            backup.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            with patch("stock_advanced_strategies.OPTIMIZED_CONFIG_FILE", primary), \
                 patch("stock_advanced_strategies.OPTIMIZED_CONFIG_BACKUP_FILE", backup):
                config = get_default_config()

        self.assertEqual(config["long"]["top_n"], 33)
        self.assertEqual(config["short"]["top_n"], 6)
        self.assertEqual(config["_optimized_defaults"]["source"], str(backup))
        self.assertEqual(config["_optimized_defaults"]["iterations_per_strategy"], 1500)

    def test_loaded_short_horizon_requires_explicit_selected_hold_days(self):
        base_payload = {
            "generated_at": "2026-07-12 10:00:00",
            "short_universe_version": "hotmoney_small_cap_v1",
            "config": {"short": {"hold_days_min": 2, "hold_days_max": 3}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            primary = Path(tmpdir) / "stock_strategy_optimized_config.json"
            backup = Path(tmpdir) / "missing_backup.json"
            primary.write_text(json.dumps(base_payload), encoding="utf-8")
            with patch.object(strategies, "OPTIMIZED_CONFIG_FILE", primary), \
                 patch.object(strategies, "OPTIMIZED_CONFIG_BACKUP_FILE", backup):
                legacy = get_default_config()["short"]

            base_payload["config"]["short"]["hold_days"] = 4
            primary.write_text(json.dumps(base_payload), encoding="utf-8")
            with patch.object(strategies, "OPTIMIZED_CONFIG_FILE", primary), \
                 patch.object(strategies, "OPTIMIZED_CONFIG_BACKUP_FILE", backup):
                current = get_default_config()["short"]

        self.assertNotIn("hold_days", legacy)
        self.assertEqual((legacy["hold_days_min"], legacy["hold_days_max"]), (1, 5))
        self.assertEqual(current["hold_days"], 4)
        self.assertEqual((current["hold_days_min"], current["hold_days_max"]), (4, 4))

    def test_smallcap_default_falls_back_to_its_own_backup(self):
        payload = {
            "generated_at": "2026-07-12 10:00:00",
            "iterations": 1500,
            "seed": 42,
            "smallcap_universe_version": "hotmoney_small_cap_v1",
            "config": {
                "smallcap": {
                    "top_n": 17,
                    "weights": {
                        "roe_mean": 1.75,
                        "market_cap": 3.0,
                    },
                },
            },
            "caveat": "test",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            primary = Path(tmpdir) / "data" / "stock_strategy_smallcap_optimized_config.json"
            backup = Path(tmpdir) / "meta_data_backup" / "stock_strategy_smallcap_optimized_config.json"
            backup.parent.mkdir(parents=True)
            backup.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            with patch("stock_advanced_strategies.SMALLCAP_OPTIMIZED_CONFIG_FILE", primary), \
                 patch("stock_advanced_strategies.SMALLCAP_OPTIMIZED_CONFIG_BACKUP_FILE", backup):
                config = get_default_config()

        self.assertEqual(config["smallcap"]["top_n"], 10)
        self.assertEqual(config["smallcap"]["weights"]["roe_mean"], 1.75)
        self.assertNotIn("market_cap", config["smallcap"]["weights"])
        self.assertEqual(config["_smallcap_optimized_defaults"]["source"], str(backup))

    def test_smallcap_default_ignores_backup_for_wrong_universe(self):
        payload = {
            "smallcap_universe_version": "legacy",
            "config": {"smallcap": {"top_n": 99}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            primary = Path(tmpdir) / "smallcap.json"
            backup = Path(tmpdir) / "missing.json"
            primary.write_text(json.dumps(payload), encoding="utf-8")
            with patch("stock_advanced_strategies.SMALLCAP_OPTIMIZED_CONFIG_FILE", primary), \
                 patch("stock_advanced_strategies.SMALLCAP_OPTIMIZED_CONFIG_BACKUP_FILE", backup):
                config = get_default_config()

        self.assertEqual(config["smallcap"]["top_n"], 10)

    def test_default_config_ignores_short_config_from_legacy_universe(self):
        payload = {
            "generated_at": "2026-06-25 14:00:00",
            "config": {
                "long": {"top_n": 33},
                "short": {"top_n": 6, "min_lhb_count": 4},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            primary = Path(tmpdir) / "stock_strategy_optimized_config.json"
            backup = Path(tmpdir) / "missing_backup.json"
            primary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            with patch("stock_advanced_strategies.OPTIMIZED_CONFIG_FILE", primary), \
                 patch("stock_advanced_strategies.OPTIMIZED_CONFIG_BACKUP_FILE", backup):
                config = get_default_config()

        self.assertEqual(config["long"]["top_n"], 33)
        self.assertEqual(config["short"]["top_n"], 10)
        self.assertEqual(config["short"]["min_lhb_count"], 0)

    def test_csi300_persistence_proxy_rewards_current_member(self):
        current_member = csi300_persistence_proxy(True, True)
        broad_member = csi300_persistence_proxy(False, True)
        non_member = csi300_persistence_proxy(False, False)

        self.assertGreater(current_member, broad_member)
        self.assertGreater(broad_member, non_member)

    def test_csi300_persistence_does_not_hard_filter_long_pool(self):
        item = {
            "name": "测试股份",
            "raw_factors": {
                "market_cap": 500,
                "csi300_current": 0.0,
                "csi300_persistence": 0.0,
            },
        }
        config = {
            "exclude_st": True,
            "min_market_cap_yi": 100,
            "min_csi300_persistence": 100,
            "require_csi300": False,
            "require_high_drawdown": False,
        }

        self.assertTrue(passes_long_hard_filters(item, config))

    def test_rsi_sweetspot_prefers_attack_zone(self):
        self.assertEqual(rsi_sweetspot_score(60), 100.0)
        self.assertLess(rsi_sweetspot_score(90), rsi_sweetspot_score(60))
        self.assertLess(rsi_sweetspot_score(25), rsi_sweetspot_score(60))

    def test_apply_scores_handles_percentile_and_boolean_factors(self):
        specs = [
            FactorSpec("size", "Size", "long", "test", "bigger is better", 1.0),
            FactorSpec("risk", "Risk", "long", "test", "lower is better", 1.0, direction="low"),
            FactorSpec("member", "Member", "long", "test", "boolean", 1.0, score="boolean"),
        ]
        items = [
            {"code": "A", "raw_factors": {"size": 10, "risk": 3, "member": 1}, "data_quality": 1},
            {"code": "B", "raw_factors": {"size": 1, "risk": 9, "member": 0}, "data_quality": 1},
        ]

        scored = apply_scores(items, specs, {"size": 1, "risk": 1, "member": 1})

        self.assertEqual(scored[0]["code"], "A")
        self.assertGreater(scored[0]["score"], scored[1]["score"])

    def test_first_not_none_keeps_legitimate_zero(self):
        self.assertEqual(first_not_none(0, 5), 0.0)
        self.assertEqual(first_not_none(None, "abc", 3), 3.0)
        self.assertIsNone(first_not_none(None, float("nan")))

    def test_sw3_industry_label_uses_parent_and_segment_without_unknown(self):
        label, bucket, sw2, sw3, segment_code = industry_label_from_sw3({
            "segment_code": "851024",
            "parent_segment": "通信设备",
            "segment_name": "通信网络设备及器件",
        }, fallback="UNKNOWN")

        self.assertEqual(label, "通信设备 / 通信网络设备及器件")
        self.assertEqual(bucket, "通信网络设备及器件")
        self.assertEqual(sw2, "通信设备")
        self.assertEqual(sw3, "通信网络设备及器件")
        self.assertEqual(segment_code, "851024")
        self.assertEqual(industry_label_from_sw3(None, fallback="UNKNOWN")[0], "")

    def test_build_long_candidates_defaults_to_sw3_segment_leaders(self):
        cfg = get_default_config()["long"]
        cfg.update({"require_csi300": False, "min_market_cap_yi": 0})
        stocks = {
            "600000": {"name": "龙头股份", "financials": {}, "indicators": {}, "dividends": {}, "daily": {}},
            "600001": {"name": "普通股份", "financials": {}, "indicators": {}, "dividends": {}, "daily": {}},
        }
        cn_index = {
            code: {"records": [{"date": "2026-01-02", "close": 10.0, "market_cap": 100.0}]}
            for code in stocks
        }

        with patch("stock_advanced_strategies.load_segment_leader_codes", return_value={"600000"}), \
             patch("stock_advanced_strategies.load_fundamental_stocks", return_value=stocks), \
             patch("stock_advanced_strategies.load_cn_stock_index", return_value=cn_index), \
             patch("stock_advanced_strategies.load_stock_universe", return_value={}), \
             patch("stock_advanced_strategies.load_market_snapshot", return_value={}), \
             patch("stock_advanced_strategies.load_sw3_segment_map", return_value={}), \
             patch("stock_advanced_strategies.load_long_capital_signals", return_value=({}, {
                 "holder": False, "repurchase": False, "lhb": False,
             })):
            candidates, notes = build_long_candidates(cfg)

        self.assertEqual([item["code"] for item in candidates], ["600000"])
        self.assertTrue(any("SW3 细分龙头池" in note for note in notes))

    def test_build_smallcap_candidates_intersects_hot_money_pool(self):
        cfg = get_default_config()["smallcap"]
        base_candidates = [{
            "code": "000002",
            "name": "测试小盘",
            "strategy": "long",
            "raw_factors": {"roe_stability": 12.0},
            "reasons": [],
            "warnings": [],
        }]
        with patch("stock_advanced_strategies.load_hot_money_codes", return_value={"000001", "000002"}), \
             patch("stock_advanced_strategies.build_long_candidates", return_value=(base_candidates, [])) as build_long:
            candidates, notes = build_smallcap_candidates(cfg, universe={"000002", "000003"})

        self.assertEqual(build_long.call_args.kwargs["universe"], {"000002"})
        self.assertFalse(build_long.call_args.args[0]["use_segment_leaders"])
        self.assertFalse(build_long.call_args.args[0]["require_csi300"])
        self.assertEqual(build_long.call_args.args[0]["min_market_cap_yi"], 0)
        self.assertEqual(candidates[0]["strategy"], "smallcap")
        self.assertTrue(any("is_hot_money=1" in note for note in notes))

    def test_strip_internal_keeps_sw3_industry_fields_for_frontend(self):
        row = strip_internal([{
            "rank": 1,
            "code": "600000",
            "name": "浦发银行",
            "event_date": "2026-07-10",
            "industry": "银行 / 股份制银行",
            "sw2_industry": "银行",
            "sw3_industry": "股份制银行",
            "sw3_segment_code": "850111",
            "score": 88.0,
        }])[0]

        self.assertEqual(row["industry"], "银行 / 股份制银行")
        self.assertEqual(row["sw2_industry"], "银行")
        self.assertEqual(row["sw3_industry"], "股份制银行")
        self.assertEqual(row["sw3_segment_code"], "850111")
        self.assertEqual(row["event_date"], "2026-07-10")

    def test_institution_conflict_distinguishes_lhb_direction(self):
        self.assertEqual(institution_conflict_score(-1e7, 5e7), 35.0)
        self.assertEqual(institution_conflict_score(-1e7, -5e7), 20.0)
        self.assertEqual(institution_conflict_score(1e7, -5e7), 45.0)
        self.assertEqual(institution_conflict_score(1e7, 5e7), 85.0)
        self.assertIsNone(institution_conflict_score(None, None))

    def test_percentile_ties_share_the_same_score(self):
        specs = [FactorSpec("size", "Size", "long", "test", "bigger is better", 1.0)]
        items = [
            {"code": "A", "raw_factors": {"size": 5}, "data_quality": 1},
            {"code": "B", "raw_factors": {"size": 5}, "data_quality": 1},
            {"code": "C", "raw_factors": {"size": 1}, "data_quality": 1},
        ]

        scored = apply_scores(items, specs, {"size": 1})
        by_code = {row["code"]: row["score"] for row in scored}

        self.assertEqual(by_code["A"], by_code["B"])
        self.assertGreater(by_code["A"], by_code["C"])

    def test_top_n_zero_selects_nothing(self):
        result = run_strategies({
            "long": {"top_n": 0},
            "smallcap": {"top_n": 0},
            "short": {"top_n": 0},
        }, persist=False)

        self.assertEqual(result["long"]["selected_count"], 0)
        self.assertEqual(result["smallcap"]["selected_count"], 0)
        self.assertEqual(result["short"]["selected_count"], 0)

    def test_high_to_latest_drawdown_pct_uses_available_history_peak(self):
        rows = [{"close": 10}, {"close": 20}, {"close": 15}]

        self.assertEqual(high_to_latest_drawdown_pct(rows), 25.0)
        self.assertIsNone(high_to_latest_drawdown_pct([{"close": 10}]))

    def test_long_high_drawdown_filter_can_exclude_all_when_impossible(self):
        cfg = get_default_config()["long"]
        cfg.update({
            "require_high_drawdown": True,
            "min_high_drawdown_pct": 101,
            "min_score": 0,
            "top_n": 9999,
        })

        result = run_long_strategy(cfg)

        self.assertEqual(result["candidate_count"], 0)

    def test_high_drawdown_filter_keeps_passing_stock_score_stable(self):
        base = get_default_config()["long"]
        base.update({
            "require_csi300": False,
            "min_market_cap_yi": 500,
            "min_high_drawdown_pct": 10,
            "min_score": 0,
            "top_n": 9999,
        })

        without_required = dict(base)
        without_required["require_high_drawdown"] = False
        with_required = dict(base)
        with_required["require_high_drawdown"] = True

        loose = run_long_strategy(without_required)
        strict = run_long_strategy(with_required)
        loose_hikvision = next((row for row in loose["picks"] if row["code"] == "002415"), None)
        strict_hikvision = next((row for row in strict["picks"] if row["code"] == "002415"), None)
        if loose_hikvision is None or strict_hikvision is None:
            self.skipTest("local data does not include Hikvision passing the drawdown threshold")

        self.assertGreaterEqual(strict_hikvision["raw_factors"]["historical_high_drawdown"], 10)
        self.assertAlmostEqual(loose_hikvision["score"], strict_hikvision["score"], places=6)

    def test_require_csi300_keeps_current_member_score_stable(self):
        base = get_default_config()["long"]
        base.update({
            "min_market_cap_yi": 500,
            "min_score": 0,
            "top_n": 9999,
        })

        without_required = dict(base)
        without_required["require_csi300"] = False
        with_required = dict(base)
        with_required["require_csi300"] = True

        loose = run_long_strategy(without_required)
        strict = run_long_strategy(with_required)
        loose_moutai = next((row for row in loose["picks"] if row["code"] == "600519"), None)
        strict_moutai = next((row for row in strict["picks"] if row["code"] == "600519"), None)
        if loose_moutai is None or strict_moutai is None:
            self.skipTest("local data does not include Kweichow Moutai in the long pool")

        self.assertEqual(strict_moutai["raw_factors"]["csi300_current"], 1.0)
        self.assertAlmostEqual(loose_moutai["score"], strict_moutai["score"], places=6)

    def test_market_cap_floor_does_not_change_passing_stock_score(self):
        base = get_default_config()["long"]
        base.update({
            "require_csi300": False,
            "min_score": 0,
            "top_n": 9999,
        })

        loose = dict(base)
        loose["min_market_cap_yi"] = 100
        strict = dict(base)
        strict["min_market_cap_yi"] = 500

        loose_result = run_long_strategy(loose)
        strict_result = run_long_strategy(strict)
        loose_moutai = next((row for row in loose_result["picks"] if row["code"] == "600519"), None)
        strict_moutai = next((row for row in strict_result["picks"] if row["code"] == "600519"), None)
        if loose_moutai is None or strict_moutai is None:
            self.skipTest("local data does not include Kweichow Moutai above both market-cap floors")

        self.assertGreaterEqual(strict_moutai["raw_factors"]["market_cap"], 500)
        self.assertAlmostEqual(loose_moutai["score"], strict_moutai["score"], places=6)

    def test_report_available_date_uses_statutory_deadlines(self):
        self.assertEqual(report_available_date("2024-12-31"), "2025-04-30")
        self.assertEqual(report_available_date("2024-03-31"), "2024-04-30")
        self.assertEqual(report_available_date("2024-06-30"), "2024-08-31")
        self.assertEqual(report_available_date("2024-09-30"), "2024-10-31")

    def test_reports_available_asof_hides_unannounced_reports(self):
        recs = [{"date": "2025-12-31", "v": 2}, {"date": "2024-12-31", "v": 1}]
        # 2025-06-01: 2025年报(法定可见日 2026-04-30)尚未公布，只剩 2024 年报
        visible = reports_available_asof(recs, "2025-06-01")
        self.assertEqual([r["date"] for r in visible], ["2024-12-31"])
        # as_of=None 原样返回(保护实盘路径)
        self.assertIs(reports_available_asof(recs, None), recs)

    def test_compute_long_factors_pit_uses_only_visible_reports(self):
        live = compute_long_raw_factors(
            "X", PIT_SYNTH_STOCK, {}, False, False, 1e10, None,
        )
        pit = compute_long_raw_factors(
            "X", PIT_SYNTH_STOCK, {}, False, False, 1e10, None, as_of="2025-06-01",
        )
        # 实盘 roe_mean 用全部记录(均值20)；PIT 只看 2024 年报已可见(=10)
        self.assertAlmostEqual(live["roe_mean"], 20.0)
        self.assertAlmostEqual(pit["roe_mean"], 10.0)
        self.assertNotAlmostEqual(live["net_margin"], pit["net_margin"])
        self.assertAlmostEqual(live["liquidity"], 2.5)
        self.assertAlmostEqual(pit["liquidity"], 2.0)

    def test_compute_long_factors_maps_capital_signals(self):
        raw = compute_long_raw_factors(
            "X", PIT_SYNTH_STOCK, {}, False, False, 1e10, None,
            capital_signal={"holder_change": -8.5, "repurchase_recent": True, "lhb_recent": True},
            capital_available={"holder": True, "repurchase": True, "lhb": True},
        )

        self.assertEqual(raw["holder_count_change"], -8.5)
        self.assertEqual(raw["repurchase_recent"], 1.0)
        self.assertEqual(raw["lhb_recent_avoid"], 1.0)

        missing = compute_long_raw_factors(
            "X", PIT_SYNTH_STOCK, {}, False, False, 1e10, None,
            capital_signal={"holder_change": -8.5, "repurchase_recent": True, "lhb_recent": True},
            capital_available={"holder": False, "repurchase": False, "lhb": False},
        )
        self.assertIsNone(missing["holder_count_change"])
        self.assertIsNone(missing["repurchase_recent"])
        self.assertIsNone(missing["lhb_recent_avoid"])

    def test_run_strategies_returns_usable_sections(self):
        result = run_strategies(persist=False)

        self.assertIn("long", result)
        self.assertIn("smallcap", result)
        self.assertIn("short", result)
        self.assertGreaterEqual(result["factor_total"], 20)
        self.assertIsInstance(result["long"]["picks"], list)
        self.assertIsInstance(result["smallcap"]["picks"], list)
        self.assertIsInstance(result["short"]["picks"], list)
        self.assertTrue(result["self_review"]["factor_count_ok"])

    def test_run_smallcap_uses_long_factors_on_hot_money_candidates(self):
        cfg = get_default_config()["smallcap"]
        cfg.update({
            "min_score": 0,
            "top_n": 10,
            "require_high_drawdown": False,
            "exclude_st": False,
        })
        candidates = [{
            "code": "000001",
            "name": "测试股份",
            "strategy": "smallcap",
            "raw_factors": {"roe_mean": 12.0},
            "data_quality": 0.5,
            "reasons": [],
            "warnings": [],
        }]
        with patch("stock_advanced_strategies.live_smallcap_candidate_pool", return_value=(candidates, [])):
            result = run_smallcap_strategy(cfg)

        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["picks"][0]["strategy"], "smallcap")
        self.assertFalse(
            set(result["picks"][0]["factor_scores"]) & SMALLCAP_EXCLUDED_FACTOR_KEYS
        )


if __name__ == "__main__":
    unittest.main()
