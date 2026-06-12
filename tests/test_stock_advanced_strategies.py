import unittest

from stock_advanced_strategies import (
    FactorSpec,
    apply_scores,
    csi300_persistence_proxy,
    first_not_none,
    get_factor_registry,
    get_default_config,
    high_to_latest_drawdown_pct,
    institution_conflict_score,
    rsi_sweetspot_score,
    run_long_strategy,
    run_strategies,
)


class StockAdvancedStrategyTest(unittest.TestCase):
    def test_factor_registry_has_required_depth(self):
        registry = get_factor_registry()

        self.assertGreaterEqual(registry["total"], 20)
        self.assertGreaterEqual(len(registry["long"]), 20)
        self.assertGreaterEqual(len(registry["short"]), 20)

    def test_csi300_persistence_proxy_rewards_current_member_and_age(self):
        old_member = csi300_persistence_proxy(True, True, 12)
        young_member = csi300_persistence_proxy(True, True, 1)
        old_non_member = csi300_persistence_proxy(False, True, 12)

        self.assertGreater(old_member, young_member)
        self.assertGreater(old_member, old_non_member)

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
        result = run_strategies({"long": {"top_n": 0}, "short": {"top_n": 0}}, persist=False)

        self.assertEqual(result["long"]["selected_count"], 0)
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
            "min_listing_years": 12,
            "min_csi300_persistence": 60,
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
            "min_listing_years": 12,
            "min_csi300_persistence": 60,
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
            "min_listing_years": 12,
            "min_csi300_persistence": 60,
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

    def test_run_strategies_returns_usable_sections(self):
        result = run_strategies(persist=False)

        self.assertIn("long", result)
        self.assertIn("short", result)
        self.assertGreaterEqual(result["factor_total"], 20)
        self.assertIsInstance(result["long"]["picks"], list)
        self.assertIsInstance(result["short"]["picks"], list)
        self.assertTrue(result["self_review"]["factor_count_ok"])


if __name__ == "__main__":
    unittest.main()

