import json
import math
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import stock_storage
import stock_hot_money_radar as radar


def _seed_history(conn, code, bars):
    """bars: list of (date, high, low, close, volume, chg, turnover)。直接写 stock_history。"""
    conn.executemany(
        "INSERT INTO stock_history (code, date, daily_high, daily_low, daily_close, "
        "daily_volume, daily_change_pct, daily_turnover_rate) VALUES (?,?,?,?,?,?,?,?)",
        [(code, *bar) for bar in bars],
    )
    conn.commit()


def _flat_bars(n, price, vol, turnover, start_day=1):
    """平淡：价格/量恒定。"""
    return [(f"2026-{1 + (start_day + i)//28:02d}-{1 + (start_day + i)%28:02d}",
             price, price, price, vol, 0.0, turnover) for i in range(n)]


def _p25_bars():
    """构造“120日低位→60日严格横盘缩量→量能启动”的180根日线。"""
    closes = [15.0 - 4.0 * i / 119 for i in range(120)]
    closes.extend(10.95 - 0.75 * i / 39 for i in range(40))
    closes.extend(10.03 + 0.08 * i / 18 for i in range(19))
    closes.append(10.13)
    bars = []
    for i, close in enumerate(closes):
        previous = closes[i - 1] if i else close
        if i < 120:
            volume = 4200 - 8 * i
        elif i < 160:
            volume = 3000 - 35 * (i - 120)
        elif i < 179:
            volume = 1200 - 22 * (i - 160)
        else:
            volume = 2100
        bars.append({
            "date": f"d{i:03d}",
            "open": previous,
            "high": close * 1.004,
            "low": min(previous, close) * 0.985,
            "close": close,
            "volume": volume,
            "amount": volume * close,
            "chg": (close / previous - 1.0) * 100 if i else 0.0,
            "turnover": volume / 1000,
        })
    return bars


def _p1_low_base_bars():
    """第178根起底盘状态连续成立，最终第180根恰好首次三日确认。"""
    bars = []
    closes = [14.0 - 3.5 * i / 119 for i in range(120)]
    closes.extend(10.12 - 0.12 * i / 59 for i in range(60))
    for i, close in enumerate(closes):
        previous = closes[i - 1] if i else close
        spread = 0.018 if i < 160 else 0.003
        high = max(previous, close) * (1.0 + spread)
        low = min(previous, close) * (1.0 - spread)
        if i == 117:
            high = 14.0  # 直到该异常高点移出60日窗口，底盘状态才成立。
        bars.append({
            "date": f"d{i:03d}",
            "open": previous,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000.0,
            "amount": close * 1000.0,
            "chg": (close / previous - 1.0) * 100.0 if previous else 0.0,
            "turnover": 1.0,
        })
    return bars


def _p2_bars(event_count, include_today=True):
    """最后10根中放入指定数量十字星，其余为普通实体K线。"""
    bars = []
    end = 180 if include_today else 179
    event_indices = set(range(end - event_count, end))
    for i in range(180):
        if i in event_indices:
            open_price = close = 10.1
            high, low = 10.15, 10.05
        else:
            open_price, close = 10.0, 10.1
            high, low = 10.12, 9.99
        bars.append({
            "date": f"d{i:03d}",
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000.0,
            "amount": 10100.0,
            "chg": 0.0,
            "turnover": 1.0,
        })
    return bars


def _p3_reclaim_bars(event_volume=800.0, current_close=10.0, undercut_close=None):
    """缩量大阴后首次完整收复跌前收盘；事件日前20日均量为1000。"""
    closes = [10.0] * 31 + [9.5, 9.7, 9.9, current_close]
    if undercut_close is not None:
        closes[32] = undercut_close
    bars = []
    for index, close in enumerate(closes):
        previous = closes[index - 1] if index else close
        volume = event_volume if index == 31 else 1000.0
        bars.append({
            "date": f"d{index:03d}",
            "open": previous,
            "high": max(previous, close) * 1.003,
            "low": min(previous, close) * 0.997,
            "close": close,
            "volume": volume,
            "amount": close * volume,
            "chg": (close / previous - 1.0) * 100 if previous else 0.0,
            "turnover": 1.0,
        })
    return bars


def _p23_state_bars():
    """构造严格箱体、振幅和量能同时压缩的180根日线。"""
    bars = []
    for i in range(180):
        if i < 120:
            close = 12.0 - i / 120
        elif i < 160:
            close = 10.5
        else:
            close = 10.0
        compressed = i >= 160
        spread = 0.004 if compressed else 0.025
        volume = 700.0 if compressed else 1100.0
        bars.append({
            "date": f"d{i:03d}",
            "open": close,
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": volume,
            "amount": close * volume,
            "chg": 0.0,
            "turnover": volume / 1000,
        })
    return bars


def _p24_state_bars():
    """价格横住但上涨日显著放量，构造强OBV底背离。"""
    bars = []
    for i in range(180):
        if i < 149:
            close, volume = 11.0, 1000.0
        elif i == 149:
            close, volume = 10.0, 1000.0
        else:
            is_up = (i - 150) % 2 == 0
            close = 10.01 if is_up else 10.0
            volume = 2000.0 if is_up else 500.0
        previous = bars[-1]["close"] if bars else close
        bars.append({
            "date": f"d{i:03d}",
            "open": previous,
            "high": max(previous, close) * 1.003,
            "low": min(previous, close) * 0.997,
            "close": close,
            "volume": volume,
            "amount": close * volume,
            "chg": (close / previous - 1.0) * 100 if previous else 0.0,
            "turnover": volume / 1000,
        })
    return bars


class HotMoneyRadarTest(unittest.TestCase):
    def test_default_mode_is_ambush(self):
        self.assertEqual(radar.build_parser().parse_args([]).mode, "ambush")
        self.assertEqual(radar.build_parser().parse_args(["distribution"]).mode, "distribution")
        self.assertEqual(radar.build_parser().parse_args(["accumulation"]).mode, "accumulation")

    def test_pattern_significance_uses_hac_and_bh_fdr(self):
        positively_correlated = [0.02] * 10 + [-0.01] * 10
        iid_t, _iid_p = radar._newey_west_mean_test(positively_correlated, 0)
        hac_t, hac_p = radar._newey_west_mean_test(positively_correlated, 5)

        self.assertLess(abs(hac_t), abs(iid_t))
        self.assertGreater(hac_p, 0.05)
        self.assertEqual(
            radar._bh_fdr_threshold([0.001, 0.01, 0.04, 0.2], 0.10),
            0.04,
        )

    def test_pattern_effective_set_matches_latest_retest(self):
        expected = {"P3", "P11", "P12", "P13", "P14", "P16", "P17", "P19", "P20", "P22", "P24", "P26"}
        self.assertEqual(radar.PATTERN_EFFECTIVE, expected)
        self.assertEqual(set(radar.PATTERN_EFFECTIVE_STYLE), expected)

        catalog = {item["code"]: item for item in radar.pattern_catalog()}
        self.assertEqual(
            {code for code, item in catalog.items() if item["effective"]},
            expected,
        )
        self.assertTrue(all("复测" in catalog[code]["desc"] for code in expected))
        self.assertEqual(len(catalog), 26)
        self.assertIn("P6", catalog)
        self.assertIn("P7", catalog)
        self.assertFalse(catalog["P6"]["effective"])
        self.assertFalse(catalog["P7"]["effective"])
        self.assertIn("P25", catalog)
        self.assertFalse(catalog["P25"]["effective"])
        self.assertEqual(catalog["P1"]["score_usage"], "吸筹分 5%")
        self.assertEqual(catalog["P25"]["score_usage"], "吸筹分 5%")
        self.assertEqual(catalog["P1"]["validation_label"], "仅小盘有效")
        self.assertEqual(catalog["P25"]["validation_label"], "仅小盘有效")
        self.assertIn("P26", catalog)
        self.assertTrue(catalog["P26"]["effective"])
        self.assertEqual(catalog["P3"]["effective_style"], "bullish")
        self.assertEqual(catalog["P24"]["effective_style"], "bullish")
        self.assertEqual(catalog["P12"]["effective_style"], "momentum")
        self.assertEqual(catalog["P13"]["effective_style"], "momentum")
        self.assertEqual(catalog["P11"]["effective_style"], "risk")
        self.assertEqual(catalog["P26"]["effective_style"], "risk")
        self.assertEqual(catalog["P6"]["effective_style"], "neutral")

    def test_p25_fires_in_all_pools(self):
        bars = _p25_bars()
        hotmoney = {item["code"] for item in radar.match_patterns("000001", bars, pool="hotmoney")}
        leader = {item["code"] for item in radar.match_patterns("000001", bars, pool="leader")}
        unspecified = {item["code"] for item in radar.match_patterns("000001", bars)}

        self.assertIn("P25", hotmoney)
        self.assertIn("P25", leader)
        self.assertIn("P25", unspecified)

    def test_p26_requires_volume_confirmation_for_winner_risk(self):
        bars = _flat_bars(180, 10.0, 1000.0, 1.0)
        self.assertTrue(radar._pat_chip_winner_risk(
            bars, {
                "chip_winner": 0.90,
                "chip_winner_prior": 0.80,
                "p26_volume_ratio": 1.50,
            }
        ))
        self.assertTrue(radar._pat_chip_winner_risk(
            bars, {
                "chip_winner": 0.60,
                "chip_winner_prior": 0.399,
                "p26_volume_ratio": 1.50,
            }
        ))
        self.assertFalse(radar._pat_chip_winner_risk(
            bars, {
                "chip_winner": 0.95,
                "chip_winner_prior": 0.80,
                "p26_volume_ratio": 1.499,
            }
        ))
        self.assertFalse(radar._pat_chip_winner_risk(
            bars, {
                "chip_winner": 0.60,
                "chip_winner_prior": 0.40,
                "p26_volume_ratio": 2.00,
            }
        ))
        self.assertFalse(radar._pat_chip_winner_risk(
            bars, {
                "chip_winner": 0.599,
                "chip_winner_prior": 0.20,
                "p26_volume_ratio": 2.00,
            }
        ))

    def test_p26_volume_ratio_uses_today_over_previous_twenty_days(self):
        self.assertAlmostEqual(radar._p26_volume_ratio([100.0] * 20 + [150.0]), 1.5)
        self.assertIsNone(radar._p26_volume_ratio([100.0] * 20))
        self.assertIsNone(radar._p26_volume_ratio([None] * 6 + [100.0] * 14 + [200.0]))

    def test_p2_requires_five_events_in_latest_ten_bars(self):
        context = {"pos": 0.20}
        self.assertTrue(radar._pat_low_shadows(_p2_bars(5), context))
        self.assertFalse(radar._pat_low_shadows(_p2_bars(4), context))
        self.assertFalse(radar._pat_low_shadows(_p2_bars(5, include_today=False), context))
        self.assertFalse(radar._pat_low_shadows(_p2_bars(5), {"pos": 0.40}))

    def test_p2_fires_in_all_pools(self):
        bars = _p2_bars(5)
        context = {"pos": 0.20}
        hotmoney = {item["code"] for item in radar.match_patterns("000001", bars, ctx=context, pool="hotmoney")}
        leader = {item["code"] for item in radar.match_patterns("000001", bars, ctx=context, pool="leader")}
        unspecified = {item["code"] for item in radar.match_patterns("000001", bars, ctx=context)}

        self.assertIn("P2", hotmoney)
        self.assertIn("P2", leader)
        self.assertIn("P2", unspecified)

    def test_p3_fires_only_on_first_full_reclaim(self):
        bars = _p3_reclaim_bars()
        context = {
            "pos": 0.20,
            "vol": [bar["volume"] for bar in bars],
            "closes": [bar["close"] for bar in bars],
        }
        self.assertTrue(radar._pat_shakedown_absorb(bars, context))

        repeated = bars + [{**bars[-1], "date": "d035", "close": 10.01, "chg": 0.1}]
        repeated_context = {
            "pos": 0.20,
            "vol": [bar["volume"] for bar in repeated],
            "closes": [bar["close"] for bar in repeated],
        }
        self.assertFalse(radar._pat_shakedown_absorb(repeated, repeated_context))

    def test_p3_rejects_partial_reclaim_high_event_volume_and_deep_undercut(self):
        for bars in (
            _p3_reclaim_bars(current_close=9.99),
            _p3_reclaim_bars(event_volume=850.0),
            _p3_reclaim_bars(undercut_close=9.20),
        ):
            context = {
                "pos": 0.20,
                "vol": [bar["volume"] for bar in bars],
                "closes": [bar["close"] for bar in bars],
            }
            self.assertFalse(radar._pat_shakedown_absorb(bars, context))

    def test_p3_rejects_high_position_or_confirmation_volume(self):
        bars = _p3_reclaim_bars()
        closes = [bar["close"] for bar in bars]
        volumes = [bar["volume"] for bar in bars]
        self.assertFalse(radar._pat_shakedown_absorb(
            bars, {"pos": 0.35, "vol": volumes, "closes": closes}
        ))
        bars[-1]["volume"] = 1301.0
        volumes[-1] = 1301.0
        self.assertFalse(radar._pat_shakedown_absorb(
            bars, {"pos": 0.20, "vol": volumes, "closes": closes}
        ))

    def test_p23_uses_original_broad_compression_rule(self):
        bars = _p23_state_bars()
        with patch.object(radar, "_amp_ratio", return_value=0.79):
            self.assertTrue(radar._pat_compression(bars, {"pos": 0.59}))
            self.assertFalse(radar._pat_compression(bars, {"pos": 0.60}))
        with patch.object(radar, "_amp_ratio", return_value=0.80):
            self.assertFalse(radar._pat_compression(bars, {"pos": 0.30}))

    def test_p24_strong_obv_state(self):
        self.assertTrue(radar._p24_obv_divergence_state(_p24_state_bars()))

    def test_p24_only_fires_on_first_five_day_confirmation_in_all_pools(self):
        bars = _p23_state_bars()
        newly_confirmed = lambda prefix: len(prefix) >= 176
        already_old = lambda _prefix: True

        with patch.object(radar, "_p24_obv_divergence_state", side_effect=newly_confirmed):
            for pool in ("leader", "hotmoney", None):
                codes = {item["code"] for item in radar.match_patterns("000001", bars, pool=pool)}
                self.assertIn("P24", codes)

        with patch.object(radar, "_p24_obv_divergence_state", side_effect=already_old):
            codes = {item["code"] for item in radar.match_patterns("000001", bars)}
            self.assertNotIn("P24", codes)

    def test_verify_as_of_dates_keeps_recent_grid_when_window_expands(self):
        dates = [f"d{i:02d}" for i in range(25)]
        with patch.object(radar, "VERIFY_STEP", 10), patch.object(radar, "VERIFY_WINDOW_DAYS", 0):
            all_history = radar._verify_as_of_dates(dates)
        with patch.object(radar, "VERIFY_STEP", 10), patch.object(radar, "VERIFY_WINDOW_DAYS", 20):
            trailing = radar._verify_as_of_dates(dates)
        self.assertEqual(all_history, ["d05", "d15"])
        self.assertEqual(trailing, ["d05", "d15"])

    def test_distribution_model_formula(self):
        weights = {
            "p14": 0.10,
            "p16": 0.10,
            "p17": 0.15,
            "p19": 0.15,
            "p20": 0.05,
            "p22": 0.05,
            "lhb_recent": 0.10,
            "technical": 0.15,
            "divergence": 0.15,
        }
        features = radar._distribution_model_features(
            50.0, ["P14", "P17", "P22"], lhb_recent=True, divergence_score=80.0
        )
        self.assertEqual(radar._distribution_model_score(features, weights), 59.5)

        p19_only = radar._distribution_model_features(0.0, ["P19"], lhb_recent=False, divergence_score=0.0)
        self.assertEqual(radar._distribution_model_score(p19_only, weights), 15.0)

    def test_accumulation_model_uses_raw_feature_weights(self):
        weights = {
            "chip": 0.10,
            "position": 0.20,
            "cmf_eff": 0.10,
            "p3": 0.20,
            "p24": 0.10,
            "p1": 0.05,
            "p25": 0.05,
            "holder_change": 0.10,
            "repurchase": 0.10,
        }
        rows = [
            {
                "ambush_score": 12.0,
                "distribution_score": 0.0,
                "holder_change": -15.0,
                "repurchase_recent": True,
                "patterns": ["P1", "P3", "P24", "P25"],
                "signals": {},
                "sub_scores": {
                    "chip": 80.0,
                    "position": 70.0,
                    "cmf_eff": 40.0,
                },
            },
            {
                "ambush_score": 34.0,
                "distribution_score": 40.0,
                "holder_change": None,
                "repurchase_recent": False,
                "patterns": [],
                "signals": {},
                "sub_scores": {
                    "chip": 20.0,
                    "position": 30.0,
                    "cmf_eff": 50.0,
                },
            },
        ]
        self.assertEqual(radar.ACCUM_MODEL_WEIGHTS, weights)
        self.assertAlmostEqual(sum(radar.ACCUM_MODEL_WEIGHTS.values()), 1.0)
        radar._apply_accumulation_model(rows)

        self.assertEqual(rows[0]["ambush_score"], 86.0)
        self.assertEqual(rows[0]["accumulation_percentile"], 100.0)
        self.assertEqual(rows[0]["distribution_percentile"], 0.0)
        self.assertEqual(rows[0]["opportunity_score"], 100.0)
        self.assertNotIn("technical_ambush_score", rows[0])
        self.assertNotIn("technical_ambush_score", rows[0]["signals"])
        self.assertEqual(rows[0]["signals"]["accumulation_model_features"]["p3"], 100.0)
        self.assertEqual(rows[0]["signals"]["accumulation_model_features"]["p24"], 100.0)
        self.assertEqual(rows[0]["signals"]["accumulation_model_features"]["p1"], 100.0)
        self.assertEqual(rows[0]["signals"]["accumulation_model_features"]["p25"], 100.0)
        self.assertEqual(rows[0]["signals"]["accumulation_model_features"]["chip"], 80.0)
        self.assertEqual(rows[0]["signals"]["accumulation_model_features"]["holder_change"], 100.0)
        self.assertEqual(rows[0]["signals"]["accumulation_model_features"]["repurchase"], 100.0)
        self.assertEqual(rows[1]["accumulation_percentile"], 0.0)
        self.assertEqual(rows[1]["distribution_percentile"], 100.0)
        self.assertEqual(rows[1]["opportunity_score"], 0.0)
        self.assertEqual(rows[1]["signals"]["accumulation_model_features"]["p3"], 0.0)
        self.assertEqual(rows[1]["signals"]["accumulation_model_features"]["p24"], 0.0)
        self.assertEqual(rows[1]["signals"]["accumulation_model_features"]["p1"], 0.0)
        self.assertEqual(rows[1]["signals"]["accumulation_model_features"]["p25"], 0.0)
        self.assertEqual(rows[1]["signals"]["accumulation_model_features"]["holder_change"], 50.0)
        self.assertEqual(rows[0]["signals"]["accumulation_percentile"], 100.0)
        self.assertEqual(rows[0]["signals"]["distribution_percentile"], 0.0)
        self.assertEqual(rows[0]["signals"]["opportunity_formula"], radar.OPPORTUNITY_FORMULA)

    def test_accumulation_weight_grid_fixes_p1_and_p25_at_five_percent(self):
        grid = radar._accumulation_weight_grid()
        self.assertEqual(len(grid), 210)
        self.assertTrue(all(abs(sum(weights.values()) - 1.0) < 1e-9 for weights in grid))
        self.assertTrue(all(weights["p1"] == 0.05 for weights in grid))
        self.assertTrue(all(weights["p25"] == 0.05 for weights in grid))
        for weights in grid:
            for feature in radar.ACCUM_FEATURES:
                if feature not in {"p1", "p25"}:
                    self.assertGreaterEqual(weights[feature], 0.10)
        self.assertIn(radar.ACCUM_MODEL_WEIGHTS, grid)

    def test_cross_section_percentiles_share_tie_rank(self):
        rows = [{"score": 10.0}, {"score": 20.0}, {"score": 20.0}, {"score": 40.0}]
        self.assertEqual(
            radar._cross_section_percentiles(rows, "score"),
            {0: 0.0, 1: 50.0, 2: 50.0, 3: 100.0},
        )

    def test_ema_recent_first(self):
        # span=3 → alpha=0.5：acc = 0.5*today + 0.25*d-1 + 0.25*d-2（today=seq[0]）
        self.assertEqual(radar._ema_recent_first([5.0], 3), 5.0)         # 单点=原值
        self.assertEqual(radar._ema_recent_first([100.0, 100.0, 100.0], 3), 100.0)
        self.assertAlmostEqual(radar._ema_recent_first([100.0, 0.0, 0.0], 3), 50.0)

    def test_reversal_snapshot_mode_matches_reverse_rank(self):
        # REVERSAL_SMOOTH_DAYS=1 → 退回单日快照：reversal_score = 过热因子反向 rank 加权，snapshot==score。
        rows = [
            {"code": "A", **{f"rev_{f}": 0.0 for f in radar.REVERSAL_FEATURES}},   # 最冷 → 最高分
            {"code": "B", **{f"rev_{f}": 0.5 for f in radar.REVERSAL_FEATURES}},
            {"code": "C", **{f"rev_{f}": 1.0 for f in radar.REVERSAL_FEATURES}},   # 最热 → 最低分
        ]
        with patch.object(radar, "REVERSAL_SMOOTH_DAYS", 1):
            radar._apply_reversal_model(rows)
        self.assertEqual(rows[0]["reversal_score"], 100.0)   # 反向：最冷=100
        self.assertEqual(rows[2]["reversal_score"], 0.0)     # 最热=0
        for r in rows:
            self.assertEqual(r["signals"]["reversal_score_snapshot"], r["reversal_score"])
            self.assertEqual(r["signals"]["reversal_smooth_days"], 1)

    def test_reversal_ema_identity_when_history_flat(self):
        # rev_hist 三日因子恒定 → 每日截面子分相同 → EMA 平滑分 == 单日快照分（恒等）。
        def row(code, val):
            feats = {f: val for f in radar.REVERSAL_FEATURES}
            return {"code": code, "rev_hist": [dict(feats) for _ in range(3)],
                    **{f"rev_{f}": val for f in radar.REVERSAL_FEATURES}}
        rows = [row("A", 0.0), row("B", 0.5), row("C", 1.0)]
        with patch.object(radar, "REVERSAL_SMOOTH_DAYS", 3):
            radar._apply_reversal_model(rows)
        for r in rows:
            self.assertEqual(r["reversal_score"], r["signals"]["reversal_score_snapshot"])
            self.assertEqual(r["signals"]["reversal_smooth_days"], 3)
            self.assertNotIn("rev_hist", r)   # 用完即清，不进 payload

    def test_reversal_ema_lifts_score_for_fresh_spike(self):
        # 某票今日才转最热(snapshot 最低)、前两日不热 → EMA 把它的反转分抬到高于纯快照。
        def hist(code, today, prior):
            t = {f: today for f in radar.REVERSAL_FEATURES}
            p = {f: prior for f in radar.REVERSAL_FEATURES}
            return {"code": code, "rev_hist": [t, dict(p), dict(p)],
                    **{f"rev_{f}": today for f in radar.REVERSAL_FEATURES}}
        # 三票今日过热度 A<B<C；C 今日突然最热但前两日最冷。
        rows = [hist("A", 0.0, 0.0), hist("B", 0.5, 0.5), hist("C", 1.0, 0.0)]
        snap_rows = [dict(r) for r in rows]
        with patch.object(radar, "REVERSAL_SMOOTH_DAYS", 1):
            radar._apply_reversal_model(snap_rows)
        with patch.object(radar, "REVERSAL_SMOOTH_DAYS", 3):
            radar._apply_reversal_model(rows)
        c_snap = snap_rows[2]["reversal_score"]
        c_ema = rows[2]["reversal_score"]
        self.assertEqual(c_snap, 0.0)            # 纯快照：今日最热→0
        self.assertGreater(c_ema, c_snap)        # 平滑：前两日不热把分抬起

    def test_market_regime_prefers_accumulated_nav_to_avoid_split_break(self):
        conn = stock_storage.connect(":memory:")
        records = []
        base = date(2026, 1, 1)
        for i in range(25):
            nav_acc = 1.24 - i * 0.01
            records.append({
                "date": (base + timedelta(days=i)).isoformat(),
                "nav": f"{(2.0 if i == 24 else nav_acc):.4f}",
                "nav_acc": f"{nav_acc:.4f}",
            })
        stock_storage.save_index_nav(conn, {"code": radar.MARKET_REGIME_INDEX, "records": records})

        regime = radar._market_regime(conn)

        self.assertTrue(regime["available"])
        self.assertEqual(regime["value_field"], "nav_acc")
        self.assertEqual(regime["fallback_count"], 0)
        self.assertAlmostEqual(regime["value"], 1.0)
        self.assertFalse(regime["above_ma20"])
        self.assertFalse(regime["favorable"])
        self.assertAlmostEqual(regime["ret5"], -0.0476)
        conn.close()

    def test_p1_market_gate_uses_ma60_or_pit_ma20_repair(self):
        rising = [1.0 + 0.01 * i for i in range(60)]
        repair = [2.0] * 35 + [1.0 + 0.01 * i for i in range(25)]
        falling = [2.0 - 0.01 * i for i in range(60)]

        self.assertTrue(radar._p1_market_gate_metrics(rising)["gate"])
        repaired = radar._p1_market_gate_metrics(repair)
        self.assertFalse(repaired["above_ma60"])
        self.assertTrue(repaired["ma20_rising_5d"])
        self.assertTrue(repaired["gate"])
        self.assertFalse(radar._p1_market_gate_metrics(falling)["gate"])
        self.assertFalse(radar._p1_market_gate_metrics(falling[:59])["available"])

        conn = stock_storage.connect(":memory:")
        base = date(2026, 1, 1)
        records = [{
            "date": (base + timedelta(days=i)).isoformat(),
            "nav": f"{value:.4f}",
            "nav_acc": f"{value:.4f}",
        } for i, value in enumerate(repair)]
        stock_storage.save_index_nav(
            conn, {"code": radar.MARKET_REGIME_INDEX, "records": records}
        )
        gates = radar._p1_market_gate_by_date(
            conn, [records[58]["date"], records[59]["date"]]
        )
        self.assertFalse(gates[records[58]["date"]])  # 仅59根，禁止偷看下一日。
        self.assertTrue(gates[records[59]["date"]])
        regime = radar._market_regime(conn)
        self.assertTrue(regime["p1_trade_gate_available"])
        self.assertTrue(regime["p1_trade_gate"])
        self.assertEqual(regime["p1_trade_gate_reason"], "ma20_rising_5d")
        conn.close()

    def test_empty_pattern_phase_uses_watch_label(self):
        self.assertEqual(radar._pattern_phase([], score=12.3), "观望⚪")
        self.assertIn("观望⚪", radar.PHASE_ORDER)
        self.assertNotIn("空仓观望⚪", radar.PHASE_ORDER)
        self.assertEqual(radar._phase_confidence("观望⚪", [], 12.3, 0), 85.0)
        self.assertLess(radar._phase_confidence("观望⚪", [], 64.0, 10.0), 35.0)

    def test_distribution_warning_uses_cumulative_pattern_points(self):
        def pattern(code):
            return {"code": code, "name": code, "phase": "出货", "signal": "sell"}

        cases = [
            (("P14",), 2, False),
            (("P16",), 2, False),
            (("P17",), 3, True),
            (("P19",), 3, True),
            (("P20", "P22"), 2, False),
            (("P14", "P20"), 3, True),
            (("P16", "P22"), 3, True),
            (("P26",), 3, True),
            (("P14", "P16"), 4, True),
            (("P15", "P18"), 0, False),
        ]
        for codes, points, warned in cases:
            fired = [pattern(code) for code in codes]
            self.assertEqual(radar._distribution_warning_points(fired), points)
            self.assertEqual(radar._pattern_phase(fired, score=0) == "出货预警🔴", warned)

        duplicate = [pattern("P17"), pattern("P17")]
        self.assertEqual(radar._distribution_warning_points(duplicate), 3)

    def test_phase_confidence_uses_score_confirmation(self):
        buy = [{"code": "P3", "name": "缩量阴线打压吸筹", "phase": "吸筹", "signal": "buy"}]
        sell = [{"code": "P17", "name": "倒V反转", "phase": "出货", "signal": "sell"}]
        hold = [{"code": "P11", "name": "放量突破启动", "phase": "突破", "signal": "hold"}]

        self.assertGreater(
            radar._phase_confidence("吸筹🟢", buy, 70.0, 5.0),
            radar._phase_confidence("吸筹🟢", buy, 70.0, 60.0),
        )
        self.assertGreater(
            radar._phase_confidence("出货预警🔴", sell, 20.0, 70.0),
            radar._phase_confidence("出货预警🔴", sell, 70.0, 10.0),
        )
        self.assertLess(
            radar._phase_confidence("▲突破🟠", hold, 50.0, 70.0),
            radar._phase_confidence("▲突破🟠", hold, 50.0, 20.0),
        )

    def _run_with_db(self, seed):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "t.sqlite3"
            out_file = Path(tmpdir) / "ambush.json"
            conn = stock_storage.connect(db_file)
            seed(conn)
            conn.close()
            with patch.object(radar, "DB_FILE", db_file), \
                 patch.object(radar, "AMBUSH_RESULT_FILE", out_file):
                payload = radar.main(["ambush"])
            self.assertTrue(out_file.exists())
            return payload

    def test_empty_pool_is_handled(self):
        payload = self._run_with_db(lambda conn: None)
        self.assertEqual(payload["candidate_count"], 0)
        self.assertEqual(payload["status"], "empty")
        self.assertEqual(payload["stocks"], [])

    def test_scoring_rewards_accumulation_over_extended(self):
        base = date(2024, 1, 1)

        def chg_bars(prices, vols, strong=False):
            """strong=True → 收盘靠近K线上半部(买压正,CMF高)；否则收在中间。"""
            hi_f, lo_f = (1.005, 0.985) if strong else (1.01, 0.99)
            bars = []
            prev = prices[0]
            for t, (px, v) in enumerate(zip(prices, vols)):
                chg = (px / prev - 1.0) * 100.0 if t else 0.0
                bars.append(((base + timedelta(days=t)).isoformat(),
                             px * hi_f, px * lo_f, px, v, chg, v / 1000.0))
                prev = px
            return bars

        def seed(conn):
            stock_storage.save_sw3_membership(conn, {"segments": [{
                "segment_code": "850111.SI", "segment_name": "测试", "parent_segment": "测试",
                "members": [{"code": c, "name": n} for c, n in [
                    ("600000", "吸筹"), ("600001", "已拉升"), ("600002", "平淡"), ("600003", "封板")]],
            }]})
            stock_storage.mark_sw3_leaders(conn, ["600000", "600001", "600002", "600003"])

            # 吸筹：先从 16 阴跌到 9.2(65日)，再低位横盘 9.0 共 25 日(收盘买压强)，最后 5 日量×3
            acc_px = [16 - (16 - 9.2) * t / 64 for t in range(65)] + \
                     [9.0 + (0.05 if t % 2 else -0.05) for t in range(25)]
            acc_vol = [1000.0] * 85 + [3000.0] * 5
            _seed_history(conn, "600000", chg_bars(acc_px, acc_vol, strong=True))

            # 已拉升：单边走高创新高，最后 5 日放量（右侧追高）
            ext_px = [8.0 + 8.0 * t / 89 for t in range(90)]
            ext_vol = [1000.0] * 85 + [3000.0] * 5
            _seed_history(conn, "600001", chg_bars(ext_px, ext_vol))

            # 平淡：恒定价、恒定低量
            _seed_history(conn, "600002", chg_bars([10.0] * 90, [1000.0] * 90))

            # 封板：横盘后最后一天一字涨停（振幅 0）
            sealed = chg_bars([8.0] * 89, [1500.0] * 89)
            sealed.append(((base + timedelta(days=89)).isoformat(), 8.8, 8.8, 8.8, 200.0, 10.0, 0.3))
            _seed_history(conn, "600003", sealed)
            conn.execute("CREATE TABLE shareholder_count (code TEXT, disclose_date TEXT, change_pct REAL)")
            conn.execute("CREATE TABLE repurchase (code TEXT, disclose_date TEXT)")
            conn.executemany(
                "INSERT INTO shareholder_count (code, disclose_date, change_pct) VALUES (?, ?, ?)",
                [
                    ("600000", "2024-03-20", -15.0),
                    ("600001", "2024-03-20", 15.0),
                ],
            )
            conn.execute(
                "INSERT INTO repurchase (code, disclose_date) VALUES (?, ?)",
                ("600000", "2024-03-20"),
            )
            conn.commit()

        payload = self._run_with_db(seed)
        self.assertEqual(payload["scored_count"], 4)
        by_code = {s["code"]: s for s in payload["stocks"]}

        # 吸筹分 > 已拉升 / 平淡，且排第一
        self.assertEqual(payload["stocks"][0]["code"], "600000")
        self.assertGreater(by_code["600000"]["ambush_score"], by_code["600001"]["ambush_score"])
        self.assertGreater(by_code["600000"]["ambush_score"], by_code["600002"]["ambush_score"])
        # 三个判别信号都算出来了
        acc = by_code["600000"]
        self.assertIsNotNone(acc["sub_scores"]["divergence"])
        self.assertIsNotNone(acc["sub_scores"]["cmf"])
        self.assertIsNotNone(acc["sub_scores"]["chip"])
        self.assertIsNotNone(acc["sub_scores"]["chip_peak_low"])
        self.assertIsNotNone(acc["sub_scores"]["chip_price_near_peak"])
        self.assertIsNotNone(acc["sub_scores"]["chip_winner_mid_low"])
        self.assertGreater(acc["signals"]["cmf"], 0)          # 收盘买压为正
        self.assertLess(acc["signals"]["close_pctile"], 0.6)  # 中低位
        for signal in ("vol_ratio", "close_pctile", "turnover_pctile", "cmf", "chip_concentration", "chip_winner", "limit_streak"):
            self.assertIn(signal, acc["signals"])
        self.assertIsNotNone(acc["signals"]["chip_peak_pctile"])
        self.assertIsNotNone(acc["signals"]["chip_price_to_peak"])
        self.assertEqual(acc["signals"]["accumulation_model_features"]["holder_change"], 100.0)
        self.assertEqual(acc["signals"]["accumulation_model_features"]["repurchase"], 100.0)
        self.assertEqual(
            acc["signals"]["accumulation_model_features"]["chip"],
            acc["sub_scores"]["chip"],
        )
        self.assertGreaterEqual(by_code["600001"]["signals"]["close_pctile"], 0.9)  # 已拉升在高位
        # 封板：不再给旧技术分打折，但仍识别并标记为已启动。
        self.assertEqual(by_code["600003"]["signals"]["sealed_recent"], 1)
        self.assertNotIn("sealed_penalty", by_code["600003"]["sub_scores"])
        self.assertIn("启动", by_code["600003"]["state"])

    def test_ambush_adds_real_sw2_heat_pctile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "t.sqlite3"
            out_file = Path(tmpdir) / "ambush.json"
            theme_file = Path(tmpdir) / "theme_candidates.json"
            conn = stock_storage.connect(db_file)
            stock_storage.save_sw3_membership(conn, {"segments": [{
                "segment_code": "850111.SI", "segment_name": "细分测试", "parent_segment": "测试",
                "members": [{"code": "600000", "name": "样本"}],
            }]})
            stock_storage.mark_sw3_leaders(conn, ["600000"])
            _seed_history(conn, "600000", _flat_bars(90, 10.0, 1000.0, 1.0))
            conn.close()
            theme_file.write_text(json.dumps({
                "generated_at": "2026-06-21 00:00:00",
                "theme_rankings": [
                    {"rank": 1, "plate_code": "A", "plate_name": "测试"},
                    {"rank": 2, "plate_code": "B", "plate_name": "相似"},
                ],
                "stock_themes": [{
                    "code": "600000",
                    "tracking_theme": "相似",
                    "tracking_theme_code": "B",
                }],
            }, ensure_ascii=False), encoding="utf-8")

            with patch.object(radar, "DB_FILE", db_file), \
                 patch.object(radar, "AMBUSH_RESULT_FILE", out_file), \
                 patch.object(radar, "THEME_CANDIDATES_FILE", theme_file):
                payload = radar.main(["ambush"])

        row = payload["stocks"][0]
        self.assertEqual(row["parent_segment"], "测试")
        self.assertEqual(row["tracking_theme"], "相似")
        self.assertEqual(row["theme_heat_pctile"], 0.0)
        self.assertEqual(row["sw2_heat_pctile"], 100.0)

    def test_realtime_quote_parsers_support_tencent_and_sina(self):
        tencent = (
            'v_sh600000="1~浦发银行~600000~9.00~8.89~8.85~544687~323481~221557~'
            '9.00~1384~8.99~2223~8.98~911~8.97~602~8.96~374~9.01~655~9.02~17648~'
            '9.03~10161~9.04~6039~9.05~10338~~20260708161454~0.11~1.24~9.03~8.79~'
            '9.00/544687/488055514~544687~48806~0.16~5.96~~9.03~8.79~2.70~2997.53~2997.53";'
        )
        tq = radar._parse_tencent_quote_text(tencent)["600000"]
        self.assertEqual(tq["source"], "tencent_batch")
        self.assertEqual(tq["price"], 9.0)
        self.assertEqual(tq["quote_date"], "2026-07-08")
        self.assertEqual(tq["amount"], 488055514.0)
        self.assertEqual(tq["turnover"], 0.16)

        sina = (
            'var hq_str_sh600000="浦发银行,8.850,8.890,9.000,9.030,8.790,9.000,'
            '9.010,54468742,488055514.000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'
            '2026-07-08,15:34:59,00";'
        )
        sq = radar._parse_sina_quote_text(sina)["600000"]
        self.assertEqual(sq["source"], "sina_batch")
        self.assertEqual(sq["price"], 9.0)
        self.assertEqual(sq["volume"], 544687.42)
        self.assertAlmostEqual(sq["change_pct"], 1.2373, places=3)

    def test_realtime_bar_projects_raw_quote_to_qfq_basis(self):
        bars = [
            {"date": "2026-07-07", "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0, "volume": 1000.0, "amount": 10000.0, "chg": 0.0, "turnover": 1.0},
            {"date": "2026-07-08", "open": 10.1, "high": 10.5, "low": 10.0, "close": 10.2, "volume": 1200.0, "amount": 12000.0, "chg": 2.0, "turnover": 1.2},
        ]
        bar = radar._quote_to_realtime_bar(bars, {
            "price": 110.0,
            "pre_close": 100.0,
            "change_pct": 10.0,
            "open": 101.0,
            "high": 112.0,
            "low": 98.0,
            "volume": 2000.0,
            "amount": 220000.0,
            "turnover": 2.0,
            "quote_date": "2026-07-08",
        })

        self.assertEqual(bar["price_adjust"], "qfq_intraday_from_change_pct")
        self.assertAlmostEqual(bar["close"], 11.0)
        self.assertAlmostEqual(bar["open"], 10.1)
        self.assertAlmostEqual(bar["high"], 11.2)
        self.assertAlmostEqual(bar["low"], 9.8)
        self.assertEqual(bar["raw_close"], 110.0)
        self.assertEqual(bar["volume"], 2000.0)
        self.assertFalse(bar["volume_projected"])
        self.assertEqual(bar["volume_projection_factor"], 1.0)
        self.assertEqual(bar["volume_projection_method"], "none")

    def test_realtime_bar_projects_intraday_volume_with_u_shape_curve_for_factors(self):
        self.assertAlmostEqual(radar._u_shape_volume_cumulative_share(60.0), 0.33)
        self.assertAlmostEqual(radar._u_shape_volume_cumulative_share(180.0), 0.68)
        bars = [
            {"date": "2026-07-07", "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0, "volume": 1000.0, "amount": 10000.0, "chg": 0.0, "turnover": 1.0},
        ]
        bar = radar._quote_to_realtime_bar(bars, {
            "price": 11.0,
            "pre_close": 10.0,
            "change_pct": 10.0,
            "open": 10.1,
            "high": 11.2,
            "low": 9.8,
            "volume": 2000.0,
            "amount": 220000.0,
            "turnover": 2.0,
            "quote_time": "2026-07-08 10:30:00",
            "quote_date": "2026-07-08",
        }, now=datetime(2026, 7, 8, 10, 30))

        self.assertTrue(bar["volume_projected"])
        self.assertEqual(bar["volume_elapsed_minutes"], 60.0)
        self.assertEqual(bar["volume_projection_method"], "u_shape_intraday")
        self.assertAlmostEqual(bar["volume_projection_factor"], 3.0303, places=4)
        self.assertEqual(bar["raw_volume"], 2000.0)
        self.assertAlmostEqual(bar["volume"], 6060.606, places=3)
        self.assertEqual(bar["raw_turnover"], 2.0)
        self.assertAlmostEqual(bar["turnover"], 6.0606, places=4)
        self.assertEqual(bar["raw_amount"], 220000.0)
        self.assertAlmostEqual(bar["amount"], 666666.667, places=3)

    def test_stale_quote_is_not_projected_as_intraday_volume(self):
        factor, elapsed, projected = radar._intraday_volume_projection(
            {
                "quote_time": "2026-07-08 10:30:00",
                "quote_date": "2026-07-08",
            },
            now=datetime(2026, 7, 9, 10, 30),
        )

        self.assertEqual(factor, 1.0)
        self.assertEqual(elapsed, 60.0)
        self.assertFalse(projected)

    def test_realtime_merge_rejects_multi_session_local_history_gap(self):
        bars = [{
            "date": "2026-07-08", "open": 10.0, "high": 10.2, "low": 9.8,
            "close": 10.0, "volume": 1000.0, "amount": 10000.0,
            "chg": 0.0, "turnover": 1.0,
        }]
        quote = {
            "price": 11.0, "pre_close": 10.5, "change_pct": 4.7619,
            "open": 10.6, "high": 11.1, "low": 10.4,
            "quote_date": "2026-07-10",
        }

        merged, used = radar._merge_realtime_quote_bars(bars, quote)

        self.assertFalse(used)
        self.assertEqual(merged, bars)
        self.assertIsNone(radar._quote_to_realtime_bar(bars, quote))

    def test_realtime_merge_allows_friday_to_monday(self):
        bars = [{
            "date": "2026-07-10", "open": 10.0, "high": 10.2, "low": 9.8,
            "close": 10.0, "volume": 1000.0, "amount": 10000.0,
            "chg": 0.0, "turnover": 1.0,
        }]
        quote = {
            "price": 10.5, "pre_close": 10.0, "change_pct": 5.0,
            "open": 10.1, "high": 10.6, "low": 10.0,
            "quote_date": "2026-07-13",
        }

        merged, used = radar._merge_realtime_quote_bars(bars, quote)

        self.assertTrue(used)
        self.assertEqual(merged[-1]["date"], "2026-07-13")
        self.assertAlmostEqual(merged[-1]["close"], 10.5)

    def test_candidate_scoring_reuses_current_chip_metrics(self):
        bars = [
            {
                "date": day,
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": close * volume * 100.0,
                "chg": change,
                "turnover": turnover,
            }
            for day, high, low, close, volume, change, turnover in _gen_bars(2, n=120)
        ]

        with patch.object(radar, "_chip_metrics", wraps=radar._chip_metrics) as chip_metrics:
            result = radar._score_candidate_from_bars({"code": "600000"}, bars, pool="leader")

        self.assertEqual(result["score_status"], "OK")
        # Current window + five-days-prior risk window.  The score and pattern
        # paths must share the current computation instead of making a third.
        self.assertEqual(chip_metrics.call_count, 2)

    def test_ambush_scores_each_stock_with_gate_at_its_own_last_bar(self):
        candidates = [{"code": "600000"}, {"code": "600001"}]
        bars_by_code = {
            "600000": [{"date": "2026-07-01"}],
            "600001": [{"date": "2026-07-10"}],
        }
        seen = []

        def fake_recent(_conn, code, **_kwargs):
            return bars_by_code[code]

        def fake_score(cand, bars, **kwargs):
            seen.append((cand["code"], bars[-1]["date"], kwargs["p1_market_gate"]))
            return {"code": cand["code"]}

        with patch.object(radar, "_recent_bars", side_effect=fake_recent), \
             patch.object(radar, "_p1_market_gate_by_date", return_value={
                 "2026-07-01": False,
                 "2026-07-10": True,
             }) as gate_map, \
             patch.object(radar, "_score_candidate_from_bars", side_effect=fake_score):
            radar._score_ambush_candidates(object(), candidates, None, "leader")

        self.assertEqual(seen, [
            ("600000", "2026-07-01", False),
            ("600001", "2026-07-10", True),
        ])
        self.assertEqual(
            set(gate_map.call_args.args[1]), {"2026-07-01", "2026-07-10"}
        )

    def test_realtime_rescore_uses_eastmoney_quote_without_persisting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "t.sqlite3"
            out_file = Path(tmpdir) / "ambush.json"
            conn = stock_storage.connect(db_file)
            stock_storage.save_sw3_membership(conn, {"segments": [{
                "segment_code": "850111.SI", "segment_name": "细分测试", "parent_segment": "测试",
                "members": [{"code": "600000", "name": "样本"}],
            }]})
            stock_storage.mark_sw3_leaders(conn, ["600000"])
            _seed_history(conn, "600000", _flat_bars(90, 10.0, 1000.0, 1.0))
            conn.close()

            with patch.object(radar, "DB_FILE", db_file), \
                 patch.object(radar, "AMBUSH_RESULT_FILE", out_file):
                payload = radar.main(["ambush"])
                latest = payload["stocks"][0]["last_date"]
                live = radar.realtime_rescore_payload(payload, {
                    "600000": {
                        "code": "600000",
                        "price": 10.8,
                        "change_pct": 8.0,
                        "open": 10.1,
                        "high": 10.9,
                        "low": 10.0,
                        "volume": 2000.0,
                        "amount": 21600000.0,
                        "turnover": 2.2,
                        "market_cap_yi": 88.0,
                        "quote_time": f"{latest} 14:58:00",
                        "quote_date": latest,
                    },
                }, fetched_at="2026-07-08 14:58:00")

        self.assertEqual(live["realtime_quote"]["matched_count"], 1)
        self.assertEqual(live["realtime_quote"]["used_count"], 1)
        row = live["stocks"][0]
        self.assertEqual(row["realtime_status"], "UPDATED")
        self.assertEqual(row["realtime_price"], 10.8)
        self.assertEqual(row["realtime_change_pct"], 8.0)
        self.assertEqual(row["realtime_price_adjust"], "qfq_intraday_from_change_pct")
        self.assertAlmostEqual(row["realtime_adjusted_close"], 10.8)
        self.assertEqual(row["market_cap_yi"], 88.0)
        self.assertEqual(row["last_date"], latest)

    def test_realtime_rescore_does_not_apply_latest_gate_to_stale_bar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "t.sqlite3"
            conn = stock_storage.connect(db_file)
            _seed_history(conn, "600000", _flat_bars(90, 10.0, 1000.0, 1.0))
            last_date = conn.execute(
                "SELECT MAX(date) FROM stock_history WHERE code='600000'"
            ).fetchone()[0]
            conn.close()
            seen = []

            def fake_score(cand, bars, **kwargs):
                seen.append((bars[-1]["date"], kwargs["p1_market_gate"]))
                return {
                    **cand,
                    "ambush_score": 1.0,
                    "patterns": [],
                    "signals": {},
                    "triggered": False,
                    "pattern_phase": "观望⚪",
                }

            with patch.object(radar, "DB_FILE", db_file), \
                 patch.object(radar, "_market_regime", return_value={
                     "available": True, "p1_trade_gate": True,
                 }), \
                 patch.object(radar, "_p1_market_gate_by_date", return_value={last_date: False}), \
                 patch.object(radar, "_score_candidate_from_bars", side_effect=fake_score), \
                 patch.object(radar, "_apply_distribution_model"), \
                 patch.object(radar, "_attach_capital_evidence"), \
                 patch.object(radar, "_apply_accumulation_model"), \
                 patch.object(radar, "_apply_reversal_model"):
                live = radar.realtime_rescore_payload({
                    "pool": "hotmoney",
                    "stocks": [{"code": "600000", "name": "停牌样本"}],
                }, {})

        self.assertEqual(seen, [(last_date, False)])
        self.assertTrue(live["market_regime"]["p1_trade_gate"])

    def test_realtime_lhb_recent_uses_latest_trade_date_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "t.sqlite3"
            out_file = Path(tmpdir) / "ambush.json"
            conn = stock_storage.connect(db_file)
            stock_storage.save_sw3_membership(conn, {"segments": [{
                "segment_code": "850111.SI", "segment_name": "细分测试", "parent_segment": "测试",
                "members": [
                    {"code": "600000", "name": "老上榜"},
                    {"code": "600001", "name": "近上榜"},
                ],
            }]})
            stock_storage.mark_sw3_leaders(conn, ["600000", "600001"])
            _seed_history(conn, "600000", _flat_bars(90, 10.0, 1000.0, 1.0))
            _seed_history(conn, "600001", _flat_bars(90, 10.0, 1000.0, 1.0))
            latest = conn.execute("SELECT MAX(date) FROM stock_history").fetchone()[0]
            old_date = (datetime.strptime(latest, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")
            recent_date = (datetime.strptime(latest, "%Y-%m-%d") - timedelta(days=10)).strftime("%Y-%m-%d")
            conn.execute("CREATE TABLE lhb_all (code TEXT, date TEXT)")
            conn.executemany(
                "INSERT INTO lhb_all (code, date) VALUES (?, ?)",
                [("600000", old_date), ("600001", recent_date)],
            )
            conn.commit()
            conn.close()

            with patch.object(radar, "DB_FILE", db_file), \
                 patch.object(radar, "AMBUSH_RESULT_FILE", out_file):
                payload = radar.main(["ambush"])

        by_code = {row["code"]: row for row in payload["stocks"]}
        self.assertFalse(by_code["600000"]["lhb_recent"])
        self.assertTrue(by_code["600001"]["lhb_recent"])
        self.assertEqual(by_code["600000"]["distribution_score"], 7.5)
        self.assertEqual(by_code["600001"]["distribution_score"], 17.5)
        self.assertEqual(by_code["600001"]["signals"]["distribution_model_features"]["technical"], 0.5)
        self.assertEqual(by_code["600000"]["signals"]["distribution_model_features"]["lhb_recent"], 0.0)
        self.assertEqual(by_code["600001"]["signals"]["distribution_model_features"]["lhb_recent"], 1.0)
        self.assertEqual(by_code["600001"]["signals"]["distribution_model_weights"]["lhb_recent"], 0.1)
        self.assertEqual(by_code["600001"]["signals"]["distribution_model_weights"]["divergence"], 0.15)
        self.assertEqual(payload["capital_counts"]["lhb_avoid"], 1)
        self.assertFalse(any("龙虎榜" in e["label"] for e in by_code["600000"]["evidence"]))
        self.assertTrue(any("近期上龙虎榜" in e["label"] for e in by_code["600001"]["evidence"]))


def _gen_bars(seed, n=240):
    """生成 n 根共享日历、随 seed 变化的有效日线，供 verify 回测测试（持有期长需更多历史）。"""
    base = date(2024, 1, 1)
    bars = []
    prev = 10.0
    for t in range(n):
        close = max(2.0, 10.0 + math.sin((t + seed) / 7.0) + t * 0.01 * (seed - 2))
        chg = (close / prev - 1.0) * 100.0 if t else 0.0
        vol = 1000.0 * (1.0 + 0.3 * math.sin((t + seed) / 5.0)) + (500.0 * seed if t > n - 6 else 0.0)
        bars.append(((base + timedelta(days=t)).isoformat(),
                     close * 1.015, close * 0.985, close, vol, chg, vol / 1000.0))
        prev = close
    return bars


class HotMoneyRadarVerifyTest(unittest.TestCase):
    def test_verify_backtest_runs_and_computes_ic(self):
        codes = ["600000", "600001", "600002", "600003", "600004", "600005"]

        def seed(conn):
            stock_storage.save_sw3_membership(conn, {"segments": [{
                "segment_code": "850111.SI", "segment_name": "测试", "parent_segment": "测试",
                "members": [{"code": c, "name": f"票{c}"} for c in codes],
            }]})
            stock_storage.mark_sw3_leaders(conn, codes)
            for k, c in enumerate(codes):
                _seed_history(conn, c, _gen_bars(k))

        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "t.sqlite3"
            out_file = Path(tmpdir) / "verify.json"
            conn = stock_storage.connect(db_file)
            seed(conn)
            conn.close()
            with patch.object(radar, "DB_FILE", db_file), \
                 patch.object(radar, "VERIFY_RESULT_FILE", out_file), \
                 patch.object(radar, "VERIFY_MIN_NAMES", 5), \
                 patch.object(radar, "VERIFY_STEP", 3):
                payload = radar.main(["verify"])
            self.assertTrue(out_file.exists())

        self.assertEqual(payload["status"], "ok")
        self.assertGreater(payload["sample_count"], 0)
        self.assertGreaterEqual(payload["section_count"], 1)
        for h in radar.VERIFY_HORIZONS:
            hz = payload["horizons"][str(h)]
            self.assertEqual(len(hz["quantile_returns"]), radar.VERIFY_BUCKETS)
        # 至少一个持有期能算出截面 IC（每截面 6 只 ≥ 5）
        self.assertTrue(any(payload["horizons"][str(h)]["ic_mean"] is not None
                            for h in radar.VERIFY_HORIZONS))

    def test_verify_empty_pool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "t.sqlite3"
            out_file = Path(tmpdir) / "verify.json"
            stock_storage.connect(db_file).close()
            with patch.object(radar, "DB_FILE", db_file), \
                 patch.object(radar, "VERIFY_RESULT_FILE", out_file):
                payload = radar.main(["verify"])
            self.assertEqual(payload["status"], "empty")
            self.assertEqual(payload["sample_count"], 0)
            self.assertTrue(out_file.exists())


class Sw3LeaderFlagTest(unittest.TestCase):
    def test_mark_and_read_leaders(self):
        conn = stock_storage.connect(":memory:")
        stock_storage.save_sw3_membership(conn, {
            "segments": [{
                "segment_code": "850111.SI", "segment_name": "种子", "parent_segment": "种植业",
                "members": [{"code": "000998", "name": "隆平高科"}, {"code": "600598", "name": "北大荒"}],
            }],
        })
        # 初始无龙头
        self.assertEqual(stock_storage.leader_members(conn), [])
        # 打标一只
        self.assertEqual(stock_storage.mark_sw3_leaders(conn, ["000998"]), 1)
        leaders = stock_storage.leader_members(conn)
        self.assertEqual([l["code"] for l in leaders], ["000998"])
        self.assertEqual(leaders[0]["segment_name"], "种子")
        # 重新打标会清掉旧的
        self.assertEqual(stock_storage.mark_sw3_leaders(conn, ["600598"]), 1)
        self.assertEqual([l["code"] for l in stock_storage.leader_members(conn)], ["600598"])
        conn.close()

    def test_leader_members_fills_missing_market_cap_from_history(self):
        conn = stock_storage.connect(":memory:")
        stock_storage.save_sw3_membership(conn, {
            "segments": [{
                "segment_code": "850111.SI", "segment_name": "种子", "parent_segment": "种植业",
                "members": [{"code": "000998", "name": "隆平高科", "market_cap_yi": None}],
            }],
        })
        stock_storage.mark_sw3_leaders(conn, ["000998"])
        conn.executemany(
            "INSERT INTO stock_history (code, date, market_cap) VALUES (?, ?, ?)",
            [
                ("000998", "2026-01-01", 111.0),
                ("000998", "2026-01-02", 222.5),
                ("000998", "2026-01-03", None),
            ],
        )

        leaders = stock_storage.leader_members(conn)

        self.assertEqual(leaders[0]["market_cap_yi"], 222.5)
        conn.close()


def _pat_bar(t, o, h, l, c, vol, prev):
    chg = (c / prev - 1.0) * 100.0 if prev else 0.0
    return {"date": (date(2024, 1, 1) + timedelta(days=t)).isoformat(),
            "open": o, "high": h, "low": l, "close": c,
            "volume": vol, "amount": c * vol, "chg": chg, "turnover": vol / 1000.0}


class PatternMatchTest(unittest.TestCase):
    def test_p1_requires_first_low_base_confirmation_chip_cost_and_market_gate(self):
        bars = _p1_low_base_bars()
        chip = {
            "concentration": 0.45,
            "peak_pctile": 0.40,
            "price_to_peak": 0.02,
            "winner": 0.50,
        }
        context = {"chip": chip}

        self.assertFalse(radar._p1_low_base_state(bars[:-3]))
        self.assertTrue(radar._p1_low_base_state(bars[:-2]))
        self.assertTrue(radar._p1_low_base_state(bars[:-1]))
        self.assertTrue(radar._p1_low_base_state(bars))
        self.assertTrue(radar._pat_low_consolidation(bars, context))
        self.assertFalse(radar._pat_low_consolidation(
            bars, {"chip": {**chip, "winner": 0.90}}
        ))

        p1_only = [("P1", "低位底盘吸筹确认", "吸筹", "buy", radar._pat_low_consolidation)]
        with patch.object(radar, "PATTERNS", p1_only):
            enabled = radar.match_patterns(
                "600000", bars, ctx=context, pool="hotmoney", p1_market_gate=True
            )
            closed = radar.match_patterns(
                "600000", bars, ctx=context, pool="hotmoney", p1_market_gate=False
            )
            leader = radar.match_patterns(
                "600000", bars, ctx=context, pool="leader", p1_market_gate=True
            )
            unspecified = radar.match_patterns(
                "600000", bars, ctx=context, p1_market_gate=True
            )
        self.assertEqual([row["code"] for row in enabled], ["P1"])
        self.assertEqual(closed, [])
        self.assertEqual([row["code"] for row in leader], ["P1"])
        self.assertEqual([row["code"] for row in unspecified], ["P1"])

    def test_accumulation_bars_fire_buy_pattern(self):
        # 先 16→9 阴跌 60 日，再低位横盘；应至少命中一个吸筹观察形态。
        bars = []
        prev = 16.0
        for t in range(90):
            c = 16 - (16 - 9) * t / 59 if t < 60 else 9.0 + (0.04 if t % 2 else -0.04)
            o = prev
            bars.append(_pat_bar(t, o, max(o, c) * 1.004, min(o, c) * 0.992,
                                 c, 1000.0 if t < 85 else 3000.0, prev))
            prev = c
        fired = radar.match_patterns("600000", bars)
        self.assertTrue(any(p["phase"] == "吸筹" and p["signal"] == "buy" for p in fired),
                        msg=f"fired={[p['code'] for p in fired]}")

    def test_distribution_bars_fire_sell_pattern(self):
        # 单边走高 8→20，最后一根高位巨量大阴（灌压出货）→ 应命中 sell 形态
        bars = []
        prev = 8.0
        for t in range(89):
            c = 8 + 12 * t / 88
            o = prev
            vol = 3000.0 if t >= 84 else 1000.0
            bars.append(_pat_bar(t, o, max(o, c) * 1.005, min(o, c) * 0.995, c, vol, prev))
            prev = c
        o = prev
        c = o * 0.90
        bars.append(_pat_bar(89, o, o * 1.01, c * 0.99, c, 5000.0, prev))
        fired = radar.match_patterns("600000", bars)
        self.assertTrue(any(p["signal"] == "sell" for p in fired),
                        msg=f"fired={[p['code'] for p in fired]}")

    def test_patterns_mode_runs(self):
        codes = ["600000", "600001", "600002", "600003", "600004", "600005"]

        def seed(conn):
            stock_storage.save_sw3_membership(conn, {"segments": [{
                "segment_code": "850111.SI", "segment_name": "测试", "parent_segment": "测试",
                "members": [{"code": c, "name": f"票{c}"} for c in codes],
            }]})
            stock_storage.mark_sw3_leaders(conn, codes)
            for k, c in enumerate(codes):
                _seed_history(conn, c, _gen_bars(k))

        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "t.sqlite3"
            out_file = Path(tmpdir) / "patterns.json"
            conn = stock_storage.connect(db_file)
            seed(conn)
            conn.close()
            with patch.object(radar, "DB_FILE", db_file), \
                 patch.object(radar, "PATTERNS_RESULT_FILE", out_file), \
                 patch.object(radar, "VERIFY_STEP", 5):
                payload = radar.main(["patterns"])
            self.assertTrue(out_file.exists())
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(payload["patterns"]), len(radar.PATTERNS))
        for p in payload["patterns"]:
            for h in radar.VERIFY_HORIZONS:
                self.assertIn(str(h), p["horizons"])
                self.assertIn("hac_lag", p["horizons"][str(h)])
                self.assertIn("fdr_10", p["horizons"][str(h)])
        self.assertEqual(payload["bh_fdr_q"], radar.PATTERN_FDR_Q)

    def test_leader_pattern_history_passes_pit_p1_gate_to_matcher(self):
        conn = stock_storage.connect(":memory:")
        _seed_history(conn, "600000", _gen_bars(1, n=240))
        seen_gates = []

        def gates_for_dates(_conn, dates):
            return {day: True for day in dates}

        def fake_match(_code, _bars, **kwargs):
            seen_gates.append(kwargs.get("p1_market_gate"))
            return []

        with patch.object(radar, "VERIFY_STEP", 5), \
             patch.object(radar, "_p1_market_gate_by_date", side_effect=gates_for_dates), \
             patch.object(radar, "match_patterns", side_effect=fake_match):
            collected = radar._collect_pattern_samples(
                conn, [{"code": "600000"}], pool="leader"
            )
        conn.close()

        self.assertTrue(collected["samples"])
        self.assertTrue(seen_gates)
        self.assertTrue(all(seen_gates))

    def test_leader_accumulation_history_scores_p1_and_p25_with_pit_gate(self):
        conn = stock_storage.connect(":memory:")
        _seed_history(conn, "600000", _gen_bars(1, n=240))
        seen = []

        def gates_for_dates(_conn, dates):
            return {day: True for day in dates}

        def fake_match(_code, _bars, **kwargs):
            seen.append((kwargs.get("pool"), kwargs.get("p1_market_gate")))
            return [
                {"code": "P1", "name": "P1", "phase": "吸筹", "signal": "buy"},
                {"code": "P25", "name": "P25", "phase": "吸筹", "signal": "buy"},
            ]

        with patch.object(radar, "VERIFY_STEP", 5), \
             patch.object(radar, "_p1_market_gate_by_date", side_effect=gates_for_dates), \
             patch.object(radar, "match_patterns", side_effect=fake_match):
            collected = radar._collect_accumulation_samples(
                conn, [{"code": "600000"}], pool="leader"
            )
        conn.close()

        self.assertTrue(collected["samples"])
        self.assertTrue(seen)
        self.assertTrue(all(pool == "leader" and gate for pool, gate in seen))
        self.assertTrue(all(row["features"]["p1"] == 100.0 for row in collected["samples"]))
        self.assertTrue(all(row["features"]["p25"] == 100.0 for row in collected["samples"]))


class HotMoneyLatentTest(unittest.TestCase):
    def test_yaogu_genes_ranks_active_higher_and_in_unit_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "t.sqlite3"
            conn = stock_storage.connect(db_file)
            conn.execute("CREATE TABLE lhb_all (code TEXT, date TEXT, net_buy REAL, net_pct REAL)")
            base = date(2024, 1, 1)

            def bars(turns, chgs):
                return [((base + timedelta(days=t)).isoformat(), 10.0, 10.0, 10.0, 1000.0, chg, tn)
                        for t, (tn, chg) in enumerate(zip(turns, chgs))]

            _seed_history(conn, "600000", bars([15.0] * 30, [10.0] * 5 + [0.0] * 25))  # 高换手+5涨停
            _seed_history(conn, "600001", bars([1.0] * 30, [0.0] * 30))               # 安静
            conn.executemany("INSERT INTO lhb_all (code, date) VALUES (?, ?)",
                             [("600000", f"2024-01-0{i+1}") for i in range(5)] + [("600001", "2024-01-10")])
            genes = radar._yaogu_genes(conn, ["600000", "600001"], as_of="2024-03-01")
            conn.close()
        self.assertGreater(genes["600000"]["gene"], genes["600001"]["gene"])
        self.assertEqual(genes["600000"]["limitups_3y"], 5.0)
        for g in genes.values():
            self.assertGreaterEqual(g["gene"], 0.0)
            self.assertLessEqual(g["gene"], 1.0)

    def _run_latent(self, seed):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "t.sqlite3"
            out_file = Path(tmpdir) / "latent.json"
            conn = stock_storage.connect(db_file)
            seed(conn)
            conn.close()
            with patch.object(radar, "DB_FILE", db_file), \
                 patch.object(radar, "LATENT_RESULT_FILE", out_file):
                payload = radar.main(["latent", "--pool", "hotmoney"])
            self.assertTrue(out_file.exists())
            return payload

    def test_latent_filters_high_position_keeps_left_side(self):
        base = date(2024, 1, 1)

        def mk(prices, vols):
            bars, prev = [], prices[0]
            for t, (px, v) in enumerate(zip(prices, vols)):
                chg = (px / prev - 1.0) * 100.0 if t else 0.0
                bars.append(((base + timedelta(days=t)).isoformat(),
                             px * 1.01, px * 0.99, px, v, chg, v / 1000.0))
                prev = px
            return bars

        def seed(conn):
            stock_storage.save_sw3_membership(conn, {"segments": [{
                "segment_code": "850111.SI", "segment_name": "测试", "parent_segment": "测试",
                "members": [{"code": "600000", "name": "潜伏"}, {"code": "600001", "name": "高位"}]}]})
            stock_storage._ensure_table_columns(conn, "sw3_member",
                                                {"is_hot_money": "INTEGER NOT NULL DEFAULT 0"})
            conn.executemany("UPDATE sw3_member SET is_hot_money=1 WHERE code=?", [("600000",), ("600001",)])
            conn.execute("CREATE TABLE lhb_all (code TEXT, date TEXT, net_buy REAL, net_pct REAL)")
            conn.execute("CREATE TABLE shareholder_count (code TEXT, disclose_date TEXT, change_pct REAL)")
            conn.execute("CREATE TABLE repurchase (code TEXT, disclose_date TEXT)")
            n = 90
            dec = [16 - (16 - 8) * t / (n - 1) for t in range(n)]          # 阴跌, 末日=新低(低位)
            decv = [2000 - (2000 - 700) * t / (n - 1) for t in range(n)]   # 量递减(末日安静)
            _seed_history(conn, "600000", mk(dec, decv))
            inc = [8 + (16 - 8) * t / (n - 1) for t in range(n)]           # 单边走高, 末日=新高(高位)
            _seed_history(conn, "600001", mk(inc, [1500.0] * n))
            conn.commit()

        payload = self._run_latent(seed)
        self.assertEqual(payload["mode"], "latent")
        codes = {s["code"] for s in payload["stocks"]}
        self.assertIn("600000", codes)        # 左侧低位+安静 → 入选
        self.assertNotIn("600001", codes)     # 高位 → 被硬过滤
        for s in payload["stocks"]:
            self.assertLessEqual(s["signals"]["close_pctile"], radar.LATENT_MAX_POS)
            self.assertLessEqual(s["distribution_score"], radar.LATENT_MAX_DIST)
            self.assertFalse(s.get("lhb_recent"))
            self.assertIn("latent_score", s)
            self.assertIn("yaogu_gene", s)


class NewsCatalystTest(unittest.TestCase):
    def test_tag_themes(self):
        import stock_crawl_news as news
        self.assertIn("存储/HBM", news.tag_themes("", "公司HBM先进封装订单放量"))
        self.assertIn("机器人", news.tag_themes("人形机器人", "灵巧手量产"))
        self.assertEqual(news.tag_themes("", "某公司召开股东大会"), "")

    def test_load_recent_news_missing_table_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = stock_storage.connect(Path(tmp) / "t.sqlite3")
            self.assertEqual(stock_storage.load_recent_news(conn, "600000"), [])  # 无 stock_news 表 → []
            conn.close()

    def test_news_catalyst_pit_and_themes(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = stock_storage.connect(Path(tmp) / "t.sqlite3")
            conn.execute("CREATE TABLE stock_news (code TEXT, pub_time TEXT, title TEXT, source TEXT, "
                         "url TEXT, keyword TEXT, themes TEXT, fetched_at TEXT, PRIMARY KEY(code,pub_time,title))")
            conn.executemany(
                "INSERT INTO stock_news (code, pub_time, title, themes) VALUES (?,?,?,?)",
                [("600000", "2026-06-20 09:00:00", "HBM放量", "存储/HBM"),
                 ("600000", "2026-06-22 09:00:00", "算力订单", "AI算力"),
                 ("600000", "2026-07-10 09:00:00", "未来新闻", "机器人")])  # as_of 之后 → PIT 排除
            conn.commit()
            cat = radar._news_catalyst(conn, "600000", as_of="2026-06-25", days=30)
            conn.close()
        self.assertEqual(cat["news_count"], 2)
        self.assertEqual(cat["latest_date"], "2026-06-22")
        self.assertEqual(cat["latest_age_days"], 3)
        self.assertIn("存储/HBM", cat["themes"])
        self.assertNotIn("机器人", cat["themes"])   # 未来题材不泄漏


if __name__ == "__main__":
    unittest.main()
