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


class HotMoneyRadarTest(unittest.TestCase):
    def test_default_mode_is_ambush(self):
        self.assertEqual(radar.build_parser().parse_args([]).mode, "ambush")
        self.assertEqual(radar.build_parser().parse_args(["distribution"]).mode, "distribution")
        self.assertEqual(radar.build_parser().parse_args(["accumulation"]).mode, "accumulation")

    def test_distribution_model_formula(self):
        weights = {
            "technical": 0.20,
            "p11": 0.20,
            "p19": 0.10,
            "p20": 0.10,
            "lhb_recent": 0.20,
            "divergence": 0.20,
        }
        features = radar._distribution_model_features(50.0, ["P11", "P20"], lhb_recent=True, divergence_score=80.0)
        self.assertEqual(radar._distribution_model_score(features, weights), 76.0)

        p19_only = radar._distribution_model_features(0.0, ["P19"], lhb_recent=False, divergence_score=0.0)
        self.assertEqual(radar._distribution_model_score(p19_only, weights), 10.0)

    def test_accumulation_model_uses_raw_feature_weights(self):
        weights = {
            "chip": 0.30,
            "position": 0.20,
            "cmf_eff": 0.10,
            "p3": 0.10,
            "holder_change": 0.10,
            "repurchase": 0.10,
        }
        rows = [
            {
                "ambush_score": 12.0,
                "distribution_score": 0.0,
                "holder_change": -15.0,
                "repurchase_recent": True,
                "patterns": ["P3"],
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
        with patch.object(radar, "ACCUM_MODEL_WEIGHTS", weights):
            radar._apply_accumulation_model(rows)

        self.assertEqual(rows[0]["ambush_score"], 72.0)
        self.assertEqual(rows[0]["accumulation_percentile"], 100.0)
        self.assertEqual(rows[0]["distribution_percentile"], 0.0)
        self.assertEqual(rows[0]["opportunity_score"], 100.0)
        self.assertEqual(rows[0]["technical_ambush_score"], 12.0)
        self.assertEqual(rows[0]["signals"]["accumulation_model_features"]["p3"], 100.0)
        self.assertEqual(rows[0]["signals"]["accumulation_model_features"]["holder_change"], 100.0)
        self.assertEqual(rows[0]["signals"]["accumulation_model_features"]["repurchase"], 100.0)
        self.assertEqual(rows[1]["accumulation_percentile"], 0.0)
        self.assertEqual(rows[1]["distribution_percentile"], 100.0)
        self.assertEqual(rows[1]["opportunity_score"], 0.0)
        self.assertEqual(rows[1]["signals"]["accumulation_model_features"]["p3"], 0.0)
        self.assertEqual(rows[1]["signals"]["accumulation_model_features"]["holder_change"], 50.0)
        self.assertEqual(rows[0]["signals"]["accumulation_percentile"], 100.0)
        self.assertEqual(rows[0]["signals"]["distribution_percentile"], 0.0)
        self.assertEqual(rows[0]["signals"]["opportunity_formula"], radar.OPPORTUNITY_FORMULA)

    def test_cross_section_percentiles_share_tie_rank(self):
        rows = [{"score": 10.0}, {"score": 20.0}, {"score": 20.0}, {"score": 40.0}]
        self.assertEqual(
            radar._cross_section_percentiles(rows, "score"),
            {0: 0.0, 1: 50.0, 2: 50.0, 3: 100.0},
        )

    def test_empty_pattern_phase_uses_watch_label(self):
        self.assertEqual(radar._pattern_phase([], score=12.3), "观望⚪")
        self.assertIn("观望⚪", radar.PHASE_ORDER)
        self.assertNotIn("空仓观望⚪", radar.PHASE_ORDER)
        self.assertEqual(radar._phase_confidence("观望⚪", [], 12.3, 0), 85.0)
        self.assertLess(radar._phase_confidence("观望⚪", [], 64.0, 10.0), 35.0)

    def test_phase_confidence_uses_score_confirmation(self):
        buy = [{"code": "P3", "name": "缩量阴线打压吸筹", "phase": "吸筹", "signal": "buy"}]
        sell = [{"code": "P20", "name": "均线放量破位", "phase": "出货", "signal": "sell"}]
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
        self.assertIsNotNone(acc["signals"]["chip_peak_pctile"])
        self.assertIsNotNone(acc["signals"]["chip_price_to_peak"])
        self.assertEqual(acc["signals"]["accumulation_model_features"]["holder_change"], 100.0)
        self.assertEqual(acc["signals"]["accumulation_model_features"]["repurchase"], 100.0)
        self.assertGreaterEqual(by_code["600001"]["signals"]["close_pctile"], 0.9)  # 已拉升在高位
        # 封板：识别到一字封死板并打折、状态标记已启动
        self.assertEqual(by_code["600003"]["signals"]["sealed_recent"], 1)
        self.assertGreater(by_code["600003"]["sub_scores"]["sealed_penalty"], 0)
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
        self.assertEqual(by_code["600000"]["distribution_score"], 10.0)
        self.assertEqual(by_code["600001"]["distribution_score"], 30.0)
        self.assertEqual(by_code["600001"]["signals"]["distribution_model_features"]["technical"], 0.5)
        self.assertEqual(by_code["600000"]["signals"]["distribution_model_features"]["lhb_recent"], 0.0)
        self.assertEqual(by_code["600001"]["signals"]["distribution_model_features"]["lhb_recent"], 1.0)
        self.assertEqual(by_code["600001"]["signals"]["distribution_model_weights"]["lhb_recent"], 0.2)
        self.assertEqual(by_code["600001"]["signals"]["distribution_model_weights"]["divergence"], 0.2)
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
    def test_accumulation_bars_fire_buy_pattern(self):
        # 先 16→9 阴跌 60 日，再低位横盘 9.0 共 30 日 → 应命中 P1 低位横盘磨人
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


if __name__ == "__main__":
    unittest.main()
