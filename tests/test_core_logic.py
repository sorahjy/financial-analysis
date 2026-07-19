import unittest
import tempfile
import sys
import os
import io
import runpy
from pathlib import Path
from unittest.mock import MagicMock, patch
import sqlite3

from fund.fund_generate_output import esc, parse_percent
from fund.fund_technical_analysis import analyze_fund, calc_adx, calc_percentile
from fund.fund_storage import (
    connect as connect_fund_db,
    load_nav_entry,
    load_profile_snapshots,
    load_realtime_estimates,
    save_nav_entry,
    save_profile_snapshots,
    save_realtime_estimates,
)
from stock_crawl_common import (
    _normalize_volume_to_hands,
    daily_stats_from_history_records,
    history_payload_from_records,
    merge_records_by_date,
)
import stock_crawl_price_valuation as stock_price_valuation
from stock_crawl_price_valuation import (
    _decide_valuation_period,
    load_stock_file,
    qfq_overlap_scale_change,
    qfq_return_discontinuity,
    records_need_ohlcv_backfill,
    save_stock_file,
)


class ParsePercentTest(unittest.TestCase):
    def test_parse_percent_accepts_percent_suffix(self):
        self.assertEqual(parse_percent("12.34%"), 12.34)

    def test_parse_percent_accepts_plain_number_string(self):
        self.assertEqual(parse_percent("-1.5"), -1.5)

    def test_parse_percent_rejects_missing_value(self):
        with self.assertRaises(ValueError):
            parse_percent("--")

    def test_html_escape_helper_escapes_external_text(self):
        self.assertEqual(esc('<script>"x"</script>'), '&lt;script&gt;&quot;x&quot;&lt;/script&gt;')


class TechnicalAnalysisTest(unittest.TestCase):
    def test_calc_percentile_flat_series_defaults_to_midpoint(self):
        self.assertEqual(calc_percentile(1.0, [1.0, 1.0, 1.0]), 50.0)

    def test_analyze_fund_handles_flat_nav_percentile(self):
        records = [
            {"date": f"2024-01-{day:02d}", "nav_acc": "1.0"}
            for day in range(1, 31)
        ] + [
            {"date": f"2024-02-{day:02d}", "nav_acc": "1.0"}
            for day in range(1, 29)
        ] + [
            {"date": f"2024-03-{day:02d}", "nav_acc": "1.0"}
            for day in range(1, 32)
        ] + [
            {"date": f"2024-04-{day:02d}", "nav_acc": "1.0"}
            for day in range(1, 32)
        ]

        result = analyze_fund(records)

        self.assertIsNotNone(result)
        self.assertEqual(result["nav_percentile"], 50.0)

    def test_calc_adx_recovers_after_flat_zero_atr_period(self):
        adx = calc_adx([1.0] * 40 + [1.1], period=14)

        self.assertEqual(len(adx), 41)
        self.assertEqual(adx[27], 0.0)
        self.assertAlmostEqual(adx[-1], 100 / 14, places=8)


class StockBenchmarkEtfTest(unittest.TestCase):
    def test_fetch_benchmark_etfs_includes_optimizer_components(self):
        calls = []

        def fake_fetch(code, output_file, *, label, start_date, years=None):
            calls.append((code, Path(output_file).name, label, start_date, years))
            return [{"date": "2026-01-02", "nav_acc": "1.0"}]

        with patch.object(stock_price_valuation, "fetch_index_etf_nav", side_effect=fake_fetch):
            result = stock_price_valuation.fetch_benchmark_etfs()

        self.assertEqual(set(result.keys()), {"510310", "510580"})
        self.assertIn(("510310", "csi300_etf_nav.json", "沪深300", "2012-01-01", None), calls)
        self.assertIn(("510580", "csi500_etf_nav.json", "中证500", "2012-01-01", None), calls)


class DataRefreshPipelineTest(unittest.TestCase):
    def test_full_refresh_runs_segment_leader_history_before_fundamentals(self):
        import stock_data_refresh as refresh

        run_steps = []
        step_commands = {}

        def fake_run_step(name, cmd, *, timeout=None, env=None, skip=False):
            run_steps.append(name)
            step_commands[name] = cmd
            return refresh.StepResult(name, " ".join(cmd), True, 0, 0.0, skipped=skip)

        def fake_local_step(name, command, func):
            run_steps.append(name)
            result = refresh.StepResult(name, command, True, 0, 0.0)
            result.meta = {}
            return result

        with patch.object(refresh, "run_step", side_effect=fake_run_step), \
             patch.object(refresh, "local_step_result", side_effect=fake_local_step), \
             patch.object(refresh, "collect_data_health", return_value={}), \
             patch.object(refresh, "write_json_file", return_value=None):
            refresh.refresh_before_server(mode="full", timeout=1)

        self.assertLess(
            run_steps.index("segment_leader_history"),
            run_steps.index("segment_leader_fundamentals"),
        )
        self.assertEqual(step_commands["segment_leader_fundamentals"][-2:], ["--segment-refresh-slice", "0"])
        self.assertLess(run_steps.index("long_capital_events"), run_steps.index("hot_money_small_cap_universe"))
        self.assertLess(run_steps.index("hot_money_small_cap_universe"), run_steps.index("short_signal_enrichment"))
        self.assertLess(run_steps.index("short_signal_enrichment"), run_steps.index("strategy_results"))
        self.assertIn("--enrich-long-factors", step_commands["hot_money_small_cap_universe"])
        self.assertEqual(
            step_commands["short_signal_enrichment"][-4:],
            ["--days", "14", "--top-yyb", "30"],
        )


class HotMoneyHistoryRefreshTest(unittest.TestCase):
    def test_universe_reuses_shared_history_refresher_before_filtering(self):
        import stock_crawl_hot_money_universe as universe

        result = {"requested": 2, "processed": 2, "write_enqueued": 1,
                  "write_completed": 1, "failed": []}
        with patch.object(universe, "refresh_stock_histories", return_value=result) as refresh_mock:
            actual = universe.refresh_seed_histories(
                ["000001", "000002"],
                {"000001": "平安银行", "000002": "万科A"},
                years=4.0,
                workers=6,
            )

        self.assertEqual(actual, result)
        refresh_mock.assert_called_once_with(
            {"000001": "平安银行", "000002": "万科A"},
            max_years=4.0,
            workers=6,
            refresh_valuation=False,
            label="游资初筛",
        )

    def test_final_hotmoney_pool_refreshes_long_factor_data(self):
        import stock_crawl_hot_money_universe as universe

        result = {"requested": 2, "processed": 2, "failed": []}
        plan = {"000001": True, "000002": False}
        pledge = {"000001": {"pledge_ratio": 3.0}}
        with patch.object(
            universe, "plan_fundamentals_refresh", return_value=(plan, pledge)
        ) as plan_mock, patch.object(
            universe, "refresh_stock_histories", return_value=result
        ) as refresh_mock:
            actual = universe.refresh_selected_factor_data(
                ["000001", "000002"],
                {"000001": "平安银行", "000002": "万科A"},
                years=4.0,
                workers=6,
            )

        self.assertEqual(actual, result)
        plan_mock.assert_called_once_with({"000001", "000002"})
        refresh_mock.assert_called_once_with(
            {"000001": "平安银行", "000002": "万科A"},
            max_years=4.0,
            workers=6,
            refresh_valuation=True,
            fundamentals_plan=plan,
            pledge_map=pledge,
            label="游资小盘长线因子",
        )

    def test_factor_refresh_failure_preserves_previous_hotmoney_pool(self):
        import stock_crawl_hot_money_universe as universe

        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = ["2026-07-17"]
        with patch.object(sys, "argv", [
                 "stock_crawl_hot_money_universe.py", "--enrich-long-factors"
             ]), \
             patch.object(universe.stock_storage, "connect", return_value=conn), \
             patch.object(universe.stock_storage, "_ensure_table_columns"), \
             patch.object(universe, "select_seed", return_value={
                 "in_member": ["000002"],
                 "not_member": [],
                 "names": {"000002": "万科A"},
             }), \
             patch.object(universe, "refresh_seed_histories", return_value={"failed": []}), \
             patch.object(universe, "latest_history_dates", return_value={"000002": "2026-07-17"}), \
             patch.object(universe, "_bars_count", return_value=300), \
             patch.object(universe, "history_is_fresh", return_value=True), \
             patch.object(universe, "float_cap_yi", return_value=50.0), \
             patch.object(universe, "refresh_selected_factor_data", return_value={
                 "failed": ["000002"],
             }):
            with self.assertRaisesRegex(RuntimeError, "保留上一份有效股票池"):
                universe.main()

        executed = [call.args[0] for call in conn.execute.call_args_list]
        self.assertFalse(any("UPDATE sw3_member SET is_hot_money = 0" in sql for sql in executed))

    def test_history_freshness_uses_actual_latest_bar_date(self):
        import stock_crawl_hot_money_universe as universe

        self.assertTrue(universe.history_is_fresh("2026-07-10", "2026-07-10"))
        self.assertTrue(universe.history_is_fresh("2026-07-03", "2026-07-10"))
        self.assertFalse(universe.history_is_fresh("2026-06-22", "2026-07-10"))
        self.assertFalse(universe.history_is_fresh(None, "2026-07-10"))

    def test_short_signal_layer_rejects_stale_local_history_without_refetching(self):
        import stock_crawl_hot_money as short_signal

        records = [{"date": f"2026-06-{day:02d}"} for day in range(1, 23)]
        records = ([{"date": f"2026-05-{day:02d}"} for day in range(1, 32)] + records)[-60:]
        with patch.object(short_signal, "load_local_kline_records", return_value=records):
            self.assertIsNone(
                short_signal.fetch_stock_data("002150", "正泰电源", reference_date="2026-07-10")
            )

        fresh = [{"date": f"2026-07-{day:02d}"} for day in range(1, 11)]
        fresh = ([{"date": f"2026-06-{day:02d}"} for day in range(1, 31)] * 2 + fresh)[-60:]
        with patch.object(short_signal, "load_local_kline_records", return_value=fresh):
            data = short_signal.fetch_stock_data(
                "002150", "正泰电源", reference_date="2026-07-10"
            )
        self.assertEqual(data["as_of_date"], "2026-07-10")


class StockShortStrategyTest(unittest.TestCase):
    def test_no_proxy_flag_sets_env_before_stripping_proxy(self):
        import plate_crawl_history as short_strategy

        with patch.dict(os.environ, {
            "HTTP_PROXY": "http://127.0.0.1:7897",
            "https_proxy": "http://127.0.0.1:7897",
        }, clear=False), \
            patch.object(short_strategy, "fetch_sw2_daily_analysis", return_value={
                "fetched": 0,
                "inserted": 0,
                "updated": 0,
                "total_records": 0,
                "latest_trade_date": None,
                "errors": [],
            }) as fetch_mock, \
            patch.object(sys, "argv", [
                "plate_crawl_history.py",
                "--start-date", "20260601",
                "--end-date", "20260601",
                "--no-proxy",
            ]):
            short_strategy.main()

            self.assertEqual(os.environ.get("STOCK_CRAWL_NO_PROXY"), "1")
            self.assertEqual(os.environ.get("NO_PROXY"), "*")
            self.assertNotIn("HTTP_PROXY", os.environ)
            self.assertNotIn("https_proxy", os.environ)
            fetch_mock.assert_called_once()

    def test_resolve_incremental_start_uses_latest_trade_date(self):
        import plate_crawl_history as short_strategy

        start = short_strategy.parse_yyyymmdd("20160101")
        end = short_strategy.parse_yyyymmdd("20260620")

        self.assertEqual(
            short_strategy.resolve_incremental_start("2026-06-18", start, end),
            short_strategy.parse_yyyymmdd("20260619"),
        )
        self.assertIsNone(
            short_strategy.resolve_incremental_start("2026-06-20", start, end)
        )


class PlateStorageTest(unittest.TestCase):
    def test_schema_has_daily_meta_comments_without_crawl_runs(self):
        import plate_storage

        conn = plate_storage.connect(":memory:")
        try:
            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            self.assertIn("plate_meta", tables)
            self.assertIn("plate_daily", tables)
            self.assertIn("plate_column_comments", tables)
            self.assertNotIn("plate_crawl_runs", tables)

            daily_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(plate_daily)")
            }
            comment_cols = {
                row["column_name"]
                for row in conn.execute(
                    "SELECT column_name FROM plate_column_comments WHERE table_name = 'plate_daily'"
                )
            }
            self.assertTrue(daily_cols <= comment_cols)
        finally:
            conn.close()

    def test_save_sw2_daily_rows_upserts_and_refreshes_meta(self):
        import plate_storage

        conn = plate_storage.connect(":memory:")
        try:
            row = {
                "swindexcode": "801125",
                "swindexname": "白酒Ⅱ",
                "bargaindate": "2026-06-18",
                "closeindex": "34766.94",
                "bargainamount": "1.97",
                "markup": "-2.08",
                "turnoverrate": "2.37",
                "pe": "18.66",
                "pb": "3.46",
                "meanprice": "98.1",
                "bargainsumrate": "0.54",
                "negotiablessharesum1": "9301.09",
                "negotiablessharesum2": "489.53",
                "dp": "4.91",
            }

            stats = plate_storage.save_sw2_daily_rows(conn, [row])
            self.assertEqual(stats, {"fetched": 1, "inserted": 1, "updated": 0})
            stats = plate_storage.save_sw2_daily_rows(conn, [row])
            self.assertEqual(stats, {"fetched": 1, "inserted": 0, "updated": 1})

            daily = conn.execute("SELECT * FROM plate_daily").fetchone()
            self.assertEqual(daily["plate_type"], "sw2")
            self.assertEqual(daily["plate_code"], "801125")
            self.assertEqual(daily["trade_date"], "2026-06-18")
            self.assertEqual(daily["float_market_cap"], 9301.09)
            self.assertIn("negotiablessharesum1", daily["raw_json"])

            meta = conn.execute("SELECT * FROM plate_meta").fetchone()
            self.assertEqual(meta["first_date"], "2026-06-18")
            self.assertEqual(meta["last_date"], "2026-06-18")
            self.assertEqual(meta["record_count"], 1)
        finally:
            conn.close()


class SegmentLeaderStorageConnectionTest(unittest.TestCase):
    def test_sw3_membership_helpers_close_connections(self):
        import stock_crawl_segment_leaders as r

        class FakeConn:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        read_conn = FakeConn()
        with patch.object(r.stock_storage, "connect", return_value=read_conn) as connect_mock, \
             patch.object(r.stock_storage, "load_sw3_membership", return_value={"segments": []}) as load_mock:
            result = r.load_sw3_membership(max_age_days=7)

        self.assertEqual(result, {"segments": []})
        connect_mock.assert_called_once_with(r.STOCK_DB_FILE)
        load_mock.assert_called_once_with(read_conn, max_age_days=7)
        self.assertTrue(read_conn.closed)

        write_conn = FakeConn()
        payload = {"schema": r.SW3_MEMBERSHIP_SCHEMA, "segments": []}
        with patch.object(r.stock_storage, "connect", return_value=write_conn), \
             patch.object(r.stock_storage, "save_sw3_membership") as save_mock:
            r._save_sw3_membership_to_db(payload)

        save_mock.assert_called_once_with(write_conn, payload)
        self.assertTrue(write_conn.closed)


class FundSQLiteStorageTest(unittest.TestCase):
    def test_fund_no_proxy_session_is_used_for_every_requests_call(self):
        import fund.fund_fetch_data as fund_fetch_data

        nav_response = MagicMock()
        nav_response.text = "var apidata={ content:'',records:0,pages:0,curpage:1};"
        estimate_response = MagicMock()
        estimate_response.text = (
            'jsonpgz({"gsz":"1.1","gszzl":"0.1","gztime":"2026-07-19 15:00",'
            '"dwjz":"1.0"})'
        )
        session = MagicMock()
        session.trust_env = True
        session.get.side_effect = [nav_response, estimate_response]

        with patch.dict(
            os.environ,
            {"FuNd_CrAwL_No_PrOxY": "true", "HTTPS_PROXY": "http://proxy.invalid"},
            clear=False,
        ), patch.object(
            fund_fetch_data.requests,
            "Session",
            return_value=session,
        ), patch.object(
            fund_fetch_data,
            "_HTTP_SESSION",
            None,
        ), patch.object(
            fund_fetch_data,
            "_HTTP_SESSION_NO_PROXY",
            None,
        ):
            fund_fetch_data._get_nav_with_retry({"code": "000001"})
            estimate = fund_fetch_data.fetch_estimate("000001")

        self.assertFalse(session.trust_env)
        self.assertEqual(session.get.call_count, 2)
        self.assertEqual(session.get.call_args_list[0].args[0], fund_fetch_data.NAV_API_URL)
        self.assertEqual(
            session.get.call_args_list[1].args[0],
            fund_fetch_data.ESTIMATE_API_URL.format(code="000001"),
        )
        self.assertEqual(estimate["gsz"], "1.1")

    def test_fund_session_without_marker_keeps_requests_environment_support(self):
        import fund.fund_fetch_data as fund_fetch_data

        environment = {
            key: value
            for key, value in os.environ.items()
            if key.lower() not in {
                "fund_crawl_no_proxy",
                "stock_crawl_no_proxy",
            }
        }
        with patch.dict(os.environ, environment, clear=True):
            session = fund_fetch_data._build_http_session()
        try:
            self.assertTrue(session.trust_env)
        finally:
            session.close()

    def test_fund_session_honors_stock_refresh_no_proxy_marker(self):
        import fund.fund_fetch_data as fund_fetch_data

        environment = {
            key: value
            for key, value in os.environ.items()
            if key.lower() not in {
                "fund_crawl_no_proxy",
                "stock_crawl_no_proxy",
            }
        }
        with patch.dict(
            os.environ,
            {
                **environment,
                "stock_crawl_no_proxy": "yes",
                "HTTPS_PROXY": "http://proxy.invalid",
            },
            clear=True,
        ):
            session = fund_fetch_data._build_http_session()
        try:
            self.assertFalse(session.trust_env)
            settings = session.merge_environment_settings(
                fund_fetch_data.NAV_API_URL,
                {},
                None,
                None,
                None,
            )
            self.assertEqual(dict(settings["proxies"]), {})
        finally:
            session.close()

    def test_scrapy_proxy_middleware_is_disabled_only_for_no_proxy_marker(self):
        settings_file = (
            Path(__file__).resolve().parents[1]
            / "fund"
            / "ttjj_spider"
            / "settings.py"
        )
        base_environment = {
            key: value
            for key, value in os.environ.items()
            if key.lower() != "fund_crawl_no_proxy"
        }

        with patch.dict(os.environ, base_environment, clear=True):
            regular_settings = runpy.run_path(str(settings_file))
        with patch.dict(
            os.environ,
            {**base_environment, "fund_crawl_no_proxy": "ON"},
            clear=True,
        ):
            direct_settings = runpy.run_path(str(settings_file))

        self.assertNotIn("HTTPPROXY_ENABLED", regular_settings)
        self.assertIs(direct_settings["HTTPPROXY_ENABLED"], False)

    def test_schema_keeps_only_core_fund_cache_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect_fund_db(Path(tmp) / "fund.sqlite3")
            try:
                tables = {
                    row["name"]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            finally:
                conn.close()

        self.assertIn("fund_nav_records", tables)
        self.assertIn("fund_nav_meta", tables)
        self.assertIn("fund_realtime_estimates", tables)
        self.assertIn("fund_profile_snapshots", tables)
        self.assertNotIn("fund_reports", tables)
        self.assertNotIn("fund_backtest_results", tables)
        self.assertNotIn("fund_signals", tables)

    def test_nav_entry_round_trips_with_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect_fund_db(Path(tmp) / "fund.sqlite3")
            try:
                save_nav_entry(conn, "000001", {
                    "start_date": "2026-01-01",
                    "end_date": "2026-01-02",
                    "records": [
                        {"date": "2026-01-02", "nav": "1.1", "nav_acc": "1.2", "daily_growth_rate": "1.00"},
                        {"date": "2026-01-01", "nav": "1.0", "nav_acc": "1.1", "daily_growth_rate": "0.00"},
                    ],
                })

                entry = load_nav_entry(conn, "000001")
            finally:
                conn.close()

        self.assertEqual(entry["start_date"], "2026-01-01")
        self.assertEqual(entry["end_date"], "2026-01-02")
        self.assertEqual([row["date"] for row in entry["records"]], ["2026-01-01", "2026-01-02"])
        self.assertEqual(entry["records"][1]["nav_acc"], "1.2")

    def test_realtime_estimates_merge_partial_snapshot_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect_fund_db(Path(tmp) / "fund.sqlite3")
            try:
                save_realtime_estimates(conn, {
                    "000001": {"gsz": "1.1", "gszzl": "1.0", "gztime": "2026-01-01 15:00", "dwjz": "1.0"},
                    "000002": {"gsz": "2.1", "gszzl": "2.0", "gztime": "2026-01-01 15:00", "dwjz": "2.0"},
                })
                save_realtime_estimates(conn, {
                    "000002": {"gsz": "2.2", "gszzl": "2.1", "gztime": "2026-01-02 15:00", "dwjz": "2.0"},
                })

                estimates = load_realtime_estimates(conn)
            finally:
                conn.close()

        self.assertEqual(list(estimates.keys()), ["000001", "000002"])
        self.assertEqual(estimates["000001"]["gsz"], "1.1")
        self.assertEqual(estimates["000002"]["gsz"], "2.2")

    def test_empty_realtime_fields_do_not_erase_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect_fund_db(Path(tmp) / "fund.sqlite3")
            try:
                save_realtime_estimates(conn, {
                    "000001": {
                        "gsz": "1.1", "gszzl": "1.0",
                        "gztime": "2026-01-01 15:00", "dwjz": "1.0",
                    },
                })
                save_realtime_estimates(conn, {
                    "000001": {"gsz": "", "gszzl": "", "gztime": "", "dwjz": ""},
                })
                with self.assertRaisesRegex(ValueError, "数据不完整"):
                    save_realtime_estimates(
                        conn,
                        {"000001": {"gsz": "", "dwjz": ""}},
                        replace=True,
                        expected_codes=["000001"],
                    )
                estimates = load_realtime_estimates(conn)
            finally:
                conn.close()

        self.assertEqual(estimates["000001"]["gsz"], "1.1")
        self.assertEqual(estimates["000001"]["gztime"], "2026-01-01 15:00")

    def test_profile_snapshot_save_merges_partial_snapshot_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_file = Path(tmp) / "fund.sqlite3"

            conn = connect_fund_db(db_file)
            try:
                self.assertEqual(save_profile_snapshots(conn, [
                    {"fundCode": "000001", "name": "基金A"},
                    {"fundCode": "000002", "name": "基金B"},
                ]), 2)

                self.assertEqual(save_profile_snapshots(conn, [
                    {"fundCode": "000002", "name": "基金B2"},
                ]), 1)
                profiles = load_profile_snapshots(conn)
            finally:
                conn.close()

        self.assertEqual(list(profiles.keys()), ["000001", "000002"])
        self.assertEqual(profiles["000001"]["name"], "基金A")
        self.assertEqual(profiles["000002"]["name"], "基金B2")

    def test_profile_snapshot_preserves_fields_from_failed_subrequests(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect_fund_db(Path(tmp) / "fund.sqlite3")
            try:
                save_profile_snapshots(conn, [{
                    "fundCode": "000001",
                    "name": "基金A",
                    "managerTrigger": "任职以来 20%",
                    "fund_manager_total_asset": "100亿元",
                    "details": {"five_year_growth": "50%", "rank": "1/10"},
                }])
                save_profile_snapshots(
                    conn,
                    [{
                        "fundCode": "000001",
                        "name": "基金A新名称",
                        "managerTrigger": "",
                        "fund_manager_total_asset": None,
                        "details": {"five_year_growth": ""},
                    }],
                    replace=True,
                    expected_codes=["000001"],
                )
                profile = load_profile_snapshots(conn)["000001"]
            finally:
                conn.close()

        self.assertEqual(profile["name"], "基金A新名称")
        self.assertEqual(profile["managerTrigger"], "任职以来 20%")
        self.assertEqual(profile["fund_manager_total_asset"], "100亿元")
        self.assertEqual(
            profile["details"], {"five_year_growth": "50%", "rank": "1/10"}
        )

    def test_full_snapshot_replace_requires_complete_expected_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect_fund_db(Path(tmp) / "fund.sqlite3")
            try:
                save_profile_snapshots(conn, [
                    {"fundCode": "000001", "name": "基金A"},
                    {"fundCode": "000002", "name": "基金B"},
                ])

                with self.assertRaisesRegex(ValueError, "数据不完整"):
                    save_profile_snapshots(
                        conn,
                        [{"fundCode": "000002", "name": "基金B2"}],
                        replace=True,
                        expected_codes=["000001", "000002"],
                    )
                profiles = load_profile_snapshots(conn)
            finally:
                conn.close()

        self.assertEqual(list(profiles.keys()), ["000001", "000002"])
        self.assertEqual(profiles["000002"]["name"], "基金B")

    def test_validated_full_snapshot_can_replace_removed_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect_fund_db(Path(tmp) / "fund.sqlite3")
            try:
                save_realtime_estimates(conn, {
                    "000001": {"gsz": "1.1"},
                    "000002": {"gsz": "2.1"},
                })
                save_realtime_estimates(
                    conn,
                    {"000002": {"gsz": "2.2"}},
                    replace=True,
                    expected_codes=["000002"],
                )
                estimates = load_realtime_estimates(conn)
            finally:
                conn.close()

        self.assertEqual(list(estimates), ["000002"])
        self.assertEqual(estimates["000002"]["gsz"], "2.2")

    def test_partial_realtime_fetch_preserves_failed_fund_snapshot(self):
        import fund.fund_fetch_data as fund_fetch_data

        with tempfile.TemporaryDirectory() as tmp:
            db_file = Path(tmp) / "fund.sqlite3"
            conn = connect_fund_db(db_file)
            try:
                save_realtime_estimates(conn, {
                    "000001": {"gsz": "1.1", "gztime": "old"},
                    "000002": {"gsz": "2.1", "gztime": "old"},
                })
            finally:
                conn.close()

            with patch.object(
                fund_fetch_data,
                "fetch_estimate",
                side_effect=[None, {"gsz": "2.2", "gszzl": "1.0", "gztime": "new", "dwjz": "2.0"}],
            ), patch.object(
                fund_fetch_data,
                "connect_fund_db",
                side_effect=lambda: connect_fund_db(db_file),
            ), patch.object(fund_fetch_data.time, "sleep"):
                fund_fetch_data.fetch_realtime_estimates(["000001", "000002"])

            conn = connect_fund_db(db_file)
            try:
                estimates = load_realtime_estimates(conn)
            finally:
                conn.close()

        self.assertEqual(estimates["000001"]["gsz"], "1.1")
        self.assertEqual(estimates["000001"]["gztime"], "old")
        self.assertEqual(estimates["000002"]["gsz"], "2.2")
        self.assertEqual(estimates["000002"]["gztime"], "new")

    def test_structurally_empty_realtime_fetch_is_not_counted_as_success(self):
        import fund.fund_fetch_data as fund_fetch_data

        with tempfile.TemporaryDirectory() as tmp:
            db_file = Path(tmp) / "fund.sqlite3"
            conn = connect_fund_db(db_file)
            try:
                save_realtime_estimates(conn, {
                    "000001": {"gsz": "1.1", "gztime": "old"},
                    "000002": {"gsz": "2.1", "gztime": "old"},
                })
            finally:
                conn.close()

            with patch.object(
                fund_fetch_data,
                "fetch_estimate",
                side_effect=[
                    {"gsz": "", "gszzl": "", "gztime": "", "dwjz": ""},
                    {"gsz": "2.2", "gszzl": "1.0", "gztime": "new", "dwjz": "2.0"},
                ],
            ), patch.object(
                fund_fetch_data,
                "connect_fund_db",
                side_effect=lambda: connect_fund_db(db_file),
            ), patch.object(fund_fetch_data.time, "sleep"):
                result = fund_fetch_data.fetch_realtime_estimates(["000001", "000002"])

            conn = connect_fund_db(db_file)
            try:
                estimates = load_realtime_estimates(conn)
            finally:
                conn.close()

        self.assertNotIn("000001", result)
        self.assertEqual(estimates["000001"]["gsz"], "1.1")
        self.assertEqual(estimates["000002"]["gsz"], "2.2")

    def test_partial_profile_pipeline_preserves_failed_fund_snapshot(self):
        from fund.ttjj_spider.pipelines import TtjjSpiderPipeline
        import fund.ttjj_spider.pipelines as pipelines

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_file = tmp_path / "fund.sqlite3"
            codes_file = tmp_path / "fund_codes.json"
            codes_file.write_text('["000001", "000002"]', encoding="utf-8")
            conn = connect_fund_db(db_file)
            try:
                save_profile_snapshots(conn, [
                    {"fundCode": "000001", "name": "基金A"},
                    {"fundCode": "000002", "name": "基金B"},
                ])
            finally:
                conn.close()

            spider = MagicMock()
            spider.fund_codes_file = str(codes_file)
            pipeline = TtjjSpiderPipeline()
            pipeline.process_item({"fundCode": "000002", "name": "基金B2"}, spider)
            with patch.object(
                pipelines,
                "connect_fund_db",
                side_effect=lambda: connect_fund_db(db_file),
            ):
                pipeline.close_spider(spider)

            conn = connect_fund_db(db_file)
            try:
                profiles = load_profile_snapshots(conn)
            finally:
                conn.close()

        self.assertEqual(profiles["000001"]["name"], "基金A")
        self.assertEqual(profiles["000002"]["name"], "基金B2")
        spider.logger.warning.assert_called()


class StockValuationDecisionTest(unittest.TestCase):
    def test_skips_when_latest_valuation_fields_are_complete(self):
        records = [{
            "date": "2026-05-29",
            "market_cap": 1000,
            "pe_ttm": 12,
            "pe_static": 15,
            "pb": 1.5,
            "pcf": 8,
        }]

        self.assertIsNone(_decide_valuation_period(records, "2026-05-30"))

    def test_fetches_when_latest_pb_or_pe_is_missing(self):
        records = [
            {
                "date": "2026-05-20",
                "market_cap": 1000,
                "pe_ttm": 12,
                "pe_static": 15,
                "pb": 1.5,
                "pcf": 8,
            },
            {
                "date": "2026-05-29",
                "market_cap": 1100,
                "pe_ttm": None,
                "pe_static": 16,
                "pb": None,
                "pcf": 9,
            },
        ]

        self.assertEqual(_decide_valuation_period(records, "2026-05-30"), "近一年")

    def test_fetches_five_years_when_complete_valuation_is_stale(self):
        records = [
            {
                "date": "2024-01-01",
                "market_cap": 1000,
                "pe_ttm": 12,
                "pe_static": 15,
                "pb": 1.5,
                "pcf": 8,
            },
            {"date": "2026-05-29", "close": 10},
        ]

        self.assertEqual(_decide_valuation_period(records, "2026-05-30"), "近五年")


class StockListingDateStorageTest(unittest.TestCase):
    def test_normalize_listing_date_accepts_common_exchange_formats(self):
        from datetime import date, datetime
        import pandas as pd
        import stock_storage as ss

        self.assertEqual(ss.normalize_listing_date("2018-12-24"), "2018-12-24")
        self.assertEqual(ss.normalize_listing_date("20181224"), "2018-12-24")
        self.assertEqual(ss.normalize_listing_date("2018/12/24"), "2018-12-24")
        self.assertEqual(ss.normalize_listing_date("2018.12.24"), "2018-12-24")
        self.assertEqual(ss.normalize_listing_date(43458), "2018-12-24")
        self.assertEqual(
            ss.normalize_listing_date("2018-12-24 00:00:00"),
            "2018-12-24",
        )
        self.assertEqual(ss.normalize_listing_date(date(2018, 12, 24)), "2018-12-24")
        self.assertEqual(
            ss.normalize_listing_date(datetime(2018, 12, 24, 9, 30)),
            "2018-12-24",
        )
        for value in (None, "", "NaT", "nan", "--", "2018-02-30", "2999-01-01"):
            with self.subTest(value=value):
                self.assertIsNone(ss.normalize_listing_date(value))
        self.assertIsNone(ss.normalize_listing_date(pd.NaT))

    def test_v6_stock_meta_migrates_listing_date_without_rewriting_existing_rows(self):
        import stock_storage as ss

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "stock.sqlite3"
            conn = sqlite3.connect(db_path)
            try:
                conn.executescript("""
                    CREATE TABLE stock_meta (
                        code                    TEXT PRIMARY KEY,
                        name                    TEXT NOT NULL,
                        instrument_type         TEXT NOT NULL DEFAULT 'stock',
                        fetch_time              TEXT,
                        financials_refetched_at TEXT,
                        daily_refetched_at      TEXT,
                        history_refetched_at    TEXT,
                        daily_stats_json        TEXT,
                        financials_json         TEXT,
                        indicators_json         TEXT,
                        dividends_json          TEXT,
                        pledge_ratio            REAL,
                        pledge_count            REAL,
                        pledge_trade_date       TEXT,
                        industry                TEXT,
                        candidate_for_json      TEXT,
                        history_source          TEXT,
                        history_start_date      TEXT,
                        history_end_date        TEXT,
                        price_adjust            TEXT DEFAULT 'qfq',
                        updated_at              TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    INSERT INTO stock_meta
                        (code, name, history_start_date, history_end_date)
                    VALUES ('603629', '利通电子', '2018-12-24', '2026-07-17');
                    PRAGMA user_version = 6;
                """)
            finally:
                conn.close()

            conn = ss.connect(db_path)
            try:
                columns = {row["name"] for row in conn.execute("PRAGMA table_info(stock_meta)")}
                row = conn.execute(
                    "SELECT name, listing_date, history_start_date, history_end_date "
                    "FROM stock_meta WHERE code = '603629'"
                ).fetchone()
                user_version = conn.execute("PRAGMA user_version").fetchone()[0]
            finally:
                conn.close()

        self.assertIn("listing_date", columns)
        self.assertEqual(user_version, 7)
        self.assertEqual(row["name"], "利通电子")
        self.assertIsNone(row["listing_date"])
        self.assertEqual(row["history_start_date"], "2018-12-24")
        self.assertEqual(row["history_end_date"], "2026-07-17")

    def test_save_stock_round_trips_listing_date_and_never_clears_cached_value(self):
        import stock_storage as ss

        conn = ss.connect(":memory:")
        try:
            ss.save_stock(conn, {
                "symbol": "603629",
                "name": "利通电子",
                "listing_date": "2018/12/24",
            }, write_history=False)
            self.assertEqual(
                ss.load_stock(conn, "603629", include_history=False)["listing_date"],
                "2018-12-24",
            )

            # 历史调用没有 listing_date，不得把已缓存值整行 UPSERT 成 NULL。
            ss.save_stock(conn, {
                "symbol": "603629",
                "name": "利通电子",
            }, write_history=False)
            # 普通刷新即使传入冲突日期也只补空，不做静默纠错。
            ss.save_stock(conn, {
                "symbol": "603629",
                "name": "利通电子",
                "listing_date": "2019-01-01",
            }, write_history=False)

            self.assertEqual(ss.listing_date_map(conn)["603629"], "2018-12-24")
            self.assertEqual(
                ss.listing_date_map(conn, ["603629", "000001"]),
                {"603629": "2018-12-24", "000001": None},
            )
        finally:
            conn.close()

    def test_upsert_listing_dates_creates_stubs_fills_only_misses_and_preserves_etf(self):
        import stock_storage as ss

        conn = ss.connect(":memory:")
        try:
            ss.save_stock(conn, {
                "symbol": "000002",
                "name": "万科A",
            }, write_history=False)
            ss.save_stock(conn, {
                "symbol": "510300",
                "name": "沪深300ETF",
                "instrument_type": "etf",
                "listing_date": "2012-05-28",
            }, write_history=False)

            changed = ss.upsert_listing_dates(conn, {
                # stock_meta 中还没有该股：先建轻量 stub，即使日线失败也保留日期。
                "603629": {"name": "利通电子", "listing_date": "20181224"},
                # 已有 meta 但日期为空：只补这一列。
                "000002": {"name": "错误名称", "listing_date": "1991-01-29"},
                # 已有 ETF 日期不覆盖，名称和品种也不得被输入重分类。
                "510300": {
                    "name": "错误股票名",
                    "instrument_type": "stock",
                    "listing_date": "2020-01-01",
                },
                # 无效值不落库，以便下次仍能重试。
                "000003": {"name": "无效日期", "listing_date": "NaT"},
            })

            self.assertEqual(changed, 2)
            self.assertEqual(
                ss.listing_date_map(conn, ["603629", "000002", "510300", "000003"]),
                {
                    "603629": "2018-12-24",
                    "000002": "1991-01-29",
                    "510300": "2012-05-28",
                    "000003": None,
                },
            )
            stock_row = conn.execute(
                "SELECT name, instrument_type FROM stock_meta WHERE code = '000002'"
            ).fetchone()
            etf_row = conn.execute(
                "SELECT name, instrument_type FROM stock_meta WHERE code = '510300'"
            ).fetchone()
            self.assertEqual(tuple(stock_row), ("万科A", "stock"))
            self.assertEqual(tuple(etf_row), ("沪深300ETF", "etf"))

            # 同一批次执行无写入；明确修复时才允许覆盖日期。
            self.assertEqual(ss.upsert_listing_dates(conn, {
                "603629": {"name": "利通电子", "listing_date": "2018-12-24"},
            }), 0)
            self.assertEqual(ss.upsert_listing_dates(conn, {
                "603629": {"name": "不应改名", "listing_date": "2018-12-25"},
            }, overwrite=True), 1)
            repaired = conn.execute(
                "SELECT name, instrument_type, listing_date FROM stock_meta WHERE code = '603629'"
            ).fetchone()
            self.assertEqual(tuple(repaired), ("利通电子", "stock", "2018-12-25"))
        finally:
            conn.close()


class StockFilePathTest(unittest.TestCase):
    def test_save_stock_file_round_trips_through_sqlite(self):
        import stock_storage as ss
        with tempfile.TemporaryDirectory() as tmp:
            conn = ss.connect(Path(tmp) / "stock.sqlite3")
            ss._thread_local.conns = {str(ss.DEFAULT_DB_FILE): conn}
            try:
                save_stock_file("000001", "坏/名字:测试", {"records": [{
                    "date": "2026-01-01",
                    "daily_open": 10,
                    "daily_high": 11,
                    "daily_low": 9,
                    "daily_close": 10.5,
                    "daily_volume": 100,
                    "daily_amount": 1050,
                }]})

                # 主键是 6 位 code：名字异常或改名都不影响按 code 取回
                loaded = load_stock_file("000001", "新名字")
                self.assertEqual(loaded["records"][0]["date"], "2026-01-01")
                self.assertEqual(loaded["records"][0]["daily_close"], 10.5)
            finally:
                ss._thread_local.conns = {}
                conn.close()

    def test_save_stock_file_normalizes_raw_daily_aliases_before_sqlite_write(self):
        import stock_storage as ss
        with tempfile.TemporaryDirectory() as tmp:
            conn = ss.connect(Path(tmp) / "stock.sqlite3")
            ss._thread_local.conns = {str(ss.DEFAULT_DB_FILE): conn}
            try:
                save_stock_file("000001", "平安银行", {"records": [{
                    "date": "2026-01-01",
                    "open": 10,
                    "high": 11,
                    "low": 9,
                    "close": 10.5,
                    "volume": 100,
                    "amount": 1050,
                    "market_cap": 1234,
                }]})

                loaded = load_stock_file("000001", "平安银行")
                row = loaded["records"][0]
                self.assertEqual(row["daily_open"], 10)
                self.assertEqual(row["daily_high"], 11)
                self.assertEqual(row["daily_low"], 9)
                self.assertEqual(row["daily_close"], 10.5)
                self.assertEqual(row["daily_volume"], 100)
                self.assertEqual(row["daily_amount"], 1050)
                self.assertEqual(row["market_cap"], 1234)
            finally:
                ss._thread_local.conns = {}
                conn.close()

    def test_sync_sw3_member_market_caps_picks_latest_non_null(self):
        import stock_storage as ss
        with tempfile.TemporaryDirectory() as tmp:
            conn = ss.connect(Path(tmp) / "stock.sqlite3")
            ss._thread_local.conns = {str(ss.DEFAULT_DB_FILE): conn}
            try:
                ss.save_sw3_membership(conn, {
                    "segments": [{
                        "segment_code": "850111.SI",
                        "segment_name": "种子",
                        "parent_segment": "种植业",
                        "members": [{"code": "000001", "name": "平安银行", "market_cap_yi": None}],
                    }],
                })

                save_stock_file("000001", "平安银行", {"records": [
                    {
                        "date": "2026-01-01", "open": 9.8, "high": 10.2, "low": 9.7,
                        "close": 10.0, "volume": 1000, "market_cap": 100.0,
                    },
                    {
                        "date": "2026-01-02", "open": 10.1, "high": 10.7, "low": 10.0,
                        "close": 10.5, "volume": 1200, "market_cap": 123.4,
                    },
                    {
                        "date": "2026-01-03", "open": 10.6, "high": 10.9, "low": 10.4,
                        "close": 10.8, "volume": 1100, "market_cap": None,
                    },
                ]})

                synced = ss.sync_sw3_member_market_caps(conn)
                self.assertEqual(synced, 1)
                cap = conn.execute(
                    "SELECT market_cap_yi FROM sw3_member WHERE code = ?", ("000001",)
                ).fetchone()["market_cap_yi"]
                self.assertEqual(cap, 123.4)
            finally:
                ss._thread_local.conns = {}
                conn.close()

    def test_stock_storage_serializes_concurrent_history_writes(self):
        from concurrent.futures import ThreadPoolExecutor
        import stock_storage as ss

        def payload(idx):
            code = f"{idx:06d}"
            records = [
                {
                    "date": f"2026-01-{day:02d}",
                    "daily_open": 10 + idx,
                    "daily_high": 11 + idx,
                    "daily_low": 9 + idx,
                    "daily_close": 10.5 + idx,
                    "daily_volume": 1000 + day,
                    "daily_amount": 10000 + day,
                }
                for day in range(1, 21)
            ]
            return {
                "symbol": code,
                "name": f"并发{idx}",
                "history": history_payload_from_records(code, f"并发{idx}", records, "test"),
            }

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "stock.sqlite3"

            def worker(idx):
                conn = ss.connect(db_path)
                try:
                    ss.save_stock(conn, payload(idx))
                finally:
                    conn.close()

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(worker, range(1, 13)))

            conn = ss.connect(db_path)
            try:
                self.assertEqual(ss.table_count(conn, "stock_meta"), 12)
                self.assertEqual(ss.table_count(conn, "stock_history"), 12 * 20)
            finally:
                conn.close()

    def test_stock_db_writer_drains_all_enqueued_saves(self):
        saved = []

        def fake_save(code, name, data):
            saved.append((code, name, data["marker"]))

        with patch.object(stock_price_valuation, "save_stock_file", fake_save):
            writer = stock_price_valuation.StockDbWriter(maxsize=2)
            writer.start()
            for idx in range(5):
                writer.enqueue(f"{idx:06d}", f"测试{idx}", {"marker": idx})
            writer.wait()

        self.assertEqual(writer.enqueued, 5)
        self.assertEqual(writer.completed, 5)
        self.assertEqual(writer.failed, [])
        self.assertEqual([row[2] for row in saved], list(range(5)))


class StockRecordMergeTest(unittest.TestCase):
    def test_merge_records_by_date_can_overwrite_with_none(self):
        merged = merge_records_by_date(
            [{"date": "2026-01-01", "pb": 1.2, "close": 10}],
            [{"date": "2026-01-01", "pb": None, "close": 11}],
            overwrite_none=True,
        )

        self.assertIsNone(merged[0]["pb"])
        self.assertEqual(merged[0]["close"], 11)

    def test_merge_records_by_date_can_preserve_existing_when_new_value_is_none(self):
        merged = merge_records_by_date(
            [{"date": "2026-01-01", "pb": 1.2, "close": 10}],
            [{"date": "2026-01-01", "pb": None, "close": 11}],
            overwrite_none=False,
        )

        self.assertEqual(merged[0]["pb"], 1.2)
        self.assertEqual(merged[0]["close"], 11)


class StockHistorySchemaTest(unittest.TestCase):
    def complete_row(self, date):
        return {
            "date": date,
            "daily_open": 10,
            "daily_high": 11,
            "daily_low": 9,
            "daily_close": 10.5,
            "daily_volume": 10000,
            "daily_amount": 105000,
            "daily_change_pct": 1.2,
            "daily_turnover_rate": 2.3,
        }

    def test_history_payload_prunes_snapshot_only_rows(self):
        payload = history_payload_from_records(
            "000001",
            "测试股份",
            [
                self.complete_row("2026-06-11"),
                {"date": "2026-06-11"},
                {"date": "2026-06-12", "daily_close": 10.8, "market_cap": 100},
            ],
            "test",
        )

        self.assertEqual([row["date"] for row in payload["records"]], ["2026-06-11"])

    def test_daily_stats_include_average_turnover_rate(self):
        stats = daily_stats_from_history_records([
            self.complete_row("2026-06-11"),
            {**self.complete_row("2026-06-12"), "daily_turnover_rate": 3.7},
        ])

        self.assertAlmostEqual(stats["history_window_avg_daily_turnover_rate"], 3.0)

    def test_single_snapshot_row_does_not_trigger_full_ohlcv_backfill(self):
        records = [
            self.complete_row("2026-06-10"),
            self.complete_row("2026-06-11"),
            {"date": "2026-06-12", "daily_close": 10.8, "market_cap": 100},
        ]

        self.assertFalse(records_need_ohlcv_backfill(records))

    def test_majority_missing_ohlcv_still_triggers_backfill(self):
        records = [
            self.complete_row("2026-06-10"),
            {"date": "2026-06-11", "daily_close": 10.6, "daily_change_pct": 0.9},
            {"date": "2026-06-12", "daily_close": 10.8, "daily_change_pct": 1.1},
        ]

        self.assertTrue(records_need_ohlcv_backfill(records))

    def test_normalize_volume_to_hands_converts_share_units(self):
        self.assertEqual(
            _normalize_volume_to_hands(12501455, 1528515000, 124.42),
            125014.55,
        )

    def test_normalize_volume_to_hands_preserves_hand_units(self):
        self.assertEqual(
            _normalize_volume_to_hands(125014.55, 1528515000, 124.42),
            125014.55,
        )

    def test_qfq_return_discontinuity_detects_mixed_adjustment_scales(self):
        seam = qfq_return_discontinuity([
            {**self.complete_row("2026-07-01"), "daily_close": 41.2, "daily_change_pct": 1.402904},
            {**self.complete_row("2026-07-02"), "daily_close": 28.17, "daily_change_pct": -2.593361},
        ])

        self.assertEqual(seam["date"], "2026-07-02")
        self.assertAlmostEqual(seam["computed_change_pct"], -31.626214, places=5)
        self.assertAlmostEqual(seam["source_change_pct"], -2.593361, places=6)

    def test_qfq_overlap_scale_change_detects_rewritten_history(self):
        changed = qfq_overlap_scale_change(
            [{"date": "2026-07-01", "daily_close": 41.2}],
            [{"date": "2026-07-01", "close": 28.92}],
        )

        self.assertEqual(changed["date"], "2026-07-01")
        self.assertEqual(changed["old_close"], 41.2)
        self.assertEqual(changed["new_close"], 28.92)

    def test_qfq_overlap_scale_change_allows_small_cross_source_rounding(self):
        changed = qfq_overlap_scale_change(
            [{"date": "2026-07-01", "daily_close": 28.94}],
            [{"date": "2026-07-01", "close": 28.92}],
        )

        self.assertIsNone(changed)

    def test_qfq_return_discontinuity_ignores_nonpositive_long_horizon_qfq(self):
        seam = qfq_return_discontinuity([
            {**self.complete_row("2026-01-01"), "daily_close": 1.0, "daily_change_pct": 0.0},
            {**self.complete_row("2026-01-02"), "daily_close": -0.5, "daily_change_pct": -150.0},
            {**self.complete_row("2026-01-05"), "daily_close": 2.0, "daily_change_pct": -500.0},
        ])

        self.assertIsNone(seam)

    def test_process_stock_rewrites_full_history_when_existing_qfq_seam_is_found(self):
        existing = [
            {**self.complete_row("2026-07-01"), "daily_close": 41.2, "daily_change_pct": 1.402904},
            {**self.complete_row("2026-07-02"), "daily_close": 28.17, "daily_change_pct": -2.593361},
        ]
        refreshed = [
            {"date": "2026-07-01", "open": 28.12, "high": 29.6, "low": 28.0,
             "close": 28.92, "volume": 108508.84, "amount": 446790220,
             "change_pct": 1.402525, "turnover_rate": 2.9065},
            {"date": "2026-07-02", "open": 28.29, "high": 29.05, "low": 26.82,
             "close": 28.17, "volume": 93989.8, "amount": 378106574,
             "change_pct": -2.593361, "turnover_rate": 2.5176},
        ]
        saved = []

        with patch.object(
            stock_price_valuation,
            "load_stock_file",
            return_value={
                "records": existing,
                "start_date": "2026-07-01",
                "end_date": "2026-07-02",
            },
        ), patch.object(
            stock_price_valuation, "latest_weekday_date", return_value="2026-07-02"
        ), patch.object(
            stock_price_valuation, "fetch_daily_range", return_value=refreshed
        ) as fetch_mock:
            stock_price_valuation.process_stock(
                "603281",
                "江瀚新材",
                1,
                1,
                save_callback=lambda code, name, data: saved.append((code, name, data)),
                max_years=0.001,
                refresh_valuation=False,
                listing_date="2020-01-01",
            )

        self.assertEqual(fetch_mock.call_count, 1)
        result = saved[0][2]
        self.assertTrue(result["history_replace"])
        self.assertAlmostEqual(result["records"][0]["daily_close"], 28.92)
        self.assertAlmostEqual(result["records"][1]["daily_close"], 28.17)

    def test_process_stock_keeps_append_fast_path_when_overlap_scale_is_stable(self):
        existing = [
            {**self.complete_row("2026-07-01"), "daily_close": 28.92, "daily_change_pct": 1.402525},
        ]
        overlap = [
            {"date": "2026-07-01", "open": 28.12, "high": 29.6, "low": 28.0,
             "close": 28.92, "volume": 108508.84, "amount": 446790220,
             "change_pct": 1.402525, "turnover_rate": 2.9065},
            {"date": "2026-07-02", "open": 28.29, "high": 29.05, "low": 26.82,
             "close": 28.17, "volume": 93989.8, "amount": 378106574,
             "change_pct": -2.593361, "turnover_rate": 2.5176},
        ]
        saved = []

        with patch.object(
            stock_price_valuation,
            "load_stock_file",
            return_value={
                "records": existing,
                "start_date": "2026-07-01",
                "end_date": "2026-07-01",
            },
        ), patch.object(
            stock_price_valuation, "latest_weekday_date", return_value="2026-07-02"
        ), patch.object(
            stock_price_valuation, "fetch_daily_range", return_value=overlap
        ):
            stock_price_valuation.process_stock(
                "603281",
                "江瀚新材",
                1,
                1,
                save_callback=lambda code, name, data: saved.append((code, name, data)),
                max_years=0.001,
                refresh_valuation=False,
                listing_date="2020-01-01",
            )

        result = saved[0][2]
        self.assertFalse(result["history_replace"])
        self.assertEqual(
            [row["date"] for row in result["history_write_records"]],
            ["2026-07-01", "2026-07-02"],
        )

    def test_process_stock_skips_redundant_overlap_after_final_bar_refresh(self):
        existing = [self.complete_row("2026-07-02")]
        saved = []

        with patch.object(
            stock_price_valuation,
            "load_stock_file",
            return_value={
                "records": existing,
                "start_date": "2026-07-02",
                "end_date": "2026-07-02",
                "history_refetched_at": "2026-07-02T15:11:00",
            },
        ), patch.object(
            stock_price_valuation, "latest_weekday_date", return_value="2026-07-02"
        ), patch.object(
            stock_price_valuation, "fetch_daily_range"
        ) as fetch_mock:
            stock_price_valuation.process_stock(
                "603281",
                "江瀚新材",
                1,
                1,
                save_callback=lambda code, name, data: saved.append(data),
                max_years=0.0,
                refresh_valuation=False,
                listing_date="2020-01-01",
            )

        fetch_mock.assert_not_called()
        self.assertEqual(saved, [])

    def test_process_stock_prunes_future_row_and_resets_metadata_frontier(self):
        existing = [
            self.complete_row("2026-07-02"),
            self.complete_row("2026-07-03"),
        ]
        saved = []

        with patch.object(
            stock_price_valuation,
            "load_stock_file",
            return_value={
                "records": existing,
                "start_date": "2026-07-02",
                "end_date": "2026-07-03",
                "history_refetched_at": "2026-07-03T15:11:00",
            },
        ), patch.object(
            stock_price_valuation, "latest_weekday_date", return_value="2026-07-02"
        ), patch.object(
            stock_price_valuation, "fetch_daily_range"
        ) as fetch_mock:
            stock_price_valuation.process_stock(
                "603281",
                "江瀚新材",
                1,
                1,
                save_callback=lambda code, name, data: saved.append(data),
                max_years=0.0,
                refresh_valuation=False,
                listing_date="2020-01-01",
            )

        fetch_mock.assert_not_called()
        self.assertEqual(saved[0]["end_date"], "2026-07-02")
        self.assertEqual([row["date"] for row in saved[0]["records"]], ["2026-07-02"])
        self.assertTrue(saved[0]["history_replace"])

    def test_sw3_member_schema_adds_official_market_cap_ratio_to_old_db(self):
        import stock_storage as ss

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "stock.sqlite3"
            conn = sqlite3.connect(db_path)
            try:
                conn.executescript("""
                    CREATE TABLE sw3_segment (
                        segment_code TEXT PRIMARY KEY,
                        segment_name TEXT,
                        parent_segment TEXT,
                        member_count INTEGER,
                        refreshed_at TEXT,
                        error TEXT,
                        updated_at TEXT
                    );
                    CREATE TABLE sw3_member (
                        code TEXT PRIMARY KEY,
                        segment_code TEXT NOT NULL,
                        name TEXT,
                        price REAL,
                        market_cap_yi REAL,
                        roe_pct REAL,
                        profit_growth_pct REAL,
                        revenue_growth_pct REAL
                    );
                    PRAGMA user_version = 2;
                """)
            finally:
                conn.close()

            conn = ss.connect(db_path)
            try:
                cols = {row["name"] for row in conn.execute("PRAGMA table_info(sw3_member)")}
                user_version = conn.execute("PRAGMA user_version").fetchone()[0]
            finally:
                conn.close()

        self.assertIn("official_market_cap_ratio", cols)
        self.assertIn("is_hot_money", cols)
        self.assertEqual(user_version, ss.SCHEMA_VERSION)

    def test_short_signal_snapshot_does_not_define_hot_money_membership(self):
        import stock_storage as ss

        with tempfile.TemporaryDirectory() as tmpdir:
            conn = ss.connect(Path(tmpdir) / "stock.sqlite3")
            try:
                ss.save_sw3_membership(conn, {
                    "segments": [{
                        "segment_code": "850111",
                        "segment_name": "测试行业",
                        "parent_segment": "测试一级",
                        "members": [
                            {"code": "000001", "name": "成员甲"},
                            {"code": "000002", "name": "成员乙"},
                        ],
                    }],
                    "errors": [],
                })
                conn.execute("UPDATE sw3_member SET is_hot_money = 1")
                conn.commit()

                saved = ss.replace_short_signals(
                    conn,
                    {"000001": {"name": "成员甲", "signals": {"lhb": {"count": 3}}}},
                    generated_at="2026-07-10 10:00:00",
                    as_of_date="2026-07-10",
                )

                self.assertEqual(saved, 1)
                self.assertEqual(
                    {row["code"] for row in ss.pool_members(conn, "hotmoney")},
                    {"000001", "000002"},
                )
                self.assertEqual(set(ss.load_short_signals(conn)), {"000001"})
                self.assertEqual(ss.short_signal_status(conn)["count"], 1)

                ss.replace_short_signals(
                    conn,
                    {"000002": {"name": "成员乙", "signals": {}}},
                    generated_at="2026-07-10 11:00:00",
                    as_of_date="2026-07-10",
                )
                self.assertEqual(set(ss.load_short_signals(conn)), {"000002"})
                self.assertEqual(
                    {row["code"] for row in ss.pool_members(conn, "hotmoney")},
                    {"000001", "000002"},
                )
            finally:
                conn.close()

    def test_sw3_membership_persists_official_market_cap_ratio(self):
        import stock_storage as ss

        membership = {
            "segments": [{
                "segment_code": "850111", "segment_name": "种子", "parent_segment": "农业",
                "member_count": 2,
                "members": [
                    {"code": "000998", "name": "隆平高科", "official_market_cap_ratio": 29.16},
                    {"code": "002041", "name": "登海种业", "index_weight": 10.36},
                ],
            }],
            "errors": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = ss.connect(Path(tmpdir) / "stock.sqlite3")
            try:
                ss.save_sw3_membership(conn, membership)
                reloaded = ss.load_sw3_membership(conn, max_age_days=None)
            finally:
                conn.close()

        members = reloaded["segments"][0]["members"]
        ratios = {m["code"]: m["official_market_cap_ratio"] for m in members}
        self.assertEqual(ratios, {"000998": 29.16, "002041": 10.36})

    def test_db_signature_tracks_sw3_membership_tables(self):
        import stock_storage as ss

        membership = {
            "segments": [{
                "segment_code": "851024",
                "segment_name": "通信网络设备及器件",
                "parent_segment": "通信设备",
                "member_count": 1,
                "members": [{"code": "300308", "name": "中际旭创"}],
            }],
            "errors": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "stock.sqlite3"
            conn = ss.connect(db_path)
            try:
                before = ss.db_signature(db_path)
                ss.save_sw3_membership(conn, membership)
                after = ss.db_signature(db_path)
            finally:
                conn.close()

        self.assertEqual(before["sw3_segments"], 0)
        self.assertEqual(before["sw3_members"], 0)
        self.assertEqual(after["sw3_segments"], 1)
        self.assertEqual(after["sw3_members"], 1)
        self.assertIsNotNone(after["sw3_max_updated_at"])


class StockStrategyOptimizerTest(unittest.TestCase):
    def test_optuna_startup_trials_keeps_small_runs_exploratory(self):
        from stock_strategy_optimizer import optuna_startup_trials

        self.assertEqual(optuna_startup_trials(1), 1)
        self.assertEqual(optuna_startup_trials(20), 20)
        self.assertEqual(optuna_startup_trials(200), 70)
        self.assertEqual(optuna_startup_trials(300), 105)

    def test_high_drawdown_threshold_is_clamped_with_independent_switch(self):
        from stock_strategy_optimizer import _default_long_params, set_long_high_drawdown_filter

        cfg = {"require_high_drawdown": False}
        set_long_high_drawdown_filter(cfg, 0)
        self.assertFalse(cfg["require_high_drawdown"])
        self.assertEqual(cfg["min_high_drawdown_pct"], 40)

        cfg["require_high_drawdown"] = True
        set_long_high_drawdown_filter(cfg, 99)
        self.assertTrue(cfg["require_high_drawdown"])
        self.assertEqual(cfg["min_high_drawdown_pct"], 70)
        params = _default_long_params()
        self.assertFalse(params["require_high_drawdown"])
        self.assertEqual(params["min_high_drawdown_pct"], 40)

    def test_broad_long_candidates_precompute_high_drawdown_factor(self):
        from stock_strategy_optimizer import broad_long_candidate_config

        cfg = broad_long_candidate_config()

        self.assertFalse(cfg["require_high_drawdown"])
        self.assertEqual(cfg["min_high_drawdown_pct"], 0)
        self.assertEqual(cfg["min_score"], 0)

    def test_long_price_history_validation_rejects_missing_closes(self):
        from stock_strategy_optimizer import ensure_long_price_history

        series = {
            "000001": [
                {"date": "2026-01-01", "close": None, "market_cap": 100.0},
                {"date": "2026-01-02", "close": None, "market_cap": 101.0},
            ]
        }

        with self.assertRaisesRegex(RuntimeError, "daily_close/OHLCV"):
            ensure_long_price_history(series)

    def test_select_best_long_trace_rejects_all_invalid_trials(self):
        from stock_strategy_optimizer import select_best_long_trace

        trace = [
            {"objective": -999.0, "selected_count": 0, "summary": {"folds": 0}},
            {"objective": -999.0, "selected_count": 3, "summary": {"folds": 2}},
        ]

        with self.assertRaisesRegex(RuntimeError, "没有产生有效 trial"):
            select_best_long_trace(trace)

    def test_default_strategy_top_n_uses_named_constants(self):
        from stock_advanced_strategies import (
            DEFAULT_CONFIG,
            LONG_DEFAULT_TOP_N,
            SHORT_DEFAULT_TOP_N,
            get_default_config,
        )

        cfg = get_default_config()
        self.assertEqual(LONG_DEFAULT_TOP_N, 20)
        self.assertEqual(SHORT_DEFAULT_TOP_N, 10)
        self.assertEqual(cfg["long"]["top_n"], LONG_DEFAULT_TOP_N)
        self.assertEqual(cfg["short"]["top_n"], SHORT_DEFAULT_TOP_N)
        self.assertFalse(DEFAULT_CONFIG["long"]["require_high_drawdown"])

    def test_long_objective_penalizes_sparse_fold_coverage(self):
        from stock_strategy_optimizer import (
            LONG_FOLD_COUNT_PENALTY,
            LONG_SOFT_TARGET_FOLDS,
            long_validation_adjusted_objective,
        )

        pairs = [
            {
                "as_of": f"2024-{(idx % 12) + 1:02d}-01",
                "cal_idx": idx * 30,
                "portfolio_return": 0.12,
                "benchmark_return": 0.02,
                "excess_return": 0.10,
                "portfolio_max_drawdown": 0.10,
                "benchmark_max_drawdown": 0.12,
            }
            for idx in range(LONG_SOFT_TARGET_FOLDS)
        ]
        sparse_pairs = pairs[:20]

        _, sparse_detail = long_validation_adjusted_objective(
            sparse_pairs[:12], sparse_pairs[12:], sparse_pairs, 125, 62
        )
        _, full_detail = long_validation_adjusted_objective(
            pairs[:24], pairs[24:], pairs, 125, 62
        )

        expected_penalty = (LONG_SOFT_TARGET_FOLDS - len(sparse_pairs)) * LONG_FOLD_COUNT_PENALTY
        self.assertEqual(sparse_detail["fold_count_penalty"], round(expected_penalty, 5))
        self.assertEqual(full_detail["fold_count_penalty"], 0)

    def test_recency_weights_are_equal_for_all_folds(self):
        from stock_strategy_optimizer import apply_recency_weights, long_fold_summary

        pairs = apply_recency_weights([
            {
                "as_of": "2024-01-01",
                "cal_idx": 10,
                "portfolio_return": -0.2,
                "benchmark_return": 0.0,
                "excess_return": -0.2,
                "portfolio_max_drawdown": 0.1,
                "benchmark_max_drawdown": 0.1,
            },
            {
                "as_of": "2024-02-01",
                "cal_idx": 20,
                "portfolio_return": 0.0,
                "benchmark_return": 0.0,
                "excess_return": 0.0,
                "portfolio_max_drawdown": 0.1,
                "benchmark_max_drawdown": 0.1,
            },
            {
                "as_of": "2024-03-01",
                "cal_idx": 30,
                "portfolio_return": 0.2,
                "benchmark_return": 0.0,
                "excess_return": 0.2,
                "portfolio_max_drawdown": 0.1,
                "benchmark_max_drawdown": 0.1,
            },
        ])

        self.assertEqual([row["fold_weight"] for row in pairs], [1.0, 1.0, 1.0])
        summary = long_fold_summary(pairs, 125)
        self.assertEqual(summary["min_fold_weight"], 1.0)
        self.assertEqual(summary["max_fold_weight"], 1.0)
        self.assertEqual(summary["avg_excess_pct"], 0.0)
        self.assertEqual(summary["hit_rate"], 33.33)

    def test_svg_marks_partial_folds_as_display_only(self):
        from stock_strategy_optimizer import write_long_fold_paths_svg

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "folds.svg"
            first_holdings = [
                {"code": f"0000{idx:02d}", "name": f"样本{idx}"}
                for idx in range(1, 13)
            ]
            second_holdings = first_holdings[:10] + [
                {"code": "000099", "name": "换手样本A"},
                {"code": "000100", "name": "换手样本B"},
            ]
            meta = write_long_fold_paths_svg(
                [
                    {
                        "as_of": "2026-01-01",
                        "dates": ["2026-01-01", "2026-01-02", "2026-01-03"],
                        "target_hold_td": 2,
                        "actual_hold_td": 2,
                        "partial": False,
                        "portfolio_path": [1.0, 1.1, 1.2],
                        "benchmark_path": [1.0, 1.05, 1.1],
                        "stock_count": 12,
                        "portfolio_return": 0.196,
                        "benchmark_return": 0.1,
                        "excess_return": 0.096,
                        "holdings": first_holdings,
                    },
                    {
                        "as_of": "2026-01-03",
                        "dates": ["2026-01-03", "2026-01-04"],
                        "target_hold_td": 2,
                        "actual_hold_td": 1,
                        "partial": True,
                        "portfolio_path": [1.0, 1.3],
                        "benchmark_path": [1.0, 1.0],
                        "stock_count": 12,
                        "portfolio_return": 0.296,
                        "benchmark_return": 0.0,
                        "excess_return": 0.296,
                        "holdings": second_holdings,
                    },
                ],
                output,
                2,
                holding_top_n=12,
            )

            svg = output.read_text(encoding="utf-8")

        self.assertEqual(meta["folds"], 1)
        self.assertEqual(meta["partial_folds"], 1)
        self.assertEqual(meta["displayed_folds"], 2)
        self.assertEqual(meta["chart_type"], "small_multiples_with_full_path")
        self.assertEqual(meta["full_path_start_date"], "2026-01-01")
        self.assertEqual(meta["full_path_end_date"], "2026-01-04")
        self.assertEqual(meta["full_portfolio_final_nav"], 1.56)
        self.assertEqual(meta["replacement_top_n"], 12)
        self.assertEqual(meta["latest_topn_replaced_count"], 2)
        self.assertEqual(meta["topn_replacement_counts"][0]["replaced_count"], None)
        self.assertEqual(meta["topn_replacement_counts"][1]["replaced_count"], 2)
        self.assertIn("未满 1/2日", svg)
        self.assertIn("Top12替换 2只", svg)
        self.assertIn("组合等权平均期末 1.20x（完整折）", svg)
        self.assertIn("完整历史拼接收益图", svg)


class HotMoneySegmentPoolTest(unittest.TestCase):
    def test_retry_fetch_logs_and_sleeps_fractional_retry_delay(self):
        import stock_crawl_segment_leaders as r

        calls = {"count": 0}

        def flaky():
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("legulegu failed")
            return "ok"

        output = io.StringIO()
        with patch.object(r.time, "sleep") as sleep_mock, patch("sys.stdout", output):
            result = r._retry_fetch(
                flaky,
                retries=2,
                sleep_sec=r.LEGULEGU_RETRY_SLEEP_SEC,
                backoff=False,
                desc="legulegu",
            )

        self.assertEqual(result, "ok")
        sleep_mock.assert_called_once_with(0.5)
        self.assertIn("0.5s 后重试", output.getvalue())

    def test_trillion_cap_leader_survives_build_filter(self):
        import pandas as pd
        from stock_crawl_segment_leaders import _parse_segment_rows

        df = pd.DataFrame([
            {"股票代码": "300308", "股票简称": "中际旭创", "价格": 1248.09, "市值（亿元）": 13919.0},
            {"股票代码": "300394", "股票简称": "天孚通信", "价格": 324.71, "市值（亿元）": 3542.0},
        ])
        # 无市值/股价上限：万亿龙头中际旭创应保留
        kept = {r["code"] for r in _parse_segment_rows(
            df, min_market_cap_yi=10.0, max_market_cap_yi=None, max_price=None)}
        self.assertEqual(kept, {"300308", "300394"})
        # 对照：若仍设 6000 亿上限，中际旭创会被过滤、天孚保留 —— 印证上限才是拦路虎
        capped = {r["code"] for r in _parse_segment_rows(
            df, min_market_cap_yi=10.0, max_market_cap_yi=6000.0, max_price=None)}
        self.assertNotIn("300308", capped)
        self.assertIn("300394", capped)

    def test_official_sw_component_fetch_normalizes_membership(self):
        import stock_crawl_segment_leaders as r

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "data": {
                        "results": [
                            {
                                "stockcode": "300308",
                                "stockname": "中际旭创",
                                "newweight": "12.3",
                                "beginningdate": "2026-01-01",
                            }
                        ]
                    }
                }

        with patch.object(r.requests, "get", return_value=FakeResponse()) as fake_get:
            df = r.fetch_sw3_segment_constituents_official("850111.SI", timeout=7)

        _args, kwargs = fake_get.call_args
        self.assertEqual(kwargs["params"]["swindexcode"], "850111")
        self.assertEqual(kwargs["timeout"], 7)
        self.assertFalse(kwargs["verify"])
        self.assertEqual(df.to_dict("records"), [{
            "证券代码": "300308",
            "证券名称": "中际旭创",
            "市值占比": "12.3",
            "计入日期": "2026-01-01",
        }])

    def test_segment_member_fetch_falls_back_to_official_source(self):
        import pandas as pd
        import stock_crawl_segment_leaders as r

        official_df = pd.DataFrame([
            {"证券代码": "300308", "证券名称": "中际旭创", "市值占比": "12.3"},
            {"证券代码": "430001", "证券名称": "北交样本"},
            {"证券代码": "600001", "证券名称": "ST样本"},
        ])
        metrics = {
            "300308": {
                "price": 1248.0,
                "market_cap_yi": 13915.0,
                "roe_pct": 22.0,
                "profit_growth_pct": 150.0,
                "revenue_growth_pct": 60.0,
            }
        }

        with patch.object(r, "fetch_sw3_segment_constituents", return_value=pd.DataFrame()), \
             patch.object(r, "fetch_sw3_segment_constituents_official", return_value=official_df), \
             patch.object(r, "load_latest_member_metrics", return_value=metrics), \
             patch.dict(r._LATEST_MEMBER_METRICS_CACHE, {}, clear=True):
            members, source, err = r._fetch_segment_members("850111")

        self.assertEqual(source, "官方")
        self.assertEqual(err, "")
        self.assertEqual(members, [{
            "code": "300308",
            "name": "中际旭创",
            "price": 1248.0,
            "market_cap_yi": 13915.0,
            "roe_pct": 22.0,
            "profit_growth_pct": 150.0,
            "revenue_growth_pct": 60.0,
            "official_market_cap_ratio": 12.3,
            "index_weight": 12.3,
        }])

    def test_segment_member_fetch_reports_official_excluded_only(self):
        import pandas as pd
        import stock_crawl_segment_leaders as r

        with patch.object(r, "fetch_sw3_segment_constituents", side_effect=RuntimeError("legulegu 504")), \
             patch.object(r, "fetch_sw3_segment_constituents_official", return_value=pd.DataFrame([
                 {"证券代码": "000615", "证券名称": "*ST美谷", "最新权重": "100.0"},
             ])), \
             patch.object(r.time, "sleep", return_value=None):
            members, source, err = r._fetch_segment_members("859832.SI")

        self.assertEqual(members, [])
        self.assertEqual(source, "")
        self.assertIn("legulegu 504", err)
        self.assertIn("官方回退仅返回被过滤股票", err)
        self.assertIn("000615 *ST美谷", err)

    def test_eastmoney_spot_metrics_fill_price_and_market_cap(self):
        import stock_crawl_segment_leaders as r

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "data": {
                        "diff": [
                            {"f12": "300308", "f14": "中际旭创", "f2": 124.8, "f20": 1391500000000},
                            {"f12": "430001", "f14": "北交样本", "f2": 10.0, "f20": 1000000000},
                        ]
                    }
                }

        with patch.object(r.requests, "get", return_value=FakeResponse()) as fake_get:
            metrics = r.fetch_a_spot_member_metrics(timeout=6)

        _args, kwargs = fake_get.call_args
        self.assertEqual(kwargs["params"]["fields"], "f2,f12,f14,f20")
        self.assertEqual(kwargs["timeout"], 6)
        self.assertEqual(metrics["300308"]["price"], 124.8)
        self.assertEqual(metrics["300308"]["market_cap_yi"], 13915.0)
        self.assertNotIn("430001", metrics)

    def test_empty_db_member_metrics_use_realtime_spot_fallback(self):
        from pathlib import Path
        import stock_crawl_segment_leaders as r
        import stock_storage as ss

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "stock.sqlite3"
            conn = ss.connect(db_path)
            conn.close()
            with patch.object(r, "fetch_a_spot_member_metrics", return_value={
                "300308": {
                    "price": 124.8,
                    "market_cap_yi": 13915.0,
                    "roe_pct": None,
                    "profit_growth_pct": None,
                    "revenue_growth_pct": None,
                }
            }):
                metrics = r.load_latest_member_metrics(db_path)

        self.assertEqual(metrics["300308"]["price"], 124.8)
        self.assertEqual(metrics["300308"]["market_cap_yi"], 13915.0)

    def test_empty_db_member_metrics_can_skip_realtime_spot_fallback(self):
        from pathlib import Path
        import stock_crawl_segment_leaders as r
        import stock_storage as ss

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "stock.sqlite3"
            conn = ss.connect(db_path)
            conn.close()
            with patch.object(r, "fetch_a_spot_member_metrics") as fake_spot:
                metrics = r.load_latest_member_metrics(db_path, allow_spot_fallback=False)

        self.assertEqual(metrics, {})
        fake_spot.assert_not_called()


class HotMoneyFastBuildTest(unittest.TestCase):
    def test_score_segment_leaders_uses_three_factor_weights(self):
        import stock_crawl_segment_leaders as r

        rows = [
            {"code": "600000", "name": "A", "size_proxy": 100.0, "roe_pct": 10.0,
             "profit_growth_pct": 20.0, "revenue_growth_pct": 20.0},
            {"code": "600001", "name": "B", "size_proxy": 50.0, "roe_pct": 5.0,
             "profit_growth_pct": 10.0, "revenue_growth_pct": 10.0},
        ]

        r.score_segment_leaders(rows)

        self.assertEqual(rows[0]["leader_score"], 100.0)
        self.assertEqual(rows[1]["leader_score"], 0.0)

    def test_build_does_not_fetch_official_weights_by_default(self):
        from datetime import datetime
        from pathlib import Path
        import stock_crawl_segment_leaders as r
        import stock_storage as ss

        membership = {
            "schema": r.SW3_MEMBERSHIP_SCHEMA,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "segments": [{
                "segment_code": "850111", "segment_name": "种子",
                "parent_segment": "农业", "member_count": 1,
                "members": [
                    {"code": "000998", "name": "隆平高科", "price": None, "market_cap_yi": None,
                     "roe_pct": None, "profit_growth_pct": None, "revenue_growth_pct": None},
                ],
            }],
            "errors": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path, pool_path = tmp / "stock.sqlite3", tmp / "pool.json"
            conn = ss.connect(db_path)
            ss.save_sw3_membership(conn, membership)
            conn.close()
            with patch.object(r, "STOCK_DB_FILE", db_path), \
                 patch.object(r, "SEGMENT_LEADER_POOL_FILE", pool_path), \
                 patch.object(r, "DATA_DIR", tmp), \
                 patch.object(r, "load_latest_member_metrics", return_value={}), \
                 patch.object(r, "_fetch_official_market_cap_ratio_map") as fetch_ratios, \
                 patch.object(r, "membership_db_needs_full_recrawl", return_value=(False, 0, 0)):
                payload = r.build_segment_leader_pool(top_per_segment=1, min_market_cap_yi=10.0)

        fetch_ratios.assert_not_called()
        self.assertEqual(payload["segment_count"], 0)

    def test_build_can_rank_with_official_market_cap_ratio_when_market_cap_missing(self):
        from datetime import datetime
        from pathlib import Path
        import stock_crawl_segment_leaders as r
        import stock_storage as ss

        membership = {
            "schema": r.SW3_MEMBERSHIP_SCHEMA,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "segments": [{
                "segment_code": "850111", "segment_name": "种子",
                "parent_segment": "农业", "member_count": 2,
                "members": [
                    {"code": "000998", "name": "隆平高科", "price": None, "market_cap_yi": None,
                     "roe_pct": None, "profit_growth_pct": None, "revenue_growth_pct": None},
                    {"code": "002041", "name": "登海种业", "price": None, "market_cap_yi": None,
                     "roe_pct": None, "profit_growth_pct": None, "revenue_growth_pct": None},
                ],
            }],
            "errors": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path, pool_path = tmp / "stock.sqlite3", tmp / "pool.json"
            conn = ss.connect(db_path)
            ss.save_sw3_membership(conn, membership)
            conn.close()
            with patch.object(r, "STOCK_DB_FILE", db_path), \
                 patch.object(r, "SEGMENT_LEADER_POOL_FILE", pool_path), \
                 patch.object(r, "DATA_DIR", tmp), \
                 patch.object(r, "load_latest_member_metrics", return_value={}), \
                 patch.object(r, "_fetch_official_market_cap_ratio_map",
                              return_value={"000998": 29.16, "002041": 10.36}), \
                 patch.object(r, "membership_db_needs_full_recrawl", return_value=(False, 0, 0)):
                payload = r.build_segment_leader_pool(
                    top_per_segment=1,
                    min_market_cap_yi=10.0,
                    enrich_weights_online=True,
                )

        leader = payload["segments"][0]["leaders"][0]
        self.assertEqual(leader["code"], "000998")
        self.assertEqual(leader["market_cap_yi"], None)
        self.assertEqual(leader["official_market_cap_ratio"], 29.16)
        self.assertEqual(leader["index_weight"], 29.16)
        self.assertEqual(leader["size_basis"], "official_market_cap_ratio")

    def test_build_enriches_cached_official_members_before_filtering(self):
        from datetime import datetime
        from pathlib import Path
        import stock_crawl_segment_leaders as r
        import stock_storage as ss

        membership = {
            "schema": r.SW3_MEMBERSHIP_SCHEMA,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "segments": [{
                "segment_code": "850111", "segment_name": "种子",
                "parent_segment": "农业", "member_count": 2,
                "members": [
                    {"code": "300308", "name": "中际旭创", "price": None, "market_cap_yi": None,
                     "roe_pct": None, "profit_growth_pct": None, "revenue_growth_pct": None},
                    {"code": "300394", "name": "天孚通信", "price": None, "market_cap_yi": None,
                     "roe_pct": None, "profit_growth_pct": None, "revenue_growth_pct": None},
                ],
            }],
            "errors": [],
        }
        metrics = {
            "300308": {"price": 124.0, "market_cap_yi": 13915.0, "roe_pct": 22.0,
                       "profit_growth_pct": 150.0, "revenue_growth_pct": 60.0},
            "300394": {"price": 324.0, "market_cap_yi": 3532.0, "roe_pct": 18.0,
                       "profit_growth_pct": 80.0, "revenue_growth_pct": 40.0},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path, pool_path = tmp / "stock.sqlite3", tmp / "pool.json"
            conn = ss.connect(db_path)
            ss.save_sw3_membership(conn, membership)
            conn.close()
            with patch.object(r, "STOCK_DB_FILE", db_path), \
                 patch.object(r, "SEGMENT_LEADER_POOL_FILE", pool_path), \
                 patch.object(r, "DATA_DIR", tmp), \
                 patch.object(r, "load_latest_member_metrics", return_value=metrics), \
                 patch.object(r, "membership_db_needs_full_recrawl", return_value=(False, 0, 0)):
                payload = r.build_segment_leader_pool(top_per_segment=1, min_market_cap_yi=10.0)
                reloaded = r.load_sw3_membership(None)

        self.assertEqual(payload["segment_count"], 1)
        self.assertEqual(payload["segments"][0]["leaders"][0]["code"], "300308")
        self.assertEqual(payload["segments"][0]["size_basis"], "market_cap_yi")
        self.assertEqual(payload["segments"][0]["leaders"][0]["size_basis"], "market_cap_yi")
        saved_members = reloaded["segments"][0]["members"]
        self.assertEqual(saved_members[0]["market_cap_yi"], 13915.0)

    def test_build_uses_one_size_basis_per_segment(self):
        from datetime import datetime
        from pathlib import Path
        import stock_crawl_segment_leaders as r

        membership = {
            "schema": r.SW3_MEMBERSHIP_SCHEMA,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "segments": [{
                "segment_code": "850000", "segment_name": "混合口径测试",
                "parent_segment": "测试", "member_count": 2,
                "members": [
                    {"code": "600000", "name": "浦发银行", "price": 10.0,
                     "market_cap_yi": 10000.0, "official_market_cap_ratio": 1.0,
                     "roe_pct": None, "profit_growth_pct": None, "revenue_growth_pct": None},
                    {"code": "600001", "name": "邯郸钢铁", "price": 10.0,
                     "market_cap_yi": None, "official_market_cap_ratio": 50.0,
                     "roe_pct": None, "profit_growth_pct": None, "revenue_growth_pct": None},
                ],
            }],
            "errors": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(r, "load_sw3_membership", return_value=membership), \
                 patch.object(r, "SEGMENT_LEADER_POOL_FILE", tmp / "pool.json"), \
                 patch.object(r, "DATA_DIR", tmp), \
                 patch.object(r, "STOCK_DB_FILE", tmp / "stock.sqlite3"), \
                 patch.object(r, "_enrich_membership_member_metrics", return_value=(0, 0)), \
                 patch.object(r, "membership_db_needs_full_recrawl", return_value=(False, 0, 0)):
                payload = r.build_segment_leader_pool(top_per_segment=1, min_market_cap_yi=10.0)

        segment = payload["segments"][0]
        leader = segment["leaders"][0]
        self.assertEqual(segment["size_basis"], "official_market_cap_ratio")
        self.assertEqual(leader["code"], "600001")
        self.assertEqual(leader["size_basis"], "official_market_cap_ratio")
        self.assertEqual(leader["official_market_cap_ratio"], 50.0)

    def test_build_uses_cached_metrics_without_recent_strength(self):
        import json
        from datetime import datetime
        from pathlib import Path
        import stock_crawl_segment_leaders as r

        # 归属缓存直接携带 legulegu 抓的 价/市值/ROE/成长，不再带动量字段
        membership = {
            "schema": r.SW3_MEMBERSHIP_SCHEMA,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "segments": [{
                "segment_code": "851024", "segment_name": "通信网络设备及器件",
                "parent_segment": "通信设备", "member_count": 35,
                "members": [
                    {"code": "300308", "name": "中际旭创", "price": 1248.0, "market_cap_yi": 13915.0,
                     "roe_pct": 22.0, "profit_growth_pct": 150.0, "revenue_growth_pct": 60.0},
                    {"code": "300394", "name": "天孚通信", "price": 324.0, "market_cap_yi": 3532.0,
                     "roe_pct": 18.0, "profit_growth_pct": 80.0, "revenue_growth_pct": 40.0},
                    {"code": "002281", "name": "光迅科技", "price": 230.0, "market_cap_yi": 1903.0,
                     "roe_pct": 10.0, "profit_growth_pct": 30.0, "revenue_growth_pct": 20.0},
                ],
            }], "errors": []}
        import stock_storage as ss
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path, pool_path = tmp / "stock.sqlite3", tmp / "pool.json"
            conn = ss.connect(db_path)
            ss.save_sw3_membership(conn, membership)
            conn.close()
            with patch.object(r, "STOCK_DB_FILE", db_path), \
                 patch.object(r, "SEGMENT_LEADER_POOL_FILE", pool_path), \
                 patch.object(r, "DATA_DIR", tmp), \
                 patch.object(r, "membership_db_needs_full_recrawl", return_value=(False, 0, 0)):
                payload = r.build_segment_leader_pool()

        # 总榜(stocks)已删除：池子只保留按赛道分组的 leaders
        self.assertNotIn("stocks", payload)
        self.assertNotIn("stock_count", payload)
        segment = payload["segments"][0]
        leaders = segment["leaders"]
        # 赛道内综合分 top2(默认 top_per_segment=2)，市值直接进 leaders 供展示
        self.assertEqual([x["code"] for x in leaders], ["300308", "300394"])
        self.assertEqual(leaders[0]["market_cap_yi"], 13915.0)

    def test_build_appends_forced_leaders_after_top_n(self):
        import tempfile
        from datetime import datetime
        from pathlib import Path
        import stock_crawl_segment_leaders as r
        import stock_storage as ss

        membership = {
            "schema": r.SW3_MEMBERSHIP_SCHEMA,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "segments": [{
                "segment_code": "850333", "segment_name": "电子化学品",
                "parent_segment": "基础化工", "member_count": 3,
                "members": [
                    {"code": "300308", "name": "中际旭创", "price": 1248.0, "market_cap_yi": 13915.0,
                     "roe_pct": 22.0, "profit_growth_pct": 150.0, "revenue_growth_pct": 60.0},
                    {"code": "300394", "name": "天孚通信", "price": 324.0, "market_cap_yi": 3532.0,
                     "roe_pct": 18.0, "profit_growth_pct": 80.0, "revenue_growth_pct": 40.0},
                    {"code": "002741", "name": "光华科技", "price": 18.0, "market_cap_yi": 80.0,
                     "roe_pct": 5.0, "profit_growth_pct": 10.0, "revenue_growth_pct": 5.0},
                ],
            }], "errors": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path, pool_path = tmp / "stock.sqlite3", tmp / "pool.json"
            conn = ss.connect(db_path)
            ss.save_sw3_membership(conn, membership)
            conn.close()
            with patch.object(r, "STOCK_DB_FILE", db_path), \
                 patch.object(r, "SEGMENT_LEADER_POOL_FILE", pool_path), \
                 patch.object(r, "DATA_DIR", tmp), \
                 patch.object(r, "membership_db_needs_full_recrawl", return_value=(False, 0, 0)):
                payload = r.build_segment_leader_pool(
                    top_per_segment=1,
                    forced_leader_codes=["002741"],
                )

        segment = payload["segments"][0]
        self.assertEqual([x["code"] for x in segment["leaders"]], ["300308", "002741"])
        self.assertEqual(segment["leaders"][1]["rank"], 2)
        self.assertEqual(payload["params"]["forced_leader_codes"], ["002741"])


class HotMoneySliceRefreshTest(unittest.TestCase):
    def test_price_valuation_refreshes_15_memberships_by_default(self):
        import stock_crawl_segment_leaders as r

        payload = {
            "segments": [{
                "leaders": [{"code": "600000", "name": "浦发银行"}],
            }],
        }
        with patch.object(stock_price_valuation, "SEGMENT_REFRESH_SLICE", 15), \
             patch.object(r, "build_segment_leader_pool", return_value=payload) as build_mock:
            stocks = stock_price_valuation.get_segment_leader_stocks()

        build_mock.assert_called_once_with(
            top_per_segment=r.DEFAULT_TOP_PER_SEGMENT,
            refresh_slice=15,
            refresh_industry_heat=True,
        )
        self.assertEqual(stocks, {"600000": "浦发银行"})

    def test_build_recrawls_when_db_segments_below_backup_threshold(self):
        import json
        from pathlib import Path
        import stock_crawl_segment_leaders as r
        import stock_storage as ss

        bad_cache = {
            "schema": r.SW3_MEMBERSHIP_SCHEMA,
            "segments": [
                {"segment_code": "A", "segment_name": "赛道A", "parent_segment": "父", "member_count": 1,
                 "members": [{"code": "600001", "name": "旧A", "price": 10.0, "market_cap_yi": 100.0}]},
                {"segment_code": "B", "segment_name": "赛道B", "parent_segment": "父", "member_count": 1,
                 "members": [{"code": "600002", "name": "旧B", "price": 10.0, "market_cap_yi": 100.0}]},
            ],
            "errors": [],
        }
        rebuilt = {
            "schema": r.SW3_MEMBERSHIP_SCHEMA,
            "generated_at": "2026-06-20 01:00:00",
            "segments": [{
                "segment_code": "850111.SI", "segment_name": "种子", "parent_segment": "种植业", "member_count": 1,
                "members": [{"code": "000998", "name": "隆平高科", "price": 10.0, "market_cap_yi": 200.0,
                             "roe_pct": 10.0, "profit_growth_pct": 20.0, "revenue_growth_pct": 15.0}],
            }],
            "errors": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "stock.sqlite3"
            backup_path = tmp / "backup.json"
            pool_path = tmp / "pool.json"
            conn = ss.connect(db_path)
            ss.save_sw3_membership(conn, bad_cache)
            conn.close()
            backup_rows = [
                {"segment_code": f"85011{i}.SI", "segment_name": f"备份{i}", "parent_segment": "父", "member_count": 1}
                for i in range(1, 5)
            ]
            backup_path.write_text(json.dumps(backup_rows, ensure_ascii=False), encoding="utf-8")

            with patch.object(r, "STOCK_DB_FILE", db_path), \
                 patch.object(r, "META_BACKUP_FILE", backup_path), \
                 patch.object(r, "SEGMENT_LEADER_POOL_FILE", pool_path), \
                 patch.object(r, "DATA_DIR", tmp), \
                 patch.object(r, "crawl_sw3_membership", return_value=rebuilt) as recrawl_mock, \
                 patch.object(r, "refresh_oldest_segments") as refresh_mock:
                payload = r.build_segment_leader_pool(top_per_segment=1, refresh_slice=15)

        recrawl_mock.assert_called_once_with(resume=False, force_official_members=True)
        refresh_mock.assert_not_called()
        self.assertEqual(payload["segments"][0]["segment_code"], "850111.SI")

    def test_first_membership_build_uses_official_members(self):
        from pathlib import Path
        import stock_crawl_segment_leaders as r

        rebuilt = {
            "schema": r.SW3_MEMBERSHIP_SCHEMA,
            "generated_at": "2026-06-20 01:00:00",
            "segments": [{
                "segment_code": "850111.SI", "segment_name": "种子", "parent_segment": "种植业", "member_count": 1,
                "members": [{"code": "000998", "name": "隆平高科", "price": 10.0, "market_cap_yi": 200.0,
                             "roe_pct": 10.0, "profit_growth_pct": 20.0, "revenue_growth_pct": 15.0}],
            }],
            "errors": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pool_path = tmp / "pool.json"
            with patch.object(r, "SEGMENT_LEADER_POOL_FILE", pool_path), \
                 patch.object(r, "DATA_DIR", tmp), \
                 patch.object(r, "STOCK_DB_FILE", tmp / "stock.sqlite3"), \
                 patch.object(r, "membership_db_needs_full_recrawl", return_value=(False, 0, 0)), \
                 patch.object(r, "load_sw3_membership", return_value=None), \
                 patch.object(r, "crawl_sw3_membership", return_value=rebuilt) as crawl_mock:
                payload = r.build_segment_leader_pool(top_per_segment=1, refresh_slice=15)

        crawl_mock.assert_called_once_with(force_official_members=True)
        self.assertEqual(payload["segments"][0]["segment_code"], "850111.SI")

    def test_refresh_fallback_uses_backup_when_db_segments_below_threshold(self):
        import json
        from pathlib import Path
        import pandas as pd
        import stock_crawl_segment_leaders as r
        import stock_storage as ss

        bad_cache = {
            "schema": r.SW3_MEMBERSHIP_SCHEMA,
            "segments": [
                {"segment_code": "A", "segment_name": "赛道A", "parent_segment": "父", "member_count": 1,
                 "members": [{"code": "600001", "name": "旧A", "price": 10.0, "market_cap_yi": 100.0}]},
                {"segment_code": "B", "segment_name": "赛道B", "parent_segment": "父", "member_count": 1,
                 "members": [{"code": "600002", "name": "旧B", "price": 10.0, "market_cap_yi": 100.0}]},
            ],
            "errors": [],
        }
        fetched = []

        def fake_official_members(code, *a, **k):
            fetched.append(code)
            num = str(600100 + len(fetched)).zfill(6)
            return pd.DataFrame([{"证券代码": num, "证券名称": f"新抓{len(fetched)}", "市值占比": 100.0}])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "stock.sqlite3"
            backup_path = tmp / "backup.json"
            conn = ss.connect(db_path)
            ss.save_sw3_membership(conn, bad_cache)
            conn.close()
            backup_rows = [
                {"segment_code": f"85011{i}.SI", "segment_name": f"备份{i}", "parent_segment": "父", "member_count": 1}
                for i in range(1, 5)
            ]
            backup_path.write_text(json.dumps(backup_rows, ensure_ascii=False), encoding="utf-8")

            with patch.object(r, "STOCK_DB_FILE", db_path), \
                 patch.object(r, "META_BACKUP_FILE", backup_path), \
                 patch.object(r, "fetch_sw3_segments", side_effect=RuntimeError("taxonomy failed")), \
                 patch.object(r, "fetch_sw3_segment_constituents_official", fake_official_members), \
                 patch.object(r, "load_latest_member_metrics", return_value={}), \
                 patch.object(r, "export_segment_backup", return_value=None):
                r.refresh_oldest_segments(n=2, sleep_sec=0)

        self.assertEqual(fetched, ["850111.SI", "850112.SI"])

    def test_refresh_oldest_segments_picks_missing_and_stalest(self):
        import json
        from pathlib import Path
        import pandas as pd
        import stock_crawl_segment_leaders as r

        # 本地总表 A/B/C：A 最旧、B 最新、C 在总表但无成分(缺失待补)。slice 用本地总表，不再远程拉。
        cache = {
            "schema": r.SW3_MEMBERSHIP_SCHEMA, "generated_at": "2026-01-01 00:00:00",
            "segments": [
                {"segment_code": "A", "segment_name": "赛道A", "parent_segment": "父", "member_count": 5,
                 "members": [{"code": "600000", "name": "旧A", "price": 10.0, "market_cap_yi": 100.0}],
                 "refreshed_at": "2026-01-01 00:00:00"},
                {"segment_code": "B", "segment_name": "赛道B", "parent_segment": "父", "member_count": 5,
                 "members": [{"code": "600001", "name": "新B", "price": 10.0, "market_cap_yi": 100.0}],
                 "refreshed_at": "2026-06-17 00:00:00"},
                {"segment_code": "C", "segment_name": "赛道C", "parent_segment": "父", "member_count": 5,
                 "members": [], "refreshed_at": "2026-01-01 00:00:00"},
            ], "errors": []}

        fetched = []

        def fake_official_members(code, *a, **k):
            fetched.append(code)
            # 不同赛道返回不同 code（1股=1三级行业，code 全表唯一）
            num = "600002" if code == "A" else "600003"
            return pd.DataFrame([{"证券代码": num, "证券名称": "新抓", "最新权重": 100.0}])

        import stock_storage as ss
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "stock.sqlite3"
            conn = ss.connect(db_path)
            ss.save_sw3_membership(conn, cache)
            conn.close()
            with patch.object(r, "STOCK_DB_FILE", db_path), \
                 patch.object(r, "META_BACKUP_FILE", Path(tmpdir) / "backup.json"), \
                 patch.object(r, "fetch_sw3_segments", side_effect=RuntimeError("taxonomy failed")), \
                 patch.object(r, "fetch_sw3_segment_constituents") as legu_members, \
                 patch.object(r, "fetch_sw3_segment_constituents_official", fake_official_members), \
                 patch.object(r, "load_latest_member_metrics", return_value={}):
                payload = r.refresh_oldest_segments(n=2, sleep_sec=0)

        legu_members.assert_not_called()
        # 只刷新缺失的 C 和最旧的 A，最新的 B 不动
        self.assertEqual(set(fetched), {"A", "C"})
        self.assertNotIn("B", fetched)
        segs = {s["segment_code"]: s for s in payload["segments"]}
        self.assertEqual(segs["B"]["refreshed_at"], "2026-06-17 00:00:00")  # B 原样保留
        self.assertEqual(segs["A"]["members"][0]["code"], "600002")          # A 被刷新
        self.assertIn("C", segs)                                             # C 被补上

    def test_refresh_prioritizes_failed_and_maintains_error_list(self):
        import json
        from pathlib import Path
        import pandas as pd
        import stock_crawl_segment_leaders as r

        sw3 = pd.DataFrame([
            {"行业代码": "A", "行业名称": "赛道A", "上级行业": "父", "成份个数": 5},
            {"行业代码": "B", "行业名称": "赛道B", "上级行业": "父", "成份个数": 5},
            {"行业代码": "C", "行业名称": "赛道C", "上级行业": "父", "成份个数": 5},
        ])
        # A 虽是最新(refreshed_at)，但在 errors 名单里 -> 应优先刷；B 较旧但不在名单
        cache = {
            "schema": r.SW3_MEMBERSHIP_SCHEMA, "generated_at": "2026-06-17 00:00:00",
            "segments": [
                {"segment_code": "A", "segment_name": "赛道A", "parent_segment": "父", "member_count": 5,
                 "members": [{"code": "600000", "name": "A", "price": 10.0, "market_cap_yi": 100.0}],
                 "refreshed_at": "2026-06-17 00:00:00"},
                {"segment_code": "B", "segment_name": "赛道B", "parent_segment": "父", "member_count": 5,
                 "members": [{"code": "600001", "name": "B", "price": 10.0, "market_cap_yi": 100.0}],
                 "refreshed_at": "2026-06-10 00:00:00"},
                {"segment_code": "C", "segment_name": "赛道C", "parent_segment": "父", "member_count": 5,
                 "members": [{"code": "600002", "name": "C", "price": 10.0, "market_cap_yi": 100.0}],
                 "refreshed_at": "2026-06-15 00:00:00"},
            ],
            "errors": [{"segment_code": "A", "segment_name": "赛道A", "error": "空表(疑似限流)"}]}

        legu_fetched = []
        official_fetched = []

        def fake_legu_members(code, *a, **k):
            legu_fetched.append(code)
            if code == "B":
                raise RuntimeError("legulegu member failed")
            return pd.DataFrame([{"股票代码": "600009", "股票简称": "新", "价格": 20.0, "市值（亿元）": 200.0}])

        def fake_official_members(code, *a, **k):
            official_fetched.append(code)
            return pd.DataFrame()

        import stock_storage as ss
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "stock.sqlite3"
            conn = ss.connect(db_path)
            ss.save_sw3_membership(conn, cache)
            conn.close()
            with patch.object(r, "STOCK_DB_FILE", db_path), \
                 patch.object(r, "META_BACKUP_FILE", Path(tmpdir) / "backup.json"), \
                 patch.object(r, "fetch_sw3_segments", lambda *a, **k: sw3), \
                 patch.object(r, "fetch_sw3_segment_constituents", fake_legu_members), \
                 patch.object(r, "fetch_sw3_segment_constituents_official", fake_official_members), \
                 patch.object(r, "load_latest_member_metrics", return_value={}), \
                 patch.object(r.time, "sleep", return_value=None):
                payload = r.refresh_oldest_segments(n=2, sleep_sec=0)

        # 失败的 A 即使最新也优先被刷；n=2 另一个取最旧的 B；C(较新)不在本轮
        self.assertIn("A", legu_fetched)
        self.assertIn("B", legu_fetched)
        self.assertEqual(official_fetched, ["B"])
        self.assertNotIn("C", legu_fetched)
        err_codes = {e["segment_code"] for e in payload["errors"]}
        self.assertNotIn("A", err_codes)   # A 刷成功 -> 移出失败名单
        self.assertIn("B", err_codes)      # B 本轮抓空 -> 加入失败名单

    def test_recrawl_prunes_segments_gone_from_taxonomy(self):
        # slice 不再剔除(总表本地优先、不远程对比)；总表增删由 recrawl(refresh_taxonomy) 统一处理。
        from pathlib import Path
        import pandas as pd
        import stock_crawl_segment_leaders as r

        # 联网重刷后总表只剩 A/B；缓存里有 A/B/C，C 已从申万删除 -> recrawl 应剔除
        sw3 = pd.DataFrame([
            {"行业代码": "A", "行业名称": "赛道A", "上级行业": "父", "成份个数": 5},
            {"行业代码": "B", "行业名称": "赛道B", "上级行业": "父", "成份个数": 5},
        ])

        def seg(code, refreshed):
            return {"segment_code": code, "segment_name": "赛道" + code, "parent_segment": "父",
                    "member_count": 5, "refreshed_at": refreshed,
                    "members": [{"code": "60000" + code[-1], "name": code, "price": 10.0, "market_cap_yi": 100.0}]}

        cache = {"schema": r.SW3_MEMBERSHIP_SCHEMA, "generated_at": "2026-06-17 00:00:00",
                 "segments": [seg("A", "2026-06-17 00:00:00"), seg("B", "2026-06-16 00:00:00"),
                              seg("C", "2026-06-15 00:00:00")],
                 "errors": []}

        import stock_storage as ss
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "stock.sqlite3"
            conn = ss.connect(db_path)
            ss.save_sw3_membership(conn, cache)
            conn.close()
            with patch.object(r, "STOCK_DB_FILE", db_path), \
                 patch.object(r, "META_BACKUP_FILE", Path(tmpdir) / "backup.json"), \
                 patch.object(r, "fetch_sw3_segments", lambda *a, **k: sw3):
                # recrawl: 重拉总表(A/B) + resume 复用 A/B 成分；C 不在新总表 -> 落库时被剔除
                payload = r.crawl_sw3_membership(sleep_sec=0, resume=True)

        codes = {s["segment_code"] for s in payload["segments"]}
        self.assertEqual(codes, {"A", "B"})  # C 已从总表删除 -> recrawl 剔除

    def test_taxonomy_fallback_skips_legulegu_members(self):
        import pandas as pd
        import stock_crawl_segment_leaders as r

        sw3_backup = pd.DataFrame([
            {"行业代码": "A", "行业名称": "赛道A", "上级行业": "父", "成份个数": 1},
            {"行业代码": "B", "行业名称": "赛道B", "上级行业": "父", "成份个数": 1},
        ])
        legu_calls = []
        official_calls = []

        def fake_legu_members(code, *a, **k):
            legu_calls.append(code)
            if code == "A":
                return pd.DataFrame([{
                    "股票代码": "600010", "股票简称": "包钢股份",
                    "价格": 10.0, "市值（亿元）": 100.0,
                }])
            raise RuntimeError("legulegu member failed")

        def fake_official_members(code, *a, **k):
            official_calls.append(code)
            stock_code = "600011" if code == "A" else "600012"
            return pd.DataFrame([{
                "证券代码": stock_code, "证券名称": "华能国际", "最新权重": 100.0,
            }])

        def fake_retry(func, *args, **kwargs):
            if func is r.fetch_sw3_segments:
                raise RuntimeError("taxonomy failed")
            return func(*args)

        with patch.object(r, "_load_sw3_membership_from_db", return_value={}), \
             patch.object(r, "_segments_df_from_backup", return_value=sw3_backup), \
             patch.object(r, "membership_db_needs_full_recrawl", return_value=(False, 0, 0)), \
             patch.object(r, "_retry_fetch", side_effect=fake_retry), \
             patch.object(r, "fetch_sw3_segment_constituents", side_effect=fake_legu_members), \
             patch.object(r, "fetch_sw3_segment_constituents_official", side_effect=fake_official_members), \
             patch.object(r, "load_latest_member_metrics", return_value={}), \
             patch.object(r, "_save_sw3_membership_to_db", return_value=None), \
             patch.object(r, "export_segment_backup", return_value=None):
            payload = r.crawl_sw3_membership(sleep_sec=0, resume=False)

        self.assertEqual(legu_calls, [])
        self.assertEqual(official_calls, ["A", "B"])
        segs = {s["segment_code"]: s for s in payload["segments"]}
        self.assertEqual(segs["A"]["members"][0]["code"], "600011")
        self.assertEqual(segs["B"]["members"][0]["code"], "600012")

    def test_first_membership_crawl_skips_legulegu_members(self):
        import pandas as pd
        import stock_crawl_segment_leaders as r

        sw3 = pd.DataFrame([
            {"行业代码": "A", "行业名称": "赛道A", "上级行业": "父", "成份个数": 1},
            {"行业代码": "B", "行业名称": "赛道B", "上级行业": "父", "成份个数": 1},
        ])
        official_calls = []

        def fake_official_members(code, *a, **k):
            official_calls.append(code)
            stock_code = "600011" if code == "A" else "600012"
            return pd.DataFrame([{
                "证券代码": stock_code, "证券名称": "华能国际", "市值占比": 100.0,
            }])

        with patch.object(r, "_load_sw3_membership_from_db", return_value={}), \
             patch.object(r, "membership_db_needs_full_recrawl", return_value=(False, 0, 0)), \
             patch.object(r, "fetch_sw3_segments", return_value=sw3), \
             patch.object(r, "fetch_sw3_segment_constituents") as legu_members, \
             patch.object(r, "fetch_sw3_segment_constituents_official", side_effect=fake_official_members), \
             patch.object(r, "load_latest_member_metrics", return_value={}), \
             patch.object(r, "_save_sw3_membership_to_db", return_value=None), \
             patch.object(r, "export_segment_backup", return_value=None):
            payload = r.crawl_sw3_membership(sleep_sec=0, resume=False)

        legu_members.assert_not_called()
        self.assertEqual(official_calls, ["A", "B"])
        self.assertEqual({s["segment_code"] for s in payload["segments"]}, {"A", "B"})


class FundamentalsRefreshDecisionTest(unittest.TestCase):
    def test_recent_report_periods(self):
        from datetime import date
        import stock_crawl_fundamentals as f
        # 2026-06-19：最近已过披露截止日的报告期 = Q1 2026(0331, 截止0430)
        self.assertEqual(f._recent_report_periods(date(2026, 6, 19), count=1), ["20260331"])
        # 2026-09-15：半年报(0630, 截止0831)已过
        self.assertEqual(f._recent_report_periods(date(2026, 9, 15), count=1), ["20260630"])
        # 取最近两期
        self.assertEqual(f._recent_report_periods(date(2026, 6, 19), count=2), ["20260331", "20251231"])

    def test_needs_fundamentals_refresh(self):
        from datetime import date
        import stock_crawl_fundamentals as f
        today = date(2026, 6, 19)
        # 从没爬过 → 刷
        self.assertTrue(f.needs_fundamentals_refresh(None, None, today=today))
        # yjbb 出新报告(公告日 > 上次刷新) → 刷
        self.assertTrue(f.needs_fundamentals_refresh("2026-04-10T10:00:00", "2026-04-25", today=today))
        # 已有该报告(公告 <= 上次刷新) 且未超期 → 跳过
        self.assertFalse(f.needs_fundamentals_refresh("2026-05-01T10:00:00", "2026-04-25", today=today))
        # 无新报告但超 90 天 → 兜底刷
        self.assertTrue(f.needs_fundamentals_refresh("2026-01-01T10:00:00", None, expire_days=90, today=today))
        # 无新报告且未超期 → 跳过
        self.assertFalse(f.needs_fundamentals_refresh("2026-05-20T10:00:00", None, today=today))

    def test_full_mode_uses_segment_leader_universe_without_default_limit(self):
        import stock_crawl_fundamentals as f

        stocks = {"600000": {"name": "浦发银行", "candidate_for": ["银行"]}}
        with patch.object(sys, "argv", [
                "stock_crawl_fundamentals.py", "--mode", "full", "--workers", "3"
             ]), \
             patch.object(f, "strip_proxy_env", return_value=None), \
             patch.object(f, "get_segment_leader_universe", return_value=stocks) as universe_mock, \
             patch.object(f, "fetch_pledge_data_bulk", return_value={}), \
             patch.object(f, "crawl_stocks", return_value=None) as crawl_mock:
            f.main()

        universe_mock.assert_called_once_with(refresh_slice=f.SEGMENT_REFRESH_SLICE)
        crawl_mock.assert_called_once_with(stocks, {}, limit=0, workers=3)

    def test_segment_leader_universe_reuses_generated_pool_when_refresh_slice_zero(self):
        import stock_crawl_fundamentals as f
        import stock_crawl_segment_leaders as leaders

        payload = {
            "segments": [{
                "segment_name": "银行",
                "leaders": [{"code": "600000", "name": "浦发银行"}],
            }],
        }
        with patch.object(leaders, "load_segment_leader_pool", return_value=payload) as load_mock, \
             patch.object(leaders, "build_segment_leader_pool") as build_mock:
            stocks = f.get_segment_leader_universe(refresh_slice=0)

        load_mock.assert_called_once()
        build_mock.assert_not_called()
        self.assertEqual(stocks["600000"]["candidate_for"], ["银行"])

    def test_load_existing_requires_fresh_complete_daily_backfill(self):
        from datetime import date, timedelta
        import stock_storage as ss
        import stock_crawl_fundamentals as f

        def stock_payload(code, records, **extra):
            payload = {
                "symbol": code,
                "name": code,
                "financials": {
                    "income": [{"date": "2025-12-31", "revenue": 1}],
                    "balance": [{"date": "2025-12-31", "total_equity": 1}],
                    "cashflow": [{"date": "2025-12-31", "operating_cashflow_net": 1}],
                },
                "indicators": {"records": [{"date": "2025-12-31", "roe": 1}]},
                "dividends": {"records": []},
                "history": history_payload_from_records(
                    code, code, records, "test.stock_crawl_fundamentals"
                ),
            }
            payload.update(extra)
            return payload

        def make_records(start, count):
            records = []
            for i in range(count):
                day = start + timedelta(days=i)
                records.append({
                    "date": day.strftime("%Y-%m-%d"),
                    "daily_open": 10 + i,
                    "daily_high": 11 + i,
                    "daily_low": 9 + i,
                    "daily_close": 10.5 + i,
                    "daily_volume": 1000 + i,
                    "daily_amount": 10000 + i,
                })
            return records

        short_records = make_records(date(2026, 1, 1), 41)
        complete_records = make_records(date(2025, 7, 15), f.MIN_COMPLETE_DAILY_ROWS)

        conn = ss.connect(":memory:")
        try:
            ss.save_stock(conn, stock_payload("600001", [
                {"date": "2026-01-01", "market_cap": 1000}
            ]))
            ss.save_stock(conn, stock_payload("600002", short_records))
            ss.save_stock(conn, stock_payload(
                "600003", short_records, history_refetched_at="2026-01-01T10:00:00"
            ))
            ss.save_stock(conn, stock_payload("600004", complete_records))

            with patch.object(f.ss, "thread_conn", return_value=conn), \
                 patch.object(f, "latest_weekday_date", return_value="2026-02-13"):
                existing = f.load_existing()
        finally:
            conn.close()

        self.assertNotIn("600001", existing)
        self.assertNotIn("600002", existing)
        self.assertNotIn("600003", existing)
        self.assertIn("600004", existing)

    def test_save_stock_preserves_history_valuation_and_existing_meta(self):
        import stock_storage as ss
        import stock_crawl_fundamentals as f

        conn = ss.connect(":memory:")
        try:
            existing_records = [{
                "date": "2026-01-02",
                "daily_open": 10.0,
                "daily_high": 11.0,
                "daily_low": 9.0,
                "daily_close": 10.5,
                "daily_volume": 1000,
                "daily_amount": 10000,
                "market_cap": 1234.0,
                "pe_ttm": 12.0,
            }]
            ss.save_stock(conn, {
                "symbol": "600010",
                "name": "旧公司",
                "daily_refetched_at": "2026-01-02T09:00:00",
                "history_refetched_at": "2026-01-02T09:30:00",
                "financials": {"old": True},
                "indicators": {"old": True},
                "dividends": {"old": True},
                "pledge": {
                    "pledge_ratio": 1.2,
                    "pledge_count": 3,
                    "trade_date": "2026-01-02",
                    "industry": "旧行业",
                },
                "history": history_payload_from_records(
                    "600010", "旧公司", existing_records, "test.existing"
                ),
            })

            incoming_records = [{
                "date": "2026-01-02",
                "daily_open": 10.1,
                "daily_high": 11.1,
                "daily_low": 9.1,
                "daily_close": 10.6,
                "daily_volume": 1100,
                "daily_amount": 11000,
            }]
            with patch.object(f.ss, "thread_conn", return_value=conn):
                f.save_stock("600010", {
                    "symbol": "600010",
                    "name": "新公司",
                    "financials": {"new": True},
                    "indicators": {"new": True},
                    "dividends": {"new": True},
                    "pledge": {
                        "pledge_ratio": None,
                        "pledge_count": None,
                        "trade_date": None,
                        "industry": None,
                    },
                    "history": history_payload_from_records(
                        "600010", "新公司", incoming_records, "test.incoming"
                    ),
                })
            saved = ss.load_stock(conn, "600010")
        finally:
            conn.close()

        self.assertEqual(saved["daily_refetched_at"], "2026-01-02T09:00:00")
        self.assertEqual(saved["history_refetched_at"], "2026-01-02T09:30:00")
        self.assertEqual(saved["pledge"]["pledge_ratio"], 1.2)
        self.assertEqual(saved["pledge"]["industry"], "旧行业")
        self.assertEqual(saved["history"]["records"][0]["daily_close"], 10.6)
        self.assertEqual(saved["history"]["records"][0]["market_cap"], 1234.0)
        self.assertEqual(saved["history"]["records"][0]["pe_ttm"], 12.0)


if __name__ == "__main__":
    unittest.main()
