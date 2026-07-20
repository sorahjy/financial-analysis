import io
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import plate_crawl_history
import plate_storage
import stock_crawl_fundamentals
import stock_crawl_common
import stock_crawl_price_valuation
import stock_crawl_segment_leaders
import stock_data_refresh
import stock_storage
from stock_crawl_common import history_payload_from_records, latest_weekday_date


def sw2_row(code, name, trade_date, close=100.0):
    return {
        "swindexcode": code,
        "swindexname": name,
        "bargaindate": trade_date,
        "closeindex": close,
        "bargainamount": 1.0,
    }


def complete_daily_rows(start, count):
    return [
        {
            "date": (start + timedelta(days=i)).strftime("%Y-%m-%d"),
            "daily_open": 10 + i,
            "daily_high": 11 + i,
            "daily_low": 9 + i,
            "daily_close": 10.5 + i,
            "daily_volume": 1000 + i,
            "daily_amount": 10000 + i,
        }
        for i in range(count)
    ]


def valid_fundamentals():
    return {
        "financials": {
            "income": [{"date": "2025-12-31", "revenue": 1}],
            "balance": [{"date": "2025-12-31", "total_equity": 1}],
            "cashflow": [{"date": "2025-12-31", "operating_cashflow_net": 1}],
        },
        "indicators": {"records": [{"date": "2025-12-31", "roe": 10}]},
        "dividends": {"records": []},
    }


class Sw3MembershipIntegrityTest(unittest.TestCase):
    def test_taxonomy_truncation_guard_ignores_tiny_fixture_but_protects_real_table(self):
        self.assertFalse(
            stock_crawl_segment_leaders._taxonomy_response_looks_truncated(2, 3)
        )
        self.assertTrue(
            stock_crawl_segment_leaders._taxonomy_response_looks_truncated(80, 100)
        )
        self.assertFalse(
            stock_crawl_segment_leaders._taxonomy_response_looks_truncated(81, 100)
        )

    def test_refresh_preserves_flags_and_signals_for_retained_members(self):
        conn = stock_storage.connect(":memory:")
        try:
            stock_storage.save_sw3_membership(conn, {
                "segments": [{
                    "segment_code": "850111",
                    "segment_name": "旧行业名",
                    "members": [
                        {
                            "code": "000001", "name": "保留成员",
                            "price": 10, "market_cap_yi": 100,
                        },
                        {"code": "000002", "name": "退出成员", "price": 20},
                    ],
                }],
                "errors": [],
            })
            conn.execute(
                "UPDATE sw3_member SET is_leader=1, is_hot_money=1 WHERE code='000001'"
            )
            conn.commit()
            stock_storage.replace_short_signals(
                conn,
                {
                    "000001": {"signals": {"lhb": {"count": 1}}},
                    "000002": {"signals": {"lhb": {"count": 2}}},
                },
                generated_at="2026-07-10 10:00:00",
                as_of_date="2026-07-10",
            )

            stock_storage.save_sw3_membership(conn, {
                "segments": [{
                    "segment_code": "850111",
                    "segment_name": "新行业名",
                    "members": [
                        {"code": "000001", "name": "保留成员", "price": 11},
                        {"code": "000003", "name": "新增成员", "price": 30},
                    ],
                }],
                "errors": [],
            })

            retained = conn.execute(
                "SELECT price, market_cap_yi, is_leader, is_hot_money "
                "FROM sw3_member WHERE code='000001'"
            ).fetchone()
            self.assertEqual(
                (
                    retained["price"], retained["market_cap_yi"],
                    retained["is_leader"], retained["is_hot_money"],
                ),
                (11.0, 100.0, 1, 1),
            )
            self.assertEqual(set(stock_storage.load_short_signals(conn)), {"000001"})
            self.assertIsNone(
                conn.execute("SELECT 1 FROM sw3_member WHERE code='000002'").fetchone()
            )
            new_member = conn.execute(
                "SELECT is_leader, is_hot_money FROM sw3_member WHERE code='000003'"
            ).fetchone()
            self.assertEqual(tuple(new_member), (0, 0))
        finally:
            conn.close()

    def test_failed_segment_and_empty_payload_preserve_existing_members_and_signals(self):
        conn = stock_storage.connect(":memory:")
        try:
            stock_storage.save_sw3_membership(conn, {
                "segments": [{
                    "segment_code": "850111",
                    "segment_name": "测试行业",
                    "parent_segment": "测试父行业",
                    "member_count": 1,
                    "refreshed_at": "2026-07-09 10:00:00",
                    "members": [{"code": "000001", "name": "成员"}],
                }],
                "errors": [],
            })
            conn.execute(
                "UPDATE sw3_member SET is_leader=1, is_hot_money=1 WHERE code='000001'"
            )
            conn.commit()
            stock_storage.replace_short_signals(
                conn,
                {"000001": {"signals": {"lhb": {"count": 1}}}},
                generated_at="2026-07-10 10:00:00",
                as_of_date="2026-07-10",
            )

            stock_storage.save_sw3_membership(conn, {
                "segments": [],
                "errors": [{
                    "segment_code": "850111",
                    "error": "upstream unavailable",
                }],
            })
            segment = conn.execute(
                "SELECT segment_name, parent_segment, member_count, refreshed_at, error "
                "FROM sw3_segment WHERE segment_code='850111'"
            ).fetchone()
            self.assertEqual(
                tuple(segment),
                ("测试行业", "测试父行业", 1, "2026-07-09 10:00:00", "upstream unavailable"),
            )
            member = conn.execute(
                "SELECT is_leader, is_hot_money FROM sw3_member WHERE code='000001'"
            ).fetchone()
            self.assertEqual(tuple(member), (1, 1))
            self.assertEqual(set(stock_storage.load_short_signals(conn)), {"000001"})

            with self.assertRaisesRegex(ValueError, "empty payload"):
                stock_storage.save_sw3_membership(conn, {"segments": [], "errors": []})
            self.assertIsNotNone(
                conn.execute("SELECT 1 FROM sw3_member WHERE code='000001'").fetchone()
            )
            self.assertEqual(set(stock_storage.load_short_signals(conn)), {"000001"})
        finally:
            conn.close()

    def test_all_failed_segments_are_protected_without_error_truncation(self):
        conn = stock_storage.connect(":memory:")
        try:
            segments = [
                {
                    "segment_code": f"85{idx:04d}",
                    "segment_name": f"行业{idx}",
                    "members": [{"code": f"{idx:06d}", "name": f"成员{idx}"}],
                }
                for idx in range(100)
            ]
            stock_storage.save_sw3_membership(conn, {"segments": segments, "errors": []})
            errors = [
                {
                    "segment_code": segment["segment_code"],
                    "segment_name": segment["segment_name"],
                    "error": "upstream unavailable",
                }
                for segment in segments
            ]

            stock_storage.save_sw3_membership(conn, {"segments": [], "errors": errors})

            self.assertEqual(stock_storage.table_count(conn, "sw3_member"), 100)
            self.assertEqual(stock_storage.table_count(conn, "sw3_segment"), 100)
        finally:
            conn.close()


class StockListingDateRefreshTest(unittest.TestCase):
    @staticmethod
    def _daily_row(trade_date, close=10.0):
        return {
            "date": trade_date,
            "daily_open": close,
            "daily_high": close + 0.5,
            "daily_low": close - 0.5,
            "daily_close": close,
            "daily_volume": 1000.0,
            "daily_amount": 1000000.0,
            "daily_change_pct": 0.0,
            "daily_turnover_rate": 1.0,
        }

    def test_exchange_lists_are_fetched_once_per_needed_market(self):
        calls = []

        def frame_fetcher(market, rows):
            def fetch():
                calls.append(market)
                return pd.DataFrame(rows)
            return fetch

        resolved, errors = stock_crawl_price_valuation.fetch_stock_listing_dates(
            ["600001", "600002", "688001", "000001", "300001"],
            fetchers={
                "sh_main": frame_fetcher("sh_main", [
                    {"证券代码": "600001", "证券简称": "沪一", "上市日期": date(1991, 1, 1)},
                    {"证券代码": "600002", "证券简称": "沪二", "上市日期": "1992-02-02"},
                ]),
                "sh_star": frame_fetcher("sh_star", [
                    {"证券代码": "688001", "证券简称": "科创", "上市日期": "20190722"},
                ]),
                "sz": frame_fetcher("sz", [
                    {"A股代码": 1.0, "A股简称": "深一", "A股上市日期": "19910403"},
                    {"A股代码": "300001", "A股简称": "创业", "A股上市日期": "2009/10/30"},
                ]),
            },
        )

        self.assertEqual(calls, ["sh_main", "sh_star", "sz"])
        self.assertEqual(errors, {})
        self.assertEqual(resolved["600001"]["listing_date"], "1991-01-01")
        self.assertEqual(resolved["688001"]["listing_date"], "2019-07-22")
        self.assertEqual(resolved["000001"]["listing_date"], "1991-04-03")
        self.assertEqual(resolved["300001"]["listing_date"], "2009-10-30")

    def test_resolver_fetches_only_missing_dates_then_uses_cache(self):
        conn = stock_storage.connect(":memory:")
        calls = []
        try:
            stock_storage.upsert_listing_dates(conn, {
                "000001": {"name": "平安银行", "listing_date": "1991-04-03"},
            })

            def fetcher(codes):
                calls.append(list(codes))
                return ({
                    "603629": {"name": "利通电子", "listing_date": "2018-12-24"},
                }, {})

            stocks = {"000001": "平安银行", "603629": "利通电子"}
            dates, errors = stock_crawl_price_valuation.resolve_stock_listing_dates(
                stocks, fetcher=fetcher, conn=conn, label="测试池"
            )
            self.assertEqual(calls, [["603629"]])
            self.assertEqual(errors, {})
            self.assertEqual(dates["603629"], "2018-12-24")

            def should_not_fetch(_codes):
                raise AssertionError("cached listing dates must not be fetched again")

            cached, errors = stock_crawl_price_valuation.resolve_stock_listing_dates(
                stocks, fetcher=should_not_fetch, conn=conn, label="测试池"
            )
            self.assertEqual(errors, {})
            self.assertEqual(cached, {
                "000001": "1991-04-03",
                "603629": "2018-12-24",
            })
        finally:
            conn.close()

    def test_missing_listing_date_is_reported_and_stock_history_is_not_requested(self):
        processed = []

        def fake_process(code, *_args, **_kwargs):
            processed.append(code)
            return {"status": "queued"}

        with patch.object(
            stock_crawl_price_valuation,
            "resolve_stock_listing_dates",
            return_value=(
                {"000001": "1991-04-03", "603629": None},
                {"603629": "上交所主板A股清单未返回该股票"},
            ),
        ), patch.object(
            stock_crawl_price_valuation,
            "process_stock",
            side_effect=fake_process,
        ), patch.object(
            stock_crawl_price_valuation.ss,
            "sync_sw3_member_market_caps",
            return_value=0,
        ):
            result = stock_crawl_price_valuation.refresh_stock_histories(
                {"000001": "平安银行", "603629": "利通电子"}, workers=1
            )

        self.assertEqual(processed, ["000001"])
        self.assertEqual(result["failed"], ["603629"])
        self.assertEqual(result["failure_details"][0]["stage"], "listing_date")
        self.assertIn("清单未返回", result["failure_details"][0]["error"])

    def test_etf_refresh_does_not_resolve_stock_listing_dates(self):
        with patch.object(
            stock_crawl_price_valuation,
            "resolve_stock_listing_dates",
        ) as resolver, patch.object(
            stock_crawl_price_valuation,
            "process_stock",
            return_value={"status": "queued"},
        ) as process_mock:
            result = stock_crawl_price_valuation.refresh_stock_histories(
                {"510300": "沪深300ETF"},
                workers=1,
                instrument_type="etf",
                refresh_valuation=False,
            )

        resolver.assert_not_called()
        self.assertEqual(process_mock.call_args.kwargs["listing_date"], None)
        self.assertEqual(result["failed"], [])

    def test_listed_stock_with_current_history_does_not_fetch_pre_listing_range(self):
        listing_date = "2018-12-24"
        record = self._daily_row(listing_date)
        with patch.object(
            stock_crawl_price_valuation,
            "load_stock_file",
            return_value={
                "records": [record],
                "start_date": listing_date,
                "end_date": listing_date,
                "history_refetched_at": "2018-12-24T15:11:00",
            },
        ), patch.object(
            stock_crawl_price_valuation,
            "latest_weekday_date",
            return_value=listing_date,
        ), patch.object(
            stock_crawl_price_valuation,
            "fetch_daily_range",
        ) as daily_fetch, patch.object(
            stock_crawl_price_valuation.ss,
            "thread_conn",
            return_value=None,
        ), patch.object(
            stock_crawl_price_valuation.ss,
            "update_stock_identity",
            return_value=True,
        ):
            outcome = stock_crawl_price_valuation.process_stock(
                "603629",
                "利通电子",
                1,
                1,
                max_years=10,
                refresh_valuation=False,
                listing_date=listing_date,
            )

        daily_fetch.assert_not_called()
        self.assertEqual(outcome["status"], "up_to_date")

    def test_first_history_request_starts_at_listing_date(self):
        calls = []
        listing_date = "2018-12-24"

        def daily_fetcher(_code, start_date, end_date):
            calls.append((start_date, end_date))
            return [{
                "date": listing_date,
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": 10.0,
                "volume": 1000.0,
                "amount": 1000000.0,
                "change_pct": 0.0,
                "turnover_rate": 1.0,
            }]

        saved = []
        with patch.object(
            stock_crawl_price_valuation,
            "load_stock_file",
            return_value={},
        ), patch.object(
            stock_crawl_price_valuation,
            "latest_weekday_date",
            return_value="2026-07-17",
        ):
            stock_crawl_price_valuation.process_stock(
                "603629",
                "利通电子",
                1,
                1,
                max_years=10,
                refresh_valuation=False,
                listing_date=listing_date,
                daily_fetcher=daily_fetcher,
                save_callback=lambda _code, _name, data: saved.append(data),
            )

        self.assertEqual(calls, [(listing_date, "2026-07-17")])
        self.assertEqual(saved[0]["listing_date"], listing_date)

    def test_embedded_fundamentals_receive_resolved_listing_date(self):
        listing_date = "2025-08-08"
        fundamentals = {
            **valid_fundamentals(),
            "financials_refetched_at": "2026-07-19T10:00:00",
        }
        saved = []
        with patch.object(
            stock_crawl_price_valuation,
            "load_stock_file",
            return_value={},
        ), patch.object(
            stock_crawl_price_valuation,
            "latest_weekday_date",
            return_value=listing_date,
        ), patch.object(
            stock_crawl_fundamentals,
            "fetch_fundamentals",
            return_value=fundamentals,
        ) as fetch_fundamentals:
            stock_crawl_price_valuation.process_stock(
                "301666",
                "大普微",
                1,
                1,
                need_fundamentals=True,
                pledge_info={"pledge_ratio": 0.0},
                max_years=10,
                refresh_valuation=False,
                listing_date=listing_date,
                daily_fetcher=lambda *_args: [{
                    "date": listing_date,
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.5,
                    "close": 10.0,
                    "volume": 1000.0,
                    "amount": 1000000.0,
                }],
                save_callback=lambda _code, _name, data: saved.append(data),
            )

        fetch_fundamentals.assert_called_once_with(
            "301666", {"pledge_ratio": 0.0}, listing_date=listing_date
        )
        self.assertEqual(saved[0]["listing_date"], listing_date)

    def test_qfq_full_refetch_is_clamped_and_clears_unproven_coverage(self):
        listing_date = "2020-01-01"
        existing = [
            {**self._daily_row("2026-07-01", 41.2), "daily_change_pct": 1.402904},
            {**self._daily_row("2026-07-02", 28.17), "daily_change_pct": -2.593361},
        ]
        refreshed = [
            {
                "date": "2026-07-01", "open": 28.12, "high": 29.6, "low": 28.0,
                "close": 28.92, "volume": 108508.84, "amount": 446790220,
                "change_pct": 1.402525, "turnover_rate": 2.9065,
            },
            {
                "date": "2026-07-02", "open": 28.29, "high": 29.05, "low": 26.82,
                "close": 28.17, "volume": 93989.8, "amount": 378106574,
                "change_pct": -2.593361, "turnover_rate": 2.5176,
            },
        ]
        calls = []
        saved = []

        def daily_fetcher(_code, start_date, end_date):
            calls.append((start_date, end_date))
            return refreshed

        with patch.object(
            stock_crawl_price_valuation,
            "load_stock_file",
            return_value={
                "records": existing,
                "start_date": "2026-07-01",
                "end_date": "2026-07-02",
                "history_coverage_start_date": listing_date,
            },
        ), patch.object(
            stock_crawl_price_valuation,
            "latest_weekday_date",
            return_value="2026-07-02",
        ):
            stock_crawl_price_valuation.process_stock(
                "603281",
                "江瀚新材",
                1,
                1,
                max_years=10,
                refresh_valuation=False,
                listing_date=listing_date,
                daily_fetcher=daily_fetcher,
                save_callback=lambda _code, _name, data: saved.append(data),
            )

        self.assertEqual(calls, [(listing_date, "2026-07-02")])
        self.assertTrue(saved[0]["history_coverage_replace"])
        self.assertNotIn("history_coverage_start_date", saved[0])

    def test_records_before_listing_date_are_removed(self):
        listing_date = "2020-01-02"
        saved = []
        with patch.object(
            stock_crawl_price_valuation,
            "load_stock_file",
            return_value={
                "records": [
                    self._daily_row("2020-01-01", 9.0),
                    self._daily_row(listing_date, 10.0),
                ],
                "start_date": "2020-01-01",
                "end_date": listing_date,
                "history_refetched_at": "2020-01-02T15:11:00",
            },
        ), patch.object(
            stock_crawl_price_valuation,
            "latest_weekday_date",
            return_value=listing_date,
        ), patch.object(
            stock_crawl_price_valuation,
            "fetch_daily_range",
        ) as daily_fetch:
            stock_crawl_price_valuation.process_stock(
                "000001",
                "测试股票",
                1,
                1,
                max_years=10,
                refresh_valuation=False,
                listing_date=listing_date,
                save_callback=lambda _code, _name, data: saved.append(data),
            )

        daily_fetch.assert_not_called()
        self.assertEqual([row["date"] for row in saved[0]["records"]], [listing_date])
        self.assertTrue(saved[0]["history_replace"])


class HistoryCoverageBoundaryTest(unittest.TestCase):
    @staticmethod
    def _canonical_row(trade_date):
        return {
            "date": trade_date,
            "daily_open": 10.0,
            "daily_high": 11.0,
            "daily_low": 9.0,
            "daily_close": 10.5,
            "daily_volume": 1000.0,
            "daily_amount": 10000.0,
            "daily_change_pct": 0.0,
            "daily_turnover_rate": 1.0,
        }

    @staticmethod
    def _source_row(trade_date):
        return {
            "date": trade_date,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 1000.0,
            "amount": 10000.0,
            "change_pct": 0.0,
            "turnover_rate": 1.0,
        }

    def test_response_covering_floor_and_anchor_persists_boundary_and_skips_next_run(self):
        start_date = "2020-01-10"
        listing_date = "2010-01-01"
        history_floor = "2019-01-10"
        conn = stock_storage.connect(":memory:")
        stock_storage.save_stock(conn, {
            "symbol": "600822",
            "name": "上海物贸",
            "listing_date": listing_date,
            "history_refetched_at": f"{start_date}T15:11:00",
            "history": history_payload_from_records(
                "600822",
                "上海物贸",
                [self._canonical_row(start_date)],
                "test",
            ),
        })
        calls = []

        def anchor_fetch(_code, range_start, range_end):
            calls.append((range_start, range_end))
            return [
                self._source_row(history_floor),
                self._source_row(start_date),
            ]

        try:
            with patch.object(
                stock_crawl_price_valuation.ss,
                "thread_conn",
                return_value=conn,
            ), patch.object(
                stock_crawl_price_valuation,
                "latest_weekday_date",
                return_value=start_date,
            ):
                first = stock_crawl_price_valuation.process_stock(
                    "600822",
                    "上海物贸",
                    1,
                    1,
                    max_years=1,
                    refresh_valuation=False,
                    listing_date=listing_date,
                    daily_fetcher=anchor_fetch,
                )
                second = stock_crawl_price_valuation.process_stock(
                    "600822",
                    "上海物贸",
                    1,
                    1,
                    max_years=1,
                    refresh_valuation=False,
                    listing_date=listing_date,
                    daily_fetcher=lambda *_args: self.fail(
                        "verified coverage must skip a repeated left-boundary request"
                    ),
                )

            self.assertEqual(first["status"], "saved")
            self.assertEqual(second["status"], "up_to_date")
            self.assertEqual(calls, [(history_floor, start_date)])
            self.assertEqual(
                stock_storage.history_coverage_start_date(conn, "600822"),
                history_floor,
            )
        finally:
            conn.close()

    def test_anchor_only_does_not_permanently_claim_the_wide_left_gap(self):
        start_date = "2020-01-10"
        history_floor = "2019-01-10"
        conn = stock_storage.connect(":memory:")
        stock_storage.save_stock(conn, {
            "symbol": "600822",
            "name": "上海物贸",
            "listing_date": "2010-01-01",
            "history_refetched_at": f"{start_date}T15:11:00",
            "history": history_payload_from_records(
                "600822",
                "上海物贸",
                [self._canonical_row(start_date)],
                "test",
            ),
        })
        calls = []

        def anchor_only(_code, range_start, range_end):
            calls.append((range_start, range_end))
            return [self._source_row(start_date)]

        try:
            with patch.object(
                stock_crawl_price_valuation.ss,
                "thread_conn",
                return_value=conn,
            ), patch.object(
                stock_crawl_price_valuation,
                "latest_weekday_date",
                return_value=start_date,
            ):
                first = stock_crawl_price_valuation.process_stock(
                    "600822",
                    "上海物贸",
                    1,
                    1,
                    max_years=1,
                    refresh_valuation=False,
                    listing_date="2010-01-01",
                    daily_fetcher=anchor_only,
                )
                second = stock_crawl_price_valuation.process_stock(
                    "600822",
                    "上海物贸",
                    1,
                    1,
                    max_years=1,
                    refresh_valuation=False,
                    listing_date="2010-01-01",
                    daily_fetcher=anchor_only,
                )

            self.assertEqual(first["status"], "up_to_date")
            self.assertEqual(second["status"], "up_to_date")
            self.assertEqual(calls, [
                (history_floor, start_date),
                (history_floor, start_date),
            ])
            self.assertIsNone(
                stock_storage.history_coverage_start_date(conn, "600822")
            )
        finally:
            conn.close()

    def test_nonempty_left_response_without_known_anchor_is_rejected(self):
        start_date = "2020-01-10"
        existing = self._canonical_row(start_date)
        with patch.object(
            stock_crawl_price_valuation,
            "load_stock_file",
            return_value={
                "records": [existing],
                "start_date": start_date,
                "end_date": start_date,
                "history_refetched_at": f"{start_date}T15:11:00",
            },
        ), patch.object(
            stock_crawl_price_valuation,
            "latest_weekday_date",
            return_value=start_date,
        ):
            outcome = stock_crawl_price_valuation.process_stock(
                "600822",
                "上海物贸",
                1,
                1,
                max_years=1,
                refresh_valuation=False,
                listing_date="2010-01-01",
                daily_fetcher=lambda *_args: [self._source_row("2020-01-09")],
            )

        self.assertEqual(outcome["status"], "source_invalid")
        self.assertEqual(outcome["reason"], "left_boundary_anchor_missing")
        self.assertEqual(outcome["expected_anchor"], start_date)
        self.assertEqual(outcome["received_dates"], ["2020-01-09"])


class RefreshFailurePropagationTest(unittest.TestCase):
    def test_failed_stock_is_not_retried_after_other_stocks_finish(self):
        calls = []

        def fake_process(code, name, *args, save_callback=None, stage_callback=None, **kwargs):
            calls.append(code)
            if code == "000001":
                stage_callback("daily_history")
                raise RuntimeError("first-pass failure")
            return {"status": "queued"}

        with patch.object(
            stock_crawl_price_valuation,
            "process_stock",
            side_effect=fake_process,
        ), patch.object(
            stock_crawl_price_valuation,
            "resolve_stock_listing_dates",
            return_value=({
                "000001": "1991-04-03",
                "000002": "1991-01-29",
                "000003": "1991-01-01",
            }, {}),
        ), patch.object(
            stock_crawl_price_valuation.ss,
            "sync_sw3_member_market_caps",
            return_value=0,
        ):
            result = stock_crawl_price_valuation.refresh_stock_histories(
                {
                    "000001": "平安银行",
                    "000002": "万科A",
                    "000003": "国华网安",
                },
                workers=1,
            )

        self.assertEqual(calls, ["000001", "000002", "000003"])
        self.assertEqual(result["failed"], ["000001"])
        self.assertEqual(result["failure_details"], [{
            "code": "000001",
            "name": "平安银行",
            "stage": "daily_history",
            "error": "RuntimeError: first-pass failure",
        }])
        self.assertNotIn("retry_attempted", result)

    def test_empty_source_response_is_reported_without_retry(self):
        with patch.object(
            stock_crawl_price_valuation,
            "process_stock",
            return_value={"status": "source_empty"},
        ) as process_mock, patch.object(
            stock_crawl_price_valuation,
            "resolve_stock_listing_dates",
            return_value=({"000001": "1991-04-03"}, {}),
        ), patch.object(
            stock_crawl_price_valuation.ss,
            "sync_sw3_member_market_caps",
            return_value=0,
        ):
            result = stock_crawl_price_valuation.refresh_stock_histories(
                {"000001": "平安银行"}, workers=1
            )

        self.assertEqual(process_mock.call_count, 1)
        self.assertEqual(result["failed"], ["000001"])
        self.assertEqual(result["failure_details"][0]["stage"], "daily_history")
        self.assertIn("未返回日线", result["failure_details"][0]["error"])

    def test_write_failure_is_reported_without_retry(self):
        payload = {"symbol": "000001", "records": [{"date": "2026-07-17"}]}
        save_calls = []

        def queue_payload(code, name, *args, save_callback=None, **kwargs):
            save_callback(code, name, payload)
            return {"status": "queued"}

        def fail_first_save(code, name, data):
            save_calls.append((code, name, data))
            raise RuntimeError("database is locked")

        with patch.object(
            stock_crawl_price_valuation,
            "process_stock",
            side_effect=queue_payload,
        ) as process_mock, patch.object(
            stock_crawl_price_valuation,
            "resolve_stock_listing_dates",
            return_value=({"000001": "1991-04-03"}, {}),
        ), patch.object(
            stock_crawl_price_valuation,
            "save_stock_file",
            side_effect=fail_first_save,
        ), patch.object(
            stock_crawl_price_valuation.ss,
            "sync_sw3_member_market_caps",
            return_value=0,
        ):
            result = stock_crawl_price_valuation.refresh_stock_histories(
                {"000001": "平安银行"}, workers=1
            )

        self.assertEqual(process_mock.call_count, 1)
        self.assertEqual(len(save_calls), 1)
        self.assertIs(save_calls[0][2], payload)
        self.assertEqual(result["failure_details"], [{
            "code": "000001",
            "name": "平安银行",
            "stage": "stock_history_write",
            "error": "RuntimeError: database is locked",
        }])

    def test_all_successful_stocks_have_no_failure_report(self):
        calls = []

        def queue_success(code, name, *args, save_callback=None, **kwargs):
            calls.append(code)
            save_callback(code, name, {"symbol": code})
            return {"status": "queued"}

        with patch.object(
            stock_crawl_price_valuation,
            "process_stock",
            side_effect=queue_success,
        ), patch.object(
            stock_crawl_price_valuation,
            "resolve_stock_listing_dates",
            return_value=({
                "000001": "1991-04-03",
                "000002": "1991-01-29",
            }, {}),
        ), patch.object(
            stock_crawl_price_valuation,
            "save_stock_file",
        ), patch.object(
            stock_crawl_price_valuation.ss,
            "sync_sw3_member_market_caps",
            return_value=0,
        ) as sync_mock:
            result = stock_crawl_price_valuation.refresh_stock_histories(
                {"000001": "平安银行", "000002": "万科A"}, workers=1
            )

        self.assertEqual(calls, ["000001", "000002"])
        sync_mock.assert_called_once()
        self.assertEqual(result["failed"], [])
        self.assertEqual(result["failure_details"], [])

    def test_per_stock_failure_keeps_code_stage_and_exception(self):
        def fail_during_history(*args, **kwargs):
            kwargs["stage_callback"]("daily_history")
            raise RuntimeError("upstream disconnected")

        with patch.object(
            stock_crawl_price_valuation,
            "process_stock",
            side_effect=fail_during_history,
        ), patch.object(
            stock_crawl_price_valuation,
            "resolve_stock_listing_dates",
            return_value=({"000001": "1991-04-03"}, {}),
        ), patch.object(
            stock_crawl_price_valuation.ss,
            "sync_sw3_member_market_caps",
            return_value=0,
        ):
            result = stock_crawl_price_valuation.refresh_stock_histories(
                {"000001": "平安银行"}, workers=1
            )

        self.assertEqual(result["failed"], ["000001"])
        self.assertEqual(result["failure_details"], [{
            "code": "000001",
            "name": "平安银行",
            "stage": "daily_history",
            "error": "RuntimeError: upstream disconnected",
        }])

    def test_stock_writer_failure_keeps_structured_error(self):
        with patch.object(
            stock_crawl_price_valuation,
            "save_stock_file",
            side_effect=RuntimeError("database is locked"),
        ):
            writer = stock_crawl_price_valuation.StockDbWriter(maxsize=1)
            writer.start()
            writer.enqueue("000001", "平安银行", {"records": []})
            writer.wait()

        self.assertEqual(writer.failed, [("000001", "平安银行")])
        self.assertEqual(writer.failure_details[0]["stage"], "stock_history_write")
        self.assertEqual(writer.failure_details[0]["error"], "RuntimeError: database is locked")

    def test_empty_refresh_is_reported_even_when_old_history_exists(self):
        old_records = [
            {
                "date": "2026-07-16",
                "daily_open": 10.0,
                "daily_high": 10.5,
                "daily_low": 9.8,
                "daily_close": 10.2,
                "daily_volume": 1000.0,
                "daily_amount": 1020000.0,
                "daily_change_pct": 1.0,
                "daily_turnover_rate": 2.0,
            }
        ]
        with patch.object(
            stock_crawl_price_valuation,
            "load_stock_file",
            return_value={
                "records": old_records,
                "start_date": "2026-07-16",
                "end_date": "2026-07-16",
                "history_refetched_at": "2026-07-16T10:00:00",
                "history_coverage_start_date": "2016-07-19",
            },
        ), patch.object(
            stock_crawl_price_valuation,
            "latest_weekday_date",
            return_value="2026-07-17",
        ):
            outcome = stock_crawl_price_valuation.process_stock(
                "000001",
                "平安银行",
                1,
                1,
                refresh_valuation=False,
                daily_fetcher=lambda *_args, **_kwargs: [],
                listing_date="1991-04-03",
            )

        self.assertEqual(outcome["status"], "source_empty")

        with patch.object(
            stock_crawl_price_valuation,
            "process_stock",
            return_value=outcome,
        ), patch.object(
            stock_crawl_price_valuation,
            "resolve_stock_listing_dates",
            return_value=({"000001": "1991-04-03"}, {}),
        ), patch.object(
            stock_crawl_price_valuation.ss,
            "sync_sw3_member_market_caps",
            return_value=0,
        ):
            result = stock_crawl_price_valuation.refresh_stock_histories(
                {"000001": "平安银行"}, workers=1
            )

        self.assertEqual(result["failed"], ["000001"])
        self.assertEqual(result["failure_details"][0]["stage"], "daily_history")
        self.assertIn("未返回日线", result["failure_details"][0]["error"])

    def test_empty_segment_pool_is_a_reported_failure(self):
        with patch.object(
            stock_crawl_price_valuation,
            "get_segment_leader_stocks",
            return_value={},
        ), patch.object(
            stock_crawl_price_valuation,
            "_plan_fundamentals_refresh",
        ) as planner:
            result = stock_crawl_price_valuation.run_segment_leader_refresh()

        planner.assert_not_called()
        self.assertEqual(result["failure_details"][0]["stage"], "segment_leader_pool")
        self.assertIn("龙头池为空", result["failure_details"][0]["error"])

    def test_unexpected_cli_failure_still_writes_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "path with spaces" / "failures.json"
            with patch.object(
                stock_crawl_price_valuation,
                "run_segment_leader_refresh",
                side_effect=RuntimeError("worker coordinator crashed"),
            ):
                exit_code = stock_crawl_price_valuation.cli(
                    ["--failure-report", str(report_path)]
                )
            payload = stock_crawl_common.load_json_file(report_path)

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            payload["failure_details"][0]["stage"], "segment_leader_refresh"
        )
        self.assertIn(
            "worker coordinator crashed", payload["failure_details"][0]["error"]
        )

    def test_segment_failure_sidecar_is_merged_into_refresh_report(self):
        detail = {
            "code": "301332",
            "name": "德尔玛",
            "stage": "daily_history",
            "error": "RuntimeError: all daily sources failed",
        }

        def fake_run_step(name, cmd, *, timeout=None, env=None, skip=False):
            if skip:
                return stock_data_refresh.StepResult(
                    name, " ".join(cmd), True, 0, 0.0, skipped=True
                )
            if name == "segment_leader_history":
                path = Path(cmd[cmd.index("--failure-report") + 1])
                stock_crawl_common.write_json_file(path, {
                    "requested": 976,
                    "processed": 976,
                    "write_enqueued": 975,
                    "write_completed": 975,
                    "failed": ["301332"],
                    "failure_details": [detail],
                })
                return stock_data_refresh.StepResult(name, " ".join(cmd), False, 1, 0.0)
            return stock_data_refresh.StepResult(name, " ".join(cmd), True, 0, 0.0)

        def fake_local_step(name, command, func):
            return stock_data_refresh.StepResult(name, command, True, 0, 0.0)

        with patch.object(
            stock_data_refresh, "run_step", side_effect=fake_run_step
        ), patch.object(
            stock_data_refresh, "local_step_result", side_effect=fake_local_step
        ), patch.object(
            stock_data_refresh, "collect_data_health", return_value={}
        ), patch.object(stock_data_refresh, "write_json_file"):
            report = stock_data_refresh.refresh_before_server(mode="full", timeout=1)

        history = next(
            step for step in report["steps"] if step["name"] == "segment_leader_history"
        )
        self.assertEqual(history["meta"]["failure_details"], [detail])
        self.assertEqual(history["meta"]["write_enqueued"], 975)
        self.assertEqual(history["meta"]["write_completed"], 975)
        self.assertIn("301332 [daily_history]", history["error"])
        self.assertEqual(report["failures"], [{"step": "segment_leader_history", **detail}])

    def test_hot_money_failure_sidecar_expands_global_exit_to_per_stock_errors(self):
        detail = {
            "code": "603075",
            "name": "热威股份",
            "stage": "fundamentals",
            "error": "RuntimeError: 603075 基本面数据不完整: indicators.records",
        }

        def fake_run_step(name, cmd, *, timeout=None, env=None, skip=False):
            if skip:
                return stock_data_refresh.StepResult(
                    name, " ".join(cmd), True, 0, 0.0, skipped=True
                )
            if name == "hot_money_small_cap_universe":
                path = Path(cmd[cmd.index("--failure-report") + 1])
                stock_crawl_common.write_json_file(path, {
                    "requested": 558,
                    "processed": 558,
                    "write_enqueued": 557,
                    "write_completed": 557,
                    "failed": ["603075"],
                    "failure_details": [detail],
                })
                return stock_data_refresh.StepResult(
                    name, " ".join(cmd), False, 1, 0.0
                )
            return stock_data_refresh.StepResult(
                name, " ".join(cmd), True, 0, 0.0
            )

        def fake_local_step(name, command, func):
            return stock_data_refresh.StepResult(name, command, True, 0, 0.0)

        with patch.object(
            stock_data_refresh, "run_step", side_effect=fake_run_step
        ), patch.object(
            stock_data_refresh, "local_step_result", side_effect=fake_local_step
        ), patch.object(
            stock_data_refresh, "collect_data_health", return_value={}
        ), patch.object(stock_data_refresh, "write_json_file"):
            report = stock_data_refresh.refresh_before_server(mode="full", timeout=1)

        hot_money = next(
            step
            for step in report["steps"]
            if step["name"] == "hot_money_small_cap_universe"
        )
        self.assertEqual(hot_money["meta"]["failure_details"], [detail])
        self.assertEqual(hot_money["meta"]["write_completed"], 557)
        self.assertIn("603075 [fundamentals]", hot_money["error"])
        self.assertIn(
            {"step": "hot_money_small_cap_universe", **detail},
            report["failures"],
        )
        short_step = next(
            step for step in report["steps"] if step["name"] == "short_signal_enrichment"
        )
        strategy_step = next(
            step for step in report["steps"] if step["name"] == "strategy_results"
        )
        self.assertTrue(short_step["skipped"])
        self.assertTrue(strategy_step["skipped"])

    def test_nonzero_child_with_empty_sidecar_still_has_an_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar = Path(tmpdir) / "failure.json"
            stock_crawl_common.write_json_file(sidecar, {"failure_details": []})
            step = stock_data_refresh.StepResult(
                "segment_leader_history", "python child.py", False, 1, 0.0
            )
            stock_data_refresh._attach_failure_sidecar(step, sidecar)

        self.assertIn("没有异常明细", step.error)

    def test_stale_health_rows_become_per_stock_report_failures(self):
        failures = stock_data_refresh._collect_health_failures({
            "hot_money_history_reference_date": "2026-07-17",
            "hot_money_history_cutoff": "2026-07-10",
            "hot_money_history_stale_details": [
                {"code": "1", "latest_date": "2026-07-01"},
                {"code": "830001", "latest_date": None},
            ],
        })

        self.assertEqual([item["code"] for item in failures], ["000001", "830001"])
        self.assertTrue(all(
            item["stage"] == "hot_money_history_freshness" for item in failures
        ))
        self.assertIn("2026-07-01", failures[0]["error"])
        self.assertIn("无日线", failures[1]["error"])

    def test_failed_dependency_skips_strategy_rebuild_and_marks_report_failed(self):
        calls = []

        def fake_run_step(name, cmd, *, timeout=None, env=None, skip=False):
            calls.append((name, skip))
            if skip:
                return stock_data_refresh.StepResult(
                    name, " ".join(cmd), True, 0, 0.0, skipped=True
                )
            ok = name != "hot_money_small_cap_universe"
            return stock_data_refresh.StepResult(
                name, " ".join(cmd), ok, 0 if ok else 2, 0.0
            )

        def fake_local_step(name, command, func):
            return stock_data_refresh.StepResult(name, command, True, 0, 0.0)

        with patch.object(stock_data_refresh, "run_step", side_effect=fake_run_step), \
             patch.object(stock_data_refresh, "local_step_result", side_effect=fake_local_step), \
             patch.object(stock_data_refresh, "collect_data_health", return_value={}), \
             patch.object(stock_data_refresh, "write_json_file"):
            report = stock_data_refresh.refresh_before_server(mode="full", timeout=1)

        self.assertFalse(report["ok"])
        self.assertIn(("short_signal_enrichment", True), calls)
        self.assertIn(("strategy_results", True), calls)

    def test_cli_exits_nonzero_for_failed_report(self):
        failures = [
            {
                "step": "segment_leader_history",
                "code": f"{index:06d}",
                "name": f"股票{index}",
                "stage": "daily_history",
                "error": f"RuntimeError: failure {index}",
            }
            for index in range(1, 12)
        ]
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(
            stock_data_refresh,
            "refresh_before_server",
            return_value={"ok": False, "failures": failures},
        ), patch.object(sys, "argv", ["stock_data_refresh.py"]), \
             redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                stock_data_refresh.main()

        self.assertEqual(raised.exception.code, 1)
        output = stderr.getvalue()
        self.assertIn("失败明细共 11 条", output)
        self.assertIn(
            "[refresh failure] 000001 股票1 [daily_history] RuntimeError: failure 1",
            output,
        )
        self.assertIn(
            "[refresh failure] 000011 股票11 [daily_history] RuntimeError: failure 11",
            output,
        )
        self.assertEqual(output.count("[refresh failure]"), 11)

    @unittest.skipIf(sys.platform == "win32", "process-group assertion is POSIX-specific")
    def test_timed_out_step_terminates_descendant_processes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            marker = Path(tmpdir) / "child-survived"
            child_code = (
                "import pathlib,time; time.sleep(0.7); "
                f"pathlib.Path({str(marker)!r}).write_text('alive')"
            )
            parent_code = (
                "import subprocess,sys,time; "
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                "time.sleep(10)"
            )

            result = stock_data_refresh.run_step(
                "timeout-tree-test",
                [sys.executable, "-c", parent_code],
                timeout=0.1,
            )
            time.sleep(0.9)
            child_survived = marker.exists()

        self.assertFalse(result.ok)
        self.assertEqual(result.returncode, 124)
        self.assertFalse(child_survived)


class DailySourceFallbackTest(unittest.TestCase):
    def test_empty_primary_source_falls_back_to_next_source(self):
        expected = [{"date": "2026-07-10", "close": 10.0}]
        warnings = []
        attempts = []
        with patch.object(
            stock_crawl_common,
            "_enabled_daily_source_groups",
            return_value=[("primary",), ("fallback",)],
        ), patch.object(
            stock_crawl_common,
            "_fetch_daily_source",
            side_effect=[[], expected],
        ) as fetch_mock, patch.object(
            stock_crawl_common, "_record_daily_source_success"
        ) as success_mock:
            records = stock_crawl_common.fetch_qfq_daily_records(
                "000001",
                "20260710",
                "20260710",
                warn=warnings.append,
                attempt_callback=attempts.append,
            )

        self.assertEqual(records, expected)
        self.assertEqual(fetch_mock.call_count, 2)
        success_mock.assert_called_once_with("fallback")
        self.assertTrue(any("空数据" in message for message in warnings))
        self.assertEqual(attempts, [
            {"source": "primary", "status": "empty"},
            {"source": "fallback", "status": "success", "rows": 1},
        ])

    def test_eastmoney_adapter_filters_rows_outside_requested_range(self):
        frame = pd.DataFrame([
            {"日期": "2026-07-09", "开盘": 9, "最高": 10, "最低": 8,
             "收盘": 9.5, "成交量": 100, "成交额": 1000, "涨跌幅": 0, "换手率": 1},
            {"日期": "2026-07-10", "开盘": 10, "最高": 11, "最低": 9,
             "收盘": 10.5, "成交量": 200, "成交额": 2000, "涨跌幅": 1, "换手率": 2},
            {"日期": "2026-07-11", "开盘": 11, "最高": 12, "最低": 10,
             "收盘": 11.5, "成交量": 300, "成交额": 3000, "涨跌幅": 2, "换手率": 3},
        ])
        with patch.object(
            stock_crawl_common.ak,
            "stock_zh_a_hist",
            return_value=frame,
        ):
            rows = stock_crawl_common._fetch_daily_eastmoney_qfq(
                "000001", "2026-07-10", "2026-07-10", include_trading_value=True
            )

        self.assertEqual([row["date"] for row in rows], ["2026-07-10"])

    def test_empty_primary_does_not_hide_fallback_transport_failure(self):
        def passthrough(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch.object(
            stock_crawl_common,
            "_enabled_daily_source_groups",
            return_value=[("primary",), ("fallback",)],
        ), patch.object(
            stock_crawl_common,
            "_fetch_daily_source",
            side_effect=[[], RuntimeError("fallback unavailable")],
        ), patch.object(
            stock_crawl_common, "retry_fetch", side_effect=passthrough
        ):
            with self.assertRaisesRegex(RuntimeError, "fallback unavailable"):
                stock_crawl_common.fetch_qfq_daily_records(
                    "000001", "20260710", "20260710"
                )

    def test_etf_empty_dedicated_source_falls_back_to_generic_public_sources(self):
        expected = [{"date": "2026-07-10", "close": 4.2}]
        warnings = []
        with patch.object(
            stock_crawl_common,
            "_fetch_daily_eastmoney_etf_qfq",
            return_value=[],
        ) as etf_mock, patch.object(
            stock_crawl_common,
            "fetch_qfq_daily_records",
            return_value=expected,
        ) as fallback_mock:
            records = stock_crawl_common.fetch_etf_qfq_daily_records(
                "510300",
                "20260701",
                "20260710",
                include_trading_value=True,
                warn=warnings.append,
            )

        self.assertEqual(records, expected)
        etf_mock.assert_called_once_with(
            "510300",
            "20260701",
            "20260710",
            include_trading_value=True,
        )
        fallback_mock.assert_called_once_with(
            "510300",
            "20260701",
            "20260710",
            include_trading_value=True,
            warn=warnings.append,
        )
        self.assertTrue(any("ETF行情返回空数据" in message for message in warnings))


class PlateCoverageIntegrityTest(unittest.TestCase):
    def test_incremental_refresh_starts_from_oldest_per_plate_frontier(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "plate.sqlite3"
            conn = plate_storage.connect(db_path)
            try:
                plate_storage.save_sw2_daily_rows(conn, [
                    sw2_row("801001", "行业甲", "2026-07-07"),
                    sw2_row("801002", "行业乙", "2026-07-07"),
                    sw2_row("801001", "行业甲", "2026-07-08"),
                ])
                self.assertEqual(
                    plate_storage.latest_trade_date(conn), "2026-07-08"
                )
                self.assertEqual(
                    plate_storage.oldest_latest_trade_date(conn), "2026-07-07"
                )
            finally:
                conn.close()

            requested = []

            def fake_fetch(start, end, *, page_size):
                requested.append((start, end))
                return [
                    sw2_row("801001", "行业甲", "2026-07-08", 101),
                    sw2_row("801002", "行业乙", "2026-07-08", 102),
                ]

            with patch.object(
                plate_crawl_history, "fetch_sw2_daily_rows", side_effect=fake_fetch
            ):
                result = plate_crawl_history.fetch_sw2_daily_analysis(
                    date(2026, 7, 1),
                    date(2026, 7, 8),
                    chunk_days=30,
                    sleep_sec=0,
                    db_file=str(db_path),
                )

            self.assertEqual(requested[0][0], date(2026, 7, 8))
            self.assertEqual(result["coverage_trade_date"], "2026-07-08")

    def test_incremental_refresh_repairs_internal_cross_section_gap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "plate.sqlite3"
            conn = plate_storage.connect(db_path)
            try:
                plate_storage.save_sw2_daily_rows(conn, [
                    sw2_row("801001", "行业甲", "2026-07-07"),
                    sw2_row("801002", "行业乙", "2026-07-07"),
                    sw2_row("801001", "行业甲", "2026-07-08"),
                    sw2_row("801001", "行业甲", "2026-07-09"),
                    sw2_row("801002", "行业乙", "2026-07-09"),
                ])
                self.assertEqual(
                    plate_storage.oldest_latest_trade_date(conn), "2026-07-09"
                )
                self.assertEqual(
                    plate_storage.recent_incomplete_trade_date(conn), "2026-07-08"
                )
            finally:
                conn.close()

            requested = []

            def fake_fetch(start, end, *, page_size):
                requested.append((start, end))
                return [
                    sw2_row("801001", "行业甲", "2026-07-08"),
                    sw2_row("801002", "行业乙", "2026-07-08"),
                    sw2_row("801001", "行业甲", "2026-07-09"),
                    sw2_row("801002", "行业乙", "2026-07-09"),
                ]

            with patch.object(
                plate_crawl_history, "fetch_sw2_daily_rows", side_effect=fake_fetch
            ):
                result = plate_crawl_history.fetch_sw2_daily_analysis(
                    date(2026, 7, 1),
                    date(2026, 7, 9),
                    chunk_days=30,
                    sleep_sec=0,
                    db_file=str(db_path),
                )

            self.assertEqual(requested[0][0], date(2026, 7, 8))
            self.assertIsNone(result["coverage_gap_date"])

    def test_first_failed_chunk_stops_later_chunks_from_advancing_watermark(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            plate_crawl_history,
            "fetch_sw2_daily_rows",
            side_effect=RuntimeError("source down"),
        ) as fetch_mock:
            result = plate_crawl_history.fetch_sw2_daily_analysis(
                date(2026, 7, 1),
                date(2026, 7, 10),
                chunk_days=3,
                sleep_sec=0,
                db_file=str(Path(tmpdir) / "plate.sqlite3"),
            )

        self.assertEqual(fetch_mock.call_count, 1)
        self.assertEqual(len(result["errors"]), 1)

    def test_retired_plate_does_not_permanently_pin_coverage_frontier(self):
        conn = plate_storage.connect(":memory:")
        try:
            start = date(2026, 4, 1)
            rows = [sw2_row("809999", "已退役", start.strftime("%Y-%m-%d"))]
            for offset in range(45):
                trade_date = (start + timedelta(days=offset + 1)).strftime("%Y-%m-%d")
                rows.extend([
                    sw2_row("801001", "行业甲", trade_date),
                    sw2_row("801002", "行业乙", trade_date),
                ])
            plate_storage.save_sw2_daily_rows(conn, rows)

            self.assertEqual(
                plate_storage.oldest_latest_trade_date(conn),
                (start + timedelta(days=45)).strftime("%Y-%m-%d"),
            )
        finally:
            conn.close()


class FundamentalsIntegrityTest(unittest.TestCase):
    def test_sina_indicators_clamp_to_listing_year_and_do_not_call_fallback(self):
        sina = pd.DataFrame([
            {"日期": "2024-12-31", "净资产收益率(%)": 8.0},
            {"日期": "2025-12-31", "净资产收益率(%)": 10.0},
        ])
        expected_start_year = str(max(
            datetime.now().year - stock_crawl_fundamentals.FINANCIAL_YEARS,
            2021,
        ))
        with patch.object(
            stock_crawl_fundamentals.ak,
            "stock_financial_analysis_indicator",
            return_value=sina,
        ) as sina_fetch, patch.object(
            stock_crawl_fundamentals.ak,
            "stock_financial_analysis_indicator_em",
        ) as em_fetch:
            result = stock_crawl_fundamentals.fetch_financial_indicators(
                "001309", listing_date="2021-11-12"
            )

        sina_fetch.assert_called_once_with(symbol="001309", start_year=expected_start_year)
        em_fetch.assert_not_called()
        self.assertEqual(result["source"], "akshare.stock_financial_analysis_indicator")
        self.assertEqual(
            [row["date"] for row in result["records"]],
            ["2025-12-31", "2024-12-31"],
        )
        self.assertEqual(result["roe_stats"]["mean"], 9.0)

    def test_empty_sina_indicators_fall_back_to_real_eastmoney_fields(self):
        em = pd.DataFrame([
            {
                "REPORT_DATE": "2025-12-31 00:00:00",
                "ROEJQ": 10.57,
                "ROEKCJQ": 99.0,
                "EPSXS": 21.76,
                "EPSKCJB": None,
                "BPS": 216.32,
                "MGJYXJJE": 21.49,
                "XSMLL": 89.76,
                "XSJLL": 52.22,
                "TOTALOPERATEREVETZ": 6.34,
                "PARENTNETPROFITTZ": 1.47,
                "EQUITY_YOYRATIO_PK": 7.5,
                "TOTAL_ASSETS_PK": 300_000_000.0,
                "KCFJCXSYJLR": 27_239_985_194.41,
                "ZCFZL": 12.12,
            },
        ])
        with patch.object(
            stock_crawl_fundamentals.ak,
            "stock_financial_analysis_indicator",
            return_value=pd.DataFrame(),
        ), patch.object(
            stock_crawl_fundamentals.ak,
            "stock_financial_analysis_indicator_em",
            return_value=em,
        ) as em_fetch:
            result = stock_crawl_fundamentals.fetch_financial_indicators(
                "600519", listing_date="2001-08-27"
            )

        em_fetch.assert_called_once_with(symbol="600519.SH", indicator="按报告期")
        record = result["records"][0]
        self.assertEqual(result["source"], "akshare.stock_financial_analysis_indicator_em")
        self.assertEqual(record["roe"], 10.57)
        self.assertEqual(record["roe_weighted"], 10.57)
        self.assertEqual(record["eps_diluted"], 21.76)
        self.assertIsNone(record["eps_adjusted"])
        self.assertEqual(record["bvps_adjusted"], 216.32)
        self.assertEqual(record["ocfps"], 21.49)
        self.assertEqual(record["gross_margin"], 89.76)
        self.assertEqual(record["revenue_growth"], 6.34)
        self.assertEqual(record["net_assets_growth"], 7.5)
        self.assertEqual(record["total_assets"], 300_000_000.0)
        self.assertEqual(record["asset_liability_ratio"], 12.12)
        self.assertNotEqual(record["roe"], 99.0)

    def test_eastmoney_indicator_symbol_maps_supported_markets_and_rejects_bse(self):
        expected = {
            "001309": "001309.SZ",
            "301666": "301666.SZ",
            "603075": "603075.SH",
            "688548": "688548.SH",
        }
        for code, symbol in expected.items():
            with self.subTest(code=code):
                self.assertEqual(
                    stock_crawl_fundamentals._eastmoney_indicator_symbol(code),
                    symbol,
                )
        with self.assertRaisesRegex(ValueError, "暂不支持"):
            stock_crawl_fundamentals._eastmoney_indicator_symbol("920001")

    def test_two_empty_indicator_sources_remain_incomplete(self):
        with patch.object(
            stock_crawl_fundamentals.ak,
            "stock_financial_analysis_indicator",
            return_value=pd.DataFrame(),
        ), patch.object(
            stock_crawl_fundamentals.ak,
            "stock_financial_analysis_indicator_em",
            return_value=pd.DataFrame(),
        ):
            indicators = stock_crawl_fundamentals.fetch_financial_indicators("301666")

        payload = valid_fundamentals()
        payload["indicators"] = indicators
        self.assertEqual(indicators["records"], [])
        with self.assertRaisesRegex(RuntimeError, "indicators.records"):
            stock_crawl_fundamentals.ensure_fundamentals_complete(payload, "301666")

    def test_fetch_fundamentals_passes_listing_date_to_indicators(self):
        payload = valid_fundamentals()
        with patch.object(
            stock_crawl_fundamentals,
            "fetch_financial_reports",
            return_value=payload["financials"],
        ), patch.object(
            stock_crawl_fundamentals,
            "fetch_financial_indicators",
            return_value=payload["indicators"],
        ) as indicator_fetch, patch.object(
            stock_crawl_fundamentals,
            "fetch_dividend_history",
            return_value=payload["dividends"],
        ):
            result = stock_crawl_fundamentals.fetch_fundamentals(
                "301666", listing_date="2025-08-08"
            )

        indicator_fetch.assert_called_once_with("301666", listing_date="2025-08-08")
        self.assertIn("financials_refetched_at", result)

    def test_empty_required_record_family_is_incomplete(self):
        payload = valid_fundamentals()
        payload["indicators"] = {"records": []}

        self.assertEqual(
            stock_crawl_fundamentals.fundamentals_missing_families(payload),
            ["indicators.records"],
        )
        with self.assertRaisesRegex(RuntimeError, "indicators.records"):
            stock_crawl_fundamentals.ensure_fundamentals_complete(payload, "1")

    def test_date_only_rows_are_incomplete_after_upstream_schema_drift(self):
        payload = {
            "financials": {
                family: [{"date": "2025-12-31"}]
                for family in ("income", "balance", "cashflow")
            },
            "indicators": {"records": [{"date": "2025-12-31"}]},
            "dividends": {"records": []},
        }

        self.assertEqual(
            stock_crawl_fundamentals.fundamentals_missing_families(payload),
            [
                "financials.income", "financials.balance",
                "financials.cashflow", "indicators.records",
            ],
        )

    def test_main_exits_nonzero_when_any_target_fundamental_fails(self):
        with patch.object(
            stock_crawl_fundamentals,
            "get_segment_leader_universe",
            return_value={"600001": {"name": "测试"}},
        ), patch.object(
            stock_crawl_fundamentals, "fetch_pledge_data_bulk", return_value={}
        ), patch.object(
            stock_crawl_fundamentals,
            "crawl_stocks",
            return_value={"errors": {"600001": {"error": "empty"}}},
        ), patch.object(
            stock_crawl_fundamentals, "strip_proxy_env"
        ), patch.object(
            sys,
            "argv",
            ["stock_crawl_fundamentals.py", "--mode", "full"],
        ):
            with self.assertRaises(SystemExit) as raised:
                stock_crawl_fundamentals.main()

        self.assertEqual(raised.exception.code, 1)

    def test_fetch_fundamentals_does_not_timestamp_empty_results(self):
        with patch.object(
            stock_crawl_fundamentals,
            "fetch_financial_reports",
            return_value={"income": [], "balance": [], "cashflow": []},
        ), patch.object(
            stock_crawl_fundamentals,
            "fetch_financial_indicators",
            return_value={"records": [], "roe_stats": {}},
        ), patch.object(
            stock_crawl_fundamentals,
            "fetch_dividend_history",
            return_value={"records": []},
        ):
            with self.assertRaisesRegex(RuntimeError, "基本面数据不完整"):
                stock_crawl_fundamentals.fetch_fundamentals("000001")

    def test_load_existing_rejects_empty_indicators_even_with_fresh_daily_rows(self):
        conn = stock_storage.connect(":memory:")
        try:
            payload = {
                "symbol": "600001",
                "name": "测试",
                **valid_fundamentals(),
                "history": history_payload_from_records(
                    "600001",
                    "测试",
                    complete_daily_rows(
                        date(2025, 7, 15), stock_crawl_fundamentals.MIN_COMPLETE_DAILY_ROWS
                    ),
                    "test",
                ),
            }
            payload["indicators"] = {"records": []}
            stock_storage.save_stock(conn, payload)
            with patch.object(stock_crawl_fundamentals.ss, "thread_conn", return_value=conn), \
                 patch.object(
                     stock_crawl_fundamentals,
                     "latest_weekday_date",
                     return_value="2026-02-13",
                 ):
                existing = stock_crawl_fundamentals.load_existing()
        finally:
            conn.close()

        self.assertNotIn("600001", existing)

    def test_fundamentals_state_map_decodes_blobs_and_marks_missing_rows(self):
        conn = stock_storage.connect(":memory:")
        try:
            stock_storage.save_stock(conn, {
                "symbol": "600001",
                "name": "完整",
                "financials_refetched_at": "2026-07-19T10:00:00",
                **valid_fundamentals(),
            }, write_history=False)
            conn.execute(
                "INSERT INTO stock_meta "
                "(code, name, financials_refetched_at, financials_json, "
                "indicators_json, dividends_json) VALUES (?, ?, ?, ?, ?, ?)",
                ("600002", "损坏", "2026-07-19T10:00:00", "{}", "{bad", "{}"),
            )
            conn.commit()

            states = stock_storage.fundamentals_state_map(
                conn, ["600001", "600002", "600003"]
            )
        finally:
            conn.close()

        self.assertEqual(states["600001"]["indicators"]["records"][0]["roe"], 10)
        self.assertIsNone(states["600002"]["indicators"])
        self.assertIsNone(states["600003"]["financials_refetched_at"])
        self.assertIsNone(states["600003"]["financials"])

    def test_planner_repairs_recent_but_incomplete_fundamentals(self):
        conn = stock_storage.connect(":memory:")
        complete = {"symbol": "600001", "name": "完整", **valid_fundamentals()}
        complete["financials_refetched_at"] = "2026-07-19T10:00:00"
        incomplete = {"symbol": "600002", "name": "缺指标", **valid_fundamentals()}
        incomplete["indicators"] = {"records": []}
        incomplete["financials_refetched_at"] = "2026-07-19T10:00:00"
        stock_storage.save_stock(conn, complete, write_history=False)
        stock_storage.save_stock(conn, incomplete, write_history=False)

        with patch.object(
            stock_crawl_price_valuation.ss,
            "connect",
            return_value=conn,
        ), patch.object(
            stock_crawl_fundamentals,
            "fetch_latest_report_announce_dates",
            return_value={},
        ), patch.object(
            stock_crawl_fundamentals,
            "fetch_pledge_data_bulk",
            return_value={},
        ), patch.object(
            stock_crawl_fundamentals,
            "needs_fundamentals_refresh",
            return_value=False,
        ):
            needs, pledge = stock_crawl_price_valuation._plan_fundamentals_refresh(
                ["600001", "600002", "600003"]
            )

        self.assertEqual(
            needs,
            {"600001": False, "600002": True, "600003": True},
        )
        self.assertEqual(pledge, {})


class CompletedDailyBarTest(unittest.TestCase):
    def test_latest_weekday_date_excludes_current_session_before_close(self):
        self.assertEqual(
            latest_weekday_date(datetime(2026, 7, 10, 14, 59)),
            "2026-07-09",
        )
        self.assertEqual(
            latest_weekday_date(datetime(2026, 7, 10, 15, 10)),
            "2026-07-10",
        )
        self.assertEqual(latest_weekday_date("2026-07-10"), "2026-07-10")

    def test_overlap_upserts_final_same_date_ohlcv_without_full_rewrite(self):
        existing = [{
            "date": "2026-07-01",
            "daily_open": 28.1,
            "daily_high": 28.9,
            "daily_low": 28.0,
            "daily_close": 28.90,
            "daily_volume": 100.0,
            "daily_amount": 1000.0,
            "daily_change_pct": 1.3,
            "daily_turnover_rate": 0.1,
        }]
        final_row = {
            "date": "2026-07-01",
            "open": 28.1,
            "high": 29.6,
            "low": 28.0,
            "close": 28.92,
            "volume": 108508.84,
            "amount": 446790220,
            "change_pct": 1.402525,
            "turnover_rate": 2.9065,
        }
        saved = []

        with patch.object(
            stock_crawl_price_valuation,
            "load_stock_file",
            return_value={
                "records": existing,
                "start_date": "2026-07-01",
                "end_date": "2026-07-01",
            },
        ), patch.object(
            stock_crawl_price_valuation,
            "latest_weekday_date",
            return_value="2026-07-01",
        ), patch.object(
            stock_crawl_price_valuation,
            "fetch_daily_range",
            return_value=[final_row],
        ):
            stock_crawl_price_valuation.process_stock(
                "603281",
                "江瀚新材",
                1,
                1,
                save_callback=lambda code, name, data: saved.append(data),
                max_years=0.0,
                refresh_valuation=False,
                listing_date="2020-01-01",
            )

        result = saved[0]
        self.assertFalse(result["history_replace"])
        self.assertEqual(
            [row["date"] for row in result["history_write_records"]],
            ["2026-07-01"],
        )
        self.assertEqual(result["history_write_records"][0]["daily_volume"], 108508.84)
        self.assertEqual(result["history_write_records"][0]["daily_high"], 29.6)


if __name__ == "__main__":
    unittest.main()
