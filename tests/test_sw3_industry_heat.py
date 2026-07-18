import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sw3_industry_heat as heat


def _dates(count=20):
    start = date(2026, 6, 1)
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def _segments():
    return [
        {
            "segment_code": "850001",
            "segment_name": "行业甲",
            "parent_segment": "父行业",
            "member_count": 1,
            "members": [{"code": "000001", "market_cap_yi": 100.0}],
        },
        {
            "segment_code": "850002",
            "segment_name": "行业乙",
            "parent_segment": "父行业",
            "member_count": 1,
            "members": [{"code": "000002", "market_cap_yi": 200.0}],
        },
    ]


def _histories():
    dates = _dates()
    return {
        "850001": {trade_date: 10.0 + index for index, trade_date in enumerate(dates)},
        "850002": {trade_date: 30.0 for trade_date in dates},
    }


class Sw3IndustryHeatTest(unittest.TestCase):
    def test_compute_uses_sws_only_sources_and_complete_rankings(self):
        histories = _histories()
        last_date = _dates()[-1]
        payload = heat.compute_sw3_industry_heat(
            _segments(),
            histories,
            generated_at="2026-07-18 10:00:00",
            estimated_amount_dates_by_code={"850001": {last_date}},
        )

        heat.validate_complete_report(payload, min_eligible_coverage=1.0)
        self.assertEqual(payload["schema"], heat.SW3_INDUSTRY_HEAT_SCHEMA)
        self.assertEqual(payload["as_of_date"], last_date)
        self.assertEqual(len(payload["rankings"]["hottest"]), 2)
        rows = {row["segment_code"]: row for row in payload["industries"]}
        self.assertEqual(
            rows["850001"]["amount_sources"],
            ["sws_analysis_share_estimate", "sws_trend"],
        )
        self.assertEqual(
            rows["850001"]["daily"][-1]["amount_source"],
            "sws_analysis_share_estimate",
        )
        self.assertEqual(
            rows["850002"]["amount_sources"],
            ["sws_trend"],
        )

    def test_compute_marks_akshare_component_sum_as_derived_not_estimated(self):
        histories = _histories()
        last_date = _dates()[-1]
        payload = heat.compute_sw3_industry_heat(
            _segments(),
            histories,
            generated_at="2026-07-18 10:00:00",
            derived_amount_dates_by_code={"850001": {last_date}},
            amount_source_by_code_date={
                "850001": {last_date: "akshare_stock_zh_a_spot_component"},
            },
            latest_amount_coverage_pct_by_code={"850001": 100.0},
        )

        rows = {row["segment_code"]: row for row in payload["industries"]}
        row = rows["850001"]
        self.assertTrue(row["amount_is_derived"])
        self.assertFalse(row["amount_is_estimate"])
        self.assertEqual(row["amount_derived_days"], 1)
        self.assertEqual(row["latest_amount_member_coverage_pct"], 100.0)
        self.assertTrue(row["daily"][-1]["amount_is_derived"])
        self.assertFalse(row["daily"][-1]["amount_is_estimate"])
        self.assertEqual(
            row["daily"][-1]["amount_source"],
            "akshare_stock_zh_a_spot_component",
        )
        self.assertEqual(payload["data_quality"]["latest_amount_source"], "mixed")

    def test_injected_latest_day_advances_report_and_records_source(self):
        histories = _histories()
        target_date = _dates(21)[-1]
        calls = []

        def latest_loader(segments, *, prior_as_of_date, cache_file):
            calls.append((len(segments), prior_as_of_date, Path(cache_file).name))
            return SimpleNamespace(
                histories={
                    "850001": {target_date: 40.0},
                    "850002": {target_date: 35.0},
                },
                expected_as_of_date=target_date,
                source="akshare_stock_zh_a_spot_component",
                source_by_code_date={
                    code: {target_date: "akshare_stock_zh_a_spot_component"}
                    for code in ("850001", "850002")
                },
                derived_dates_by_code={
                    code: {target_date} for code in ("850001", "850002")
                },
                coverage_pct_by_code={"850001": 100.0, "850002": 100.0},
                errors=[],
                quality={"snapshot_valid": True, "snapshot_row_count": 5500},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "heat.json"
            payload = heat.build_sw3_industry_heat_report(
                _segments(),
                history_fetcher=lambda code: histories[code],
                latest_amount_fetcher=latest_loader,
                output_file=output,
                max_workers=1,
                retries=1,
                min_eligible_coverage=1.0,
                analysis_fetcher=lambda *_args: [],
                progress=False,
            )

        self.assertEqual(calls, [(2, _dates()[-1], "sw3_akshare_latest_cache.json")])
        self.assertEqual(payload["as_of_date"], target_date)
        self.assertEqual(payload["data_quality"]["expected_as_of_date"], target_date)
        self.assertEqual(payload["data_quality"]["latest_amount_added_industry_count"], 2)
        self.assertEqual(
            payload["data_quality"]["latest_amount_source"],
            "akshare_stock_zh_a_spot_component",
        )
        self.assertEqual(payload["data_quality"]["derived_daily_points"], 2)

    def test_expected_latest_date_gate_preserves_previous_report(self):
        histories = _histories()
        target_date = _dates(21)[-1]

        def empty_latest(*_args, **_kwargs):
            return SimpleNamespace(
                histories={},
                expected_as_of_date=target_date,
                source="akshare_stock_zh_a_spot_component",
                source_by_code_date={},
                derived_dates_by_code={},
                coverage_pct_by_code={},
                errors=[{"stage": "market_snapshot", "error": "upstream unavailable"}],
                quality={"snapshot_valid": False},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "heat.json"
            output.write_text('{"sentinel": true}', encoding="utf-8")
            with self.assertRaisesRegex(
                heat.IndustryHeatIncompleteError,
                "报告截止日未到最新已收盘交易日",
            ):
                heat.build_sw3_industry_heat_report(
                    _segments(),
                    history_fetcher=lambda code: histories[code],
                    latest_amount_fetcher=empty_latest,
                    output_file=output,
                    max_workers=1,
                    retries=1,
                    min_eligible_coverage=1.0,
                    analysis_fetcher=lambda *_args: [],
                    progress=False,
                )
            disk_payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(disk_payload, {"sentinel": True})

    def test_previous_valid_estimates_bridge_analysis_outage_before_latest_day(self):
        previous_histories = _histories()
        previous_dates = set(_dates())
        previous = heat.compute_sw3_industry_heat(
            _segments(),
            previous_histories,
            estimated_amount_dates_by_code={"850002": previous_dates},
        )
        heat.validate_complete_report(previous, min_eligible_coverage=1.0)
        stale_dates = [
            (date(2024, 1, 1) + timedelta(days=index)).isoformat()
            for index in range(20)
        ]
        current_histories = {
            "850001": previous_histories["850001"],
            "850002": {trade_date: 5.0 for trade_date in stale_dates},
        }
        target_date = _dates(21)[-1]

        def latest_loader(*_args, **_kwargs):
            return SimpleNamespace(
                histories={
                    "850001": {target_date: 40.0},
                    "850002": {target_date: 35.0},
                },
                expected_as_of_date=target_date,
                source="akshare_stock_zh_a_spot_component",
                source_by_code_date={
                    code: {target_date: "akshare_stock_zh_a_spot_component"}
                    for code in ("850001", "850002")
                },
                derived_dates_by_code={
                    code: {target_date} for code in ("850001", "850002")
                },
                coverage_pct_by_code={"850001": 100.0, "850002": 100.0},
                errors=[],
                quality={"snapshot_valid": True},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "heat.json"
            output.write_text(json.dumps(previous), encoding="utf-8")
            payload = heat.build_sw3_industry_heat_report(
                _segments(),
                history_fetcher=lambda code: current_histories[code],
                latest_amount_fetcher=latest_loader,
                output_file=output,
                max_workers=1,
                retries=1,
                min_eligible_coverage=1.0,
                analysis_fetcher=lambda *_args: [],
                progress=False,
            )

        self.assertEqual(payload["as_of_date"], target_date)
        quality = payload["data_quality"]
        self.assertEqual(quality["previous_report_estimated_industry_count"], 1)
        self.assertEqual(quality["previous_report_estimated_daily_points"], 20)
        self.assertEqual(quality["estimated_industry_count"], 1)
        self.assertEqual(quality["estimated_daily_points"], 19)
        row = next(row for row in payload["industries"] if row["segment_code"] == "850002")
        self.assertEqual(row["daily"][-1]["amount_source"], "akshare_stock_zh_a_spot_component")
        self.assertTrue(row["daily"][-2]["amount_is_estimate"])

    def test_previous_report_reuse_rejects_derived_or_akshare_estimate_points(self):
        previous = heat.compute_sw3_industry_heat(
            _segments(),
            _histories(),
            estimated_amount_dates_by_code={"850002": set(_dates())},
        )
        row = next(row for row in previous["industries"] if row["segment_code"] == "850002")
        forbidden_date = row["daily"][0]["date"]
        row["daily"][0]["amount_is_derived"] = True
        row["daily"][0]["amount_source"] = "akshare_stock_zh_a_spot_component"

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "heat.json"
            output.write_text(json.dumps(previous), encoding="utf-8")
            histories, estimated_dates, as_of_date = heat._load_previous_report_estimates(
                output,
                allowed_codes={"850001", "850002"},
                window_days=20,
                min_eligible_coverage=1.0,
            )

        self.assertEqual(as_of_date, _dates()[-1])
        self.assertNotIn(forbidden_date, histories["850002"])
        self.assertNotIn(forbidden_date, estimated_dates["850002"])
        self.assertEqual(len(histories["850002"]), 19)

    def test_direct_sws_value_wins_when_latest_batch_has_same_date(self):
        histories = _histories()
        last_date = _dates()[-1]

        def duplicate_latest(*_args, **_kwargs):
            return SimpleNamespace(
                histories={"850001": {last_date: 999.0}},
                expected_as_of_date=last_date,
                source="akshare_stock_zh_a_spot_component",
                source_by_code_date={
                    "850001": {last_date: "akshare_stock_zh_a_spot_component"},
                },
                derived_dates_by_code={"850001": {last_date}},
                coverage_pct_by_code={"850001": 100.0},
                errors=[],
                quality={"already_current": True},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            payload = heat.build_sw3_industry_heat_report(
                _segments(),
                history_fetcher=lambda code: histories[code],
                latest_amount_fetcher=duplicate_latest,
                output_file=Path(tmpdir) / "heat.json",
                max_workers=1,
                retries=1,
                min_eligible_coverage=1.0,
                analysis_fetcher=lambda *_args: [],
                progress=False,
            )

        row = next(row for row in payload["industries"] if row["segment_code"] == "850001")
        self.assertEqual(row["latest_amount_yi"], histories["850001"][last_date])
        self.assertEqual(row["daily"][-1]["amount_source"], "sws_trend")
        self.assertFalse(row["daily"][-1]["amount_is_derived"])
        self.assertIsNone(row["latest_amount_member_coverage_pct"])
        self.assertEqual(payload["data_quality"]["latest_amount_added_industry_count"], 0)

    def test_injected_build_is_offline_atomic_and_reuses_fresh_cache(self):
        histories = _histories()
        fetch_calls = []

        def fetcher(code):
            fetch_calls.append(code)
            return histories[code]

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "heat.json"
            cache = Path(tmpdir) / "cache.json"
            payload = heat.build_sw3_industry_heat_report(
                _segments(),
                history_fetcher=fetcher,
                output_file=output,
                history_cache_file=cache,
                max_workers=1,
                retries=1,
                min_eligible_coverage=1.0,
                analysis_fetcher=lambda *_args: [],
                progress=False,
            )
            first_disk_payload = json.loads(output.read_text(encoding="utf-8"))

            second_calls = []
            second = heat.build_sw3_industry_heat_report(
                _segments(),
                history_fetcher=lambda code: second_calls.append(code),
                output_file=output,
                history_cache_file=cache,
                max_workers=1,
                retries=1,
                min_eligible_coverage=1.0,
                analysis_fetcher=lambda *_args: [],
                progress=False,
            )

        self.assertEqual(fetch_calls, ["850001", "850002"])
        self.assertEqual(second_calls, [])
        self.assertEqual(payload["as_of_date"], _dates()[-1])
        self.assertEqual(first_disk_payload["as_of_date"], _dates()[-1])
        self.assertEqual(second["data_quality"]["history_cache_reused_count"], 2)

    def test_incomplete_refresh_preserves_previous_report(self):
        histories = _histories()

        def partial_fetcher(code):
            if code == "850002":
                raise RuntimeError("upstream unavailable")
            return histories[code]

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "heat.json"
            cache = Path(tmpdir) / "cache.json"
            output.write_text('{"sentinel": true}', encoding="utf-8")

            with self.assertRaises(heat.IndustryHeatIncompleteError):
                heat.build_sw3_industry_heat_report(
                    _segments(),
                    history_fetcher=partial_fetcher,
                    output_file=output,
                    history_cache_file=cache,
                    max_workers=1,
                    retries=1,
                    min_eligible_coverage=1.0,
                    analysis_fetcher=lambda *_args: [],
                    progress=False,
                )

            disk_payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(disk_payload, {"sentinel": True})

    def test_default_production_source_is_not_skipped_by_fresh_cache(self):
        histories = _histories()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "heat.json"
            cache = Path(tmpdir) / "cache.json"
            heat.build_sw3_industry_heat_report(
                _segments(),
                history_fetcher=lambda code: histories[code],
                output_file=output,
                history_cache_file=cache,
                max_workers=1,
                retries=1,
                min_eligible_coverage=1.0,
                analysis_fetcher=lambda *_args: [],
                progress=False,
            )
            production_calls = []

            def production_fetcher(code):
                production_calls.append(code)
                return histories[code]

            with patch.object(
                heat,
                "fetch_sw3_daily_amount_history",
                side_effect=production_fetcher,
            ), patch.object(
                heat,
                "fetch_akshare_latest_amount_batch",
                return_value=SimpleNamespace(
                    histories={},
                    expected_as_of_date=_dates()[-1],
                    source="akshare_stock_zh_a_spot_component",
                    source_by_code_date={},
                    derived_dates_by_code={},
                    coverage_pct_by_code={},
                    errors=[],
                    quality={"already_current": True},
                ),
            ):
                heat.build_sw3_industry_heat_report(
                    _segments(),
                    output_file=output,
                    history_cache_file=cache,
                    max_workers=1,
                    retries=1,
                    min_eligible_coverage=1.0,
                    analysis_fetcher=lambda *_args: [],
                    progress=False,
                )

        self.assertEqual(production_calls, ["850001", "850002"])


if __name__ == "__main__":
    unittest.main()
