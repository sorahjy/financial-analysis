import unittest
from datetime import datetime
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import sw3_akshare_latest as latest


def _anchor(date="2026-07-17", quote_time="15:00:03"):
    return {"date": date, "time": quote_time}


def _segments():
    return [
        {
            "segment_code": "850001",
            "segment_name": "行业甲",
            "member_count": 2,
            "members": [{"code": "000001"}, {"code": "sh600000"}],
        },
        {
            "segment_code": "850002",
            "segment_name": "行业乙",
            "member_count": 1,
            "members": [{"code": "000002"}],
        },
    ]


def _spot_rows():
    return [
        {"代码": "000001", "成交额": 100_000_000},
        {"代码": "600000.SH", "成交额": "200,000,000"},
        {"代码": 2, "成交额": 50_000_000},
    ]


class FakeFrame:
    def __init__(self, rows):
        self.rows = rows

    def to_dict(self, orient):
        if orient != "records":
            raise AssertionError(orient)
        return list(self.rows)


def _build(segments, **kwargs):
    kwargs.setdefault("prior_as_of_date", "2026-07-16")
    kwargs.setdefault(
        "trade_calendar_fetcher",
        lambda: ["2026-07-15", "2026-07-16", "2026-07-17"],
    )
    kwargs.setdefault("cache_file", None)
    return latest.build_akshare_latest_amount_batch(segments, **kwargs)


class AkShareLatestAmountTest(unittest.TestCase):
    def test_happy_path_fetches_snapshot_once_and_aggregates_yuan_to_yi(self):
        calls = {"anchor": 0, "spot": 0}

        def anchor_fetcher():
            calls["anchor"] += 1
            return _anchor()

        def spot_fetcher():
            calls["spot"] += 1
            return FakeFrame(_spot_rows())

        batch = _build(
            _segments(),
            anchor_fetcher=anchor_fetcher,
            spot_fetcher=spot_fetcher,
            now=datetime(2026, 7, 18, 10, 0),
            min_market_rows=3,
            min_segment_coverage=1.0,
        )

        self.assertEqual(calls, {"anchor": 2, "spot": 1})
        self.assertEqual(batch.trade_date, "2026-07-17")
        self.assertEqual(batch.expected_as_of_date, "2026-07-17")
        self.assertEqual(batch.source, latest.SOURCE_NAME)
        self.assertEqual(batch.histories["850001"], {"2026-07-17": 3.0})
        self.assertEqual(batch.histories["850002"], {"2026-07-17": 0.5})
        self.assertEqual(batch.coverage_pct_by_code, {"850001": 100.0, "850002": 100.0})
        self.assertEqual(
            batch.source_by_code_date["850001"]["2026-07-17"],
            latest.SOURCE_NAME,
        )
        self.assertEqual(batch.derived_dates_by_code["850001"], {"2026-07-17"})
        self.assertEqual(batch.quality["snapshot_row_count"], 3)
        self.assertEqual(batch.quality["trade_day_gap"], 1)
        self.assertTrue(batch.quality["snapshot_called"])
        self.assertTrue(batch.quality["snapshot_valid"])
        self.assertTrue(batch.quality["anchor_confirmed"])
        self.assertEqual(batch.quality["anchor_confirmation"], "before_and_after_snapshot")
        self.assertEqual(batch.errors, [])

    def test_same_trade_date_never_calls_market_snapshot(self):
        spot_calls = []
        batch = _build(
            _segments(),
            prior_as_of_date="2026-07-17",
            anchor_fetcher=lambda: _anchor(),
            spot_fetcher=lambda: spot_calls.append(True),
            now=datetime(2026, 7, 18, 10, 0),
        )

        self.assertEqual(spot_calls, [])
        self.assertEqual(batch.histories, {})
        self.assertEqual(batch.expected_as_of_date, "2026-07-17")
        self.assertTrue(batch.quality["already_current"])
        self.assertFalse(batch.quality["snapshot_required"])
        self.assertFalse(batch.quality["snapshot_called"])

    def test_gap_over_one_trade_day_fails_before_market_snapshot(self):
        spot_calls = []
        batch = _build(
            _segments(),
            prior_as_of_date="2026-07-15",
            anchor_fetcher=lambda: _anchor(),
            spot_fetcher=lambda: spot_calls.append(True),
            now=datetime(2026, 7, 18, 10, 0),
        )

        self.assertEqual(spot_calls, [])
        self.assertEqual(batch.histories, {})
        self.assertEqual(batch.expected_as_of_date, "2026-07-17")
        self.assertEqual(batch.errors[0]["stage"], "trade_gap")
        self.assertIn("相差 2 个交易日", batch.errors[0]["error"])

    def test_cache_hit_for_same_date_and_membership_skips_market_snapshot(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "latest.json"
            first = _build(
                _segments(),
                anchor_fetcher=lambda: _anchor(),
                spot_fetcher=lambda: calls.append("first") or _spot_rows(),
                now=datetime(2026, 7, 18, 10, 0),
                min_market_rows=3,
                cache_file=cache,
            )
            second = _build(
                _segments(),
                anchor_fetcher=lambda: _anchor(),
                spot_fetcher=lambda: calls.append("second") or _spot_rows(),
                now=datetime(2026, 7, 18, 10, 0),
                min_market_rows=3,
                cache_file=cache,
            )
            temporary_files = list(cache.parent.glob(f".{cache.name}.*.tmp"))

        self.assertEqual(calls, ["first"])
        self.assertFalse(first.quality["cache_hit"])
        self.assertTrue(second.quality["cache_hit"])
        self.assertFalse(second.quality["snapshot_called"])
        self.assertTrue(second.quality["anchor_confirmed"])
        self.assertEqual(
            second.quality["anchor_confirmation"],
            "cache_snapshot_date_matches_current_anchor",
        )
        self.assertEqual(second.histories, first.histories)
        self.assertEqual(second.derived_dates_by_code, first.derived_dates_by_code)
        self.assertEqual(temporary_files, [])

    def test_stricter_quality_policy_invalidates_cached_batch(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "latest.json"
            _build(
                _segments(),
                anchor_fetcher=lambda: _anchor(),
                spot_fetcher=lambda: calls.append("loose") or _spot_rows(),
                now=datetime(2026, 7, 18, 10, 0),
                min_market_rows=3,
                min_segment_coverage=0.5,
                min_weight_coverage=0.5,
                cache_file=cache,
            )
            strict = _build(
                _segments(),
                anchor_fetcher=lambda: _anchor(),
                spot_fetcher=lambda: calls.append("strict") or _spot_rows(),
                now=datetime(2026, 7, 18, 10, 0),
                min_market_rows=3,
                min_segment_coverage=1.0,
                min_weight_coverage=0.95,
                cache_file=cache,
            )

        self.assertEqual(calls, ["loose", "strict"])
        self.assertFalse(strict.quality["cache_hit"])
        self.assertEqual(strict.quality["minimum_segment_coverage_pct"], 100.0)

    def test_snapshot_is_rejected_if_anchor_date_changes_during_pagination(self):
        anchors = [_anchor(), _anchor(date="2026-07-20")]
        spot_calls = []
        batch = _build(
            _segments(),
            anchor_fetcher=lambda: anchors.pop(0),
            spot_fetcher=lambda: spot_calls.append(True) or _spot_rows(),
            now=datetime(2026, 7, 20, 16, 0),
            min_market_rows=3,
        )

        self.assertEqual(spot_calls, [True])
        self.assertEqual(batch.histories, {})
        self.assertEqual(batch.errors[0]["stage"], "date_anchor_confirm")
        self.assertIn("锚点变化", batch.errors[0]["error"])
        self.assertTrue(batch.quality["snapshot_called"])

    def test_membership_change_invalidates_latest_day_cache(self):
        calls = []
        changed = _segments()
        changed[0]["member_count"] = 3
        changed[0]["members"].append({"code": "000002"})
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "latest.json"
            _build(
                _segments(),
                anchor_fetcher=lambda: _anchor(),
                spot_fetcher=lambda: calls.append("first") or _spot_rows(),
                now=datetime(2026, 7, 18, 10, 0),
                min_market_rows=3,
                cache_file=cache,
            )
            second = _build(
                changed,
                anchor_fetcher=lambda: _anchor(),
                spot_fetcher=lambda: calls.append("changed") or _spot_rows(),
                now=datetime(2026, 7, 18, 10, 0),
                min_market_rows=3,
                cache_file=cache,
            )

        self.assertEqual(calls, ["first", "changed"])
        self.assertFalse(second.quality["cache_hit"])
        self.assertEqual(second.histories["850001"]["2026-07-17"], 3.5)

    def test_weight_change_invalidates_latest_day_cache_even_when_sum_is_unchanged(self):
        calls = []
        original = [_segments()[0]]
        original[0]["members"][0]["index_weight"] = 50.0
        original[0]["members"][1]["index_weight"] = 50.0
        changed = [_segments()[0]]
        changed[0]["members"][0]["index_weight"] = 51.0
        changed[0]["members"][1]["index_weight"] = 49.0
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "latest.json"
            _build(
                original,
                anchor_fetcher=lambda: _anchor(),
                spot_fetcher=lambda: calls.append("original") or _spot_rows(),
                now=datetime(2026, 7, 18, 10, 0),
                min_market_rows=3,
                cache_file=cache,
            )
            second = _build(
                changed,
                anchor_fetcher=lambda: _anchor(),
                spot_fetcher=lambda: calls.append("changed") or _spot_rows(),
                now=datetime(2026, 7, 18, 10, 0),
                min_market_rows=3,
                cache_file=cache,
            )

        self.assertEqual(calls, ["original", "changed"])
        self.assertFalse(second.quality["cache_hit"])

    def test_missing_any_locally_stored_member_skips_that_segment(self):
        rows = [
            {"代码": "000001", "成交额": 100_000_000},
            {"代码": "000002", "成交额": 50_000_000},
            {"代码": "000003", "成交额": 25_000_000},
        ]
        batch = _build(
            _segments(),
            anchor_fetcher=lambda: _anchor(),
            spot_fetcher=lambda: rows,
            now=datetime(2026, 7, 18, 10, 0),
            min_market_rows=3,
        )

        self.assertNotIn("850001", batch.histories)
        self.assertIn("850002", batch.histories)
        self.assertEqual(batch.coverage_pct_by_code["850001"], 50.0)
        quality = batch.quality["segment_quality_by_code"]["850001"]
        self.assertEqual(quality["snapshot_coverage_pct"], 50.0)
        self.assertFalse(quality["snapshot_complete"])
        self.assertFalse(quality["aggregated"])
        self.assertTrue(any(
            error["stage"] == "segment_snapshot_coverage"
            and "600000" in error["error"]
            for error in batch.errors
        ))

    def test_complete_official_weights_can_pass_with_partial_declared_count(self):
        segment = {
            "segment_code": "850001",
            "segment_name": "行业甲",
            "member_count": 10,
            "members": [
                {"code": "000001", "official_market_cap_ratio": 55.0},
                {"code": "600000", "index_weight": 41.0},
            ],
        }
        batch = _build(
            [segment],
            anchor_fetcher=lambda: _anchor(),
            spot_fetcher=lambda: _spot_rows(),
            now=datetime(2026, 7, 18, 10, 0),
            min_market_rows=3,
        )

        self.assertEqual(batch.histories["850001"], {"2026-07-17": 3.0})
        self.assertEqual(batch.coverage_pct_by_code["850001"], 96.0)
        quality = batch.quality["segment_quality_by_code"]["850001"]
        self.assertEqual(quality["membership_gate"], "weight")
        self.assertEqual(quality["membership_count_coverage_pct"], 20.0)
        self.assertEqual(quality["membership_weight_sum_pct"], 96.0)
        self.assertTrue(quality["membership_complete"])
        self.assertTrue(quality["snapshot_complete"])
        self.assertTrue(quality["aggregated"])

    def test_low_complete_weight_sum_skips_segment_without_scaling(self):
        segment = {
            "segment_code": "850001",
            "segment_name": "行业甲",
            "member_count": 2,
            "members": [
                {"code": "000001", "official_market_cap_ratio": 50.0},
                {"code": "600000", "index_weight": 44.0},
            ],
        }
        batch = _build(
            [segment],
            anchor_fetcher=lambda: _anchor(),
            spot_fetcher=lambda: _spot_rows(),
            now=datetime(2026, 7, 18, 10, 0),
            min_market_rows=3,
        )

        self.assertNotIn("850001", batch.histories)
        self.assertEqual(batch.coverage_pct_by_code["850001"], 94.0)
        quality = batch.quality["segment_quality_by_code"]["850001"]
        self.assertEqual(quality["membership_gate"], "weight")
        self.assertEqual(quality["membership_weight_sum_pct"], 94.0)
        self.assertFalse(quality["membership_complete"])
        self.assertFalse(quality["aggregated"])
        self.assertTrue(any(
            error["stage"] == "segment_coverage" and "94.00%" in error["error"]
            for error in batch.errors
        ))

    def test_snapshot_is_not_called_when_anchor_is_intraday(self):
        spot_calls = []
        batch = _build(
            _segments(),
            anchor_fetcher=lambda: _anchor(quote_time="14:59:59"),
            spot_fetcher=lambda: spot_calls.append(True),
            now=datetime(2026, 7, 17, 15, 5),
            min_market_rows=1,
        )

        self.assertEqual(spot_calls, [])
        self.assertEqual(batch.histories, {})
        self.assertFalse(batch.quality["snapshot_valid"])
        self.assertEqual(batch.errors[0]["stage"], "date_anchor")
        self.assertIn("尚未收盘", batch.errors[0]["error"])

    def test_weekend_anchor_is_rejected(self):
        batch = _build(
            _segments(),
            anchor_fetcher=lambda: _anchor(date="2026-07-18"),
            spot_fetcher=lambda: _spot_rows(),
            now=datetime(2026, 7, 18, 16, 0),
            min_market_rows=3,
        )

        self.assertEqual(batch.histories, {})
        self.assertIn("非交易日", batch.errors[0]["error"])

    def test_stale_anchor_is_rejected(self):
        batch = _build(
            _segments(),
            anchor_fetcher=lambda: _anchor(date="2026-06-01"),
            spot_fetcher=lambda: _spot_rows(),
            now=datetime(2026, 7, 18, 16, 0),
            max_anchor_age_days=14,
            min_market_rows=3,
        )

        self.assertEqual(batch.histories, {})
        self.assertIn("过旧", batch.errors[0]["error"])

    def test_truncated_market_snapshot_is_rejected(self):
        batch = _build(
            _segments(),
            anchor_fetcher=lambda: _anchor(),
            spot_fetcher=lambda: _spot_rows(),
            now=datetime(2026, 7, 18, 10, 0),
            min_market_rows=4,
        )

        self.assertEqual(batch.histories, {})
        self.assertEqual(batch.errors[0]["stage"], "market_snapshot")
        self.assertIn("疑似截断", batch.errors[0]["error"])

    def test_duplicate_market_code_rejects_whole_snapshot(self):
        rows = _spot_rows() + [{"代码": "sz000001", "成交额": 1}]
        batch = _build(
            _segments(),
            anchor_fetcher=lambda: _anchor(),
            spot_fetcher=lambda: rows,
            now=datetime(2026, 7, 18, 10, 0),
            min_market_rows=3,
        )

        self.assertEqual(batch.histories, {})
        self.assertIn("不唯一", batch.errors[0]["error"])

    def test_negative_or_non_finite_amount_rejects_whole_snapshot(self):
        for value in (-1, float("nan")):
            with self.subTest(value=value):
                rows = _spot_rows()
                rows[0] = {"代码": "000001", "成交额": value}
                batch = _build(
                    _segments(),
                    anchor_fetcher=lambda: _anchor(),
                    spot_fetcher=lambda rows=rows: rows,
                    now=datetime(2026, 7, 18, 10, 0),
                    min_market_rows=3,
                )
                self.assertEqual(batch.histories, {})
                self.assertIn("无效", batch.errors[0]["error"])

    def test_low_segment_coverage_skips_only_that_segment(self):
        segments = _segments()
        segments[0]["member_count"] = 3
        batch = _build(
            segments,
            anchor_fetcher=lambda: _anchor(),
            spot_fetcher=lambda: _spot_rows(),
            now=datetime(2026, 7, 18, 10, 0),
            min_market_rows=3,
            min_segment_coverage=0.8,
        )

        self.assertNotIn("850001", batch.histories)
        self.assertIn("850002", batch.histories)
        self.assertEqual(batch.coverage_pct_by_code["850001"], 66.67)
        self.assertTrue(any(error["stage"] == "segment_coverage" for error in batch.errors))

    def test_duplicate_members_are_not_double_counted(self):
        segment = {
            "segment_code": "850001",
            "member_count": 1,
            "members": [{"code": "000001"}, {"code": "sz000001"}],
        }
        batch = _build(
            [segment],
            anchor_fetcher=lambda: _anchor(),
            spot_fetcher=lambda: _spot_rows(),
            now=datetime(2026, 7, 18, 10, 0),
            min_market_rows=3,
            min_segment_coverage=1.0,
        )

        self.assertEqual(batch.histories["850001"]["2026-07-17"], 1.0)
        self.assertEqual(batch.coverage_pct_by_code["850001"], 100.0)

    def test_sina_quote_parser_reads_date_and_time(self):
        fields = ["浦发银行"] + ["0"] * 29 + ["2026-07-17", "15:00:03", "00"]
        text = f'var hq_str_sh600000="{",".join(fields)}";'
        self.assertEqual(
            latest.parse_sina_quote_anchor(text),
            {"symbol": "sh600000", "date": "2026-07-17", "time": "15:00:03"},
        )

    def test_akshare_import_is_lazy_and_fetch_called_once(self):
        fake_fetch_calls = []
        fake_module = SimpleNamespace(
            stock_zh_a_spot=lambda: fake_fetch_calls.append(True) or _spot_rows()
        )
        with patch.object(latest.importlib, "import_module", return_value=fake_module) as importer:
            rows = latest.fetch_akshare_a_spot()

        importer.assert_called_once_with("akshare")
        self.assertEqual(fake_fetch_calls, [True])
        self.assertEqual(rows, _spot_rows())

    def test_spot_exception_becomes_visible_error(self):
        def fail():
            raise RuntimeError("sina blocked")

        batch = _build(
            _segments(),
            anchor_fetcher=lambda: _anchor(),
            spot_fetcher=fail,
            now=datetime(2026, 7, 18, 10, 0),
            min_market_rows=3,
        )

        self.assertEqual(batch.histories, {})
        self.assertIn("sina blocked", batch.errors[0]["error"])


if __name__ == "__main__":
    unittest.main()
