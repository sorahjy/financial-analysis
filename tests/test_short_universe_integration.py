import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import stock_advanced_strategies as strategies
import stock_storage


class ShortUniverseIntegrationTest(unittest.TestCase):
    def setUp(self):
        strategies.invalidate_dir_fingerprints()

    def tearDown(self):
        strategies.invalidate_dir_fingerprints()

    def test_strategy_base_candidates_equal_radar_hotmoney_members(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "stock.sqlite3"
            conn = stock_storage.connect(db_path)
            try:
                stock_storage.save_sw3_membership(conn, {
                    "segments": [{
                        "segment_code": "850111",
                        "segment_name": "测试三级",
                        "parent_segment": "测试二级",
                        "members": [
                            {"code": "000001", "name": "成员甲"},
                            {"code": "000002", "name": "成员乙"},
                            {"code": "000003", "name": "非池成员"},
                        ],
                    }],
                    "errors": [],
                })
                conn.execute(
                    "UPDATE sw3_member SET is_hot_money = 1 WHERE code IN ('000001', '000002')"
                )
                conn.commit()
                stock_storage.replace_short_signals(
                    conn,
                    {
                        "000001": {
                            "name": "成员甲",
                            "followers": [],
                            "signals": {"lhb": {"count": 2}, "tech": {}},
                        }
                    },
                    generated_at="2026-07-10 10:00:00",
                    as_of_date="2026-07-10",
                )
                radar_codes = {
                    item["code"] for item in stock_storage.pool_members(conn, "hotmoney")
                }
            finally:
                conn.close()

            real_connect = stock_storage.connect

            def connect_test_db(*_args, **_kwargs):
                return real_connect(db_path)

            config = copy.deepcopy(strategies.DEFAULT_CONFIG["short"])
            config.update({
                "exclude_st": False,
                "min_lhb_count": 0,
                "min_hot_money_concurrent": 0,
                "max_consecutive_limit_up": 99,
            })
            with patch.object(strategies.stock_storage, "connect", side_effect=connect_test_db), \
                 patch.object(
                     strategies.stock_storage,
                     "sw3_signature",
                     return_value={"test": "short-universe"},
                 ):
                candidates, notes = strategies.build_short_candidates(config)

        self.assertEqual({item["code"] for item in candidates}, radar_codes)
        self.assertTrue(any("1/2" in note for note in notes))
        missing_signal = next(item for item in candidates if item["code"] == "000002")
        self.assertEqual(missing_signal["sources"], ["hotmoney_universe"])


if __name__ == "__main__":
    unittest.main()
