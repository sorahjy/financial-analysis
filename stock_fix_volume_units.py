"""一次性迁移：把 stock_history.daily_volume 统一归一到「手」(1手=100股)。

背景：成交量来自 3 个源——新浪 stock_zh_a_daily 返回「股」、东财/腾讯返回「手」，
旧爬虫原样入库导致全库约一半票是股、一半是手(差 100×)。`daily_amount` 三源均为元、可靠，
故按「成交额/收盘 ≈ 成交均价」逐行判定单位：若 |amount/vol − close| < |amount/(vol*100) − close|
则该行 volume 现为「股」→ ÷100 折成「手」。已是「手」的行不动，故脚本可重复运行(幂等)。

用法：
  python stock_fix_volume_units.py            # dry-run，只报告将改动多少行 + 样例
  python stock_fix_volume_units.py --apply     # 落库
"""

from __future__ import annotations

import argparse
import sqlite3

import stock_storage


# 按股判定的行：amount/vol 比 amount/(vol*100) 更接近收盘价 → 当前单位是「股」，需 ÷100。
SHARE_ROW_PREDICATE = (
    "daily_volume IS NOT NULL AND daily_volume > 0 "
    "AND daily_amount IS NOT NULL AND daily_amount > 0 "
    "AND daily_close IS NOT NULL AND daily_close > 0 "
    "AND ABS(daily_amount / daily_volume - daily_close) "
    "  < ABS(daily_amount / (daily_volume * 100.0) - daily_close)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="统一 daily_volume 单位到「手」")
    parser.add_argument("--apply", action="store_true", help="实际写库（默认仅 dry-run）")
    args = parser.parse_args()

    conn = sqlite3.connect(stock_storage.DEFAULT_DB_FILE)
    try:
        total = conn.execute("SELECT COUNT(*) FROM stock_history").fetchone()[0]
        share_rows = conn.execute(
            f"SELECT COUNT(*) FROM stock_history WHERE {SHARE_ROW_PREDICATE}"
        ).fetchone()[0]
        share_codes = conn.execute(
            f"SELECT COUNT(DISTINCT code) FROM stock_history WHERE {SHARE_ROW_PREDICATE}"
        ).fetchone()[0]
        print(f"全库行数: {total}")
        print(f"判定为「股」需折算的行: {share_rows}（涉及 {share_codes} 只股票）")

        examples = conn.execute(
            "SELECT code, date, daily_close, daily_volume, daily_amount FROM stock_history "
            f"WHERE {SHARE_ROW_PREDICATE} ORDER BY date DESC LIMIT 5"
        ).fetchall()
        print("样例（折算前 → 折算后/手）:")
        for code, date, c, v, a in examples:
            print(f"  {code} {date} 收{c} 量{v:.0f}股 → {v/100:.2f}手  (额{a:.0f}元, 额/量={a/v:.2f}≈收盘)")

        if not args.apply:
            print("\n[dry-run] 未写库。确认无误后加 --apply 落库。")
            return

        cur = conn.execute(
            f"UPDATE stock_history SET daily_volume = daily_volume / 100.0 WHERE {SHARE_ROW_PREDICATE}"
        )
        conn.commit()
        print(f"\n[applied] 已折算 {cur.rowcount} 行 → 单位统一为「手」。")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
