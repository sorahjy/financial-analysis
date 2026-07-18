"""刷新游资雷达 ETF 池的前复权行情。

ETF 只抓技术分析需要的 OHLCV、成交额、涨跌幅和换手率；不会触发财报、
估值、质押、股东户数、回购或龙虎榜接口。
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

import akshare as ak

from stock_crawl_common import (
    fetch_etf_qfq_daily_records,
    safe_print,
    write_json_file,
)
from stock_crawl_price_valuation import refresh_stock_histories
from stock_etf_pool import load_etf_pool


DEFAULT_YEARS = 5.0
DEFAULT_WORKERS = 6
REPORT_FILE = Path(__file__).resolve().parent / "data" / "etf_pool_refresh_report.json"


def fetch_etf_spot_metadata(
    fetcher: Optional[Callable[[], Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """读取 ETF 主清单，用于名称更新和严格品种校验。

    默认按东财 → 同花顺 → 新浪故障转移；测试或调用方显式传入 ``fetcher``
    时只使用该数据源，不会静默换源。
    """
    sources: List[tuple[str, Callable[[], Any]]]
    if fetcher is not None:
        sources = [(getattr(fetcher, "__name__", "custom"), fetcher)]
    else:
        sources = [
            ("eastmoney", ak.fund_etf_spot_em),
            ("ths", ak.fund_etf_spot_ths),
            ("sina", lambda: ak.fund_etf_category_sina(symbol="ETF基金")),
        ]
    frame = None
    source = ""
    errors: List[str] = []
    for label, source_fetcher in sources:
        try:
            candidate = source_fetcher()
            if candidate is None or getattr(candidate, "empty", True):
                errors.append(f"{label}: empty")
                continue
            frame = candidate
            source = label
            break
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            if fetcher is not None:
                raise
            safe_print(f"[WARN] ETF 主清单源 {label} 不可用，尝试备用源: {exc}")
    if frame is None:
        if errors:
            raise RuntimeError("ETF 主清单源全部失败：" + " | ".join(errors))
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    for _, item in frame.iterrows():
        raw_code = str(item.get("代码") or item.get("基金代码") or "").strip()
        matched = re.search(r"(\d{6})$", raw_code)
        if not matched:
            continue
        code = matched.group(1)
        rows[code] = {
            "code": code,
            "name": str(item.get("名称") or item.get("基金名称") or "").strip(),
            "price": item.get("最新价", item.get("当前-单位净值")),
            "iopv": item.get("IOPV实时估值"),
            "discount_rate": item.get("基金折价率"),
            "shares": item.get("最新份额"),
            "validation_source": source,
        }
    return rows


def _etf_daily_fetcher(code: str, start_date: str, end_date: str):
    return fetch_etf_qfq_daily_records(
        code,
        start_date,
        end_date,
        include_trading_value=True,
        warn=lambda message: safe_print(f"  [ETF-FALLBACK] {message}"),
    )


def resolve_etf_rows(
    configured: Iterable[Dict[str, Any]],
    spot_map: Optional[Dict[str, Dict[str, Any]]],
) -> tuple[List[Dict[str, Any]], List[str]]:
    """用交易所 ETF 主清单校验配置；传 ``None`` 表示用户显式关闭校验。"""
    resolved: List[Dict[str, Any]] = []
    invalid: List[str] = []
    for item in configured:
        code = str(item["code"])
        if spot_map is not None and code not in spot_map:
            invalid.append(code)
            continue
        market = (spot_map or {}).get(code) or {}
        resolved.append({
            **item,
            "name": str(market.get("name") or item.get("name") or f"ETF {code}"),
        })
    return resolved, invalid


def refresh_etf_pool(
    *,
    years: float = DEFAULT_YEARS,
    workers: int = DEFAULT_WORKERS,
    validate_spot: bool = True,
    spot_fetcher: Optional[Callable[[], Any]] = None,
    history_refresher: Callable[..., Dict[str, Any]] = refresh_stock_histories,
) -> Dict[str, Any]:
    configured = load_etf_pool()
    spot_map: Optional[Dict[str, Dict[str, Any]]] = None
    spot_error = ""
    if validate_spot:
        try:
            fetched = fetch_etf_spot_metadata(spot_fetcher)
            if not fetched:
                raise RuntimeError("ETF 主清单返回空数据")
            spot_map = fetched
        except Exception as exc:
            spot_error = str(exc)
            report = {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "validation_failed",
                "configured_count": len(configured),
                "validated_count": 0,
                "validated_codes": [],
                "spot_validation_available": False,
                "spot_validation_error": spot_error,
                "invalid_codes": [],
                "skipped_company_data": [
                    "financials", "financial_indicators", "dividends", "valuation",
                    "pledge", "shareholder_count", "repurchase", "lhb", "sw3_industry",
                ],
                "refresh": None,
            }
            write_json_file(REPORT_FILE, report)
            raise RuntimeError(
                f"ETF 主清单校验失败，已停止刷新；如确需绕过请显式使用 --no-validate-spot: {exc}"
            ) from exc

    rows, invalid_codes = resolve_etf_rows(configured, spot_map)
    stocks = {row["code"]: row["name"] for row in rows}
    safe_print(
        f"ETF 池配置 {len(configured)} 只，待刷新 {len(stocks)} 只"
        + (f"，非 ETF/已失效 {len(invalid_codes)} 只" if invalid_codes else "")
    )
    result = history_refresher(
        stocks,
        max_years=max(float(years), 1.0),
        workers=max(int(workers), 1),
        refresh_valuation=False,
        fundamentals_plan={},
        pledge_map={},
        label="ETF池",
        daily_fetcher=_etf_daily_fetcher,
        instrument_type="etf",
    )
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "partial" if result.get("failed") else "ok",
        "configured_count": len(configured),
        "validated_count": len(stocks),
        "validated_codes": list(stocks),
        "spot_validation_available": spot_map is not None,
        "spot_validation_sources": sorted({
            str(item.get("validation_source") or "unknown")
            for item in (spot_map or {}).values()
        }),
        "spot_validation_error": spot_error or None,
        "invalid_codes": invalid_codes,
        "skipped_company_data": [
            "financials", "financial_indicators", "dividends", "valuation",
            "pledge", "shareholder_count", "repurchase", "lhb", "sw3_industry",
        ],
        "refresh": result,
    }
    write_json_file(REPORT_FILE, report)
    safe_print(f"ETF 刷新报告: {REPORT_FILE}")
    return report


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="刷新游资雷达 ETF 池行情")
    parser.add_argument("--years", type=float, default=DEFAULT_YEARS, help="历史年数，默认5年")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并发数，默认6")
    parser.add_argument(
        "--validate-spot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否用交易所 ETF 主清单严格校验代码（默认是；失败即停止）",
    )
    args = parser.parse_args()
    return refresh_etf_pool(
        years=args.years,
        workers=args.workers,
        validate_spot=args.validate_spot,
    )


if __name__ == "__main__":
    main()
