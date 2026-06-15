"""
爬取三指数候选池全部股票数据

两种模式:
  --mode full    全量爬取: 中证全指+全A股合并 → 逐只爬取详细数据
  --mode staged  分步粗筛(默认): 先获取全市场快照做粗筛, 再只爬通过粗筛的股票

爬取内容（数据获取底层函数见本文件「单股数据获取」段，原 stock_fetch_data.py 已并入）：
  - 日频行情、财务报表、财务指标、分红历史、质押比例

特性：
  - 增量爬取：已爬过的股票自动跳过
  - 每只股票爬完立即写盘，断点可续
  - 质押数据全量获取一次，避免重复请求
  - 失败的股票记录到 errors，不影响后续
"""

import os
os.environ["TQDM_DISABLE"] = "1"

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

import stock_storage as ss
from stock_crawl_common import (
    strip_proxy_env,
    safe_print,
    safe_float as _safe_float,
    retry_fetch as _retry_fetch,
    is_bse_stock,
    fetch_index_constituents,
    fetch_qfq_daily_records,
    daily_payload_from_history_records,
    daily_stats_from_history_records,
    history_payload_from_records,
    latest_weekday_date,
    merge_records_by_date,
    normalize_history_records,
)

DATA_DIR = Path(__file__).parent / "data"
CONS_FILE = DATA_DIR / "index_constituents.json"
SNAPSHOT_FILE = DATA_DIR / "market_snapshot.json"
UNIVERSE_FILE = DATA_DIR / "stock_universe.json"

# 财报历史深度：10 年。PIT 回测要 join 多年日线，财报需同样深。
FINANCIAL_YEARS = 10
FINANCIAL_QUARTERS = FINANCIAL_YEARS * 4
STAGE_TIMING_REPORT_EVERY = 100
STAGE_TIMING_ORDER = ("daily", "financials", "indicators", "dividends", "save", "total")


# 科技主题行业关键词（中证二级/三级行业），用于科技100指数筛选科技行业股票
TECH_KEYWORDS = [
    "航空航天", "国防", "电网设备", "储能", "光伏", "风电",
    "医疗器械", "生物", "药品", "动物保健", "育种", "化学药", "制药",
    "数字媒体", "数字营销", "软件", "电子", "半导体", "通信",
    "计算机", "信息技术", "互联网", "人工智能", "芯片", "集成电路",
    "新能源", "科技", "IT服务",
]


# ═══════════════════════════════════════════════════════════
# 单股数据获取（原 stock_fetch_data.py 并入）
# ═══════════════════════════════════════════════════════════



def _df_to_records(df, cols_map, date_col="报告日"):
    """从 DataFrame 中提取指定列，转为 list of dict。
    cols_map 的值可以是字符串或字符串列表（按优先级尝试多个列名）。
    """
    records = []
    for _, row in df.iterrows():
        rec = {}
        date_val = str(row[date_col])
        # 将 '20251231' 格式转为 '2025-12-31'
        if len(date_val) == 8 and date_val.isdigit():
            rec["date"] = f"{date_val[:4]}-{date_val[4:6]}-{date_val[6:8]}"
        else:
            rec["date"] = str(date_val)[:10]
        for new_key, old_col in cols_map.items():
            candidates = old_col if isinstance(old_col, list) else [old_col]
            val = None
            for col_name in candidates:
                if col_name in df.columns:
                    val = _safe_float(row[col_name])
                    break
            rec[new_key] = val
        records.append(rec)
    return records


def fetch_daily_price(symbol, years=1):
    """获取近 N 年前复权日频行情，并派生 daily.stats。"""
    end_date = latest_weekday_date()
    start_date = (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365 * years)
    ).strftime("%Y-%m-%d")
    rows = fetch_qfq_daily_records(
        symbol,
        start_date,
        end_date,
        include_trading_value=True,
        warn=lambda message: safe_print(f"  [FALLBACK] {message}"),
    )
    if not rows:
        raise RuntimeError(f"{symbol} 日线为空")

    history_records = normalize_history_records(rows, include_valuation=False)
    return {
        "stats": daily_stats_from_history_records(history_records),
        "history_records": history_records,
    }


def fetch_financial_reports(stock_code):
    """获取三大财务报表（利润表、资产负债表、现金流量表）"""
    result = {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        income_future = pool.submit(
            _retry_fetch, ak.stock_financial_report_sina, stock=stock_code, symbol="利润表"
        )
        balance_future = pool.submit(
            _retry_fetch, ak.stock_financial_report_sina, stock=stock_code, symbol="资产负债表"
        )
        cashflow_future = pool.submit(
            _retry_fetch, ak.stock_financial_report_sina, stock=stock_code, symbol="现金流量表"
        )

        df_income = income_future.result()
        df_balance = balance_future.result()
        df_cashflow = cashflow_future.result()

    income_cols = {
        "revenue": ["营业总收入", "营业收入"],  # 银行股用"营业收入"
        "operating_cost": ["营业总成本", "营业支出"],  # 银行股用"营业支出"
        "cost_of_revenue": "营业成本",
        "rd_expense": "研发费用",
        "net_profit": "净利润",
        "net_profit_parent": ["归属于母公司所有者的净利润", "归属于母公司的净利润"],
    }
    result["income"] = _df_to_records(df_income, income_cols)[:FINANCIAL_QUARTERS]  # 近10年(40个季度)

    balance_cols = {
        "total_equity_parent": ["归属于母公司股东权益合计", "归属于母公司股东的权益"],
        "minority_equity": "少数股东权益",
        "total_equity": ["所有者权益(或股东权益)合计", "负债及股东权益总计"],
        "total_assets_liabilities": ["负债和所有者权益(或股东权益)总计", "负债及股东权益总计"],
        "dev_expenditure": "开发支出",
    }
    result["balance"] = _df_to_records(df_balance, balance_cols)[:FINANCIAL_QUARTERS]

    cashflow_cols = {
        "operating_cashflow_net": "经营活动产生的现金流量净额",
        "operating_cashflow_in": "经营活动现金流入小计",
        "operating_cashflow_out": "经营活动现金流出小计",
    }
    result["cashflow"] = _df_to_records(df_cashflow, cashflow_cols)[:FINANCIAL_QUARTERS]

    return result


def fetch_financial_indicators(symbol):
    """获取财务分析指标（ROE、增长率、每股净资产等）"""
    start_year = str(datetime.now().year - FINANCIAL_YEARS)
    df = _retry_fetch(ak.stock_financial_analysis_indicator, symbol=symbol, start_year=start_year)
    # 按日期降序排列，最新在前
    df = df.iloc[::-1].reset_index(drop=True)

    indicator_cols = {
        "roe": "净资产收益率(%)",
        "roe_weighted": "加权净资产收益率(%)",
        "eps_diluted": "摊薄每股收益(元)",
        "eps_adjusted": "每股收益_调整后(元)",
        "eps_deducted": "扣除非经常性损益后的每股收益(元)",
        "bvps_adjusted": "每股净资产_调整后(元)",
        "ocfps": "每股经营性现金流(元)",
        "gross_margin": "销售毛利率(%)",
        "net_margin": "销售净利率(%)",
        "revenue_growth": "主营业务收入增长率(%)",
        "net_profit_growth": "净利润增长率(%)",
        "net_assets_growth": "净资产增长率(%)",
        "total_assets": "总资产(元)",
        "deducted_net_profit": "扣除非经常性损益后的净利润(元)",
        "dividend_payout_ratio": "股息发放率(%)",
        "asset_liability_ratio": "资产负债率(%)",
    }

    records = _df_to_records(df, indicator_cols, date_col="日期")

    # 提取 ROE 序列（用于计算 ROE 均值-标准差因子）
    roe_series = [r["roe"] for r in records if r["roe"] is not None]
    roe_stats = {}
    if roe_series:
        roe_stats = {
            "mean": _safe_float(np.mean(roe_series)),
            "std": _safe_float(np.std(roe_series)),
            "count": len(roe_series),
        }

    return {"records": records, "roe_stats": roe_stats}


def fetch_dividend_history(symbol):
    """获取分红历史"""
    df = _retry_fetch(ak.stock_history_dividend_detail, symbol=symbol, indicator="分红")

    records = []
    for _, row in df.iterrows():
        rec = {
            "announce_date": str(row["公告日期"])[:10] if pd.notna(row["公告日期"]) else None,
            "bonus_shares": _safe_float(row["送股"]),
            "transfer_shares": _safe_float(row["转增"]),
            "dividend_per_10": _safe_float(row["派息"]),  # 每10股派息(元)
            "progress": str(row["进度"]) if pd.notna(row["进度"]) else None,
            "ex_date": str(row["除权除息日"])[:10] if pd.notna(row["除权除息日"]) else None,
        }
        records.append(rec)

    # 计算近3年是否连续分红
    current_year = datetime.now().year
    yearly_dividends = {}
    for r in records:
        if r["announce_date"] and r["dividend_per_10"] and r["dividend_per_10"] > 0:
            year = int(r["announce_date"][:4])
            if year not in yearly_dividends:
                yearly_dividends[year] = 0
            yearly_dividends[year] += r["dividend_per_10"]

    consecutive_3y = all(
        (current_year - i) in yearly_dividends
        for i in range(1, 4)
    )

    return {
        "records": records[:FINANCIAL_QUARTERS],
        "yearly_dividends": {str(k): _safe_float(v) for k, v in sorted(yearly_dividends.items(), reverse=True)[:11]},
        "consecutive_3y_dividend": consecutive_3y,
    }


# ═══════════════════════════════════════════════════════════
# 公共工具
# ═══════════════════════════════════════════════════════════



def save_stock_universe(csi300_map, csi_all_map):
    """写 data/stock_universe.json, 供 stock_advanced_strategies.py 的
    沪深300硬过滤(require_csi300)与 csi300_current/csi300_persistence 因子使用。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "akshare index_stock_cons_csindex",
        "csi300": sorted(csi300_map),
        "all": sorted(csi_all_map),
    }
    with open(UNIVERSE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  股票池已保存: {UNIVERSE_FILE} (csi300={len(csi300_map)}, all={len(csi_all_map)})")


def fetch_pledge_data_bulk():
    """一次性获取全市场质押数据"""
    print("获取全市场质押数据...")
    try:
        df = _retry_fetch(ak.stock_gpzy_pledge_ratio_em)
        pledge_map = {}
        for _, row in df.iterrows():
            code = str(row["股票代码"]).zfill(6)
            pledge_map[code] = {
                "pledge_ratio": _safe_float(row["质押比例"]),  # 股权质押比例(%), 用于风险筛选
                "pledge_count": int(row["质押笔数"]) if pd.notna(row["质押笔数"]) else None,  # 质押笔数
                "trade_date": str(row["交易日期"])[:10],  # 数据日期
                "industry": str(row["所属行业"]),  # 所属行业(中证行业分类)
            }
        print(f"  质押数据: {len(pledge_map)} 条")
        return pledge_map
    except Exception as e:
        print(f"  质押数据获取失败: {e}")
        return {}


def fetch_one_stock(symbol, name, pledge_map, timing_callback=None):
    """爬取单只股票的全部数据:
    - daily: 日频行情(收盘价/成交量/波动率等)
    - financials: 三大财务报表(利润表/资产负债表/现金流量表)
    - indicators: 财务指标(ROE/增长率/每股净资产/资产负债率等)
    - dividends: 分红历史(每10股派息/是否连续分红)
    - pledge: 股权质押数据(质押比例/质押笔数/所属行业)
    """
    data = {"symbol": symbol, "name": name, "fetch_time": datetime.now().isoformat()}

    def _record_timing(stage, start):
        if timing_callback:
            timing_callback(stage, time.perf_counter() - start)

    def _timed_call(stage, func, *args):
        start = time.perf_counter()
        try:
            return func(*args)
        finally:
            _record_timing(stage, start)

    daily = _timed_call("daily", fetch_daily_price, symbol)
    history_records = daily.pop("history_records", [])
    data["daily"] = daily
    data["history"] = history_payload_from_records(
        symbol, name, history_records, "stock_crawl_fundamentals.daily"
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        financials_future = pool.submit(_timed_call, "financials", fetch_financial_reports, symbol)
        indicators_future = pool.submit(_timed_call, "indicators", fetch_financial_indicators, symbol)
        data["financials"] = financials_future.result()
        data["indicators"] = indicators_future.result()

    data["dividends"] = _timed_call("dividends", fetch_dividend_history, symbol)

    data["pledge"] = pledge_map.get(symbol, {
        "pledge_ratio": None, "pledge_count": None,
        "trade_date": None, "industry": None,
    })

    return data


def load_existing():
    conn = ss.thread_conn()
    return ss.existing_codes(conn)


def save_stock(symbol, data):
    conn = ss.thread_conn()
    symbol = str(symbol).zfill(6)
    existing_records = ss.load_history_records(conn, symbol)
    if existing_records:
        incoming_records = (data.get("history") or {}).get("records", [])
        merged_records = merge_records_by_date(existing_records, incoming_records, overwrite_none=False)
        if merged_records:
            data["history"] = history_payload_from_records(
                symbol,
                data.get("name", symbol),
                merged_records,
                "stock_data.history+stock_crawl_fundamentals.daily",
            )
            data["daily"] = daily_payload_from_history_records(merged_records)
    prev_refetched = ss.history_refetched_at(conn, symbol)
    if prev_refetched and "history_refetched_at" not in data:
        data["history_refetched_at"] = prev_refetched
    ss.save_stock(conn, data)


def crawl_stocks(stocks, pledge_map, limit=0, workers=20):
    """通用爬取逻辑: stocks = {code: {name, ...}}，多线程并发"""
    existing_codes = load_existing()
    existing = len(existing_codes)
    if existing > 0:
        print(f"已有 {existing} 只股票数据，将跳过\n")

    # 筛选待爬列表
    todo = []
    for symbol, info in stocks.items():
        if symbol in existing_codes:
            continue
        todo.append((symbol, info))
    if limit > 0:
        todo = todo[:limit]

    total = existing + len(todo)
    errors = {}
    done_count = [existing]
    processed_count = [0]
    progress_lock = threading.Lock()
    timing_lock = threading.Lock()
    stage_timings = {}

    def _record_stage_timing(stage, elapsed):
        with timing_lock:
            stats = stage_timings.setdefault(stage, {"count": 0, "total": 0.0, "max": 0.0})
            stats["count"] += 1
            stats["total"] += elapsed
            stats["max"] = max(stats["max"], elapsed)

    def _format_stage_timing_summary():
        with timing_lock:
            snapshot = {
                stage: dict(values)
                for stage, values in stage_timings.items()
                if values["count"] > 0
            }
        parts = []
        for stage in STAGE_TIMING_ORDER:
            values = snapshot.get(stage)
            if not values:
                continue
            avg = values["total"] / values["count"]
            parts.append(f"{stage} avg={avg:.2f}s max={values['max']:.2f}s")
        return " | ".join(parts)

    def _maybe_print_stage_timing(processed):
        if not todo:
            return
        if processed % STAGE_TIMING_REPORT_EVERY != 0 and processed != len(todo):
            return
        summary = _format_stage_timing_summary()
        if summary:
            safe_print(f"  [耗时] 已处理 {processed}/{len(todo)}: {summary}")

    def _worker(symbol, info):
        name = info["name"]
        worker_start = time.perf_counter()
        ok = False
        try:
            stock_data = fetch_one_stock(symbol, name, pledge_map, timing_callback=_record_stage_timing)
            stock_data["candidate_for"] = info.get("candidate_for", [])
            start = time.perf_counter()
            save_stock(symbol, stock_data)
            _record_stage_timing("save", time.perf_counter() - start)
            ok = True
        except Exception as e:
            error = e
        finally:
            _record_stage_timing("total", time.perf_counter() - worker_start)
            with progress_lock:
                processed_count[0] += 1
                processed = processed_count[0]
                if ok:
                    done_count[0] += 1
                    done = done_count[0]
                else:
                    errors[symbol] = {"name": name, "error": str(error)}
                    done = done_count[0]
            if ok:
                safe_print(f"  ✓ {name}({symbol}) 完成 ({done}/{total})")
            else:
                safe_print(f"  ✗ {name}({symbol}) 失败: {error}")
            _maybe_print_stage_timing(processed)

    print(f"待爬取: {len(todo)} 只, 线程数: {workers}\n")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, sym, info): sym for sym, info in todo}
        for future in as_completed(futures):
            future.result()

    # 汇总
    print(f"\n{'=' * 60}")
    print(f"爬取完成! 成功: {done_count[0]}/{total}, 失败: {len(errors)}")
    if errors:
        print(f"\n失败列表:")
        for sym, err in errors.items():
            print(f"  {sym} {err['name']}: {err['error']}")
        with open(DATA_DIR / "crawl_errors.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)

    stock_count = ss.table_count(ss.thread_conn(), "stock_meta")
    print(f"\n数据库: {ss.DEFAULT_DB_FILE} ({stock_count} 只股票)")


# ═══════════════════════════════════════════════════════════
# 方案1: 全量爬取
# ═══════════════════════════════════════════════════════════

def get_full_universe():
    """获取三指数需要的完整样本空间, 合并去重, 排除北交所"""
    stocks = {}

    # 1. 中证全指 (红利低波100 + 科技100 的样本空间)
    print("获取中证全指成分股(000985)...")
    csi_all_map = fetch_index_constituents("000985")
    for code, name in csi_all_map.items():
        if is_bse_stock(code):
            continue
        stocks[code] = {"name": name, "source": "中证全指"}
    print(f"  中证全指: {len(csi_all_map)} 只")

    # 2. 全A股 (中金300 的样本空间 = 沪深A股, 上市>500日, 非ST)
    #    中证全指已包含大部分, 补充未覆盖的
    print("获取上证A股列表...")
    df_sh_main = _retry_fetch(ak.stock_info_sh_name_code, symbol="主板A股")
    df_sh_kcb = _retry_fetch(ak.stock_info_sh_name_code, symbol="科创板")
    df_sh = pd.concat([df_sh_main, df_sh_kcb], ignore_index=True)
    sh_listing = {}
    for _, row in df_sh.iterrows():
        code = str(row["证券代码"]).zfill(6)
        sh_listing[code] = {
            "name": str(row["证券简称"]),
            "listing_date": str(row["上市日期"])[:10],
        }
    print(f"  上证A股: {len(df_sh)} 只")

    print("获取深证A股列表...")
    df_sz = _retry_fetch(ak.stock_info_sz_name_code, symbol="A股列表")
    sz_listing = {}
    for _, row in df_sz.iterrows():
        code = str(row["A股代码"]).zfill(6)
        sz_listing[code] = {
            "name": str(row["A股简称"]),
            "listing_date": str(row["A股上市日期"])[:10],
        }
    print(f"  深证A股: {len(df_sz)} 只")

    # 合并上市日期信息
    all_listing = {**sh_listing, **sz_listing}

    # 中金300 额外候选: 上市>500交易日(约2年) 且 非ST, 排除北交所
    cutoff_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    added = 0
    for code, info in all_listing.items():
        if code in stocks:
            continue
        if is_bse_stock(code):
            continue
        name = info["name"]
        if "ST" in name:
            continue
        if info["listing_date"] > cutoff_date:
            continue
        stocks[code] = {"name": name, "source": "全A股(中金300)"}
        added += 1
    print(f"  中金300补充: {added} 只")

    print(f"  合并去重后共: {len(stocks)} 只\n")
    return stocks


# ═══════════════════════════════════════════════════════════
# 方案2: 分步粗筛
# ═══════════════════════════════════════════════════════════

def get_market_snapshot():
    """Stage 1: 获取全市场基础数据快照"""
    print("=" * 60)
    print("  Stage 1: 获取全市场基础数据")
    print("=" * 60)

    # 1. 实时行情 (代码/名称/最新价/成交量/成交额)
    print("\n获取全A股实时行情 (新浪接口)...")
    df_spot = _retry_fetch(ak.stock_zh_a_spot)
    # 代码格式: sh600000 / sz000001 → 提取纯数字
    df_spot["code"] = df_spot["代码"].str.extract(r"(\d{6})")
    df_spot = df_spot.dropna(subset=["code"])
    spot_map = {}
    for _, row in df_spot.iterrows():
        code = row["code"]
        if is_bse_stock(code):
            continue
        spot_map[code] = {
            "name": str(row["名称"]),           # 股票简称
            "price": _safe_float(row["最新价"]),   # 最新成交价(元)
            "volume": _safe_float(row["成交量"]),  # 当日成交量(手)
            "turnover": _safe_float(row["成交额"]),  # 当日成交额(元), 用于流动性筛选
        }
    print(f"  实时行情: {len(spot_map)} 只")

    # 2. 上证上市日期
    print("获取上证上市日期...")
    df_sh_main = _retry_fetch(ak.stock_info_sh_name_code, symbol="主板A股")
    df_sh_kcb = _retry_fetch(ak.stock_info_sh_name_code, symbol="科创板")
    df_sh = pd.concat([df_sh_main, df_sh_kcb], ignore_index=True)
    for _, row in df_sh.iterrows():
        code = str(row["证券代码"]).zfill(6)
        if code in spot_map:
            spot_map[code]["listing_date"] = str(row["上市日期"])[:10]
    print(f"  上证: {len(df_sh)} 只")

    # 3. 深证上市日期 + 行业 + 股本
    print("获取深证信息...")
    df_sz = _retry_fetch(ak.stock_info_sz_name_code, symbol="A股列表")
    for _, row in df_sz.iterrows():
        code = str(row["A股代码"]).zfill(6)
        if code in spot_map:
            spot_map[code]["listing_date"] = str(row["A股上市日期"])[:10]  # 上市日期, 用于判断上市时长
            spot_map[code]["sz_industry"] = str(row["所属行业"])  # 深交所行业分类
            # 总股本(股), 用于估算总市值
            total_shares_str = str(row.get("A股总股本", "")).replace(",", "")
            try:
                spot_map[code]["total_shares"] = int(total_shares_str)
            except (ValueError, TypeError):
                pass
    print(f"  深证: {len(df_sz)} 只")

    # 4. 中证全指 + 沪深300 成分股列表(官网全量接口)
    print("获取中证全指成分股列表(000985)...")
    csi_all_map = fetch_index_constituents("000985")
    csi_all_set = set(csi_all_map)
    for code in spot_map:
        spot_map[code]["in_csi_all"] = code in csi_all_set
    print(f"  中证全指: {len(csi_all_set)} 只")

    print("获取沪深300成分股列表(000300)...")
    csi300_map = fetch_index_constituents("000300")
    print(f"  沪深300: {len(csi300_map)} 只")
    save_stock_universe(csi300_map, csi_all_map)

    # 5. 质押数据 (含行业分类, 用于科技100的风险筛选和行业判断)
    print("获取质押数据...")
    pledge_map = fetch_pledge_data_bulk()
    for code, pdata in pledge_map.items():
        if code in spot_map:
            spot_map[code]["pledge_ratio"] = pdata.get("pledge_ratio")  # 质押比例(%)
            spot_map[code]["pledge_industry"] = pdata.get("industry")   # 中证行业分类

    # 估算总市值(market_cap_est): 最新价(price) × 总股本(total_shares)
    # 深证有 total_shares 数据, 上证无此字段则标记为 None
    for code, info in spot_map.items():
        price = info.get("price")
        shares = info.get("total_shares")
        if price and shares:
            info["market_cap_est"] = price * shares
        else:
            info["market_cap_est"] = None

    # 保存快照
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
        json.dump(spot_map, f, ensure_ascii=False, indent=2)
    print(f"\n快照已保存: {SNAPSHOT_FILE} ({len(spot_map)} 只)")

    return spot_map


def pre_screen_candidates(snapshot):
    """Stage 2: 按三个指数规则粗筛, 返回需要爬详细数据的股票"""
    print("\n" + "=" * 60)
    print("  Stage 2: 按指数规则粗筛")
    print("=" * 60)

    candidates = {}  # code → {name, candidate_for: [...]}

    def add_candidate(code, index_name):
        info = snapshot.get(code, {})
        name = info.get("name", "")
        if code not in candidates:
            candidates[code] = {"name": name, "candidate_for": [], "source": ""}
        candidates[code]["candidate_for"].append(index_name)

    # ─── 红利低波100 ───
    # 样本空间: 中证全指成分股
    # 流动性筛选: 剔除日均成交额(turnover)排名后20%的股票
    print("\n红利低波100 粗筛:")
    csi_stocks = [(code, info) for code, info in snapshot.items()
                  if info.get("in_csi_all")]  # in_csi_all: 是否属于中证全指成分股
    csi_with_turnover = [(code, info) for code, info in csi_stocks
                         if info.get("turnover") and info["turnover"] > 0]  # turnover: 日成交额(元)
    csi_with_turnover.sort(key=lambda x: x[1]["turnover"], reverse=True)
    cutoff = int(len(csi_with_turnover) * 0.8)  # 保留成交额前80%
    hl_candidates = csi_with_turnover[:max(cutoff, 1)]
    for code, _ in hl_candidates:
        add_candidate(code, "红利低波100")
    print(f"  中证全指: {len(csi_stocks)} → 剔除成交额后20% → {len(hl_candidates)}")

    # ─── 中金300 ───
    # 样本空间: 沪深A股(排除北交所), 上市>500交易日(约2年), 非ST
    print("\n中金300 粗筛:")
    cutoff_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")  # 上市日期截止线(约500交易日)
    zj_all = []
    for code, info in snapshot.items():
        if is_bse_stock(code):
            continue
        name = info.get("name", "")
        if "ST" in name:
            continue
        listing = info.get("listing_date")
        if listing and listing > cutoff_date:
            continue
        zj_all.append((code, info))

    # 行业内综合排名需要详细财务数据, 这里只做基本过滤
    for code, _ in zj_all:
        add_candidate(code, "中金300")
    print(f"  全A股非ST且上市>2年: {len(zj_all)}")

    # ─── 科技100 ───
    # 样本空间: 中证全指中属于科技行业的股票
    # 流动性筛选: 剔除总市值(market_cap_est)或日成交额(turnover)任一排名后20%
    # 风险筛选: 剔除股权质押比例(pledge_ratio)排名前5%(质押率最高的)
    print("\n科技100 粗筛:")
    tech_stocks = []
    for code, info in snapshot.items():
        if not info.get("in_csi_all"):  # 必须属于中证全指
            continue
        # pledge_industry: 质押数据中的行业分类; sz_industry: 深交所行业分类
        industry = info.get("pledge_industry") or info.get("sz_industry") or ""
        if any(kw in industry for kw in TECH_KEYWORDS):  # 匹配科技行业关键词
            tech_stocks.append((code, info))

    # 剔除总市值(market_cap_est = 最新价 × 总股本)排名后20%
    tech_with_cap = [(c, i) for c, i in tech_stocks
                     if i.get("market_cap_est") and i["market_cap_est"] > 0]
    tech_with_cap.sort(key=lambda x: x[1]["market_cap_est"], reverse=True)
    cap_cutoff = int(len(tech_with_cap) * 0.8)  # 保留市值前80%
    cap_bottom = set(c for c, _ in tech_with_cap[cap_cutoff:])

    # 剔除日成交额(turnover)排名后20%
    tech_with_turn = [(c, i) for c, i in tech_stocks
                      if i.get("turnover") and i["turnover"] > 0]
    tech_with_turn.sort(key=lambda x: x[1]["turnover"], reverse=True)
    turn_cutoff = int(len(tech_with_turn) * 0.8)  # 保留成交额前80%
    turn_bottom = set(c for c, _ in tech_with_turn[turn_cutoff:])

    # 市值或成交额任一在后20%的均剔除
    tech_filtered = [(c, i) for c, i in tech_stocks
                     if c not in cap_bottom and c not in turn_bottom]

    # 剔除股权质押比例(pledge_ratio, 单位%)排名前5%(质押率最高的股票)
    tech_pledge = [(c, i.get("pledge_ratio", 0) or 0) for c, i in tech_filtered]
    tech_pledge.sort(key=lambda x: x[1], reverse=True)
    pledge_cutoff = max(int(len(tech_pledge) * 0.05), 1)  # 剔除质押率最高的5%
    high_pledge = set(c for c, _ in tech_pledge[:pledge_cutoff])
    tech_final = [(c, i) for c, i in tech_filtered if c not in high_pledge]

    for code, _ in tech_final:
        add_candidate(code, "科技100")
    print(f"  科技行业: {len(tech_stocks)} → 流动性筛选 → {len(tech_filtered)} → 去高质押 → {len(tech_final)}")

    # 汇总
    for code, info in candidates.items():
        info["source"] = "+".join(info["candidate_for"])

    print(f"\n三指数候选池合并去重: {len(candidates)} 只")

    # 按候选指数数量排序（优先爬多指数候选的）
    candidates = dict(sorted(candidates.items(),
                             key=lambda x: len(x[1]["candidate_for"]), reverse=True))

    # 保存候选列表
    with open(CONS_FILE, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    return candidates


# ═══════════════════════════════════════════════════════════
# 财报加深重爬（PIT 回测用，跳过日线）
# ═══════════════════════════════════════════════════════════

def refetch_financials(codes=None, workers=8):
    """只重拉三大报表+指标+分红(跳过日线)，把现有 stock_data 文件的财报加深到
    FINANCIAL_YEARS 年。日线长历史已在 stock_data.history，这里不动，避免重爬慢接口。

    codes=None 时取带 history.records 的 stock_data 文件，正是 PIT 回测要 join 的股票池。
    """
    if codes is None:
        codes = ss.codes_with_history(ss.thread_conn())
    print(f"重拉财报到 {FINANCIAL_YEARS} 年(跳过日线): {len(codes)} 只 / {workers} 线程\n")

    def _worker(code):
        conn = ss.thread_conn()
        data = ss.load_stock(conn, code)
        if not data:
            return code, False, "无 stock_data 记录"
        try:
            data["financials"] = fetch_financial_reports(code)
            data["indicators"] = fetch_financial_indicators(code)
            data["dividends"] = fetch_dividend_history(code)
            data["financials_refetched_at"] = datetime.now().isoformat()
            # 只改财报/指标/分红，不重写 stock_history（187 万行无谓重写）
            ss.save_stock(conn, data, write_history=False)
            return code, True, ""
        except Exception as e:
            return code, False, str(e)

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, c) for c in codes]
        for i, fut in enumerate(as_completed(futures), 1):
            code, ok, err = fut.result()
            if not ok:
                failed.append((code, err))
            if i % 100 == 0:
                print(f"  进度 {i}/{len(codes)} (失败 {len(failed)})")

    print(f"\n完成: {len(codes) - len(failed)}/{len(codes)} 成功, {len(failed)} 失败")
    for code, err in failed[:10]:
        print(f"  ✗ {code}: {err}")
    if len(failed) > 10:
        print(f"  ... 其余 {len(failed) - 10} 个失败")


def refetch_daily(codes=None, years=1, workers=8, limit=0):
    """只重拉已有 stock_data 文件的 daily 字段，用于统一日线复权和涨跌幅口径。"""
    if codes is None:
        codes = ss.list_codes(ss.thread_conn())
    else:
        conn = ss.thread_conn()
        codes = [str(c).zfill(6) for c in codes if ss.stock_exists(conn, c)]
    if limit and limit > 0:
        codes = codes[:limit]

    print(f"重拉前复权日线到 {years} 年: {len(codes)} 只 / {workers} 线程\n")

    failed = []
    done = 0
    lock = threading.Lock()

    def _worker(code):
        conn = ss.thread_conn()
        data = ss.load_stock(conn, code)
        if not data:
            return code, False, "无 stock_data 记录"
        try:
            daily = fetch_daily_price(code, years=years)
            new_records = daily.pop("history_records", [])
            existing_records = (data.get("history") or {}).get("records", [])
            merged_records = merge_records_by_date(existing_records, new_records, overwrite_none=False)
            data["history"] = history_payload_from_records(
                code,
                data.get("name") or code,
                merged_records,
                "stock_data.history+stock_crawl_fundamentals.refetch_daily",
            )
            data["daily"] = daily_payload_from_history_records(merged_records)
            data["daily_refetched_at"] = datetime.now().isoformat()
            ss.save_stock(conn, data)
            return code, True, ""
        except Exception as e:
            return code, False, str(e)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, c) for c in codes]
        for fut in as_completed(futures):
            code, ok, err = fut.result()
            with lock:
                done += 1
                if not ok:
                    failed.append((code, err))
                if done % 50 == 0 or done == len(codes):
                    print(f"  进度 {done}/{len(codes)} (失败 {len(failed)})")

    print(f"\n完成: {len(codes) - len(failed)}/{len(codes)} 成功, {len(failed)} 失败")
    for code, err in failed[:10]:
        print(f"  ✗ {code}: {err}")
    if len(failed) > 10:
        print(f"  ... 其余 {len(failed) - 10} 个失败")


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="爬取三指数候选池股票数据")
    parser.add_argument("--mode", choices=["full", "staged"], default="staged",
                        help="full=全量爬取, staged=分步粗筛(默认)")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制爬取数量, 0=不限 (full模式默认20)")
    parser.add_argument("--workers", type=int, default=40,
                        help="并发线程数(默认40)")
    parser.add_argument("--refetch-financials", action="store_true",
                        help="只把已有 stock_data 的财报加深重拉到10年(跳过日线), 用于PIT回测")
    parser.add_argument("--refetch-daily", action="store_true",
                        help="只重拉已有 stock_data 的前复权日线并统一 close-to-close 涨跌幅")
    parser.add_argument("--daily-years", type=int, default=1,
                        help="--refetch-daily 重拉最近多少年日线，默认 1")
    args = parser.parse_args()

    strip_proxy_env()

    if args.refetch_financials:
        refetch_financials(workers=max(args.workers, 1) if args.workers else 8)
        return

    if args.refetch_daily:
        refetch_daily(
            years=max(args.daily_years, 1),
            workers=max(args.workers, 1) if args.workers else 8,
            limit=args.limit,
        )
        return

    if args.mode == "full":
        limit = args.limit if args.limit > 0 else 20
        print(f"模式: 全量爬取 (limit={limit}, workers={args.workers})\n")
        stocks = get_full_universe()
        pledge_map = fetch_pledge_data_bulk()
        crawl_stocks(stocks, pledge_map, limit=limit, workers=args.workers)

    elif args.mode == "staged":
        limit = args.limit
        print(f"模式: 分步粗筛 (workers={args.workers})\n")

        # Stage 1: 全市场快照
        snapshot = get_market_snapshot()

        # Stage 2: 粗筛
        candidates = pre_screen_candidates(snapshot)

        # Stage 3: 爬取详细数据
        print(f"\n{'=' * 60}")
        print(f"  Stage 3: 爬取详细财务数据")
        print(f"{'=' * 60}")
        pledge_map = {}
        for code, info in snapshot.items():
            if info.get("pledge_ratio") is not None or info.get("pledge_industry"):
                pledge_map[code] = {
                    "pledge_ratio": info.get("pledge_ratio"),
                    "pledge_count": None,
                    "trade_date": None,
                    "industry": info.get("pledge_industry"),
                }
        crawl_stocks(candidates, pledge_map, limit=limit, workers=args.workers)


if __name__ == "__main__":
    main()
