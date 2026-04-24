"""
股票数据爬取工具

为自定义指数编制（sorahjy_stock_index.py）获取所需的股票基础数据。
数据源: akshare (新浪/东方财富接口)

所需数据:
  A. 日频行情: 收盘价、成交额、涨跌幅 → 计算波动率、日均成交额
  B. 利润表: 营业总收入、营业成本、研发费用、净利润、扣非净利润
  C. 资产负债表: 股东权益(净资产)、开发支出
  D. 现金流量表: 经营活动现金流量净额
  E. 财务指标: ROE、净利润增长率、营收增长率、每股净资产、股息率
  F. 分红历史: 每年派息金额
  G. 质押比例
"""

import json
import time
import math
from datetime import datetime, timedelta
from pathlib import Path
import akshare as ak
import pandas as pd
import numpy as np


DATA_DIR = Path(__file__).parent / "data"
STOCK_DATA_FILE = DATA_DIR / "stock_data.json"

STOCKS = {
    "600036": "招商银行",
    "000895": "双汇发展",
}

MAX_RETRIES = 3


def _retry_fetch(func, *args, retries=MAX_RETRIES, **kwargs):
    """带重试的数据获取，应对连接不稳定"""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt + 1)
            else:
                raise


def _safe_float(val):
    """将值转为 float，NaN/None 转为 None（方便 JSON 序列化）"""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    except (ValueError, TypeError):
        return None


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


# ─── 1. 日频行情 ───────────────────────────────────────────────

def fetch_daily_price(symbol, years=1):
    """获取近 N 年日频行情，计算波动率和日均成交额
    优先使用腾讯接口(稳定)，回退到东方财富接口
    """
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y%m%d")

    # 确定腾讯接口需要的 symbol 前缀
    tx_symbol = f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}"

    try:
        df = _retry_fetch(
            ak.stock_zh_a_hist_tx,
            symbol=tx_symbol, start_date=start_date, end_date=end_date
        )
        # 腾讯接口: date, open, close, high, low, amount(成交量,手)
        close_prices = df["close"].astype(float)
        daily_returns = close_prices.pct_change().dropna()
        # amount 是成交量(手)，近似成交额 = 成交量 * 100 * 收盘价
        df["turnover_approx"] = df["amount"].astype(float) * 100 * close_prices

        stats = {
            "trading_days": len(df),
            "avg_daily_volume": _safe_float(df["amount"].astype(float).mean()),
            "avg_daily_turnover_approx": _safe_float(df["turnover_approx"].mean()),
            "volatility_annual": _safe_float(daily_returns.std() * np.sqrt(252)),
            "volatility_daily_std": _safe_float(daily_returns.std()),
            "latest_close": _safe_float(close_prices.iloc[-1]),
            "start_date": str(df["date"].iloc[0])[:10],
            "end_date": str(df["date"].iloc[-1])[:10],
        }

        recent = []
        for _, row in df.tail(30).iterrows():
            recent.append({
                "date": str(row["date"])[:10],
                "close": _safe_float(row["close"]),
                "volume": _safe_float(row["amount"]),
                "change_pct": _safe_float(
                    (row["close"] - row["open"]) / row["open"] * 100 if row["open"] else None
                ),
            })

    except Exception as e:
        pass
        df = _retry_fetch(
            ak.stock_zh_a_hist,
            symbol=symbol, period="daily",
            start_date=start_date, end_date=end_date, adjust="qfq"
        )
        close_prices = df["收盘"].astype(float)
        daily_returns = close_prices.pct_change().dropna()

        stats = {
            "trading_days": len(df),
            "avg_daily_turnover": _safe_float(df["成交额"].astype(float).mean()),
            "volatility_annual": _safe_float(daily_returns.std() * np.sqrt(252)),
            "volatility_daily_std": _safe_float(daily_returns.std()),
            "latest_close": _safe_float(close_prices.iloc[-1]),
            "start_date": str(df["日期"].iloc[0])[:10],
            "end_date": str(df["日期"].iloc[-1])[:10],
        }

        recent = []
        for _, row in df.tail(30).iterrows():
            recent.append({
                "date": str(row["日期"])[:10],
                "close": _safe_float(row["收盘"]),
                "turnover": _safe_float(row["成交额"]),
                "change_pct": _safe_float(row["涨跌幅"]),
            })

    return {"stats": stats, "recent_daily": recent}


# ─── 2. 财务报表 ───────────────────────────────────────────────

def fetch_financial_reports(stock_code):
    """获取三大财务报表（利润表、资产负债表、现金流量表）"""
    result = {}

    # 利润表
    df_income = _retry_fetch(ak.stock_financial_report_sina, stock=stock_code, symbol="利润表")
    income_cols = {
        "revenue": ["营业总收入", "营业收入"],  # 银行股用"营业收入"
        "operating_cost": ["营业总成本", "营业支出"],  # 银行股用"营业支出"
        "cost_of_revenue": "营业成本",
        "rd_expense": "研发费用",
        "net_profit": "净利润",
        "net_profit_parent": ["归属于母公司所有者的净利润", "归属于母公司的净利润"],
    }
    result["income"] = _df_to_records(df_income, income_cols)[:20]  # 近5年(20个季度)

    time.sleep(0.3)

    # 资产负债表
    df_balance = _retry_fetch(ak.stock_financial_report_sina, stock=stock_code, symbol="资产负债表")
    balance_cols = {
        "total_equity_parent": ["归属于母公司股东权益合计", "归属于母公司股东的权益"],
        "minority_equity": "少数股东权益",
        "total_equity": ["所有者权益(或股东权益)合计", "负债及股东权益总计"],
        "total_assets_liabilities": ["负债和所有者权益(或股东权益)总计", "负债及股东权益总计"],
        "dev_expenditure": "开发支出",
    }
    result["balance"] = _df_to_records(df_balance, balance_cols)[:20]

    time.sleep(0.3)

    # 现金流量表
    df_cashflow = _retry_fetch(ak.stock_financial_report_sina, stock=stock_code, symbol="现金流量表")
    cashflow_cols = {
        "operating_cashflow_net": "经营活动产生的现金流量净额",
        "operating_cashflow_in": "经营活动现金流入小计",
        "operating_cashflow_out": "经营活动现金流出小计",
    }
    result["cashflow"] = _df_to_records(df_cashflow, cashflow_cols)[:20]

    return result


# ─── 3. 财务指标 ───────────────────────────────────────────────

def fetch_financial_indicators(symbol):
    """获取财务分析指标（ROE、增长率、每股净资产等）"""
    start_year = str(datetime.now().year - 5)
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


# ─── 4. 分红历史 ───────────────────────────────────────────────

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
        "records": records[:20],
        "yearly_dividends": {str(k): _safe_float(v) for k, v in sorted(yearly_dividends.items(), reverse=True)[:6]},
        "consecutive_3y_dividend": consecutive_3y,
    }


# ─── 5. 质押比例 ───────────────────────────────────────────────

def fetch_pledge_ratio(symbol):
    """获取股权质押比例"""
    try:
        df = _retry_fetch(ak.stock_gpzy_pledge_ratio_em)
        row = df[df["股票代码"] == symbol]
        if not row.empty:
            return {
                "pledge_ratio": _safe_float(row.iloc[0]["质押比例"]),
                "pledge_count": int(row.iloc[0]["质押笔数"]) if pd.notna(row.iloc[0]["质押笔数"]) else None,
                "trade_date": str(row.iloc[0]["交易日期"])[:10],
                "industry": str(row.iloc[0]["所属行业"]),
            }
    except Exception:
        pass
    return {"pledge_ratio": None, "pledge_count": None, "trade_date": None, "industry": None}


# ─── 整合 ─────────────────────────────────────────────────────

def fetch_all_stock_data(symbol, name):
    """整合获取一只股票的全部数据"""
    print(f"  获取 {name}({symbol}) ...")
    data = {"symbol": symbol, "name": name, "fetch_time": datetime.now().isoformat()}

    # 日频行情
    data["daily"] = fetch_daily_price(symbol)
    time.sleep(0.5)

    # 财务报表
    data["financials"] = fetch_financial_reports(symbol)
    time.sleep(0.5)

    # 财务指标
    data["indicators"] = fetch_financial_indicators(symbol)
    time.sleep(0.5)

    # 分红
    data["dividends"] = fetch_dividend_history(symbol)
    time.sleep(0.5)

    # 质押
    data["pledge"] = fetch_pledge_ratio(symbol)

    return data


def print_summary(data):
    """打印数据汇总"""
    name = data["name"]
    symbol = data["symbol"]

    print(f"\n{'━'*60}")
    print(f"  {name} ({symbol}) 数据汇总")
    print(f"{'━'*60}")

    # 日频统计
    ds = data["daily"]["stats"]
    print(f"\n  【日频行情统计】({ds['start_date']} ~ {ds['end_date']})")
    print(f"    交易天数:        {ds['trading_days']}")
    print(f"    最新收盘价:      {ds['latest_close']}")
    turnover = ds.get('avg_daily_turnover') or ds.get('avg_daily_turnover_approx')
    print(f"    日均成交额(估):  {turnover:,.0f} 元" if turnover else "    日均成交额: N/A")
    if ds.get('avg_daily_volume'):
        print(f"    日均成交量:      {ds['avg_daily_volume']:,.0f} 手")
    print(f"    年化波动率:      {ds['volatility_annual']:.4f}" if ds['volatility_annual'] else "    年化波动率: N/A")
    print(f"    日收益率标准差:  {ds['volatility_daily_std']:.6f}" if ds['volatility_daily_std'] else "    日收益率标准差: N/A")

    # 财务报表概览
    fin = data["financials"]
    if fin["income"]:
        latest = fin["income"][0]
        print(f"\n  【最新利润表】({latest['date']})")
        print(f"    营业收入:        {latest['revenue']:,.0f} 元" if latest['revenue'] else "    营业收入: N/A")
        print(f"    营业成本:        {latest.get('cost_of_revenue') or latest.get('operating_cost') or 0:,.0f} 元" if (latest.get('cost_of_revenue') or latest.get('operating_cost')) else "    营业成本: N/A")
        print(f"    研发费用:        {latest['rd_expense']:,.0f} 元" if latest.get('rd_expense') else "    研发费用: N/A")
        print(f"    净利润:          {latest['net_profit']:,.0f} 元" if latest['net_profit'] else "    净利润: N/A")

    if fin["balance"]:
        latest = fin["balance"][0]
        print(f"\n  【最新资产负债表】({latest['date']})")
        print(f"    归属母公司股东权益: {latest['total_equity_parent']:,.0f} 元" if latest['total_equity_parent'] else "    归属母公司股东权益: N/A")

    if fin["cashflow"]:
        latest = fin["cashflow"][0]
        print(f"\n  【最新现金流量表】({latest['date']})")
        print(f"    经营活动现金流净额: {latest['operating_cashflow_net']:,.0f} 元" if latest['operating_cashflow_net'] else "    经营活动现金流净额: N/A")

    # 财务指标
    ind = data["indicators"]
    if ind["records"]:
        latest = ind["records"][0]
        print(f"\n  【最新财务指标】({latest['date']})")
        print(f"    ROE:             {latest['roe']}%" if latest['roe'] is not None else "    ROE: N/A")
        print(f"    加权ROE:         {latest['roe_weighted']}%" if latest['roe_weighted'] is not None else "    加权ROE: N/A")
        print(f"    营收增长率:      {latest['revenue_growth']}%" if latest['revenue_growth'] is not None else "    营收增长率: N/A")
        print(f"    净利润增长率:    {latest['net_profit_growth']}%" if latest['net_profit_growth'] is not None else "    净利润增长率: N/A")
        print(f"    每股净资产:      {latest['bvps_adjusted']} 元" if latest['bvps_adjusted'] is not None else "    每股净资产: N/A")

    if ind["roe_stats"]:
        rs = ind["roe_stats"]
        print(f"\n  【ROE统计】(近{rs['count']}期)")
        print(f"    ROE均值:         {rs['mean']}%")
        print(f"    ROE标准差:       {rs['std']}%")
        print(f"    ROE均值-标准差:  {_safe_float(rs['mean'] - rs['std'])}%")

    # 分红
    div = data["dividends"]
    print(f"\n  【分红情况】")
    print(f"    近3年连续分红:   {'是' if div['consecutive_3y_dividend'] else '否'}")
    if div["yearly_dividends"]:
        print(f"    各年度派息(每10股):")
        for year, amt in list(div["yearly_dividends"].items())[:5]:
            print(f"      {year}年: {amt} 元")

    # 质押
    pl = data["pledge"]
    print(f"\n  【质押情况】")
    print(f"    质押比例:        {pl['pledge_ratio']}%" if pl['pledge_ratio'] is not None else "    质押比例: N/A")
    print(f"    所属行业:        {pl['industry']}" if pl['industry'] else "")

    print(f"\n{'━'*60}\n")


def main():
    all_data = {}

    for symbol, name in STOCKS.items():
        stock_data = fetch_all_stock_data(symbol, name)
        all_data[symbol] = stock_data
        print_summary(stock_data)

    # 持久化存储
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(STOCK_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n数据已保存到: {STOCK_DATA_FILE}")
    print(f"文件大小: {STOCK_DATA_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()