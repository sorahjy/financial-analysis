"""
自定义指数编制工具 - 三指数选股实现

复现三个中证指数的编制规则：
  1. 红利低波100 (930955) - 高股息率、低波动率
  2. 中金300 (931069) - 高ROE稳定性、分红+成长
  3. 科技100 (931187) - 研发强度+盈利+成长

数据来源: data/stock_data/CN_*.json (由 crawl_index_stocks.py 爬取)

参考指数编制方案：
  - 中证红利低波动100 (930955): index_report/红利低波100.pdf
  - 中证中金优选300 (931069): index_report/中金300.pdf
  - 中证科技100 (931187): index_report/科技100.pdf
"""

import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np

DATA_DIR = Path(__file__).parent / "data"


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def load_all_stocks():
    """加载所有已爬取的股票数据（从 data/stock_data/ 目录逐文件读取）"""
    all_stocks = {}

    stock_dir = DATA_DIR / "stock_data"
    if stock_dir.is_dir():
        for f in stock_dir.glob("CN_*.json"):
            code = f.stem.split("_")[1]
            with open(f, "r", encoding="utf-8") as fp:
                all_stocks[code] = json.load(fp)

    # 兼容旧格式 stock_data.json
    sd_file = DATA_DIR / "stock_data.json"
    if sd_file.exists():
        with open(sd_file, "r", encoding="utf-8") as f:
            for k, v in json.load(f).items():
                if k not in all_stocks:
                    all_stocks[k] = v

    print(f"共加载 {len(all_stocks)} 只股票数据")
    return all_stocks


def safe(val):
    """安全获取数值，None/NaN/Inf → None"""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (ValueError, TypeError):
        return None


def get_annual_records(records, field, n_years=3):
    """从财报记录中取最近 n 个年报（12-31）的指定字段值"""
    values = []
    for r in records:
        if r["date"].endswith("-12-31"):
            v = safe(r.get(field))
            if v is not None:
                values.append(v)
                if len(values) >= n_years:
                    break
    return values


def estimate_shares(stock):
    """估算总股本 = 归母净资产 / 每股净资产"""
    ind = stock.get("indicators", {}).get("records", [])
    bal = stock.get("financials", {}).get("balance", [])
    if not ind or not bal:
        return None
    bvps = safe(ind[0].get("bvps_adjusted"))
    equity = safe(bal[0].get("total_equity_parent"))
    if bvps and bvps > 0 and equity and equity > 0:
        return equity / bvps
    return None


def estimate_market_cap(stock):
    """估算总市值 = 总股本 × 最新收盘价"""
    shares = estimate_shares(stock)
    price = safe(stock.get("daily", {}).get("stats", {}).get("latest_close"))
    if shares and price:
        return shares * price
    return None


def compute_dividend_yield(stock, years=3):
    """计算 N 年平均股息率 = 近 N 年年均每股分红 / 当前股价
    PDF: "过去三年平均现金股息率" (红利低波100) / "过去5年平均股息率" (中金300)
    近似: 使用最新收盘价代替调整日总市值/股本来计算每股股息率
    """
    yearly = stock.get("dividends", {}).get("yearly_dividends", {})
    price = safe(stock.get("daily", {}).get("stats", {}).get("latest_close"))
    if not price or price <= 0:
        return None

    current_year = datetime.now().year
    divs = []
    for yr_offset in range(1, years + 1):
        yr = str(current_year - yr_offset)
        d = safe(yearly.get(yr))
        if d is not None and d > 0:
            divs.append(d / 10.0)  # 每10股派息 → 每股派息
        else:
            if years <= 3:
                return None  # 红利低波100要求近3年每年都有现金分红
            # 中金300的5年平均允许缺失年份，跳过
    if not divs:
        return None
    return (sum(divs) / len(divs)) / price


def compute_ttm(records, field):
    """计算 TTM（滚动12个月）值

    财报为累计值: Q1=1季度, H1=上半年, Q3=前三季度, FY=全年
    TTM = 最新累计值 + 上年全年 - 上年同期累计值
    """
    if not records:
        return None

    latest = records[0]
    latest_val = safe(latest.get(field))
    if latest_val is None:
        return None

    date = latest["date"]
    month = date[5:7]

    if month == "12":
        return latest_val

    latest_year = int(date[:4])
    prev_fy = None
    prev_same = None

    for r in records[1:]:
        yr = int(r["date"][:4])
        mo = r["date"][5:7]
        v = safe(r.get(field))
        if v is None:
            continue
        if yr == latest_year - 1:
            if mo == "12":
                prev_fy = v
            if mo == month:
                prev_same = v

    if prev_fy is not None and prev_same is not None:
        return latest_val + prev_fy - prev_same

    return latest_val  # 无法精确计算时近似返回最新值


def compute_yoy_growth(records, field):
    """计算 TTM 同比增速 = (当前TTM - 去年同期TTM) / |去年同期TTM|"""
    ttm_now = compute_ttm(records, field)
    if ttm_now is None or not records:
        return None

    latest_date = records[0]["date"]
    latest_year = int(latest_date[:4])
    latest_month = latest_date[5:7]

    # 构建去年同期起始的子记录列表
    target_date = f"{latest_year - 1}-{latest_month}"
    sub_records = [r for r in records if r["date"] <= target_date]
    if not sub_records:
        return None

    ttm_prev = compute_ttm(sub_records, field)
    if ttm_prev is None or ttm_prev == 0:
        return None

    return (ttm_now - ttm_prev) / abs(ttm_prev)


def get_industry(stock):
    """获取行业分类（来自质押数据）"""
    return stock.get("pledge", {}).get("industry") or "未知"


def rank_desc(values):
    """降序排名 (值最大排名为1)，None 排最后"""
    n = len(values)
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: (x[1] is None, -(x[1] if x[1] is not None else 0)))
    ranks = [0] * n
    for rank_pos, (idx, _) in enumerate(indexed):
        ranks[idx] = rank_pos + 1
    return ranks


def percentile_rank(values):
    """升序百分比排名 (0~1)，None 排最低"""
    n = len(values)
    if n == 0:
        return {}
    sorted_vals = sorted(enumerate(values),
                         key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0))
    ranks = {}
    for rank_pos, (idx, val) in enumerate(sorted_vals):
        ranks[idx] = 0.0 if val is None else (rank_pos + 1) / n
    return ranks


def apply_industry_weight_cap(results, cap=0.20):
    """应用行业权重上限，迭代调整
    PDF(红利低波100): "单个中证二级行业权重不超过20%，超出部分按比例分配"
    """
    total_raw = sum(r["raw_weight"] for r in results)
    if total_raw <= 0:
        w = round(1.0 / len(results), 6) if results else 0
        for r in results:
            r["weight"] = w
        return results

    for r in results:
        r["weight"] = r["raw_weight"] / total_raw

    for _ in range(20):
        industry_weights = {}
        for r in results:
            industry_weights[r["industry"]] = industry_weights.get(r["industry"], 0) + r["weight"]

        capped = False
        for ind, iw in industry_weights.items():
            if iw > cap + 1e-9:
                scale = cap / iw
                for r in results:
                    if r["industry"] == ind:
                        r["weight"] *= scale
                capped = True

        if not capped:
            break

        total = sum(r["weight"] for r in results)
        if total > 0:
            for r in results:
                r["weight"] /= total

    for r in results:
        r["weight"] = round(r["weight"] * 100, 4)  # 百分比

    return results


def apply_market_cap_weight(results, cap=0.05):
    """市值加权，迭代压缩单票权重至上限
    PDF(中金300/科技100): "采用自由流通市值加权，单只股票权重不超过5%"
    近似: 使用总市值估算值代替自由流通市值
    """
    total_cap = sum(r["market_cap"] for r in results if r.get("market_cap"))
    if total_cap <= 0:
        w = round(100.0 / max(len(results), 1), 4)
        for r in results:
            r["weight"] = w
        return results

    for r in results:
        r["weight"] = r.get("market_cap", 0) / total_cap

    for _ in range(50):
        capped = False
        excess = 0.0
        uncapped_weight = 0.0
        for r in results:
            if r["weight"] > cap + 1e-9:
                excess += r["weight"] - cap
                r["weight"] = cap
                capped = True
            else:
                uncapped_weight += r["weight"]

        if not capped:
            break

        if uncapped_weight > 0:
            scale = (uncapped_weight + excess) / uncapped_weight
            for r in results:
                if r["weight"] < cap - 1e-9:
                    r["weight"] *= scale

    for r in results:
        r["weight"] = round(r["weight"] * 100, 4)  # 百分比

    return results


# ═══════════════════════════════════════════════════════════
# 指数一: 红利低波100 (930955)
# ═══════════════════════════════════════════════════════════
#
# PDF编制方案:
#   样本空间: 中证全指成分股
#   1. 剔除过去一年日均成交额排名后20%
#   2. 过去三年连续现金分红，且每年现金股息率>0
#   3. 按过去三年平均现金股息率降序排名，取前300名
#   4. 对前300名按过去一年波动率升序排名，取前100名
#   加权: 股息率/波动率，单个中证二级行业权重不超过20%

def build_dividend_low_vol_100(all_stocks):
    print("\n" + "=" * 60)
    print("  红利低波100 选股")
    print("=" * 60)

    # 准备有效数据
    candidates = []
    for sym, stock in all_stocks.items():
        stats = stock.get("daily", {}).get("stats", {})
        turnover = safe(stats.get("avg_daily_turnover_approx")) or safe(stats.get("avg_daily_turnover"))
        vol = safe(stats.get("volatility_daily_std"))
        if turnover and vol:
            candidates.append((sym, stock, turnover, vol))

    print(f"  有效股票: {len(candidates)}")

    # Step 1: PDF "剔除过去一年日均成交额排名后20%的股票"
    candidates.sort(key=lambda x: x[2], reverse=True)
    cutoff = int(len(candidates) * 0.8)
    candidates = candidates[:max(cutoff, 1)]
    print(f"  剔除成交额后20%: {len(candidates)}")

    # Step 2: PDF "过去三年连续现金分红，且每年的现金股息率均大于0"
    div_candidates = []
    for sym, stock, turnover, vol in candidates:
        if not stock.get("dividends", {}).get("consecutive_3y_dividend"):
            continue
        # PDF "过去三年平均现金股息率" = 近3年年均每股分红 / 股价
        dy = compute_dividend_yield(stock, years=3)
        if dy is not None and dy > 0:
            div_candidates.append((sym, stock, dy, vol))

    print(f"  连续3年分红且股息率>0: {len(div_candidates)}")

    # Step 3: PDF "对样本按照过去三年平均现金股息率由高到低排名，选取排名前300的股票"
    div_candidates.sort(key=lambda x: x[2], reverse=True)
    top300 = div_candidates[:300]
    print(f"  股息率前300: {len(top300)}")

    # Step 4: PDF "对上述300只股票，按过去一年波动率由低到高排名，选取排名前100的股票"
    # 近似: 使用日收益率标准差代替完整的日波动率计算
    top300.sort(key=lambda x: x[3])
    top100 = top300[:100]
    print(f"  波动率最低100: {len(top100)}")

    # 加权: PDF "权重因子=股息率/波动率，单个中证二级行业权重不超过20%"
    results = []
    for sym, stock, dy, vol in top100:
        results.append({
            "symbol": sym,
            "name": stock.get("name", ""),
            "dividend_yield_3y": round(dy * 100, 4),
            "volatility": round(vol, 6),
            "raw_weight": dy / vol if vol > 0 else 0,  # PDF: Wi = Di / Vi
            "industry": get_industry(stock),
        })

    results = apply_industry_weight_cap(results, cap=0.20)  # PDF: 行业权重≤20%
    results.sort(key=lambda r: r["weight"], reverse=True)

    print(f"\n  红利低波100 最终成分股: {len(results)} 只")
    return results


# ═══════════════════════════════════════════════════════════
# 指数二: 中金300 (931069)
# ═══════════════════════════════════════════════════════════
#
# PDF编制方案:
#   样本空间: 沪深A股, 上市>500交易日, 非ST
#   1. 按中证二级行业分组，剔除净资产≤0
#   2. 行业内按"TTM营收+日均流通市值+日均成交额"综合排名，剔除后40%
#   3. 计算ROE均值-标准差因子(近5年季度ROE)，剔除后50%
#   4. 按"过去5年平均股息率+TTM净利润增长率"综合排名，取前300名
#   加权: 自由流通市值加权，单票权重上限5%
#   近似: ROE使用akshare预报值而非自算TTM序列; 市值用总市值估算代替流通市值

def build_cicc_300(all_stocks):
    print("\n" + "=" * 60)
    print("  中金300 选股")
    print("=" * 60)

    candidates = []
    for sym, stock in all_stocks.items():
        stats = stock.get("daily", {}).get("stats", {})
        balance = stock.get("financials", {}).get("balance", [])
        income = stock.get("financials", {}).get("income", [])
        indicators = stock.get("indicators", {})

        # PDF "剔除净资产为负或为零的股票"
        equity = safe(balance[0].get("total_equity_parent")) if balance else None
        if equity is None or equity <= 0:
            continue

        # PDF "TTM营业收入" (行业内综合排名用)
        ttm_revenue = compute_ttm(income, "revenue")
        if ttm_revenue is None:
            continue

        # PDF "过去一年日均成交额" (行业内综合排名用)
        turnover = safe(stats.get("avg_daily_turnover_approx")) or safe(stats.get("avg_daily_turnover"))
        if turnover is None:
            continue

        # PDF "日均总市值" (行业内综合排名用)
        # 近似: 用总市值估算值代替"日均自由流通市值"
        mkt_cap = estimate_market_cap(stock)
        if mkt_cap is None:
            continue

        # PDF "ROE均值-标准差因子" (近5年季度ROE序列)
        # 近似: 使用akshare预报的ROE值，而非自算 TTM净利润/平均净资产 序列
        roe_stats = indicators.get("roe_stats", {})
        roe_mean = safe(roe_stats.get("mean"))
        roe_std = safe(roe_stats.get("std"))
        roe_factor = (roe_mean - roe_std) if (roe_mean is not None and roe_std is not None) else None

        # PDF "过去5年平均股息率"
        dy5 = compute_dividend_yield(stock, years=5)

        # PDF "TTM净利润增长率" = (当期TTM净利润 - 去年同期TTM净利润) / |去年同期TTM净利润|
        np_growth = compute_yoy_growth(income, "net_profit")

        candidates.append({
            "symbol": sym,
            "name": stock.get("name", ""),
            "industry": get_industry(stock),
            "ttm_revenue": ttm_revenue,
            "market_cap": mkt_cap,
            "turnover": turnover,
            "roe_factor": roe_factor,
            "dy5": dy5,
            "np_growth": np_growth,
        })

    print(f"  有效候选股: {len(candidates)}")

    # Step 1: PDF "在各中证二级行业内，按TTM营收+日均流通市值+日均成交额综合排名，剔除后40%"
    by_industry = {}
    for c in candidates:
        by_industry.setdefault(c["industry"], []).append(c)

    after_industry = []
    for ind, group in by_industry.items():
        n = len(group)
        if n == 0:
            continue

        rev_ranks = rank_desc([c["ttm_revenue"] for c in group])
        cap_ranks = rank_desc([c["market_cap"] for c in group])
        turn_ranks = rank_desc([c["turnover"] for c in group])

        for i, c in enumerate(group):
            c["composite_rank"] = rev_ranks[i] + cap_ranks[i] + turn_ranks[i]

        group.sort(key=lambda x: x["composite_rank"])
        keep = max(int(n * 0.6), 1)
        after_industry.extend(group[:keep])

    print(f"  行业内剔除后40%: {len(after_industry)}")

    # Step 2: PDF "计算ROE均值-标准差因子，剔除排名后50%"
    roe_valid = [c for c in after_industry if c["roe_factor"] is not None]
    roe_valid.sort(key=lambda x: x["roe_factor"], reverse=True)
    keep_roe = max(int(len(roe_valid) * 0.5), 1)
    after_roe = roe_valid[:keep_roe]
    print(f"  ROE筛选后: {len(after_roe)}")

    # Step 3: PDF "按过去5年平均股息率+TTM净利润增长率综合排名，取前300名"
    dy_ranks = rank_desc([c["dy5"] for c in after_roe])
    ng_ranks = rank_desc([c["np_growth"] for c in after_roe])

    for i, c in enumerate(after_roe):
        c["final_rank"] = dy_ranks[i] + ng_ranks[i]

    after_roe.sort(key=lambda x: x["final_rank"])
    top300 = after_roe[:300]

    print(f"  中金300 最终成分股: {len(top300)} 只")

    results = []
    for c in top300:
        results.append({
            "symbol": c["symbol"],
            "name": c["name"],
            "industry": c["industry"],
            "market_cap": c["market_cap"],
            "roe_factor": round(c["roe_factor"], 4) if c["roe_factor"] else None,
            "dividend_yield_5y": round(c["dy5"] * 100, 4) if c["dy5"] else None,
            "net_profit_growth": round(c["np_growth"], 4) if c["np_growth"] else None,
        })

    results = apply_market_cap_weight(results, cap=0.05)  # PDF: 自由流通市值加权，单票≤5%
    results.sort(key=lambda r: r["weight"], reverse=True)

    for r in results:
        del r["market_cap"]

    return results


# ═══════════════════════════════════════════════════════════
# 指数三: 科技100 (931187)
# ═══════════════════════════════════════════════════════════
#
# PDF编制方案:
#   样本空间: 中证全指中属于科技主题行业的股票
#   1. 剔除总市值或日均成交额排名后20%
#   2. 剔除股权质押比例排名前5%
#   3. 行业内基本面得分(近2年平均营收/现金流/净资产/毛利占比), 剔除后40%
#   4. 综合得分 = 研发强度×40% + 盈利能力×30% + 成长能力×30%, 取前100
#   加权: 自由流通市值加权，单票权重上限5%

# 科技主题行业关键词 (中证三级行业)
TECH_KEYWORDS = [
    "航空航天", "国防", "电网设备", "储能", "光伏", "风电",
    "医疗器械", "生物", "药品", "动物保健", "育种", "化学药", "制药",
    "数字媒体", "数字营销", "软件", "电子", "半导体", "通信",
    "计算机", "信息技术", "互联网", "人工智能", "芯片", "集成电路",
    "新能源", "科技", "IT服务",
]


def is_tech_stock(stock):
    """判断是否属于科技主题行业"""
    industry = get_industry(stock)
    if industry == "未知":
        return False
    return any(kw in industry for kw in TECH_KEYWORDS)


def build_tech_100(all_stocks):
    print("\n" + "=" * 60)
    print("  科技100 选股")
    print("=" * 60)

    # Step 1: 筛选科技行业
    tech_stocks = [(sym, stock) for sym, stock in all_stocks.items() if is_tech_stock(stock)]
    print(f"  科技行业股票: {len(tech_stocks)}")

    if len(tech_stocks) == 0:
        print("  无科技行业股票，跳过")
        return []

    # Step 2: PDF "剔除总市值或日均成交额任一排名后20%的股票"
    valid = []
    for sym, stock in tech_stocks:
        stats = stock.get("daily", {}).get("stats", {})
        turnover = safe(stats.get("avg_daily_turnover_approx")) or safe(stats.get("avg_daily_turnover"))
        mkt_cap = estimate_market_cap(stock)
        if turnover and mkt_cap:
            valid.append((sym, stock, mkt_cap, turnover))

    n = len(valid)
    if n == 0:
        print("  无有效数据，跳过")
        return []

    # 市值后20%
    valid_sorted_cap = sorted(valid, key=lambda x: x[2], reverse=True)
    cap_bottom20 = set(s for i, (s, _, _, _) in enumerate(valid_sorted_cap) if i >= n * 0.8)

    # 成交额后20%
    valid_sorted_turn = sorted(valid, key=lambda x: x[3], reverse=True)
    turn_bottom20 = set(s for i, (s, _, _, _) in enumerate(valid_sorted_turn) if i >= n * 0.8)

    valid = [(s, st, mc, t) for s, st, mc, t in valid
             if s not in cap_bottom20 and s not in turn_bottom20]
    print(f"  剔除市值/成交额后20%: {len(valid)}")

    # Step 3: PDF "剔除股权质押比例排名前5%的股票"
    pledges_with_val = [(sym, safe(stock.get("pledge", {}).get("pledge_ratio")))
                        for sym, stock, _, _ in valid]
    pledges_valid = [(s, p) for s, p in pledges_with_val if p is not None and p > 0]
    high_pledge = set()
    if pledges_valid:
        pledges_valid.sort(key=lambda x: x[1], reverse=True)
        cutoff_5pct = max(int(len(pledges_valid) * 0.05), 1)
        high_pledge = set(s for s, _ in pledges_valid[:cutoff_5pct])

    valid = [(s, st, mc, t) for s, st, mc, t in valid if s not in high_pledge]
    print(f"  剔除高质押后: {len(valid)}")

    # Step 4: PDF "行业内按基本面得分(近2年平均营收/现金流/净资产/毛利占行业比例)排名，剔除后40%"
    candidates = []
    for sym, stock, mkt_cap, turnover in valid:
        income = stock.get("financials", {}).get("income", [])
        cashflow = stock.get("financials", {}).get("cashflow", [])
        balance = stock.get("financials", {}).get("balance", [])
        indicators = stock.get("indicators", {}).get("records", [])

        avg_revenue = _mean_or_none(get_annual_records(income, "revenue", 2))
        avg_opcf = _mean_or_none(get_annual_records(cashflow, "operating_cashflow_net", 2))
        equity = safe(balance[0].get("total_equity_parent")) if balance else None

        # 毛利 = 营收 - 营业成本
        rev_2y = get_annual_records(income, "revenue", 2)
        cost_2y = get_annual_records(income, "cost_of_revenue", 2)
        gross_2y = [r - c for r, c in zip(rev_2y, cost_2y) if r is not None and c is not None]
        avg_gross = _mean_or_none(gross_2y)

        # 研发支出 (过去2年均值)
        avg_rd = _mean_or_none(get_annual_records(income, "rd_expense", 2))

        # 营业支出 (过去2年均值)
        avg_opcost = _mean_or_none(get_annual_records(income, "operating_cost", 2))

        # PB = price / bvps
        bvps = safe(indicators[0].get("bvps_adjusted")) if indicators else None
        price = safe(stock.get("daily", {}).get("stats", {}).get("latest_close"))
        pb = (price / bvps) if (price and bvps and bvps > 0) else None

        # 扣非净利润
        deducted_np = safe(indicators[0].get("deducted_net_profit")) if indicators else None

        # 成长指标
        rev_growth = compute_yoy_growth(income, "revenue")
        opcf_growth = compute_yoy_growth(cashflow, "operating_cashflow_net")

        candidates.append({
            "symbol": sym,
            "name": stock.get("name", ""),
            "industry": get_industry(stock),
            "mkt_cap": mkt_cap,
            "avg_revenue": avg_revenue,
            "avg_opcf": avg_opcf,
            "equity": equity,
            "avg_gross": avg_gross,
            "avg_rd": avg_rd,
            "avg_opcost": avg_opcost,
            "pb": pb,
            "deducted_np": deducted_np,
            "rev_growth": rev_growth,
            "opcf_growth": opcf_growth,
        })

    # 行业内基本面得分
    by_industry = {}
    for c in candidates:
        by_industry.setdefault(c["industry"], []).append(c)

    after_fundamental = []
    for ind, group in by_industry.items():
        n = len(group)
        if n == 0:
            continue

        total_rev = sum(c["avg_revenue"] for c in group if c["avg_revenue"]) or 1
        total_opcf = sum(abs(c["avg_opcf"]) for c in group if c["avg_opcf"]) or 1
        total_equity = sum(c["equity"] for c in group if c["equity"]) or 1
        total_gross = sum(c["avg_gross"] for c in group if c["avg_gross"]) or 1

        for c in group:
            scores = []
            if c["avg_revenue"]:
                scores.append(c["avg_revenue"] / total_rev)
            if c["avg_opcf"]:
                scores.append(abs(c["avg_opcf"]) / total_opcf)
            if c["equity"]:
                scores.append(c["equity"] / total_equity)
            if c["avg_gross"]:
                scores.append(c["avg_gross"] / total_gross)
            c["fundamental_score"] = float(np.mean(scores)) if scores else 0

        group.sort(key=lambda x: x["fundamental_score"], reverse=True)
        keep = max(int(n * 0.6), 1)
        after_fundamental.extend(group[:keep])

    print(f"  基本面筛选后: {len(after_fundamental)}")

    if not after_fundamental:
        return []

    # Step 5: PDF "综合得分 = 研发强度×40% + 盈利能力×30% + 成长能力×30%，取前100"
    n = len(after_fundamental)

    # --- 研发强度 (40%) ---
    # PDF: 指标1 = 近2年平均研发支出/PB, 指标2 = 近2年平均研发支出/日均总市值
    # 研发强度 = 指标1百分位×40% + 指标2百分位×60%
    ind1_vals = [c["avg_rd"] / c["pb"] if (c["avg_rd"] and c["pb"] and c["pb"] > 0) else None
                 for c in after_fundamental]
    ind2_vals = [c["avg_rd"] / c["mkt_cap"] if (c["avg_rd"] and c["mkt_cap"] and c["mkt_cap"] > 0) else None
                 for c in after_fundamental]

    rank1 = percentile_rank(ind1_vals)
    rank2 = percentile_rank(ind2_vals)
    rd_scores = [rank1.get(i, 0) * 0.4 + rank2.get(i, 0) * 0.6 for i in range(n)]

    # --- 盈利能力 (30%) ---
    # PDF: 指标3 = (营收-营支)/营收, 指标4 = 扣非净利/净资产, 指标5 = 经营现金流/营收
    # 盈利能力 = 三个指标百分位排名的均值
    ind3_vals = []
    for c in after_fundamental:
        if c["avg_revenue"] and c["avg_opcost"] and c["avg_revenue"] > 0:
            ind3_vals.append((c["avg_revenue"] - c["avg_opcost"]) / c["avg_revenue"])
        else:
            ind3_vals.append(None)

    ind4_vals = [c["deducted_np"] / c["equity"]
                 if (c["deducted_np"] and c["equity"] and c["equity"] > 0) else None
                 for c in after_fundamental]

    ind5_vals = [c["avg_opcf"] / c["avg_revenue"]
                 if (c["avg_opcf"] and c["avg_revenue"] and c["avg_revenue"] > 0) else None
                 for c in after_fundamental]

    rank3 = percentile_rank(ind3_vals)
    rank4 = percentile_rank(ind4_vals)
    rank5 = percentile_rank(ind5_vals)
    profit_scores = [(rank3.get(i, 0) + rank4.get(i, 0) + rank5.get(i, 0)) / 3 for i in range(n)]

    # --- 成长能力 (30%) ---
    # PDF: 指标6 = 营收TTM滚动增速, 指标7 = 经营现金流TTM滚动增速
    # 成长能力 = 两个指标百分位排名的均值
    ind6_vals = [c["rev_growth"] for c in after_fundamental]
    ind7_vals = [c["opcf_growth"] for c in after_fundamental]

    rank6 = percentile_rank(ind6_vals)
    rank7 = percentile_rank(ind7_vals)
    growth_scores = [(rank6.get(i, 0) + rank7.get(i, 0)) / 2 for i in range(n)]

    # PDF: 综合得分 = 研发强度×40% + 盈利能力×30% + 成长能力×30%
    for i, c in enumerate(after_fundamental):
        c["rd_score"] = rd_scores[i]
        c["profit_score"] = profit_scores[i]
        c["growth_score"] = growth_scores[i]
        c["composite_score"] = rd_scores[i] * 0.4 + profit_scores[i] * 0.3 + growth_scores[i] * 0.3

    after_fundamental.sort(key=lambda x: x["composite_score"], reverse=True)
    top100 = after_fundamental[:100]

    print(f"  科技100 最终成分股: {len(top100)} 只")

    results = []
    for c in top100:
        results.append({
            "symbol": c["symbol"],
            "name": c["name"],
            "industry": c["industry"],
            "market_cap": c["mkt_cap"],
            "composite_score": round(c["composite_score"], 6),
            "rd_score": round(c["rd_score"], 6),
            "profit_score": round(c["profit_score"], 6),
            "growth_score": round(c["growth_score"], 6),
        })

    results = apply_market_cap_weight(results, cap=0.05)  # PDF: 自由流通市值加权，单票≤5%
    results.sort(key=lambda r: r["weight"], reverse=True)

    for r in results:
        del r["market_cap"]

    return results


def _mean_or_none(values):
    """计算均值，空列表返回 None"""
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


# ═══════════════════════════════════════════════════════════
# 打印与输出
# ═══════════════════════════════════════════════════════════

def print_results(name, results, top_n=20):
    print(f"\n{'━' * 60}")
    print(f"  {name} 成分股 (共{len(results)}只, 显示前{min(top_n, len(results))}只)")
    print(f"{'━' * 60}")

    for i, r in enumerate(results[:top_n]):
        line = f"  {i + 1:3d}. {r['symbol']} {r['name']:8s}"
        if "dividend_yield_3y" in r:
            line += f"  股息率={r['dividend_yield_3y']:6.2f}%  波动率={r['volatility']:.4f}"
        if "roe_factor" in r:
            rf = r["roe_factor"]
            line += f"  ROE因子={rf:.2f}" if rf else "  ROE因子=N/A"
        if "composite_score" in r:
            line += f"  综合得分={r['composite_score']:.4f}"
        if "weight" in r:
            line += f"  权重={r['weight']:.2f}%"
        print(line)

    if len(results) > top_n:
        print(f"  ... 省略 {len(results) - top_n} 只 ...")


def main():
    all_stocks = load_all_stocks()

    # 红利低波100
    hl100 = build_dividend_low_vol_100(all_stocks)
    print_results("红利低波100", hl100)

    # 中金300
    zj300 = build_cicc_300(all_stocks)
    print_results("中金300", zj300)

    # 科技100
    kj100 = build_tech_100(all_stocks)
    print_results("科技100", kj100)

    # 保存结果
    output = {
        "generated_at": datetime.now().isoformat(),
        "stock_count": len(all_stocks),
        "note": f"基于 {len(all_stocks)} 只已爬取股票, 数据可能不完整",
        "dividend_low_vol_100": hl100,
        "cicc_300": zj300,
        "tech_100": kj100,
    }

    output_file = DATA_DIR / "index_results.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
