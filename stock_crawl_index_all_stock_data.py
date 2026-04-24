"""
爬取三指数候选池全部股票数据

两种模式:
  --mode full    全量爬取: 中证全指+全A股合并 → 逐只爬取详细数据
  --mode staged  分步粗筛(默认): 先获取全市场快照做粗筛, 再只爬通过粗筛的股票

使用 stock_fetch_data.py 中的数据获取函数，爬取：
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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd

from stock_fetch_data import (
    fetch_daily_price,
    fetch_financial_reports,
    fetch_financial_indicators,
    fetch_dividend_history,
    _safe_float,
    _retry_fetch,
    DATA_DIR,
)

OUTPUT_DIR = DATA_DIR / "stock_data"
CONS_FILE = DATA_DIR / "index_constituents.json"
SNAPSHOT_FILE = DATA_DIR / "market_snapshot.json"


# 科技主题行业关键词（中证二级/三级行业），用于科技100指数筛选科技行业股票
TECH_KEYWORDS = [
    "航空航天", "国防", "电网设备", "储能", "光伏", "风电",
    "医疗器械", "生物", "药品", "动物保健", "育种", "化学药", "制药",
    "数字媒体", "数字营销", "软件", "电子", "半导体", "通信",
    "计算机", "信息技术", "互联网", "人工智能", "芯片", "集成电路",
    "新能源", "科技", "IT服务",
]


def is_bse_stock(code: str) -> bool:
    """判断是否为北交所股票。北交所代码以 4 或 8 开头（如 430xxx, 830xxx, 920xxx 等）"""
    return code.startswith("4") or code.startswith("8") or code.startswith("9")


# ═══════════════════════════════════════════════════════════
# 公共工具
# ═══════════════════════════════════════════════════════════

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


def fetch_one_stock(symbol, name, pledge_map):
    """爬取单只股票的全部数据:
    - daily: 日频行情(收盘价/成交量/波动率等)
    - financials: 三大财务报表(利润表/资产负债表/现金流量表)
    - indicators: 财务指标(ROE/增长率/每股净资产/资产负债率等)
    - dividends: 分红历史(每10股派息/是否连续分红)
    - pledge: 股权质押数据(质押比例/质押笔数/所属行业)
    """
    data = {"symbol": symbol, "name": name, "fetch_time": datetime.now().isoformat()}

    data["daily"] = fetch_daily_price(symbol)
    time.sleep(0.5)

    data["financials"] = fetch_financial_reports(symbol)
    time.sleep(0.5)

    data["indicators"] = fetch_financial_indicators(symbol)
    time.sleep(0.5)

    data["dividends"] = fetch_dividend_history(symbol)
    time.sleep(0.3)

    data["pledge"] = pledge_map.get(symbol, {
        "pledge_ratio": None, "pledge_count": None,
        "trade_date": None, "industry": None,
    })

    return data


def _stock_file(symbol, name):
    safe_name = name.replace("/", "_").replace("\\", "_")
    return OUTPUT_DIR / f"CN_{symbol}_{safe_name}.json"


def load_existing():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = set()
    for f in OUTPUT_DIR.glob("CN_*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                json.load(fp)
            code = f.stem.split("_")[1]
            existing.add(code)
        except (json.JSONDecodeError, ValueError, KeyError):
            print(f"  数据损坏，已删除: {f.name}")
            f.unlink()
    return existing


def save_stock(symbol, data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _stock_file(symbol, data.get("name", symbol))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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

    def _worker(symbol, info):
        name = info["name"]
        try:
            stock_data = fetch_one_stock(symbol, name, pledge_map)
            stock_data["candidate_for"] = info.get("candidate_for", [])
            save_stock(symbol, stock_data)
            done_count[0] += 1
            print(f"  ✓ {name}({symbol}) 完成 ({done_count[0]}/{total})")
        except Exception as e:
            errors[symbol] = {"name": name, "error": str(e)}
            print(f"  ✗ {name}({symbol}) 失败: {e}")

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

    file_count = len(list(OUTPUT_DIR.glob("CN_*.json")))
    print(f"\n数据目录: {OUTPUT_DIR} ({file_count} 个文件)")


# ═══════════════════════════════════════════════════════════
# 方案1: 全量爬取
# ═══════════════════════════════════════════════════════════

def get_full_universe():
    """获取三指数需要的完整样本空间, 合并去重, 排除北交所"""
    stocks = {}

    # 1. 中证全指 (红利低波100 + 科技100 的样本空间)
    print("获取中证全指成分股(000985)...")
    df_all = _retry_fetch(ak.index_stock_cons, symbol="000985")
    for _, row in df_all.iterrows():
        code = row["品种代码"]
        if is_bse_stock(code):
            continue
        stocks[code] = {"name": row["品种名称"], "source": "中证全指"}
    print(f"  中证全指: {len(df_all)} 只")

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

    # 4. 中证全指成分股列表
    print("获取中证全指成分股列表(000985)...")
    df_csi = _retry_fetch(ak.index_stock_cons, symbol="000985")
    csi_all_set = set(df_csi["品种代码"].tolist())
    for code in spot_map:
        spot_map[code]["in_csi_all"] = code in csi_all_set
    print(f"  中证全指: {len(csi_all_set)} 只")

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
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="爬取三指数候选池股票数据")
    parser.add_argument("--mode", choices=["full", "staged"], default="staged",
                        help="full=全量爬取, staged=分步粗筛(默认)")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制爬取数量, 0=不限 (full模式默认20)")
    parser.add_argument("--workers", type=int, default=20,
                        help="并发线程数(默认20)")
    args = parser.parse_args()

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