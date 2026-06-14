"""
实验性：游资雷达 (Hot-Money Radar) —— 两种模式对应庄股三阶段

按「潜伏(吸筹) → 拉升 → 出货」三阶段框架：

【默认模式·拉升雷达】盘中捕捉正在进攻的票（已进入拉升期，节奏快、风险高）：
  盘口异动事件流 + 同花顺大单/即时资金流 + 涨停池 + 本地席位记忆。

【--ambush·潜伏雷达】★ 找游资正在吸筹、还没启动的票（第一阶段才赚大钱）：
  核心指纹是「钱持续进、价格没动」+ 吸筹K线形态，盘后每天跑一次即可：
  1. 多日资金流背离：同花顺 5/10/20 日资金净流入为正且占流通市值比高，
     但同期阶段涨幅很小（钱进了价格没动 = 有人在压价吸货）
  2. 吸筹K线形态（60日日线）：波动收敛、距前高有空间(低位)、温和放量
     而非爆量、收盘强度高（盘中砸下去总被接回）、下影线频繁
  3. 筹码集中：最近一期股东户数环比下降（散户被洗出、筹码向主力集中）
  4. 试盘痕迹：15~90天前上过龙虎榜（游资试盘）之后回落横盘 + 席位记忆
  硬剔除：近10日出现过涨停（已是拉升期，建仓风险大）、近10日涨幅>12%、
  近60日涨幅>50%（高位出货区）、日均成交<3000万（流动性陷阱）。

局限（实验性质，务必理解后再用）：
  - 盘中/盘后都不存在席位级实时数据，本质是「行为指纹」推断
  - 吸筹是以天~周为单位的过程，潜伏分数高不代表马上启动，可能潜伏数周
  - 分数为启发式权重；先用 --verify / --ambush-verify 积累命中率再参考

用法：
  python stock_hot_money_radar.py                    # 拉升雷达，扫一次 Top 20
  python stock_hot_money_radar.py --watch 60         # 拉升雷达，盘中每60秒刷新
  python stock_hot_money_radar.py --verify           # 盘后用龙虎榜验证拉升雷达
  python stock_hot_money_radar.py --ambush           # ★ 潜伏雷达（盘后跑）
  python stock_hot_money_radar.py --ambush-verify --date 2026-06-13 --horizon 5
                                                     # N天后回看潜伏票的前向收益
"""

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timedelta
from statistics import mean

import akshare as ak

from stock_crawl_capital import (
    DATA_DIR,
    _kline_lock,
    _num,
    _pad_visual,
    _retry,
    _symbol_with_prefix,
    fetch_stock_data,
)

RADAR_LATEST_FILE = DATA_DIR / "hot_money_radar.json"
AMBUSH_LATEST_FILE = DATA_DIR / "hot_money_ambush.json"
SCORED_FILE = DATA_DIR / "scored_stocks.json"

# 异动事件 → 进攻性权重（单类事件多次出现按次数累计，封顶见评分函数）
ALERT_CATEGORIES = {
    "封涨停板": 12,
    "大笔买入": 10,
    "火箭发射": 8,
    "有大买盘": 8,
    "竞价上涨": 6,
    "60日新高": 5,
}


# ─── 工具 ─────────────────────────────────────────────────────

def _parse_cn_amount(text):
    """'1.35亿'/'-4926.05万'/'3765' → float(元)；解析失败返回 0.0。"""
    if text is None:
        return 0.0
    s = str(text).strip().replace(",", "")
    if not s or s in ("-", "--", "nan"):
        return 0.0
    mult = 1.0
    if s.endswith("亿"):
        mult, s = 1e8, s[:-1]
    elif s.endswith("万"):
        mult, s = 1e4, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return 0.0


def _parse_pct(text):
    """'5.11%' / 5.11 → 5.11；失败返回 None。"""
    if text is None:
        return None
    s = str(text).strip().rstrip("%")
    if not s or s in ("-", "--", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _norm_code(value):
    code = re.sub(r"\D", "", str(value))[:6]
    return code.zfill(6) if code else ""


def _is_small_board_excluded(code, name):
    """硬排除：北交所、ST、N/C 新股次新。"""
    if code.startswith(("4", "8", "9")):
        return True
    upper = str(name).upper()
    if "ST" in upper:
        return True
    if upper.startswith(("N", "C")) and not upper.startswith("CW"):  # 东财/同花顺新股前缀
        return True
    return False


def _radar_day_file(date_str):
    return DATA_DIR / f"hot_money_radar_{date_str.replace('-', '')}.json"


def _is_trading_time(now=None):
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    hm = now.hour * 100 + now.minute
    return 915 <= hm <= 1135 or 1255 <= hm <= 1505


# ─── 数据抓取（任一失败降级为空，不阻塞整体）──────────────────

def fetch_intraday_alerts():
    """盘口异动事件流 → {code: {"events": Counter, "name": str, "last_time": str}}"""
    out = {}
    for category in ALERT_CATEGORIES:
        try:
            df = _retry(ak.stock_changes_em, symbol=category)
        except Exception as e:
            print(f"  [WARN] 盘口异动({category}) 获取失败: {e}")
            continue
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            code = _norm_code(row.get("代码"))
            if not code:
                continue
            d = out.setdefault(code, {"events": Counter(), "name": str(row.get("名称", "")), "last_time": ""})
            d["events"][category] += 1
            t = str(row.get("时间", ""))
            if t > d["last_time"]:
                d["last_time"] = t
    return out


def fetch_ths_flow():
    """同花顺即时资金流（全市场）→ {code: 行情/资金字段 + 反推流通市值}"""
    df = _retry(ak.stock_fund_flow_individual, symbol="即时")
    out = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        code = _norm_code(row.get("股票代码"))
        if not code:
            continue
        turnover = _parse_pct(row.get("换手率"))
        amount = _parse_cn_amount(row.get("成交额"))
        # 流通市值 ≈ 成交额 / 换手率；换手过低时估计噪音太大，置空
        float_cap = amount / (turnover / 100.0) if turnover and turnover >= 0.5 and amount > 0 else None
        out[code] = {
            "name": str(row.get("股票简称", "")),
            "price": _num(row.get("最新价")),
            "chg_pct": _parse_pct(row.get("涨跌幅")),
            "turnover_pct": turnover,
            "net_inflow": _parse_cn_amount(row.get("净额")),
            "amount": amount,
            "float_cap_est": float_cap,
        }
    return out


def fetch_big_deals():
    """同花顺逐笔大单 → {code: {"buy_amt","sell_amt","buy_cnt","sell_cnt","net_ratio"}}"""
    df = _retry(ak.stock_fund_flow_big_deal)
    out = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        code = _norm_code(row.get("股票代码"))
        if not code:
            continue
        d = out.setdefault(code, {"buy_amt": 0.0, "sell_amt": 0.0, "buy_cnt": 0, "sell_cnt": 0})
        nature = str(row.get("大单性质", ""))
        amount = _parse_cn_amount(row.get("成交额")) or _num(row.get("成交额"))
        if "买" in nature:
            d["buy_amt"] += amount
            d["buy_cnt"] += 1
        elif "卖" in nature:
            d["sell_amt"] += amount
            d["sell_cnt"] += 1
    for d in out.values():
        total = d["buy_amt"] + d["sell_amt"]
        d["net_ratio"] = (d["buy_amt"] - d["sell_amt"]) / total if total > 0 else 0.0
    return out


MAIN_CAPITAL_TTL_SEC = 600  # ① 主力分级资金缓存(秒)：盘中10分钟内复用，控 push2 频率（东财分级唯一源、易限流）


def _main_capital_cache_file(indicator):
    return DATA_DIR / f"main_capital_{indicator}.json"


def _parse_main_capital_df(df):
    """解析 stock_individual_fund_flow_rank 返回 → {code:{main_net,main_ratio,small_net}}；列名容错。"""
    cols = list(df.columns)
    find = lambda key: next((c for c in cols if key in c), None)
    code_col = find("代码")
    main_net_col = find("主力净流入-净额") or find("主力净流入")
    main_ratio_col = find("主力净流入-净占比") or find("主力净流入净占比")
    small_net_col = find("小单净流入-净额") or find("小单净流入")
    if not code_col or not main_net_col:
        return {}
    out = {}
    for _, row in df.iterrows():
        code = _norm_code(row.get(code_col))
        if not code:
            continue
        out[code] = {
            "main_net": _num(row.get(main_net_col)),
            "main_ratio": _parse_pct(row.get(main_ratio_col)) if main_ratio_col else None,
            "small_net": _num(row.get(small_net_col)) if small_net_col else None,
        }
    return out


def fetch_main_capital_flow(indicator="今日", ttl_sec=MAIN_CAPITAL_TTL_SEC):
    """东财全市场个股资金分级 → {code: {main_net, main_ratio, small_net}}。
    indicator: "今日"(拉升用) / "5日" / "10日"(潜伏吸筹用)。main=主力(超大+大单)净额，
    small=小单净额；用于 ①「主力进散户出」背离判断。

    带本地缓存(默认 ttl_sec 内复用)+退避重试+旧缓存兜底，把对 push2 的调用压到最低
    （东财分级是唯一源、对频率极敏感）；需代理关闭直连。彻底无数据→{}（背离降级为0）。"""
    cache_fp = _main_capital_cache_file(indicator)
    cached = None
    try:
        with open(cache_fp, encoding="utf-8") as f:
            cached = json.load(f)
        if time.time() - cached.get("fetched_ts", 0) < ttl_sec and cached.get("data"):
            return cached["data"]   # 新鲜缓存命中，不打 push2
    except (OSError, json.JSONDecodeError):
        cached = None

    df = _retry(ak.stock_individual_fund_flow_rank, indicator=indicator)
    out = _parse_main_capital_df(df) if df is not None and not getattr(df, "empty", True) else {}
    if not out:
        # 抓取失败/解析空 → 用旧缓存兜底（任意时效，聊胜于无）
        return cached["data"] if cached and cached.get("data") else {}

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_fp, "w", encoding="utf-8") as f:
            json.dump({"fetched_ts": time.time(), "indicator": indicator, "data": out}, f, ensure_ascii=False)
    except OSError:
        pass
    return out


def fetch_limit_pool():
    """涨停池（取最近一个有数据的交易日）→ {code: 封板行为字段}"""
    for back in range(6):
        date = (datetime.now() - timedelta(days=back)).strftime("%Y%m%d")
        try:
            df = _retry(ak.stock_zt_pool_em, date=date)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        out = {}
        for _, row in df.iterrows():
            code = _norm_code(row.get("代码"))
            if not code:
                continue
            out[code] = {
                "seal_amount": _num(row.get("封板资金")),
                "float_cap": _num(row.get("流通市值")),
                "first_seal_time": str(row.get("首次封板时间", ""))[:8],
                "break_times": int(_num(row.get("炸板次数"))),
                "consecutive": int(_num(row.get("连板数"))),
                "industry": str(row.get("所属行业", "")).strip(),
            }
        return out, date
    return {}, None


def load_seat_memory():
    """本地席位记忆：近一轮龙虎榜爬取里被 Top 席位买过的票。"""
    try:
        with open(SCORED_FILE, encoding="utf-8") as f:
            scored = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    memory = {}
    for stock in scored.get("stocks", []):
        code = _norm_code(stock.get("code"))
        followers = stock.get("followers") or []
        dates = [str(f.get("date", ""))[:10] for f in followers if isinstance(f, dict)]
        memory[code] = {
            "seats": stock.get("total_buyers") or len(followers),
            "known": stock.get("known_seat_count") or 0,
            "last_date": max(dates) if dates else "",
        }
    return memory


# ─── ③ 板块共振：个股→行业映射（东财主源 / 新浪回退 / 本地缓存）──

INDUSTRY_MAP_FILE = DATA_DIR / "industry_map.json"
INDUSTRY_MAP_TTL_DAYS = 7   # 行业归属变化极慢，缓存长期复用，摊薄建映射的请求量


def _build_industry_map_em():
    """东财行业成分遍历 → {code: 行业名}（与涨停池"所属行业"同口径）。失败返回 {}。"""
    names = _retry(ak.stock_board_industry_name_em)
    if names is None or names.empty:
        return {}
    col = "板块名称" if "板块名称" in names.columns else names.columns[1]
    mapping = {}
    for ind in names[col].dropna().tolist():
        cons = _retry(ak.stock_board_industry_cons_em, symbol=str(ind))
        if cons is None or cons.empty:
            continue
        code_col = next((c for c in cons.columns if "代码" in c), None)
        if not code_col:
            continue
        for raw in cons[code_col]:
            code = _norm_code(raw)
            if code:
                mapping[code] = str(ind)
        time.sleep(0.15)
    return mapping


def _build_industry_map_sina():
    """新浪板块成分遍历 → {code: 行业名}（新浪口径，作东财不可用时的回退）。失败返回 {}。"""
    spot = _retry(ak.stock_sector_spot, indicator="行业")
    if spot is None or spot.empty or "label" not in spot.columns:
        return {}
    name_col = "板块" if "板块" in spot.columns else spot.columns[1]
    mapping = {}
    for label, ind in zip(spot["label"], spot[name_col]):
        det = _retry(ak.stock_sector_detail, sector=str(label))
        if det is None or det.empty:
            continue
        code_col = "code" if "code" in det.columns else next(
            (c for c in det.columns if "代码" in c or c == "symbol"), None)
        if not code_col:
            continue
        for raw in det[code_col]:
            code = _norm_code(raw)
            if code:
                mapping[code] = str(ind)
        time.sleep(0.15)
    return mapping


def build_industry_map(force=False):
    """个股→行业 映射：东财主源 + 新浪回退 + 本地缓存(TTL内复用) + 旧缓存兜底。

    返回 (mapping{code:行业名}, source)。两源口径不同（东财与涨停池一致；新浪为新浪分类），
    缓存里记 source 供上层判断板块热度该用哪套口径。
    """
    if not force:
        try:
            with open(INDUSTRY_MAP_FILE, encoding="utf-8") as f:
                cached = json.load(f)
            fetched = datetime.strptime(cached.get("fetched", "2000-01-01"), "%Y-%m-%d")
            if (datetime.now() - fetched).days <= INDUSTRY_MAP_TTL_DAYS and cached.get("map"):
                return cached["map"], cached.get("source", "cache")
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    mapping, source = _build_industry_map_em(), "eastmoney"
    if not mapping:
        print("  [行业映射] 东财不可用，回退新浪...")
        mapping, source = _build_industry_map_sina(), "sina"

    if mapping:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(INDUSTRY_MAP_FILE, "w", encoding="utf-8") as f:
            json.dump({"fetched": datetime.now().strftime("%Y-%m-%d"),
                       "source": source, "map": mapping}, f, ensure_ascii=False)
        print(f"  [行业映射] {source} 建成 {len(mapping)} 只，缓存 {INDUSTRY_MAP_TTL_DAYS} 天")
        return mapping, source

    # 两源都失败 → 过期旧缓存兜底
    try:
        with open(INDUSTRY_MAP_FILE, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("map"):
            print("  [行业映射] 实时源全失败，沿用旧缓存")
            return cached["map"], cached.get("source", "stale-cache")
    except (OSError, json.JSONDecodeError):
        pass
    return {}, None


# ─── 评分 ─────────────────────────────────────────────────────

def score_candidate(code, flow, alerts, deals, pool, memory, float_rank_pct,
                    capital=None, resonance_ctx=None):
    """返回 (score 0-100, reasons, parts)。各分项为启发式权重，实验性质。
    100分重配：攻28 + 背14 + 封22 + 小16 + 共10 + 忆10。"""
    reasons = []
    parts = {}

    # 1) 资金攻击 (0-28)：异动事件 + 大单净买 + 净流入占比
    alert_pts = 0.0
    if alerts:
        for cat, cnt in alerts["events"].items():
            alert_pts += ALERT_CATEGORIES[cat] * min(cnt, 3)
        top_events = "、".join(f"{c}×{n}" for c, n in alerts["events"].most_common(2))
        reasons.append(top_events)
    alert_pts = min(alert_pts, 18.0)

    deal_pts = 0.0
    if deals and (deals["buy_cnt"] + deals["sell_cnt"]) >= 2:
        deal_pts = max(0.0, deals["net_ratio"]) * 8.0
        if deals["net_ratio"] >= 0.3:
            reasons.append(f"大单净买{deals['net_ratio'] * 100:.0f}%")

    inflow_pts = 0.0
    if flow["amount"] > 0 and flow["net_inflow"] > 0:
        inflow_ratio = flow["net_inflow"] / flow["amount"]
        inflow_pts = min(inflow_ratio / 0.20, 1.0) * 4.0
    parts["attack"] = round(min(alert_pts + deal_pts + inflow_pts, 28.0), 1)

    # 2) 主力-散户背离 (0-14)：主力净占比 + 主力进散户出 (①，东财分级资金)
    div_pts = 0.0
    if capital:
        main_net = capital.get("main_net")
        main_ratio = capital.get("main_ratio")
        small_net = capital.get("small_net")
        if main_ratio is not None:
            div_pts += min(max(main_ratio, 0.0) / 10.0, 1.0) * 8.0
            if main_ratio >= 3.0:
                reasons.append(f"主力净占比{main_ratio:.1f}%")
        if main_net is not None and small_net is not None:
            if main_net > 0 and small_net < 0:
                div_pts += 6.0
                reasons.append("主力进散户出")
            elif main_net < 0 and small_net > 0:
                div_pts = 0.0  # 主力出散户接=派发，背离分清零
                reasons.append("⚠主力出散户进")
    parts["divergence"] = round(min(div_pts, 14.0), 1)

    # 3) 涨停行为 (0-22)
    seal_pts = 0.0
    if pool:
        if pool["float_cap"] > 0 and pool["seal_amount"] > 0:
            seal_ratio = pool["seal_amount"] / pool["float_cap"]
            seal_pts += min(seal_ratio / 0.08, 1.0) * 11.0
            if seal_ratio >= 0.02:
                reasons.append(f"封单/流通{seal_ratio * 100:.1f}%")
        first = pool["first_seal_time"].replace(":", "")
        if first and first <= "100000":
            seal_pts += 7.0
            reasons.append(f"首封{first[:2]}:{first[2:4]}")
        elif first and first <= "133000":
            seal_pts += 4.0
        if pool["break_times"] == 0:
            seal_pts += 4.0
        else:
            seal_pts -= pool["break_times"] * 2.0
            reasons.append(f"炸板{pool['break_times']}次")
        consec_bonus = {1: 4.0, 2: 3.0, 3: 2.0}
        seal_pts += consec_bonus.get(pool["consecutive"], 0.0)
        if pool["consecutive"] >= 1:
            reasons.append(f"{pool['consecutive']}连板")
    parts["seal"] = round(max(0.0, min(seal_pts, 22.0)), 1)

    # 4) 小票弹性 (0-16)：流通市值越小越高 + 换手甜区
    size_pts = (1.0 - float_rank_pct) * 10.0
    turnover = flow.get("turnover_pct")
    if turnover is not None:
        if 8.0 <= turnover <= 25.0:
            size_pts += 6.0
        elif 4.0 <= turnover < 8.0 or 25.0 < turnover <= 35.0:
            size_pts += 3.0
    parts["size"] = round(min(size_pts, 16.0), 1)
    if flow.get("float_cap_est"):
        reasons.append(f"流通约{flow['float_cap_est'] / 1e8:.0f}亿")

    # 5) 板块共振 (0-10)：所属行业涨停潮 + 候选扎堆 (③)
    res_pts = 0.0
    if resonance_ctx:
        ind = (resonance_ctx.get("industry_map") or {}).get(code)
        if ind:
            zt = (resonance_ctx.get("zt_counter") or {}).get(ind, 0)
            cd = (resonance_ctx.get("cand_counter") or {}).get(ind, 0)
            if zt >= 3:
                res_pts += 6.0
            elif zt == 2:
                res_pts += 4.0
            elif zt == 1:
                res_pts += 2.0
            if cd >= 4:
                res_pts += 4.0
            elif cd >= 2:
                res_pts += 2.0
            if res_pts > 0:
                reasons.append(f"{ind}板块涨停{zt}/候选{cd}")
    parts["resonance"] = round(min(res_pts, 10.0), 1)

    # 6) 席位记忆 (0-10)：游资回头客
    mem_pts = 0.0
    if memory:
        if memory["seats"] >= 2:
            mem_pts += 5.0
        elif memory["seats"] >= 1:
            mem_pts += 3.0
        mem_pts += min(memory["known"], 3) * 1.5
        if memory["known"] > 0:
            reasons.append(f"知名席位{memory['known']}个近期买过")
        elif memory["seats"] > 0:
            reasons.append("追踪席位近期买过")
    parts["memory"] = round(min(mem_pts, 10.0), 1)

    score = sum(parts.values())
    return round(score, 1), reasons, parts


def scan(top_n=20, max_float_cap_yi=150.0, min_float_cap_yi=50.0, min_chg_pct=-2.0):
    """单轮扫描：抓数据 → 过滤小票 → 评分排序，返回 picks 列表。"""
    print(f"[{datetime.now():%H:%M:%S}] 拉取盘中数据...")
    alerts_map = fetch_intraday_alerts()
    print(f"  盘口异动: {len(alerts_map)} 只")
    flow_map = fetch_ths_flow()
    print(f"  即时资金流: {len(flow_map)} 只")
    deals_map = fetch_big_deals()
    print(f"  大单追踪: {len(deals_map)} 只")
    pool_map, pool_date = fetch_limit_pool()
    print(f"  涨停池({pool_date}): {len(pool_map)} 只")
    memory_map = load_seat_memory()
    print(f"  席位记忆: {len(memory_map)} 只")
    capital_map = fetch_main_capital_flow()
    print(f"  主力分级资金: {len(capital_map)} 只")
    industry_map, ind_src = build_industry_map()
    print(f"  行业映射({ind_src}): {len(industry_map)} 只")

    # 候选 = 有进攻信号的票（异动 ∪ 涨停池 ∪ 大单净买 ∪ 席位记忆），再做小票过滤
    candidate_codes = set(alerts_map) | set(pool_map) | set(memory_map)
    candidate_codes |= {c for c, d in deals_map.items() if d["net_ratio"] > 0}

    candidates = []
    for code in candidate_codes:
        flow = flow_map.get(code)
        if not flow:
            continue
        if _is_small_board_excluded(code, flow["name"]):
            continue
        price = flow.get("price")
        if price is None or not (2.0 <= price <= 80.0):
            continue
        chg = flow.get("chg_pct")
        if chg is None or chg < min_chg_pct:
            continue
        float_cap = flow.get("float_cap_est") or (pool_map.get(code, {}) or {}).get("float_cap")
        if float_cap and float_cap > max_float_cap_yi * 1e8:
            continue
        # 过小的流通盘流动性差、容易操纵也容易闷杀，默认 50 亿以下不要
        if float_cap and min_float_cap_yi and float_cap < min_float_cap_yi * 1e8:
            continue
        flow = dict(flow)
        flow["float_cap_est"] = float_cap
        candidates.append((code, flow))

    # 流通市值横截面排名（越小 rank 越低 → 弹性分越高）；缺失按中位处理
    caps = sorted(c[1]["float_cap_est"] for c in candidates if c[1]["float_cap_est"])
    def cap_rank(value):
        if not caps or not value:
            return 0.5
        idx = sum(1 for v in caps if v <= value)
        return idx / len(caps)

    # 板块共振上下文：涨停池各行业涨停数 + 候选池同行业扎堆数
    zt_counter = Counter(d["industry"] for d in pool_map.values() if d.get("industry"))
    cand_counter = Counter()
    for cand_code, _flow in candidates:
        ind = industry_map.get(cand_code)
        if ind:
            cand_counter[ind] += 1
    resonance_ctx = {"industry_map": industry_map, "zt_counter": zt_counter, "cand_counter": cand_counter}

    picks = []
    for code, flow in candidates:
        score, reasons, parts = score_candidate(
            code, flow,
            alerts_map.get(code), deals_map.get(code),
            pool_map.get(code), memory_map.get(code),
            cap_rank(flow["float_cap_est"]),
            capital=capital_map.get(code), resonance_ctx=resonance_ctx,
        )
        picks.append({
            "code": code,
            "name": flow["name"],
            "price": flow["price"],
            "chg_pct": flow["chg_pct"],
            "float_cap_yi": round(flow["float_cap_est"] / 1e8, 1) if flow["float_cap_est"] else None,
            "score": score,
            "parts": parts,
            "reasons": reasons[:5],
            "alert_last_time": (alerts_map.get(code) or {}).get("last_time", ""),
        })
    picks.sort(key=lambda p: -p["score"])
    return picks[:top_n], len(candidates)


# ─── 输出与落盘 ───────────────────────────────────────────────

def print_picks(picks, candidate_count):
    print()
    print("=" * 110)
    print(f"  游资买入雷达（实验） · 候选 {candidate_count} → Top {len(picks)}"
          f" · 攻/背/封/小/共/忆 = 资金攻击/主力背离/涨停行为/小票弹性/板块共振/席位记忆")
    print("=" * 110)
    header = (
        _pad_visual("代码", 8) + _pad_visual("名称", 12) + _pad_visual("现价", 8)
        + _pad_visual("涨幅%", 8) + _pad_visual("流通亿", 8) + _pad_visual("总分", 7)
        + _pad_visual("攻", 6) + _pad_visual("背", 6) + _pad_visual("封", 6)
        + _pad_visual("小", 6) + _pad_visual("共", 6) + _pad_visual("忆", 6)
        + "理由"
    )
    print(header)
    print("-" * 110)
    for p in picks:
        print(
            _pad_visual(p["code"], 8)
            + _pad_visual(p["name"], 12)
            + _pad_visual(f"{p['price']:.2f}" if p["price"] else "-", 8)
            + _pad_visual(f"{p['chg_pct']:+.1f}" if p["chg_pct"] is not None else "-", 8)
            + _pad_visual(f"{p['float_cap_yi']:.0f}" if p["float_cap_yi"] else "-", 8)
            + _pad_visual(f"{p['score']:.1f}", 7)
            + _pad_visual(f"{p['parts']['attack']:.0f}", 6)
            + _pad_visual(f"{p['parts']['divergence']:.0f}", 6)
            + _pad_visual(f"{p['parts']['seal']:.0f}", 6)
            + _pad_visual(f"{p['parts']['size']:.0f}", 6)
            + _pad_visual(f"{p['parts']['resonance']:.0f}", 6)
            + _pad_visual(f"{p['parts']['memory']:.0f}", 6)
            + " | ".join(p["reasons"])
        )
    print("=" * 110)
    if not _is_trading_time():
        print("注: 当前非交易时间，以上为最近交易日的全天口径数据。")
    print("注: 实验性推断，非席位级实锤；用 --verify 在盘后对照龙虎榜统计命中率。")


def persist_picks(picks, candidate_count):
    """最新结果 + 当日累积文件（记录每票当天首次被雷达发现的时间，供验证提前性）。"""
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    payload = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_count": candidate_count,
        "picks": picks,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RADAR_LATEST_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    day_file = _radar_day_file(today)
    try:
        with open(day_file, encoding="utf-8") as f:
            day = json.load(f)
    except (OSError, json.JSONDecodeError):
        day = {"date": today, "scans": 0, "stocks": {}}
    day["scans"] += 1
    for p in picks:
        entry = day["stocks"].setdefault(p["code"], {
            "name": p["name"],
            "first_seen_at": now.strftime("%H:%M:%S"),
            "best_score": p["score"],
        })
        entry["best_score"] = max(entry["best_score"], p["score"])
        entry["last_score"] = p["score"]
    with open(day_file, "w", encoding="utf-8") as f:
        json.dump(day, f, ensure_ascii=False, indent=2)
    print(f"  → 已落盘 {RADAR_LATEST_FILE} / {day_file}")


# ─── 盘后验证 ─────────────────────────────────────────────────

def verify(date_str=None):
    """用当日龙虎榜每日明细对照雷达当天发现的票，统计命中与提前量。"""
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    day_file = _radar_day_file(date_str)
    try:
        with open(day_file, encoding="utf-8") as f:
            day = json.load(f)
    except (OSError, json.JSONDecodeError):
        print(f"[ERROR] 找不到 {day_file}，当天没有跑过雷达。")
        return
    compact = date_str.replace("-", "")
    df = _retry(ak.stock_lhb_detail_em, start_date=compact, end_date=compact)
    lhb = {}
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            code = _norm_code(row.get("代码"))
            if code:
                lhb[code] = {
                    "net_buy": _num(row.get("龙虎榜净买额")),
                    "reason": str(row.get("上榜原因", "")),
                }
    if not lhb:
        print(f"[WARN] {date_str} 无龙虎榜数据（可能尚未公布或为非交易日）。")
        return

    stocks = day.get("stocks", {})
    hits = []
    for code, info in stocks.items():
        if code in lhb:
            hits.append((code, info, lhb[code]))
    print("=" * 96)
    print(f"  雷达验证 {date_str} · 雷达当日累计发现 {len(stocks)} 只 · 当晚上榜 {len(hits)} 只"
          f" · 命中率 {len(hits) / len(stocks) * 100:.1f}%" if stocks else "  雷达当日无记录")
    print("=" * 96)
    for code, info, l in sorted(hits, key=lambda x: -x[2]["net_buy"]):
        sign = "✓净买" if l["net_buy"] > 0 else "✗净卖"
        print(f"  {code} {_pad_visual(info['name'], 12)} 雷达首见 {info['first_seen_at']}"
              f"  最高分 {info['best_score']:>5.1f}  当晚{sign} {l['net_buy'] / 1e8:+.2f}亿")
    misses = [c for c in lhb if c not in stocks]
    print(f"\n  当晚上榜但雷达未覆盖: {len(misses)} 只（含大票/尾盘异动等）")


# ─── 潜伏模式（吸筹检测）──────────────────────────────────────

def _ambush_day_file(date_str):
    return DATA_DIR / f"hot_money_ambush_{date_str.replace('-', '')}.json"


def fetch_ths_flow_rank(symbol):
    """同花顺阶段资金流排行（5日/10日/20日）→ {code: {chg_pct, net_inflow, turnover_cont}}"""
    df = _retry(ak.stock_fund_flow_individual, symbol=symbol)
    out = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        code = _norm_code(row.get("股票代码"))
        if not code:
            continue
        out[code] = {
            "chg_pct": _parse_pct(row.get("阶段涨跌幅")),
            "net_inflow": _parse_cn_amount(row.get("资金流入净额")),
            "turnover_cont": _parse_pct(row.get("连续换手率")),
        }
    return out


def fetch_lhb_history(days=90):
    """近 days 天龙虎榜逐日事件 → {code: [(上榜日, 净买额), ...]}（升序）"""
    end = datetime.now()
    start = end - timedelta(days=days)
    df = _retry(ak.stock_lhb_detail_em,
                start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
    out = {}
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        code = _norm_code(row.get("代码"))
        if not code:
            continue
        out.setdefault(code, []).append(
            (str(row.get("上榜日", ""))[:10], _num(row.get("龙虎榜净买额")))
        )
    for events in out.values():
        events.sort()
    return out


def fetch_lhb_recent_dates(days=90):
    """近 days 天龙虎榜上榜日 → {code: {"dates": [...], "net_buy": 合计}}（识别试盘痕迹）"""
    return {
        code: {"dates": [d for d, _ in events], "net_buy": sum(n for _, n in events)}
        for code, events in fetch_lhb_history(days).items()
    }


def fetch_holder_concentration(code):
    """最近一期股东户数环比增减比例(%)。负值=户数下降=筹码集中。失败返回 (None, None)。"""
    df = _retry(ak.stock_zh_a_gdhs_detail_em, symbol=code, retries=2)
    if df is None or df.empty:
        return None, None
    last = df.iloc[-1]
    return _num(last.get("股东户数-增减比例")), str(last.get("股东户数统计截止日", ""))[:10]


def analyze_kline_shape(records, code):
    """60日K线吸筹形态分析。返回 (指标dict, None)；被硬过滤剔除时返回 (None, 原因)。"""
    if len(records) < 40:
        return None, "历史不足40日(次新)"
    limit_pct = 19.5 if code.startswith(("30", "68")) else 9.5
    last10 = records[-10:]
    if any(r["change_pct"] >= limit_pct for r in last10):
        return None, "近10日有涨停(已是拉升期)"
    closes = [r["close"] for r in records]
    if closes[-11] > 0 and (closes[-1] / closes[-11] - 1) * 100 > 12:
        return None, "近10日涨幅>12%(疑似已启动)"
    if closes[0] > 0 and (closes[-1] / closes[0] - 1) * 100 > 50:
        return None, "近60日涨幅>50%(高位出货区)"
    if mean(r["amount"] for r in records[-5:]) < 3e7:
        return None, "日均成交<3000万(流动性差)"

    def day_range(r):
        return (r["high"] - r["low"]) / r["close"] if r["close"] > 0 else 0.0

    range10 = mean(day_range(r) for r in last10)
    range60 = mean(day_range(r) for r in records)
    high60 = max(r["high"] for r in records)
    lo, hi = min(closes), max(closes)
    vols = [r["volume"] for r in records]
    vol5 = mean(vols[-5:])
    vol_base = mean(vols[-25:-5]) if len(vols) >= 25 else mean(vols[:-5])
    strengths = [(r["close"] - r["low"]) / (r["high"] - r["low"])
                 for r in last10 if r["high"] > r["low"]]
    lower_shadows = sum(
        1 for r in last10
        if min(r["close"], r["open"]) - r["low"] > abs(r["close"] - r["open"])
    )
    return {
        "contraction": range10 / range60 if range60 > 0 else 1.0,   # <1 = 波动收敛
        "drawdown_pct": (high60 - closes[-1]) / high60 * 100 if high60 > 0 else 0.0,
        "position": (closes[-1] - lo) / (hi - lo) if hi > lo else 0.5,
        "vol_lift": vol5 / vol_base if vol_base > 0 else 1.0,       # 温和放量 1.1~2.5
        "close_strength": mean(strengths) if strengths else 0.5,    # 收盘强度 0~1
        "lower_shadows": lower_shadows,                              # 近10日下影线天数
    }, None


def ambush_divergence_pts(capital, cap=14.0):
    """① 主力-散户背离分（潜伏用多日分级资金）。返回 (pts, reason)。
    无数据→(0,'')；主力进散户出→加分；主力出散户进→清零并告警。"""
    if not capital:
        return 0.0, ""
    main_net, main_ratio, small_net = (
        capital.get("main_net"), capital.get("main_ratio"), capital.get("small_net"))
    pts, reason = 0.0, ""
    if main_ratio is not None:
        pts += min(max(main_ratio, 0.0) / 8.0, 1.0) * 8.0   # 吸筹期主力占比阈值低于拉升
    if main_net is not None and small_net is not None:
        if main_net > 0 and small_net < 0:
            pts += 6.0
            reason = f"10日主力进散户出({main_ratio:.1f}%)" if main_ratio is not None else "10日主力进散户出"
        elif main_net < 0 and small_net > 0:
            pts = 0.0
            reason = "⚠10日主力出散户进"
    return round(min(pts, cap), 1), reason


def score_ambush(flow5, flow10, flow20, float_cap, shape, lhb_info, memory, as_of=None):
    """潜伏分（不含筹码分项，户数在终选阶段补充）。返回 (score, reasons, parts)。

    as_of: 回测时传历史时点，试盘痕迹的"天数差"按该时点计算。
    """
    reasons = []
    parts = {}

    # 1) 资金潜入 (0-35)：多日净流入占流通市值比 + 钱进价不动
    r5 = flow5["net_inflow"] / float_cap if float_cap else 0.0
    r10 = flow10["net_inflow"] / float_cap if float_cap else 0.0
    pts = min(max(r5, 0.0) / 0.02, 1.0) * 15 + min(max(r10, 0.0) / 0.035, 1.0) * 12
    if flow20 and flow20["net_inflow"] > 0:
        pts += 8.0
    parts["inflow"] = round(min(pts, 35.0), 1)
    reasons.append(f"5日净流入{flow5['net_inflow'] / 1e8:.2f}亿(占流通{r5 * 100:.1f}%)"
                   f"而涨幅仅{flow5['chg_pct']:+.1f}%")
    if flow20 and flow20["net_inflow"] > 0:
        reasons.append("20日资金亦为净流入")

    # 2) 吸筹形态 (0-25)
    s = 0.0
    if shape["contraction"] <= 0.6:
        s += 8.0
        reasons.append("波动显著收敛")
    elif shape["contraction"] <= 0.8:
        s += 6.0
        reasons.append("波动收敛")
    if shape["drawdown_pct"] >= 20:
        s += 6.0
        reasons.append(f"距60日高点-{shape['drawdown_pct']:.0f}%")
    elif shape["drawdown_pct"] >= 10:
        s += 3.0
    if shape["position"] <= 0.45:
        s += 2.0
    if 1.1 <= shape["vol_lift"] <= 2.5:
        s += 6.0
        reasons.append(f"温和放量×{shape['vol_lift']:.1f}")
    elif 2.5 < shape["vol_lift"] <= 4.0:
        s += 2.0
    if shape["close_strength"] >= 0.58:
        s += 5.0
        reasons.append("收盘强度高(砸盘被接)")
    elif shape["close_strength"] >= 0.52:
        s += 3.0
    if shape["lower_shadows"] >= 4:
        s += 2.0
    parts["shape"] = round(min(s, 25.0), 1)

    # 3) 筹码集中 (0-20)：终选阶段由股东户数补充，先置 0
    parts["holder"] = 0.0

    # 4) 试盘痕迹 (0-20)：15~90天前上过榜、之后归于平静 = 经典试盘→吸筹节奏
    t = 0.0
    today = as_of or datetime.now()
    if lhb_info:
        trail_days = []
        for d in lhb_info["dates"]:
            try:
                ago = (today - datetime.strptime(d, "%Y-%m-%d")).days
            except ValueError:
                continue
            if 15 <= ago <= 90:
                trail_days.append(ago)
        if trail_days:
            t += 8.0
            if lhb_info["net_buy"] > 0:
                t += 4.0
            reasons.append(f"{min(trail_days)}天前龙虎榜试盘")
    if memory:
        if memory["known"] > 0:
            t += 8.0
            reasons.append(f"知名席位{memory['known']}个买过")
        elif memory["seats"] >= 1:
            t += 4.0
    parts["trail"] = round(min(t, 20.0), 1)

    return round(sum(parts.values()), 1), reasons, parts


def ambush_scan(args):
    """潜伏雷达主流程：资金背离初筛 → K线形态终筛 → 股东户数加权 → Top N。"""
    print(f"[{datetime.now():%H:%M:%S}] 潜伏雷达：拉取多日资金流...")
    flow_now = fetch_ths_flow()
    flow5 = fetch_ths_flow_rank("5日排行")
    flow10 = fetch_ths_flow_rank("10日排行")
    # 20日资金分项改由终选阶段的 A/D 代理提供（与 ambush_backtest 口径一致），不再单独抓同花顺20日排行
    print(f"  即时/5日/10日资金流: {len(flow_now)}/{len(flow5)}/{len(flow10)} 只")
    lhb_hist = fetch_lhb_recent_dates(days=90)
    print(f"  近90日龙虎榜: {len(lhb_hist)} 只")
    memory = load_seat_memory()

    # 初筛：钱在进、价没动、流通盘符合区间
    pre = []
    for code, f5 in flow5.items():
        f10, now = flow10.get(code), flow_now.get(code)
        if not f10 or not now:
            continue
        if _is_small_board_excluded(code, now["name"]):
            continue
        price = now.get("price")
        if price is None or not (2.0 <= price <= 80.0):
            continue
        if f5["net_inflow"] <= 0 or f10["net_inflow"] <= 0:
            continue
        if f5["chg_pct"] is None or not (-5.0 <= f5["chg_pct"] <= 10.0):
            continue
        if f10["chg_pct"] is not None and f10["chg_pct"] > 15.0:
            continue
        float_cap = now.get("float_cap_est")
        if not float_cap:
            continue
        if not (args.min_float_cap * 1e8 <= float_cap <= args.max_float_cap * 1e8):
            continue
        intensity = f5["net_inflow"] / float_cap + f10["net_inflow"] / float_cap
        pre.append((intensity, code, now, f5, f10, float_cap))
    pre.sort(key=lambda x: -x[0])
    finalists = pre[:args.finalists]
    print(f"  资金背离初筛: {len(pre)} 只 → 取前 {len(finalists)} 只拉60日K线（串行，约{len(finalists) // 2}~{len(finalists)}秒）")

    survivors = []
    rejects = Counter()
    for idx, (_, code, now, _f5_ths, _f10_ths, _cap_pre) in enumerate(finalists, 1):
        data = fetch_stock_data(code, now["name"])
        if not data or not data.get("records"):
            rejects["K线获取失败"] += 1
            continue
        records = data["records"]
        # 终选用与 ambush_backtest 一致的 A/D 代理口径，在"当前收盘"时点重算入选门槛与资金分项。
        # 同花顺"主动净额"会漏掉主力挂买盘接砸盘/对倒式的被动吸筹，A/D（收盘日内位置×成交额）能抓到；
        # 同时让实盘打分口径 = 已回测验证过的口径（消除实盘/回测口径裂缝）。
        reason_box = []
        gate = ambush_gate_check(records, len(records) - 1, code, now["name"],
                                 args.min_float_cap, args.max_float_cap, reason_out=reason_box)
        if gate is None:
            rejects[reason_box[0] if reason_box else "未知"] += 1
            continue
        score, reasons, parts = score_ambush(
            gate["flow5"], gate["flow10"], gate["flow20"],
            gate["float_cap"], gate["shape"],
            lhb_hist.get(code), memory.get(code),
        )
        # 入选新鲜度：回看前3个交易日是否同样满足潜伏指纹(同一 A/D 口径)
        streak = 1
        for back in range(1, 4):
            if ambush_gate_check(records, len(records) - 1 - back, code, now["name"],
                                 args.min_float_cap, args.max_float_cap):
                streak += 1
            else:
                break
        parts["freshness"] = freshness_bonus(streak)
        score = round(score + parts["freshness"], 1)
        if streak == 1:
            reasons.append("首次入选(新鲜)")
        elif streak >= 3:
            reasons.append(f"连续{streak}天未启动(降权)")
        survivors.append({
            "code": code,
            "name": now["name"],
            "price": gate["price"],
            "chg5_pct": gate["flow5"]["chg_pct"],
            "float_cap_yi": round(gate["float_cap"] / 1e8, 1),
            "score": score,
            "parts": parts,
            "reasons": reasons,
            "streak": streak,
        })
        if idx % 20 == 0:
            print(f"    K线进度 {idx}/{len(finalists)}")
    if rejects:
        print("  形态过滤剔除:", "、".join(f"{k}×{v}" for k, v in rejects.most_common()))

    # 附加 ① 主力背离(10日分级资金) + ③ 板块共振(同行业潜伏扎堆)。
    # 实盘专属实时维度，ambush_backtest 不含——这两项无历史快照、回放不了，
    # 不进核心 score_ambush 以保持回测口径纯净。
    capital_map = fetch_main_capital_flow(indicator="10日")
    industry_map, ind_src = build_industry_map()
    print(f"  主力分级资金(10日): {len(capital_map)} 只 · 行业映射({ind_src}): {len(industry_map)} 只")
    cand_counter = Counter(
        industry_map.get(p["code"]) for p in survivors if industry_map.get(p["code"]))
    for p in survivors:
        div, dreason = ambush_divergence_pts(capital_map.get(p["code"]))
        p["parts"]["divergence"] = div
        p["score"] = round(p["score"] + div, 1)
        if dreason:
            p["reasons"].append(dreason)
        ind = industry_map.get(p["code"])
        cd = cand_counter.get(ind, 0) if ind else 0
        res = 4.0 if cd >= 4 else (2.0 if cd >= 2 else 0.0)   # 潜伏只看同行业吸筹扎堆（不用涨停潮）
        p["parts"]["resonance"] = res
        p["score"] = round(p["score"] + res, 1)
        if res > 0:
            p["reasons"].append(f"{ind}板块{cd}只潜伏扎堆")

    # 股东户数（筹码集中）只查中期排名靠前的，控制请求量
    survivors.sort(key=lambda p: -p["score"])
    check = survivors[: min(50, len(survivors))]
    if not args.skip_holders and check:
        print(f"  查询 {len(check)} 只股东户数（筹码集中度）...")
        for p in check:
            chg, as_of = fetch_holder_concentration(p["code"])
            if chg is None:
                continue
            if chg <= -5:
                pts = 20.0
            elif chg <= -2:
                pts = 12.0
            elif chg < 0:
                pts = 6.0
            else:
                pts = 0.0
            p["parts"]["holder"] = pts
            p["score"] = round(p["score"] + pts, 1)
            if chg < 0:
                p["reasons"].append(f"股东户数降{abs(chg):.1f}%({as_of})")
            elif chg > 8:
                p["reasons"].append(f"⚠户数增{chg:.1f}%(筹码发散)")
            time.sleep(0.2)
        survivors.sort(key=lambda p: -p["score"])

    picks = survivors[: args.top]
    for p in picks:
        p["reasons"] = p["reasons"][:5]
    return picks, len(pre)


def print_ambush(picks, candidate_count):
    print()
    print("=" * 112)
    print(f"  潜伏雷达（实验·吸筹阶段） · 资金背离候选 {candidate_count} → Top {len(picks)}"
          f" · 潜/背/形/筹/共/忆/鲜 = 资金潜入/主力背离/吸筹形态/筹码集中/板块共振/试盘痕迹/入选新鲜度")
    print("=" * 112)
    header = (
        _pad_visual("代码", 8) + _pad_visual("名称", 12) + _pad_visual("现价", 8)
        + _pad_visual("5日%", 8) + _pad_visual("流通亿", 8) + _pad_visual("总分", 7)
        + _pad_visual("潜", 6) + _pad_visual("背", 6) + _pad_visual("形", 6) + _pad_visual("筹", 6)
        + _pad_visual("共", 6) + _pad_visual("忆", 6) + _pad_visual("鲜", 6)
        + "理由"
    )
    print(header)
    print("-" * 112)
    for p in picks:
        print(
            _pad_visual(p["code"], 8)
            + _pad_visual(p["name"], 12)
            + _pad_visual(f"{p['price']:.2f}" if p["price"] else "-", 8)
            + _pad_visual(f"{p['chg5_pct']:+.1f}" if p["chg5_pct"] is not None else "-", 8)
            + _pad_visual(f"{p['float_cap_yi']:.0f}", 8)
            + _pad_visual(f"{p['score']:.1f}", 7)
            + _pad_visual(f"{p['parts']['inflow']:.0f}", 6)
            + _pad_visual(f"{p['parts'].get('divergence', 0):.0f}", 6)
            + _pad_visual(f"{p['parts']['shape']:.0f}", 6)
            + _pad_visual(f"{p['parts']['holder']:.0f}", 6)
            + _pad_visual(f"{p['parts'].get('resonance', 0):.0f}", 6)
            + _pad_visual(f"{p['parts']['trail']:.0f}", 6)
            + _pad_visual(f"{p['parts'].get('freshness', 0):+.0f}", 6)
            + " | ".join(p["reasons"])
        )
    print("=" * 112)
    print("注: 潜伏≠马上启动，可能横盘数周；已硬剔除近10日涨停/涨幅>12%/60日涨幅>50%的票。")
    print("注: 背/共为实盘实时维度，--ambush-backtest 回测不含（无历史快照）。")
    print("注: 实验性推断；隔几天用 --ambush-verify 回看前向收益验证。")


def persist_ambush(picks, candidate_count):
    now = datetime.now()
    payload = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "candidate_count": candidate_count,
        "picks": picks,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(AMBUSH_LATEST_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    day_file = _ambush_day_file(payload["date"])
    with open(day_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  → 已落盘 {AMBUSH_LATEST_FILE} / {day_file}")


def ambush_verify(date_str=None, horizon=5):
    """回看某日潜伏名单此后 horizon 个交易日的表现（最高价涨幅/收盘涨幅/涨停数）。"""
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    day_file = _ambush_day_file(date_str)
    try:
        with open(day_file, encoding="utf-8") as f:
            day = json.load(f)
    except (OSError, json.JSONDecodeError):
        print(f"[ERROR] 找不到 {day_file}，{date_str} 没有跑过潜伏雷达。")
        return
    picks = day.get("picks", [])
    print("=" * 100)
    print(f"  潜伏雷达前向验证 · 名单日 {date_str} · 持有 {horizon} 个交易日 · 共 {len(picks)} 只")
    print("=" * 100)
    rows = []
    for p in picks:
        data = fetch_stock_data(p["code"], p["name"])
        records = (data or {}).get("records") or []
        forward = [r for r in records if r["date"] > date_str][:horizon]
        base = next((r["close"] for r in reversed(records) if r["date"] <= date_str), None)
        if not forward or not base:
            rows.append((p, None, None, 0, 0))
            continue
        limit_pct = 19.5 if p["code"].startswith(("30", "68")) else 9.5
        max_gain = max(r["high"] for r in forward) / base - 1
        close_gain = forward[-1]["close"] / base - 1
        limit_ups = sum(1 for r in forward if r["change_pct"] >= limit_pct)
        rows.append((p, max_gain * 100, close_gain * 100, limit_ups, len(forward)))
    valid = [r for r in rows if r[1] is not None]
    for p, max_g, close_g, lu, n in rows:
        if max_g is None:
            print(f"  {p['code']} {_pad_visual(p['name'], 12)} 分{p['score']:>5.1f}  尚无足够前向交易日")
            continue
        mark = " ★涨停" * lu
        print(f"  {p['code']} {_pad_visual(p['name'], 12)} 分{p['score']:>5.1f}"
              f"  {n}日内最高{max_g:+6.1f}%  期末{close_g:+6.1f}%{mark}")
    if valid:
        avg_max = mean(r[1] for r in valid)
        avg_close = mean(r[2] for r in valid)
        hit5 = sum(1 for r in valid if r[1] >= 5) / len(valid) * 100
        hit10 = sum(1 for r in valid if r[1] >= 10) / len(valid) * 100
        launched = sum(1 for r in valid if r[3] > 0)
        print("-" * 100)
        print(f"  均值: 期间最高 {avg_max:+.1f}% / 期末 {avg_close:+.1f}%"
              f" · 摸高≥5%: {hit5:.0f}% · 摸高≥10%: {hit10:.0f}% · 启动(出现涨停): {launched}/{len(valid)}")


# ─── 潜伏回测（历史逐日回放）──────────────────────────────────
#
# 同花顺多日资金流排行只有"当天快照"，无法直接回看历史；东财的单股历史资金流
# 与历史K线接口(push2系)在本机均被拒。因此：
#   - 历史K线用新浪 stock_zh_a_daily（全字段 OHLCV+成交额+换手，底层 mini_racer
#     执行JS不可多线程，借用 stock_crawl_capital 的全局锁串行拉取，磁盘缓存当日复用）
#   - "资金潜入"分项改用可完全重建的 Chaikin A/D 吸筹资金代理：Σ CLV×成交额，
#     CLV=((收-低)-(高-收))/(高-低)。口径与实盘同花顺净额不同但方向一致；
#     形态/筹码/试盘三个分项与实盘口径完全相同。
# 股东户数按"公告日期 ≤ 回放日"取数、龙虎榜按上榜日过滤，避免未来函数。

AMBUSH_CACHE_DIR = DATA_DIR / "ambush_cache"
AMBUSH_BACKTEST_FILE = DATA_DIR / "ambush_backtest.json"


def _fetch_kline_history(code, days_back=300):
    """新浪历史日线 → 统一记录格式；失败返回 []。turnover 小数换手→百分比。"""
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
    with _kline_lock:
        df = _retry(ak.stock_zh_a_daily, symbol=_symbol_with_prefix(code),
                    start_date=start, end_date=datetime.now().strftime("%Y%m%d"),
                    adjust="qfq", retries=2)
    if df is None or df.empty:
        return []
    records = []
    prev_close = None
    for _, row in df.iterrows():
        close = _num(row.get("close"))
        chg = (close / prev_close - 1) * 100 if prev_close else 0.0
        records.append({
            "date": str(row.get("date", ""))[:10],
            "open": _num(row.get("open")),
            "high": _num(row.get("high")),
            "low": _num(row.get("low")),
            "close": close,
            "volume": _num(row.get("volume")),
            "amount": _num(row.get("amount")),
            "turnover_rate": _num(row.get("turnover")) * 100,
            "change_pct": round(chg, 4),
        })
        prev_close = close
    return records


def _load_kline_universe(universe):
    """带磁盘缓存的批量历史K线（当日缓存直接复用；新浪源串行约0.5秒/只）。"""
    AMBUSH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    cache = {}
    todo = []
    for code, name in universe:
        fp = AMBUSH_CACHE_DIR / f"{code}.json"
        if fp.exists():
            try:
                with open(fp, encoding="utf-8") as f:
                    saved = json.load(f)
                if saved.get("fetched") == today and saved.get("records"):
                    cache[code] = {"name": saved.get("name") or name, "records": saved["records"]}
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        todo.append((code, name))

    if todo:
        print(f"  K线缓存命中 {len(cache)} 只，需要拉取 {len(todo)} 只"
              f"（新浪源串行，预计约 {len(todo) // 2}~{len(todo)} 秒）...")
    else:
        print(f"  K线缓存全部命中（{len(cache)} 只）")
    for i, (code, name) in enumerate(todo, 1):
        records = _fetch_kline_history(code)
        if records:
            with open(AMBUSH_CACHE_DIR / f"{code}.json", "w", encoding="utf-8") as f:
                json.dump({"fetched": today, "name": name, "records": records}, f, ensure_ascii=False)
            cache[code] = {"name": name, "records": records}
        if i % 100 == 0:
            print(f"    拉取进度 {i}/{len(todo)}（成功 {len(cache)}）")
        time.sleep(0.05)
    # 预建 日期→下标 索引
    for info in cache.values():
        info["index"] = {r["date"]: i for i, r in enumerate(info["records"])}
    return cache


def _ad_flow(records_slice, window):
    """Chaikin A/D 资金代理：窗口内 Σ CLV×成交额（元）。"""
    total = 0.0
    for r in records_slice[-window:]:
        h, l, c = r["high"], r["low"], r["close"]
        if h > l and r["amount"] > 0:
            total += (2 * c - h - l) / (h - l) * r["amount"]
    return total


def _fetch_gdhs_cached(code, max_age_days=7):
    """股东户数历史（磁盘缓存 max_age_days 天）→ [(截止日, 公告日, 增减比例)]"""
    fp = AMBUSH_CACHE_DIR / f"gdhs_{code}.json"
    if fp.exists():
        try:
            with open(fp, encoding="utf-8") as f:
                saved = json.load(f)
            fetched = datetime.strptime(saved.get("fetched", "2000-01-01"), "%Y-%m-%d")
            if (datetime.now() - fetched).days <= max_age_days:
                return saved.get("rows", [])
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    df = _retry(ak.stock_zh_a_gdhs_detail_em, symbol=code, retries=2)
    rows = []
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            rows.append([
                str(row.get("股东户数统计截止日", ""))[:10],
                str(row.get("股东户数公告日期", ""))[:10],
                _num(row.get("股东户数-增减比例")),
            ])
    AMBUSH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump({"fetched": datetime.now().strftime("%Y-%m-%d"), "rows": rows}, f, ensure_ascii=False)
    return rows


def ambush_gate_check(records, idx, code, name, min_cap_yi, max_cap_yi, reason_out=None):
    """潜伏指纹门槛（时点 idx 收盘后口径），实盘回看与回测回放共用。

    通过返回 {flow5, flow10, flow20, float_cap, shape, price}；任一门槛不过返回 None。
    资金口径为 A/D 代理（实盘当日入选门槛走同花顺净额，回看历史只能用代理）。
    reason_out: 可选 list，传入时把"未通过的关卡"写进去（默认 None=行为不变，回测调用不受影响）。
    """
    def _fail(reason):
        if reason_out is not None:
            reason_out.append(reason)
        return None

    if idx is None or idx < 60 or idx >= len(records):
        return _fail("历史不足60日")
    past = records[: idx + 1]
    price = past[-1]["close"]
    if not (2.0 <= price <= 80.0):
        return _fail("价格区间外(2~80)")
    if _is_small_board_excluded(code, name):
        return _fail("排除板块(北交/ST/次新)")
    turnover, amount = past[-1]["turnover_rate"], past[-1]["amount"]
    float_cap = amount / (turnover / 100.0) if turnover and turnover >= 0.5 and amount > 0 else None
    if not float_cap or not (min_cap_yi * 1e8 <= float_cap <= max_cap_yi * 1e8):
        return _fail("市值区间外/换手过低")
    in5, in10, in20 = _ad_flow(past, 5), _ad_flow(past, 10), _ad_flow(past, 20)
    if in5 <= 0 or in10 <= 0:
        return _fail("A/D资金非净流入")
    chg5 = (price / past[-6]["close"] - 1) * 100 if past[-6]["close"] > 0 else None
    chg10 = (price / past[-11]["close"] - 1) * 100 if past[-11]["close"] > 0 else None
    if chg5 is None or not (-5.0 <= chg5 <= 10.0):
        return _fail("5日涨幅超界(吸筹需-5~10%)")
    if chg10 is not None and chg10 > 15.0:
        return _fail("10日涨幅>15%")
    shape, _why = analyze_kline_shape(past[-60:], code)
    if shape is None:
        return _fail(f"形态:{_why}")
    return {
        "flow5": {"net_inflow": in5, "chg_pct": chg5},
        "flow10": {"net_inflow": in10, "chg_pct": chg10},
        "flow20": {"net_inflow": in20, "chg_pct": None},
        "float_cap": float_cap,
        "shape": shape,
        "price": price,
    }


def freshness_bonus(streak):
    """入选新鲜度：首次入选加分，挂榜多日反而减分。

    60交易日/294买点回测显示，吸筹指纹刚出现时点火概率最高（首次入选启动率18%），
    而连续≥3天还没动的票多半是没人来拉的横盘股（启动率仅8%）——故新鲜度反向计分。
    """
    if streak == 1:
        return 10.0
    if streak == 2:
        return 2.0
    return -8.0  # 连续≥3天未启动，降权


def _holder_pts_asof(rows, date_str):
    """按公告日期 ≤ date_str 取最近一期户数增减比例，换算筹码分。"""
    latest = None
    for as_of, announce, chg in rows:
        if announce and announce <= date_str:
            if latest is None or as_of > latest[0]:
                latest = (as_of, chg)
    if latest is None or latest[1] is None:
        return 0.0
    chg = latest[1]
    if chg <= -5:
        return 20.0
    if chg <= -2:
        return 12.0
    if chg < 0:
        return 6.0
    return 0.0


def ambush_backtest(args):
    """逐日回放过去 N 个交易日的潜伏雷达，并统计前向收益与全池基准对比。"""
    print(f"[{datetime.now():%H:%M:%S}] 潜伏回测：构建股票池...")
    flow_now = fetch_ths_flow()
    universe = []
    for code, now in flow_now.items():
        if _is_small_board_excluded(code, now["name"]):
            continue
        fc = now.get("float_cap_est")
        # 留缓冲：历史时点的流通市值会逐日重算，这里只做粗圈定
        if not fc or not (args.min_float_cap * 0.7e8 <= fc <= args.max_float_cap * 1.3e8):
            continue
        if (now.get("amount") or 0) < 3e7:
            continue
        universe.append((now["amount"], code, now["name"]))
    universe.sort(key=lambda x: -x[0])
    universe = [(c, n) for _, c, n in universe[: args.universe_cap]]
    print(f"  股票池: {len(universe)} 只（按成交额取前 {args.universe_cap}）")

    cache = _load_kline_universe(universe)
    print(f"  历史K线就绪: {len(cache)} 只")
    lhb_hist = fetch_lhb_history(days=90 + args.days * 2 + 10)
    print(f"  龙虎榜历史: {len(lhb_hist)} 只")

    # 交易日历：出现在 ≥30% 股票K线中的日期（多取3天用于"入选新鲜度"预热）
    date_counts = Counter()
    for info in cache.values():
        for r in info["records"][-(args.days + args.horizon + 15):]:
            date_counts[r["date"]] += 1
    calendar = sorted(d for d, c in date_counts.items() if c >= len(cache) * 0.3)
    replay_dates = calendar[-args.days:]
    if not replay_dates:
        print("[ERROR] 无可用交易日历。")
        return
    prewarm_dates = calendar[-(args.days + 3):-args.days]
    print(f"  回放区间: {replay_dates[0]} ~ {replay_dates[-1]}（{len(replay_dates)} 个交易日，"
          f"预热 {len(prewarm_dates)} 天算入选新鲜度）\n")

    # 预热：回放起点前几个交易日的候选集合，让首日就能识别新鲜度
    candidate_history = []
    for date_str in prewarm_dates:
        day_set = set()
        for code, info in cache.items():
            if ambush_gate_check(info["records"], info["index"].get(date_str), code,
                                 info["name"], args.min_float_cap, args.max_float_cap):
                day_set.add(code)
        candidate_history.append(day_set)

    all_picks = []
    day_lines = []
    for date_str in replay_dates:
        as_of = datetime.strptime(date_str, "%Y-%m-%d")
        day_candidates = []
        baseline_gains = []
        today_set = set()
        for code, info in cache.items():
            idx = info["index"].get(date_str)
            if idx is None or idx < 60:
                continue
            records = info["records"]
            last = records[idx]
            price = last["close"]
            if not (2.0 <= price <= 80.0) or _is_small_board_excluded(code, info["name"]):
                continue
            turnover, amount = last["turnover_rate"], last["amount"]
            float_cap = amount / (turnover / 100.0) if turnover and turnover >= 0.5 and amount > 0 else None
            if not float_cap or not (args.min_float_cap * 1e8 <= float_cap <= args.max_float_cap * 1e8):
                continue
            forward = records[idx + 1: idx + 1 + args.horizon]
            if forward:
                baseline_gains.append(max(r["high"] for r in forward) / price - 1)
            gate = ambush_gate_check(records, idx, code, info["name"],
                                     args.min_float_cap, args.max_float_cap)
            if not gate:
                continue
            today_set.add(code)
            streak = 1
            for prev_set in reversed(candidate_history):
                if code in prev_set:
                    streak += 1
                else:
                    break
            events = [(d, n) for d, n in (lhb_hist.get(code) or []) if d <= date_str]
            lhb_info = {"dates": [d for d, _ in events],
                        "net_buy": sum(n for _, n in events)} if events else None
            score, reasons, parts = score_ambush(
                gate["flow5"], gate["flow10"], gate["flow20"],
                gate["float_cap"], gate["shape"], lhb_info, None, as_of=as_of,
            )
            parts["freshness"] = freshness_bonus(streak)
            score = round(score + parts["freshness"], 1)
            if streak == 1:
                reasons.append("首次入选(新鲜)")
            elif streak >= 3:
                reasons.append(f"连续{streak}天未启动(降权)")
            day_candidates.append({
                "date": date_str, "code": code, "name": info["name"],
                "score": score, "parts": parts, "price": price,
                "idx": idx, "reasons": reasons, "streak": streak,
            })
        candidate_history.append(today_set)
        candidate_history = candidate_history[-3:]
        day_candidates.sort(key=lambda p: -p["score"])
        top = day_candidates[: args.top]

        if not args.skip_holders:
            for p in top:
                pts = _holder_pts_asof(_fetch_gdhs_cached(p["code"]), date_str)
                p["parts"]["holder"] = pts
                p["score"] = round(p["score"] + pts, 1)
            top.sort(key=lambda p: -p["score"])

        launches = 0
        for p in top:
            records = cache[p["code"]]["records"]
            forward = records[p["idx"] + 1: p["idx"] + 1 + args.horizon]
            if forward:
                limit_pct = 19.5 if p["code"].startswith(("30", "68")) else 9.5
                p["max_gain_pct"] = round((max(r["high"] for r in forward) / p["price"] - 1) * 100, 2)
                p["close_gain_pct"] = round((forward[-1]["close"] / p["price"] - 1) * 100, 2)
                p["limit_ups"] = sum(1 for r in forward if r["change_pct"] >= limit_pct)
                p["fwd_days"] = len(forward)
                launches += 1 if p["limit_ups"] > 0 else 0
            else:
                p["max_gain_pct"] = p["close_gain_pct"] = None
                p["limit_ups"] = 0
                p["fwd_days"] = 0
            del p["idx"]
        all_picks.extend(top)
        valid = [p for p in top if p["max_gain_pct"] is not None]
        bench = mean(baseline_gains) * 100 if baseline_gains else None
        avg_max = mean(p["max_gain_pct"] for p in valid) if valid else None
        day_lines.append((date_str, len(day_candidates), len(top), avg_max, bench, launches))
        line = f"  {date_str} 候选{len(day_candidates):>3} 入选{len(top):>3}"
        if avg_max is not None and bench is not None:
            line += f"  期间最高均值 {avg_max:+.2f}%  全池基准 {bench:+.2f}%  启动{launches}"
        else:
            line += "  （无前向数据，最近交易日）"
        print(line)

    valid = [p for p in all_picks if p["max_gain_pct"] is not None]
    print()
    print("=" * 100)
    print(f"  潜伏回测汇总 · {len(replay_dates)} 个交易日 · 共 {len(all_picks)} 个买点（有前向数据 {len(valid)}）")
    print("=" * 100)
    if valid:
        avg_max = mean(p["max_gain_pct"] for p in valid)
        avg_close = mean(p["close_gain_pct"] for p in valid)
        bench_all = [b for _, _, _, _, b, _ in day_lines if b is not None]
        bench_avg = mean(bench_all) if bench_all else None
        hit5 = sum(1 for p in valid if p["max_gain_pct"] >= 5) / len(valid) * 100
        hit10 = sum(1 for p in valid if p["max_gain_pct"] >= 10) / len(valid) * 100
        launched = sum(1 for p in valid if p["limit_ups"] > 0)
        print(f"  期间最高均值 {avg_max:+.2f}%"
              + (f"（全池基准 {bench_avg:+.2f}%，超额 {avg_max - bench_avg:+.2f}%）" if bench_avg is not None else ""))
        print(f"  期末均值 {avg_close:+.2f}% · 摸高≥5%: {hit5:.0f}% · 摸高≥10%: {hit10:.0f}%"
              f" · 启动(出现涨停): {launched}/{len(valid)}")
        best = sorted(valid, key=lambda p: -p["max_gain_pct"])[:8]
        print("\n  最佳买点:")
        for p in best:
            print(f"    {p['date']} {p['code']} {_pad_visual(p['name'], 12)} 分{p['score']:>5.1f}"
                  f"  {p['fwd_days']}日内最高 {p['max_gain_pct']:+6.1f}%"
                  + ("  ★涨停" * p["limit_ups"]))
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": {"days": args.days, "horizon": args.horizon, "top": args.top,
                   "universe": len(cache), "min_float_cap": args.min_float_cap,
                   "max_float_cap": args.max_float_cap,
                   "note": "回测资金分项用A/D代理，与实盘同花顺净额口径不同"},
        "picks": all_picks,
    }
    with open(AMBUSH_BACKTEST_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n  → 已落盘 {AMBUSH_BACKTEST_FILE}")


# ─── 分项有效性分析（IC，离线读回测产物，不联网）──────────────
#
# 读 ambush_backtest.json，对每个评分分项算与前向收益的 Spearman 秩相关(IC)，
# 量化"哪个信号真的领先于主力拉升"，用于砍无效因子 / 数据驱动重配权重。
# 纯离线、纯计算；需先跑过一次 --ambush-backtest 产出数据。

def _rankdata(values):
    """平均秩(1-based)，并列取平均。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx <= 0 or sy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (sx ** 0.5 * sy ** 0.5)


def _spearman_ic(xs, ys):
    """Spearman 秩相关 = 秩上的 Pearson；任一序列为常数返回 None。"""
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _ic_tstat(ic, n):
    """IC 的近似 t 值：ic·sqrt((n-2)/(1-ic²))。"""
    if ic is None or n <= 2 or abs(ic) >= 1.0:
        return None
    return ic * ((n - 2) / (1.0 - ic * ic)) ** 0.5


def _quantile_means(xs, ys, q=5):
    """按 x 升序分 q 档，返回每档 y 均值（样本不足或 x 为常数返回 None）。"""
    if len(xs) < q * 2 or len(set(xs)) < 2:
        return None
    pairs = sorted(zip(xs, ys))
    n = len(pairs)
    return [mean(y for _, y in pairs[b * n // q:(b + 1) * n // q]) for b in range(q)]


def analyze_ambush_ic(target="max_gain_pct"):
    """离线读 ambush_backtest.json，算各评分分项与前向收益的 Spearman IC。"""
    try:
        with open(AMBUSH_BACKTEST_FILE, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        print(f"[ERROR] 找不到/无法读取 {AMBUSH_BACKTEST_FILE}，请先跑一次 --ambush-backtest。")
        return

    picks = payload.get("picks", [])
    rows = [p for p in picks if p.get(target) is not None]
    if not rows:
        print(f"[ERROR] {AMBUSH_BACKTEST_FILE.name} 里没有带前向收益的买点。")
        return
    if len(rows) < 20:
        print(f"[WARN] 有效买点仅 {len(rows)} 个，样本太少、IC 噪声极大，仅供参考。\n")

    horizon = (payload.get("params") or {}).get("horizon")
    factor_keys = sorted({k for p in rows for k in (p.get("parts") or {}).keys()})

    print("=" * 96)
    print("  潜伏雷达 · 分项有效性 IC 分析（Spearman 秩相关，越高=越领先于前向涨幅）")
    print(f"  产物 {AMBUSH_BACKTEST_FILE.name} · 买点 {len(picks)}（有前向 {len(rows)}）"
          f" · horizon={horizon}日 · 目标=期间最高涨幅")
    print("=" * 96)
    print(_pad_visual("分项", 12) + _pad_visual("样本", 6)
          + _pad_visual("IC", 9) + _pad_visual("t值", 8)
          + "  低→高分5档 期间最高均值%（单调上升=有效）")
    print("-" * 96)

    def _ic_row(name, getter):
        xs, ys = [], []
        for p in rows:
            x, y = getter(p), p.get(target)
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
        ic = _spearman_ic(xs, ys)
        t = _ic_tstat(ic, len(xs))
        buckets = _quantile_means(xs, ys)
        ic_s = f"{ic:+.3f}" if ic is not None else "N/A"
        t_s = f"{t:+.2f}" if t is not None else "-"
        flag = ""
        if ic is not None and t is not None and abs(ic) >= 0.05 and abs(t) >= 2.0:
            flag = " ✅有效" if ic > 0 else " ⚠反向"
        if buckets:
            bucket_s = " → ".join(f"{b:+.1f}" for b in buckets)
        else:
            bucket_s = "(常数/无效)" if ic is None else "(样本不足)"
        print(_pad_visual(name, 12) + _pad_visual(str(len(xs)), 6)
              + _pad_visual(ic_s, 9) + _pad_visual(t_s, 8) + "  " + bucket_s + flag)

    for key in factor_keys:
        _ic_row(key, lambda p, k=key: (p.get("parts") or {}).get(k))
    _ic_row("score总分", lambda p: p.get("score"))

    print("-" * 96)
    print("  读法: IC>0=因子值越高前向涨幅越高；|IC|≥0.05 且 |t|≥2 标 ✅有效 / ⚠反向。")
    print("  分5档均值单调上升=方向可靠；据此砍无效/反向分项、给有效分项加权。")
    print("  注: 单次回测、买点样本有限，IC 噪声大；多跑几段 --ambush-backtest 再下结论。")


# ─── 入口 ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="实验性：盘中游资买入雷达")
    parser.add_argument("--top", type=int, default=20, help="输出条数，默认20")
    parser.add_argument("--max-float-cap", type=float, default=150.0,
                        help="流通市值上限(亿)，默认150（游资偏好小票）")
    parser.add_argument("--min-float-cap", type=float, default=50.0,
                        help="流通市值下限(亿)，默认50，0=不限")
    parser.add_argument("--min-chg", type=float, default=-2.0,
                        help="最低当日涨幅(%%)过滤，默认-2")
    parser.add_argument("--watch", type=int, default=0,
                        help="轮询秒数；0=只扫一次")
    parser.add_argument("--verify", action="store_true", help="盘后用龙虎榜验证当日拉升雷达")
    parser.add_argument("--date", default=None, help="验证日期 YYYY-MM-DD，默认今天")
    parser.add_argument("--ambush", action="store_true",
                        help="潜伏雷达：找游资正在吸筹、尚未启动的票（盘后跑）")
    parser.add_argument("--ambush-verify", action="store_true",
                        help="回看某日潜伏名单的前向收益（配 --date/--horizon）")
    parser.add_argument("--ambush-backtest", action="store_true",
                        help="逐日回放过去 N 个交易日的潜伏雷达并统计前向收益（配 --days/--horizon）")
    parser.add_argument("--ambush-ic", action="store_true",
                        help="离线读 ambush_backtest.json，算各评分分项与前向收益的 Spearman IC（不联网）")
    parser.add_argument("--days", type=int, default=15,
                        help="--ambush-backtest 回放的交易日数，默认15")
    parser.add_argument("--horizon", type=int, default=5,
                        help="验证/回测的前向交易日数，默认5")
    parser.add_argument("--universe-cap", type=int, default=900,
                        help="--ambush-backtest 股票池上限（按成交额取前N），默认900")
    parser.add_argument("--finalists", type=int, default=80,
                        help="潜伏模式进入K线分析的初筛数量，默认80")
    parser.add_argument("--skip-holders", action="store_true",
                        help="潜伏/回测模式跳过股东户数查询（更快）")
    args = parser.parse_args()

    if args.verify:
        verify(args.date)
        return
    if args.ambush_verify:
        ambush_verify(args.date, horizon=args.horizon)
        return
    if args.ambush_backtest:
        ambush_backtest(args)
        return
    if args.ambush_ic:
        analyze_ambush_ic()
        return
    if args.ambush:
        picks, candidate_count = ambush_scan(args)
        print_ambush(picks, candidate_count)
        persist_ambush(picks, candidate_count)
        return

    while True:
        picks, candidate_count = scan(
            top_n=args.top,
            max_float_cap_yi=args.max_float_cap,
            min_float_cap_yi=args.min_float_cap,
            min_chg_pct=args.min_chg,
        )
        print_picks(picks, candidate_count)
        persist_picks(picks, candidate_count)
        if not args.watch:
            break
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n已停止。")
            break


if __name__ == "__main__":
    main()
