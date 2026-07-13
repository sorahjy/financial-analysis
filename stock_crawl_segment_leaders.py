"""
细分行业龙头爬虫。

爬取申万三级行业(细分领域)成分，在每个细分行业内部挑出龙头，落盘成龙头股票池，
供后续模型/雷达消费。

三个命令：
  show     展示已爬取的细分龙头池(默认)。
  crawl    正常爬取：读归属缓存(滚动补抓最旧赛道)，重建龙头池。
  recrawl  全量重爬：重抓全部申万三级归属后再重建。

选股逻辑：
  1. 申万三级行业作为细分领域。
  2. 仅过滤 ST/北交/新股前缀与极低价；不设市值/股价绝对上限。
  3. 在每个细分行业内部按规模、盈利、成长打龙头分。
  4. 每个细分行业默认保留 3 只龙头候选。
"""

import argparse
import json
import random
import re
import sqlite3
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

import stock_storage
from stock_crawl_common import strip_proxy_env


strip_proxy_env()


DATA_DIR = Path("data/capital")
SEGMENT_LEADER_POOL_FILE = DATA_DIR / "segment_leader_pool.json"
SEGMENT_LEADER_SCHEMA = "segment_leader_pool.v1"

# 申万三级"行业→成分股"归属缓存：唯一需要 legulegu 的部分，变动极少，缓存复用；持久化到
# 主库 stock_data.sqlite3 的 sw3_segment/sw3_member 两表(见 stock_storage)，日常 crawl 直接读它。
SW3_MEMBERSHIP_SCHEMA = "sw3_membership.v1"
MEMBERSHIP_MAX_AGE_DAYS = 30
DEFAULT_REFRESH_SLICE = 15  # crawl 默认每次滚动补抓最旧的赛道数

# 三级行业"总表"灾备文件：legulegu 彻底失败(如 504 宕机)时的兜底来源。
# 只存段层(代码/名称/上级/成份数)，成分股恢复时由官方接口补；每次切片刷新后自动从主库回写保鲜。
META_BACKUP_FILE = Path("meta_data_backup/sw3_industry_segment.json")

# legulegu 反爬较敏感、冷缓存时偶发返回残缺/空页面。用更接近真实浏览器的请求头降低被
# 限流概率(不设 Accept-Encoding，避免 requests 未装 brotli 时拿到无法解码的响应)。
SW3_OVERVIEW_URL = "https://legulegu.com/stockdata/sw-industry-overview"
SWS_COMPONENT_URL = "https://www.swsresearch.com/institute-sw/api/index_publish/details/component_stocks/"
EASTMONEY_A_SPOT_URL = "https://82.push2.eastmoney.com/api/qt/clist/get"
LEGULEGU_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": SW3_OVERVIEW_URL,
    "Upgrade-Insecure-Requests": "1",
}
LEGULEGU_RETRY_SLEEP_SEC = 0.5
SWS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.swsresearch.com/institute_sw/allIndex/releasedIndex",
}
EASTMONEY_HEADERS = {
    "User-Agent": SWS_HEADERS["User-Agent"],
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://quote.eastmoney.com/center/gridlist.html",
}

# 本地股票主库(K线/财报)：crawl 时从这里补价格/市值/基本面
STOCK_DB_FILE = DATA_DIR.parent / "stock_data.sqlite3"

DEFAULT_TOP_PER_SEGMENT = 3
FORCED_SEGMENT_LEADER_CODES = [ # 筛完各 SW3 topN 后仍强制纳入其所属赛道
    # 消费
    "002507",  # 涪陵榨菜
    "002956",  # 西麦食品
    "601888",  # 中国中免
    "000333",  # 美的集团 *
    "600690",  # 海尔智家 *
    "000858",  # 五粮液
    "600887",  # 伊利股份
    "603288",  # 海天味业
    "605499",  # 东鹏饮料
    "000895",  # 双汇发展
    # 金融
    "300033",  # 同花顺
    "300059",  # 东方财富
    "600030",  # 中信证券
    "600570",  # 恒生电子
    "600036",  # 招商银行
    "002142",  # 宁波银行
    "601318",  # 中国平安
    "601336",  # 新华保险
    # 传统化工
    "600309",  # 万华化学 *  MDI保温材料
    "002648",  # 卫星化学 轻烃
    "600989",  # 宝丰能源 煤-烯烃
    "600160",  # 巨化股份 制冷剂
    "600486",  # 扬农化工
    # 能源电力
    "600406",  # 国电南瑞
    "600089",  # 特变电工 *
    "000400",  # 许继电气
    "601179",  # 中国西电 *
    "600312",  # 平高电气
    "600522",  # 中天科技 *
    "300274",  # 阳光电源 * 光伏逆变器、储能
    "603606",  # 东方电缆 海缆、海上风电
    "002202",  # 金风科技 风电整机
    "600900",  # 长江电力 水电
    "601985",  # 中国核电 核电运营
    "003816",  # 中国广核 核电运营
    # 工程
    "000338",  # 潍柴动力 *
    "600031",  # 三一重工 *
    # 医药医疗医美
    "600276",  # 恒瑞医药
    "688235",  # 百济神州 *
    "603259",  # 药明康德
    "688331",  # 荣昌生物
    "603087",  # 甘李药业
    "002001",  # 新和成
    "000963",  # 华东医药
    "000538",  # 云南白药
    "300760",  # 迈瑞医疗 *
    "300896",  # 爱美客
    # 机器人
    "300124",  # 汇川技术 工业机器人、汽车工控自动化
    "002747",  # 埃斯顿  工业机器人本体、智能系统
    "002472",  # 双环传动 * 减速器起步、智能执行系统、汽车齿轮
    "002896",  # 中大力德 减速器核心龙头、执行器龙头
    "601689",  # 拓普集团 执行器起步、汽车配饰和底盘
    "002050",  # 三花智控 * 执行器起步、汽车冷却零部件
    "603728",  # 鸣志电器 电机运动控制
    "002979",  # 雷赛智能 运动控制
    "603662",  # 柯力传感 机器人力传感起步，工业物联网力传感器
    "688017",  # 绿的谐波 谐波减速器、计算机视觉
    # cpo
    "300308",  # 中际旭创 *
    "300394",  # 天孚通信
    # 晶圆厂
    "002371",  # 北方华创
    "688981",  # 中芯国际
    # 先进封装
    "600584",  # 长电科技 *
    "002156",  # 通富微电
    "002185",  # 华天科技
    # PCB
    "300476",  # 胜宏科技 *
    "002463",  # 沪电股份 *
    "002916",  # 深南电路 二线
    # 半导体材料
    "688146",  # 中船特气
    # 消费电子
    "601138",  # 工业富联 *
    "002475",  # 立讯精密 *
    "002241",  # 歌尔股份
    "603501",  # 豪威集团
    "000725",  # 京东方A *
    "000100",  # TCL科技
    "002938",  # 鹏鼎控股
    "002415",  # 海康威视
    # 存储芯片
    "603986",  # 兆易创新
    "300223",  # 北京君正
    # MLCC
    "000636",  # 风华高科
    "300408",  # 三环集团
    "300285",  # 国瓷材料
    # 其他
    "603298",  # 杭叉集团
    "002230",  # 科大讯飞
]

DEFAULT_POOL_MAX_AGE_DAYS = 14
LOCAL_TAXONOMY_MIN_BACKUP_RATIO = 0.80
LOCAL_TAXONOMY_TRUNCATION_GUARD_MIN_COUNT = 20


def _taxonomy_response_looks_truncated(fetched_count: int, reference_count: int) -> bool:
    """Flag large taxonomy responses that would unexpectedly delete >=20%."""
    return (
        reference_count >= LOCAL_TAXONOMY_TRUNCATION_GUARD_MIN_COUNT
        and fetched_count <= reference_count * LOCAL_TAXONOMY_MIN_BACKUP_RATIO
    )


def _norm_code(value: Any) -> str:
    code = re.sub(r"\D", "", str(value))[:6]
    return code.zfill(6) if code else ""


def _is_stock_excluded(code: str, name: str) -> bool:
    """排除北交所、ST、新股前缀。"""
    if code.startswith(("4", "8", "9")):
        return True
    upper = str(name).upper()
    if "ST" in upper:
        return True
    if upper.startswith(("N", "C")) and not upper.startswith("CW"):
        return True
    return False


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("%", "")
    if not text or text in ("-", "--", "—", "nan", "None", "加载中..."):
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _find_col(columns: Iterable[Any], *keywords: str) -> Optional[Any]:
    for keyword in keywords:
        for col in columns:
            if keyword in str(col):
                return col
    return None


def _percentile_scores(rows: List[Dict[str, Any]], key: str, missing: float = 40.0) -> Dict[str, float]:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return {row["code"]: missing for row in rows}
    if len(values) == 1:
        only = values[0]
        return {row["code"]: (100.0 if row.get(key) == only else missing) for row in rows}

    scores = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            scores[row["code"]] = missing
            continue
        lower = sum(1 for item in values if item < value)
        scores[row["code"]] = lower / (len(values) - 1) * 100.0
    return scores


def _retry_fetch(func, *args, retries: int = 3, sleep_sec: float = 1.0,
                 backoff: bool = True, jitter: float = 0.0, desc: str = "", **kwargs):
    last_error = None
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                delay = sleep_sec * (attempt + 1) if backoff else sleep_sec
                if jitter:
                    delay += random.uniform(0, jitter)
                # 传了 desc 才报进度(避免 per-segment 重试刷屏)，让长退避不再静默
                if desc:
                    print(f"  [retry] {desc} 第 {attempt + 1}/{retries} 次失败：{exc}；"
                          f"{delay:g}s 后重试...", flush=True)
                time.sleep(delay)
    raise last_error


def _parse_sw3_overview(html: str) -> pd.DataFrame:
    """从 legulegu 概览页 HTML 解析申万三级行业总表(只取下游需要的四列)。

    legulegu 限流时会返回缺 level3Items 容器或内容为空的残缺页。这里对这类情况
    显式抛 RuntimeError，交给上层 _retry_fetch 退避重试；而 ak.sw_index_third_info
    遇到残缺页会直接 None.find_all() 崩成 AttributeError、9 次重试也救不回来。
    """
    soup = BeautifulSoup(html, features="lxml")
    container = soup.find(name="div", attrs={"id": "level3Items"})
    if container is None:
        raise RuntimeError("legulegu 申万三级页缺少 level3Items 容器(疑似限流/残缺响应)")

    title_raw = container.find_all(name="div", attrs={"class": "lg-industries-item-chinese-title"})
    number_raw = container.find_all(name="div", attrs={"class": "lg-industries-item-number"})
    if not title_raw or not number_raw or len(title_raw) != len(number_raw):
        raise RuntimeError(
            f"legulegu 申万三级解析异常(title={len(title_raw)}/number={len(number_raw)}，疑似限流)"
        )

    codes, names, parents, counts = [], [], [], []
    for title, number in zip(title_raw, number_raw):
        text = number.get_text()
        span = number.find("span")
        codes.append(title.get_text(strip=True))
        names.append(text.split("(")[0].strip())
        parents.append(span.get_text().split("(")[0][1:-1] if span else "")
        counts.append(text.split("(")[1].split(")")[0] if "(" in text else None)

    return pd.DataFrame({
        "行业代码": codes,
        "行业名称": names,
        "上级行业": parents,
        "成份个数": pd.to_numeric(counts, errors="coerce"),
    })


def fetch_sw3_segments(timeout: int = 20) -> pd.DataFrame:
    """申万三级行业列表(直连 legulegu 概览页解析)。

    用浏览器化请求头抓取，解析失败/残缺时抛错让上层退避重试。
    """
    response = requests.get(SW3_OVERVIEW_URL, headers=LEGULEGU_HEADERS, timeout=timeout)
    response.raise_for_status()
    return _parse_sw3_overview(response.text)


def fetch_sw3_segment_constituents(segment_code: str, timeout: int = 15) -> pd.DataFrame:
    """申万三级行业成分。

    AkShare 的 sw_index_third_cons 在当前上游表结构下存在列数写死问题；
    这里直接解析 Legulegu 原始 HTML 表。
    """
    url = f"https://legulegu.com/stockdata/index-composition?industryCode={segment_code}"
    response = requests.get(url, headers=LEGULEGU_HEADERS, timeout=timeout)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    return tables[0] if tables else pd.DataFrame()


def fetch_sw3_segment_constituents_official(segment_code: str, timeout: int = 15) -> pd.DataFrame:
    """申万宏源研究官方成分接口。

    官方接口与 legulegu 使用同一套申万指数代码(如 850111 / 850111.SI)，但只返回
    成分股代码、名称、市值占比、计入日期。这里保留原始 membership 语义，后续再用本地主库补
    legulegu 表里的价格/市值/基本面字段。
    """
    code = re.sub(r"\D", "", str(segment_code))[:6]
    if not code:
        return pd.DataFrame()
    response = requests.get(
        SWS_COMPONENT_URL,
        params={"swindexcode": code, "page": "1", "page_size": "10000"},
        headers=SWS_HEADERS,
        timeout=timeout,
        verify=False,
    )
    response.raise_for_status()
    payload = response.json()
    results = ((payload.get("data") or {}).get("results") or [])
    if not isinstance(results, list):
        raise RuntimeError("申万宏源官方成分接口返回结构异常")
    df = pd.DataFrame(results)
    if df.empty:
        return df
    df.rename(
        columns={
            "stockcode": "证券代码",
            "stockname": "证券名称",
            "newweight": "市值占比",
            "beginningdate": "计入日期",
        },
        inplace=True,
    )
    keep = [
        col for col in ("证券代码", "证券名称", "市值占比", "最新权重", "权重", "计入日期")
        if col in df.columns
    ]
    if "证券代码" not in keep or "证券名称" not in keep:
        return pd.DataFrame()
    return df[keep]


def _latest_indicator_record(indicators: Any) -> Dict[str, Any]:
    if isinstance(indicators, str):
        try:
            indicators = json.loads(indicators)
        except json.JSONDecodeError:
            return {}
    if not isinstance(indicators, dict):
        return {}
    records = [r for r in indicators.get("records", []) if isinstance(r, dict)]
    if not records:
        return {}
    dated = [r for r in records if r.get("date")]
    if dated:
        return max(dated, key=lambda r: str(r.get("date")))
    return records[0]


def _blank_member_metrics() -> Dict[str, Optional[float]]:
    return {
        "price": None,
        "market_cap_yi": None,
        "roe_pct": None,
        "profit_growth_pct": None,
        "revenue_growth_pct": None,
    }


def fetch_a_spot_member_metrics(timeout: int = 15) -> Dict[str, Dict[str, Optional[float]]]:
    """东方财富全 A 实时行情：冷启动时补官方 membership 的最新价/总市值。

    申万宏源官方成分接口只给代码/名称；如果本地主库还没有 stock_history，就用这里的
    最新价和总市值先让龙头池能选出候选，随后 stock_crawl_price_valuation 再爬完整历史。
    """
    response = requests.get(
        EASTMONEY_A_SPOT_URL,
        params={
            "pn": "1",
            "pz": "10000",
            "po": "1",
            "np": "1",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": "2",
            "invt": "2",
            "fid": "f12",
            "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
            "fields": "f2,f12,f14,f20",
        },
        headers=EASTMONEY_HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()
    rows = ((response.json().get("data") or {}).get("diff") or [])
    if not isinstance(rows, list):
        raise RuntimeError("东方财富全A实时行情返回结构异常")

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = _norm_code(row.get("f12"))
        name = str(row.get("f14") or "").strip()
        if not code or not name or _is_stock_excluded(code, name):
            continue
        price = _to_float_or_none(row.get("f2"))
        market_cap_yuan = _to_float_or_none(row.get("f20"))
        out[code] = {
            "price": price,
            "market_cap_yi": round(market_cap_yuan / 1e8, 4) if market_cap_yuan else None,
            "roe_pct": None,
            "profit_growth_pct": None,
            "revenue_growth_pct": None,
        }
    return out


def _fetch_a_spot_member_metrics_or_empty() -> Dict[str, Dict[str, Optional[float]]]:
    try:
        return fetch_a_spot_member_metrics()
    except Exception as exc:
        print(f"  [membership] 东财全A市值补全失败：{exc}", flush=True)
        return {}


def load_latest_member_metrics(
        db_file: Path = STOCK_DB_FILE,
        *,
        allow_spot_fallback: bool = True,
) -> Dict[str, Dict[str, Optional[float]]]:
    """本地主库补齐 legulegu 成分表字段：价格、总市值、ROE、利润/营收增速。

    官方成分接口只负责 membership；这些指标从本地库补，尽量贴近 legulegu 表结构。
    主库缺失/无数据则返回 {}，补不到的字段留 None，build 时按既有中性/过滤规则处理。
    allow_spot_fallback=True 时才尝试东财实时行情补价/市值；常规龙头池 build 不需要这一步。
    """
    if not Path(db_file).exists():
        return _fetch_a_spot_member_metrics_or_empty() if allow_spot_fallback else {}
    try:
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    except sqlite3.Error:
        return _fetch_a_spot_member_metrics_or_empty() if allow_spot_fallback else {}
    out: Dict[str, Dict[str, Optional[float]]] = {}
    try:
        row = conn.execute("SELECT MAX(date) FROM stock_history").fetchone()
        max_date = row[0] if row else None
        if max_date:
            cutoff = (datetime.strptime(max_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
            for code, _date, close, cap in conn.execute(
                    "SELECT code, date, daily_close, market_cap FROM stock_history "
                    "WHERE date >= ? AND (daily_close IS NOT NULL OR market_cap IS NOT NULL) "
                    "ORDER BY code, date", (cutoff,)):
                norm_code = _norm_code(code)
                item = out.setdefault(norm_code, _blank_member_metrics())
                if close is not None:
                    item["price"] = close
                if cap is not None:
                    item["market_cap_yi"] = cap
        for code, indicators_json in conn.execute(
                "SELECT code, indicators_json FROM stock_meta WHERE indicators_json IS NOT NULL"):
            norm_code = _norm_code(code)
            latest = _latest_indicator_record(indicators_json)
            if not latest:
                continue
            item = out.setdefault(norm_code, _blank_member_metrics())
            item["roe_pct"] = _to_float_or_none(
                latest.get("roe_weighted") if latest.get("roe_weighted") is not None else latest.get("roe")
            )
            item["profit_growth_pct"] = _to_float_or_none(latest.get("net_profit_growth"))
            item["revenue_growth_pct"] = _to_float_or_none(latest.get("revenue_growth"))
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    if allow_spot_fallback and not any(metrics.get("market_cap_yi") is not None for metrics in out.values()):
        for code, metrics in _fetch_a_spot_member_metrics_or_empty().items():
            item = out.setdefault(code, _blank_member_metrics())
            for key in ("price", "market_cap_yi"):
                if item.get(key) is None and metrics.get(key) is not None:
                    item[key] = metrics[key]
    return out


def load_latest_market_cap_yi(db_file: Path = STOCK_DB_FILE) -> Dict[str, float]:
    """兼容旧调用：返回本地主库每只股票最近一条非空总市值(亿元)。"""
    return {
        code: metrics["market_cap_yi"]
        for code, metrics in load_latest_member_metrics(db_file).items()
        if metrics.get("market_cap_yi") is not None
    }


# 官方成分补全：进程内只算一次(membership 一趟跑完即结束)
_LATEST_MEMBER_METRICS_CACHE: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}


def _member_metrics_cached(*, allow_spot_fallback: bool = True) -> Dict[str, Dict[str, Optional[float]]]:
    cache_key = "spot" if allow_spot_fallback else "local"
    if cache_key not in _LATEST_MEMBER_METRICS_CACHE:
        _LATEST_MEMBER_METRICS_CACHE[cache_key] = load_latest_member_metrics(
            allow_spot_fallback=allow_spot_fallback
        )
    return _LATEST_MEMBER_METRICS_CACHE[cache_key]


def _parse_official_member_rows(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], str]:
    if df is None or df.empty:
        return [], "官方回退空表"
    code_col = _find_col(df.columns, "证券代码", "股票代码", "代码")
    name_col = _find_col(df.columns, "证券名称", "股票简称", "名称")
    ratio_col = _find_col(df.columns, "市值占比", "最新权重", "权重")
    if not code_col or not name_col:
        return [], f"官方回退缺少代码/名称列: {list(df.columns)}"
    metrics_map: Optional[Dict[str, Dict[str, Optional[float]]]] = None
    members = []
    excluded = []
    raw_count = 0
    for _, r in df.iterrows():
        c = _norm_code(r.get(code_col))
        n = str(r.get(name_col, "")).strip()
        if not c or not n:
            continue
        raw_count += 1
        if _is_stock_excluded(c, n):
            excluded.append(f"{c} {n}")
            continue
        if metrics_map is None:
            metrics_map = _member_metrics_cached(allow_spot_fallback=False)
        metrics = metrics_map.get(c, {})
        official_ratio = _to_float_or_none(r.get(ratio_col)) if ratio_col else None
        members.append({
            "code": c,
            "name": n,
            "price": metrics.get("price"),
            "market_cap_yi": metrics.get("market_cap_yi"),
            "roe_pct": metrics.get("roe_pct"),
            "profit_growth_pct": metrics.get("profit_growth_pct"),
            "revenue_growth_pct": metrics.get("revenue_growth_pct"),
            "official_market_cap_ratio": official_ratio,
            "index_weight": official_ratio,
        })
    if members:
        return members, ""
    if raw_count and excluded:
        shown = "、".join(excluded[:5])
        more = f" 等 {len(excluded)} 只" if len(excluded) > 5 else ""
        return [], f"官方回退仅返回被过滤股票: {shown}{more}"
    return [], "官方回退无有效股票"


def _fetch_members_official(segment_code: str) -> List[Dict[str, Any]]:
    """申万宏源官方成分接口兜底(legulegu 失败/空时)。

    官方接口与 legulegu 同款指数代码(850xxx，去掉 .SI)，但只回 证券代码/证券名称等成分字段；
    这里用本地主库补价格/市值/基本面，使落库 schema 与 legulegu 成分表保持一致。
    """
    df = _retry_fetch(fetch_sw3_segment_constituents_official, segment_code,
                      retries=3, sleep_sec=0.5, backoff=True)
    members, _reason = _parse_official_member_rows(df)
    return members


def _fetch_official_market_cap_ratio_map(segment_code: str) -> Dict[str, float]:
    df = _retry_fetch(fetch_sw3_segment_constituents_official, segment_code,
                      retries=2, sleep_sec=0.8, backoff=True)
    if df is None or df.empty:
        return {}
    code_col = _find_col(df.columns, "证券代码", "股票代码", "代码")
    ratio_col = _find_col(df.columns, "市值占比", "最新权重", "权重")
    if not code_col or not ratio_col:
        return {}
    out = {}
    for _, row in df.iterrows():
        code = _norm_code(row.get(code_col))
        ratio = _to_float_or_none(row.get(ratio_col))
        if code and ratio is not None:
            out[code] = ratio
    return out


def _fetch_official_weight_map(segment_code: str) -> Dict[str, float]:
    """Compatibility alias: old callers used "weight" for official market-cap ratio."""
    return _fetch_official_market_cap_ratio_map(segment_code)


def _fetch_segment_members(segment_code: str, skip_legulegu: bool = False,
                           retry_desc: str = "") -> Tuple[List[Dict[str, Any]], str, str]:
    """取单个三级赛道成分：主源 legulegu HTML 表 → 空/失败则换申万宏源官方成分接口兜底。

    skip_legulegu=True 时直接走官方接口——legulegu 已确认宕机/连续失败后用，避免每个赛道
    都白白先撞 3 次死掉的 legulegu(那会把 336 赛道拖成几十分钟)。
    返回 (members, source, err)：source ∈ {"legulegu","官方",""}；members 非空时 err 为 ""。
    """
    err = ""
    if not skip_legulegu:
        try:
            df = _retry_fetch(fetch_sw3_segment_constituents, segment_code,
                              retries=2, sleep_sec=LEGULEGU_RETRY_SLEEP_SEC, backoff=False,
                              desc=f"{retry_desc} legulegu" if retry_desc else "")
            members = _parse_segment_rows(df, min_market_cap_yi=0.0) if df is not None and not df.empty else []
            if members:
                return members, "legulegu", ""
        except Exception as exc:
            err = str(exc)
    try:
        df = _retry_fetch(fetch_sw3_segment_constituents_official, segment_code,
                          retries=1, sleep_sec=0.5, backoff=True,
                          desc=f"{retry_desc} 官方" if retry_desc else "")
        members, official_reason = _parse_official_member_rows(df)
        if members:
            return members, "官方", ""
    except Exception as exc:
        official_reason = str(exc)
    reasons = [item for item in (err, official_reason) if item]
    return [], "", "；".join(reasons) if reasons else "空表(疑似限流)"


def _enrich_membership_member_metrics(
        membership: Dict[str, Any],
        *,
        allow_weight_fetch: bool = False,
) -> Tuple[int, int]:
    """给旧 membership 缓存补价格/市值/基本面。

    早期官方兜底只落了 code/name，旧缓存会导致 build 阶段因 market_cap_yi 缺失全被过滤；
    这里在 build 前按本地主库/东财实时行情补齐，补到后顺手回写 sw3_member。
    """
    metrics_map: Optional[Dict[str, Dict[str, Optional[float]]]] = None
    filled_metrics = 0
    filled_ratios = 0
    fields = ("price", "market_cap_yi", "roe_pct", "profit_growth_pct", "revenue_growth_pct")
    for seg in membership.get("segments", []):
        ratio_map: Optional[Dict[str, float]] = None
        for member in seg.get("members", []):
            code = _norm_code(member.get("code"))
            if not code:
                continue
            if member.get("official_market_cap_ratio") is None and member.get("index_weight") is not None:
                member["official_market_cap_ratio"] = member.get("index_weight")
            if not all(member.get(field) is not None for field in fields):
                if metrics_map is None:
                    metrics_map = load_latest_member_metrics(allow_spot_fallback=False)
                metrics = metrics_map.get(code)
                if metrics:
                    changed = False
                    for field in fields:
                        if member.get(field) is None and metrics.get(field) is not None:
                            member[field] = metrics[field]
                            changed = True
                    if changed:
                        filled_metrics += 1
            if (allow_weight_fetch
                    and member.get("market_cap_yi") is None
                    and member.get("official_market_cap_ratio") is None):
                if ratio_map is None:
                    try:
                        ratio_map = _fetch_official_market_cap_ratio_map(seg.get("segment_code", ""))
                    except Exception:
                        ratio_map = {}
                ratio = ratio_map.get(code)
                if ratio is not None:
                    member["official_market_cap_ratio"] = ratio
                    member["index_weight"] = ratio
                    filled_ratios += 1
    return filled_metrics, filled_ratios


def _segments_df_from_cache(cache: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """用归属快照/备份(segments 列表)拼出三级总表 DataFrame(legulegu 列表失败时回退)。"""
    rows = [{
        "行业代码": s.get("segment_code", ""),
        "行业名称": s.get("segment_name", ""),
        "上级行业": s.get("parent_segment", ""),
        "成份个数": s.get("member_count", 0),
    } for s in (cache or {}).get("segments", [])
        if s.get("segment_code") and s.get("segment_name")]
    return pd.DataFrame(rows)


def load_segment_backup() -> List[Dict[str, Any]]:
    """读取三级行业总表灾备文件(legulegu 彻底失败时兜底)。返回 sw3_segment 行列表，缺失返回 []。"""
    try:
        with open(META_BACKUP_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("segments") or []
    return []


def _segments_df_from_backup() -> pd.DataFrame:
    """从灾备文件拼出三级总表 DataFrame。"""
    return _segments_df_from_cache({"segments": load_segment_backup()})


def _db_sw3_segment_rows() -> List[Dict[str, Any]]:
    if not Path(STOCK_DB_FILE).exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{STOCK_DB_FILE}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT segment_code, segment_name, parent_segment, member_count "
                "FROM sw3_segment ORDER BY segment_code"
            ).fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows]
    except sqlite3.Error:
        return []


def _membership_db_segment_count() -> int:
    return len(_db_sw3_segment_rows())


def _db_segment_count_too_small(db_count: int, backup_count: int) -> bool:
    return backup_count > 0 and db_count < backup_count * LOCAL_TAXONOMY_MIN_BACKUP_RATIO


def membership_db_needs_full_recrawl() -> Tuple[bool, int, int]:
    """DB 总表明显少于灾备时视为损坏，先全量重建 membership DB。"""
    backup_count = len(load_segment_backup())
    db_count = _membership_db_segment_count()
    return _db_segment_count_too_small(db_count, backup_count), db_count, backup_count


def _load_local_taxonomy(prev: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """本地三级总表(全量段，含无成分/失败段)：优先读 DB sw3_segment 全表，再退灾备文件。

    总表近乎静态(申万约一年才重分类一次)：crawl_sw3_membership 联网拉一次落库后，切片刷新直接复用，
    不必每次重拉 legulegu。注意必须读 sw3_segment 全表，而非 load 出来的 membership——后者只含有
    成分的段，会漏掉"在总表但成分待补/上次失败"的段，导致切片永远发现不了它们。
    """
    backup = load_segment_backup()
    db_rows = _db_sw3_segment_rows()
    if db_rows and not _db_segment_count_too_small(len(db_rows), len(backup)):
        df = _segments_df_from_cache({"segments": db_rows})
        if not df.empty:
            return df
    if db_rows and backup and _db_segment_count_too_small(len(db_rows), len(backup)):
        print(f"  [membership] 本地DB三级总表仅 {len(db_rows)}/{len(backup)} 段，低于备份80%，改用灾备总表",
              flush=True)
    df = _segments_df_from_backup()
    if not df.empty:
        return df
    return _segments_df_from_cache(prev)


def export_segment_backup() -> None:
    """把主库 sw3_segment 总表全量回写到灾备文件，保持备份新鲜。

    只导出段层(代码/名称/上级/成份数/refreshed_at/error)，与人工备份格式一致；成分股不入备份
    (恢复时由官方接口补)。带防回退保护：主库段数明显少于现有备份(<80%)时跳过，避免覆盖关键备份。
    """
    conn = stock_storage.connect(STOCK_DB_FILE)
    try:
        rows = conn.execute(
            "SELECT segment_code, segment_name, parent_segment, member_count, "
            "refreshed_at, error, updated_at FROM sw3_segment ORDER BY segment_code"
        ).fetchall()
    finally:
        conn.close()
    segs = [dict(r) for r in rows]
    if not segs:
        print("  [membership] 备份导出跳过：主库无三级总表", flush=True)
        return
    existing = load_segment_backup()
    if existing and len(segs) < len(existing) * 0.8:
        print(f"  [membership] 备份导出跳过：主库 {len(segs)} 段 < 现有备份 {len(existing)} 段的80%，"
              f"避免覆盖更全的关键备份", flush=True)
        return
    META_BACKUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    # 维持原备份的紧凑格式(每个对象一行)，避免每次导出把整文件 diff 炸开
    body = ",\n".join(json.dumps(s, ensure_ascii=False, separators=(",", ":")) for s in segs)
    with open(META_BACKUP_FILE, "w", encoding="utf-8") as f:
        f.write(f"[{body}]")
    print(f"  [membership] 已回写 {len(segs)} 个三级行业总表 -> {META_BACKUP_FILE}", flush=True)


def _passes_pool_filters(
        code: str,
        name: str,
        price: Optional[float],
        market_cap_yi: Optional[float],
        *,
        min_market_cap_yi: float,
        max_market_cap_yi: Optional[float],
        max_price: Optional[float],
) -> bool:
    """池子准入过滤：排除北交/ST/新股、极低价、绝对股价上限(可选)、市值区间。

    给 build 期 DataFrame 解析和 fast-build 缓存合并两条路径共用，保证口径一致。
    """
    if not _passes_basic_stock_filters(code, name, price, max_price=max_price):
        return False
    if market_cap_yi is None or market_cap_yi < min_market_cap_yi:
        return False
    # 不再对市值设绝对上限；TopK 只在各自三级行业内部比较。
    if max_market_cap_yi is not None and market_cap_yi > max_market_cap_yi:
        return False
    return True


def _passes_basic_stock_filters(
        code: str,
        name: str,
        price: Optional[float],
        *,
        max_price: Optional[float],
) -> bool:
    """不依赖市值口径的基础过滤，用于市值/官方市值占比两种规模口径。"""
    if not code or not name or _is_stock_excluded(code, name):
        return False
    if price is not None:
        # 只保留极低价剔除；不再用绝对股价上限误杀行业龙头
        if price < 2.0:
            return False
        if max_price is not None and price > max_price:
            return False
    return True


def _choose_segment_size_basis(rows: List[Dict[str, Any]]) -> str:
    """每个行业只选一种规模口径：全员有市值则用市值，否则全员用官方市值占比。"""
    if not rows:
        return ""
    if all(row.get("market_cap_yi") is not None for row in rows):
        return "market_cap_yi"
    if all(row.get("official_market_cap_ratio") is not None for row in rows):
        return "official_market_cap_ratio"
    return ""


def _parse_segment_rows(
        df: pd.DataFrame,
        *,
        min_market_cap_yi: float,
        max_market_cap_yi: Optional[float] = None,
        max_price: Optional[float] = None,
) -> List[Dict[str, Any]]:
    cols = list(df.columns)
    code_col = _find_col(cols, "股票代码")
    name_col = _find_col(cols, "股票简称")
    price_col = _find_col(cols, "价格")
    cap_col = _find_col(cols, "市值")
    roe_col = _find_col(cols, "ROE")
    profit_col = _find_col(cols, "净利润增速")
    revenue_col = _find_col(cols, "营收增速")
    if not code_col or not name_col or not cap_col:
        return []

    rows = []
    for _, stock in df.iterrows():
        code = _norm_code(stock.get(code_col))
        name = str(stock.get(name_col, "")).strip()
        price = _to_float_or_none(stock.get(price_col)) if price_col else None
        market_cap_yi = _to_float_or_none(stock.get(cap_col))
        if not _passes_pool_filters(
                code, name, price, market_cap_yi,
                min_market_cap_yi=min_market_cap_yi,
                max_market_cap_yi=max_market_cap_yi,
                max_price=max_price):
            continue

        rows.append({
            "code": code,
            "name": name,
            "price": price,
            "market_cap_yi": market_cap_yi,
            "roe_pct": _to_float_or_none(stock.get(roe_col)) if roe_col else None,
            "profit_growth_pct": _to_float_or_none(stock.get(profit_col)) if profit_col else None,
            "revenue_growth_pct": _to_float_or_none(stock.get(revenue_col)) if revenue_col else None,
        })
    return rows


def score_segment_leaders(rows: List[Dict[str, Any]]) -> None:
    """行业内部龙头分：规模 50% + ROE 25% + 成长 25%。"""
    for row in rows:
        row.setdefault("size_proxy", row.get("market_cap_yi"))
    cap_score = _percentile_scores(rows, "size_proxy")
    roe_score = _percentile_scores(rows, "roe_pct")
    profit_score = _percentile_scores(rows, "profit_growth_pct")
    revenue_score = _percentile_scores(rows, "revenue_growth_pct")

    for item in rows:
        code = item["code"]
        growth_score = max(profit_score[code], revenue_score[code])
        item["leader_score"] = round(
            cap_score[code] * 0.50
            + roe_score[code] * 0.25
            + growth_score * 0.25,
            1,
        )


def _report_segment_changes(prev_names: Dict[str, str], cur_names: Dict[str, str]) -> None:
    """对比旧缓存与当前申万三级总表，打印新增/消失的三级行业。"""
    added = [c for c in cur_names if c not in prev_names]
    removed = [c for c in prev_names if c not in cur_names]
    if not added and not removed:
        return
    print(f"  [membership] 结构变化：新增 {len(added)} / 消失 {len(removed)} 个三级行业", flush=True)
    if added:
        names = ", ".join(f"{cur_names[c]}({c})" for c in added[:8])
        print(f"    + 新增: {names}{' ...' if len(added) > 8 else ''}", flush=True)
    if removed:
        names = ", ".join(f"{prev_names[c]}({c})" for c in removed[:8])
        print(f"    - 消失: {names}{' ...' if len(removed) > 8 else ''}", flush=True)


def _load_sw3_membership_from_db(max_age_days: Optional[int]) -> Optional[Dict[str, Any]]:
    conn = stock_storage.connect(STOCK_DB_FILE)
    try:
        return stock_storage.load_sw3_membership(conn, max_age_days=max_age_days)
    finally:
        conn.close()


def _save_sw3_membership_to_db(payload: Dict[str, Any]) -> None:
    conn = stock_storage.connect(STOCK_DB_FILE)
    try:
        stock_storage.save_sw3_membership(conn, payload)
    finally:
        conn.close()


def _mark_leaders_in_db(segments: List[Dict[str, Any]]) -> int:
    """把龙头池里每个赛道选出的龙头 code 回写到主库 sw3_member.is_leader。

    save_sw3_membership 的增量同步会保留现有标记；龙头池生成完成后仍以本轮结果全量
    重打标，让 is_leader 始终反映最新选股，供主力雷达直接 WHERE is_leader=1 取池。
    """
    codes = {
        lead.get("code")
        for seg in segments
        for lead in seg.get("leaders", [])
        if lead.get("code")
    }
    if not codes:
        return 0
    conn = stock_storage.connect(STOCK_DB_FILE)
    try:
        marked = stock_storage.mark_sw3_leaders(conn, codes)
    finally:
        conn.close()
    print(f"  [membership] 已回写 {marked} 只龙头标记 -> sw3_member.is_leader", flush=True)
    return marked


def crawl_sw3_membership(
        sleep_sec: float = 0.6,
        resume: bool = True,
        force_official_members: bool = False,
) -> Dict[str, Any]:
    """爬取申万三级"行业→成分股"归属并落盘缓存（支持断点续传）。

    总表(三级行业列表)在此拉取一次(带重试)：成功即据此重建并落库，作为后续切片的本地总表；
    失败则回退本地备份文件/旧快照；成分阶段直接走官方兜底，避免对异常 legulegu 逐行业等待。
    成分股变动需刷新，故 per-segment 带退避重试 + 官方兜底；resume 复用已抓赛道、只补缺失。
    日常滚动保鲜走 refresh_oldest_segments(复用缓存总表，不重拉)。
    """
    print(f"[{datetime.now():%H:%M:%S}] [membership] 拉取申万三级行业总表...", flush=True)
    # 先载入旧缓存：既给 resume 续传用，又是总表拉取失败时的回退底稿(配合备份文件)
    prev = _load_sw3_membership_from_db(max_age_days=None) or {}
    db_needs_recrawl, db_count, backup_count = membership_db_needs_full_recrawl()
    first_membership_run = not any(seg.get("members") for seg in prev.get("segments", []))
    official_members_only = force_official_members or db_needs_recrawl or first_membership_run
    taxonomy_from_fallback = False
    try:
        sw3 = _retry_fetch(fetch_sw3_segments, retries=3, sleep_sec=LEGULEGU_RETRY_SLEEP_SEC,
                           backoff=False, jitter=0.0,
                           desc="申万三级行业列表")
    except Exception as exc:
        print(f"  [membership] 三级总表抓取失败：{exc}", flush=True)
        sw3 = None
    reference_count = max(
        int(db_count or 0),
        int(backup_count or 0),
        len(prev.get("segments") or []),
    )
    if (
        sw3 is not None
        and not sw3.empty
        and _taxonomy_response_looks_truncated(len(sw3), reference_count)
    ):
        print(
            f"  [membership] 三级总表疑似截断({len(sw3)}/{reference_count}，不高于80%)，"
            "拒绝据此删除旧赛道并回退完整快照",
            flush=True,
        )
        sw3 = None
    if sw3 is None or sw3.empty:
        taxonomy_from_fallback = True
        # 总表拉取失败(重试仍挂) → 优先备份文件、再退本地旧快照；成分阶段直接走官方兜底。
        sw3 = _segments_df_from_backup()
        src = f"备份文件 {META_BACKUP_FILE}"
        if sw3.empty:
            sw3 = _segments_df_from_cache(prev)
            src = "本地旧快照"
        if sw3.empty:
            raise RuntimeError("无法获取三级总表：legulegu 失败且无备份/快照，请先联网拉一次")
        print(f"  [membership] 总表拉取失败，回退{src}的 {len(sw3)} 个三级行业"
              "(成分跳过 legulegu，直接官方兜底)", flush=True)
    else:
        print(f"  [membership] 已拉取三级总表，共 {len(sw3)} 个行业", flush=True)
    if official_members_only and not taxonomy_from_fallback:
        reason = "首次运行" if first_membership_run else "DB低于备份80%" if db_needs_recrawl else "强制重建"
        print(f"  [membership] 成分抓取模式：{reason}，直接官方接口(跳过逐行业 legulegu)", flush=True)

    # 记录旧赛道(名)用于报结构变化；resume 时复用已抓到的赛道
    done: Dict[str, Dict[str, Any]] = {}
    prev_names: Dict[str, str] = {}
    for seg in prev.get("segments", []):
        if seg.get("members"):
            prev_names[seg.get("segment_code")] = seg.get("segment_name", "")
            if resume:
                done[seg.get("segment_code")] = seg
    if resume and done:
        print(f"  [membership] 续传：已有 {len(done)} 个赛道，本趟只补抓缺失的", flush=True)

    segments = []
    errors = []
    universe_names: Dict[str, str] = {}  # 当前总表里"在范围内"的赛道(用于报结构变化)
    reused = fetched = official_used = 0
    last_log = time.monotonic()
    total = len(sw3)
    pending = sum(1 for _, r in sw3.iterrows()
                  if str(r.get("行业代码", "")).strip() not in done)
    if pending:
        source_hint = "直接官方兜底" if (taxonomy_from_fallback or official_members_only) else "逐行业 legulegu→官方兜底"
        print(f"  [membership] 待抓成分 {pending}/{total} 个赛道"
              f"（{source_hint}，无固定间隔）...", flush=True)
    for idx, row in sw3.iterrows():
        segment_code = str(row.get("行业代码", "")).strip()
        segment_name = str(row.get("行业名称", "")).strip()
        parent_name = str(row.get("上级行业", "")).strip()
        member_count = int(_to_float_or_none(row.get("成份个数")) or 0)
        if not segment_code or not segment_name:
            continue
        universe_names[segment_code] = segment_name

        if segment_code in done:
            segments.append(done[segment_code])
            reused += 1
            continue

        # 总表已回退时跳过 legulegu；否则主源 legulegu → 空/失败再官方兜底。
        members, source, err = _fetch_segment_members(
            segment_code,
            skip_legulegu=taxonomy_from_fallback or official_members_only,
        )

        if members:
            segments.append({
                "segment_code": segment_code,
                "segment_name": segment_name,
                "parent_segment": parent_name,
                "member_count": member_count,
                "members": members,
                "refreshed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            fetched += 1
            if source == "官方":
                official_used += 1
        else:
            errors.append({"segment_code": segment_code, "segment_name": segment_name, "error": err})
        # 心跳：每 ~10s 报一次进度(新抓/失败都会推进)，避免抓成分阶段长时间静默
        now = time.monotonic()
        if now - last_log >= 10.0:
            last_log = now
            print(f"  [membership] 进度 {idx + 1}/{total}"
                  f"（新抓 {fetched} / 复用 {reused} / 失败 {len(errors)}）当前 {segment_name}", flush=True)
    _report_segment_changes(prev_names, universe_names)
    payload = {
        "schema": SW3_MEMBERSHIP_SCHEMA,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "申万三级行业成分(Legulegu 主源 / 申万宏源官方兜底, 缓存)",
        "segment_count": len(segments),
        "segments": segments,
        # DB 同步必须拿到完整失败集合，才能保护每个失败赛道的旧成员/信号；
        # 展示层如需限长，应在输出时截断，不能在持久化前截断。
        "errors": errors,
    }
    _save_sw3_membership_to_db(payload)
    print(f"  -> 归属缓存已落库 {STOCK_DB_FILE}"
          f"（共 {len(segments)} 个行业：复用 {reused} + 新抓 {fetched}"
          f"{f'(其中官方兜底 {official_used})' if official_used else ''}，本趟失败 {len(errors)}）")
    export_segment_backup()  # 全量重建成功 → 回写灾备文件保鲜
    return payload


def refresh_oldest_segments(n: int = DEFAULT_REFRESH_SLICE, sleep_sec: float = 0.6) -> Dict[str, Any]:
    """切片增量：优先补"上次跑批失败/缺失"的，再刷最旧的，共 n 个申万三级赛道。

    先探测 legulegu 总表：总表可用则按 legulegu 成分主源刷新，单赛道失败再官方兜底；
    总表不可用则退回本地总表，并直接走官方成分接口。
    优先级：上次失败(errors 名单)或缺失/空 的赛道最先补，其余按 refreshed_at 由旧到新。
    """
    prev = _load_sw3_membership_from_db(max_age_days=None) or {}
    cached = {s.get("segment_code"): s for s in prev.get("segments", [])}
    # 失败名单：刷成功就移除，仍失败就保留/加入
    errors_map = {e.get("segment_code"): e for e in prev.get("errors", []) if e.get("segment_code")}

    taxonomy_from_fallback = False
    try:
        sw3 = _retry_fetch(fetch_sw3_segments, retries=1, sleep_sec=0, backoff=False, timeout=8)
        if sw3 is None or sw3.empty:
            raise RuntimeError("legulegu 三级总表为空")
        print(f"  [slice] 已拉取 legulegu 三级总表，共 {len(sw3)} 个行业", flush=True)
    except Exception as exc:
        taxonomy_from_fallback = True
        print(f"  [slice] legulegu 总表拉取失败：{exc}；回退本地总表，成分直接官方兜底", flush=True)
        sw3 = _load_local_taxonomy(prev)
    if sw3.empty:
        raise RuntimeError("无本地三级总表(DB快照/备份均空)，无法切片刷新；请先 recrawl 联网拉一次总表")
    universe = []
    for _, row in sw3.iterrows():
        code = str(row.get("行业代码", "")).strip()
        name = str(row.get("行业名称", "")).strip()
        parent = str(row.get("上级行业", "")).strip()
        mc = int(_to_float_or_none(row.get("成份个数")) or 0)
        if not code or not name:
            continue
        universe.append((code, name, parent, mc))

    def sort_key(u):
        seg = cached.get(u[0])
        # bucket 0 = 上次失败 或 缺失/空 -> 最优先；bucket 1 = 已有数据按 refreshed_at 由旧到新
        if u[0] in errors_map or not seg or not seg.get("members"):
            return (0, seg.get("refreshed_at", "") if seg else "")
        return (1, seg.get("refreshed_at") or "")

    universe.sort(key=sort_key)
    targets = universe[:max(0, n)]
    source_hint = "官方兜底" if taxonomy_from_fallback else "legulegu→官方兜底"
    print(f"[{datetime.now():%H:%M:%S}] [slice] 刷新最旧 {len(targets)} 个赛道"
          f"（缓存共 {len(cached)} / 全集 {len(universe)}，{source_hint}）...", flush=True)

    refreshed = failed = official_used = 0
    for idx, (code, name, parent, mc) in enumerate(targets, 1):
        retry_desc = f"slice {idx}/{len(targets)} {name}({code})"
        print(f"  [slice] {idx}/{len(targets)} 获取 {name}({code}) 成分...", flush=True)
        members, source, err = _fetch_segment_members(
            code,
            skip_legulegu=taxonomy_from_fallback,
            retry_desc=retry_desc,
        )
        if members:
            cached[code] = {
                "segment_code": code, "segment_name": name, "parent_segment": parent,
                "member_count": mc, "members": members,
                "refreshed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            refreshed += 1
            if source == "官方":
                official_used += 1
            errors_map.pop(code, None)  # 刷成功 -> 移出失败名单
            print(f"  [slice] {idx}/{len(targets)} {name} 完成：{len(members)} 只，来源 {source}", flush=True)
        else:
            failed += 1
            errors_map[code] = {"segment_code": code, "segment_name": name, "error": err}
            print(f"  [slice] {idx}/{len(targets)} {name} 失败：{err}", flush=True)
    segments = list(cached.values())
    payload = {
        "schema": SW3_MEMBERSHIP_SCHEMA,
        # 切片即视为"缓存被主动维护"，刷新 generated_at 让 TTL 不会整体过期触发全量重爬
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "申万三级行业成分(Legulegu 主源 / 申万宏源官方兜底, 缓存+切片增量)",
        "segment_count": len(segments),
        "segments": segments,
        "errors": list(errors_map.values()),
    }
    _save_sw3_membership_to_db(payload)
    print(f"  -> 切片刷新完成：成功 {refreshed} / 失败 {failed}"
          f"{f'(其中官方兜底 {official_used})' if official_used else ''}，缓存现有 {len(segments)} 个赛道", flush=True)
    export_segment_backup()  # 刷完最老 topK 后，把主库三级总表全量回写灾备文件
    return payload


def load_sw3_membership(max_age_days: Optional[int] = MEMBERSHIP_MAX_AGE_DAYS) -> Optional[Dict[str, Any]]:
    """从主库 sw3_segment/sw3_member 拼回归属缓存 dict；过期/为空返回 None。"""
    return _load_sw3_membership_from_db(max_age_days)


def build_segment_leader_pool(
        top_per_segment: int = 2,
        min_market_cap_yi: float = 10.0,
        refresh_membership: bool = False,
        refresh_slice: int = 0,
        enrich_weights_online: bool = False,
        forced_leader_codes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """生成细分行业龙头池并落盘（归属缓存 + 主库K线，可选滚动刷新 membership）。

    refresh_membership=True 强制全量重抓归属(resume=False, 感知改名/成分调整/删除)；
    refresh_slice>0 则只切片补抓最旧的几个赛道(滚动保鲜)；两者都不给时直接用现有缓存。
    """
    if refresh_membership:
        print("  [membership] recrawl: 重拉总表 + 全量重抓(resume=False)，感知行业改名/成分/删除...", flush=True)
        needs_recrawl, db_count, backup_count = membership_db_needs_full_recrawl()
        membership = crawl_sw3_membership(
            resume=False,
            force_official_members=needs_recrawl or db_count == 0,
        )
    else:
        rebuilt_membership = False
        needs_recrawl, db_count, backup_count = membership_db_needs_full_recrawl()
        if needs_recrawl:
            print(f"  [membership] 主库三级总表仅 {db_count}/{backup_count} 段，低于备份80%，先全量重建DB...",
                  flush=True)
            membership = crawl_sw3_membership(resume=False, force_official_members=True)
            rebuilt_membership = True
        else:
            membership = load_sw3_membership()
        if membership is None:
            print("  [membership] 缓存缺失/过期，全量爬取(可续传)...", flush=True)
            membership = crawl_sw3_membership(force_official_members=True)
        elif refresh_slice > 0 and not rebuilt_membership:
            # 有可用缓存：只切片补抓最旧的几个赛道，几次请求即可，不触发限流。
            # 切片失败(legulegu 抽风)不应让重建崩，沿用现有缓存继续。
            try:
                membership = refresh_oldest_segments(refresh_slice)
            except Exception as exc:
                print(f"  [slice] 切片补抓失败({exc!r})，沿用现有缓存继续", flush=True)

    filled_metrics, filled_ratios = _enrich_membership_member_metrics(
        membership,
        allow_weight_fetch=enrich_weights_online,
    )
    if filled_metrics or filled_ratios:
        parts = []
        if filled_metrics:
            parts.append(f"{filled_metrics} 只价格/市值指标")
        if filled_ratios:
            parts.append(f"{filled_ratios} 只官方市值占比")
        print(f"  [membership] 已补齐 {'、'.join(parts)}", flush=True)
        try:
            _save_sw3_membership_to_db(membership)
        except Exception as exc:
            print(f"  [membership] 指标补齐回写失败({exc!r})，继续生成龙头池", flush=True)

    forced_source = FORCED_SEGMENT_LEADER_CODES if forced_leader_codes is None else forced_leader_codes
    forced_codes = {code for code in (_norm_code(item) for item in forced_source) if code}
    segments = []

    for seg in membership.get("segments", []):
        segment_code = seg.get("segment_code", "")
        segment_name = seg.get("segment_name", "")
        parent_name = seg.get("parent_segment", "")
        member_count = seg.get("member_count", 0)

        raw_rows = []
        for m in seg.get("members", []):
            code = m.get("code", "")
            name = m.get("name", "")
            price = m.get("price")
            market_cap_yi = m.get("market_cap_yi")
            official_ratio = m.get("official_market_cap_ratio")
            if official_ratio is None:
                official_ratio = m.get("index_weight")
            if not _passes_basic_stock_filters(code, name, price, max_price=None):
                continue
            raw_rows.append({
                "code": code,
                "name": name,
                "price": price,
                "market_cap_yi": market_cap_yi,
                "official_market_cap_ratio": official_ratio,
                "index_weight": official_ratio,
                "roe_pct": m.get("roe_pct"),
                "profit_growth_pct": m.get("profit_growth_pct"),
                "revenue_growth_pct": m.get("revenue_growth_pct"),
            })
        size_basis = _choose_segment_size_basis(raw_rows)
        rows = []
        for row in raw_rows:
            if size_basis == "market_cap_yi":
                if not _passes_pool_filters(
                    row["code"], row["name"], row["price"], row["market_cap_yi"],
                    min_market_cap_yi=min_market_cap_yi,
                    max_market_cap_yi=None,
                    max_price=None,
                ):
                    continue
                row["size_proxy"] = row["market_cap_yi"]
            elif size_basis == "official_market_cap_ratio":
                if row.get("official_market_cap_ratio") is None:
                    continue
                row["size_proxy"] = row["official_market_cap_ratio"]
            else:
                continue
            row["size_basis"] = size_basis
            rows.append(row)
        if not rows:
            continue

        score_segment_leaders(rows)
        rows.sort(key=lambda item: (-item["leader_score"], -(item.get("size_proxy") or 0.0)))
        leaders = rows[:top_per_segment]
        if forced_codes:
            selected_codes = {item["code"] for item in leaders}
            leaders.extend(
                item for item in rows
                if item["code"] in forced_codes and item["code"] not in selected_codes
            )

        segments.append({
            "segment_code": segment_code,
            "segment_name": segment_name,
            "parent_segment": parent_name,
            "member_count": member_count,
            "candidate_count": len(rows),
            "size_basis": size_basis,
            "leaders": [
                {
                    "rank": rank,
                    "code": item["code"],
                    "name": item["name"],
                    "leader_score": item["leader_score"],
                    "market_cap_yi": item["market_cap_yi"],
                    "official_market_cap_ratio": item.get("official_market_cap_ratio"),
                    "index_weight": item.get("index_weight"),
                    "size_basis": item.get("size_basis"),
                }
                for rank, item in enumerate(leaders, 1)
            ],
        })

    leader_count = len({lead["code"] for seg in segments for lead in seg["leaders"]})
    payload = {
        "schema": SEGMENT_LEADER_SCHEMA,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "申万三级归属+价/市值/基本面(Legulegu缓存) + 行业内综合排名",
        "params": {
            "top_per_segment": top_per_segment,
            "min_market_cap_yi": min_market_cap_yi,
            "forced_leader_codes": sorted(forced_codes),
            "membership_generated_at": membership.get("generated_at"),
        },
        "segment_count": len(segments),
        "leader_count": leader_count,
        "segments": segments,
        "errors": membership.get("errors", []),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if refresh_slice > 0:
        try:
            with open(SEGMENT_LEADER_POOL_FILE, "r", encoding="utf-8") as f:
                previous = json.load(f)
            previous_segments = previous.get("segments") or []
            previous_leaders = int(previous.get("leader_count") or 0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            previous_segments = []
            previous_leaders = 0
        if previous_segments and previous_leaders > leader_count:
            current_by_code = {seg.get("segment_code"): seg for seg in segments if seg.get("segment_code")}
            merged_segments = []
            retained = replaced = 0
            for old_seg in previous_segments:
                code = old_seg.get("segment_code")
                if code in current_by_code:
                    merged_segments.append(current_by_code.pop(code))
                    replaced += 1
                else:
                    merged_segments.append(old_seg)
                    retained += 1
            merged_segments.extend(current_by_code.values())
            segments = merged_segments
            leader_count = len({lead["code"] for seg in segments for lead in seg.get("leaders", [])})
            payload["segment_count"] = len(segments)
            payload["leader_count"] = leader_count
            payload["segments"] = segments
            print(f"  [membership] 滚动刷新合并旧龙头池：更新 {replaced} 个赛道，保留 {retained} 个赛道")
    if leader_count == 0 and any(seg.get("members") for seg in membership.get("segments", [])):
        try:
            with open(SEGMENT_LEADER_POOL_FILE, "r", encoding="utf-8") as f:
                previous = json.load(f)
            previous_leaders = int(previous.get("leader_count") or 0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            previous = None
            previous_leaders = 0
        if previous and previous_leaders > 0:
            print(f"  [membership] 本次未生成龙头，保留已有 {previous_leaders} 只龙头池，避免空结果覆盖")
            _mark_leaders_in_db(previous.get("segments", []))
            return previous
    with open(SEGMENT_LEADER_POOL_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  -> 已生成 {len(segments)} 个细分行业 / {leader_count} 只龙头，落盘 {SEGMENT_LEADER_POOL_FILE}")
    _mark_leaders_in_db(segments)
    return payload


def load_segment_leader_pool(max_age_days: Optional[int] = 14) -> Optional[Dict[str, Any]]:
    try:
        with open(SEGMENT_LEADER_POOL_FILE, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema") != SEGMENT_LEADER_SCHEMA:
        return None
    if max_age_days is None or max_age_days <= 0:
        return payload
    try:
        generated = datetime.strptime(payload.get("generated_at", ""), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    if (datetime.now() - generated).days > max_age_days:
        return None
    return payload


def show_segment_leader_pool(segment_show_limit: int = 0) -> None:
    payload = load_segment_leader_pool(max_age_days=DEFAULT_POOL_MAX_AGE_DAYS)
    if payload is None:
        print(f"[ERROR] 找不到可用股票池: {SEGMENT_LEADER_POOL_FILE}")
        print("  先运行: python stock_crawl_segment_leaders.py crawl")
        return

    segments = payload.get("segments", [])
    shown_segments = segments[:segment_show_limit] if segment_show_limit > 0 else segments

    print("=" * 108)
    print("  细分领域龙头股票池")
    print(f"  生成时间: {payload.get('generated_at')} · 细分行业: {payload.get('segment_count')}"
          f" · 龙头候选: {payload.get('leader_count')} · 失败行业: {len(payload.get('errors') or [])}")
    print(f"  文件: {SEGMENT_LEADER_POOL_FILE}")
    print("=" * 108)
    print("  按申万三级细分行业分组:")
    print("-" * 108)
    for segment in shown_segments:
        parts = []
        for leader in segment.get("leaders", []):
            cap = leader.get("market_cap_yi")
            weight = leader.get("official_market_cap_ratio")
            if weight is None:
                weight = leader.get("index_weight")
            basis = leader.get("size_basis") or segment.get("size_basis")
            if basis == "market_cap_yi" and cap is not None:
                cap_text = f" 市值{cap:.0f}亿"
            elif basis == "official_market_cap_ratio" and weight is not None:
                cap_text = f" 市值占比{weight:.2f}%"
            elif cap is not None:
                cap_text = f" 市值{cap:.0f}亿"
            elif weight is not None:
                cap_text = f" 市值占比{weight:.2f}%"
            else:
                cap_text = ""
            parts.append(
                f"{leader.get('rank')}.{leader.get('code')} {leader.get('name')}"
                f"(分{leader.get('leader_score')}{cap_text})"
            )
        print(f"  {segment.get('parent_segment')} | {segment.get('segment_name')}[{segment.get('member_count')}]: "
              + (" / ".join(parts) if parts else "-"))
    if len(shown_segments) < len(segments):
        print(f"  ... 已省略 {len(segments) - len(shown_segments)} 个细分行业；完整数据见 JSON 文件")
    errors = payload.get("errors") or []
    if errors:
        print("-" * 108)
        print("  失败/空候选行业:")
        for item in errors[:20]:
            print(f"  {item.get('segment_name')}({item.get('segment_code')}): {item.get('error')}")
        if len(errors) > 20:
            print(f"  ... 已省略 {len(errors) - 20} 个失败行业；完整数据见 JSON 文件")
    print("=" * 108)


def _build_arg_parser() -> argparse.ArgumentParser:
    """三个命令：show 展示 / crawl 正常爬取 / recrawl 全量重爬。"""
    examples = (
        "示例:\n"
        "  python stock_crawl_segment_leaders.py            展示已爬取的细分龙头池(默认)\n"
        "  python stock_crawl_segment_leaders.py crawl      正常爬取: 读归属缓存(滚动补抓最旧赛道) + 主库K线重建\n"
        "  python stock_crawl_segment_leaders.py recrawl    全量重爬: 重抓全部申万三级归属后再重建\n"
    )
    parser = argparse.ArgumentParser(
        description="细分行业龙头爬虫（申万三级归属=legulegu缓存，指标=归属缓存+本地主库基本面）",
        epilog=examples, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "action", nargs="?", default="show", choices=["show", "crawl", "recrawl"],
        help="show=展示已爬数据(默认) / crawl=正常爬取重建 / recrawl=全量重爬")
    return parser


def main() -> None:
    action = _build_arg_parser().parse_args().action
    if action == "show":
        show_segment_leader_pool()
    elif action == "crawl":
        build_segment_leader_pool(
            top_per_segment=DEFAULT_TOP_PER_SEGMENT,
            refresh_slice=DEFAULT_REFRESH_SLICE,
        )
    elif action == "recrawl":
        build_segment_leader_pool(
            top_per_segment=DEFAULT_TOP_PER_SEGMENT,
            refresh_membership=True,
        )


if __name__ == "__main__":
    main()
