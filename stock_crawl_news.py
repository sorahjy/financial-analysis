"""个股新闻/题材数据层：爬取 + 入库 + 题材关键词打标。最小管线。

动机：游资标的 5 条件里唯一筛不出来的是①「题材富矿 + 对齐风口」——主库此前无 news 表，
只能用 sw2_heat 板块动量做弱代理。本层补"个股新闻 +
题材标签"，作为 latent 潜伏妖股观察名单的催化剂证据（⚠️展示/证据用，未做预测力验证、不进排序权重）。

数据源：akshare stock_news_em（东财·个股新闻，每只返回最近约 10 条，含关键词/标题/时间/来源/链接）。
  ——这是「当下催化」滚动快照（非历史全量）；多次运行用 INSERT OR IGNORE 跨日累积。
  东财概念成分(push2)本机被拒、同花顺概念成分函数已被 akshare 删（见记忆），故走"新闻 + 关键词打标"路线。

题材标签：用 THEME_KEYWORDS 词典对(关键词 + 标题)做包含匹配，落 themes 列(逗号分隔)。

用法：
  python stock_crawl_news.py --no-proxy --pool hotmoney        # 爬游资小盘池(默认)
  python stock_crawl_news.py --no-proxy --pool leader --limit 50
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

import stock_storage
from stock_crawl_common import retry_fetch_or_none, safe_print, strip_proxy_env

TABLE = "stock_news"
DEFAULT_WORKERS = 6        # 别 >6：东财新闻接口本机限流
DEFAULT_POOL = "hotmoney"

# 题材词典：标签 → 命中关键词。匹配(东财关键词列 + 标题)的子串。粗粒度、可持续扩充。
THEME_KEYWORDS: Dict[str, List[str]] = {
    "存储/HBM": ["HBM", "存储", "DDR", "DRAM", "NAND", "闪存", "先进封装", "固态硬盘", "颗粒"],
    "AI算力": ["算力", "AI", "人工智能", "大模型", "CPO", "光模块", "液冷", "数据中心", "英伟达", "GPU"],
    "半导体": ["半导体", "芯片", "晶圆", "光刻", "封测", "IGBT", "碳化硅", "EDA", "流片"],
    "信创/国产替代": ["信创", "国产替代", "自主可控", "操作系统", "数据库", "鸿蒙"],
    "华为链": ["华为", "鸿蒙", "昇腾", "麒麟", "Mate", "盘古"],
    "机器人": ["机器人", "人形", "灵巧手", "减速器", "丝杠", "具身"],
    "新能源车/电池": ["固态电池", "钠电", "锂电", "电池", "充电", "新能源车", "动力电池"],
    "光伏风电": ["光伏", "风电", "组件", "硅料", "储能", "逆变器"],
    "并购重组": ["并购", "重组", "资产注入", "重大资产", "借壳", "收购"],
    "稀土磁材": ["稀土", "磁材", "永磁", "钕铁硼"],
    "军工/低空": ["军工", "低空", "eVTOL", "无人机", "导弹", "卫星", "商业航天"],
    "国资/中特估": ["国资", "中特估", "央企", "市值管理", "国企改革"],
    "回购增持": ["回购", "增持", "注销"],
    "医药": ["创新药", "减肥药", "GLP", "CXO", "疫苗", "医保"],
    "消费": ["消费", "白酒", "免税", "旅游", "零售"],
}


def tag_themes(keyword: str, title: str) -> str:
    """从(关键词 + 标题)抽题材标签，逗号分隔；无命中返回空串。"""
    text = f"{keyword or ''} {title or ''}"
    hits = [theme for theme, kws in THEME_KEYWORDS.items() if any(kw in text for kw in kws)]
    return ",".join(hits)


def ensure_table(conn) -> None:
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {TABLE} (
            code TEXT NOT NULL,
            pub_time TEXT NOT NULL,        -- 发布时间 YYYY-MM-DD HH:MM:SS
            title TEXT NOT NULL,
            source TEXT,
            url TEXT,
            keyword TEXT,                  -- 东财返回的命中关键词
            themes TEXT,                   -- 题材词典打标(逗号分隔)
            fetched_at TEXT,
            PRIMARY KEY (code, pub_time, title)
        )"""
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_code_time ON {TABLE}(code, pub_time)")
    conn.commit()


def _fetch_one(code: str) -> List[Dict[str, Any]]:
    import akshare as ak
    df = retry_fetch_or_none(ak.stock_news_em, symbol=str(code).zfill(6))
    if df is None or getattr(df, "empty", True):
        return []
    recs: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        pub = str(row.get("发布时间") or "")[:19]
        title = str(row.get("新闻标题") or "").strip()
        if not pub or not title:
            continue
        keyword = str(row.get("关键词") or "")
        recs.append({
            "pub_time": pub,
            "title": title,
            "source": str(row.get("文章来源") or "") or None,
            "url": str(row.get("新闻链接") or "") or None,
            "keyword": keyword or None,
            "themes": tag_themes(keyword, title) or None,
        })
    return recs


def _save(conn, code: str, recs: List[Dict[str, Any]]) -> int:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.executemany(
        f"INSERT OR IGNORE INTO {TABLE} "
        "(code, pub_time, title, source, url, keyword, themes, fetched_at) VALUES (?,?,?,?,?,?,?,?)",
        [(code, r["pub_time"], r["title"], r["source"], r["url"], r["keyword"], r["themes"], now)
         for r in recs],
    )
    conn.commit()
    return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0


def crawl(pool: str, limit: Optional[int], workers: int) -> None:
    conn = stock_storage.connect()
    ensure_table(conn)
    codes = [stock_storage._normalize_code(m["code"]) for m in stock_storage.pool_members(conn, pool)]
    codes = [c for c in codes if c]
    if limit:
        codes = codes[:limit]
    safe_print(f"[news] 池={pool} · 待爬 {len(codes)} 只 · workers={workers}")
    if not codes:
        conn.close()
        safe_print("[news] 候选池为空。")
        return
    done = ok = total_new = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, c): c for c in codes}
        for fut in as_completed(futs):
            code = futs[fut]
            done += 1
            try:
                recs = fut.result()
            except Exception as exc:
                recs = []
                safe_print(f"  [WARN] {code} 失败: {exc}")
            if recs:
                total_new += _save(conn, code, recs)
                ok += 1
            if done % 50 == 0 or done == len(codes):
                safe_print(f"  进度 {done}/{len(codes)} · 有新闻 {ok} · 新增 {total_new} 行")
    n_total = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    n_themed = conn.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE themes IS NOT NULL").fetchone()[0]
    conn.close()
    safe_print(f"[news] 完成：{ok}/{len(codes)} 只有新闻，本次新增 {total_new} 行；"
               f"库内共 {n_total} 行({n_themed} 带题材标签)入库 {TABLE}")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="个股新闻/题材数据层爬取")
    ap.add_argument("--pool", choices=("leader", "hotmoney"),
                    default=DEFAULT_POOL, help="候选池：hotmoney(默认游资小盘) / leader(细分龙头)")
    ap.add_argument("--limit", type=int, default=None, help="只爬前 N 只(小样本试)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="爬取线程数(别>6)")
    ap.add_argument("--no-proxy", action="store_true", help="清理代理环境变量(境内行情/新闻接口要求)")
    args = ap.parse_args(argv)
    if args.no_proxy:
        os.environ["STOCK_CRAWL_NO_PROXY"] = "1"
    strip_proxy_env()
    crawl(args.pool, args.limit, args.workers)


if __name__ == "__main__":
    main()
