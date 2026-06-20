"""主力资金雷达。

候选池 = 细分行业龙头（stock_crawl_segment_leaders 选出、回写主库 sw3_member.is_leader）。
潜伏分目标：捕捉「左侧吸筹」——主力在低位悄悄建仓、但价格还没起飞的阶段。

  调研背景：「放量+创新高」经 verify 回测证明是右侧追高(截面 RankIC 显著为负)。吸筹的真正
  指纹是方向性的——参考 Wyckoff/VSA、A股筹码分布、OBV/ADL 三套体系，落地三个判别信号：

ambush 吸筹分（换手率优先、缺失退回成交量）：
  位置  价格中低位(留出上行空间)                    权重 0.20  —— 筹码在低成本区
  背离  努力与结果背离：放量(努力) × 价格横住(结果)    权重 0.25  —— 放量却不涨=有人吸货
  买压  Chaikin Money Flow：收盘持续靠近K线上半部     权重 0.25  —— 收上半部=买方吸收
  筹码  低位筹码单峰密集(换手衰减+三角成本分布)       权重 0.30  —— 成本趋同=主力控盘
  penalty 一字封板/连板                              打折       —— 已启动,非吸筹

另叠加「游资形态匹配」(规格见 meta_data_backup/hot_money_patterns.md)：把游资坐庄的「吸筹→试盘→
洗盘→突破→拉升→出货」六段套路编码成 match_patterns() 的布尔匹配器，给每只票打形态标签，再由
_pattern_phase() 汇总成一个主导阶段（详见下表 + 该函数 docstring）。

─────────────────────────────────────────────────────────────────────────────
游资形态总表（PATTERNS，6 类 / 20 个，编号 P1-P20）
  · 列：编号  名称  —— 命中条件（位置=收盘价近60日分位；量比=近5日均量/前20日均量；
        漂移=近20日涨跌幅；CMF=Chaikin资金流；筹码集中=主峰±7%价带内筹码占比）。
  · 信号方向 buy/hold/sell；阶段配色按操作进程由早到晚渐变：吸筹🟢→试盘/洗盘🟡→突破/拉升🟠→出货🔴（空仓观望⚪）。
  · 阶段优先级见 _pattern_phase：出货 > 突破 > 吸筹/洗盘 > 拉升 > 试盘。
─────────────────────────────────────────────────────────────────────────────
【吸筹 🟢buy】主力在低位悄悄建仓
  P1 低位横盘磨人      位置<0.40 + 近20日振幅<18% + |漂移|<8%：低位窄幅横盘磨人
  P2 低位影线吸筹      位置<0.40 + 近15日≥3根十字星/长下影(下影>2×实体且>1.5%)：反复探底留下影
  P3 缩量阴线打压吸筹  位置<0.45 + 近10日有"大阴跌≥4%却缩量(<0.85×30日均量)、随后收复"：隐性吸货
  P4 量增价稳吸收      位置<0.60 + 量比>1.2 + |漂移|<6% + CMF>0：放量但价稳、资金净流入
  P5 底部形态构筑      位置<0.45 + 近端两摆动低点等高/低点抬高(-4%~+8%) + 中间反弹≥6%(颈线) + 当前价回升未破颈线：双底/W底
【试盘 🟡hold】拉升前试探上方抛压
  P6 试盘长上影        近8日有长上影(>3%且>2×实体)创20日新高后收盘缩回：探顶又压回
  P7 底部异动放量      位置<0.40 + 量比>1.5 + 近5日最大振幅>7%：低位突然放量异动
【洗盘 🟡buy】震仓甩浮筹、不破结构
  P8 缩量回踩洗盘      站上MA20 + 回踩近10日高点-2%~-15% + 跌日量<涨日量 + 筹码集中≥0.40：挖坑不破位
  P9 边拉边洗          多头排列(MA5>10>20) + 近8日涨跌符号切换≥4次 + 低点抬高：边拉边洗
  P10 高换手洗盘       量比>1.5 + 收盘>MA20 + 筹码集中≥0.45：高换手震仓但筹码峰不发散
【突破 🟠hold（阶段标签=头号买点，优先级仅次出货）】放量右进
  P11 放量突破启动     右进左出触发器命中：从低/中位放量(>1.3×)突破近20日收盘高点，刚启动的右侧买点
【拉升 🟠hold】主升浪，追入=接盘
  P12 连板拉升         连续涨停 streak≥2
  P13 首板卡位         今日首板(近20日无涨停) + 换手10%~45%
【出货 🔴sell】高位派发，当风控示警
  P14 高位放量滞涨     位置≥0.85 + 量比>1.5 + 近5日涨幅≤2% + 上影>2%：高位放量不涨
  P15 量价背离         创60日新高却缩量(量比<1.0)、或放量(>1.8)却5日涨幅<1%：量价背离顶
  P16 阴天量           位置≥0.80 + 近5日出现40日最大量且收阴：天量收阴
  P17 倒V反转          位置≥0.80 + 较前期冲高>15%后从峰值回落≤-8%：冲高倒V
  P18 顶部大阴包阳      位置≥0.80 + 昨阳今阴且今实体完全吞没昨实体：顶部看跌吞没
  P19 灌压巨量大阴      位置≥0.70 + 量比>1.8 + 实体跌>6%且收在当日价区下1/4：巨量灌压大阴
  P20 均线放量破位      收盘跌破MA20(5日前还在MA20上方) + 量比>1.2：放量破位
─────────────────────────────────────────────────────────────────────────────

Modes:
  ambush   默认。给候选龙头算吸筹分 + 匹配游资形态，排名落盘。
  watch    实时交易监控入口（仍为骨架，后续接盘口/实时行情）。
  verify   吸筹分后验回测：PIT 重算分数 vs 未来前向收益(分位单调性/截面RankIC/多空价差/触发器超额)。
  patterns 形态预测力后验：每个形态命中后，未来 N 日相对同日全体的超额收益(剔大盘 beta)+t，
           验证哪些形态真有预测力(buy 看正超额、sell 看负超额)。

═══════════════════════════════════════════════════════════════════════════════
研究纪要（verify 回测沉淀，universe=细分龙头 / 区间 2023-2026 / 截面 Rank IC 为准）
═══════════════════════════════════════════════════════════════════════════════
口径说明：pooled 分位收益含大盘 beta（高分扎堆普涨日→假象）；**截面 Rank IC / 多空价差 /
触发器超额** 是每个 as-of 日内部排序再跨日平均，天然剔除 beta，才是真实选股力。

迭代与结论：
  1) 放量+创新高（最初）  → 截面 IC 5/10/20日 -0.035/-0.039/-0.048 (t≈-3.6~-4.5)。
     = 右侧追高/见顶探测器；高分票未来 1-4 周反而走弱（短期反转）。和「天价见天量」自洽。
  2) 左侧吸筹 v1（量增价稳）→ 负 IC 砍半到 ~-0.02，仍未转正（量比本身就是反转特征）。
  3) 特征级诊断（10/60/120日）→ vol_ratio/振幅/动量/位置/换手 **全周期 IC 全负、6 个月不翻正**；
     龙头池被「短期反转 + 低波动异象」主导，没有任何「强势/放量」特征有正预测力。
  4) 三信号吸筹分 v2（位置+努力结果背离+CMF买压+低位筹码集中，本文件实现）→ 20/40/60日
     IC -0.001/-0.003/-0.017。**负 IC 被彻底中和到中性**（信号不再追高接盘），但没转正。
  5) 市值分层 → 大盘(>368亿)显著负(60日 IC-0.048,t-3.78)，**全池负 IC 是大盘贡献的**；
     小/中盘(<368亿)翻微正。故加市值上限 MAX_MARKET_CAP_YI=300 剔除大盘。
  6) cap-filtered(591只) + 右进左出突破触发器 →
       · 吸筹分截面 IC 翻微正：20/40/60日 +0.004/+0.006/+0.004（胜率达60%@40日），但 t<1 不显著；
       · **右进左出触发器失败**：触发样本绝对收益看着高(+1.31%>全体+0.85%)是 beta 假象，
         截面超额为负 -1.2%/-2.0%/-3.0% (t-1.3~-1.7)，胜率<50%。买突破=接盘，与(1)同源。
  7) v3 三角筹码分布 + 市值缺失用 stock_history 回补后，cap-filtered(582只) →
       · 吸筹分截面 IC 20/40/60日 +0.008/+0.005/-0.001，t<1，仍是微正/中性；
       · 右进触发器继续负超额：40日 -2.0%(t-1.4)。结论仍是只看状态，不追突破。

核心结论：龙头池（大盘质量股）截面被**短期反转**主导——追强/突破/放量创高都亏，只有「安静的
吸筹状态」（低位+资金悄悄流入+不动）微弱正向但不显著。量价 hot-money 命题在此池拿不到稳健
alpha。**根因疑为 universe 错配**：游资/主力打的是小盘低流通题材股，不是细分龙头；DB 目前只有
龙头 K 线（956 龙头+41 非龙头），无独立小盘池。下一步若要真正检验命题，需联网爬真·小盘/题材
股池在游资主场重跑 verify。当前代码=cap-filtered 三信号吸筹分 + v3 三角筹码分布，作为研究基线沉淀。

  8) 游资形态层（P1-P20, match_patterns + patterns 模式）→ patterns 后验(582龙头/4.3万样本)：
     · 出货类经验证有效：P19 灌压巨量大阴 40日超额 -4.97%、P20 均线放量破位 -1.39%(t-2.16)。
     · buy 类多为反转陷阱(P8 洗盘 -1.17%/t-3.0、P10 -1.61%/t-3.0)；唯 P3 缩量阴线打压吸筹
       +3.35%/60日+3.94%(t1.47 近显著)有正向苗头。拉升类 P11/P12 负超额，追入=接盘。
     · P5 底部形态构筑(双底/W底, 后补)→ 命中2080, 20/40/60日 -0.65%/-0.84%/-0.82%(t-1.46)，
       buy 形态却负超额=又一反转陷阱；同 universe 错配，识别本身没问题但龙头池无预测力。
     · 结论：出货预警层(P19/P20)可直接当风控；buy 侧除 P3 外(含P5)在龙头池失效——与全文一致。
     · 备注：形态只产出"阶段标签"、不进吸筹分加权(分数=4因子)，故 P5 失效无需"降权"，留作研究基线。
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import stock_storage


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CAPITAL_DIR = DATA_DIR / "capital"
DB_FILE = stock_storage.DEFAULT_DB_FILE
AMBUSH_RESULT_FILE = CAPITAL_DIR / "hot_money_ambush.json"
THEME_CANDIDATES_FILE = CAPITAL_DIR / "theme_candidates.json"   # stock_theme_candidates.py 落盘
WATCH_STATE_FILE = CAPITAL_DIR / "hot_money_watch.json"
VERIFY_RESULT_FILE = CAPITAL_DIR / "hot_money_verify.json"
PATTERNS_RESULT_FILE = CAPITAL_DIR / "hot_money_patterns.json"
SCHEMA = "hot_money_radar.v3"
MODES = ("ambush", "watch", "verify", "patterns")
DEFAULT_MODE = "ambush"

# ── 打分参数（集中放置，便于后续调参/优化器接管）──────────────
LOOKBACK = 90            # 每只龙头读取的近端有效交易日数（够算筹码分布 + 60 日分位）
MIN_BARS = 40            # 少于这么多有效 bar 视为数据不足，不打分
SHORT_WIN = 5            # 近端放量窗口
BASE_WIN = 20            # 量比基线窗口（紧邻近端窗口之前）
HIGH_WIN = 60            # 价格分位窗口
DRIFT_WIN = 20           # 努力与结果背离：看近 20 日累计涨跌幅是否横住
VOL_RATIO_FULL = 2.5     # 量比达到该值给满分（1.0→0 分，线性）
POS_LOW = 0.25           # 收盘价分位 ≤ 此值 → 中低位满分（留足上行空间）
POS_HIGH = 0.70          # 收盘价分位 ≥ 此值 → 已拉升，位置分 0
CMF_WIN = 20             # Chaikin Money Flow 窗口
CMF_FULL = 0.20          # CMF 达到该值给满分（≤0 → 0 分）
CHIP_BUCKETS = 80        # 成本分布价格网格数
CHIP_DECAY = 1.0         # 历史换手衰减系数（n=1：今日换手多少就搬移多少昨日筹码）
CHIP_BAND = 0.07         # 主峰 ±7% 价带内筹码占比 = 单峰密集集中度
CHIP_CONC_LO = 0.25      # 集中度 ≤ 此值 → 0 分
CHIP_CONC_HI = 0.60      # 集中度 ≥ 此值 → 满分
CHIP_PEAK_FULL = 0.35    # 主峰处在近端价格区间低 35% 内 → 低位满分
CHIP_PEAK_ZERO = 0.70    # 主峰处在近端价格区间高 70% 以上 → 低位 0 分
CHIP_PRICE_BELOW_ZERO = -0.08  # 当前价低于主峰太多，说明尚未站回成本区
CHIP_PRICE_ABOVE_FULL = 0.08   # 当前价略高于主峰内仍视为吸筹未充分拉升
CHIP_PRICE_ABOVE_ZERO = 0.22   # 当前价高于主峰过多，视为已启动
CHIP_WINNER_MIN = 0.15         # 获利盘过低，上方套牢压力重
CHIP_WINNER_FULL_LO = 0.25
CHIP_WINNER_FULL_HI = 0.60     # 获利盘中低位最佳：不是全套牢，也不是满获利
CHIP_WINNER_MAX = 0.82
SEALED_AMP = 0.005       # 日内振幅 ≤ 0.5% 视为一字封死板
SEALED_PENALTY_PER = 0.2  # 每个一字封板的打折系数
SEALED_PENALTY_CAP = 0.6  # 一字封板最多打掉 60%
TURNOVER_COVERAGE = 0.7  # 近端窗口换手率覆盖率达标才用换手率，否则退回成交量
WEIGHTS = {"position": 0.20, "divergence": 0.25, "cmf": 0.25, "chip": 0.30}
SUSPECT_ACCUM_SCORE = 65  # 无形态命中但吸筹分≥此值 → 疑似吸筹(待确认)；低于则空仓观望

# 阶段标签按游资操作顺序排列（疑似吸筹→吸筹→试盘→洗盘→突破→拉升→出货，空仓观望=场外）。
# 表头计数据此从左到右展示；标签字符串须与 _pattern_phase() 的返回值完全一致。
PHASE_ORDER: Tuple[str, ...] = (
    "疑似吸筹(待确认)🟢",
    "吸筹🟢",
    "试盘🟡",
    "洗盘🟡",
    "吸筹+洗盘🟡",
    "▲突破🟠",
    "拉升中🟠",
    "出货预警🔴",
    "空仓观望⚪",
)

# 市值上限：大盘是反转陷阱(verify 分层显示 >368亿 IC≈-0.05 显著负)，雷达只看小/中盘龙头。
MAX_MARKET_CAP_YI = 300.0

# 右进左出触发器：吸筹分给"状态"，触发器给"买点"——从低/中位整理放量突破近端高点。
TRIGGER_WIN = 20          # 突破窗口：放量突破近 20 日收盘高点
TRIGGER_VOL_MULT = 1.3    # 突破当日量 > 1.3× 前 20 日均量
TRIGGER_BASE_MAX_POS = 0.70  # 被突破的前高在近 60 日分位 < 0.70 → 从低/中位突破(非高位追涨)

# ── 后验回测参数 ──────────────────────────────────────────────
VERIFY_HORIZONS = (20, 40, 60)     # 吸筹是中期信号，前向持有期拉长（交易日）
VERIFY_STEP = 10                   # 每隔多少个交易日取一个 as-of 截面
VERIFY_WINDOW_DAYS = 750           # 回测窗口（交易日，约 3 年）
VERIFY_MIN_NAMES = 30              # 单个截面至少多少只票才计入截面 IC / 多空
VERIFY_BUCKETS = 5                 # 吸筹分分位桶数（五等分）


# ── 基础工具 ──────────────────────────────────────────────────

def _limit_pct(code: str) -> float:
    """个股涨跌停幅度（粗分；ST 5% 细节忽略，作为噪声接受）。"""
    if code.startswith(("300", "301", "688", "689")):
        return 20.0
    if code.startswith(("8", "4")):  # 北交所
        return 30.0
    return 10.0


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 5:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def _ranks(values: Sequence[float]) -> List[float]:
    """平均秩（处理并列），供 Spearman 用。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 5:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _clip01(value: float) -> float:
    return 0.0 if value < 0 else 1.0 if value > 1 else value


def _safe(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ── 候选池：细分龙头（主库 is_leader=1）────────────────────────

def load_leader_candidates(conn: sqlite3.Connection,
                           max_cap: Optional[float] = MAX_MARKET_CAP_YI) -> List[Dict[str, Any]]:
    """从主库取已标记的细分龙头作为候选池（stock_crawl_segment_leaders crawl 后回写）。

    max_cap>0 时剔除总市值超过该值(亿元)的大盘股——verify 分层证明大盘吸筹分是反转陷阱；
    sw3_member 市值缺失会先由 stock_storage.leader_members 从 stock_history 最新非空 market_cap 回补。
    max_cap=None/0 则不按市值过滤。
    """
    rows = stock_storage.leader_members(conn)
    out: List[Dict[str, Any]] = []
    for row in rows:
        code = stock_storage._normalize_code(row.get("code"))
        if not code:
            continue
        cap = row.get("market_cap_yi")
        if max_cap and cap is not None and cap > max_cap:
            continue
        out.append({
            "code": code,
            "name": str(row.get("name") or ""),
            "market_cap_yi": cap,
            "segment_name": row.get("segment_name") or "",
            "parent_segment": row.get("parent_segment") or "",
        })
    return out


def _bar(row: sqlite3.Row) -> Dict[str, Any]:
    """把一行日线整成轻量短键 dict（解耦 DB 列名、PIT 切片更快）。"""
    return {
        "date": row["date"],
        "open": _safe(row["daily_open"]),
        "high": _safe(row["daily_high"]),
        "low": _safe(row["daily_low"]),
        "close": _safe(row["daily_close"]),
        "volume": _safe(row["daily_volume"]),
        "amount": _safe(row["daily_amount"]),
        "chg": _safe(row["daily_change_pct"]),
        "turnover": _safe(row["daily_turnover_rate"]),
    }


_BAR_SQL = (
    "SELECT date, daily_open, daily_high, daily_low, daily_close, daily_volume, daily_amount, "
    "daily_change_pct, daily_turnover_rate FROM stock_history "
    "WHERE code = ? AND daily_close IS NOT NULL AND daily_volume IS NOT NULL "
)


def _recent_bars(conn: sqlite3.Connection, code: str, limit: int = LOOKBACK) -> List[Dict[str, Any]]:
    """该 code 近 limit 个有效日线 bar（升序），跳过估值快照空行。"""
    rows = conn.execute(_BAR_SQL + "ORDER BY date DESC LIMIT ?", (code, limit)).fetchall()
    return [_bar(r) for r in reversed(rows)]


def _all_bars(conn: sqlite3.Connection, code: str) -> List[Dict[str, Any]]:
    """该 code 全部有效日线 bar（升序）。供 verify 做 PIT 滑窗。"""
    rows = conn.execute(_BAR_SQL + "ORDER BY date", (code,)).fetchall()
    return [_bar(r) for r in rows]


# ── 潜伏分（纯函数，ambush 取最新窗口 / verify 取历史窗口都复用）──

def _volume_series(bars: List[Dict[str, Any]]) -> Tuple[List[Optional[float]], str]:
    """挑量度量：近端换手率覆盖率达标用换手率（更干净，qfq 不复权 volume 有跳变），否则退回成交量。"""
    turns = [b["turnover"] for b in bars]
    coverage = sum(1 for t in turns if t is not None) / len(turns) if turns else 0.0
    if coverage >= TURNOVER_COVERAGE:
        return turns, "turnover"
    return [b["volume"] for b in bars], "volume"


def _score_volume_ratio(vol: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    """F1 量比抬升：近 SHORT_WIN 日均量 / 紧邻其前 BASE_WIN 日均量。"""
    if len(vol) < SHORT_WIN + BASE_WIN:
        return None, None
    short = [v for v in vol[-SHORT_WIN:] if v is not None]
    base = [v for v in vol[-(SHORT_WIN + BASE_WIN):-SHORT_WIN] if v is not None]
    short_avg, base_avg = _mean(short), _mean(base)
    if not short_avg or not base_avg or base_avg <= 0:
        return None, None
    ratio = short_avg / base_avg
    score = _clip01((ratio - 1.0) / (VOL_RATIO_FULL - 1.0)) * 100.0
    return score, ratio


def _score_position(bars: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """A2 价格中低位：收盘价在近 HIGH_WIN 日分位越低越好（留出上行空间，已拉升=0）。"""
    closes = [b["close"] for b in bars[-HIGH_WIN:] if b["close"] is not None]
    if len(closes) < 20:
        return None, None
    last = closes[-1]
    pct = sum(1 for c in closes if c <= last) / len(closes)
    score = _clip01((POS_HIGH - pct) / (POS_HIGH - POS_LOW)) * 100.0
    return score, pct


def _absorption_score(drift: float) -> float:
    """20 日累计涨跌幅 → 吸筹分。横盘微涨最佳；大跌(杀跌/出货)或大涨(已拉升)都归零。"""
    if drift <= -0.12 or drift >= 0.25:
        return 0.0
    if drift < -0.03:
        return (drift + 0.12) / 0.09 * 100.0     # -12%→0 升到 -3%→100
    if drift <= 0.08:
        return 100.0                              # -3%~+8% 横盘微涨：满分
    return (0.25 - drift) / 0.17 * 100.0          # +8%→100 降到 +25%→0


def _score_absorption(bars: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """近 DRIFT_WIN 日价格横住程度（结果端），杀跌/暴涨都不算吸筹。"""
    closes = [b["close"] for b in bars if b["close"] is not None]
    if len(closes) < DRIFT_WIN + 1:
        return None, None
    ref = closes[-1 - DRIFT_WIN]
    if not ref:
        return None, None
    drift = closes[-1] / ref - 1.0
    return _absorption_score(drift), drift


def _score_divergence(bars: List[Dict[str, Any]], vol: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    """信号①努力与结果背离：放量(努力) × 价格横住(结果) 连乘。放量却不涨=有人在吸货。"""
    s_vol, vol_ratio = _score_volume_ratio(vol)
    s_abs, drift = _score_absorption(bars)
    if vol_ratio is None or s_abs is None:
        return None, drift
    effort = _clip01((vol_ratio - 1.0) / (VOL_RATIO_FULL - 1.0))   # 量能抬升强度 0~1
    return effort * (s_abs / 100.0) * 100.0, drift


def _money_flow_mult(bar: Dict[str, Any]) -> Optional[float]:
    """Chaikin 资金流乘数 MFM = ((C−L)−(H−C))/(H−L) ∈ [−1,1]，收越靠上半部越正。"""
    h, l, c = bar["high"], bar["low"], bar["close"]
    if h is None or l is None or c is None or h <= l:
        return None   # 一字板 high==low → 无定义，跳过
    return ((c - l) - (h - c)) / (h - l)


def _score_cmf(bars: List[Dict[str, Any]], vol: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    """信号②收盘买压：Chaikin Money Flow = Σ(MFM×量)/Σ量，收盘持续靠上半部=买方吸收。"""
    num = den = 0.0
    for b, v in zip(bars[-CMF_WIN:], vol[-CMF_WIN:]):
        m = _money_flow_mult(b)
        if m is None or v is None:
            continue
        num += m * v
        den += v
    if den <= 0:
        return None, None
    cmf = num / den   # [-1, 1]
    return _clip01(cmf / CMF_FULL) * 100.0, cmf


def _bar_avg_price(bar: Dict[str, Any]) -> Optional[float]:
    """用成交额/成交量估算均价；不可用时退回典型价，供三角筹码分布作为峰值。"""
    h, l, c = bar["high"], bar["low"], bar["close"]
    amount, volume = bar.get("amount"), bar.get("volume")
    if amount and volume and amount > 0 and volume > 0 and h is not None and l is not None:
        for avg in (amount / (volume * 100.0), amount / volume):
            if l <= avg <= h:
                return avg
    vals = [x for x in (h, l, c) if x is not None]
    return sum(vals) / len(vals) if vals else None


def _triangular_weights(prices: np.ndarray, low: float, high: float, peak: float) -> np.ndarray:
    """当日换手筹码在 low-peak-high 间做三角分布。"""
    if len(prices) == 1 or high <= low:
        return np.ones(len(prices))
    peak = min(high, max(low, peak))
    weights = np.zeros(len(prices))
    if peak <= low:
        weights = (high - prices) / (high - low)
    elif peak >= high:
        weights = (prices - low) / (high - low)
    else:
        left = prices <= peak
        weights[left] = (prices[left] - low) / (peak - low)
        weights[~left] = (high - prices[~left]) / (high - peak)
    weights = np.maximum(weights, 0.0)
    if weights.sum() <= 0:
        weights[np.argmin(np.abs(prices - peak))] = 1.0
    return weights


def _chip_metrics(bars: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """信号③成本分布：换手衰减 + 当日筹码在 low-avg-high 三角铺开，重建流通筹码成本分布。

    返回 concentration(主峰±CHIP_BAND 内筹码占比)、peak_pctile(主峰价位分位)、
    price_to_peak(当前价相对主峰)、winner(获利盘比例)。换手率缺失太多则返回 None。
    """
    lows, highs, peaks, turns = [], [], [], []
    for b in bars:
        avg = _bar_avg_price(b)
        if b["low"] and b["high"] and avg is not None and b["turnover"] is not None and b["high"] > b["low"]:
            lows.append(b["low"]); highs.append(b["high"]); peaks.append(avg); turns.append(b["turnover"])
    if len(lows) < 30:
        return None
    pmin, pmax = min(lows), max(highs)
    if pmax <= pmin:
        return None
    grid = np.linspace(pmin, pmax, CHIP_BUCKETS)
    chips = np.zeros(CHIP_BUCKETS)
    span = pmax - pmin
    for lo, hi, peak, t in zip(lows, highs, peaks, turns):
        frac = min(1.0, max(0.0, t / 100.0 * CHIP_DECAY))   # 当日搬移的筹码比例
        chips *= (1.0 - frac)                                # 旧筹码按换手衰减
        i0 = int((lo - pmin) / span * (CHIP_BUCKETS - 1))
        i1 = int((hi - pmin) / span * (CHIP_BUCKETS - 1))
        i0 = max(0, i0); i1 = min(CHIP_BUCKETS - 1, max(i0, i1))
        weights = _triangular_weights(grid[i0:i1 + 1], lo, hi, peak)
        chips[i0:i1 + 1] += frac * weights / weights.sum()
    tot = chips.sum()
    if tot <= 0:
        return None
    chips /= tot
    close = bars[-1]["close"]
    peak_price = float(grid[int(chips.argmax())])
    concentration = float(chips[np.abs(grid - peak_price) <= CHIP_BAND * peak_price].sum())
    winner = float(chips[grid <= close].sum()) if close else 0.0
    peak_pctile = (peak_price - pmin) / span
    price_to_peak = close / peak_price - 1.0 if close and peak_price else None
    return {
        "concentration": concentration,
        "winner": winner,
        "peak_price": peak_price,
        "peak_pctile": peak_pctile,
        "price_to_peak": price_to_peak,
    }


def _score_chip(bars: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
    """信号③低位筹码集中：单峰密集 + 主峰低位 + 价格贴近主峰 + 获利盘中低位。"""
    m = _chip_metrics(bars)
    if m is None:
        return None, None
    conc = _clip01((m["concentration"] - CHIP_CONC_LO) / (CHIP_CONC_HI - CHIP_CONC_LO)) * 100.0
    peak_low = _clip01((CHIP_PEAK_ZERO - m["peak_pctile"]) / (CHIP_PEAK_ZERO - CHIP_PEAK_FULL)) * 100.0
    dist = m.get("price_to_peak")
    if dist is None or dist <= CHIP_PRICE_BELOW_ZERO or dist >= CHIP_PRICE_ABOVE_ZERO:
        price_near_peak = 0.0
    elif dist <= 0:
        price_near_peak = _clip01((dist - CHIP_PRICE_BELOW_ZERO) / (0.0 - CHIP_PRICE_BELOW_ZERO)) * 100.0
    elif dist <= CHIP_PRICE_ABOVE_FULL:
        price_near_peak = 100.0
    else:
        price_near_peak = _clip01((CHIP_PRICE_ABOVE_ZERO - dist) / (CHIP_PRICE_ABOVE_ZERO - CHIP_PRICE_ABOVE_FULL)) * 100.0

    winner = m["winner"]
    if winner <= CHIP_WINNER_MIN or winner >= CHIP_WINNER_MAX:
        winner_mid_low = 0.0
    elif winner < CHIP_WINNER_FULL_LO:
        winner_mid_low = _clip01((winner - CHIP_WINNER_MIN) / (CHIP_WINNER_FULL_LO - CHIP_WINNER_MIN)) * 100.0
    elif winner <= CHIP_WINNER_FULL_HI:
        winner_mid_low = 100.0
    else:
        winner_mid_low = _clip01((CHIP_WINNER_MAX - winner) / (CHIP_WINNER_MAX - CHIP_WINNER_FULL_HI)) * 100.0

    score = 0.45 * conc + 0.25 * peak_low + 0.20 * price_near_peak + 0.10 * winner_mid_low
    m["sub_concentration"] = conc
    m["sub_peak_low"] = peak_low
    m["sub_price_near_peak"] = price_near_peak
    m["sub_winner_mid_low"] = winner_mid_low
    return score, m


def _sealed_and_streak(bars: List[Dict[str, Any]], code: str) -> Tuple[int, int]:
    """近 SHORT_WIN 日一字封死板数量；以及当前连续涨停 streak 长度。"""
    limit = _limit_pct(code) - 0.3
    sealed = 0
    for i in range(max(1, len(bars) - SHORT_WIN), len(bars)):
        chg, prev_close = bars[i]["chg"], bars[i - 1]["close"]
        high, low = bars[i]["high"], bars[i]["low"]
        if chg is None or not prev_close or high is None or low is None:
            continue
        if chg >= limit and (high - low) / prev_close <= SEALED_AMP:
            sealed += 1
    streak = 0
    for b in reversed(bars):
        if b["chg"] is not None and b["chg"] >= limit:
            streak += 1
        else:
            break
    return sealed, streak


def _breakout_trigger(bars: List[Dict[str, Any]], vol: List[Optional[float]]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """右进左出触发器：从低/中位整理中放量突破近 TRIGGER_WIN 日收盘高点。

    左侧吸筹分给「在不在吸筹」的状态；触发器给「是否刚启动」的买点——只有当价格突破
    一个『位于近 60 日中低位的整理平台』且当日放量，才算右侧确认(避免高位追涨/缩量假突破)。
    """
    if len(bars) < HIGH_WIN + 1:
        return False, None
    closes = [b["close"] for b in bars]
    c = closes[-1]
    prior = [x for x in closes[-1 - TRIGGER_WIN:-1] if x]
    if not c or len(prior) < TRIGGER_WIN // 2:
        return False, None
    ceiling = max(prior)
    broke = c > ceiling
    base_vol = _mean([v for v in vol[-1 - TRIGGER_WIN:-1] if v])
    vol_confirm = bool(base_vol and vol[-1] and vol[-1] > TRIGGER_VOL_MULT * base_vol)
    win = [x for x in closes[-HIGH_WIN - 1:-1] if x]   # 突破前的近 60 日(不含今日)
    base_pctile = sum(1 for x in win if x <= ceiling) / len(win) if win else 1.0
    from_low = base_pctile < TRIGGER_BASE_MAX_POS
    triggered = bool(broke and vol_confirm and from_low)
    return triggered, {"breakout": bool(broke), "vol_confirm": vol_confirm, "base_pctile": round(base_pctile, 2)}


# ── 游资形态匹配器（P1-P20，规格见 data/hot_money_patterns.md）──────────
# 每个匹配器输入 (bars, ctx) 返回 bool（命中），PIT 安全（只用窗口内 bar）。
# 信号方向：buy=吸筹/洗盘(左侧) · hold=拉升(只标记不追) · sell=出货(风控/回避)。

def _kl_pc(bars: List[Dict[str, Any]], i: int) -> Optional[float]:
    return bars[i - 1]["close"] if i > 0 else None


def _kl_amp(bars, i):
    pc, h, l = _kl_pc(bars, i), bars[i]["high"], bars[i]["low"]
    return (h - l) / pc if pc and h is not None and l is not None else None


def _kl_body(bars, i):
    pc, o, c = _kl_pc(bars, i), bars[i]["open"], bars[i]["close"]
    return abs(c - o) / pc if pc and o is not None and c is not None else None


def _kl_upper(bars, i):
    pc, h, o, c = _kl_pc(bars, i), bars[i]["high"], bars[i]["open"], bars[i]["close"]
    return (h - max(o, c)) / pc if pc and None not in (h, o, c) else None


def _kl_lower(bars, i):
    pc, l, o, c = _kl_pc(bars, i), bars[i]["low"], bars[i]["open"], bars[i]["close"]
    return (min(o, c) - l) / pc if pc and None not in (l, o, c) else None


def _kl_doji(bars, i):
    b = _kl_body(bars, i)
    return b is not None and b < 0.005


def _ret_k(closes: List[Optional[float]], k: int) -> Optional[float]:
    if len(closes) > k and closes[-1] and closes[-1 - k]:
        return closes[-1] / closes[-1 - k] - 1
    return None


def _ma_last(closes: List[Optional[float]], n: int) -> Optional[float]:
    vals = [c for c in closes[-n:] if c is not None]
    return sum(vals) / len(vals) if len(vals) >= max(2, int(n * 0.6)) else None


def _avg_vol(vol: List[Optional[float]], a: int, b: int) -> Optional[float]:
    seg = [v for v in vol[a:b] if v]
    return sum(seg) / len(seg) if seg else None


def _swing_lows(values: List[float], k: int = 3) -> List[int]:
    """局部低点(摆动低)索引：values[i] 是前后各 k 根内的最小值。

    用于底部形态识别（双底/W底的两个谷）；相邻 k 根内只保留更低的一个，避免平台重复计点。
    """
    lows: List[int] = []
    n = len(values)
    for i in range(k, n - k):
        v = values[i]
        if v is None or v != min(values[i - k:i + k + 1]):
            continue
        if lows and i - lows[-1] <= k:        # 太近：保留更低者
            if v < values[lows[-1]]:
                lows[-1] = i
        else:
            lows.append(i)
    return lows


def _build_pattern_context(code: str, bars: List[Dict[str, Any]]) -> Dict[str, Any]:
    """一次性算齐形态匹配要用的量价上下文（避免每个匹配器各算一遍）。"""
    closes = [b["close"] for b in bars]
    vol, _ = _volume_series(bars)
    _, pos = _score_position(bars)
    _, vol_ratio = _score_volume_ratio(vol)
    _, drift = _score_absorption(bars)
    _, cmf = _score_cmf(bars, vol)
    chip = _chip_metrics(bars)
    sealed, streak = _sealed_and_streak(bars, code)
    triggered, _ = _breakout_trigger(bars, vol)
    ma = {n: _ma_last(closes, n) for n in (5, 10, 20, 60)}
    ma_bull = bool(ma[5] and ma[10] and ma[20] and ma[5] > ma[10] > ma[20]
                   and closes[-1] and closes[-1] > ma[5])
    return {
        "code": code, "closes": closes, "vol": vol,
        "pos": pos, "vol_ratio": vol_ratio, "drift": drift, "cmf": cmf,
        "chip": chip, "sealed": sealed, "streak": streak, "triggered": triggered,
        "ma5": ma[5], "ma10": ma[10], "ma20": ma[20], "ma60": ma[60], "ma_bull": ma_bull,
    }


# --- 吸筹 🟢buy ---
def _pat_low_consolidation(bars, ctx):           # P1 低位横盘磨人
    pos, drift, closes = ctx["pos"], ctx["drift"], ctx["closes"]
    if pos is None or drift is None:
        return False
    win = [c for c in closes[-20:] if c]
    if len(win) < 15:
        return False
    rng = (max(win) - min(win)) / (sum(win) / len(win))
    return pos < 0.40 and rng < 0.18 and abs(drift) < 0.08


def _pat_low_shadows(bars, ctx):                 # P2 低位十字星/长下影反复
    if ctx["pos"] is None or ctx["pos"] >= 0.40 or len(bars) < 16:
        return False
    cnt = 0
    for i in range(len(bars) - 15, len(bars)):
        body, low = _kl_body(bars, i), _kl_lower(bars, i)
        if _kl_doji(bars, i) or (body and low and low > 2 * body and low > 0.015):
            cnt += 1
    return cnt >= 3


def _pat_shakedown_absorb(bars, ctx):            # P3 隐性收集(缩量阴线打压吸筹)
    pos, vol, closes = ctx["pos"], ctx["vol"], ctx["closes"]
    if pos is None or pos >= 0.45 or len(bars) < 30:
        return False
    base = _avg_vol(vol, -30, -1)
    if not base:
        return False
    for i in range(len(bars) - 10, len(bars) - 1):
        chg, v = bars[i]["chg"], vol[i]
        if (chg is not None and chg <= -4 and v and v < 0.85 * base
                and closes[-1] and closes[i] and closes[-1] > closes[i]):
            return True
    return False


def _pat_absorption(bars, ctx):                  # P4 量增价稳(吸收)
    pos, vr, drift, cmf = ctx["pos"], ctx["vol_ratio"], ctx["drift"], ctx["cmf"]
    return (pos is not None and pos < 0.60 and vr is not None and vr > 1.2
            and drift is not None and abs(drift) < 0.06 and cmf is not None and cmf > 0)


def _pat_bottom_formation(bars, ctx):            # P5 底部形态构筑(双底/W底·低点抬高)
    """低位筑底：近端两个摆动低点等高或低点抬高、中间有像样反弹(颈线)，当前价正从二次探底
    回升但尚未显著突破颈线（突破后归 P11/拉升）。可选筹码不发散增强可信度。"""
    pos, closes, chip = ctx["pos"], ctx["closes"], ctx["chip"]
    if pos is None or pos >= 0.45:
        return False
    cs = [c for c in closes[-60:] if c]
    if len(cs) < 40:
        return False
    lows = _swing_lows(cs, k=3)
    if len(lows) < 2:
        return False
    i1, i2 = lows[-2], lows[-1]
    if i2 - i1 < 5:                               # 两底间隔太近不算结构
        return False
    l1, l2 = cs[i1], cs[i2]
    if not -0.04 <= l2 / l1 - 1 <= 0.08:          # 等高(±)或低点抬高，排除二次破位
        return False
    neck = max(cs[i1:i2 + 1])                     # 两底之间的反弹高点=颈线
    if neck / min(l1, l2) - 1 < 0.06:             # 中间反弹太弱，是平台不是双底
        return False
    c = cs[-1]
    if c <= l2 or c / neck - 1 > 0.03:            # 已回升但还没显著突破颈线(突破归拉升)
        return False
    return chip is None or chip["concentration"] >= 0.35


# --- 试盘 🟡hold ---
def _pat_test_upper_shadow(bars, ctx):           # P6 试盘长上影破平台又缩回
    if len(bars) < 30:
        return False
    closes = ctx["closes"]
    for i in range(len(bars) - 8, len(bars) - 1):
        up, body = _kl_upper(bars, i), _kl_body(bars, i)
        prior = [bars[j]["high"] for j in range(max(0, i - 20), i) if bars[j]["high"] is not None]
        if (up and body and up > 0.03 and up > 2 * body and prior and bars[i]["high"]
                and bars[i]["high"] > max(prior)
                and closes[-1] and closes[i] and closes[-1] < closes[i]):
            return True
    return False


def _pat_bottom_spike(bars, ctx):                # P7 底部异动放量
    pos, vr = ctx["pos"], ctx["vol_ratio"]
    if pos is None or pos >= 0.40 or vr is None or vr <= 1.5 or len(bars) < 6:
        return False
    amps = [a for a in (_kl_amp(bars, i) for i in range(len(bars) - 5, len(bars))) if a is not None]
    return bool(amps and max(amps) > 0.07)


# --- 洗盘 🟡buy ---
def _pat_pullback_shakeout(bars, ctx):           # P8 缩量回踩洗盘(挖坑)
    closes, vol, ma20, chip = ctx["closes"], ctx["vol"], ctx["ma20"], ctx["chip"]
    if ma20 is None or len(bars) < 12:
        return False
    c = closes[-1]
    if not c or c < ma20:
        return False
    hi = max([x for x in closes[-10:] if x] or [c])
    if not (-0.15 <= c / hi - 1 <= -0.02):
        return False
    down_v = [vol[i] for i in range(len(bars) - 8, len(bars))
              if bars[i]["chg"] is not None and bars[i]["chg"] < 0 and vol[i]]
    up_v = [vol[i] for i in range(len(bars) - 8, len(bars))
            if bars[i]["chg"] is not None and bars[i]["chg"] > 0 and vol[i]]
    if not down_v or not up_v:
        return False
    conc_ok = chip is None or chip["concentration"] >= 0.40
    return (sum(down_v) / len(down_v)) < (sum(up_v) / len(up_v)) and conc_ok


def _pat_climb_wash(bars, ctx):                  # P9 边拉边洗
    if not ctx["ma_bull"] or len(bars) < 12:
        return False
    signs = [1 if (bars[i]["chg"] or 0) > 0 else -1 for i in range(len(bars) - 8, len(bars))]
    alt = sum(1 for k in range(1, len(signs)) if signs[k] != signs[k - 1])
    lows = [bars[i]["low"] for i in range(len(bars) - 10, len(bars)) if bars[i]["low"] is not None]
    higher_low = len(lows) >= 8 and min(lows[-5:]) > min(lows[:5])
    return alt >= 4 and higher_low


def _pat_high_turnover_wash(bars, ctx):          # P10 高换手洗盘(筹码峰不发散)
    vr, ma20, closes, chip = ctx["vol_ratio"], ctx["ma20"], ctx["closes"], ctx["chip"]
    return (vr is not None and vr > 1.5 and ma20 and closes[-1] and closes[-1] > ma20
            and chip is not None and chip["concentration"] >= 0.45)


# --- 突破 🟠hold ---
def _pat_breakout(bars, ctx):                    # P11 放量突破启动(右进左出)
    return bool(ctx["triggered"])


# --- 拉升 🟠hold ---
def _pat_consecutive_limit(bars, ctx):           # P12 连板拉升
    return ctx["streak"] >= 2


def _pat_first_board(bars, ctx):                 # P13 首板卡位
    if len(bars) < 22:
        return False
    limit = _limit_pct(ctx["code"]) - 0.3
    today = bars[-1]["chg"]
    if today is None or today < limit or ctx["streak"] != 1:
        return False
    for i in range(len(bars) - 21, len(bars) - 1):
        if bars[i]["chg"] is not None and bars[i]["chg"] >= limit:
            return False
    t = bars[-1]["turnover"]
    return t is not None and 10.0 <= t <= 45.0


# --- 出货 🔴sell ---
def _pat_high_vol_stall(bars, ctx):              # P14 高位放量滞涨
    pos, vr, closes = ctx["pos"], ctx["vol_ratio"], ctx["closes"]
    if pos is None or pos < 0.85 or vr is None or vr <= 1.5:
        return False
    r5 = _ret_k(closes, 5)
    up = _kl_upper(bars, len(bars) - 1)
    return r5 is not None and r5 <= 0.02 and up is not None and up > 0.02


def _pat_vol_price_div(bars, ctx):               # P15 量价背离
    closes, vr = ctx["closes"], ctx["vol_ratio"]
    win = [c for c in closes[-60:] if c]
    if len(win) < 30 or vr is None or closes[-1] is None:
        return False
    if closes[-1] < max(win):
        return False
    r5 = _ret_k(closes, 5)
    return vr < 1.0 or (vr > 1.8 and r5 is not None and r5 < 0.01)


def _pat_bearish_max_vol(bars, ctx):             # P16 阴天量(近期最大量收阴)
    pos, vol = ctx["pos"], ctx["vol"]
    if pos is None or pos < 0.80 or len(bars) < 40:
        return False
    mx = max([v for v in vol[-40:] if v] or [0])
    for i in range(len(bars) - 5, len(bars)):
        if vol[i] and vol[i] >= mx and bars[i]["chg"] is not None and bars[i]["chg"] < 0:
            return True
    return False


def _pat_inverted_v(bars, ctx):                  # P17 倒V反转
    pos, closes = ctx["pos"], ctx["closes"]
    if pos is None or pos < 0.80 or len(bars) < 16:
        return False
    recent = [c for c in closes[-10:] if c]
    base = [c for c in closes[-16:-10] if c]
    if len(recent) < 8 or not base:
        return False
    peak = max(recent)
    return (peak / min(base) - 1 > 0.15) and (closes[-1] / peak - 1 <= -0.08)


def _pat_bearish_engulf(bars, ctx):              # P18 顶部大阴包阳
    if ctx["pos"] is None or ctx["pos"] < 0.80 or len(bars) < 2:
        return False
    o1, c1 = bars[-2]["open"], bars[-2]["close"]
    o0, c0 = bars[-1]["open"], bars[-1]["close"]
    if None in (o1, c1, o0, c0):
        return False
    return c1 > o1 and c0 < o0 and o0 >= c1 and c0 <= o1


def _pat_dump_bigbear(bars, ctx):                # P19 灌压出货(巨量大阴)
    pos, vr = ctx["pos"], ctx["vol_ratio"]
    if pos is None or pos < 0.70 or vr is None or vr <= 1.8:
        return False
    body = _kl_body(bars, len(bars) - 1)
    o, c, h, l = bars[-1]["open"], bars[-1]["close"], bars[-1]["high"], bars[-1]["low"]
    if None in (o, c, h, l) or h <= l:
        return False
    return body is not None and body > 0.06 and c < o and (c - l) / (h - l) < 0.25


def _pat_ma_breakdown(bars, ctx):                # P20 均线放量破位
    closes, ma20, vr = ctx["closes"], ctx["ma20"], ctx["vol_ratio"]
    if ma20 is None or len(closes) < 25 or vr is None:
        return False
    c = closes[-1]
    if not c or c >= ma20:
        return False
    prior, ma20_prior = closes[-5], _ma_last(closes[:-4], 20)
    return bool(prior and ma20_prior and prior > ma20_prior and vr > 1.2)


# (code, 名称, 阶段, 信号方向, 匹配函数)
PATTERNS: List[Tuple[str, str, str, str, Any]] = [
    ("P1", "低位横盘磨人", "吸筹", "buy", _pat_low_consolidation),
    ("P2", "低位影线吸筹", "吸筹", "buy", _pat_low_shadows),
    ("P3", "缩量阴线打压吸筹", "吸筹", "buy", _pat_shakedown_absorb),
    ("P4", "量增价稳吸收", "吸筹", "buy", _pat_absorption),
    ("P5", "底部形态构筑", "吸筹", "buy", _pat_bottom_formation),
    ("P6", "试盘长上影", "试盘", "hold", _pat_test_upper_shadow),
    ("P7", "底部异动放量", "试盘", "hold", _pat_bottom_spike),
    ("P8", "缩量回踩洗盘", "洗盘", "buy", _pat_pullback_shakeout),
    ("P9", "边拉边洗", "洗盘", "buy", _pat_climb_wash),
    ("P10", "高换手洗盘", "洗盘", "buy", _pat_high_turnover_wash),
    ("P11", "放量突破启动", "突破", "hold", _pat_breakout),
    ("P12", "连板拉升", "拉升", "hold", _pat_consecutive_limit),
    ("P13", "首板卡位", "拉升", "hold", _pat_first_board),
    ("P14", "高位放量滞涨", "出货", "sell", _pat_high_vol_stall),
    ("P15", "量价背离", "出货", "sell", _pat_vol_price_div),
    ("P16", "阴天量", "出货", "sell", _pat_bearish_max_vol),
    ("P17", "倒V反转", "出货", "sell", _pat_inverted_v),
    ("P18", "顶部大阴包阳", "出货", "sell", _pat_bearish_engulf),
    ("P19", "灌压巨量大阴", "出货", "sell", _pat_dump_bigbear),
    ("P20", "均线放量破位", "出货", "sell", _pat_ma_breakdown),
]


def match_patterns(code: str, bars: List[Dict[str, Any]],
                   ctx: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """对一段日线窗口匹配全部形态，返回命中列表（PIT 安全）。"""
    if len(bars) < MIN_BARS:
        return []
    ctx = ctx or _build_pattern_context(code, bars)
    fired: List[Dict[str, str]] = []
    for pcode, name, phase, signal, fn in PATTERNS:
        try:
            if fn(bars, ctx):
                fired.append({"code": pcode, "name": name, "phase": phase, "signal": signal})
        except Exception:
            continue
    return fired


def _pattern_phase(fired: List[Dict[str, str]], score: Optional[float] = None) -> str:
    """命中形态汇总成一个主导阶段标签（优先级：出货 > 突破 > 买入区 > 拉升 > 试盘）。

    突破=放量右进买点，最 actionable，仅次于出货风控示警、优先于被动的吸筹/洗盘；
    买入区按类别细分吸筹 / 洗盘（两类都中则合并标注）；剩余 hold 区拉升中 > 试盘；
    无任何形态命中：吸筹分≥SUSPECT_ACCUM_SCORE → 疑似吸筹(待确认)，否则 → 空仓观望(场外不参与)。
    """
    sigs = {p["signal"] for p in fired}
    cats = {p["phase"] for p in fired}
    if "sell" in sigs:
        return "出货预警🔴"
    if "突破" in cats:
        return "▲突破🟠"
    if "buy" in sigs:
        buy_cats = {p["phase"] for p in fired if p["signal"] == "buy"}
        if {"吸筹", "洗盘"} <= buy_cats:
            return "吸筹+洗盘🟡"
        if "洗盘" in buy_cats:
            return "洗盘🟡"
        return "吸筹🟢"
    if "hold" in sigs:                # 此时仅剩 拉升 / 试盘
        return "拉升中🟠" if "拉升" in cats else "试盘🟡"
    if score is not None and score >= SUSPECT_ACCUM_SCORE:
        return "疑似吸筹(待确认)🟢"
    return "空仓观望⚪"


def _score_bars(code: str, bars: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """对一段日线窗口算吸筹分（位置 + 努力结果背离 + 收盘买压 + 低位筹码集中）。数据不足返回 None。"""
    if len(bars) < MIN_BARS:
        return None
    vol, vol_measure = _volume_series(bars)
    s_pos, close_pctile = _score_position(bars)            # 位置
    s_div, drift = _score_divergence(bars, vol)            # ① 努力与结果背离
    s_cmf, cmf = _score_cmf(bars, vol)                     # ② 收盘买压
    s_chip, chip = _score_chip(bars)                       # ③ 低位筹码集中
    _, vol_ratio = _score_volume_ratio(vol)               # 仅供展示
    sealed, streak = _sealed_and_streak(bars, code)
    triggered, trigger = _breakout_trigger(bars, vol)     # 右进左出触发器

    raw = (WEIGHTS["position"] * (s_pos or 0.0)
           + WEIGHTS["divergence"] * (s_div or 0.0)
           + WEIGHTS["cmf"] * (s_cmf or 0.0)
           + WEIGHTS["chip"] * (s_chip or 0.0))
    penalty = min(SEALED_PENALTY_CAP, SEALED_PENALTY_PER * sealed)
    score = round(raw * (1.0 - penalty), 1)
    return {
        "ambush_score": score,
        "raw": raw,
        "sealed": sealed,
        "streak": streak,
        "triggered": triggered,
        "signals": {
            "vol_measure": vol_measure,
            "vol_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
            "close_pctile": round(close_pctile, 2) if close_pctile is not None else None,
            "drift_20d": round(drift, 3) if drift is not None else None,
            "cmf": round(cmf, 3) if cmf is not None else None,
            "chip_concentration": round(chip["concentration"], 2) if chip else None,
            "chip_winner": round(chip["winner"], 2) if chip else None,
            "chip_peak_price": round(chip["peak_price"], 2) if chip else None,
            "chip_peak_pctile": round(chip["peak_pctile"], 2) if chip else None,
            "chip_price_to_peak": round(chip["price_to_peak"], 3) if chip and chip["price_to_peak"] is not None else None,
            "triggered": triggered,
            "trigger": trigger,
            "sealed_recent": sealed,
            "limit_streak": streak,
            "latest_turnover": bars[-1]["turnover"],
        },
        "sub_scores": {
            "position": round(s_pos, 1) if s_pos is not None else None,
            "divergence": round(s_div, 1) if s_div is not None else None,
            "cmf": round(s_cmf, 1) if s_cmf is not None else None,
            "chip": round(s_chip, 1) if s_chip is not None else None,
            "chip_concentration": round(chip["sub_concentration"], 1) if chip else None,
            "chip_peak_low": round(chip["sub_peak_low"], 1) if chip else None,
            "chip_price_near_peak": round(chip["sub_price_near_peak"], 1) if chip else None,
            "chip_winner_mid_low": round(chip["sub_winner_mid_low"], 1) if chip else None,
            "sealed_penalty": round(penalty, 2),
        },
    }


def _state_label(raw: Optional[float], sealed: int, streak: int, triggered: bool = False) -> str:
    if sealed > 0 or streak >= 4:
        return "已启动(封板/连板,非吸筹)"
    if raw is None:
        return "数据不足"
    if triggered and raw >= 40:
        return "放量突破(右进)"      # 事件标记；verify 显示截面不占优，慎追
    if raw >= 65:
        return "疑似吸筹"
    if raw >= 40:
        return "温和量增"
    return "平淡"


def score_candidate(conn: sqlite3.Connection, cand: Dict[str, Any]) -> Dict[str, Any]:
    """给单只龙头算当下潜伏分 + 游资形态匹配，返回带子分/原始信号/形态的明细行。"""
    bars = _recent_bars(conn, cand["code"])
    out = dict(cand)
    res = _score_bars(cand["code"], bars)
    if res is None:
        out.update({"ambush_score": None, "score_status": "INSUFFICIENT_DATA",
                    "state": "数据不足", "last_date": bars[-1]["date"] if bars else None})
        return out
    fired = match_patterns(cand["code"], bars)
    out.update({
        "ambush_score": res["ambush_score"],
        "score_status": "OK",
        "triggered": res["triggered"],
        "state": _state_label(res["raw"], res["sealed"], res["streak"], res["triggered"]),
        "last_date": bars[-1]["date"],
        "patterns": [p["code"] for p in fired],
        "pattern_detail": fired,
        "pattern_phase": _pattern_phase(fired, res["ambush_score"]),
        "signals": res["signals"],
        "sub_scores": res["sub_scores"],
    })
    return out


# ── 输出 ──────────────────────────────────────────────────────

def base_payload(mode: str, candidate_count: int) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": mode,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "sw3_member.is_leader (细分行业龙头池)",
        "candidate_count": candidate_count,
    }


def write_payload(path: Path, payload: Dict[str, Any]) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


# ── ambush ────────────────────────────────────────────────────

THEME_STALE_DAYS = 7        # 题材数据超过该天数视为偏旧（热度时效性强）


def _load_theme_map() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """读取 stock_theme_candidates.py 落盘的题材映射。

    返回 (映射, 元信息)：
      映射 = {code: {theme: 拟合二级行业名, theme_code, heat_pctile: 行业热度百分位}}；
      元信息 = {available, generated_at, age_days, stale}。
    热度百分位由题材热度排名换算（rank 越靠前越接近 100，rank=1→100、rank=N→0）。
    文件缺失/解析失败则映射为空、available=False，雷达照常出表（题材列回退到静态细分行业）。
    """
    meta: Dict[str, Any] = {"available": False, "generated_at": None, "age_days": None, "stale": False}
    try:
        data = json.loads(THEME_CANDIDATES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, meta
    meta["available"] = True
    gen = data.get("generated_at")
    meta["generated_at"] = gen
    if gen:
        try:
            age = (datetime.now() - datetime.strptime(gen, "%Y-%m-%d %H:%M:%S")).days
            meta["age_days"] = age
            meta["stale"] = age >= THEME_STALE_DAYS
        except ValueError:
            pass
    rankings = data.get("theme_rankings") or []
    n = len(rankings)
    pctile_by_code: Dict[str, float] = {}
    for row in rankings:
        rank, plate_code = row.get("rank"), row.get("plate_code")
        if rank is None or plate_code is None:
            continue
        pctile_by_code[str(plate_code)] = 100.0 if n <= 1 else (n - rank) / (n - 1) * 100.0
    out: Dict[str, Dict[str, Any]] = {}
    for st in data.get("stock_themes") or []:
        code = stock_storage._normalize_code(st.get("code"))
        if not code:
            continue
        theme_code = st.get("tracking_theme_code")
        out[code] = {
            "theme": st.get("tracking_theme") or "",
            "theme_code": theme_code,
            "heat_pctile": pctile_by_code.get(str(theme_code)),
        }
    return out, meta


def run_ambush() -> Dict[str, Any]:
    conn = stock_storage.connect(DB_FILE)
    try:
        candidates = load_leader_candidates(conn)
        scored = [score_candidate(conn, cand) for cand in candidates]
    finally:
        conn.close()

    theme_map, theme_meta = _load_theme_map()
    for r in scored:
        info = theme_map.get(r["code"])
        if info:
            r["tracking_theme"] = info["theme"]
            r["tracking_theme_code"] = info["theme_code"]
            r["theme_heat_pctile"] = info["heat_pctile"]

    ranked = [r for r in scored if r.get("ambush_score") is not None]
    ranked.sort(key=lambda r: r["ambush_score"], reverse=True)
    skipped = len(scored) - len(ranked)

    payload = base_payload("ambush", len(candidates))
    payload.update({
        "status": "ok" if candidates else "empty",
        "description": "细分龙头吸筹分（价格中低位 + 努力结果背离 + 收盘买压CMF + 低位筹码集中 − 封板/连板）。",
        "params": {
            "weights": WEIGHTS, "lookback": LOOKBACK,
            "cmf_full": CMF_FULL, "chip_band": CHIP_BAND, "pos_low": POS_LOW, "pos_high": POS_HIGH,
        },
        "scored_count": len(ranked),
        "skipped_count": skipped,
        "triggered_count": sum(1 for r in ranked if r.get("triggered")),
        "phase_counts": dict(Counter(r.get("pattern_phase") for r in ranked)),
        "max_market_cap_yi": MAX_MARKET_CAP_YI,
        "theme_source": theme_meta,
        "stocks": ranked,
    })
    if not candidates:
        payload["notes"] = ["候选池为空：先运行 python stock_crawl_segment_leaders.py crawl 选龙头并回写 is_leader。"]
    write_payload(AMBUSH_RESULT_FILE, payload)
    _print_ambush_summary(payload)
    return payload


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{value:g}"


def _disp_width(text: str) -> int:
    """终端显示宽度：东亚全角/宽字符（含 emoji）按 2 计，其余按 1。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in str(text))


def _ljust(text: Any, width: int) -> str:
    """按显示宽度左对齐（中文每字算 2 宽，避免列错位）。"""
    text = str(text)
    return text + " " * max(0, width - _disp_width(text))


def _rjust(text: Any, width: int) -> str:
    """按显示宽度右对齐。"""
    text = str(text)
    return " " * max(0, width - _disp_width(text)) + text


def _theme_cell(s: Dict[str, Any]) -> str:
    """跟踪二级行业 + 行业热度百分位（来自 stock_theme_candidates.py）。

    有题材映射时显示「行业名 热度NN%」；缺映射时回退到静态细分行业(SW3)。
    """
    theme = s.get("tracking_theme")
    if not theme:
        return s.get("segment_name") or ""
    pctile = s.get("theme_heat_pctile")
    return theme if pctile is None else f"{theme} 热度{pctile:.0f}%"


def _theme_freshness_note(meta: Dict[str, Any]) -> str:
    """题材热度数据来源 + 新鲜度提示行。"""
    if not meta.get("available"):
        return ("题材热度: ⚠️ 缺 theme_candidates.json，"
                "「跟踪二级行业」列回退静态细分行业 → 先跑 python stock_theme_candidates.py")
    gen = meta.get("generated_at") or "?"
    age = meta.get("age_days")
    if age is None:
        return f"题材热度: 来自 stock_theme_candidates.py（生成于 {gen}）"
    freshness = f"{age} 天前生成"
    if meta.get("stale"):
        return (f"题材热度: ⚠️ 数据偏旧（{freshness}，≥{THEME_STALE_DAYS}天）"
                f" → 重跑 python stock_theme_candidates.py 刷新热度")
    return f"题材热度: 来自 stock_theme_candidates.py（{freshness}，{gen}）"


def _print_ambush_summary(payload: Dict[str, Any]) -> None:
    stocks = payload.get("stocks", [])
    print("=" * 112)
    print("  主力资金雷达 · 吸筹分 + 游资形态 (ambush)")
    counts = payload.get("phase_counts") or {}
    dist = " · ".join(f"{ph}{counts[ph]}" for ph in PHASE_ORDER if counts.get(ph))
    print(f"  生成时间: {payload['generated_at']} · 候选(≤{payload.get('max_market_cap_yi', '∞')}亿小中盘龙头): "
          f"{payload['candidate_count']} · 已打分: {payload.get('scored_count', 0)}")
    print(f"  阶段分布(游资操作顺序): {dist}")
    print(f"  落盘: {display_path(AMBUSH_RESULT_FILE)}")
    print(f"  {_theme_freshness_note(payload.get('theme_source') or {})}")
    print("-" * 112)
    if not stocks:
        for note in payload.get("notes", ["（无候选）"]):
            print(f"  {note}")
        print("=" * 112)
        return
    print(f"  {'#':>2} {_ljust('代码', 7)}{_ljust('名称', 9)}{_rjust('吸筹分', 6)}  "
          f"{_rjust('量比', 5)} {_rjust('价分位', 6)} {'CMF':>6} {_rjust('筹码集中', 7)}"
          f" {_rjust('连板', 4)}  {_ljust('命中形态', 16)} 阶段 / 跟踪二级行业(热度%)")
    for i, s in enumerate(stocks[:30], 1):
        sig = s.get("signals", {})
        name = (s.get("name") or "")[:8]
        pats = ",".join(s.get("patterns") or []) or "-"
        phase = s.get("pattern_phase") or s.get("state", "")
        print(f"  {i:>2} {s['code']:<7}{_ljust(name, 9)}{s['ambush_score']:>6.1f}  "
              f"{_fmt(sig.get('vol_ratio')):>5} {_fmt(sig.get('close_pctile')):>6} "
              f"{_fmt(sig.get('cmf')):>6} {_fmt(sig.get('chip_concentration')):>7} "
              f"{sig.get('limit_streak', 0):>4}  {pats:<16} {phase} / {_theme_cell(s)}")
    if len(stocks) > 30:
        print(f"  ... 其余 {len(stocks) - 30} 只见 {display_path(AMBUSH_RESULT_FILE)}")
    print("=" * 112)


# ── verify：潜伏分后验回测 ────────────────────────────────────

def _collect_verify_samples(conn: sqlite3.Connection, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对每只龙头滑动取历史截面，PIT 重算潜伏分并配对未来前向收益。

    返回 {samples, dates, codes}。samples 每项 = {date, code, score, rets:{h:ret}}。
    PIT：打分只用截止 as-of 当日的 LOOKBACK 根 bar；前向收益用其后第 h 根 bar 的收盘价。
    """
    max_h = max(VERIFY_HORIZONS)
    series: Dict[str, Tuple[List[Dict[str, Any]], Dict[str, int]]] = {}
    for cand in candidates:
        bars = _all_bars(conn, cand["code"])
        if len(bars) < LOOKBACK + max_h + 1:
            continue
        series[cand["code"]] = (bars, {b["date"]: i for i, b in enumerate(bars)})

    if not series:
        return {"samples": [], "dates": [], "codes": []}

    all_dates = sorted({d for bars, _ in series.values() for d in (b["date"] for b in bars)})
    usable = all_dates[:-max_h]                       # 末段没有前向数据，剔除
    window = usable[-VERIFY_WINDOW_DAYS:]
    as_of_dates = window[::VERIFY_STEP]

    samples: List[Dict[str, Any]] = []
    used_dates: set = set()
    for d in as_of_dates:
        for code, (bars, idx_map) in series.items():
            i = idx_map.get(d)
            if i is None or i < LOOKBACK - 1 or i + max_h >= len(bars):
                continue
            res = _score_bars(code, bars[i - LOOKBACK + 1:i + 1])
            if res is None:
                continue
            close_i = bars[i]["close"]
            if not close_i:
                continue
            rets: Dict[int, float] = {}
            ok = True
            for h in VERIFY_HORIZONS:
                cf = bars[i + h]["close"]
                if not cf:
                    ok = False
                    break
                rets[h] = cf / close_i - 1.0
            if not ok:
                continue
            samples.append({"date": d, "code": code, "score": res["ambush_score"],
                            "triggered": res["triggered"], "rets": rets})
            used_dates.add(d)
    return {"samples": samples, "dates": sorted(used_dates), "codes": list(series.keys())}


def _aggregate_verify(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分位前向收益（pooled）+ 截面 Rank IC + 多空分位价差（按 as-of 日聚合再平均）。"""
    by_horizon: Dict[str, Any] = {}
    by_date: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        by_date.setdefault(s["date"], []).append(s)

    for h in VERIFY_HORIZONS:
        # pooled：全样本按潜伏分五等分，看各桶平均前向收益（绝对收益含大盘 beta）
        ordered = sorted(samples, key=lambda s: s["score"])
        n = len(ordered)
        buckets = []
        for q in range(VERIFY_BUCKETS):
            grp = ordered[q * n // VERIFY_BUCKETS:(q + 1) * n // VERIFY_BUCKETS]
            rets = [g["rets"][h] for g in grp]
            buckets.append({
                "mean_ret": round(_mean(rets), 4) if rets else None,
                "n": len(grp),
                "score_lo": round(grp[0]["score"], 1) if grp else None,
                "score_hi": round(grp[-1]["score"], 1) if grp else None,
            })
        # 截面：每个 as-of 日单独算 Rank IC 与 top-bottom 五分位价差，再跨日平均（剔除大盘 beta）
        ics: List[float] = []
        spreads: List[float] = []
        for grp in by_date.values():
            if len(grp) < VERIFY_MIN_NAMES:
                continue
            scores = [g["score"] for g in grp]
            rets = [g["rets"][h] for g in grp]
            ic = _spearman(scores, rets)
            if ic is not None:
                ics.append(ic)
            sg = sorted(grp, key=lambda g: g["score"])
            k = max(1, len(sg) // VERIFY_BUCKETS)
            top = _mean([g["rets"][h] for g in sg[-k:]])
            bot = _mean([g["rets"][h] for g in sg[:k]])
            if top is not None and bot is not None:
                spreads.append(top - bot)

        ic_mean = _mean(ics)
        ic_std = (sum((x - ic_mean) ** 2 for x in ics) / len(ics)) ** 0.5 if len(ics) > 1 else None
        t_stat = (ic_mean / ic_std * math.sqrt(len(ics))) if ic_mean is not None and ic_std else None
        pooled_hi, pooled_lo = buckets[-1]["mean_ret"], buckets[0]["mean_ret"]
        by_horizon[str(h)] = {
            "quantile_returns": buckets,
            "pooled_q5_minus_q1": round(pooled_hi - pooled_lo, 4) if pooled_hi is not None and pooled_lo is not None else None,
            "long_short_spread": round(_mean(spreads), 4) if spreads else None,
            "ic_mean": round(ic_mean, 4) if ic_mean is not None else None,
            "ic_std": round(ic_std, 4) if ic_std is not None else None,
            "ic_t_stat": round(t_stat, 2) if t_stat is not None else None,
            "ic_hit_rate": round(sum(1 for x in ics if x > 0) / len(ics), 3) if ics else None,
            "n_sections": len(ics),
            "trigger": _trigger_study(by_date, h),
        }
    return by_horizon


def _trigger_study(by_date: Dict[str, List[Dict[str, Any]]], h: int) -> Dict[str, Any]:
    """右进左出事件研究：触发样本 vs 全体的前向收益。

    excess = 每个 as-of 日「触发样本均收益 − 该日全样本均收益」(剔除大盘 beta)，跨日平均 + t 值。
    这才是右进左出的真实择时力——触发那一刻买，能否跑赢同日普通龙头。
    """
    excess: List[float] = []
    trig_rets: List[float] = []
    base_rets: List[float] = []
    for grp in by_date.values():
        fired = [g["rets"][h] for g in grp if g.get("triggered")]
        if not fired:
            continue
        section_mean = _mean([g["rets"][h] for g in grp])
        trig_mean = _mean(fired)
        excess.append(trig_mean - section_mean)
        trig_rets.extend(fired)
        base_rets.extend(g["rets"][h] for g in grp)
    if not excess:
        return {"n_triggered": 0, "n_sections": 0}
    em = _mean(excess)
    es = (sum((x - em) ** 2 for x in excess) / len(excess)) ** 0.5 if len(excess) > 1 else None
    t = (em / es * math.sqrt(len(excess))) if es else None
    return {
        "n_triggered": len(trig_rets),
        "n_sections": len(excess),
        "triggered_mean_ret": round(_mean(trig_rets), 4),
        "all_mean_ret": round(_mean(base_rets), 4),
        "excess_mean": round(em, 4),
        "excess_t_stat": round(t, 2) if t is not None else None,
        "win_rate": round(sum(1 for x in trig_rets if x > 0) / len(trig_rets), 3),
    }


def _verdict(by_horizon: Dict[str, Any]) -> str:
    mid = by_horizon.get(str(VERIFY_HORIZONS[len(VERIFY_HORIZONS) // 2]), {})
    ic, t = mid.get("ic_mean"), mid.get("ic_t_stat")
    tg = mid.get("trigger", {})
    ex, et = tg.get("excess_mean"), tg.get("excess_t_stat")
    parts = []
    # 触发器(右进左出)是主信号：触发那刻买能否跑赢同日龙头
    if ex is not None and et is not None:
        if ex > 0 and et >= 2:
            parts.append(f"右进左出触发器有效：触发后超额 {_pct(ex).strip()} (t={et}, 显著)")
        elif ex > 0 and et >= 1:
            parts.append(f"触发器方向为正但偏弱：超额 {_pct(ex).strip()} (t={et})")
        elif ex <= 0:
            parts.append(f"触发器无超额：{_pct(ex).strip()} (t={et})")
    if ic is None:
        return "; ".join(parts) or "样本不足，无法判定"
    if ic >= 0.03 and (t or 0) >= 2:
        parts.append(f"吸筹分截面 IC 显著正 ({ic}, t={t})")
    elif ic > 0.005:
        parts.append(f"吸筹分截面 IC 微正 ({ic}, t={t})")
    elif ic <= -0.01:
        parts.append(f"吸筹分截面 IC 偏负 ({ic}, t={t})")
    else:
        parts.append(f"吸筹分截面 IC 中性 ({ic}, t={t})")
    return "; ".join(parts)


def run_verify() -> Dict[str, Any]:
    conn = stock_storage.connect(DB_FILE)
    try:
        candidates = load_leader_candidates(conn)
        collected = _collect_verify_samples(conn, candidates)
    finally:
        conn.close()

    samples = collected["samples"]
    by_horizon = _aggregate_verify(samples) if samples else {}
    payload = base_payload("verify", len(candidates))
    payload.update({
        "status": "ok" if samples else "empty",
        "description": "潜伏分后验回测：PIT 重算潜伏分 vs 未来前向收益（分位单调性 / 截面RankIC / 多空价差）。",
        "params": {
            "horizons": list(VERIFY_HORIZONS), "step": VERIFY_STEP,
            "window_days": VERIFY_WINDOW_DAYS, "buckets": VERIFY_BUCKETS,
            "min_names_per_section": VERIFY_MIN_NAMES,
        },
        "section_count": len(collected["dates"]),
        "sample_count": len(samples),
        "scored_codes": len(collected["codes"]),
        "date_range": [collected["dates"][0], collected["dates"][-1]] if collected["dates"] else None,
        "horizons": by_horizon,
        "verdict": _verdict(by_horizon) if samples else "候选池为空或历史不足，无法回测。",
    })
    if not samples:
        payload["notes"] = ["无可回测样本：先确保 sw3_member.is_leader 有龙头、且其历史日线足够长。"]
    write_payload(VERIFY_RESULT_FILE, payload)
    _print_verify_summary(payload)
    return payload


def _pct(value: Any) -> str:
    return "  -  " if value is None else f"{value * 100:+5.2f}%"


def _print_verify_summary(payload: Dict[str, Any]) -> None:
    print("=" * 92)
    print("  主力资金雷达 · 潜伏分后验回测 (verify)")
    rng = payload.get("date_range")
    print(f"  生成时间: {payload['generated_at']} · 候选龙头: {payload['candidate_count']}"
          f" · 截面: {payload.get('section_count', 0)} · 样本: {payload.get('sample_count', 0)}"
          + (f" · 区间: {rng[0]}~{rng[1]}" if rng else ""))
    print(f"  落盘: {display_path(VERIFY_RESULT_FILE)}")
    print("-" * 92)
    horizons = payload.get("horizons", {})
    if not horizons:
        for note in payload.get("notes", ["（无样本）"]):
            print(f"  {note}")
        print("=" * 92)
        return
    print("  分位前向收益（按潜伏分五等分，Q1低→Q5高；绝对收益含大盘 beta，pooled）:")
    print(f"  {'持有期':>6} {'Q1低':>8} {'Q2':>8} {'Q3':>8} {'Q4':>8} {'Q5高':>8} | {'Q5-Q1':>8}")
    for h in VERIFY_HORIZONS:
        hz = horizons.get(str(h), {})
        b = hz.get("quantile_returns", [])
        cells = " ".join(f"{_pct(b[q]['mean_ret']) if q < len(b) else '  -  ':>8}" for q in range(VERIFY_BUCKETS))
        print(f"  {str(h)+'日':>6} {cells} | {_pct(hz.get('pooled_q5_minus_q1')):>8}")
    print("-" * 92)
    print("  截面口径（每个 as-of 日内部排序，跨日平均；天然剔除大盘 beta）— 这才是真实选股力:")
    print(f"  {'持有期':>6} {'RankIC':>9} {'IC_t':>7} {'胜率':>7} {'多空价差':>9} {'截面数':>7}")
    for h in VERIFY_HORIZONS:
        hz = horizons.get(str(h), {})
        hit = hz.get("ic_hit_rate")
        print(f"  {str(h)+'日':>6} {_fmt(hz.get('ic_mean')):>9} {_fmt(hz.get('ic_t_stat')):>7} "
              f"{(f'{hit*100:.0f}%' if hit is not None else '-'):>7} {_pct(hz.get('long_short_spread')):>9} "
              f"{hz.get('n_sections', 0):>7}")
    print("-" * 92)
    tg0 = horizons.get(str(VERIFY_HORIZONS[0]), {}).get("trigger", {})
    print(f"  右进左出触发器事件研究（触发=吸筹中放量突破近端高点；触发样本数 {tg0.get('n_triggered', 0)}）:")
    print(f"  {'持有期':>6} {'触发后均收益':>11} {'同日全体':>9} {'超额':>9} {'超额_t':>7} {'胜率':>7}")
    for h in VERIFY_HORIZONS:
        tg = horizons.get(str(h), {}).get("trigger", {})
        win = tg.get("win_rate")
        print(f"  {str(h)+'日':>6} {_pct(tg.get('triggered_mean_ret')):>11} {_pct(tg.get('all_mean_ret')):>9} "
              f"{_pct(tg.get('excess_mean')):>9} {_fmt(tg.get('excess_t_stat')):>7} "
              f"{(f'{win*100:.0f}%' if win is not None else '-'):>7}")
    print("-" * 92)
    print(f"  结论: {payload.get('verdict')}")
    print("=" * 92)


# ── patterns：形态预测力后验（PIT 事件研究）─────────────────────

def _collect_pattern_samples(conn: sqlite3.Connection, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对每只龙头滑动取历史截面，PIT 匹配全部形态并配对未来前向收益。

    samples 每项 = {date, fired:set(形态code), rets:{h:ret}}。PIT 安全：形态只用截止当日窗口。
    """
    max_h = max(VERIFY_HORIZONS)
    series: Dict[str, Tuple[List[Dict[str, Any]], Dict[str, int]]] = {}
    for cand in candidates:
        bars = _all_bars(conn, cand["code"])
        if len(bars) < LOOKBACK + max_h + 1:
            continue
        series[cand["code"]] = (bars, {b["date"]: i for i, b in enumerate(bars)})
    if not series:
        return {"samples": [], "dates": [], "codes": []}

    all_dates = sorted({d for bars, _ in series.values() for d in (b["date"] for b in bars)})
    as_of_dates = all_dates[:-max_h][-VERIFY_WINDOW_DAYS:][::VERIFY_STEP]
    samples: List[Dict[str, Any]] = []
    used_dates: set = set()
    for d in as_of_dates:
        for code, (bars, idx_map) in series.items():
            i = idx_map.get(d)
            if i is None or i < LOOKBACK - 1 or i + max_h >= len(bars):
                continue
            window = bars[i - LOOKBACK + 1:i + 1]
            fired = match_patterns(code, window)
            close_i = bars[i]["close"]
            if not close_i:
                continue
            rets: Dict[int, float] = {}
            ok = True
            for h in VERIFY_HORIZONS:
                cf = bars[i + h]["close"]
                if not cf:
                    ok = False
                    break
                rets[h] = cf / close_i - 1.0
            if not ok:
                continue
            samples.append({"date": d, "fired": {p["code"] for p in fired}, "rets": rets})
            used_dates.add(d)
    return {"samples": samples, "dates": sorted(used_dates), "codes": list(series.keys())}


def _pattern_event_study(by_date: Dict[str, List[Dict[str, Any]]], pcode: str, h: int) -> Dict[str, Any]:
    """单形态事件研究：命中样本 vs 同日全体的前向收益超额（剔除大盘 beta），跨日平均 + t。"""
    excess: List[float] = []
    hit_rets: List[float] = []
    for grp in by_date.values():
        hit = [g["rets"][h] for g in grp if pcode in g["fired"]]
        if not hit:
            continue
        section_mean = _mean([g["rets"][h] for g in grp])
        excess.append(_mean(hit) - section_mean)
        hit_rets.extend(hit)
    if not hit_rets:
        return {"n_hits": 0, "n_sections": 0, "excess_mean": None, "excess_t_stat": None, "win_rate": None}
    em = _mean(excess)
    es = (sum((x - em) ** 2 for x in excess) / len(excess)) ** 0.5 if len(excess) > 1 else None
    t = (em / es * math.sqrt(len(excess))) if es else None
    return {
        "n_hits": len(hit_rets),
        "n_sections": len(excess),
        "excess_mean": round(em, 4),
        "excess_t_stat": round(t, 2) if t is not None else None,
        "win_rate": round(sum(1 for x in hit_rets if x > 0) / len(hit_rets), 3),
    }


def run_patterns() -> Dict[str, Any]:
    conn = stock_storage.connect(DB_FILE)
    try:
        candidates = load_leader_candidates(conn)
        collected = _collect_pattern_samples(conn, candidates)
    finally:
        conn.close()

    samples = collected["samples"]
    by_date: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        by_date.setdefault(s["date"], []).append(s)

    results = []
    for pcode, name, phase, signal, _fn in PATTERNS:
        row = {"code": pcode, "name": name, "phase": phase, "signal": signal,
               "horizons": {str(h): _pattern_event_study(by_date, pcode, h) for h in VERIFY_HORIZONS}}
        results.append(row)

    payload = base_payload("patterns", len(candidates))
    payload.update({
        "status": "ok" if samples else "empty",
        "description": "游资形态预测力后验：每个形态命中后，未来 N 日相对同日全体的超额收益（剔除大盘 beta）。",
        "params": {"horizons": list(VERIFY_HORIZONS), "step": VERIFY_STEP, "window_days": VERIFY_WINDOW_DAYS},
        "section_count": len(collected["dates"]),
        "sample_count": len(samples),
        "date_range": [collected["dates"][0], collected["dates"][-1]] if collected["dates"] else None,
        "patterns": results,
    })
    if not samples:
        payload["notes"] = ["无可回测样本：先确保 sw3_member.is_leader 有龙头、且历史日线足够长。"]
    write_payload(PATTERNS_RESULT_FILE, payload)
    _print_patterns_summary(payload)
    return payload


def _print_patterns_summary(payload: Dict[str, Any]) -> None:
    print("=" * 100)
    print("  主力资金雷达 · 游资形态预测力后验 (patterns)")
    rng = payload.get("date_range")
    print(f"  生成时间: {payload['generated_at']} · 候选龙头: {payload['candidate_count']}"
          f" · 截面: {payload.get('section_count', 0)} · 样本: {payload.get('sample_count', 0)}"
          + (f" · 区间: {rng[0]}~{rng[1]}" if rng else ""))
    print(f"  落盘: {display_path(PATTERNS_RESULT_FILE)}")
    print("-" * 100)
    results = payload.get("patterns", [])
    if not results:
        for note in payload.get("notes", ["（无样本）"]):
            print(f"  {note}")
        print("=" * 100)
        return
    mid = str(VERIFY_HORIZONS[len(VERIFY_HORIZONS) // 2])
    print(f"  逐形态：命中后相对同日全体的超额收益（buy 看正、sell 看负为有效；按{mid}日超额排序）")
    print(f"  {_ljust('形态', 6)}{_ljust('名称', 17)}{_ljust('阶段', 5)}{_ljust('信号', 5)}{_rjust('命中', 6)}  "
          + "".join(_rjust(f"{h}日超额", 9) for h in VERIFY_HORIZONS) + _rjust(f"  {mid}日_t", 8) + _rjust('胜率', 7))
    rows = sorted(results, key=lambda r: (r["horizons"][mid].get("excess_mean") or 0), reverse=True)
    for r in rows:
        hz = r["horizons"]
        m = hz[mid]
        cells = "".join(f"{_pct(hz[str(h)].get('excess_mean')):>9}" for h in VERIFY_HORIZONS)
        win = m.get("win_rate")
        flag = ""
        if r["signal"] == "buy" and (m.get("excess_t_stat") or 0) >= 1.5:
            flag = " ✅有效"
        elif r["signal"] == "sell" and (m.get("excess_t_stat") or 0) <= -1.5:
            flag = " ✅有效"
        print(f"  {r['code']:<6}{_ljust(r['name'], 17)}{_ljust(r['phase'], 5)}{r['signal']:<5}{m.get('n_hits', 0):>6}  "
              f"{cells}{_fmt(m.get('excess_t_stat')):>8}"
              f"{(f'{win*100:.0f}%' if win is not None else '-'):>7}{flag}")
    print("-" * 100)
    print("  说明：buy 形态超额>0 且 t≥1.5、或 sell 形态超额<0 且 t≤-1.5，视为有预测力（标 ✅）。")
    print("=" * 100)


# ── watch（骨架）──────────────────────────────────────────────

def run_watch() -> Dict[str, Any]:
    """实时交易监控入口（骨架）：取候选龙头占位，后续接盘口/实时行情事件。"""
    conn = stock_storage.connect(DB_FILE)
    try:
        candidates = load_leader_candidates(conn)
    finally:
        conn.close()
    payload = base_payload("watch", len(candidates))
    payload.update({
        "status": "scaffold",
        "description": "实时交易监控框架（待接入实时行情/盘口）。",
        "stocks": [{**c, "watch_status": "TODO", "last_trade_snapshot": None} for c in candidates],
    })
    write_payload(WATCH_STATE_FILE, payload)
    print(f"[watch] scaffold written · candidates={len(candidates)}")
    return payload


# ── CLI ───────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="主力资金雷达")
    parser.add_argument(
        "mode", nargs="?", choices=MODES, default=DEFAULT_MODE,
        help="运行模式：ambush(默认,吸筹分+形态) / watch(实时监控) / "
             "verify(吸筹分回测) / patterns(形态预测力回测)",
    )
    return parser


def run_mode(mode: str) -> Dict[str, Any]:
    if mode == "watch":
        return run_watch()
    if mode == "verify":
        return run_verify()
    if mode == "patterns":
        return run_patterns()
    return run_ambush()


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    return run_mode(args.mode)


if __name__ == "__main__":
    main()
