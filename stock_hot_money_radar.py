"""主力资金雷达。

候选池 = 细分行业龙头（stock_crawl_segment_leaders 选出、回写主库 sw3_member.is_leader）。
潜伏分目标：捕捉「左侧吸筹」——主力在低位悄悄建仓、但价格还没起飞的阶段。

  调研背景：「放量+创新高」经 verify 回测证明是右侧追高(截面 RankIC 显著为负)。吸筹的真正
  指纹是方向性的——参考 Wyckoff/VSA、A股筹码分布、OBV/ADL 三套体系，落地三个判别信号：

ambush 吸筹分（换手率优先、缺失退回成交量）：
  位置  价格中低位(留出上行空间)                    权重 0.20  —— 筹码在低成本区
  背离  努力与结果背离：放量(努力) × 价格横住(结果)    权重 0.25  —— 放量却不涨=有人吸货
  买压  Chaikin Money Flow：收盘持续靠近K线上半部     权重 0.25  —— 收上半部=买方吸收
  筹码  低位筹码单峰密集(换手衰减+成本分布)           权重 0.30  —— 成本趋同=主力控盘
  penalty 一字封板/连板                              打折       —— 已启动,非吸筹

Modes:
  ambush  默认。给候选龙头算当下吸筹分并排名落盘。
  watch   实时交易监控入口（仍为骨架，后续接盘口/实时行情）。
  verify  吸筹分后验回测：把分数按历史时点 PIT 重算（不看未来），统计未来前向收益的
          分位单调性、截面 Rank IC、多空分位价差、右进左出触发器超额，验证信号是否真有预测力。

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

核心结论：龙头池（大盘质量股）截面被**短期反转**主导——追强/突破/放量创高都亏，只有「安静的
吸筹状态」（低位+资金悄悄流入+不动）微弱正向但不显著。量价 hot-money 命题在此池拿不到稳健
alpha。**根因疑为 universe 错配**：游资/主力打的是小盘低流通题材股，不是细分龙头；DB 目前只有
龙头 K 线（956 龙头+41 非龙头），无独立小盘池。下一步若要真正检验命题，需联网爬真·小盘/题材
股池在游资主场重跑 verify。当前代码=cap-filtered 三信号吸筹分，作为已验证的研究基线沉淀。
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import math
import sqlite3
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
WATCH_STATE_FILE = CAPITAL_DIR / "hot_money_watch.json"
VERIFY_RESULT_FILE = CAPITAL_DIR / "hot_money_verify.json"
SCHEMA = "hot_money_radar.v2"
MODES = ("ambush", "watch", "verify")
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
SEALED_AMP = 0.005       # 日内振幅 ≤ 0.5% 视为一字封死板
SEALED_PENALTY_PER = 0.2  # 每个一字封板的打折系数
SEALED_PENALTY_CAP = 0.6  # 一字封板最多打掉 60%
TURNOVER_COVERAGE = 0.7  # 近端窗口换手率覆盖率达标才用换手率，否则退回成交量
WEIGHTS = {"position": 0.20, "divergence": 0.25, "cmf": 0.25, "chip": 0.30}

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
    市值缺失的股票保留(不因缺数据误杀)。max_cap=None/0 则不按市值过滤。
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
        "high": _safe(row["daily_high"]),
        "low": _safe(row["daily_low"]),
        "close": _safe(row["daily_close"]),
        "volume": _safe(row["daily_volume"]),
        "chg": _safe(row["daily_change_pct"]),
        "turnover": _safe(row["daily_turnover_rate"]),
    }


_BAR_SQL = (
    "SELECT date, daily_high, daily_low, daily_close, daily_volume, "
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


def _chip_metrics(bars: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """信号③成本分布：换手衰减 + 当日筹码在[low,high]均匀铺开，重建流通筹码成本分布。

    返回 concentration(主峰±CHIP_BAND 内筹码占比=单峰密集度)、winner(获利盘比例)、
    peak_price(主峰成本价)。换手率缺失太多则返回 None。
    """
    lows, highs, turns = [], [], []
    for b in bars:
        if b["low"] and b["high"] and b["turnover"] is not None and b["high"] > b["low"]:
            lows.append(b["low"]); highs.append(b["high"]); turns.append(b["turnover"])
    if len(lows) < 30:
        return None
    pmin, pmax = min(lows), max(highs)
    if pmax <= pmin:
        return None
    grid = np.linspace(pmin, pmax, CHIP_BUCKETS)
    chips = np.zeros(CHIP_BUCKETS)
    span = pmax - pmin
    for lo, hi, t in zip(lows, highs, turns):
        frac = min(1.0, max(0.0, t / 100.0 * CHIP_DECAY))   # 当日搬移的筹码比例
        chips *= (1.0 - frac)                                # 旧筹码按换手衰减
        i0 = int((lo - pmin) / span * (CHIP_BUCKETS - 1))
        i1 = int((hi - pmin) / span * (CHIP_BUCKETS - 1))
        i0 = max(0, i0); i1 = min(CHIP_BUCKETS - 1, max(i0, i1))
        chips[i0:i1 + 1] += frac / (i1 - i0 + 1)             # 今日筹码在[low,high]均匀铺开
    tot = chips.sum()
    if tot <= 0:
        return None
    chips /= tot
    close = bars[-1]["close"]
    peak_price = float(grid[int(chips.argmax())])
    concentration = float(chips[np.abs(grid - peak_price) <= CHIP_BAND * peak_price].sum())
    winner = float(chips[grid <= close].sum()) if close else 0.0
    return {"concentration": concentration, "winner": winner, "peak_price": peak_price}


def _score_chip(bars: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
    """信号③低位筹码集中：单峰越密集分越高；价格已站上主峰(获利盘多=有支撑)再加成。"""
    m = _chip_metrics(bars)
    if m is None:
        return None, None
    conc = _clip01((m["concentration"] - CHIP_CONC_LO) / (CHIP_CONC_HI - CHIP_CONC_LO)) * 100.0
    score = conc * (0.5 + 0.5 * _clip01(m["winner"]))   # 价站上密集区(获利盘高)→满权，套牢在上→半权
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
    """给单只龙头算当下潜伏分，返回带子分/原始信号的明细行。"""
    bars = _recent_bars(conn, cand["code"])
    out = dict(cand)
    res = _score_bars(cand["code"], bars)
    if res is None:
        out.update({"ambush_score": None, "score_status": "INSUFFICIENT_DATA",
                    "state": "数据不足", "last_date": bars[-1]["date"] if bars else None})
        return out
    out.update({
        "ambush_score": res["ambush_score"],
        "score_status": "OK",
        "triggered": res["triggered"],
        "state": _state_label(res["raw"], res["sealed"], res["streak"], res["triggered"]),
        "last_date": bars[-1]["date"],
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

def run_ambush() -> Dict[str, Any]:
    conn = stock_storage.connect(DB_FILE)
    try:
        candidates = load_leader_candidates(conn)
        scored = [score_candidate(conn, cand) for cand in candidates]
    finally:
        conn.close()

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
        "max_market_cap_yi": MAX_MARKET_CAP_YI,
        "stocks": ranked,
    })
    if not candidates:
        payload["notes"] = ["候选池为空：先运行 python stock_crawl_segment_leaders.py crawl 选龙头并回写 is_leader。"]
    write_payload(AMBUSH_RESULT_FILE, payload)
    _print_ambush_summary(payload)
    return payload


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{value:g}"


def _print_ambush_summary(payload: Dict[str, Any]) -> None:
    stocks = payload.get("stocks", [])
    print("=" * 100)
    print("  主力资金雷达 · 吸筹分 (ambush)")
    print(f"  生成时间: {payload['generated_at']} · 候选(≤{payload.get('max_market_cap_yi', '∞')}亿小中盘龙头): "
          f"{payload['candidate_count']} · 已打分: {payload.get('scored_count', 0)}"
          f" · ▲突破启动: {payload.get('triggered_count', 0)}")
    print(f"  落盘: {display_path(AMBUSH_RESULT_FILE)}")
    print("-" * 100)
    if not stocks:
        for note in payload.get("notes", ["（无候选）"]):
            print(f"  {note}")
        print("=" * 100)
        return
    print(f"  {'#':>2} {'代码':<7}{'名称':<9}{'吸筹分':>6}  {'量比':>5} {'价分位':>6} {'CMF':>6} {'筹码集中':>7} {'连板':>4}  状态 / 细分行业")
    for i, s in enumerate(stocks[:25], 1):
        sig = s.get("signals", {})
        name = (s.get("name") or "")[:8]
        print(f"  {i:>2} {s['code']:<7}{name:<9}{s['ambush_score']:>6.1f}  "
              f"{_fmt(sig.get('vol_ratio')):>5} {_fmt(sig.get('close_pctile')):>6} "
              f"{_fmt(sig.get('cmf')):>6} {_fmt(sig.get('chip_concentration')):>7} "
              f"{sig.get('limit_streak', 0):>4}  {s.get('state', '')} / {s.get('segment_name') or ''}")
    if len(stocks) > 25:
        print(f"  ... 其余 {len(stocks) - 25} 只见 {display_path(AMBUSH_RESULT_FILE)}")
    print("=" * 100)


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
        help="运行模式：ambush(默认,潜伏分) / watch(实时监控) / verify(后验回测)",
    )
    return parser


def run_mode(mode: str) -> Dict[str, Any]:
    if mode == "watch":
        return run_watch()
    if mode == "verify":
        return run_verify()
    return run_ambush()


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    return run_mode(args.mode)


if __name__ == "__main__":
    main()
