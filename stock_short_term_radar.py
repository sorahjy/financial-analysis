"""短线雷达（super-short swing）。

目标：捕捉「未来 2-5 天能涨」的标的，持有不超过 10 个交易日——和 stock_hot_money_radar.py
的「左侧吸筹」长线命题正交。后者经 verify 反复证明买入侧是 10-40 日慢信号、2-5 日很弱
(纪要10)、龙头池超短无 alpha；本文件另起炉灶专做超短窗口。

构建纪律：买入侧每加一个因子，先在 verify 模式后验「有信息」(截面 RankIC 符号合方向且
   |t|≥1.5、核心周期 3/5 日)才并入 short_score，无效不留（避免堆无效因子/过拟合）。

构建顺序（增量落地）：
  ① 风控层：从 stock_hot_money_radar.py 搬来用户验证最有效的三个「派发/拒绝」风控因子
     P17 倒V反转 / P19 灌压巨量大阴 / P22 放量假突破，作为做多前的一票否决（避雷）。
     —— 这三个都是 sell 方向的高位派发信号；超短博反弹时，当天冒出任一信号即剔除。
  ② 买入侧短线分 short_score（超短反转，等权反向 rank 合成）：候选因子经 verify 后验，
     leader 池(2023-2026/71046样本)通过 3 个、均为「过热反向」反转因子：
       turn_pctile 换手拥挤度(IC全周期-0.03~-0.05,t≤-2.6)、amp_today 当日振幅(全周期t-1.6~-2.6)、
       dist_ma20 偏离MA20(5/10日t-1.6/-2.5)。
     DROP：mom_5d/ret_1d/vol_ratio 核心周期不显著；limitup_5d 连板显著但方向相反——纪要10
       里它在游资/全市值池是 +动量(P12@2/5日正)，龙头池实测却是反转(IC负)，印证 universe 错配。
     择时：market_regime.favorable(大盘>MA20) 才做多(纪要14)。
  ③ TODO：游资 hotmoney 小盘池本机未建(返回0)，建池后应在其主场重跑 verify——超短反转/连板
     动量的真主场是小盘题材股，leader 池结论仅是有数据的代理样本。

⚠️ 复用而非复制：风控判据、量价因子原料(_reversal_raw_features 等)直接 import 自
   stock_hot_money_radar，口径与原文件单一来源、永不漂移（同该文件对 P20「避免两处逻辑漂移」）。

Modes:
  screen   默认。逐票跑风控三因子(避雷) + 对通过者按 short_score 排序，输出买入候选 + 避雷名单。
  verify   候选因子有效性后验：PIT 截面 RankIC + t + 多空价差，逐因子裁 KEEP/DROP。
"""

from __future__ import annotations

import argparse
import bisect
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import stock_storage
import stock_hot_money_radar as hmr


SCHEMA = "short_term_radar.v1"
RESULT_FILE = hmr.CAPITAL_DIR / "short_term_radar.json"
VERIFY_RESULT_FILE = hmr.CAPITAL_DIR / "short_term_verify.json"
DB_FILE = hmr.DB_FILE
DEFAULT_POOL = hmr.DEFAULT_POOL
POOLS = hmr.POOLS
MODES = ("screen", "verify", "backtest")
DEFAULT_MODE = "screen"

# ── backtest 参数：买入组绝对收益 + 对中证500超额 ──
BENCHMARK_INDEX = "510580"      # 中证500ETF(index_nav)，做多超额基准
BACKTEST_HORIZONS = (2, 5, 10)  # 用户关注的持有期（交易日）
BUY_TOP_FRAC = 0.2              # 买入组=按买入信号取截面前20%(top quintile)
BACKTEST_MIN_GROUP = 5          # 单截面买入组至少多少只才计入

# ── 短线后验参数（贴 2-5日目标 / 持有≤10日；口径与 hmr.verify 一致，结论可比）──
LOOKBACK = hmr.LOOKBACK                 # 每只票 PIT 窗口长度（够算量比/换手分位/MA20）
SHORT_HORIZONS = (2, 3, 5, 10)          # 前向收益周期（交易日）：2-5日博反弹 + 10日持有上限
VERIFY_STEP = hmr.VERIFY_STEP           # 每隔多少个交易日取一个 as-of 截面
VERIFY_WINDOW_DAYS = hmr.VERIFY_WINDOW_DAYS   # 回测窗口（交易日，约3年）
VERIFY_MIN_NAMES = hmr.VERIFY_MIN_NAMES       # 单截面至少多少只票才计 IC/多空
VERIFY_BUCKETS = hmr.VERIFY_BUCKETS           # 多空分位桶数
KEEP_HORIZONS = (3, 5)                  # 判 KEEP/DROP 的核心周期（命中任一核心周期即留）
EFFECTIVE_T = 1.5                       # |t|≥此值算「有信息」（与 hmr._horizon_effective 同口径）

# 经 verify 后验通过（核心周期3/5日有效）的买入侧因子 → 等权并入 short_score。
# 证据(leader池/2023-2026/71046样本)：
#   turn_pctile IC 全周期 -0.03~-0.05(t≤-2.6)、amp_today 全周期(t-1.6~-2.6)、dist_ma20 5/10日(t-1.6/-2.5)；
#   三者方向均为 -1（过热/拉伸→短反转），取反向 rank 后高分=「不过热/被错杀」。
# DROP(未并入)：mom_5d/ret_1d/vol_ratio 核心周期不显著；limitup_5d 显著但方向相反
#   （龙头池连板=反转非动量，与游资池研究相左=universe 错配，详见 verify 落盘）。
# 加新因子务必先跑 `verify` 达标再登记此处（纪律：无效不加）。
SHORT_SCORE_FACTORS = [
    ("turn_pctile", -1),
    ("amp_today",   -1),
    ("dist_ma20",   -1),
]


# ── 风控层：搬自 stock_hot_money_radar 的高位派发风控因子（P17/P19/P22）──
# (编号, 名称, 判据函数)。判据函数签名 fn(bars, ctx)->bool，与原文件 PATTERNS 同源。
RISK_PATTERNS = [
    ("P17", "倒V反转", hmr._pat_inverted_v),       # 位置≥0.80 + 冲高>15%后从峰值回落≤-8%
    ("P19", "灌压巨量大阴", hmr._pat_dump_bigbear),  # 位置≥0.70 + 量比>1.8 + 实体跌>6%且收在下1/4
    ("P22", "放量假突破", hmr._pat_failed_breakout), # 破前40日高但收盘没站上 + 当日放量>1.8×
]


def risk_catalog() -> List[Dict[str, str]]:
    """风控因子总表（供前端解释 / payload 透明化）。desc 复用原文件口径。"""
    return [
        {"code": code, "name": name, "signal": "sell",
         "desc": hmr.PATTERN_DESC.get(code, "")}
        for code, name, _ in RISK_PATTERNS
    ]


def risk_control(code: str, bars: List[Dict[str, Any]],
                 ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """对一段日线窗口跑风控三因子，命中任一即否决做多（PIT 安全）。

    返回 {status, vetoed, fired:[{code,name,desc}]}：
      · status=INSUFFICIENT_DATA 数据不足、不参与；OK 才计入避雷/通过统计。
      · vetoed=True  当天出现派发/拒绝信号，超短不做多。
    """
    if len(bars) < hmr.MIN_BARS:
        return {"status": "INSUFFICIENT_DATA", "vetoed": False, "fired": []}
    ctx = ctx or hmr._build_pattern_context(code, bars)
    fired: List[Dict[str, str]] = []
    for pcode, name, fn in RISK_PATTERNS:
        try:
            if fn(bars, ctx):
                fired.append({"code": pcode, "name": name,
                              "desc": hmr.PATTERN_DESC.get(pcode, "")})
        except Exception:
            continue
    return {"status": "OK", "vetoed": bool(fired), "fired": fired}


def screen_candidate(conn, cand: Dict[str, Any],
                     as_of: Optional[str] = None) -> Dict[str, Any]:
    """给单只候选跑风控 + 算买入侧因子原始值；返回明细行。

    as_of 给定时只用该日期及以前的 bar（PIT 防泄漏，供历史复盘，复用 hmr._recent_bars）。
    short_score 是截面相对分，由 _apply_short_score 在全池上算（此处先置 None）。
    """
    bars = hmr._recent_bars(conn, cand["code"], as_of=as_of)
    out = dict(cand)
    risk = risk_control(cand["code"], bars)
    out["risk"] = risk
    out["risk_status"] = risk["status"]
    out["risk_vetoed"] = risk["vetoed"]
    out["risk_patterns"] = [f["code"] for f in risk["fired"]]
    out["last_date"] = bars[-1]["date"] if bars else None
    out["factors"] = _candidate_factor_values(cand["code"], bars) if len(bars) >= hmr.MIN_BARS else {}
    out["short_score"] = None          # 由 _apply_short_score 截面计算
    return out


def _apply_short_score(rows: List[Dict[str, Any]]) -> None:
    """买入侧短线分（0~100，截面相对分）：KEEP 因子各转截面百分位、按方向取向后等权平均。

    direction=-1 取反(100−pct)：过热/拉伸越低、分越高（超短反转=买被错杀的安静票）；
    +1 直接用 pct。某因子缺失=该项中性 50，不污染其它因子。空池直接返回。
    """
    if not rows:
        return
    pct_by_factor = {
        name: hmr._percentiles_from_values([r.get("factors", {}).get(name) for r in rows])
        for name, _ in SHORT_SCORE_FACTORS
    }
    w = 1.0 / len(SHORT_SCORE_FACTORS)
    for idx, r in enumerate(rows):
        subs: Dict[str, float] = {}
        acc = 0.0
        for name, direction in SHORT_SCORE_FACTORS:
            pct = pct_by_factor[name].get(idx, 50.0)
            contrib = (100.0 - pct) if direction < 0 else pct
            subs[name] = round(contrib, 1)
            acc += w * contrib
        r["short_score"] = round(acc, 1)
        r["short_sub_scores"] = subs


# ── 输出 ──────────────────────────────────────────────────────

def base_payload(mode: str, candidate_count: int, pool: str) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": mode,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": f"pool={pool}（短线雷达：风控避雷层）",
        "candidate_count": candidate_count,
    }


def run_screen(as_of: Optional[str] = None,
               max_cap: Optional[float] = None,
               pool: str = DEFAULT_POOL,
               write: bool = True,
               print_summary: bool = True) -> Dict[str, Any]:
    conn = stock_storage.connect(DB_FILE)
    try:
        candidates = hmr.load_candidates(conn, pool, max_cap=max_cap)
        screened = [screen_candidate(conn, c, as_of=as_of) for c in candidates]
        market_regime = hmr._market_regime(conn, as_of)
    finally:
        conn.close()

    valid = [r for r in screened if r["risk_status"] == "OK"]
    _apply_short_score(valid)                       # 截面买入分：全可评估池为横截面
    vetoed = [r for r in valid if r["risk_vetoed"]]
    clean = [r for r in valid if not r["risk_vetoed"]]
    veto_counts = Counter(c for r in vetoed for c in r["risk_patterns"])
    vetoed.sort(key=lambda r: (len(r["risk_patterns"]), r.get("market_cap_yi") or 0), reverse=True)
    clean.sort(key=lambda r: r.get("short_score") or 0.0, reverse=True)   # 买入候选：短线分高→低

    payload = base_payload("screen", len(candidates), pool)
    payload.update({
        "status": "ok" if candidates else "empty",
        "description": "短线雷达：先用 P17/P19/P22 风控避雷，再对通过者按买入侧 short_score(超短反转,verify已验证) 排序。",
        "as_of": as_of,
        "pool": pool,
        "max_market_cap_yi": max_cap,
        "market_regime": market_regime,
        "risk_model": {
            "patterns": risk_catalog(),
            "note": "三因子搬自 stock_hot_money_radar(P17/P19/P22)，import 复用、口径单一来源不漂移。",
        },
        "short_score_model": {
            "factors": [{"factor": n, "direction": d} for n, d in SHORT_SCORE_FACTORS],
            "note": "等权反向 rank 合成(过热低→分高)；因子经 verify 后验核心周期3/5日有效才并入。",
            "timing": "做多择时：market_regime.favorable(大盘>MA20) 才出手，否则观望/对冲。",
        },
        "scored_count": len(valid),
        "vetoed_count": len(vetoed),
        "clean_count": len(clean),
        "insufficient_count": sum(1 for r in screened if r["risk_status"] == "INSUFFICIENT_DATA"),
        "veto_counts": dict(veto_counts),
        "buy_candidates": clean,
        "vetoed": vetoed,
    })
    if not candidates:
        payload["notes"] = [f"候选池 {pool} 为空：先建池（leader 跑 segment_leaders / hotmoney 跑 hot_money_universe）。"]
    if write:
        hmr.write_payload(RESULT_FILE, payload)
    if print_summary:
        _print_screen_summary(payload)
    return payload


def _print_screen_summary(payload: Dict[str, Any]) -> None:
    print("=" * 100)
    print("  短线雷达 · 选股 (screen：风控避雷 + 超短反转买入分)")
    print(f"  生成时间: {payload['generated_at']} · 池: {payload.get('pool')}"
          f"({payload.get('candidate_count', 0)}) · 落盘: {hmr.display_path(RESULT_FILE)}")
    mr = payload.get("market_regime")
    if isinstance(mr, dict) and mr.get("available"):
        print(f"  大盘: {'站上' if mr.get('favorable') else '跌破'}MA20 — {mr.get('note', '')}")
    vc = payload.get("veto_counts", {})
    vc_str = " ".join(f"{code}×{vc.get(code, 0)}" for code, _, _ in RISK_PATTERNS)
    favorable = bool(isinstance(mr, dict) and mr.get("favorable"))
    print(f"  评估 {payload.get('scored_count', 0)} 只 → 避雷 {payload.get('vetoed_count', 0)} / "
          f"买入候选 {payload.get('clean_count', 0)} / 数据不足 {payload.get('insufficient_count', 0)}"
          f"   风控命中: {vc_str}")
    sf = " ".join(f"{n}({d:+d})" for n, d in SHORT_SCORE_FACTORS)
    print(f"  买入分因子(verify已验证·反向): {sf}   ⚠️择时: {'✅大盘站上MA20·可做多' if favorable else '❌大盘弱·宜观望'}")
    print("-" * 100)

    # ① 买入候选：风控通过 + 短线分高→低（超短反转，买不过热/被错杀的安静票）
    buys = payload.get("buy_candidates", [])
    print(f"  ▼ 买入候选 Top（short_score 高=越不过热/越被错杀；持有2-5日博反弹·上限10日）")
    if not buys:
        print("    （全池被风控否决或池空）")
    else:
        bh = (hmr._ljust("代码", 8) + hmr._ljust("名称", 11) + hmr._rjust("短线分", 7)
              + hmr._rjust("市值亿", 8) + "  " + " ".join(hmr._rjust(n[:7], 8) for n, _ in SHORT_SCORE_FACTORS)
              + "  行业/板块")
        print("  " + bh)
        for s in buys[:15]:
            cap = s.get("market_cap_yi")
            cap_s = f"{cap:.0f}" if cap is not None else "-"
            subs = s.get("short_sub_scores", {})
            sub_cells = " ".join(hmr._rjust(f"{subs.get(n, 0):.0f}", 8) for n, _ in SHORT_SCORE_FACTORS)
            seg = s.get("segment_name") or s.get("parent_segment") or ""
            row = (hmr._ljust(s["code"], 8) + hmr._ljust(s.get("name", "")[:5], 11)
                   + hmr._rjust(s.get("short_score", 0), 7) + hmr._rjust(cap_s, 8)
                   + "  " + sub_cells + "  " + seg[:14])
            print("  " + row)
    print("-" * 100)

    # ② 避雷名单：命中派发风控（不做多）
    vetoed = payload.get("vetoed", [])
    print(f"  ▼ 避雷名单（命中 P17/P19/P22 派发信号=不做多，共 {len(vetoed)}）")
    if not vetoed:
        print("    （无标的命中派发风控）")
    else:
        header = (hmr._ljust("代码", 8) + hmr._ljust("名称", 11) + hmr._rjust("市值亿", 8)
                  + hmr._rjust("命中", 6) + "  风控信号                行业/板块")
        print("  " + header)
        for s in vetoed[:20]:
            cap = s.get("market_cap_yi")
            cap_s = f"{cap:.0f}" if cap is not None else "-"
            sig = "+".join(f"{f['code']}{f['name']}" for f in s["risk"]["fired"])
            seg = s.get("segment_name") or s.get("parent_segment") or ""
            row = (hmr._ljust(s["code"], 8) + hmr._ljust(s.get("name", "")[:5], 11)
                   + hmr._rjust(cap_s, 8) + hmr._rjust(len(s["risk_patterns"]), 6)
                   + "  " + hmr._ljust(sig, 24) + seg[:14])
            print("  " + row)
        if len(vetoed) > 20:
            print(f"    …另有 {len(vetoed) - 20} 只命中，详见落盘 JSON。")
    print("=" * 100)


# ── verify：短线因子有效性后验（PIT 截面 RankIC，逐因子裁 KEEP/DROP）──
#
# 纪律：买入侧每加一个因子，先在此台子验证「有信息」才并入 short_score，无效不留。
# 口径与 hmr.verify 一致：逐 as-of 日截面内部 Spearman(因子, 前向收益) → 跨日平均 + t 值
# （天然剔大盘 beta）；多空价差辅证。direction=因子的预期方向：
#   -1 = 过热/反转（预期 IC 为负：值越高未来越弱，做多时取反向）；
#   +1 = 动量（预期 IC 为正：值越高未来越强）。
# 有效 = sign(IC)==direction 且 |t|≥EFFECTIVE_T。

def _candidate_factor_values(code: str, window: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """PIT 计算所有候选因子原始值（window 末根 = as-of 当日）。

    复用 hmr._reversal_raw_features（5 个过热因子，已 PIT 验证）再补两个均值回归候选。
    """
    feats: Dict[str, Optional[float]] = dict(hmr._reversal_raw_features(code, window))
    closes = [b["close"] for b in window]
    feats["ret_1d"] = (closes[-1] / closes[-2] - 1.0) if len(closes) > 1 and closes[-1] and closes[-2] else None
    ma20 = hmr._ma_last(closes, 20)
    feats["dist_ma20"] = (closes[-1] / ma20 - 1.0) if ma20 and closes[-1] else None
    return feats


# (因子名, 预期方向, 角色说明)。新增候选因子在此登记，跑 verify 看是否够格并入买入侧分。
CANDIDATE_FACTORS: List[tuple] = [
    ("mom_5d",      -1, "近5日涨幅(过热→短反转)"),
    ("ret_1d",      -1, "昨日涨幅(隔日反转)"),
    ("turn_pctile", -1, "换手拥挤度(过热)"),
    ("amp_today",   -1, "当日振幅(过热)"),
    ("vol_ratio",   -1, "量比(过热)"),
    ("limitup_5d",  +1, "近5日涨停数(连板动量)"),
    ("dist_ma20",   -1, "偏离MA20(均值回归)"),
]


def _collect_factor_samples(conn, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """PIT 滑窗：对每只票每个 as-of 截面算齐候选因子原始值 + 未来前向收益。

    samples 每项 = {date, code, factors:{name:val}, rets:{h:ret}}。
    PIT：因子只用截止 as-of 当日的 LOOKBACK 根 bar；收益用其后第 h 根收盘价。
    """
    max_h = max(SHORT_HORIZONS)
    series: Dict[str, tuple] = {}
    for cand in candidates:
        bars = hmr._all_bars(conn, cand["code"])
        if len(bars) < LOOKBACK + max_h + 1:
            continue
        series[cand["code"]] = (bars, {b["date"]: i for i, b in enumerate(bars)})
    if not series:
        return {"samples": [], "dates": [], "codes": []}

    all_dates = sorted({d for bars, _ in series.values() for d in (b["date"] for b in bars)})
    as_of_dates = all_dates[:-max_h][-VERIFY_WINDOW_DAYS:][::VERIFY_STEP]
    samples: List[Dict[str, Any]] = []
    used: set = set()
    for d in as_of_dates:
        for code, (bars, idx_map) in series.items():
            i = idx_map.get(d)
            if i is None or i < LOOKBACK - 1 or i + max_h >= len(bars):
                continue
            close_i = bars[i]["close"]
            if not close_i:
                continue
            rets: Dict[int, float] = {}
            ok = True
            for h in SHORT_HORIZONS:
                cf = bars[i + h]["close"]
                if not cf:
                    ok = False
                    break
                rets[h] = cf / close_i - 1.0
            if not ok:
                continue
            feats = _candidate_factor_values(code, bars[i - LOOKBACK + 1:i + 1])
            samples.append({"date": d, "code": code, "factors": feats, "rets": rets})
            used.add(d)
    return {"samples": samples, "dates": sorted(used), "codes": list(series.keys())}


def _factor_horizon_ic(samples: List[Dict[str, Any]], name: str, h: int) -> Dict[str, Any]:
    """单因子×单周期：逐 as-of 日截面 Spearman(因子,收益) → 跨日均值 + t + 多空价差。"""
    by_date: Dict[str, List[tuple]] = {}
    for s in samples:
        v = s["factors"].get(name)
        if v is None:
            continue
        by_date.setdefault(s["date"], []).append((v, s["rets"][h]))
    ics: List[float] = []
    spreads: List[float] = []
    for grp in by_date.values():
        if len(grp) < VERIFY_MIN_NAMES:
            continue
        ic = hmr._spearman([g[0] for g in grp], [g[1] for g in grp])
        if ic is not None:
            ics.append(ic)
        sg = sorted(grp, key=lambda g: g[0])
        k = max(1, len(sg) // VERIFY_BUCKETS)
        top = hmr._mean([g[1] for g in sg[-k:]])
        bot = hmr._mean([g[1] for g in sg[:k]])
        if top is not None and bot is not None:
            spreads.append(top - bot)
    ic_mean = hmr._mean(ics)
    ic_std = (sum((x - ic_mean) ** 2 for x in ics) / len(ics)) ** 0.5 if len(ics) > 1 else None
    t = (ic_mean / ic_std * math.sqrt(len(ics))) if ic_mean is not None and ic_std else None
    return {
        "ic_mean": round(ic_mean, 4) if ic_mean is not None else None,
        "ic_t_stat": round(t, 2) if t is not None else None,
        "ic_hit_rate": round(sum(1 for x in ics if x > 0) / len(ics), 3) if ics else None,
        "long_short_spread": round(hmr._mean(spreads), 4) if spreads else None,
        "n_sections": len(ics),
    }


def _factor_effective(direction: int, hz: Dict[str, Any]) -> bool:
    """该周期是否「有信息」：IC 符号与预期方向一致且 |t|≥EFFECTIVE_T。"""
    ic, t = hz.get("ic_mean"), hz.get("ic_t_stat")
    if ic is None or t is None:
        return False
    return (ic * direction > 0) and abs(t) >= EFFECTIVE_T


def _factor_report(samples: List[Dict[str, Any]], name: str, direction: int, role: str) -> Dict[str, Any]:
    horizons = {str(h): _factor_horizon_ic(samples, name, h) for h in SHORT_HORIZONS}
    keep = any(_factor_effective(direction, horizons[str(h)]) for h in KEEP_HORIZONS)
    eff_h = [h for h in SHORT_HORIZONS if _factor_effective(direction, horizons[str(h)])]
    return {"factor": name, "direction": direction, "role": role,
            "horizons": horizons, "keep": keep, "effective_horizons": eff_h}


def run_verify(as_of: Optional[str] = None, max_cap: Optional[float] = None,
               pool: str = DEFAULT_POOL) -> Dict[str, Any]:
    conn = stock_storage.connect(DB_FILE)
    try:
        candidates = hmr.load_candidates(conn, pool, max_cap=max_cap)
        collected = _collect_factor_samples(conn, candidates)
    finally:
        conn.close()
    samples = collected["samples"]
    reports = [_factor_report(samples, n, d, r) for n, d, r in CANDIDATE_FACTORS] if samples else []

    payload = base_payload("verify", len(candidates), pool)
    payload.update({
        "status": "ok" if samples else "empty",
        "description": "短线因子有效性后验：PIT 截面 RankIC(因子,前向收益) + t + 多空价差，逐因子裁 KEEP/DROP。",
        "params": {
            "horizons": list(SHORT_HORIZONS), "step": VERIFY_STEP,
            "window_days": VERIFY_WINDOW_DAYS, "buckets": VERIFY_BUCKETS,
            "min_names_per_section": VERIFY_MIN_NAMES,
            "keep_horizons": list(KEEP_HORIZONS), "effective_t": EFFECTIVE_T,
        },
        "pool": pool,
        "section_count": len(collected["dates"]),
        "sample_count": len(samples),
        "scored_codes": len(collected["codes"]),
        "date_range": [collected["dates"][0], collected["dates"][-1]] if collected["dates"] else None,
        "factors": reports,
        "kept_factors": [r["factor"] for r in reports if r["keep"]],
    })
    if not samples:
        payload["notes"] = ["无可回测样本：候选池空或历史日线不足。"]
    hmr.write_payload(VERIFY_RESULT_FILE, payload)
    _print_verify_summary(payload)
    return payload


def _print_verify_summary(payload: Dict[str, Any]) -> None:
    print("=" * 100)
    print("  短线雷达 · 因子有效性后验 (verify)")
    rng = payload.get("date_range")
    print(f"  生成时间: {payload['generated_at']} · 池: {payload.get('pool')}({payload['candidate_count']})"
          f" · 截面: {payload.get('section_count', 0)} · 样本: {payload.get('sample_count', 0)}"
          + (f" · 区间: {rng[0]}~{rng[1]}" if rng else ""))
    print(f"  落盘: {hmr.display_path(VERIFY_RESULT_FILE)}")
    print(f"  口径: 逐as-of日截面 Spearman(因子,前向收益)→跨日均值+t（剔大盘beta）；"
          f"有效=符号合方向且|t|≥{EFFECTIVE_T}；KEEP=核心周期{KEEP_HORIZONS}任一有效")
    print("-" * 100)
    reports = payload.get("factors", [])
    if not reports:
        for note in payload.get("notes", ["（无样本）"]):
            print(f"  {note}")
        print("=" * 100)
        return
    hdr = hmr._ljust("因子", 13) + hmr._rjust("方向", 5) + " "
    hdr += " ".join(hmr._rjust(f"{h}日IC(t)", 14) for h in SHORT_HORIZONS)
    hdr += "  " + hmr._ljust("裁决", 8) + "角色"
    print("  " + hdr)
    for r in reports:
        cells = []
        for h in SHORT_HORIZONS:
            hz = r["horizons"][str(h)]
            ic, t = hz.get("ic_mean"), hz.get("ic_t_stat")
            mark = "*" if _factor_effective(r["direction"], hz) else " "
            cells.append(hmr._rjust(f"{ic:+.3f}({t:+.1f}){mark}" if ic is not None and t is not None else "-", 14))
        verdict = "✅KEEP" if r["keep"] else "❌DROP"
        eff = (",".join(f"{h}d" for h in r["effective_horizons"])) if r["effective_horizons"] else ""
        row = (hmr._ljust(r["factor"], 13) + hmr._rjust(f"{r['direction']:+d}", 5) + " "
               + " ".join(cells) + "  " + hmr._ljust(verdict, 8) + (r["role"] + (f" [{eff}]" if eff else "")))
        print("  " + row)
    print("-" * 100)
    kept = payload.get("kept_factors", [])
    print(f"  通过(KEEP) {len(kept)}/{len(reports)}: {', '.join(kept) if kept else '（本批无因子达标）'}")
    print(f"  * 标记=该周期有效(符号合方向且|t|≥{EFFECTIVE_T})。只把 KEEP 因子并入买入侧 short_score。")
    print("=" * 100)


# ── backtest：买入组绝对收益 + 对中证500超额（持有 2/5/10 日）──
#
# 回答「三个买入因子分别能涨多少 / 跑赢中证500多少」：每个 as-of 日按买入信号取截面前 20%
# 买入组，算其后 h 日均涨幅；中证500(510580)同期 h 个交易日涨幅为基准；超额=买入组−中证500。
# 跨日平均 + t（剔大盘 beta 的另一种口径：直接减指数收益），胜率=跑赢中证500的截面占比。

BENCH_JUMP_THRESHOLD = 0.20    # 相邻日 |涨跌|>此值视为数据断裂（index_nav 拼接错误），其窗口作废


def _load_benchmark(conn) -> Dict[str, Any]:
    """加载基准指数 nav → {dates(升序), navs(并列数组), name, bad(断裂位的右端索引集合)}。

    ⚠️ index_nav 的 510580/510310 存在跨标的拼接断裂（|日收益|>20% 的假跳变），
    前向窗口一旦跨越断裂日，比值就是假的——故标记断裂位、_benchmark_forward 跨越即作废。
    """
    entry = stock_storage.load_index_nav(conn, BENCHMARK_INDEX)
    recs = (entry.get("records") if isinstance(entry, dict) else None) or []
    series = []
    for r in recs:
        d, v = r.get("date"), hmr._safe(r.get("nav"))
        if d and v is not None and v > 0:
            series.append((d, v))
    series.sort()
    dates = [d for d, _ in series]
    navs = [v for _, v in series]
    bad = {i for i in range(1, len(navs)) if abs(navs[i] / navs[i - 1] - 1.0) > BENCH_JUMP_THRESHOLD}
    name = (entry.get("name") if isinstance(entry, dict) else None) or BENCHMARK_INDEX
    return {"dates": dates, "navs": navs, "bad": bad, "name": name, "n_jumps": len(bad)}


def _benchmark_forward(bench: Dict[str, Any], d: str, h: int) -> Optional[float]:
    """基准从「最近的 ≤ d 的交易日」起 h 个交易日的前向收益；窗口跨越断裂日则返回 None。"""
    dates, navs, bad = bench["dates"], bench["navs"], bench["bad"]
    pos = bisect.bisect_right(dates, d) - 1
    if pos < 0 or pos + h >= len(dates):
        return None
    if any(j in bad for j in range(pos + 1, pos + h + 1)):   # 窗口内含断裂日 → 作废
        return None
    base = navs[pos]
    return (navs[pos + h] / base - 1.0) if base else None


def _buygroup_raw(grp: List[Dict[str, Any]], name: str, direction: int,
                  top_frac: float) -> List[Dict[str, Any]]:
    """单因子买入组：direction<0 取原始值最低的前 top_frac（不过热），>0 取最高的。"""
    vals = [(s, s["factors"].get(name)) for s in grp]
    vals = [(s, v) for s, v in vals if v is not None]
    if not vals:
        return []
    vals.sort(key=lambda x: x[1])
    k = max(BACKTEST_MIN_GROUP, round(len(vals) * top_frac))
    chosen = vals[:k] if direction < 0 else vals[-k:]
    return [s for s, _ in chosen]


def _buygroup_short_score(grp: List[Dict[str, Any]], top_frac: float) -> List[Dict[str, Any]]:
    """合成分买入组：截面 reverse-rank 等权合成 short_score，取最高的前 top_frac。"""
    pcts = {name: hmr._percentiles_from_values([s["factors"].get(name) for s in grp])
            for name, _ in SHORT_SCORE_FACTORS}
    w = 1.0 / len(SHORT_SCORE_FACTORS)
    scored = []
    for idx, s in enumerate(grp):
        acc = 0.0
        for name, direction in SHORT_SCORE_FACTORS:
            pct = pcts[name].get(idx, 50.0)
            acc += w * ((100.0 - pct) if direction < 0 else pct)
        scored.append((s, acc))
    scored.sort(key=lambda x: x[1])
    k = max(BACKTEST_MIN_GROUP, round(len(scored) * top_frac))
    return [s for s, _ in scored[-k:]]


def _backtest_one(samples: List[Dict[str, Any]], bench: Dict[str, Any],
                  selector, h: int) -> Dict[str, Any]:
    """跨 as-of 日：买入组均涨幅 / 中证500同期 / 超额(+t/胜率) / 全池均涨(参考)。"""
    by_date: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        if h in s["rets"]:
            by_date.setdefault(s["date"], []).append(s)
    grp_rets, bench_rets, excesses, pool_rets = [], [], [], []
    for d, grp in by_date.items():
        bret = _benchmark_forward(bench, d, h)
        if bret is None:
            continue
        buy = selector(grp)
        rets = [s["rets"][h] for s in buy]
        if len(rets) < BACKTEST_MIN_GROUP:
            continue
        g = hmr._mean(rets)
        grp_rets.append(g)
        bench_rets.append(bret)
        excesses.append(g - bret)
        pool_rets.append(hmr._mean([s["rets"][h] for s in grp]))
    if not excesses:
        return {"n_sections": 0}
    em = hmr._mean(excesses)
    es = (sum((x - em) ** 2 for x in excesses) / len(excesses)) ** 0.5 if len(excesses) > 1 else None
    t = (em / es * math.sqrt(len(excesses))) if es else None
    return {
        "n_sections": len(excesses),
        "buy_ret": round(hmr._mean(grp_rets), 4),
        "benchmark_ret": round(hmr._mean(bench_rets), 4),
        "pool_ret": round(hmr._mean(pool_rets), 4),
        "excess_mean": round(em, 4),
        "excess_t_stat": round(t, 2) if t is not None else None,
        "win_rate": round(sum(1 for x in excesses if x > 0) / len(excesses), 3),
    }


def run_backtest(as_of: Optional[str] = None, max_cap: Optional[float] = None,
                 pool: str = DEFAULT_POOL) -> Dict[str, Any]:
    conn = stock_storage.connect(DB_FILE)
    try:
        candidates = hmr.load_candidates(conn, pool, max_cap=max_cap)
        collected = _collect_factor_samples(conn, candidates)
        bench = _load_benchmark(conn)
    finally:
        conn.close()
    samples = collected["samples"]
    role_of = {n: r for n, _, r in CANDIDATE_FACTORS}

    entries: List[Dict[str, Any]] = []
    if samples:
        for name, direction in SHORT_SCORE_FACTORS:               # 三个买入因子分别
            sel = (lambda grp, nm=name, dr=direction: _buygroup_raw(grp, nm, dr, BUY_TOP_FRAC))
            entries.append({"factor": name, "role": role_of.get(name, ""),
                            "horizons": {str(h): _backtest_one(samples, bench, sel, h) for h in BACKTEST_HORIZONS}})
        sel_c = (lambda grp: _buygroup_short_score(grp, BUY_TOP_FRAC))   # 合成分
        entries.append({"factor": "short_score(合成)", "role": "三因子等权",
                        "horizons": {str(h): _backtest_one(samples, bench, sel_c, h) for h in BACKTEST_HORIZONS}})

    payload = base_payload("backtest", len(candidates), pool)
    payload.update({
        "status": "ok" if samples else "empty",
        "description": "买入组(截面前20%)绝对收益 + 对中证500超额，持有 2/5/10 日；跨as-of日平均+t+胜率。",
        "params": {
            "horizons": list(BACKTEST_HORIZONS), "buy_top_frac": BUY_TOP_FRAC,
            "step": VERIFY_STEP, "window_days": VERIFY_WINDOW_DAYS, "min_group": BACKTEST_MIN_GROUP,
            "benchmark": {"index": BENCHMARK_INDEX, "name": bench["name"], "n_jumps": bench["n_jumps"],
                          "caveat": "index_nav 数据有拼接断裂，跨断裂日的前向窗口已剔除(h越大剔越多)；建议重爬指数净值。"},
        },
        "pool": pool,
        "section_count": len(collected["dates"]),
        "sample_count": len(samples),
        "date_range": [collected["dates"][0], collected["dates"][-1]] if collected["dates"] else None,
        "entries": entries,
    })
    if not samples:
        payload["notes"] = ["无可回测样本：候选池空或历史日线不足。"]
    hmr.write_payload(hmr.CAPITAL_DIR / "short_term_backtest.json", payload)
    _print_backtest_summary(payload)
    return payload


def _print_backtest_summary(payload: Dict[str, Any]) -> None:
    print("=" * 104)
    print("  短线雷达 · 买入因子回测 (backtest)")
    rng = payload.get("date_range")
    bm = payload.get("params", {}).get("benchmark", {})
    print(f"  生成时间: {payload['generated_at']} · 池: {payload.get('pool')}({payload['candidate_count']})"
          f" · 截面: {payload.get('section_count', 0)} · 样本: {payload.get('sample_count', 0)}"
          + (f" · 区间: {rng[0]}~{rng[1]}" if rng else ""))
    print(f"  买入组=截面前{payload.get('params', {}).get('buy_top_frac', 0) * 100:.0f}% · 基准: 中证500({bm.get('index')})"
          f" · 落盘: {hmr.display_path(hmr.CAPITAL_DIR / 'short_term_backtest.json')}")
    if bm.get("n_jumps"):
        print(f"  ⚠️基准 index_nav 有 {bm.get('n_jumps')} 处数据断裂(跨标的拼接)，跨断裂窗口已剔除(h越大剔越多)；数字偏代理，建议重爬指数净值")
    print("-" * 104)
    entries = payload.get("entries", [])
    if not entries:
        for note in payload.get("notes", ["（无样本）"]):
            print(f"  {note}")
        print("=" * 104)
        return
    print(f"  {'因子':<14}{'持有':>5}{'买入组涨幅':>11}{'中证500':>10}{'超额':>9}{'超额t':>7}{'胜率':>7}{'(全池均涨)':>11}")
    for e in entries:
        first = True
        for h in BACKTEST_HORIZONS:
            hz = e["horizons"].get(str(h), {})
            label = e["factor"] if first else ""
            win = hz.get("win_rate")
            win_s = f"{win * 100:.0f}%" if win is not None else "-"
            print(f"  {label:<14}{str(h) + '日':>5}{hmr._pct(hz.get('buy_ret')):>11}{hmr._pct(hz.get('benchmark_ret')):>10}"
                  f"{hmr._pct(hz.get('excess_mean')):>9}{hmr._fmt(hz.get('excess_t_stat')):>7}{win_s:>7}"
                  f"{hmr._pct(hz.get('pool_ret')):>11}")
            first = False
        print(f"  {e.get('role', '')}")
    print("-" * 104)
    print("  说明: 买入组涨幅=能涨多少(含大盘beta)；超额=买入组−中证500同期=能跑赢多少(剔beta)；"
          "胜率=跑赢中证500的截面占比；持有期到点即平。")
    print("=" * 104)


# ── CLI ───────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="短线雷达（2-5日博反弹/持有≤10日）")
    parser.add_argument(
        "mode", nargs="?", choices=MODES, default=DEFAULT_MODE,
        help="运行模式：screen(默认,风控避雷+买入分排序) / verify(候选因子有效性后验,逐因子裁KEEP/DROP) / "
             "backtest(买入因子在2/5/10日的绝对收益与对中证500超额)。",
    )
    parser.add_argument(
        "--as-of", default=None, metavar="YYYY-MM-DD",
        help="只用该日期及以前的 bar（PIT 历史复盘）；默认最新交易日。",
    )
    parser.add_argument(
        "--exclude-large-cap", action=argparse.BooleanOptionalAction, default=False,
        help=f"剔除总市值>{hmr.MAX_MARKET_CAP_YI:g}亿（仅对 leader 池生效）。",
    )
    parser.add_argument(
        "--pool", choices=POOLS, default=DEFAULT_POOL,
        help="候选池：leader=细分龙头(默认) / hotmoney=游资小盘universe。",
    )
    return parser


def run_mode(mode: str, as_of: Optional[str] = None,
             max_cap: Optional[float] = None, pool: str = DEFAULT_POOL) -> Dict[str, Any]:
    if mode == "verify":
        return run_verify(as_of=as_of, max_cap=max_cap, pool=pool)
    if mode == "backtest":
        return run_backtest(as_of=as_of, max_cap=max_cap, pool=pool)
    return run_screen(as_of=as_of, max_cap=max_cap, pool=pool)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    # hotmoney 池建池时已按流通市值过滤；市值剔除仅对 leader 池生效（与 hmr 一致）
    max_cap = hmr.MAX_MARKET_CAP_YI if (args.exclude_large_cap and args.pool == "leader") else None
    return run_mode(args.mode, as_of=args.as_of, max_cap=max_cap, pool=args.pool)


if __name__ == "__main__":
    main()
