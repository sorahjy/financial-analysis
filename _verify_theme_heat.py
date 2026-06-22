"""verify 实验：行业热度对吸筹分的边际 IC 增量。

★【是否固化成雷达的正式实验模式：待定】★（暂留作研究脚本，先不删）

结论(2026-06-22 跑出)：剔大盘590只/43截面/24826样本——行业热度是【负增量】。
  S0原始 IC 2/5/10/20/40 +0.058/+0.056/+0.056/+0.059/+0.065(t2.7~4.5)；
  加热度后全线塌：S2+真实 边际 -0.03~-0.049、S1+拟合 -0.04~-0.056、S3两者 -0.064~-0.086。
  行业热度单因子全周期显著负(真实-0.045~-0.024、拟合-0.056~-0.038)=中期反转/退潮(追热点=接盘)。
  与"6/22单日 ρ=+0.135 正相关"相反：热度只在【当天】是动量beta，2-40日反转。真实优于拟合(走势拟合引噪)。

4组对比(同口径同样本)：
  S0 原始吸筹分
  S1 原始 + 拟合行业热度(走势相似)
  S2 原始 + 真实SW2行业热度
  S3 原始 + 真实 + 拟合
PIT：吸筹分只用 <=d 的 bar；行业热度用 plate_daily<=d 重算；拟合关系用 <=d 的相关性。
剔大盘(<=300亿)、前向 2/5/10/20/40 日，截面 Rank IC 跨日平均。
"""
import os
os.environ["STOCK_CRAWL_NO_PROXY"] = "1"
import sqlite3
import math
import numpy as np
import pandas as pd

import stock_storage
import stock_hot_money_radar as R

DB = stock_storage.DEFAULT_DB_FILE
PLATE_DB = R.DATA_DIR / "plate_data.sqlite3"
HORIZONS = (2, 5, 10, 20, 40)
STEP = 10
WINDOW = 750
MIN_NAMES = 30
MAX_CAP = 300.0          # 剔大盘(纪要口径，基线应≈0.04)
TRACK_MIN_DAYS = 120     # 拟合相关性最少匹配日
TRACK_WIN = 280          # 拟合用的近端窗口(交易日，≈180*1.55)

def pct_rank(s):
    return s.rank(pct=True, method="average") * 100.0

# ---------- 1. 板块历史 + 滚动指标(全量预计算) ----------
print("加载板块历史...", flush=True)
pc = sqlite3.connect(PLATE_DB)
plates = pd.read_sql_query(
    "SELECT plate_code, plate_name, trade_date, close_index, change_pct, turnover_rate, amount_share_pct "
    "FROM plate_daily WHERE plate_type='sw2' ORDER BY plate_code, trade_date", pc)
pc.close()
frames = []
for _, g in plates.groupby("plate_code"):
    g = g.sort_values("trade_date").copy()
    g["ret20"] = g["close_index"].pct_change(20) * 100
    g["ret60"] = g["close_index"].pct_change(60) * 100
    g["turn20"] = g["turnover_rate"].rolling(20, min_periods=5).mean()
    g["turn120"] = g["turnover_rate"].rolling(120, min_periods=20).mean()
    g["amt20"] = g["amount_share_pct"].rolling(20, min_periods=5).mean()
    frames.append(g)
plate_full = pd.concat(frames, ignore_index=True)
plate_name_by_code = plates.groupby("plate_code")["plate_name"].last().to_dict()
# 拟合用矩阵(date×plate)
M_change = plate_full.pivot(index="trade_date", columns="plate_code", values="change_pct")
M_turn = plate_full.pivot(index="trade_date", columns="plate_code", values="turnover_rate")
M_ret60 = plate_full.pivot(index="trade_date", columns="plate_code", values="ret60")
print(f"  板块 {plate_full['plate_code'].nunique()} 个, {plate_full['trade_date'].nunique()} 日", flush=True)

def heat_pctile_on(date):
    """as_of=date 的截面：每个 plate_code -> 热度百分位(0-100)。PIT。"""
    snap = plate_full[plate_full["trade_date"] == date]
    if snap.empty:
        return {}
    s = snap.copy()
    score = (pct_rank(s["ret20"]) * 0.25 + pct_rank(s["ret60"]) * 0.20
             + pct_rank(s["turn20"]) * 0.25 + pct_rank(s["amt20"]) * 0.20
             + pct_rank((s["turn20"] / s["turn120"].replace(0, np.nan))) * 0.10)
    heat = pct_rank(score)
    return dict(zip(s["plate_code"], heat))

# ---------- 2. 候选池 + 真实行业归属(静态近似) ----------
conn = stock_storage.connect(DB)
cands = R.load_leader_candidates(conn, max_cap=MAX_CAP)
print(f"候选(剔大盘<= {MAX_CAP}亿): {len(cands)}", flush=True)
# plate_name -> plate_code（真实行业热度按 parent_segment 名匹配）
code_by_name = {v: k for k, v in plate_name_by_code.items()}

# 预取每只股票全历史 bar(吸筹分用) + 日收益/换手(拟合用)
stock_bars = {}
stock_ts = {}   # DataFrame index=date: ret1, ret60, turn
for c in cands:
    bars = R._all_bars(conn, c["code"])
    if len(bars) < R.LOOKBACK + max(HORIZONS) + 1:
        continue
    stock_bars[c["code"]] = bars
    df = pd.DataFrame({
        "trade_date": [b["date"] for b in bars],
        "close": [b["close"] for b in bars],
        "turn": [b["turnover"] for b in bars],
    }).set_index("trade_date")
    df["ret1"] = df["close"].pct_change() * 100
    df["ret60"] = df["close"].pct_change(60) * 100
    stock_ts[c["code"]] = df
conn.close()
cand_by_code = {c["code"]: c for c in cands}
codes = list(stock_bars.keys())
print(f"  有效(够bar): {len(codes)}", flush=True)

# ---------- 3. as_of 日 ----------
all_dates = sorted(set(M_change.index) & set().union(*[set(s.index) for s in stock_ts.values()]))
all_dates = sorted({b["date"] for c in codes for b in stock_bars[c]})
maxh = max(HORIZONS)
usable = all_dates[:-maxh]
window = usable[-WINDOW:]
as_of_dates = window[::STEP]
print(f"as_of 截面数: {len(as_of_dates)} ({as_of_dates[0]}..{as_of_dates[-1]})", flush=True)

def best_track_heat(code, d, heat_map):
    """PIT 拟合行业 -> 其热度百分位。"""
    sdf = stock_ts[code]
    sdf = sdf[sdf.index <= d].tail(TRACK_WIN)
    if len(sdf) < 180:
        return None
    idx = sdf.index
    pc_ch = M_change.reindex(idx)
    pc_tn = M_turn.reindex(idx)
    pc_r60 = M_ret60.reindex(idx)
    matched = pc_ch.notna().mul(sdf["ret1"].notna(), axis=0).sum()
    rc = pc_ch.corrwith(sdf["ret1"])
    tc = pc_tn.corrwith(sdf["turn"])
    t60 = pc_r60.corrwith(sdf["ret60"])
    track = rc.fillna(0) * 0.55 + t60.fillna(0) * 0.25 + tc.fillna(0) * 0.20
    track = track.where(matched >= TRACK_MIN_DAYS).dropna()
    if track.empty:
        return None
    pcode = str(track.idxmax())
    return heat_map.get(pcode)

# ---------- 4. 收集样本 ----------
samples = []   # {date, code, amb, ret{h}, theme_heat, sw2_heat}
for di, d in enumerate(as_of_dates):
    heat_map = heat_pctile_on(d)
    if not heat_map:
        continue
    for code in codes:
        bars = stock_bars[code]
        idx_map = stock_ts[code]
        # 找 d 在 bars 的位置
        pos = None
        # bars 升序, 用二分
        dates_list = idx_map.index
        # idx_map index 与 bars date 对齐
        try:
            i = list(dates_list).index(d)
        except ValueError:
            continue
        if i < R.LOOKBACK - 1 or i + maxh >= len(bars):
            continue
        res = R._score_bars(code, bars[i - R.LOOKBACK + 1:i + 1])
        if res is None:
            continue
        close_i = bars[i]["close"]
        if not close_i:
            continue
        rets = {}
        ok = True
        for h in HORIZONS:
            cf = bars[i + h]["close"]
            if not cf:
                ok = False; break
            rets[h] = cf / close_i - 1.0
        if not ok:
            continue
        # 真实行业热度
        sw2_name = cand_by_code[code].get("parent_segment") or ""
        sw2_heat = heat_map.get(code_by_name.get(sw2_name))
        # 拟合行业热度
        theme_heat = best_track_heat(code, d, heat_map)
        samples.append({"date": d, "code": code, "amb": res["ambush_score"],
                        "rets": rets, "theme_heat": theme_heat, "sw2_heat": sw2_heat})
    if (di + 1) % 10 == 0:
        print(f"  ...{di+1}/{len(as_of_dates)} 截面, 累计样本 {len(samples)}", flush=True)

print(f"总样本: {len(samples)}", flush=True)

# 只保留4组都可算的同口径样本
full = [s for s in samples if s["theme_heat"] is not None and s["sw2_heat"] is not None]
print(f"同口径(有拟合+真实热度)样本: {len(full)}", flush=True)

# ---------- 5. 合成 + 截面 IC ----------
from collections import defaultdict
by_date = defaultdict(list)
for s in full:
    by_date[s["date"]].append(s)

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 5:
        return None
    rx = pd.Series(x).rank().values; ry = pd.Series(y).rank().values
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])

def pctile(arr):
    return pd.Series(arr).rank(pct=True).values * 100.0

variants = {
    "S0 原始": lambda amb, th, sw: amb,
    "S1 +拟合行业": lambda amb, th, sw: 0.5 * pctile(amb) + 0.5 * np.asarray(th),
    "S2 +真实行业": lambda amb, th, sw: 0.5 * pctile(amb) + 0.5 * np.asarray(sw),
    "S3 +真实+拟合": lambda amb, th, sw: (pctile(amb) + np.asarray(th) + np.asarray(sw)) / 3.0,
}

print("\n" + "=" * 78)
print(f"{'变体':16}", end="")
for h in HORIZONS:
    print(f"  IC{h:>2}d(t)      ", end="")
print()
print("-" * 78)
results = {}
for name, fn in variants.items():
    print(f"{name:16}", end="")
    for h in HORIZONS:
        ics = []
        for d, grp in by_date.items():
            if len(grp) < MIN_NAMES:
                continue
            amb = [g["amb"] for g in grp]
            th = [g["theme_heat"] for g in grp]
            sw = [g["sw2_heat"] for g in grp]
            ret = [g["rets"][h] for g in grp]
            score = fn(amb, th, sw)
            ic = spearman(score, ret)
            if ic is not None:
                ics.append(ic)
        m = np.mean(ics) if ics else float("nan")
        sd = np.std(ics, ddof=1) if len(ics) > 1 else float("nan")
        t = m / sd * math.sqrt(len(ics)) if sd and not math.isnan(sd) else float("nan")
        results[(name, h)] = (m, t, len(ics))
        print(f"  {m:+.3f}({t:+.1f})", end="")
    print()
print("=" * 78)
print(f"截面数(各周期n_dates≈{len(by_date)}), 同口径样本 {len(full)}, 剔大盘<= {MAX_CAP}亿")
print("\n边际增量(相对S0):")
for name in ["S1 +拟合行业", "S2 +真实行业", "S3 +真实+拟合"]:
    deltas = [f"{h}d:{results[(name,h)][0]-results[('S0 原始',h)][0]:+.3f}" for h in HORIZONS]
    print(f"  {name:16} " + "  ".join(deltas))

# 也单独看行业热度自身的单因子IC
print("\n行业热度单因子 IC(自身预测力):")
for hk, label in [("theme_heat", "拟合行业热度"), ("sw2_heat", "真实行业热度")]:
    print(f"  {label:12}", end="")
    for h in HORIZONS:
        ics = []
        for d, grp in by_date.items():
            if len(grp) < MIN_NAMES:
                continue
            x = [g[hk] for g in grp]; ret = [g["rets"][h] for g in grp]
            ic = spearman(x, ret)
            if ic is not None:
                ics.append(ic)
        m = np.mean(ics); sd = np.std(ics, ddof=1) if len(ics) > 1 else float("nan")
        t = m / sd * math.sqrt(len(ics)) if sd else float("nan")
        print(f"  {m:+.3f}({t:+.1f})", end="")
    print()
