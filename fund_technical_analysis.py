"""
技术分析模块：基于历史净值计算多维度技术指标，生成买卖信号。

═══════════════════════════════════════════════════════════════
一、指标体系（6大评分指标 + 辅助展示指标）
═══════════════════════════════════════════════════════════════

  评分指标（参与综合评分）：
    1. MA均线交叉 (MA5/MA20)    — 金叉=买入(+分)，死叉=卖出(-分)
    2. RSI(14)                 — <RSI_OVERSOLD 超卖=买入，>RSI_OVERBOUGHT 超买=卖出
    3. MACD(12,26,9)           — DIF上穿DEA=买入，DIF下穿DEA=卖出
    4. KDJ交叉 + J值           — K上穿D=买入，K下穿D=卖出；J>100超买，J<0超卖
       (ADX>=25 趋势行情用 KDJ(KDJ_N_TREND,3,3)，ADX<25 震荡行情用 KDJ(KDJ_N_OSCILLATE,3,3))
    5. 布林带 (20日, 2倍标准差) — %B>BOLL_OVERBOUGHT 超买=卖出，%B<BOLL_OVERSOLD 超卖=买入
    6. MA60趋势方向             — 仅趋势市生效：价格>=MA60=多头(+TREND_60_SCORE)，价格<MA60=空头(-TREND_60_SCORE)；震荡市=0

  辅助展示指标（不参与评分）：
    7. ADX(14) 平均趋向指数     — 判断趋势/震荡，动态调整上述指标权重
    8. ATR(14) 真实波幅         — 展示波动率
    9. 历史净值百分位（近6年）   — 展示估值位置，同时用于过滤买卖信号
    10. 最大回撤（近5年）        — 展示风险
    11. MA120 趋势方向           — 展示长期趋势（多头/空头）

═══════════════════════════════════════════════════════════════
二、评分机制（ADX 动态权重）
═══════════════════════════════════════════════════════════════

  ADX >= 25（趋势行情）：
    MA/MACD 权重 × ADX_WEIGHT(1.5)，RSI/布林 权重 × 1，KDJ 固定 × 1，MA60趋势 固定 ±TREND_60_SCORE
  ADX < 25（震荡行情）：
    MA/MACD 权重 × 1，RSI/布林 权重 × ADX_WEIGHT(1.5)，KDJ 固定 × 1，MA60趋势 = 0

  各指标得分范围：
    MA: ±ADX_WEIGHT 或 ±1 | RSI: ±ADX_WEIGHT 或 ±1
    MACD: ±ADX_WEIGHT 或 ±1 | KDJ交叉: ±1 | KDJ J值: ±1
    布林: ±ADX_WEIGHT 或 ±1 | MA60趋势: ±TREND_60_SCORE（趋势市）或 0（震荡市）

  综合评分 = MA + RSI + MACD + KDJ交叉 + KDJ_J + 布林 + MA60趋势
  理论极值：趋势市 ±(7 + TREND_60_SCORE)，震荡市 ±7

═══════════════════════════════════════════════════════════════
三、买卖信号生成规则
═══════════════════════════════════════════════════════════════

  1. 阈值判定：综合评分 >= BUY_SIGNAL → 买入；<= SELL_SIGNAL → 卖出
  2. 百分位过滤：百分位 > BUY_PERCENTILE_CAP 时买入无效；
                 百分位 < SELL_PERCENTILE_FLOOR 时卖出无效
  3. 连续信号去重：
     - 连续买入：新买点净值须 <= 上次买入价 ×(1 - CONSEC_CHANGE_PCT)
     - 连续卖出：新卖点净值须 >= 上次卖出价 ×(1 + CONSEC_CHANGE_PCT)
  4. 强制止盈：净值 > 上次买入价 × FORCE_TAKE_PROFIT 时触发止盈信号

  百分位评分（辅助参考）：< PERCENTILE_LOW → +1 | > PERCENTILE_HIGH → -1
"""
import json
import math
import os


# ============================================================
# 可调参数（修改后 fund_generate_output.py 会自动引用）
# ============================================================
ADX_WEIGHT = 1.5              # ADX 动态权重倍数
BUY_SIGNAL = 4              # 综合评分 >= 此值 → 买入信号
SELL_SIGNAL = -4            # 综合评分 <= 此值 → 卖出信号
CONSEC_CHANGE_PCT = 0.03    # 连续买卖去重：变动须超过此比例（3%）才再标记
FORCE_TAKE_PROFIT = 1.25    # 净值涨超上次买入价 × 此值 → 强制止盈（涨25%）
BUY_PERCENTILE_CAP = 90     # 百分位 > 此值时，买入信号无效
SELL_PERCENTILE_FLOOR = 10  # 百分位 < 此值时，卖出信号无效
RSI_OVERBOUGHT = 70         # RSI 超买阈值
RSI_OVERSOLD = 30           # RSI 超卖阈值
KDJ_J_OVERBOUGHT = 100      # KDJ J值超买
KDJ_J_OVERSOLD = 0          # KDJ J值超卖
BOLL_OVERBOUGHT = 0.8       # 布林 %B 超买
BOLL_OVERSOLD = 0.2         # 布林 %B 超卖
KDJ_N_TREND = 14             # KDJ 趋势行情(ADX>=25)的 N 参数
KDJ_N_OSCILLATE = 9          # KDJ 震荡行情(ADX<25)的 N 参数
TREND_60_ENABLED = True       # 是否启用 MA60 趋势评分
TREND_60_SCORE = 2            # MA60 趋势评分的分值（趋势市±此值，震荡市=0）
FORCE_TAKE_PROFIT_ENABLED = True  # 是否启用强制止盈信号
PERCENTILE_LOW = 5           # 百分位评分：< 此值 → +0.5分（低估）
PERCENTILE_HIGH = 95         # 百分位评分：> 此值 → -0.5分（高估）
CHART_LOOKBACK_DAYS = 1250   # 走势图 / 最大回撤回看天数（不建议修改）
PERCENTILE_LOOKBACK_DAYS = 1500  # 历史百分位回看天数（不建议修改）


# ============================================================
# 基础计算函数
# ============================================================

def calc_ma(navs, period):
    """简单移动平均线，有滞后性，震荡行情频繁假信号，适合趋势行情。"""
    result = [None] * len(navs)
    for i in range(period - 1, len(navs)):
        result[i] = sum(navs[i - period + 1:i + 1]) / period
    return result


def calc_ema(values, period):
    """指数移动平均线"""
    result = [None] * len(values)
    k = 2 / (period + 1)
    first_valid = None
    for i, v in enumerate(values):
        if v is not None and first_valid is None:
            first_valid = i
    if first_valid is None:
        return result
    result[first_valid] = values[first_valid]
    for i in range(first_valid + 1, len(values)):
        if values[i] is not None and result[i - 1] is not None:
            result[i] = values[i] * k + result[i - 1] * (1 - k)
        else:
            result[i] = result[i - 1]
    return result


def latest_valid(series):
    """返回序列中最后一个非 None 值，没有则返回 None"""
    for v in reversed(series):
        if v is not None:
            return v
    return None


# ============================================================
# 技术指标
# ============================================================

def calc_rsi(navs, period=14):
    """RSI 相对强弱指标，强趋势时可能长期超卖/超买，适合短线震荡行情。"""
    if len(navs) < period + 1:
        return [None] * len(navs)

    changes = [0] + [navs[i] - navs[i - 1] for i in range(1, len(navs))]
    result = [None] * len(navs)

    gains = [max(c, 0) for c in changes[1:period + 1]]
    losses = [max(-c, 0) for c in changes[1:period + 1]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        result[period] = 100 - 100 / (1 + avg_gain / avg_loss)

    for i in range(period + 1, len(navs)):
        gain = max(changes[i], 0)
        loss = max(-changes[i], 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            result[i] = 100 - 100 / (1 + avg_gain / avg_loss)

    return result


def calc_macd(navs, fast=12, slow=26, signal=9):
    """MACD 指标，返回 (dif, dea, macd_hist)，背离信号可靠性高，有滞后性，震荡行情假信号多，适合趋势行情。"""
    ema_fast = calc_ema(navs, fast)
    ema_slow = calc_ema(navs, slow)

    dif = [None] * len(navs)
    for i in range(len(navs)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            dif[i] = ema_fast[i] - ema_slow[i]

    dea = calc_ema(dif, signal)

    macd_hist = [None] * len(navs)
    for i in range(len(navs)):
        if dif[i] is not None and dea[i] is not None:
            macd_hist[i] = 2 * (dif[i] - dea[i])

    return dif, dea, macd_hist


def calc_kdj(navs, n=9, m1=3, m2=3):
    """KDJ 随机指标，返回 (K, D, J)，信号灵敏，但容易过于灵敏，适合超短线、震荡行情。"""
    length = len(navs)
    k_list = [None] * length
    d_list = [None] * length
    j_list = [None] * length

    if length < n:
        return k_list, d_list, j_list

    prev_k = 50.0
    prev_d = 50.0

    for i in range(n - 1, length):
        window = navs[i - n + 1:i + 1]
        low_n = min(window)
        high_n = max(window)
        if high_n == low_n:
            rsv = 50.0
        else:
            rsv = (navs[i] - low_n) / (high_n - low_n) * 100

        k = (m1 - 1) / m1 * prev_k + 1 / m1 * rsv
        d = (m2 - 1) / m2 * prev_d + 1 / m2 * k
        j = 3 * k - 2 * d

        k_list[i] = round(k, 2)
        d_list[i] = round(d, 2)
        j_list[i] = round(j, 2)

        prev_k = k
        prev_d = d

    return k_list, d_list, j_list


def calc_bollinger(navs, period=20, num_std=2):
    """布林带，返回 (upper, middle, lower, percent_b)，对无效波动不敏感，普适性强"""
    length = len(navs)
    upper = [None] * length
    middle = [None] * length
    lower = [None] * length
    pct_b = [None] * length

    for i in range(period - 1, length):
        window = navs[i - period + 1:i + 1]
        ma = sum(window) / period
        std = math.sqrt(sum((x - ma) ** 2 for x in window) / period)
        middle[i] = ma
        upper[i] = ma + num_std * std
        lower[i] = ma - num_std * std
        band_width = upper[i] - lower[i]
        if band_width > 0:
            pct_b[i] = round((navs[i] - lower[i]) / band_width, 4)
        else:
            pct_b[i] = 0.5

    return upper, middle, lower, pct_b


def calc_max_drawdown(navs, lookback=365):
    """近 lookback 日最大回撤（百分比）"""
    recent = navs[-lookback:] if len(navs) >= lookback else navs
    if len(recent) < 2:
        return 0.0
    peak = recent[0]
    max_dd = 0.0
    for v in recent:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    return round(max_dd * 100, 2)


def calc_atr(navs, period=14):
    """ATR 真实波幅（基金净值仅有收盘价，TR 简化为日间变动绝对值）"""
    length = len(navs)
    if length < 2:
        return [None] * length
    tr = [None] * length
    for i in range(1, length):
        tr[i] = abs(navs[i] - navs[i - 1])
    atr = [None] * length
    if length < period + 1:
        return atr
    atr[period] = sum(v for v in tr[1:period + 1] if v is not None) / period
    for i in range(period + 1, length):
        if atr[i - 1] is not None and tr[i] is not None:
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def calc_adx(navs, period=14):
    """ADX 平均趋向指数（基金净值仅有收盘价，用日间变动近似 +DM/-DM）"""
    length = len(navs)
    if length < period * 2 + 1:
        return [None] * length

    plus_dm = [0.0] * length
    minus_dm = [0.0] * length
    tr = [0.0] * length
    for i in range(1, length):
        up = navs[i] - navs[i - 1]
        down = navs[i - 1] - navs[i]
        plus_dm[i] = up if up > 0 and up > down else 0.0
        minus_dm[i] = down if down > 0 and down > up else 0.0
        tr[i] = abs(navs[i] - navs[i - 1])

    atr_s = sum(tr[1:period + 1])
    plus_dm_s = sum(plus_dm[1:period + 1])
    minus_dm_s = sum(minus_dm[1:period + 1])

    adx_list = [None] * length
    dx_vals = []

    for i in range(period, length):
        if i == period:
            pass
        else:
            atr_s = atr_s - atr_s / period + tr[i]
            plus_dm_s = plus_dm_s - plus_dm_s / period + plus_dm[i]
            minus_dm_s = minus_dm_s - minus_dm_s / period + minus_dm[i]

        if atr_s == 0:
            dx_vals.append(0.0)
            continue
        plus_di = plus_dm_s / atr_s * 100
        minus_di = minus_dm_s / atr_s * 100
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx_vals.append(0.0)
        else:
            dx_vals.append(abs(plus_di - minus_di) / di_sum * 100)

        if len(dx_vals) == period:
            adx_list[i] = sum(dx_vals) / period
        elif len(dx_vals) > period:
            adx_list[i] = (adx_list[i - 1] * (period - 1) + dx_vals[-1]) / period

    return adx_list


# ============================================================
# 信号判断
# ============================================================

def get_cross_signal(short_line, long_line):
    """检测最近交叉：金叉=买入，死叉=卖出"""
    for i in range(len(short_line) - 1, 0, -1):
        if short_line[i] is None or long_line[i] is None:
            continue
        if short_line[i - 1] is None or long_line[i - 1] is None:
            continue
        diff_now = short_line[i] - long_line[i]
        diff_prev = short_line[i - 1] - long_line[i - 1]
        if diff_prev <= 0 < diff_now:
            return '买入', i
        if diff_prev >= 0 > diff_now:
            return '卖出', i
    return '持有', -1


def get_trend(navs, ma_line):
    """判断趋势：价格在均线上方=多头，下方=空头"""
    for i in range(len(navs) - 1, -1, -1):
        if ma_line[i] is not None:
            return '多头' if navs[i] >= ma_line[i] else '空头'
    return '未知'


def signal_to_score(signal):
    """信号转分数"""
    if signal == '买入':
        return 1
    elif signal == '卖出':
        return -1
    return 0


# ============================================================
# 单点综合评分（消除重复逻辑的核心函数）
# ============================================================

def calc_point_score(gi, navs, ma60, ma5, ma20, rsi, dif, dea,
                     k_trend, d_trend, j_trend,
                     k_osc, d_osc, j_osc,
                     pct_b, adx, adx_weight, detail=False):
    """计算单个时间点 gi 的综合评分。

    detail=False: 返回 float（总分），用于每日标记循环
    detail=True:  返回 dict，包含各指标分项得分，用于最终展示

    权重规则：
      - ADX >= 25 (趋势): MA/MACD ×adx_weight, RSI/布林 ×1, KDJ ×1, MA60趋势 固定±TREND_60_SCORE
      - ADX < 25  (震荡): MA/MACD ×1, RSI/布林 ×adx_weight, KDJ ×1, MA60趋势 = 0
    """
    adx_i = adx[gi] if gi < len(adx) else None
    is_trend = adx_i is not None and adx_i >= 25

    if is_trend:
        tw, ow = adx_weight, 1
    else:
        tw, ow = 1, adx_weight

    ma_score = 0
    rsi_score = 0
    macd_score = 0
    kdj_cross_score = 0
    kdj_j_score = 0
    boll_score = 0

    # MA5/20 交叉（趋势类）
    if (ma5[gi] is not None and ma20[gi] is not None
            and ma5[gi - 1] is not None and ma20[gi - 1] is not None):
        diff_now = ma5[gi] - ma20[gi]
        diff_prev = ma5[gi - 1] - ma20[gi - 1]
        if diff_prev <= 0 < diff_now:
            ma_score = tw
        elif diff_prev >= 0 > diff_now:
            ma_score = -tw

    # RSI（震荡类）
    if rsi[gi] is not None:
        if rsi[gi] > RSI_OVERBOUGHT:
            rsi_score = -ow
        elif rsi[gi] < RSI_OVERSOLD:
            rsi_score = ow

    # MACD（趋势类）
    if (dif[gi] is not None and dea[gi] is not None
            and dif[gi - 1] is not None and dea[gi - 1] is not None):
        md_now = dif[gi] - dea[gi]
        md_prev = dif[gi - 1] - dea[gi - 1]
        if md_prev <= 0 < md_now:
            macd_score = tw
        elif md_prev >= 0 > md_now:
            macd_score = -tw

    # KDJ（根据ADX选参数，固定权重×1）
    ki, di, ji = (k_trend, d_trend, j_trend) if is_trend else (k_osc, d_osc, j_osc)
    if (ki[gi] is not None and di[gi] is not None
            and ki[gi - 1] is not None and di[gi - 1] is not None):
        kd_now = ki[gi] - di[gi]
        kd_prev = ki[gi - 1] - di[gi - 1]
        if kd_prev <= 0 < kd_now:
            kdj_cross_score = 1
        elif kd_prev >= 0 > kd_now:
            kdj_cross_score = -1
        if ji[gi] is not None:
            if ji[gi] > KDJ_J_OVERBOUGHT:
                kdj_j_score = -1
            elif ji[gi] < KDJ_J_OVERSOLD:
                kdj_j_score = 1

    # 布林带
    if pct_b[gi] is not None:
        if pct_b[gi] > BOLL_OVERBOUGHT:
            boll_score = -ow
        elif pct_b[gi] < BOLL_OVERSOLD:
            boll_score = ow

    # MA60 趋势方向（趋势市±TREND_60_SCORE，震荡市=0）
    trend_60_score = 0
    if TREND_60_ENABLED and ma60[gi] is not None and is_trend:
        if navs[gi] >= ma60[gi]:
            trend_60_score = TREND_60_SCORE
        else:
            trend_60_score = -TREND_60_SCORE

    total = ma_score + rsi_score + macd_score + kdj_cross_score + kdj_j_score + boll_score + trend_60_score

    if not detail:
        return total

    return {
        'total': total,
        'ma': ma_score,
        'rsi': rsi_score,
        'macd': macd_score,
        'kdj_cross': kdj_cross_score,
        'kdj_j': kdj_j_score,
        'boll': boll_score,
        'trend_60': trend_60_score,
    }


# ============================================================
# 核心分析
# ============================================================

def analyze_fund(nav_records, estimate=None):
    """分析单只基金，返回完整信号字典。estimate: 实时估算数据 {gsz, gszzl, gztime}"""
    records = sorted(nav_records, key=lambda x: x['date'])

    navs = []
    dates = []
    for r in records:
        try:
            navs.append(float(r['nav_acc']))
            dates.append(r['date'])
        except (ValueError, TypeError):
            continue

    if len(navs) < 30:
        return None

    # --- 实时估算净值 ---
    estimated = False
    gztime = None
    if estimate and estimate.get('gszzl') and estimate.get('gztime'):
        try:
            gz_date = estimate['gztime'].split(' ')[0]
            if gz_date > dates[-1]:
                gszzl = float(estimate['gszzl'])
                estimated_nav = navs[-1] * (1 + gszzl / 100)
                navs.append(round(estimated_nav, 4))
                dates.append(gz_date)
                estimated = True
                gztime = estimate['gztime']
        except (ValueError, IndexError):
            pass

    # --- 指标计算 ---
    ma5 = calc_ma(navs, 5)
    ma20 = calc_ma(navs, 20)
    ma60 = calc_ma(navs, 60)
    ma120 = calc_ma(navs, 120)
    rsi = calc_rsi(navs, 14)
    dif, dea, macd_hist = calc_macd(navs)
    k_trend, d_trend, j_trend = calc_kdj(navs, n=KDJ_N_TREND)
    k_osc, d_osc, j_osc = calc_kdj(navs, n=KDJ_N_OSCILLATE)
    boll_upper, boll_middle, boll_lower, pct_b = calc_bollinger(navs)
    adx = calc_adx(navs, 14)
    atr = calc_atr(navs, 14)
    max_drawdown = calc_max_drawdown(navs, CHART_LOOKBACK_DAYS)

    # --- ADX / ATR 最新值 & 市场状态 ---
    latest_adx = latest_valid(adx)
    if latest_adx is not None:
        latest_adx = round(latest_adx, 1)

    latest_atr = latest_valid(atr)
    if latest_atr is not None:
        latest_atr = round(latest_atr, 6)
    atr_pct = round(latest_atr / navs[-1] * 100, 3) if latest_atr and navs[-1] else None

    is_trend = latest_adx is not None and latest_adx >= 25
    market_state = '趋势' if is_trend else '震荡'

    # --- 最新指标原始值（用于展示）---
    latest_rsi = latest_valid(rsi)
    if latest_rsi is not None:
        latest_rsi = round(latest_rsi, 2)

    if is_trend:
        k_line, d_line, j_line = k_trend, d_trend, j_trend
    else:
        k_line, d_line, j_line = k_osc, d_osc, j_osc
    latest_j = latest_valid(j_line)

    latest_pct_b = latest_valid(pct_b)
    if latest_pct_b is not None:
        latest_pct_b = round(latest_pct_b, 4)

    # 历史净值百分位（最6年）
    lookback_navs = navs[-PERCENTILE_LOOKBACK_DAYS:] if len(navs) > PERCENTILE_LOOKBACK_DAYS else navs
    current_nav = navs[-1]
    nav_low = min(lookback_navs)
    nav_high = max(lookback_navs)
    nav_percentile = round((current_nav - nav_low) / (nav_high - nav_low) * 100, 1)

    if nav_percentile < PERCENTILE_LOW:
        percentile_score = 1
    elif nav_percentile > PERCENTILE_HIGH:
        percentile_score = -1
    else:
        percentile_score = 0

    # 趋势方向
    trend_60 = get_trend(navs, ma60)
    trend_120 = get_trend(navs, ma120)

    # --- 综合评分（调用统一函数，detail=True 获取分项得分）---
    score_args = (navs, ma60, ma5, ma20, rsi, dif, dea,
                  k_trend, d_trend, j_trend,
                  k_osc, d_osc, j_osc,
                  pct_b, adx, ADX_WEIGHT)

    score_detail = calc_point_score(len(navs) - 1, *score_args, detail=True)
    score = score_detail['total']

    # 从分项得分推导信号
    def _score_to_signal(s):
        if s > 0:
            return '买入'
        elif s < 0:
            return '卖出'
        return '持有'

    ma_signal = _score_to_signal(score_detail['ma'])
    rsi_signal = _score_to_signal(score_detail['rsi'])
    macd_signal = _score_to_signal(score_detail['macd'])
    kdj_cross_signal = _score_to_signal(score_detail['kdj_cross'])
    j_signal = _score_to_signal(score_detail['kdj_j'])
    boll_signal = _score_to_signal(score_detail['boll'])
    trend_60_signal = _score_to_signal(score_detail['trend_60'])

    # --- 走势图数据 (近5年≈1250日) + 每日信号标记（含历史去重 & 百分位过滤）---
    recent_n = min(CHART_LOOKBACK_DAYS, len(navs))
    sl = slice(-recent_n, None)
    start_idx = len(navs) - recent_n

    buy_markers = []
    sell_markers = []
    force_sell_markers = []
    last_buy_nav = 0.0
    last_sell_nav = 0.0
    overall = '持有'
    filter_reason = ''
    is_force_sell = False

    for ci in range(recent_n):
        gi = start_idx + ci
        if gi < 120:
            continue
        # 计算该时刻的近6年百分位
        lb_start = max(0, gi - PERCENTILE_LOOKBACK_DAYS + 1)
        lb_navs = navs[lb_start:gi + 1]
        nav_lo = min(lb_navs)
        nav_hi = max(lb_navs)
        pt_pct = (navs[gi] - nav_lo) / (nav_hi - nav_lo) * 100 if nav_hi > nav_lo else 50.0
        s = calc_point_score(gi, *score_args)

        is_last = (gi == len(navs) - 1)
        point_signal = '持有'
        point_forced = False

        # ① 强制止盈
        if FORCE_TAKE_PROFIT_ENABLED and last_buy_nav != 0 and navs[gi] > last_buy_nav * FORCE_TAKE_PROFIT:
            point_forced = True
            # 强制卖出也算卖点，执行规则2流程
            last_buy_nav = 0.0
            last_sell_nav = navs[gi]
            force_sell_markers.append((ci, navs[gi]))
            point_signal = '卖出'

        # ② 普通买卖信号（非强制止盈时才判断）
        if not point_forced:
            if s >= BUY_SIGNAL:
                if pt_pct <= BUY_PERCENTILE_CAP:
                    last_sell_nav = 0.0
                    if last_buy_nav == 0:
                        last_buy_nav = navs[gi]
                        buy_markers.append((ci, navs[gi]))
                        point_signal = '买入'
                    else:
                        if navs[gi] <= last_buy_nav * (1 - CONSEC_CHANGE_PCT):
                            last_buy_nav = navs[gi]
                            buy_markers.append((ci, navs[gi]))
                            point_signal = '买入'
                        elif is_last:
                            filter_reason = f'连续买入去重(距上次买入跌幅不足{CONSEC_CHANGE_PCT*100:.0f}%)'
                elif is_last:
                    filter_reason = f'百分位过滤(当前{pt_pct:.1f}%>{BUY_PERCENTILE_CAP}%)'
            elif s <= SELL_SIGNAL:
                if pt_pct >= SELL_PERCENTILE_FLOOR:
                    last_buy_nav = 0.0
                    if last_sell_nav == 0:
                        last_sell_nav = navs[gi]
                        sell_markers.append((ci, navs[gi]))
                        point_signal = '卖出'
                    else:
                        if navs[gi] >= last_sell_nav * (1 + CONSEC_CHANGE_PCT):
                            last_sell_nav = navs[gi]
                            sell_markers.append((ci, navs[gi]))
                            point_signal = '卖出'
                        elif is_last:
                            filter_reason = f'连续卖出去重(距上次卖出涨幅不足{CONSEC_CHANGE_PCT*100:.0f}%)'
                elif is_last:
                    filter_reason = f'百分位过滤(当前{pt_pct:.1f}%<{SELL_PERCENTILE_FLOOR}%)'

        if is_last:
            overall = point_signal
            is_force_sell = point_forced and point_signal == '卖出'

    # --- 最新指标值 ---
    latest_dif = latest_dea = latest_macd_val = None
    for i in range(len(dif) - 1, -1, -1):
        if dif[i] is not None:
            latest_dif = round(dif[i], 4)
            latest_dea = round(dea[i], 4) if dea[i] is not None else None
            latest_macd_val = round(macd_hist[i], 4) if macd_hist[i] is not None else None
            break

    latest_k = latest_valid(k_line)
    latest_d = latest_valid(d_line)

    return {
        'ma_signal': ma_signal,
        'ma_score': score_detail['ma'],
        'rsi_signal': rsi_signal,
        'rsi_value': latest_rsi,
        'rsi_score': score_detail['rsi'],
        'macd_signal': macd_signal,
        'macd_score': score_detail['macd'],
        'dif': latest_dif,
        'dea': latest_dea,
        'macd_hist': latest_macd_val,
        'kdj_cross_signal': kdj_cross_signal,
        'kdj_cross_score': score_detail['kdj_cross'],
        'j_signal': j_signal,
        'j_score': score_detail['kdj_j'],
        'j_value': latest_j,
        'k_value': latest_k,
        'd_value': latest_d,
        'boll_signal': boll_signal,
        'boll_score': score_detail['boll'],
        'trend_60_signal': trend_60_signal,
        'trend_60_score': score_detail['trend_60'],
        'pct_b': latest_pct_b,
        'adx_value': latest_adx,
        'atr_pct': atr_pct,
        'market_state': market_state,
        'nav_percentile': nav_percentile,
        'percentile_score': percentile_score,
        'trend_60': trend_60,
        'trend_120': trend_120,
        'max_drawdown': max_drawdown,
        'score': score,
        'overall': overall,
        'filter_reason': filter_reason,
        'recent_navs': navs[sl],
        'recent_dates': dates[sl],
        'recent_ma5': ma5[sl],
        'recent_ma20': ma20[sl],
        'recent_ma60': ma60[sl],
        'recent_boll_upper': boll_upper[sl],
        'recent_boll_lower': boll_lower[sl],
        'buy_markers': buy_markers,
        'sell_markers': sell_markers,
        'force_sell_markers': force_sell_markers,
        'is_force_sell': is_force_sell,
        'latest_nav': navs[-1],
        'latest_date': dates[-1],
        'estimated': estimated,
        'gztime': gztime,
    }


def main():
    with open('data/fund_codes.json', encoding='utf-8') as f:
        fund_codes = set(json.load(f))

    with open('data/nav_history.json', encoding='utf-8') as f:
        history = json.load(f)

    # 加载实时估算数据（可选）
    estimate_data = {}
    estimate_file = os.path.join('data', 'realtime_estimate.json')
    if os.path.exists(estimate_file):
        with open(estimate_file, encoding='utf-8') as f:
            estimate_data = json.load(f)
        print(f'已加载 {len(estimate_data)} 只基金的实时估算数据')

    signals = {}
    for code in fund_codes:
        records = history.get(code)
        if not records:
            print(f'  [SKIP] {code}: 无历史净值数据')
            continue
        result = analyze_fund(records, estimate=estimate_data.get(code))
        if result is not None:
            clean = {}
            for k, v in result.items():
                if isinstance(v, list):
                    clean[k] = [x if x is not None else 'null' for x in v]
                else:
                    clean[k] = v
            signals[code] = clean

    with open('data/signals.json', 'w', encoding='utf-8') as f:
        json.dump(signals, f, ensure_ascii=False, indent=2)

    print(f'技术分析完成，共分析 {len(signals)} 只基金，结果已写入 data/signals.json')
    print(f'{"代码":>8}  {"MA":>4} {"RSI":>4} {"MACD":>4} {"KDJ":>4} {"布林":>4} {"趋60":>4} {"百分位":>6}  '
          f'{"ADX":>5} {"状态":>4} {"ATR%":>6}  '
          f'{"评分":>4}  {"建议":>4}  {"回撤":>6}  {"趋势60":>6} {"趋势120":>6}')
    print('-' * 130)
    for code, sig in signals.items():
        adx_str = f'{sig["adx_value"]:.1f}' if sig.get("adx_value") is not None else '--'
        atr_str = f'{sig["atr_pct"]:.3f}' if sig.get("atr_pct") is not None else '--'
        print(f'{code:>8}  '
              f'{sig["ma_signal"]:>4} '
              f'{sig["rsi_signal"]:>4} '
              f'{sig["macd_signal"]:>4} '
              f'{sig["kdj_cross_signal"]:>4} '
              f'{sig["boll_signal"]:>4} '
              f'{sig["trend_60_signal"]:>4} '
              f'{sig["nav_percentile"]:>5.1f}%  '
              f'{adx_str:>5} '
              f'{sig["market_state"]:>4} '
              f'{atr_str:>6}  '
              f'{sig["score"]:>+5.1f}  '
              f'{sig["overall"]:>4}  '
              f'{sig["max_drawdown"]:>5.1f}%  '
              f'{sig["trend_60"]:>6} '
              f'{sig["trend_120"]:>6}')


if __name__ == '__main__':
    main()
