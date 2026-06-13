import json
import os
import html
import warnings
from datetime import datetime
from funds import get_funds, get_funds_bond
from fund_technical_analysis import (
    ADX_WEIGHT, BUY_SIGNAL, SELL_SIGNAL,
    CONSEC_CHANGE_PCT, FORCE_TAKE_PROFIT,
    BUY_PERCENTILE_CAP, SELL_PERCENTILE_FLOOR,
    TREND_60_ENABLED, TREND_60_SCORE,
    FORCE_TAKE_PROFIT_ENABLED,
)


PERIOD_KEYS = [
    'netAssetValueRestoredGrowthRateRecentWeek',
    'netAssetValueRestoredGrowthRateRecentMonth',
    'netAssetValueRestoredGrowthRateRecentThreeMonth',
    'netAssetValueRestoredGrowthRateRecentSixMonth',
    'netAssetValueRestoredGrowthRateRecentOneYear',
    'netAssetValueRestoredGrowthRateRecentTwoYear',
    'netAssetValueRestoredGrowthRateRecentThreeYear',
    'netAssetValueRestoredGrowthRateRecentFiveYear',
]

PERIOD_LABELS = ['近一周', '1周~1月', '1月~3月', '3月~6月', '6月~1年', '1年~2年', '2年~3年', '3年~5年']
TOT_METRIC = 8
MISSING_VALUE = '--'

# 超额收益高亮阈值，与 PERIOD_LABELS 一一对应（8个周期）
EQUITY_HIGHLIGHT_RED = [1.5, 3, 5, 7, 12, 20, 20, 25]
EQUITY_HIGHLIGHT_GREEN = [-1.5, -3, -4.5, -6, -10, -15, -15, -20]
BOND_HIGHLIGHT_RED = [0.04, 0.1, 0.3, 0.4, 0.75, 1.5, 1.5, 2.0]
BOND_HIGHLIGHT_GREEN = [-0.03, 0, -0.1, -0.15, -0.25, -0.5, -0.5, -0.75]

# 综合评分理论极值：趋势市 MA/MACD ×ADX_WEIGHT + RSI/布林/KDJ交叉/J值 ±1×4 + MA60趋势
SCORE_RANGE = 2 * ADX_WEIGHT + 4 + (TREND_60_SCORE if TREND_60_ENABLED else 0)


def esc(value):
    return html.escape(str(value), quote=True)


def parse_percent(val):
    if val is None:
        raise ValueError('percent value is missing')
    text = str(val).strip()
    if text in ('', '--', '---'):
        raise ValueError('percent value is empty')
    if text.endswith('%'):
        text = text[:-1]
    return float(text)


def _parse_percent_or_missing(json_file, key, code):
    try:
        return parse_percent(json_file[key])
    except KeyError:
        warnings.warn(f'{code}: 缺少字段 {key}', RuntimeWarning)
    except (TypeError, ValueError) as exc:
        warnings.warn(f'{code}: 字段 {key} 不是有效百分比: {json_file.get(key)!r} ({exc})', RuntimeWarning)
    return MISSING_VALUE


def _safe_asset_class(total_asset):
    try:
        asset_val = float(total_asset)
    except (TypeError, ValueError):
        return ''
    if asset_val <= 100:
        return 'red'
    if asset_val > 300:
        return 'green'
    return ''


def process_item(json_file):
    code = json_file['fundCode']
    name = json_file['name']
    # 抓取失败时爬虫填空字符串（旧数据可能是哨兵 99999），统一占位
    total_asset = json_file.get('fund_manager_total_asset') or MISSING_VALUE
    if total_asset == 99999 or total_asset == '99999':
        total_asset = MISSING_VALUE

    week_val = _parse_percent_or_missing(json_file, PERIOD_KEYS[0], code)
    week = round(week_val, 2) if isinstance(week_val, (int, float)) else MISSING_VALUE
    increments = [week]

    prev = week + 100 if isinstance(week, (int, float)) else None
    for key in PERIOD_KEYS[1:]:
        cur_val = _parse_percent_or_missing(json_file, key, code)
        if isinstance(cur_val, (int, float)) and prev is not None:
            cur = cur_val + 100
            increments.append(round(cur / prev * 100 - 100, 2))
            prev = cur
        else:
            increments.append(MISSING_VALUE)
            if isinstance(cur_val, (int, float)):
                prev = cur_val + 100

    return (code, name, *increments, total_asset, json_file.get('managerTrigger', ''))


def detect_manager_changes(fund_list, tmp_data):
    """检测任职时长 < 20 天的基金经理（视为新近变更）。

    天天基金的任职时长格式为 "N天" 或 "X年又N天"：含"年"的一律视为老经理，
    避免整年数（如"2年"）被误解析成 2 天。
    """
    changes = []
    for item in fund_list:
        if item not in tmp_data:
            continue
        trigger = str(tmp_data[item][-1])
        if '年' in trigger or not trigger.endswith('天'):
            continue
        try:
            if int(trigger[:-1]) < 20:
                changes.append((tmp_data[item][0], tmp_data[item][1]))
        except ValueError:
            pass
    return changes


def compute_excess_table(fund_list, compare_index, tmp_data, highlight_red, highlight_green, hold_index=None):
    hold_index = hold_index or []
    rows = []
    for item in fund_list:
        if item not in tmp_data:
            warnings.warn(f'{item}: temp.json 中缺少该基金，HTML 报告中以 -- 占位', RuntimeWarning)
            rows.append({
                'code': item,
                'name': MISSING_VALUE,
                'total_asset': MISSING_VALUE,
                'is_held': item in hold_index,
                'manager_trigger': '',
                'asset_class': '',
                'benchmarks': [[(MISSING_VALUE, '') for _ in range(TOT_METRIC)] for _ in compare_index],
            })
            continue
        row = {
            'code': item,
            'name': tmp_data[item][1],
            'total_asset': tmp_data[item][-2],
            'is_held': item in hold_index,
            'manager_trigger': tmp_data[item][-1],
            'benchmarks': [],
        }
        row['asset_class'] = _safe_asset_class(tmp_data[item][-2])

        for index in compare_index:
            cells = []
            for i in range(TOT_METRIC):
                try:
                    value = round(tmp_data[item][i + 2] - tmp_data[index][i + 2], 2)
                    css = ''
                    if i < len(highlight_green) and value < highlight_green[i]:
                        css = 'green'
                    if i < len(highlight_red) and value > highlight_red[i]:
                        css = 'red'
                    cells.append((f'{value:.2f}', css))
                except (TypeError, KeyError):
                    cells.append((MISSING_VALUE, ''))
            row['benchmarks'].append(cells)
        rows.append(row)
    return rows


# ============================================================
# SVG 走势图（250日 + MA5/MA20/MA60 + 布林带）
# ============================================================

def render_sparkline_svg(sig, width=650, height=140):
    """生成带布林带、MA60、买卖标记的大走势图"""
    navs = sig.get('recent_navs', [])
    ma5 = sig.get('recent_ma5', [])
    ma20 = sig.get('recent_ma20', [])
    ma60 = sig.get('recent_ma60', [])
    boll_upper = sig.get('recent_boll_upper', [])
    boll_lower = sig.get('recent_boll_lower', [])
    dates = sig.get('recent_dates', [])
    buy_markers = sig.get('buy_markers', [])
    sell_markers = sig.get('sell_markers', [])
    force_sell_markers = sig.get('force_sell_markers', [])

    valid_navs = [v for v in navs if v is not None and v != 'null']
    if len(valid_navs) < 2:
        return ''

    # 收集所有有效值来确定Y轴范围
    all_vals = list(valid_navs)
    for series in [boll_upper, boll_lower]:
        for v in series:
            if v is not None and v != 'null':
                all_vals.append(v)

    pad = 8
    chart_w = width - 2 * pad
    chart_h = height - 2 * pad
    nav_min = min(all_vals)
    nav_max = max(all_vals)
    nav_range = nav_max - nav_min
    if nav_range == 0:
        nav_range = 1

    n = len(navs)

    def to_xy(i, v):
        x = pad + i / (n - 1) * chart_w if n > 1 else pad
        y = pad + chart_h - (v - nav_min) / nav_range * chart_h
        return x, y

    def polyline_str(series):
        pts = []
        for i, v in enumerate(series):
            if v is not None and v != 'null':
                x, y = to_xy(i, v)
                pts.append(f'{x:.1f},{y:.1f}')
        return ' '.join(pts)

    svg = f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'

    # 布林带填充区域
    upper_pts = []
    lower_pts = []
    for i in range(n):
        u = boll_upper[i] if i < len(boll_upper) else None
        l = boll_lower[i] if i < len(boll_lower) else None
        if u is not None and u != 'null' and l is not None and l != 'null':
            ux, uy = to_xy(i, u)
            lx, ly = to_xy(i, l)
            upper_pts.append(f'{ux:.1f},{uy:.1f}')
            lower_pts.append(f'{lx:.1f},{ly:.1f}')
    if upper_pts and lower_pts:
        polygon = ' '.join(upper_pts) + ' ' + ' '.join(reversed(lower_pts))
        svg += f'<polygon points="{polygon}" fill="#e6f0fa" fill-opacity="0.5" stroke="none" />'

    # 均线和净值
    svg += f'<polyline points="{polyline_str(navs)}" fill="none" stroke="#1890ff" stroke-width="1.5" />'
    ma5_str = polyline_str(ma5)
    if ma5_str:
        svg += f'<polyline points="{ma5_str}" fill="none" stroke="#faad14" stroke-width="1" stroke-dasharray="3,2" />'
    ma20_str = polyline_str(ma20)
    if ma20_str:
        svg += f'<polyline points="{ma20_str}" fill="none" stroke="#f5222d" stroke-width="1" stroke-dasharray="5,3" />'
    ma60_str = polyline_str(ma60)
    if ma60_str:
        svg += f'<polyline points="{ma60_str}" fill="none" stroke="#722ed1" stroke-width="1" stroke-dasharray="8,4" />'

    # X轴月份刻度（根据数据跨度自动稀疏，避免重叠）
    all_months = []
    seen_months = set()
    for i, d in enumerate(dates):
        if not d or d == 'null':
            continue
        month_key = d[:7]
        if month_key not in seen_months:
            seen_months.add(month_key)
            all_months.append((i, d))

    # 根据总月数决定间隔：<=24月每月显示，<=60月每3月，>60月每6月
    total_months = len(all_months)
    if total_months <= 24:
        step = 1
    elif total_months <= 60:
        step = 3
    else:
        step = 6

    for idx, (i, d) in enumerate(all_months):
        x, _ = to_xy(i, nav_min)
        # 细网格线每月都画
        svg += f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{height - pad}" stroke="#e8e8e8" stroke-width="0.5" />'
        # 文字标签按间隔显示
        if idx % step == 0:
            month_num = d[5:7]
            label = f'{d[:4]}' if month_num == '01' else f'{month_num}月'
            svg += f'<text x="{x:.1f}" y="{height - 1}" text-anchor="middle" font-size="9" fill="#999">{label}</text>'

    # 买入标记（红色向上三角）
    for ci, nav_val in buy_markers:
        if nav_val is not None and nav_val != 'null':
            x, y = to_xy(ci, nav_val)
            svg += f'<polygon points="{x:.1f},{y + 3:.1f} {x - 4:.1f},{y + 11:.1f} {x + 4:.1f},{y + 11:.1f}" fill="#f5222d" opacity="0.85" />'

    # 卖出标记（绿色向下三角）
    for ci, nav_val in sell_markers:
        if nav_val is not None and nav_val != 'null':
            x, y = to_xy(ci, nav_val)
            svg += f'<polygon points="{x:.1f},{y - 3:.1f} {x - 4:.1f},{y - 11:.1f} {x + 4:.1f},{y - 11:.1f}" fill="#52c41a" opacity="0.85" />'

    # 止盈标记（紫色向下三角）
    for ci, nav_val in force_sell_markers:
        if nav_val is not None and nav_val != 'null':
            x, y = to_xy(ci, nav_val)
            svg += f'<polygon points="{x:.1f},{y - 3:.1f} {x - 4:.1f},{y - 11:.1f} {x + 4:.1f},{y - 11:.1f}" fill="#722ed1" opacity="0.85" />'

    svg += '</svg>'
    return svg


def render_rsi_bar(rsi_value):
    if rsi_value is None:
        return '<span class="rsi-na">N/A</span>'
    pct = max(0, min(100, rsi_value))
    if pct > 70:
        color = '#cf1322'
        label = '超买'
    elif pct < 30:
        color = '#389e0d'
        label = '超卖'
    else:
        color = '#1890ff'
        label = '中性'
    return (
        f'<div class="rsi-bar">'
        f'<div class="rsi-fill" style="width:{pct}%;background:{color}"></div>'
        f'<span class="rsi-label">{rsi_value} ({label})</span>'
        f'</div>'
    )


def render_signal_badge(signal, score=None):
    cls = 'badge-buy' if signal == '买入' else ('badge-sell' if signal == '卖出' else 'badge-hold')
    score_str = f' ({score:+g})' if score is not None and score != 0 else ''
    return f'<span class="signal-badge {cls}">{signal}{score_str}</span>'


def render_trend_badge(trend):
    if trend == '多头':
        return '<span class="trend-bull">多头</span>'
    elif trend == '空头':
        return '<span class="trend-bear">空头</span>'
    return '<span class="trend-neutral">未知</span>'


def render_percentile_bar(percentile):
    if percentile is None:
        return '<span class="rsi-na">N/A</span>'
    pct = max(0, min(100, percentile))
    if pct < 10:
        color = '#389e0d'
        label = '极低'
    elif pct < 20:
        color = '#52c41a'
        label = '偏低'
    elif pct > 90:
        color = '#cf1322'
        label = '极高'
    elif pct > 80:
        color = '#f5222d'
        label = '偏高'
    else:
        color = '#1890ff'
        label = '中性'
    return (
        f'<div class="rsi-bar">'
        f'<div class="rsi-fill" style="width:{pct}%;background:{color}"></div>'
        f'<span class="rsi-label">{percentile:.1f}% ({label})</span>'
        f'</div>'
    )


def render_adx_bar(adx_value, market_state):
    if adx_value is None:
        return '<span class="rsi-na">N/A</span>'
    pct = max(0, min(100, adx_value))
    if adx_value >= 25:
        color = '#1890ff'
        label = '趋势'
    else:
        color = '#faad14'
        label = '震荡'
    return (
        f'<div class="rsi-bar">'
        f'<div class="rsi-fill" style="width:{pct}%;background:{color}"></div>'
        f'<span class="rsi-label">{adx_value:.1f} ({label})</span>'
        f'</div>'
    )


def render_atr_display(atr_pct):
    if atr_pct is None:
        return '<span class="rsi-na">N/A</span>'
    if atr_pct > 2.0:
        cls = 'dd-high'
    elif atr_pct > 1.0:
        cls = 'dd-mid'
    else:
        cls = 'dd-low'
    return f'<span class="{cls}">{atr_pct:.3f}%</span>'


def render_score_bar(score, max_score=SCORE_RANGE):
    """评分条：范围取自评分理论极值，买卖阈值与技术分析参数联动"""
    pct = (score + max_score) / (2 * max_score) * 100
    pct = max(0, min(100, pct))
    if score >= BUY_SIGNAL:
        color = '#f5222d'
    elif score <= SELL_SIGNAL:
        color = '#52c41a'
    elif score > 0:
        color = '#ff7875'
    elif score < 0:
        color = '#95de64'
    else:
        color = '#8c8c8c'
    return (
        f'<div class="score-bar">'
        f'<div class="score-fill" style="width:{pct}%;background:{color}"></div>'
        f'<span class="score-label">{score:+.1f}</span>'
        f'</div>'
    )


def render_drawdown(dd):
    if dd is None:
        return '--'
    if dd > 15:
        cls = 'dd-high'
    elif dd > 8:
        cls = 'dd-mid'
    else:
        cls = 'dd-low'
    return f'<span class="{cls}">{dd:.1f}%</span>'


def generate_html(tmp_data, equity_config, bond_config, change_manager, signals=None):
    equity_compare = equity_config['compare_index']
    bond_compare = bond_config['compare_index']

    equity_rows = compute_excess_table(
        equity_config['fund'], equity_compare, tmp_data,
        EQUITY_HIGHLIGHT_RED, EQUITY_HIGHLIGHT_GREEN, equity_config.get('hold_index', []))
    bond_rows = compute_excess_table(
        bond_config['fund'], bond_compare, tmp_data,
        BOND_HIGHLIGHT_RED, BOND_HIGHLIGHT_GREEN)

    today = datetime.now().strftime('%Y-%m-%d %H:%M')
    signal_counts = {'buy': 0, 'sell': 0, 'hold': 0, 'total': 0}
    if signals:
        signal_counts['buy'] = sum(1 for s in signals.values() if s.get('overall') == '买入')
        signal_counts['sell'] = sum(1 for s in signals.values() if s.get('overall') == '卖出')
        signal_counts['hold'] = sum(1 for s in signals.values() if s.get('overall') == '持有')
        signal_counts['total'] = len(signals)

    summary_html = f'''
    <div class="stat-grid">
        <div class="stat-card">
            <span>股票型基金</span>
            <strong>{len(equity_rows)}</strong>
        </div>
        <div class="stat-card">
            <span>债券型基金</span>
            <strong>{len(bond_rows)}</strong>
        </div>
        <div class="stat-card">
            <span>技术信号</span>
            <strong>{signal_counts["total"]}</strong>
        </div>
        <div class="stat-card signal-stat">
            <span>买 / 卖 / 持有</span>
            <strong><b class="buy-text">{signal_counts["buy"]}</b> / <b class="sell-text">{signal_counts["sell"]}</b> / {signal_counts["hold"]}</strong>
        </div>
    </div>'''

    def render_table(title, compare_index, rows):
        benchmark_names = [tmp_data[idx][1] if idx in tmp_data else idx for idx in compare_index]
        n_benchmarks = len(compare_index)
        header1 = '<tr><th class="sticky-col sticky-code"></th><th class="sticky-col sticky-name"></th><th></th>'
        for name in benchmark_names:
            header1 += f'<th colspan="{TOT_METRIC}" class="benchmark-header">{esc(name)}</th>'
        header1 += '</tr>'

        header2 = '<tr><th class="sticky-col sticky-code">代码</th><th class="sticky-col sticky-name">名字</th><th>管理规模(亿)</th>'
        for _ in range(n_benchmarks):
            for label in PERIOD_LABELS:
                header2 += f'<th>{esc(label)}</th>'
        header2 += '</tr>'

        body = ''
        for row in rows:
            code_classes = ['sticky-col', 'sticky-code', 'code-cell']
            if row['is_held']:
                code_classes.append('held')
            asset_cls = f' class="{row["asset_class"]}"' if row['asset_class'] else ''
            body += '<tr>'
            body += f'<td class="{" ".join(code_classes)}">{esc(row["code"])}</td>'
            body += f'<td class="sticky-col sticky-name name-cell">{esc(row["name"])}</td>'
            body += f'<td{asset_cls}>{esc(row["total_asset"])}</td>'
            for cells in row['benchmarks']:
                for val, css in cells:
                    cls = f' class="{css}"' if css else ''
                    body += f'<td{cls}>{esc(val)}</td>'
            body += '</tr>\n'

        return f'''
        <section class="report-section">
            <div class="section-head">
                <h2>{title}</h2>
                <span>{len(rows)} 只基金 · {n_benchmarks} 个基准</span>
            </div>
            <div class="table-wrapper">
                <table class="metric-table">
                    <thead>{header1}{header2}</thead>
                    <tbody>{body}</tbody>
                </table>
            </div>
        </section>'''

    manager_html = ''
    if change_manager:
        items = ''.join(f'<li>{esc(code)} - {esc(name)}</li>' for code, name in change_manager)
        manager_html = f'<div class="status-banner alert">近20天内以下基金经理发生变更：<ul>{items}</ul></div>'
    else:
        manager_html = '<div class="status-banner ok">近20天内列表中基金经理没有变更。</div>'

    # 技术分析板块
    signal_html = ''
    if signals:
        all_fund_codes = equity_config['fund'] + bond_config['fund']
        signal_rows = ''
        for code in all_fund_codes:
            if code not in signals:
                continue
            sig = signals[code]
            name = tmp_data[code][1] if code in tmp_data else code

            sparkline = render_sparkline_svg(sig)

            j_val = sig.get('j_value', '')
            j_display = f'{j_val:.1f}' if isinstance(j_val, (int, float)) else '--'
            pct_b = sig.get('pct_b', '')
            pct_b_display = f'{pct_b:.2f}' if isinstance(pct_b, (int, float)) else '--'

            # 净值列：如果有估算数据，追加显示
            nav_display = esc(sig.get('latest_nav', ''))
            if sig.get('estimated') and sig.get('latest_nav'):
                nav_display += '<br><small class="estimate-label">估值</small>'

            # 建议列：追加估算时间
            gztime = sig.get('gztime', '')
            gztime_display = ''
            if gztime:
                # "2026-04-14 13:52" → "04-14 13:52"
                gztime_short = gztime[5:] if len(gztime) > 5 else gztime
                gztime_display = f'<br><small>估算: {esc(gztime_short)}</small>'

            ma_s = sig.get('ma_score', 0)
            rsi_s = sig.get('rsi_score', 0)
            macd_s = sig.get('macd_score', 0)
            kdj_cross_s = sig.get('kdj_cross_score', 0)
            j_s = sig.get('j_score', 0)
            boll_s = sig.get('boll_score', 0)

            force_note = '<br><small class="force-label">触发止盈</small>' if sig.get('is_force_sell') else ''
            filter_reason = sig.get('filter_reason', '')
            filter_note = f'<br><small class="filter-label">{esc(filter_reason)}</small>' if filter_reason else ''
            if sig.get('is_force_sell'):
                overall_badge = '<span class="signal-badge badge-force-sell">止盈信号</span>'
            else:
                overall_badge = render_signal_badge(sig['overall'])

            signal_rows += f'''<tr>
                <td class="code-cell">{esc(code)}</td>
                <td class="name-cell">{esc(name)}</td>
                <td>{nav_display}</td>
                <td class="chart-cell">{sparkline}</td>
                <td>{render_signal_badge(sig['ma_signal'], ma_s)}</td>
                <td>{render_rsi_bar(sig.get('rsi_value'))}<br><small>得分: {rsi_s:+g}</small></td>
                <td>{render_signal_badge(sig['macd_signal'], macd_s)}</td>
                <td>{render_signal_badge(sig.get('kdj_cross_signal', '持有'), kdj_cross_s)}<br>{render_signal_badge(sig.get('j_signal', '持有'), j_s)}<br><small>J={j_display}</small></td>
                <td>{render_signal_badge(sig.get('boll_signal', '持有'), boll_s)}<br><small>%B={pct_b_display}</small></td>
                <td>{render_adx_bar(sig.get('adx_value'), sig.get('market_state', ''))}</td>
                <td>{render_atr_display(sig.get('atr_pct'))}</td>
                <td>{render_percentile_bar(sig.get('nav_percentile'))}<br></td>
                <td>{render_trend_badge(sig.get('trend_60', '')) + '<br><small>得分: ' + (f'{sig.get("trend_60_score", 0):+g}' if sig.get('market_state') == '趋势' else '0 (非趋势市)') + '</small>' if TREND_60_ENABLED else '--'}</td> 
                <td>{render_score_bar(sig.get('score', 0))}{force_note}</td>
                <td>{overall_badge}{filter_note}{gztime_display}</td>
            </tr>\n'''

        signal_html = f'''
        <h2>技术分析 - 多维买卖信号</h2>
        <div class="legend">
            <span><svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="#1890ff" stroke-width="1.5"/></svg> 净值</span>
            <span><svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="#faad14" stroke-width="1" stroke-dasharray="3,2"/></svg> MA5</span>
            <span><svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="#f5222d" stroke-width="1" stroke-dasharray="5,3"/></svg> MA20</span>
            <span><svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="#722ed1" stroke-width="1" stroke-dasharray="8,4"/></svg> MA60</span>
            <span><svg width="20" height="10"><rect x="0" y="2" width="20" height="6" fill="#e6f0fa" opacity="0.7"/></svg> 布林带</span>
            <span><svg width="14" height="12"><polygon points="7,1 2,11 12,11" fill="#f5222d" opacity="0.85"/></svg> 买入信号</span>
            <span><svg width="14" height="12"><polygon points="7,11 2,1 12,1" fill="#52c41a" opacity="0.85"/></svg> 卖出信号</span>
            <span><svg width="14" height="12"><polygon points="7,11 2,1 12,1" fill="#722ed1" opacity="0.85"/></svg> 止盈信号</span>
        </div>
        <p class="signal-note">ADX&ge;25=趋势行情（MA/MACD权重&times;{ADX_WEIGHT}，MA60趋势&plusmn;{TREND_60_SCORE}），ADX&lt;25=震荡行情（RSI/布林权重&times;{ADX_WEIGHT}，MA60趋势=0），KDJ固定权重&times;1 | 综合建议: 评分 &ge;+{BUY_SIGNAL} 买入, &le;{SELL_SIGNAL} 卖出 | 买卖信号经历史去重(变动需超{round(CONSEC_CHANGE_PCT * 100)}%)、百分位过滤(买&gt;{BUY_PERCENTILE_CAP}%/卖&lt;{SELL_PERCENTILE_FLOOR}%无效) | {'净值涨幅&gt;' + str(round((FORCE_TAKE_PROFIT - 1) * 100)) + '%触发<span class="force-label">止盈信号</span>' if FORCE_TAKE_PROFIT_ENABLED else '止盈信号已关闭'} | ATR/百分位仅展示</p>
        <div class="table-wrapper">
        <table class="signal-table">
            <thead>
                <tr>
                    <th>代码</th><th>名字</th><th>净值</th><th>近5年走势</th>
                    <th>MA(5/20)</th><th>RSI(14)</th><th>MACD</th><th>KDJ(交叉/J值)</th><th>布林带</th>
                    <th>ADX(14)</th><th>ATR%</th><th>百分位（近6年）</th><th>趋势60</th>
                    <th>评分</th><th>建议</th>
                </tr>
            </thead>
            <tbody>{signal_rows}</tbody>
        </table>
        </div>'''

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>基金量化指标报告</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    :root {{
        --bg: #f4f6f8;
        --surface: #ffffff;
        --surface-soft: #f8fafc;
        --surface-tint: #eef6ff;
        --ink: #1f2937;
        --muted: #64748b;
        --line: #d8e0ea;
        --line-soft: #e8edf3;
        --blue: #2563eb;
        --cyan: #0e7490;
        --green: #15803d;
        --green-soft: #edf8f1;
        --red: #b91c1c;
        --red-soft: #fff1f1;
        --amber: #b45309;
        --amber-soft: #fff7ed;
        --purple: #6d28d9;
        --purple-soft: #f5f0ff;
        --shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
    }}
    body {{
        min-width: 1100px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
        background: var(--bg);
        color: var(--ink);
        padding: 0;
        font-size: 13px;
        line-height: 1.45;
    }}
    .page {{
        width: 90%;
        margin: 0 auto;
        padding: 0;
    }}
    .report-header {{
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 20px;
        padding: 2px 0 18px;
        border-bottom: 1px solid var(--line);
    }}
    .eyebrow {{
        margin-bottom: 5px;
        color: var(--cyan);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
    }}
    h1 {{
        margin: 0;
        font-size: 28px;
        line-height: 1.15;
        font-weight: 760;
        color: #172033;
    }}
    .date {{
        color: var(--muted);
        font-size: 12px;
        text-align: right;
        white-space: nowrap;
    }}
    .date strong {{
        display: block;
        margin-top: 4px;
        color: var(--ink);
        font-size: 15px;
        font-weight: 700;
    }}
    .stat-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(150px, 1fr));
        gap: 12px;
        margin: 18px 0;
    }}
    .stat-card {{
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--surface);
        padding: 13px 15px;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
    }}
    .stat-card span {{
        display: block;
        margin-bottom: 5px;
        color: var(--muted);
        font-size: 12px;
    }}
    .stat-card strong {{
        font-size: 23px;
        line-height: 1;
        font-weight: 760;
        color: var(--ink);
    }}
    .stat-card b {{
        font-weight: 760;
    }}
    .buy-text {{ color: var(--red); }}
    .sell-text {{ color: var(--green); }}
    .status-banner {{
        margin: 16px 0 22px;
        padding: 12px 14px;
        border-radius: 8px;
        font-weight: 650;
    }}
    .status-banner ul {{ margin: 8px 0 0 20px; font-weight: 500; }}
    .alert {{ background: var(--amber-soft); border: 1px solid #fed7aa; color: var(--amber); }}
    .ok {{ background: var(--green-soft); border: 1px solid #bbf7d0; color: var(--green); }}
    .report-section {{ margin-top: 24px; }}
    .section-head {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 14px;
        margin-bottom: 10px;
    }}
    h2 {{
        margin: 0;
        font-size: 18px;
        line-height: 1.25;
        color: #172033;
    }}
    .section-head span {{
        color: var(--muted);
        font-size: 12px;
        white-space: nowrap;
    }}
    .table-wrapper {{
        overflow: auto;
        max-height: 72vh;
        margin-bottom: 22px;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--surface);
        box-shadow: var(--shadow);
    }}
    table {{
        border-collapse: separate;
        border-spacing: 0;
        white-space: nowrap;
        min-width: 100%;
        font-variant-numeric: tabular-nums;
    }}
    th, td {{
        border-right: 1px solid var(--line-soft);
        border-bottom: 1px solid var(--line-soft);
        padding: 8px 10px;
        text-align: center;
        background: var(--surface);
    }}
    th:last-child, td:last-child {{ border-right: 0; }}
    thead th {{
        background: var(--surface-soft);
        color: #334155;
        font-size: 12px;
        font-weight: 700;
        position: sticky;
        z-index: 4;
    }}
    thead tr:first-child th {{ top: 0; }}
    thead tr:nth-child(2) th {{ top: 35px; }}
    .benchmark-header {{
        background: var(--surface-tint);
        color: var(--blue);
        border-bottom-color: #bfdbfe;
    }}
    tbody tr:nth-child(even) td {{ background: #fbfcfe; }}
    tbody tr:hover td {{ background: #f3f8ff; }}
    .sticky-col {{
        position: sticky;
        z-index: 3;
        background: var(--surface);
    }}
    thead .sticky-col {{
        z-index: 6;
        background: var(--surface-soft);
    }}
    tbody tr:nth-child(even) .sticky-col {{ background: #fbfcfe; }}
    tbody tr:hover .sticky-col {{ background: #f3f8ff; }}
    .sticky-code {{
        left: 0;
        width: 90px;
        min-width: 90px;
        max-width: 90px;
    }}
    .sticky-name {{
        left: 90px;
        width: 230px;
        min-width: 230px;
        max-width: 230px;
        text-align: left;
        box-shadow: 1px 0 0 var(--line);
    }}
    .code-cell {{ font-weight: 700; color: #334155; }}
    .name-cell {{
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    td.red {{
        background: var(--red-soft) !important;
        color: var(--red);
        font-weight: 700;
    }}
    td.green {{
        background: var(--green-soft) !important;
        color: var(--green);
        font-weight: 700;
    }}
    td.held {{
        background: var(--purple-soft) !important;
        color: var(--purple);
        font-weight: 800;
    }}

    /* 技术分析样式 */
    .signal-table {{ min-width: 1780px; }}
    .signal-table td {{ vertical-align: middle; }}
    .signal-table thead tr:first-child th {{ top: 0; }}
    .chart-cell {{ padding: 6px !important; }}
    .chart-cell svg {{ display: block; margin: 0 auto; }}
    .signal-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 46px;
        padding: 3px 8px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        color: #fff;
        margin: 1px;
    }}
    .badge-buy {{ background: var(--red); }}
    .badge-sell {{ background: var(--green); }}
    .badge-hold {{ background: #64748b; }}
    .badge-force-sell {{ background: var(--purple); }}
    .trend-bull {{ color: var(--red); font-weight: 700; font-size: 12px; }}
    .trend-bear {{ color: var(--green); font-weight: 700; font-size: 12px; }}
    .trend-neutral {{ color: var(--muted); font-size: 12px; }}
    .rsi-bar {{
        position: relative; width: 116px; height: 20px;
        background: #edf2f7; border-radius: 999px; overflow: hidden;
        border: 1px solid #d8e0ea;
        display: inline-block; vertical-align: middle;
    }}
    .rsi-fill {{ height: 100%; border-radius: 999px; opacity: 0.82; }}
    .rsi-label {{
        position: absolute; top: 0; left: 0; right: 0;
        text-align: center; font-size: 11px; line-height: 18px;
        color: #172033; font-weight: 650;
    }}
    .rsi-na {{ color: #94a3b8; font-size: 12px; }}
    .score-bar {{
        position: relative; width: 92px; height: 20px;
        background: #edf2f7; border-radius: 999px; overflow: hidden;
        border: 1px solid #d8e0ea;
        display: inline-block; vertical-align: middle;
    }}
    .score-fill {{ height: 100%; border-radius: 999px; opacity: 0.82; }}
    .score-label {{
        position: absolute; top: 0; left: 0; right: 0;
        text-align: center; font-size: 11px; line-height: 18px;
        color: #172033; font-weight: 750;
    }}
    .dd-high {{ color: var(--red); font-weight: 700; }}
    .dd-mid {{ color: var(--amber); font-weight: 650; }}
    .dd-low {{ color: var(--green); font-weight: 650; }}
    .legend {{
        margin: 10px 0;
        display: flex;
        gap: 12px 18px;
        align-items: center;
        font-size: 12px;
        color: var(--muted);
        flex-wrap: wrap;
    }}
    .legend span {{ display: flex; align-items: center; gap: 4px; }}
    .signal-note {{
        max-width: 1280px;
        font-size: 12px;
        color: var(--muted);
        margin: 4px 0 12px;
    }}
    .estimate-label {{ color: var(--blue); }}
    .force-label {{ color: var(--purple); font-weight: 700; }}
    .filter-label {{ color: var(--amber); }}
    small {{ color: var(--muted); font-size: 11px; }}
    @media (max-width: 900px) {{
        body {{ min-width: 760px; }}
        .page {{ width: 90%; padding: 0; }}
        .report-header {{ align-items: flex-start; flex-direction: column; }}
        .date {{ text-align: left; }}
        .stat-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        h1 {{ font-size: 24px; }}
    }}
</style>
</head>
<body>
<div class="page">
    <header class="report-header">
        <div>
            <p class="eyebrow">financial-analysis</p>
            <h1>基金量化指标报告</h1>
        </div>
        <p class="date">生成时间<strong>{today}</strong></p>
    </header>
    {summary_html}
    {manager_html}
    {render_table("股票型基金（中高风险）", equity_compare, equity_rows)}
    {render_table("债券型基金（中低风险）", bond_compare, bond_rows)}
    {signal_html}
</div>
</body>
</html>'''

    with open('fund_report.html', 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == '__main__':
    tmp_data = {}
    with open('data/temp.json', encoding='utf8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            item = process_item(data)
            tmp_data[item[0]] = item

    equity_config = get_funds()
    bond_config = get_funds_bond()

    change_manager = detect_manager_changes(equity_config['fund'], tmp_data)
    change_manager.extend(detect_manager_changes(bond_config['fund'], tmp_data))

    # 加载技术信号
    signals = None
    signals_file = 'data/signals.json'
    if os.path.exists(signals_file):
        with open(signals_file, encoding='utf-8') as f:
            signals = json.load(f)

    # 生成 HTML 报告
    generate_html(tmp_data, equity_config, bond_config, change_manager, signals)

    print('*' * 30, 'result', '*' * 30)
    print('处理完毕，文件 fund_report.html 已输出。')
    if signals:
        buy_count = sum(1 for s in signals.values() if s['overall'] == '买入')
        sell_count = sum(1 for s in signals.values() if s['overall'] == '卖出')
        hold_count = sum(1 for s in signals.values() if s['overall'] == '持有')
        print(f'技术分析信号：买入 {buy_count} 只，卖出 {sell_count} 只，持有 {hold_count} 只')
    if not change_manager:
        print('经检查，近20天内列表中基金的经理没有更换。')
    else:
        print('经检查，近20天以下基金的经理发生了人事变动：')
        for code, name in change_manager:
            print(f'  {code} - {name}')
