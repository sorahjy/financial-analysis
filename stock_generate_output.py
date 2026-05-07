"""
读取 data/CN_stock/selected_stocks.json 选中的股票，计算技术分析信号并生成 stock_report.html。
只包含"技术分析 - 多维买卖信号"板块，不输出 Excel，不包含基金专属板块。
"""
import json
import os
from datetime import datetime
from pathlib import Path

from fund_technical_analysis import (
    analyze_fund,
    ADX_WEIGHT, BUY_SIGNAL, SELL_SIGNAL,
    CONSEC_CHANGE_PCT, FORCE_TAKE_PROFIT,
    BUY_PERCENTILE_CAP, SELL_PERCENTILE_FLOOR,
    TREND_60_ENABLED, TREND_60_SCORE,
    FORCE_TAKE_PROFIT_ENABLED,
)
from fund_generate_output import (
    render_sparkline_svg, render_rsi_bar, render_signal_badge,
    render_trend_badge, render_percentile_bar, render_adx_bar,
    render_atr_display, render_score_bar,
)

DATA_DIR = Path("data/CN_stock")
SELECTED_FILE = DATA_DIR / "selected_stocks.json"


def load_stock_records(code, name):
    """读取单只股票文件，返回 analyze_fund 能接受的 records（nav_acc=close）"""
    fp = DATA_DIR / f"CN_{code}_{name}.json"
    if not fp.exists():
        return None
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for r in data.get("records", []):
        close = r.get("close")
        if close is None:
            continue
        records.append({"date": r["date"], "nav_acc": close})
    return records


def compute_signals(selected_stocks):
    """对每只股票算信号，返回 {code: signal_dict}"""
    signals = {}
    for s in selected_stocks:
        code = s["code"]
        name = s["name"]
        records = load_stock_records(code, name)
        if not records:
            continue
        sig = analyze_fund(records)
        if sig is None:
            continue
        # 清理 None → 'null'（与 fund 侧保持一致，便于 SVG 判断）
        clean = {}
        for k, v in sig.items():
            if isinstance(v, list):
                clean[k] = [x if x is not None else 'null' for x in v]
            else:
                clean[k] = v
        signals[code] = clean
    return signals


def render_signal_row(code, name, sig):
    sparkline = render_sparkline_svg(sig)

    j_val = sig.get('j_value', '')
    j_display = f'{j_val:.1f}' if isinstance(j_val, (int, float)) else '--'
    pct_b = sig.get('pct_b', '')
    pct_b_display = f'{pct_b:.2f}' if isinstance(pct_b, (int, float)) else '--'

    nav_display = str(sig.get('latest_nav', ''))

    ma_s = sig.get('ma_score', 0)
    rsi_s = sig.get('rsi_score', 0)
    macd_s = sig.get('macd_score', 0)
    kdj_cross_s = sig.get('kdj_cross_score', 0)
    j_s = sig.get('j_score', 0)
    boll_s = sig.get('boll_score', 0)

    force_note = '<br><small style="color:#722ed1">触发止盈</small>' if sig.get('is_force_sell') else ''
    filter_reason = sig.get('filter_reason', '')
    filter_note = f'<br><small style="color:#fa8c16">{filter_reason}</small>' if filter_reason else ''
    if sig.get('is_force_sell'):
        overall_badge = '<span class="signal-badge badge-force-sell">止盈信号</span>'
    else:
        overall_badge = render_signal_badge(sig['overall'])

    trend_cell = '--'
    if TREND_60_ENABLED:
        t60_score = f'{sig.get("trend_60_score", 0):+g}' if sig.get('market_state') == '趋势' else '0 (非趋势市)'
        trend_cell = render_trend_badge(sig.get('trend_60', '')) + f'<br><small>得分: {t60_score}</small>'

    return f'''<tr>
        <td>{code}</td>
        <td>{name}</td>
        <td>{nav_display}</td>
        <td class="chart-cell">{sparkline}</td>
        <td>{render_signal_badge(sig['ma_signal'], ma_s)}</td>
        <td>{render_rsi_bar(sig.get('rsi_value'))}<br><small>得分: {rsi_s:+g}</small></td>
        <td>{render_signal_badge(sig['macd_signal'], macd_s)}</td>
        <td>{render_signal_badge(sig.get('kdj_cross_signal', '持有'), kdj_cross_s)}<br>{render_signal_badge(sig.get('j_signal', '持有'), j_s)}<br><small>J={j_display}</small></td>
        <td>{render_signal_badge(sig.get('boll_signal', '持有'), boll_s)}<br><small>%B={pct_b_display}</small></td>
        <td>{render_adx_bar(sig.get('adx_value'), sig.get('market_state', ''))}</td>
        <td>{render_atr_display(sig.get('atr_pct'))}</td>
        <td>{render_percentile_bar(sig.get('nav_percentile'))}</td>
        <td>{trend_cell}</td>
        <td>{render_score_bar(sig.get('score', 0))}{force_note}</td>
        <td>{overall_badge}{filter_note}</td>
    </tr>
'''


def generate_html(selected_data, signals, output_path='stock_report.html'):
    today = datetime.now().strftime('%Y-%m-%d %H:%M')
    params = selected_data.get('params', {})
    count = selected_data.get('count', len(selected_data.get('stocks', [])))

    code_to_name = {s['code']: s['name'] for s in selected_data.get('stocks', [])}

    signal_rows = ''
    for s in selected_data.get('stocks', []):
        code = s['code']
        if code not in signals:
            continue
        signal_rows += render_signal_row(code, code_to_name.get(code, code), signals[code])

    params_html = (
        f'窗口 {params.get("window_start", "?")} ~ {params.get("start_time", "?")} · '
        f'跌幅 &gt; {params.get("max_drawdown_pct", "?")}% · '
        f'市值 &gt; {params.get("min_market_cap", "?")}亿 · '
        f'PB &lt; {params.get("max_pb", "?")} · '
        f'命中 {count} 只'
    )

    if FORCE_TAKE_PROFIT_ENABLED:
        force_desc = (
            f'收盘价涨幅&gt;{round((FORCE_TAKE_PROFIT - 1) * 100)}%触发'
            f'<span style="color:#722ed1">止盈信号</span>'
        )
    else:
        force_desc = '止盈信号已关闭'

    signal_note = (
        f'ADX&ge;25=趋势行情（MA/MACD权重&times;{ADX_WEIGHT}，MA60趋势&plusmn;{TREND_60_SCORE}），'
        f'ADX&lt;25=震荡行情（RSI/布林权重&times;{ADX_WEIGHT}，MA60趋势=0），KDJ固定权重&times;1 | '
        f'综合建议: 评分 &ge;+{BUY_SIGNAL} 买入, &le;{SELL_SIGNAL} 卖出 | '
        f'买卖信号经历史去重(变动需超{round(CONSEC_CHANGE_PCT * 100)}%)、'
        f'百分位过滤(买&gt;{BUY_PERCENTILE_CAP}%/卖&lt;{SELL_PERCENTILE_FLOOR}%无效) | '
        f'{force_desc} | '
        f'ATR/百分位仅展示'
    )

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>股票量化指标报告</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; background: #f5f7fa; color: #333; padding: 20px; }}
    h1 {{ text-align: center; margin: 20px 0 5px; font-size: 24px; }}
    .date {{ text-align: center; color: #888; margin-bottom: 20px; font-size: 14px; }}
    .params {{ text-align: center; color: #555; margin-bottom: 16px; font-size: 13px; }}
    h2 {{ margin: 30px 0 10px; padding-left: 10px; border-left: 4px solid #1890ff; font-size: 18px; }}
    .table-wrapper {{ overflow-x: auto; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; font-size: 13px; white-space: nowrap; min-width: 100%; }}
    th, td {{ border: 1px solid #d9d9d9; padding: 6px 10px; text-align: center; }}
    thead th {{ background: #fafafa; font-weight: 600; position: sticky; top: 0; }}
    tbody tr:hover {{ background: #f0f5ff; }}

    .signal-table td {{ vertical-align: middle; }}
    .chart-cell {{ padding: 4px !important; }}
    .signal-badge {{
        display: inline-block; padding: 2px 8px; border-radius: 10px;
        font-size: 11px; font-weight: 600; color: #fff; margin: 1px;
    }}
    .badge-buy {{ background: #f5222d; }}
    .badge-sell {{ background: #52c41a; }}
    .badge-hold {{ background: #8c8c8c; }}
    .badge-force-sell {{ background: #722ed1; }}
    .trend-bull {{ color: #cf1322; font-weight: 600; font-size: 12px; }}
    .trend-bear {{ color: #389e0d; font-weight: 600; font-size: 12px; }}
    .trend-neutral {{ color: #8c8c8c; font-size: 12px; }}
    .rsi-bar {{
        position: relative; width: 110px; height: 18px;
        background: #f0f0f0; border-radius: 9px; overflow: hidden;
        display: inline-block; vertical-align: middle;
    }}
    .rsi-fill {{ height: 100%; border-radius: 9px; }}
    .rsi-label {{
        position: absolute; top: 0; left: 0; right: 0;
        text-align: center; font-size: 11px; line-height: 18px;
        color: #333; font-weight: 500;
    }}
    .rsi-na {{ color: #bbb; font-size: 12px; }}
    .score-bar {{
        position: relative; width: 80px; height: 18px;
        background: #f0f0f0; border-radius: 9px; overflow: hidden;
        display: inline-block; vertical-align: middle;
    }}
    .score-fill {{ height: 100%; border-radius: 9px; }}
    .score-label {{
        position: absolute; top: 0; left: 0; right: 0;
        text-align: center; font-size: 11px; line-height: 18px;
        color: #333; font-weight: 600;
    }}
    .dd-high {{ color: #cf1322; font-weight: 600; }}
    .dd-mid {{ color: #fa8c16; font-weight: 500; }}
    .dd-low {{ color: #389e0d; }}
    .legend {{
        margin: 10px 0; display: flex; gap: 16px; align-items: center;
        font-size: 13px; color: #666; flex-wrap: wrap;
    }}
    .legend span {{ display: flex; align-items: center; gap: 4px; }}
    .signal-note {{ font-size: 12px; color: #999; margin: 4px 0 12px; }}
    small {{ color: #999; font-size: 11px; }}
</style>
</head>
<body>
<h1>股票量化指标报告</h1>
<p class="date">生成时间：{today}</p>
<p class="params">{params_html}</p>

<h2>技术分析 - 多维买卖信号</h2>
<div class="legend">
    <span><svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="#1890ff" stroke-width="1.5"/></svg> 收盘价</span>
    <span><svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="#faad14" stroke-width="1" stroke-dasharray="3,2"/></svg> MA5</span>
    <span><svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="#f5222d" stroke-width="1" stroke-dasharray="5,3"/></svg> MA20</span>
    <span><svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="#722ed1" stroke-width="1" stroke-dasharray="8,4"/></svg> MA60</span>
    <span><svg width="20" height="10"><rect x="0" y="2" width="20" height="6" fill="#e6f0fa" opacity="0.7"/></svg> 布林带</span>
    <span><svg width="14" height="12"><polygon points="7,1 2,11 12,11" fill="#f5222d" opacity="0.85"/></svg> 买入信号</span>
    <span><svg width="14" height="12"><polygon points="7,11 2,1 12,1" fill="#52c41a" opacity="0.85"/></svg> 卖出信号</span>
    <span><svg width="14" height="12"><polygon points="7,11 2,1 12,1" fill="#722ed1" opacity="0.85"/></svg> 止盈信号</span>
</div>
<p class="signal-note">{signal_note}</p>
<div class="table-wrapper">
<table class="signal-table">
    <thead>
        <tr>
            <th>代码</th><th>名字</th><th>收盘价</th><th>近5年走势</th>
            <th>MA(5/20)</th><th>RSI(14)</th><th>MACD</th><th>KDJ(交叉/J值)</th><th>布林带</th>
            <th>ADX(14)</th><th>ATR%</th><th>百分位（近6年）</th><th>趋势60</th>
            <th>评分</th><th>建议</th>
        </tr>
    </thead>
    <tbody>{signal_rows}</tbody>
</table>
</div>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    if not SELECTED_FILE.exists():
        raise SystemExit(
            f"未找到 {SELECTED_FILE}，请先运行 stock_crawl_top_800_data.py 中的 select_by_drawdown()"
        )

    with open(SELECTED_FILE, encoding='utf-8') as f:
        selected_data = json.load(f)

    stocks = selected_data.get('stocks', [])
    print(f'加载 {len(stocks)} 只选中股票，开始计算技术信号...')

    signals = compute_signals(stocks)
    print(f'技术分析完成，{len(signals)} 只股票生成了信号')

    generate_html(selected_data, signals)
    print('stock_report.html 已生成')

    if signals:
        buy = sum(1 for s in signals.values() if s['overall'] == '买入')
        sell = sum(1 for s in signals.values() if s['overall'] == '卖出')
        hold = sum(1 for s in signals.values() if s['overall'] == '持有')
        print(f'买入 {buy} · 卖出 {sell} · 持有 {hold}')


if __name__ == '__main__':
    main()
