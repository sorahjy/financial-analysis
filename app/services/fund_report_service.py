from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from fund.fund_technical_analysis import (
    ADX_WEIGHT,
    BUY_PERCENTILE_CAP,
    BUY_SIGNAL,
    CONSEC_CHANGE_PCT,
    FORCE_TAKE_PROFIT,
    FORCE_TAKE_PROFIT_ENABLED,
    SELL_PERCENTILE_FLOOR,
    SELL_SIGNAL,
    TREND_60_ENABLED,
    TREND_60_SCORE,
)


@dataclass
class FundReportView:
    exists: bool
    generated_at: str = ""
    period_labels: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    manager_changes: List[Dict[str, Any]] = field(default_factory=list)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    signal_counts: Dict[str, Any] = field(default_factory=dict)
    signal_rows: List[Dict[str, Any]] = field(default_factory=list)
    signal_note: str = ""
    error: str = ""


def load_fund_report_view(report_data_file: Path) -> FundReportView:
    if not report_data_file.exists():
        return FundReportView(exists=False)
    try:
        payload = json.loads(report_data_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return FundReportView(exists=False, error=str(exc))
    return build_fund_report_view(payload)


def build_fund_report_view(payload: Dict[str, Any]) -> FundReportView:
    period_labels = list(payload.get("period_labels") or [])
    raw_sections = payload.get("sections") or {}
    raw_signal_rows = list(payload.get("signal_rows") or [])
    signal_states = _signal_state_map(raw_signal_rows)
    sections = [
        _prepare_metric_section(raw_sections[key], period_labels, signal_states)
        for key in ("equity", "bond")
        if key in raw_sections
    ]
    return FundReportView(
        exists=True,
        generated_at=payload.get("generated_at", ""),
        period_labels=period_labels,
        summary=payload.get("summary") or {},
        manager_changes=list(payload.get("manager_changes") or []),
        sections=sections,
        signal_counts=payload.get("signal_counts") or {},
        signal_rows=[_prepare_signal_row(row) for row in raw_signal_rows],
        signal_note=_signal_note(),
    )


def _signal_state_map(signal_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    states = {}
    for row in signal_rows:
        code = str(row.get("code") or "")
        if code:
            states[code] = str(row.get("signal_state") or "")
    return states


def _prepare_metric_section(
    section: Dict[str, Any],
    period_labels: List[str],
    signal_states: Dict[str, str],
) -> Dict[str, Any]:
    rows = []
    for row in section.get("rows") or []:
        prepared = {**row}
        prepared["signal_state"] = signal_states.get(str(prepared.get("code") or ""), "")
        rows.append(prepared)
    benchmark_names = list(section.get("benchmark_names") or [])
    return {
        "id": section.get("id", ""),
        "title": section.get("title", ""),
        "benchmark_names": benchmark_names,
        "benchmark_count": len(benchmark_names),
        "metric_count": len(period_labels),
        "total_count": len(rows),
        "rows": rows,
    }


def _prepare_signal_row(row: Dict[str, Any]) -> Dict[str, Any]:
    sig = row.get("signal") or {}
    return {
        **row,
        "display": {
            "nav": _render_nav(sig),
            "sparkline": render_sparkline_svg(sig),
            "ma": render_signal_badge(sig.get("ma_signal", "持有"), sig.get("ma_score", 0)),
            "rsi": f"{render_rsi_bar(sig.get('rsi_value'))}<br><small>得分: {_fmt_score(sig.get('rsi_score', 0))}</small>",
            "macd": render_signal_badge(sig.get("macd_signal", "持有"), sig.get("macd_score", 0)),
            "kdj": _render_kdj(sig),
            "boll": _render_boll(sig),
            "adx": render_adx_bar(sig.get("adx_value")),
            "atr": render_atr_display(sig.get("atr_pct")),
            "percentile": f"{render_percentile_bar(sig.get('nav_percentile'))}<br>",
            "trend": _render_trend_score(sig),
            "score": _render_total_score(sig),
            "advice": _render_advice(sig),
        },
    }


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt_score(value: Any) -> str:
    try:
        return f"{float(value):+g}"
    except (TypeError, ValueError):
        return "+0"


def render_sparkline_svg(sig: Dict[str, Any], width: int = 650, height: int = 140) -> str:
    navs = sig.get("recent_navs", [])
    ma5 = sig.get("recent_ma5", [])
    ma20 = sig.get("recent_ma20", [])
    ma60 = sig.get("recent_ma60", [])
    boll_upper = sig.get("recent_boll_upper", [])
    boll_lower = sig.get("recent_boll_lower", [])
    dates = sig.get("recent_dates", [])
    buy_markers = sig.get("buy_markers", [])
    sell_markers = sig.get("sell_markers", [])
    force_sell_markers = sig.get("force_sell_markers", [])

    valid_navs = [v for v in navs if v is not None and v != "null"]
    if len(valid_navs) < 2:
        return ""

    all_vals = list(valid_navs)
    for series in (boll_upper, boll_lower):
        for value in series:
            if value is not None and value != "null":
                all_vals.append(value)

    pad = 8
    chart_w = width - 2 * pad
    chart_h = height - 2 * pad
    nav_min = min(all_vals)
    nav_max = max(all_vals)
    nav_range = nav_max - nav_min or 1
    count = len(navs)

    def to_xy(index: int, value: float) -> tuple[float, float]:
        x = pad + index / (count - 1) * chart_w if count > 1 else pad
        y = pad + chart_h - (value - nav_min) / nav_range * chart_h
        return x, y

    def polyline_str(series: List[Any]) -> str:
        points = []
        for index, value in enumerate(series):
            if value is not None and value != "null":
                x, y = to_xy(index, value)
                points.append(f"{x:.1f},{y:.1f}")
        return " ".join(points)

    svg = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']

    upper_pts = []
    lower_pts = []
    for index in range(count):
        upper = boll_upper[index] if index < len(boll_upper) else None
        lower = boll_lower[index] if index < len(boll_lower) else None
        if upper is not None and upper != "null" and lower is not None and lower != "null":
            upper_x, upper_y = to_xy(index, upper)
            lower_x, lower_y = to_xy(index, lower)
            upper_pts.append(f"{upper_x:.1f},{upper_y:.1f}")
            lower_pts.append(f"{lower_x:.1f},{lower_y:.1f}")
    if upper_pts and lower_pts:
        polygon = " ".join(upper_pts) + " " + " ".join(reversed(lower_pts))
        svg.append(f'<polygon points="{polygon}" fill="#e6f0fa" fill-opacity="0.5" stroke="none" />')

    svg.append(f'<polyline points="{polyline_str(navs)}" fill="none" stroke="#1890ff" stroke-width="1.5" />')
    for series, color, dash in (
        (ma5, "#faad14", "3,2"),
        (ma20, "#f5222d", "5,3"),
        (ma60, "#722ed1", "8,4"),
    ):
        points = polyline_str(series)
        if points:
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1" stroke-dasharray="{dash}" />')

    month_ticks = []
    seen_months = set()
    for index, date_text in enumerate(dates):
        if not date_text or date_text == "null":
            continue
        month_key = str(date_text)[:7]
        if month_key not in seen_months:
            seen_months.add(month_key)
            month_ticks.append((index, str(date_text)))
    if len(month_ticks) <= 24:
        step = 1
    elif len(month_ticks) <= 60:
        step = 3
    else:
        step = 6
    for tick_index, (index, date_text) in enumerate(month_ticks):
        x, _ = to_xy(index, nav_min)
        svg.append(f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{height - pad}" stroke="#e8e8e8" stroke-width="0.5" />')
        if tick_index % step == 0:
            month_num = date_text[5:7]
            label = date_text[:4] if month_num == "01" else f"{month_num}月"
            svg.append(f'<text x="{x:.1f}" y="{height - 1}" text-anchor="middle" font-size="9" fill="#999">{_esc(label)}</text>')

    _append_markers(svg, buy_markers, to_xy, "buy")
    _append_markers(svg, sell_markers, to_xy, "sell")
    _append_markers(svg, force_sell_markers, to_xy, "force-sell")
    svg.append("</svg>")
    return "".join(svg)


def _append_markers(svg: List[str], markers: List[Any], to_xy: Any, marker_type: str) -> None:
    color = {"buy": "#f5222d", "sell": "#52c41a", "force-sell": "#722ed1"}[marker_type]
    for marker in markers:
        if not isinstance(marker, (list, tuple)) or len(marker) < 2:
            continue
        index, nav_value = marker[0], marker[1]
        if nav_value is None or nav_value == "null":
            continue
        x, y = to_xy(index, nav_value)
        if marker_type == "buy":
            points = f"{x:.1f},{y + 3:.1f} {x - 4:.1f},{y + 11:.1f} {x + 4:.1f},{y + 11:.1f}"
        else:
            points = f"{x:.1f},{y - 3:.1f} {x - 4:.1f},{y - 11:.1f} {x + 4:.1f},{y - 11:.1f}"
        svg.append(f'<polygon points="{points}" fill="{color}" opacity="0.85" />')


def render_rsi_bar(rsi_value: Any) -> str:
    if rsi_value is None:
        return '<span class="rsi-na">N/A</span>'
    pct = max(0, min(100, rsi_value))
    if pct > 70:
        color = "#cf1322"
        label = "超买"
    elif pct < 30:
        color = "#389e0d"
        label = "超卖"
    else:
        color = "#1890ff"
        label = "中性"
    return (
        '<div class="rsi-bar">'
        f'<div class="rsi-fill" style="width:{pct}%;background:{color}"></div>'
        f'<span class="rsi-label">{_esc(rsi_value)} ({label})</span>'
        "</div>"
    )


def render_signal_badge(signal: Any, score: Any = None) -> str:
    signal_text = str(signal or "持有")
    cls = "badge-buy" if signal_text == "买入" else ("badge-sell" if signal_text == "卖出" else "badge-hold")
    score_str = ""
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0
    if score is not None and score_value != 0:
        score_str = f" ({score_value:+g})"
    return f'<span class="signal-badge {cls}">{_esc(signal_text)}{score_str}</span>'


def render_trend_badge(trend: Any) -> str:
    if trend == "多头":
        return '<span class="trend-bull">多头</span>'
    if trend == "空头":
        return '<span class="trend-bear">空头</span>'
    return '<span class="trend-neutral">未知</span>'


def render_percentile_bar(percentile: Any) -> str:
    if percentile is None:
        return '<span class="rsi-na">N/A</span>'
    pct = max(0, min(100, percentile))
    if pct < 10:
        color = "#389e0d"
        label = "极低"
    elif pct < 20:
        color = "#52c41a"
        label = "偏低"
    elif pct > 90:
        color = "#cf1322"
        label = "极高"
    elif pct > 80:
        color = "#f5222d"
        label = "偏高"
    else:
        color = "#1890ff"
        label = "中性"
    return (
        '<div class="rsi-bar">'
        f'<div class="rsi-fill" style="width:{pct}%;background:{color}"></div>'
        f'<span class="rsi-label">{percentile:.1f}% ({label})</span>'
        "</div>"
    )


def render_adx_bar(adx_value: Any, market_state: Any = None) -> str:
    if adx_value is None:
        return '<span class="rsi-na">N/A</span>'
    pct = max(0, min(100, adx_value))
    if adx_value >= 25:
        color = "#1890ff"
        label = "趋势"
    else:
        color = "#faad14"
        label = "震荡"
    return (
        '<div class="rsi-bar">'
        f'<div class="rsi-fill" style="width:{pct}%;background:{color}"></div>'
        f'<span class="rsi-label">{adx_value:.1f} ({label})</span>'
        "</div>"
    )


def render_atr_display(atr_pct: Any) -> str:
    if atr_pct is None:
        return '<span class="rsi-na">N/A</span>'
    if atr_pct > 2.0:
        cls = "dd-high"
    elif atr_pct > 1.0:
        cls = "dd-mid"
    else:
        cls = "dd-low"
    return f'<span class="{cls}">{atr_pct:.3f}%</span>'


def render_score_bar(score: Any) -> str:
    max_score = 2 * ADX_WEIGHT + 4 + (TREND_60_SCORE if TREND_60_ENABLED else 0)
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0
    pct = (score_value + max_score) / (2 * max_score) * 100
    pct = max(0, min(100, pct))
    if score_value >= BUY_SIGNAL:
        color = "#f5222d"
    elif score_value <= SELL_SIGNAL:
        color = "#52c41a"
    elif score_value > 0:
        color = "#ff7875"
    elif score_value < 0:
        color = "#95de64"
    else:
        color = "#8c8c8c"
    return (
        '<div class="score-bar">'
        f'<div class="score-fill" style="width:{pct}%;background:{color}"></div>'
        f'<span class="score-label">{score_value:+.1f}</span>'
        "</div>"
    )


def _render_nav(sig: Dict[str, Any]) -> str:
    nav_display = _esc(sig.get("latest_nav", ""))
    if sig.get("estimated") and sig.get("latest_nav"):
        nav_display += '<br><small class="estimate-label">估值</small>'
    return nav_display


def _render_kdj(sig: Dict[str, Any]) -> str:
    j_value = sig.get("j_value")
    j_display = f"{j_value:.1f}" if isinstance(j_value, (int, float)) else "--"
    return (
        f"{render_signal_badge(sig.get('kdj_cross_signal', '持有'), sig.get('kdj_cross_score', 0))}<br>"
        f"{render_signal_badge(sig.get('j_signal', '持有'), sig.get('j_score', 0))}<br>"
        f"<small>J={j_display}</small>"
    )


def _render_boll(sig: Dict[str, Any]) -> str:
    pct_b = sig.get("pct_b")
    pct_b_display = f"{pct_b:.2f}" if isinstance(pct_b, (int, float)) else "--"
    return (
        f"{render_signal_badge(sig.get('boll_signal', '持有'), sig.get('boll_score', 0))}<br>"
        f"<small>%B={pct_b_display}</small>"
    )


def _render_trend_score(sig: Dict[str, Any]) -> str:
    if not TREND_60_ENABLED:
        return "--"
    score_text = _fmt_score(sig.get("trend_60_score", 0)) if sig.get("market_state") == "趋势" else "0 (非趋势市)"
    return f"{render_trend_badge(sig.get('trend_60', ''))}<br><small>得分: {_esc(score_text)}</small>"


def _render_total_score(sig: Dict[str, Any]) -> str:
    force_note = '<br><small class="force-label">触发止盈</small>' if sig.get("is_force_sell") else ""
    return f"{render_score_bar(sig.get('score', 0))}{force_note}"


def _render_advice(sig: Dict[str, Any]) -> str:
    if sig.get("is_force_sell"):
        overall_badge = '<span class="signal-badge badge-force-sell">止盈信号</span>'
    else:
        overall_badge = render_signal_badge(sig.get("overall", "持有"))

    filter_reason = sig.get("filter_reason", "")
    filter_note = f'<br><small class="filter-label">{_esc(filter_reason)}</small>' if filter_reason else ""
    gztime = sig.get("gztime", "")
    if gztime:
        gztime_short = str(gztime)[5:] if len(str(gztime)) > 5 else str(gztime)
        gztime_display = f"<br><small>估算: {_esc(gztime_short)}</small>"
    else:
        gztime_display = ""
    return f"{overall_badge}{filter_note}{gztime_display}"


def _signal_note() -> str:
    if FORCE_TAKE_PROFIT_ENABLED:
        take_profit_note = (
            f'净值涨幅&gt;{round((FORCE_TAKE_PROFIT - 1) * 100)}%'
            '触发<span class="force-label">止盈信号</span>'
        )
    else:
        take_profit_note = "止盈信号已关闭"
    return (
        f"ADX&ge;25=趋势行情（MA/MACD权重&times;{ADX_WEIGHT}，MA60趋势&plusmn;{TREND_60_SCORE}），"
        f"ADX&lt;25=震荡行情（RSI/布林权重&times;{ADX_WEIGHT}，MA60趋势=0），"
        f"KDJ固定权重&times;1 | 综合建议: 评分 &ge;+{BUY_SIGNAL} 买入, &le;{SELL_SIGNAL} 卖出 | "
        f"买卖信号经历史去重(变动需超{round(CONSEC_CHANGE_PCT * 100)}%)、"
        f"百分位过滤(买&gt;{BUY_PERCENTILE_CAP}%/卖&lt;{SELL_PERCENTILE_FLOOR}%无效) | "
        f"{take_profit_note} | ATR/百分位仅展示"
    )
