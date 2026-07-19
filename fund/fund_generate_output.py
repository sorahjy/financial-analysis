import json
import os
import html
import warnings
from datetime import datetime
from .funds import get_funds, get_funds_bond
from .fund_storage import (
    connect as connect_fund_db,
    load_profile_snapshots,
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
            warnings.warn(f'{item}: SQLite 基金概况快照中缺少该基金，报告数据中以 -- 占位', RuntimeWarning)
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


def _benchmark_names(compare_index, tmp_data):
    return [tmp_data[idx][1] if idx in tmp_data else idx for idx in compare_index]


def _signal_counts(signals, all_fund_codes):
    counts = {'buy': 0, 'sell': 0, 'hold': 0, 'total': 0}
    if not signals:
        return counts
    report_signals = [signals[code] for code in all_fund_codes if code in signals]
    counts['buy'] = sum(1 for s in report_signals if s.get('overall') == '买入')
    counts['sell'] = sum(1 for s in report_signals if s.get('overall') == '卖出')
    counts['hold'] = sum(1 for s in report_signals if s.get('overall') == '持有')
    counts['total'] = len(report_signals)
    return counts


def _signal_rows(tmp_data, signals, all_fund_codes, held_codes):
    if not signals:
        return []
    rows = []
    for code in all_fund_codes:
        if code not in signals:
            continue
        sig = signals[code]
        rows.append({
            'code': code,
            'name': tmp_data[code][1] if code in tmp_data else code,
            'is_held': code in held_codes,
            'signal_state': 'force-sell' if sig.get('is_force_sell') else sig.get('overall', ''),
            'signal': sig,
        })
    return rows


def build_report_payload(tmp_data, equity_config, bond_config, change_manager, signals=None):
    equity_compare = equity_config['compare_index']
    bond_compare = bond_config['compare_index']

    equity_rows = compute_excess_table(
        equity_config['fund'], equity_compare, tmp_data,
        EQUITY_HIGHLIGHT_RED, EQUITY_HIGHLIGHT_GREEN, equity_config.get('hold_index', []))
    bond_rows = compute_excess_table(
        bond_config['fund'], bond_compare, tmp_data,
        BOND_HIGHLIGHT_RED, BOND_HIGHLIGHT_GREEN, bond_config.get('hold_index', []))

    all_fund_codes = equity_config['fund'] + bond_config['fund']
    held_codes = set(equity_config.get('hold_index', [])) | set(bond_config.get('hold_index', []))
    signal_counts = _signal_counts(signals, all_fund_codes)

    return {
        'schema_version': 1,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'period_labels': PERIOD_LABELS,
        'summary': {
            'equity_count': len(equity_rows),
            'bond_count': len(bond_rows),
            'signal_count': signal_counts['total'],
            'buy_count': signal_counts['buy'],
            'sell_count': signal_counts['sell'],
            'hold_count': signal_counts['hold'],
        },
        'manager_changes': [
            {'code': code, 'name': name}
            for code, name in change_manager
        ],
        'sections': {
            'equity': {
                'id': 'equity',
                'title': '股票型基金（中高风险）',
                'compare_index': equity_compare,
                'benchmark_names': _benchmark_names(equity_compare, tmp_data),
                'rows': equity_rows,
            },
            'bond': {
                'id': 'bond',
                'title': '债券型基金（中低风险）',
                'compare_index': bond_compare,
                'benchmark_names': _benchmark_names(bond_compare, tmp_data),
                'rows': bond_rows,
            },
        },
        'signal_counts': signal_counts,
        'signal_rows': _signal_rows(tmp_data, signals, all_fund_codes, held_codes),
    }


def write_report_data(tmp_data, equity_config, bond_config, change_manager, signals=None, output_file='data/fund_report_data.json'):
    payload = build_report_payload(tmp_data, equity_config, bond_config, change_manager, signals)
    output_path = os.fspath(output_file)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path, payload


def load_profile_items():
    conn = connect_fund_db()
    try:
        return load_profile_snapshots(conn)
    finally:
        conn.close()


if __name__ == '__main__':
    tmp_data = {}
    for data in load_profile_items().values():
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

    # 生成结构化报告数据，HTML 渲染由 Flask 负责
    output_path, _payload = write_report_data(tmp_data, equity_config, bond_config, change_manager, signals)

    print('*' * 30, 'result', '*' * 30)
    print(f'处理完毕，文件 {output_path} 已输出。')
    if signals:
        report_signals = [
            signals[code]
            for code in equity_config['fund'] + bond_config['fund']
            if code in signals
        ]
        buy_count = sum(1 for s in report_signals if s.get('overall') == '买入')
        sell_count = sum(1 for s in report_signals if s.get('overall') == '卖出')
        hold_count = sum(1 for s in report_signals if s.get('overall') == '持有')
        print(f'技术分析信号：买入 {buy_count} 只，卖出 {sell_count} 只，持有 {hold_count} 只')
    if not change_manager:
        print('经检查，近20天内列表中基金的经理没有更换。')
    else:
        print('经检查，近20天以下基金的经理发生了人事变动：')
        for code, name in change_manager:
            print(f'  {code} - {name}')
