"""
回测模块：读取 fund_technical_analysis.py 生成的 data/signals.json，回测买卖信号收益。

修改 fund_technical_analysis.py 参数后重新运行 fund_technical_analysis.py 生成 signals.json，
再运行本脚本即可回测，无需修改本文件。

回测规则：
  - 初始本金：INIT_CAPITAL 元（买入等值份额）
  - 买入信号：追加 TRADE_AMOUNT 元本金买入（本金 >= MAX_CAPITAL 时跳过）
  - 卖出信号：撤回 TRADE_AMOUNT 元本金卖出（本金 <= MIN_CAPITAL 时跳过）
  - 本金 = 累计投入金额，范围 [MIN_CAPITAL, MAX_CAPITAL]
  - 组合价值 = 持仓市值 + 现金（现金可为负，表示额外投入）
  - 基准：买入并持有 INIT_CAPITAL 元不动
"""
import json
import os
import unicodedata

# ============================================================
# 回测参数
# ============================================================
INIT_CAPITAL = 100000       # 初始投入本金（元）
TRADE_AMOUNT = 50000        # 每次买卖金额（元）
MAX_CAPITAL = 300000        # 本金上限（累计投入达到后不再买入）
MIN_CAPITAL = 0             # 本金下限（累计投入降到后不再卖出）


def _display_width(s):
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in s)

def _rjust(s, width):
    return ' ' * max(0, width - _display_width(s)) + s

def _ljust(s, width):
    return s + ' ' * max(0, width - _display_width(s))

def _truncate(s, width):
    result, w = [], 0
    for c in s:
        cw = 2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1
        if w + cw > width:
            break
        result.append(c)
        w += cw
    return ''.join(result)


def backtest_fund(sig):
    """基于 signals.json 中单只基金的信号数据执行回测。"""
    navs = sig.get('recent_navs', [])
    dates = sig.get('recent_dates', [])
    buy_markers = sig.get('buy_markers', [])
    sell_markers = sig.get('sell_markers', [])
    force_sell_markers = sig.get('force_sell_markers', [])

    # 过滤 null
    valid_navs = [(i, v) for i, v in enumerate(navs) if v is not None and v != 'null']
    if len(valid_navs) < 2:
        return None

    first_idx, first_nav = valid_navs[0]

    # 构建事件表：{ci: action}
    events = {}
    for ci, nav_val in buy_markers:
        events[ci] = 'buy'
    for ci, nav_val in sell_markers:
        events[ci] = 'sell'
    for ci, nav_val in force_sell_markers:
        events[ci] = 'force_sell'

    # --- 初始化 ---
    shares = INIT_CAPITAL / first_nav       # 持有份额
    cash = 0.0                              # 现金（负数=额外投入）
    capital = INIT_CAPITAL                  # 本金（累计投入）

    buy_count = 0
    sell_count = 0
    force_sell_count = 0

    # --- 遍历每日，遇到事件执行交易 ---
    for ci in range(first_idx + 1, len(navs)):
        nav = navs[ci]
        if nav is None or nav == 'null':
            continue

        action = events.get(ci)
        if action is None:
            continue

        if action == 'buy':
            if capital < MAX_CAPITAL:
                buy_val = min(TRADE_AMOUNT, MAX_CAPITAL - capital)
                shares += buy_val / nav
                cash -= buy_val
                capital += buy_val
                buy_count += 1

        elif action in ('sell', 'force_sell'):
            if capital > MIN_CAPITAL:
                sell_val = min(TRADE_AMOUNT, capital - MIN_CAPITAL)
                sell_shares = sell_val / nav
                if sell_shares <= shares:
                    shares -= sell_shares
                    cash += sell_val
                    capital -= sell_val
                    if action == 'force_sell':
                        force_sell_count += 1
                    else:
                        sell_count += 1

    # --- 计算收益 ---
    last_nav = navs[-1]
    if last_nav is None or last_nav == 'null':
        # 找最后一个有效值
        for v in reversed(navs):
            if v is not None and v != 'null':
                last_nav = v
                break
        else:
            return None

    final_position = shares * last_nav      # 持仓市值
    strategy_value = final_position + cash  # 组合总价值

    benchmark_value = (INIT_CAPITAL / first_nav) * last_nav  # 买入持有

    strategy_return = (strategy_value / INIT_CAPITAL - 1) * 100
    benchmark_return = (benchmark_value / INIT_CAPITAL - 1) * 100
    excess_return = strategy_return - benchmark_return

    start_date = dates[first_idx] if first_idx < len(dates) else '?'
    end_date = dates[-1] if dates else '?'

    return {
        'start_date': start_date,
        'end_date': end_date,
        'data_days': len(valid_navs),
        'strategy_value': round(strategy_value, 2),
        'benchmark_value': round(benchmark_value, 2),
        'strategy_return': round(strategy_return, 2),
        'benchmark_return': round(benchmark_return, 2),
        'excess_return': round(excess_return, 2),
        'buy_count': buy_count,
        'sell_count': sell_count,
        'force_sell_count': force_sell_count,
        'final_position': round(final_position, 2),
        'final_cash': round(cash, 2),
        'final_capital': round(capital, 2),
    }


def load_fund_names():
    """从 data/temp.json (JSONL) 加载基金名称映射 {code: name}。"""
    names = {}
    path = os.path.join('data', 'temp.json')
    if not os.path.exists(path):
        return names
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                names[obj['fundCode']] = obj.get('name', '')
            except (json.JSONDecodeError, KeyError):
                continue
    return names


def main():
    signals_file = 'data/signals.json'
    if not os.path.exists(signals_file):
        print(f'错误：{signals_file} 不存在，请先运行 fund_technical_analysis.py 生成信号数据')
        return

    with open(signals_file, encoding='utf-8') as f:
        signals = json.load(f)

    fund_names = load_fund_names()

    results = {}
    for code in sorted(signals.keys()):
        sig = signals[code]
        result = backtest_fund(sig)
        if result is None:
            print(f'  [SKIP] {code}: 数据不足')
            continue
        results[code] = result

    # 保存结果
    os.makedirs('data', exist_ok=True)
    with open('data/backtest_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 输出汇总
    print(f'\n回测完成，共 {len(results)} 只基金')
    print(f'回测参数：初始本金={INIT_CAPITAL}, 单次交易={TRADE_AMOUNT}, '
          f'本金上限={MAX_CAPITAL}, 本金下限={MIN_CAPITAL}')
    print()
    print(f'{_rjust("代码", 8)}  {_ljust("名称", 16)} {_rjust("起始日期", 10)} {_rjust("结束日期", 10)} {_rjust("天数", 5)}  '
          f'{_rjust("策略价值", 14)} {_rjust("基准价值", 14)}  '
          f'{_rjust("策略收益%", 12)} {_rjust("基准收益%", 12)} {_rjust("超额收益%", 12)}  '
          f'{_rjust("买入", 4)} {_rjust("卖出", 4)} {_rjust("止盈", 4)}  '
          f'{_rjust("持仓市值", 14)} {_rjust("现金", 14)} {_rjust("本金", 12)}')
    print('-' * 185)

    total_strategy = 0
    total_benchmark = 0
    excess_positive = 0

    for code, r in sorted(results.items()):
        total_strategy += r['strategy_return']
        total_benchmark += r['benchmark_return']
        if r['excess_return'] > 0:
            excess_positive += 1

        name = fund_names.get(code, '')
        display_name = _truncate(name, 16)
        print(f'{code:>8}  {_ljust(display_name, 16)} '
              f'{r["start_date"]:>10} {r["end_date"]:>10} {r["data_days"]:>5}  '
              f'{r["strategy_value"]:>14,.2f} {r["benchmark_value"]:>14,.2f}  '
              f'{r["strategy_return"]:>+12.2f} {r["benchmark_return"]:>+12.2f} {r["excess_return"]:>+12.2f}  '
              f'{r["buy_count"]:>4} {r["sell_count"]:>4} {r["force_sell_count"]:>4}  '
              f'{r["final_position"]:>14,.2f} {r["final_cash"]:>14,.2f} {r["final_capital"]:>12,.0f}')

    n = len(results)
    if n > 0:
        print('-' * 160)
        avg_strategy = total_strategy / n
        avg_benchmark = total_benchmark / n
        avg_excess = avg_strategy - avg_benchmark
        print(f'\n汇总统计（{n} 只基金）：')
        print(f'  平均策略收益: {avg_strategy:+.2f}%')
        print(f'  平均基准收益: {avg_benchmark:+.2f}%')
        print(f'  平均超额收益: {avg_excess:+.2f}%')
        print(f'  跑赢基准数量: {excess_positive}/{n} ({excess_positive/n*100:.1f}%)')


if __name__ == '__main__':
    main()
