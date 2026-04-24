
"""
独立脚本：从天天基金 F10DataApi 爬取历史净值数据并持久化。

持久化文件: data/nav_store.json
结构: {
  "007345": {
    "start_date": "2019-07-16",   # 已有数据的最早日期
    "end_date": "2026-04-10",     # 已有数据的最晚日期
    "records": [ {date, nav, nav_acc, daily_growth_rate}, ... ]
  },
  ...
}

每次运行逻辑：
  1. 加载已有持久化数据
  2. 对每只基金：
     a. 向后补数据：从 end_date+1 到今天
     b. 向前追溯：从 start_date 往前，每次1年，直到接口无数据（基金成立前）
  3. 去重、按日期排序后写回持久化文件
  4. 同时导出 data/nav_history.json（仅 records，兼容下游 fund_technical_analysis.py）
"""
import json
import os
import re
import time
import random
from datetime import datetime, timedelta
import requests


HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
    'Referer': 'https://fundf10.eastmoney.com/',
}

API_URL = 'https://fundf10.eastmoney.com/F10DataApi.aspx'

STORE_FILE = os.path.join('data', 'nav_store.json')
EXPORT_FILE = os.path.join('data', 'nav_history.json')

META_RE = re.compile(r'records:(\d+),pages:(\d+),curpage:(\d+)')
ROW_RE = re.compile(r'<td[^>]*>(.*?)</td>')


def parse_response(text):
    """解析 F10DataApi 返回的 JS 变量，提取数据行和分页信息"""
    meta_match = META_RE.search(text)
    if not meta_match:
        return [], 0, 0

    total_records = int(meta_match.group(1))
    total_pages = int(meta_match.group(2))

    rows = []
    tr_parts = text.split('<tr>')[2:]
    for tr in tr_parts:
        cells = ROW_RE.findall(tr)
        if len(cells) >= 4:
            date = cells[0].strip()
            nav = cells[1].strip()
            nav_acc = cells[2].strip()
            growth = cells[3].strip().replace('%', '')
            rows.append({
                'date': date,
                'nav': nav,
                'nav_acc': nav_acc,
                'daily_growth_rate': growth,
            })

    return rows, total_records, total_pages


def fetch_range(code, sdate, edate):
    """爬取单只基金 [sdate, edate] 区间内的全部净值数据"""
    collected = []
    page = 1

    while True:
        params = {
            'type': 'lsjz',
            'code': code,
            'page': page,
            'per': 20,
            'sdate': sdate,
            'edate': edate,
        }
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
            rows, total_records, total_pages = parse_response(r.text)
        except Exception as e:
            print(f'  [ERROR] {code} page {page}: {e}')
            break

        if not rows:
            break

        collected.extend(rows)

        if page >= total_pages:
            break
        page += 1
        time.sleep(random.uniform(0.15, 0.35))

    return collected


def merge_records(existing, new_records):
    """合并两组记录，按日期去重并排序"""
    by_date = {}
    for r in existing:
        by_date[r['date']] = r
    for r in new_records:
        by_date[r['date']] = r
    return sorted(by_date.values(), key=lambda x: x['date'])


def load_store():
    """加载持久化数据"""
    if os.path.exists(STORE_FILE):
        with open(STORE_FILE, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_store(store):
    """写入持久化数据"""
    os.makedirs('data', exist_ok=True)
    with open(STORE_FILE, 'w', encoding='utf-8') as f:
        json.dump(store, f, ensure_ascii=False)


def export_nav_history(store):
    """导出兼容格式的 nav_history.json（仅 records 列表）"""
    export = {}
    for code, entry in store.items():
        export[code] = entry['records']
    with open(EXPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False)


def fetch_fund_incremental(code, entry):
    """
    增量爬取单只基金：
      1. 向后补：end_date+1 → 今天
      2. 向前追溯：从 start_date 往前，每批1年，直到无数据
    返回更新后的 entry。
    """
    today = datetime.now().strftime('%Y-%m-%d')
    existing_records = entry.get('records', [])
    start_date = entry.get('start_date')
    end_date = entry.get('end_date')

    new_records = []

    # === 1. 向后补数据 ===
    if end_date and end_date < today:
        next_day = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f'  → 补后段 {next_day} ~ {today}', end='', flush=True)
        rows = fetch_range(code, next_day, today)
        if rows:
            new_records.extend(rows)
            print(f' +{len(rows)}条')
        else:
            print(' 无新数据')
    elif not end_date:
        # 全新基金，先爬最近1年
        sdate = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        print(f'  → 首次爬取 {sdate} ~ {today}', end='', flush=True)
        rows = fetch_range(code, sdate, today)
        if rows:
            new_records.extend(rows)
            print(f' +{len(rows)}条')
        else:
            print(' 无数据')
            return entry

    # === 2. 向前追溯（每批1年，直到接口返回空） ===
    # 确定当前已有的最早日期
    all_so_far = merge_records(existing_records, new_records)
    if all_so_far:
        earliest = all_so_far[0]['date']
    elif start_date:
        earliest = start_date
    else:
        return entry

    batch = 0
    max_back_batches = 20  # 最多向前追溯20年
    while batch < max_back_batches:
        batch_end = (datetime.strptime(earliest, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        batch_start = (datetime.strptime(batch_end, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')

        print(f'  ← 追溯 {batch_start} ~ {batch_end}', end='', flush=True)
        rows = fetch_range(code, batch_start, batch_end)
        if not rows:
            print(' 已到最早')
            break

        new_records.extend(rows)
        print(f' +{len(rows)}条')
        earliest = batch_start
        batch += 1
        time.sleep(random.uniform(0.2, 0.5))

    # === 3. 合并去重 ===
    merged = merge_records(existing_records, new_records)

    if merged:
        return {
            'start_date': merged[0]['date'],
            'end_date': merged[-1]['date'],
            'records': merged,
        }
    return entry


def main():
    with open('data/fund_codes.json', encoding='utf-8') as f:
        codes = json.load(f)

    store = load_store()
    total = len(codes)

    for i, code in enumerate(codes):
        entry = store.get(code, {})
        sd = entry.get('start_date', '无')
        ed = entry.get('end_date', '无')
        n = len(entry.get('records', []))
        print(f'[{i+1}/{total}] {code}  已有: {sd} ~ {ed} ({n}条)')

        updated = fetch_fund_incremental(code, entry)
        store[code] = updated

        new_n = len(updated.get('records', []))
        if new_n > n:
            print(f'  ✓ 更新后: {updated["start_date"]} ~ {updated["end_date"]} ({new_n}条)')
        else:
            print(f'  - 无新增')

        # 每只基金爬完后就持久化，防止中断丢失
        save_store(store)

    # 导出兼容格式
    export_nav_history(store)

    print(f'\n完成！共 {len(store)} 只基金')
    for code in codes:
        e = store.get(code, {})
        print(f'  {code}: {e.get("start_date","?")} ~ {e.get("end_date","?")} ({len(e.get("records",[]))}条)')


if __name__ == '__main__':
    main()