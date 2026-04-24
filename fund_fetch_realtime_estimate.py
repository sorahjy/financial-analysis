

"""
获取基金实时净值估算数据。

数据源: 天天基金估值接口 https://fundgz.1234567.com.cn/js/{code}.js
返回 JSONP: jsonpgz({fundcode, name, jzrq, dwjz, gsz, gszzl, gztime})

输出: data/realtime_estimate.json
结构: { "007345": {"gsz": "1.4367", "gszzl": "0.08", "gztime": "2026-04-14 13:52", "dwjz": "1.4356"}, ... }
"""
import json
import os
import re
import time
import random
import requests


HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
    'Referer': 'https://fund.eastmoney.com/',
}

API_URL = 'https://fundgz.1234567.com.cn/js/{code}.js'
OUTPUT_FILE = os.path.join('data', 'realtime_estimate.json')

JSONP_RE = re.compile(r'jsonpgz\((.*)\)')


def fetch_estimate(code):
    """获取单只基金的实时估算数据，返回 dict 或 None"""
    url = API_URL.format(code=code)
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.encoding = 'utf-8'
        m = JSONP_RE.search(r.text)
        if not m:
            return None
        data = json.loads(m.group(1))
        return {
            'gsz': data.get('gsz', ''),
            'gszzl': data.get('gszzl', ''),
            'gztime': data.get('gztime', ''),
            'dwjz': data.get('dwjz', ''),
        }
    except Exception as e:
        print(f'  [ERROR] {code}: {e}')
        return None


def main():
    with open('data/fund_codes.json', encoding='utf-8') as f:
        codes = json.load(f)

    estimates = {}
    total = len(codes)

    for i, code in enumerate(codes):
        print(f'[{i + 1}/{total}] 获取估算: {code}', end='', flush=True)
        result = fetch_estimate(code)
        if result:
            estimates[code] = result
            print(f'  gsz={result["gsz"]} gszzl={result["gszzl"]}% gztime={result["gztime"]}')
        else:
            print('  无数据')
        if i < total - 1:
            time.sleep(random.uniform(0.1, 0.3))

    os.makedirs('data', exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(estimates, f, ensure_ascii=False, indent=2)

    print(f'\n完成！共获取 {len(estimates)}/{total} 只基金的实时估算数据')


if __name__ == '__main__':
    main()
