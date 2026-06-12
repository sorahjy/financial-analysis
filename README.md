# financial-analysis

一些自用的金融指标分析工具

当前版本：v3.0

## 1. Requirements

* pip install -r requirements.txt


## 2. 基金相关配置与使用
### 2.1 STEP 1 参数修改

* funds.py 里可以配置具体的基金。

* technical_analysis.py 中可以设置各个指标的参数。

### 2.2 STEP 2 运行

* bash fund_run.sh

### 2.3 Outputs
会输出一个html文件 'fund_report.html' 展示：
 * 基金近期超额收益表现。
 * 基金当日的买卖指标和技术面指标。基金的历史买卖点。 
 * 提示经理人事变动情况

## 3. 股票相关配置与使用

下面将说明如何运行 A 股长线/短线策略框架、刷新数据、启动可视化配置页面，以及重新搜索默认参数。

- 第一次使用建议按照以下步骤
  - 3.3节命令先拉取全量数据
  - 3.4节命令搜索因子初始超参
  - 3.2节命令启动服务
- 日常打开页面优先用 `--skip-refresh`，避免每次启动都等待外部数据源。
- 需要最新数据时再运行 full/quick/capital-only 刷新。
- `stock_crawl_top_800_data.py` 的外部接口可能很慢或超时；刷新器会记录失败，并用本地 `stock_data + market_snapshot` 兜底补齐 `data/CN_stock`。
- 短线策略依赖龙虎榜数据，刷新后建议重新跑一次参数搜索。
- 页面默认读取 `data/stock_strategy_optimized_config.json` 作为优化后的默认参数。


### 3.1 启动可视化策略页面

默认启动会先刷新数据，再启动 server。刷新会访问行情、指数、龙虎榜等外部数据源，耗时可能较长。



```bash
python stock_strategy_dashboard.py --port 8765
```

启动后打开：

```text
http://127.0.0.1:8765
```

### 3.2 已经刷新过数据时快速启动

如果刚刚已经刷新过数据，只想立刻打开页面：

```bash
python stock_strategy_dashboard.py --port 8765 --skip-refresh
```

### 3.3 单独刷新数据

全量刷新：

```bash
python stock_data_refresh.py --mode full --timeout 1800 --no-proxy
```

只刷新短线龙虎榜/主力席位数据：

```bash
python stock_data_refresh.py --mode capital-only --timeout 1800
```

快速刷新模式：

```bash
python stock_data_refresh.py --mode quick --timeout 1800
```

刷新报告会写入：

```text
data/stock_data_refresh_report.json
```

### 3.4 重新搜索默认参数

刷新完数据后，可以重新跑 100 次参数搜索，更新默认配置：

```bash
python stock_strategy_optimizer.py --iterations 100
```

输出文件：

```text
data/stock_strategy_optimization.json
data/stock_strategy_optimized_config.json
```

### 3.5 单独生成策略结果

如果只想按当前默认配置生成最新长线/短线选股结果：

```bash
python stock_advanced_strategies.py --persist
```

输出文件：

```text
data/stock_advanced_strategy_results.json
```

### 3.6 常用验证命令

检查 dashboard 健康状态：

```bash
curl --noproxy '*' -s http://127.0.0.1:8765/api/health
```

运行一次策略 API：

```bash
curl --noproxy '*' -s -X POST http://127.0.0.1:8765/api/run \
  -H 'Content-Type: application/json' \
  -d '{}'
```

运行单元测试：

```bash
python -m unittest tests/test_stock_advanced_strategies.py
```

## 4. 更新历史

#### Update v1.1  2021.7
新增该基金的基金经理管理规模提示，小于100亿标红（表示管理规模小），大于300亿标绿（表示管理规模较大）。

#### Update v1.2  2022.1
在fund.py文件中会更新我的持仓，希望市场能让我们写代码赚的辛苦钱持续稳健增值。我的持仓风格是多元化全球资产配置，股8债2，投资中国、美国、香港、日本市场。

#### Update v1.2  2022.5
增加了对中低风险基金的详细对比支持。

#### Update v1.3  2022.11
增加了对中信股指期货持仓的统计。

#### Update v1.4  2023.1
代码细节优化

#### Update v1.4.1  2023.7.22
适配基金估值下线后的天天基金前端

#### Update v2.0  2026.4.15
增加技术面指标。增加技术面买卖点推荐。增加html展示。

#### Update v2.0.1  2026.4.15
修复百分位计算问题

#### Update v2.0.2  2026.4.18
新增买卖点判断逻辑以及止盈卖出点位

#### Update v2.0.3  2026.4.20
新增60日趋势线指标，新增参数配置功能。修改了repo名字。

#### Update v2.1.0  2026.4.23
新增回测能力 backtest.py

#### Update v2.1.0.post1  2026.4.24
更新股票分析相关骨架

#### Update v2.2.dev0  2026.5.7
更新左侧交易股票分析相关代码

#### Update v2.2.dev1  2026.6.1
修复基金爬虫阻塞请求、数据缺失静默吞错、百分位边界、股票文件名兼容与股票抓取失败可见性问题；新增核心纯函数单测。

#### Update v3.0  2026.6.12
上线 A 股长线/短线策略 v3.0：新增多因子策略引擎、龙虎榜/游资共振短线模型、T+1/T+2 回测、数据刷新编排、参数搜索和本地可视化配置台；优化股票行情爬取的代理控制、数据源回退和历史回补；补充股票策略使用文档与单元测试；同步刷新基金报告输出。

## 5. Acknowledgment

感谢东方财富，新浪财经，腾讯财经，百度。爱您们，感恩！
