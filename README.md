# financial-analysis

一些自用的金融量化分析工具，覆盖基金超额收益与技术信号、A 股长线/短线策略、龙虎榜/游资行为跟踪、参数搜索和本地可视化配置台。基金报告与 A 股策略台统一整合在一个本地 Flask 工作台中，一条命令即可启动：`python run.py --port 8765`。

当前版本：v3.2.0

> 仅用于个人研究、复盘和辅助分析，不构成任何投资建议。外部数据源可能延迟、缺失或变更接口，所有结果都应结合原始数据与人工判断复核。

## 目录

- [1. 项目预览](#1-项目预览)
- [2. 功能概览](#2-功能概览)
- [3. 快速开始](#3-快速开始)
- [4. 基金分析](#4-基金分析)
- [5. A 股策略配置台](#5-a-股策略配置台)
- [6. 游资雷达](#6-游资雷达)
- [7. 行业周期](#7-行业周期)
- [8. 输出文件](#8-输出文件)
- [9. 项目结构](#9-项目结构)
- [10. 运行提示](#10-运行提示)
- [11. 更新历史](#11-更新历史)
- [12. Acknowledgment](#12-acknowledgment)

## 1. 项目预览

### 1.1 基金量化报告

<img src="img/基金量化示例1.png" alt="基金量化报告总览" width="100%">

<img src="img/基金量化示例2.png" alt="基金技术分析信号" width="100%">

### 1.2 A 股策略配置台

<img src="img/股票平台示例1.png" alt="A 股长线策略配置台" width="100%">

<img src="img/股票平台示例2.png" alt="A 股短线策略配置台" width="100%">

## 2. 功能概览

所有模块统一在一个本地 Flask 工作台中运行：`python run.py --port 8765`。首页 `/` 是工作台入口，`/fund` 查看基金报告，`/stock` 进入 A 股策略台，数据抓取与策略计算仍由独立脚本完成。

| 模块 | 解决的问题 | 主要输出 |
| --- | --- | --- |
| 基金超额收益报告 | 对持仓基金和关注基金做跨周期超额收益比较，并提示基金经理变动 | `data/fund_report_data.json`（Flask `/fund` 页面渲染） |
| 基金技术分析 | 结合 MA、RSI、MACD、KDJ、布林带、ADX、ATR、百分位和止盈逻辑生成买卖信号 | `data/signals.json` |
| A 股长线策略 | 面向 2-5 年持有期，从细分行业龙头池中筛选质量、价值、盈利、低波、反转/动量等多因子候选 | `data/stock_advanced_strategy_results.json` |
| A 股短线策略 | 面向 1-5 个交易日，围绕龙虎榜、游资席位、机构共振、价量和风控因子选股 | 同上 |
| 参数搜索 | 对长线/短线策略做 Optuna/TPE 搜索和代理回测，写入优化后的默认参数 | `data/stock_strategy_optimized_config.json` |
| 统一 Flask 工作台 | 在本地网页中查看基金报告、调参运行 A 股策略、保存配置、查看入选股票和关键因子 | `python run.py --port 8765`（`/`、`/fund`、`/stock`） |
| 游资雷达 | 实验性识别盘中拉升票和盘后潜伏吸筹票，并支持命中率验证 | `data/capital/hot_money_*.json` |

## 3. 快速开始

首先安装环境：

```bash
pip install -r requirements.txt
```

直接使用以下命令后台启动 Flask ：

```bash
python run.py --port 8765
```

打开：

```text
http://127.0.0.1:8765
```


## 4. 基金分析

基金模块从天天基金等数据源抓取基金基础信息、历史净值和实时估算，生成结构化报告数据 `data/fund_report_data.json`，由 Flask 工作台的 `/fund` 页面渲染。报告重点不是单只基金的绝对涨跌，而是把基金与指定基准做多周期超额收益比较。

### 4.1 能看什么

- 股票型基金和债券型基金分别展示，支持不同基准指数或基准基金。
- 近一周、1 周到 1 月、1 月到 3 月、3 月到 6 月、6 月到 1 年、1 年到 2 年、2 年到 3 年、3 年到 5 年分段比较。
- 对持仓基金、管理规模偏小/偏大、超额收益显著强弱做颜色标记。
- 检查近 20 天内基金经理是否发生变更。
- 技术分析区展示近 5 年走势、买卖点、MA/RSI/MACD/KDJ/布林带/ADX/ATR/百分位、综合评分和建议。
- 实时估算会作为最新点补入技术序列，便于当日盘中观察。

### 4.2 相关文件

- `funds.py`：配置基金列表、持仓列表和比较基准。
- `fund_fetch_data.py`：统一抓取历史净值和实时估算。
- `fund_storage.py`：基金 SQLite 核心缓存，保存历史净值、实时估算和 Scrapy 基金概况快照。
- `fund_technical_analysis.py`：生成技术指标和买卖信号。
- `fund_generate_output.py`：生成 `data/fund_report_data.json`（HTML 由 Flask `/fund` 页面渲染）。

### 4.3 基金分析命令行工具

基金分析报告可以直接运行：

```bash
bash fund_run.sh
```

如果境内数据源通过本地代理容易失败，可以直连运行：

```bash
FUND_CRAWL_NO_PROXY=1 bash fund_run.sh
```

## 5. A 股策略配置台

A 股模块分为长线和短线两套策略，统一由 `stock_advanced_strategies.py` 评分，既可以命令行运行，也可以通过本地 Dashboard 调参。

### 5.1 长线大盘股策略

长线策略面向 2-5 年持有期，核心目标是从申万三级细分行业龙头池中筛选财务质量稳定、有一定安全边际、流动性足够、估值和风险相对合理的股票。v3.2.0 起，细分行业龙头池先按行业内规模、ROE、成长打分，再交给长线多因子模型二次筛选；其中规模口径优先使用具体市值，若该行业任一成员缺失市值，则统一回退到官方接口返回的“市值占比”。主要因子包括：

- 规模与流动性：总市值、小市值弹性、成交额、行业规模地位、细分行业龙头分。
- 质量与盈利：ROE 稳定性、ROA、经营盈利能力、毛利资产比、现金流质量、低应计、Piotroski F 分。
- 价值与股东回报：账面市值比、盈利收益率、现金流收益率、营收市值比、股息率、连续分红。
- 风险与稳健性：低波动、低负债、低质押、杠杆改善、资产扩张约束。
- 价格行为：一月反转、12-1 月动量、52 周高点距离、长期反转、异常换手。

### 5.2 短线龙虎榜策略

短线策略面向 1-5 个交易日，重点观察龙虎榜、机构席位、游资共振、资金强度、价量形态和交易可行性。典型因子包括：

- 龙虎榜近期上榜次数、净买额、净买占成交、买方主导度、净买占流通市值。
- 游资共振数、知名游资占比、席位多样性、席位持续性、席位平均买额。
- 机构净买、机构/游资共振、机构分歧惩罚。
- 均线多头、量比、RSI 甜区、MACD 强度、短反 Alpha、涨停热度。
- 连板约束、过热惩罚、一字板/T 字板等可交易性过滤。

### 5.3 股票数据持久化

v3.1.4 起，A 股主数据不再以 `data/stock_data/CN_{code}_{name}.json` 作为主链路，而是统一写入 `data/stock_data.sqlite3`。`stock_storage.py` 负责建库、连接、schema 版本、upsert 和旧 JSON 导入：`stock_meta` 以 6 位股票代码为主键，保存名称、抓取时间、行业、质押、日线统计，以及财报、指标、分红、候选来源等 JSON blob；`stock_history` 按 `(code, date)` 存前复权日线 OHLCV、换手、涨跌幅和估值字段；`sw3_member` 保存申万三级成分、官方市值占比和本地主库回补后的市值/ROE/成长字段；`index_nav` / `index_nav_meta` 保存 510310、510580 等基准 ETF NAV。主键从文件名切到代码后，股票改名只更新 `name`，不会造成旧缓存失联。

股票爬虫、刷新、策略、游资雷达和优化器都改为优先读写 SQLite：`stock_crawl_price_valuation.py` 先维护细分行业龙头池并把长历史写入 `stock_history`，`stock_crawl_fundamentals.py --mode full` 随后只补齐龙头股票的基本面，`stock_data_refresh.py` 的健康检查和 fallback 面向 DB 表计数，`stock_advanced_strategies.py` 用 `iter_history()` / `db_signature()` 构建和失效候选池缓存，`stock_hot_money_radar.py` 回放 K 线优先读 `stock_history`，`stock_strategy_optimizer.py` 优先从 `index_nav` 读取 510310+510580 等权基准。`stock_storage.import_stock_data_dir()` 保留旧 `CN_*.json` 批量导入，便于已有缓存过渡。

### 5.4 参数搜索与可视化

`stock_strategy_optimizer.py` 会搜索策略权重和硬过滤参数。长线采用 **Point-in-Time (PIT) walk-forward 回测**消除前视偏差：每个历史折以全市场交易日历的某个时点为基准，财报按 A 股法定披露截止日（年报次年 4-30、季报 4-30/8-31/10-31）、价格/估值/分红按当时可见切片后重算因子再选股——绝不用未来数据选过去的股。v3.2.0 当前口径为每 30 个交易日取一个折起点、固定持有 30 个交易日；组合等权收益减成本后，对比 510310 沪深300ETF 与 510580 中证500ETF 按日等权再平衡的混合基准。折样本按确定性随机键切成约 60/40 的训练/验证集，并对越接近现在的折给更高权重；选参同时看训练折和验证折，并惩罚 train/val 差距、最差单折超额、尾部 CVaR、持有期回撤、折数不足和验证折为负。高点回撤过滤开关参与搜索，开启时阈值搜索范围为 40%-70%。

> 注意：受本地约 10 年日线与基准 ETF 覆盖区间所限，PIT 有效折数有限，超额数值是相对而非可直接兑现的收益；且沪深 300 成分与质押用的是当前快照（无历史数据），这两维仍有轻微残留前视。默认搜索 300 次，运行时间取决于本地缓存规模。

Dashboard 支持：

- 长线/短线 tabs 切换。
- 调整硬过滤参数、因子权重、输出数量和最低分。
- 运行策略、保存配置、重置参数。
- 直接触发 300 次参数搜索，并查看后台搜索状态。
- 展示候选数、入选数、平均分、分数区间、数据覆盖率、筛选解释和主要因子贡献。


### 5.5 常用CLI命令

首次使用推荐后台跑以下命令组合

```bash
python stock_data_refresh.py --mode full --no-proxy
python stock_strategy_optimizer.py --iterations 300
```

股票数据刷新

```bash
python stock_data_refresh.py --mode full --timeout 1800 --no-proxy
python stock_data_refresh.py --mode quick --timeout 1800
python stock_data_refresh.py --mode capital-only --timeout 1800
```

股票策略

```bash
python run.py --port 8765
python stock_advanced_strategies.py --persist
python stock_advanced_strategies.py --persist --rebuild-cache
python stock_advanced_strategies.py --strategy long --json
python stock_advanced_strategies.py --strategy short --json
python stock_strategy_optimizer.py --iterations 300
```

`stock_data_refresh.py` 的 full 流程会先刷新细分行业龙头行情与估值，再补齐龙头股票基本面，随后刷新基准 ETF、指数成分、龙虎榜/游资候选，最后自动运行 `stock_advanced_strategies.py --persist --rebuild-cache` 生成最新策略结果并预构建候选池缓存。申万三级 membership 默认每轮滚动刷新最旧 15 个行业：legulegu 总表可用时优先使用 legulegu 成分接口，成员接口失败才回退官方；总表不可用时直接用官方数据刷新最旧 15 个。日常打开 Flask 不再启动时重算候选池；如果只是修改因子权重、最低分或输出数量，Dashboard 会复用缓存中的候选池并快速重打分。

## 6. 游资雷达

`stock_hot_money_radar.py` 是 v3.1 新增的实验性模块，按「潜伏吸筹 → 拉升 → 出货」三阶段理解短线资金行为。

v3.1.4 起雷达改为双分制：默认扫描会同时输出吸筹分、拉升分、阶段判断和操作建议；`--ambush` 模式也会补充盘后拉升风险分，避免只看潜伏分而忽略已启动或过热状态。历史 K 线优先从 `stock_history` 读取，缺失时才实时回退，`data/capital/ambush_cache` 现主要保留股东户数缓存。

### 6.1 拉升雷达

默认模式用于盘中扫描正在进攻的股票。它综合盘口异动、同花顺大单/即时资金流、涨停池和本地席位记忆，输出短线进攻候选：

```bash
python stock_hot_money_radar.py
python stock_hot_money_radar.py --watch 60
python stock_hot_money_radar.py --verify
```

### 6.2 潜伏雷达

`--ambush` 模式用于盘后扫描可能处于吸筹阶段、尚未明显启动的股票。它关注资金持续净流入但价格未明显上涨、波动收敛、低位横盘、温和放量、股东户数下降、历史龙虎榜试盘痕迹、入选新鲜度（刚出现指纹加分、挂榜多日未启动反而降权）等行为指纹：

```bash
python stock_hot_money_radar.py --ambush
python stock_hot_money_radar.py --ambush --skip-holders
python stock_hot_money_radar.py --ambush-verify --date 2026-06-13 --horizon 5
python stock_hot_money_radar.py --ambush-backtest --days 15 --horizon 5
STOCK_CRAWL_NO_PROXY=1 python stock_hot_money_radar.py --ambush-backtest --days 60 --horizon 3 --top 5
```

### 6.3 使用边界

- 盘中没有席位级实时数据，雷达输出是行为推断，不是席位实锤。
- 潜伏吸筹可能持续多天到数周，高分不代表马上启动。
- 建议先积累 `--verify`、`--ambush-verify` 和 `--ambush-backtest` 的命中率，再决定是否纳入交易流程。


## 7. 行业周期

`industry_cycle_extractor.py` 是 v3.1.2 新增的行业指数周期位置提取能力骨架，后续用于把宽基指数、一级行业、热门行业、大宗商品、聪明资金行为和景气度数据统一成周期位置特征。

计划覆盖的数据维度包括：

- 宽基与市场指数：万得全 A（除科创板）、上证指数、沪深300除金融、上证50、中证500、中证1000、国证2000、创业板指、创业板50、港股、短债、长债。
- 行业指数：申万一级行业指数周期位置（31 个）、热门行业指数周期位置（10+）。
- 大宗商品：黄金、白银、原油的周期位置。
- 行业强弱：每周更新的行业强弱模型数据。
- 聪明资金行为：周期底部更有效的进场动作模型。
- 景气度：行业景气度跟踪。

当前文件只搭建任务拆解、数据源占位、接口和输出结构，真正的数据抓取、特征计算和策略接入会在后续版本继续实现。

```bash
python industry_cycle_extractor.py
```

## 8. 输出文件

| 文件 | 说明 |
| --- | --- |
| `data/fund_report_data.json` | 基金超额收益和技术信号报告数据，由 Flask `/fund` 页面渲染（生成物，不入库） |
| `data/signals.json` | 基金技术信号 |
| `data/fund_data.sqlite3` | 基金核心缓存，包含历史净值、实时估算和 Scrapy 基金概况快照 |
| `data/stock_advanced_strategy_results.json` | A 股长线/短线策略结果 |
| `data/stock_strategy_candidate_cache.json` | A 股长线/短线候选池缓存，由 `stock_data_refresh.py` 刷新后重建，用于 Dashboard 快速调参 |
| `data/stock_strategy_optimization.json` | 参数搜索过程和结果摘要 |
| `data/stock_strategy_optimized_config.json` | Dashboard 默认读取的优化参数 |
| `data/capital/segment_leader_pool.json` | 申万三级细分行业龙头池，包含行业内龙头分、规模口径和候选来源 |
| `data/stock_data.sqlite3` | A 股主数据库：`stock_meta` 存个股财报、指标、分红、日线统计和质押等 meta JSON，`stock_history` 存长历史 OHLCV 与估值序列，`sw3_member` 存申万三级成分和官方市值占比，`index_nav` / `index_nav_meta` 存基准 ETF NAV |
| `data/stock_data/CN_*.json` | 旧版个股 JSON 缓存；v3.1.4 主流程不再依赖，可通过 `stock_storage.import_stock_data_dir()` 批量导入 SQLite |
| `data/stock_data_refresh_report.json` | 数据刷新步骤、耗时和失败信息 |
| `data/capital/hot_money_candidates.json` | 龙虎榜/游资候选池原始信号，短线最终分数由 `stock_advanced_strategies.py` 计算 |
| `data/capital/hot_money_radar*.json` | 拉升雷达结果与每日快照 |
| `data/capital/hot_money_ambush*.json` | 潜伏雷达结果与每日快照 |
| `data/capital/ambush_backtest.json` | 潜伏雷达历史回放结果 |

## 9. 项目结构

```text
.
├── app/                           # Flask 统一工作台、路由、模板和静态资源
├── run.py                         # Flask 本地启动入口
├── fund_*.py                     # 基金数据、技术分析、回测和报告生成
├── fund_storage.py                # 基金 SQLite 缓存 schema、读写和导入工具
├── funds.py                      # 基金列表和基准配置
├── stock_advanced_strategies.py   # A 股长线/短线策略引擎
├── stock_strategy_optimizer.py    # 参数搜索和代理回测
├── stock_data_refresh.py          # 股票数据刷新编排
├── stock_crawl_common.py          # 股票爬虫公共文件、JSON、历史行情和日线统计工具
├── stock_storage.py               # A 股 SQLite 持久化层、schema、导入和读写工具
├── stock_crawl_segment_leaders.py # 申万三级行业 membership 与细分行业龙头池
├── stock_crawl_*.py               # 股票基础数据、指数池、龙虎榜/资金数据抓取
├── stock_hot_money_radar.py       # 游资拉升/潜伏雷达
├── industry_cycle_extractor.py    # 行业指数周期位置提取能力骨架
├── app/static/vendor/live2d-widget # 本地化 Live2D 小组件运行时资源
├── tests/                         # 核心逻辑和股票策略单测
├── img/                           # README 和说明文档展示图
└── data/                          # 本地缓存、策略结果和刷新报告
```

## 10. 运行提示

- 外部数据接口可能很慢或超时，刷新股票全量数据建议给足 `--timeout 1800`。
- 境内行情接口通过本地代理有时会失败，可使用 `--no-proxy` 或 `FUND_CRAWL_NO_PROXY=1`。
- Dashboard 日常打开优先使用 `--skip-refresh`，需要最新数据时再手动刷新。
- 短线策略和游资雷达强依赖龙虎榜/资金数据，刷新后建议重新跑参数搜索。
- 所有策略结果都来自本地缓存和可得数据，缺失数据会影响排序和评分。

## 11. 更新历史

#### Update v1.1  2021.7
新增该基金的基金经理管理规模提示，小于100亿标红（表示管理规模小），大于300亿标绿（表示管理规模较大）。

#### Update v1.2  2022.1
在fund.py文件中会更新我的持仓，希望市场能让我们写代码赚的辛苦钱持续稳健增值。我的持仓风格是多元化全球资产配置，股8债2，投资中国、美国、香港、日本市场。

#### Update v1.2  2022.5
增加了对中低风险基金的详细对比支持。

#### Update v1.3  2022.11
增加了对中信股指期货持仓的统计。

#### Update v1.4  2023.1
代码细节优化。

#### Update v1.4.1  2023.7.22
适配基金估值下线后的天天基金前端。

#### Update v2.0  2026.4.15
增加技术面指标。增加技术面买卖点推荐。增加html展示。

#### Update v2.0.1  2026.4.15
修复百分位计算问题。

#### Update v2.0.2  2026.4.18
新增买卖点判断逻辑以及止盈卖出点位。

#### Update v2.0.3  2026.4.20
新增60日趋势线指标，新增参数配置功能。修改了repo名字。

#### Update v2.1.0  2026.4.23
新增回测能力 `fund_backtest.py`。

#### Update v2.1.0.post1  2026.4.24
更新股票分析相关骨架。

#### Update v2.2.dev0  2026.5.7
更新左侧交易股票分析相关代码。

#### Update v2.2.dev1  2026.6.1
修复基金爬虫阻塞请求、数据缺失静默吞错、百分位边界、股票文件名兼容与股票抓取失败可见性问题；新增核心纯函数单测。

#### Update v3.0  2026.6.6
上线 A 股长线/短线策略：引入多因子选股、龙虎榜短线模型、数据刷新、参数搜索和本地策略配置台。

#### Update v3.0.1  2026.6.12
优化基金报告生成与数据抓取稳定性，修复净值、估值、回测和短历史信号中的若干边界问题。

#### Update v3.1  2026.6.13
新增实验性游资雷达，扩展长线/短线因子库，并将长线参数搜索升级为 PIT walk-forward 回测以降低前视偏差。

#### Update v3.1.1  2026.6.13
优化长线回测展示和基准对比，统一基金报告与 A 股策略台到同一个本地工作台，并清理旧报告链路。

#### Update v3.1.1.post1  2026.6.14
优化股票爬虫速度和复用结构，小幅调整游资雷达与 Flask 页面体验。

#### Update v3.1.2  2026.6.15
重构 A 股刷新与策略缓存链路，让 Dashboard 复用候选池快速重打分，并为行业周期特征预留能力骨架。

#### Update v3.1.2.post1  2026.6.15
修复股票刷新 fallback 的打印问题，并移除旧策略台入口，统一从本地工作台启动。

#### Update v3.1.3  2026.6.15
基金核心缓存迁移到 SQLite；股票数据刷新同步提速，增强行情 fallback、并发抓取和阶段耗时输出。

#### Update v3.1.4  2026.6.15
A 股主数据迁移到 SQLite，股票代码成为稳定主键；长线搜索升级为 300 次 Optuna/TPE，游资雷达改为吸筹分/拉升分双分制。

#### Update v3.2.0  2026.6.20
长线候选改为细分行业龙头池优先，龙头分简化为规模/ROE/成长，缺市值时用官方市值占比兜底。刷新链路先补龙头行情再补龙头基本面；优化器改为 30 日持有/30 日折间隔、混合宽基基准，并搜索高点回撤过滤开关与 40%-70% 阈值范围。

## 12. Acknowledgment

感谢东方财富、新浪财经、腾讯财经、百度、同花顺、AkShare 以及相关公开数据源。爱您们，感恩！

## 13. License

本项目基于 [MIT License](LICENSE) 开源，详见仓库根目录的 `LICENSE` 文件。
