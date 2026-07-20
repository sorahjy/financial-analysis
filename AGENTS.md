# AGENTS.md

本文件适用于仓库根目录及所有子目录，供后续编码 agent 和维护者使用。开始修改前先读 [README.md](README.md) 与 [ARCHITECTURE.md](ARCHITECTURE.md)。如果未来某个子目录增加更具体的 `AGENTS.md`，以离目标文件最近的规则为准。

## 1. 工作原则

1. 先确认真实实现，再改文档或代码。入口、CLI、数据源和产物经常一起演进，不要只根据文件名或旧注释推断。
2. 把当前工作区视为用户现场。先运行 `git status --short` 和相关 `git diff`；已有修改、未跟踪研究结果和本地数据库都属于用户，不能顺手清理、覆盖或回滚。
3. 保持改动聚焦。不要把文档任务扩成模型调参，不要把诊断任务扩成生产修复，也不要为通过测试而改变未被请求的策略口径。
4. 用最小但足够的验证证明改动。优先跑直接相关测试，再根据风险决定是否跑完整套件。
5. 所有金融输出都是研究信号。新增页面文案、导出或报告时，不得暗示确定收益或自动交易能力。

## 2. 仓库事实

- 运行环境：Python 3.10+；当前代码使用 `X | None`、`argparse.BooleanOptionalAction` 和内置泛型等语法，文档编写时在 Python 3.12.12 验证。
- Web：Flask + Jinja + 原生 JavaScript/CSS，无前端构建步骤。
- 测试：标准库 `unittest`；没有 `pytest.ini`、`pyproject.toml`、tox 或项目级 lint/format 配置。
- 依赖：`requirements.txt` 没有锁定精确版本，也没有单独的开发依赖文件。
- 主存储：本地 SQLite；运行产物为 JSON/SVG。
- 工作目录很重要：多个存储模块使用相对路径 `Path("data")`，命令必须从仓库根目录执行。
- 网络抓取依赖公开接口，可能受代理、限流、字段变化和非交易日影响。

## 3. 环境与常用命令

macOS、Linux 及基于 Linux/Unix 的国产系统从仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

启动本地工作台：

```bash
python run.py --port 8765
```

跑单个测试文件：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_hot_money_radar.py' -v
```

跑完整测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Windows PowerShell 对应写法：

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m unittest discover -s tests -v
```

交付前至少运行：

```bash
git diff --check
git status --short
```

不要用联网刷新命令充当普通单元测试。测试外部数据适配时注入 fetcher、mock HTTP/AkShare 响应，并用临时目录或 `:memory:` SQLite。

## 4. 写入与副作用分级

### 明确会联网并写本地状态

- `python fund_data_refresh.py`
- `python stock_radar_fresh_data.py`
- `stock_crawl_*.py`、`plate_crawl_history.py` 等抓取器
- `stock_data_refresh.py`
- `stock_hot_money_radar.py` 的生产/实验模式
- `stock_crawl_news.py`、`mf_pilot.py crawl`

除非任务明确要求刷新真实数据，否则不要在验证阶段运行这些命令。

两个 `.py` 刷新入口都会复用当前解释器，并在 Windows 与 POSIX 系统上通过参数列表启动子进程。`fund_data_refresh.py` 保持为根目录基金刷新入口，内部子步骤通过 `python -m fund...` 运行。

### 看似读取、实际仍可能写

- `stock_storage.connect()` 会执行 schema 初始化/迁移；连接真实 `data/stock_data.sqlite3` 不能视为严格只读。
- `stock_advanced_strategies.py` 即使不带 `--persist`，候选缓存缺失或失效时也可能写 `data/stock_strategy_candidate_cache.json`；`--rebuild-cache` 还会先删除旧缓存。
- `stock_strategy_optimizer.py` 默认持久化配置和报告；只做试验时至少使用 `--no-persist`，但仍要考虑数据库连接迁移和运行成本。
- 启动 Flask 会预热股票缓存；不要把启动服务当作零副作用语法检查。

严格只读研究应像 `research_p15_optimization.py` 一样用 SQLite URI `mode=ro`，并拒绝在数据库不存在时自动创建。

## 5. 文件归属

### 源码与版本化依据

- `app/`、`fund/`、顶层 `*.py`、`*.sh`、`tests/`：生产代码和测试。
- `README.md`：用户能力、操作命令和使用边界。
- `ARCHITECTURE.md`：组件边界、数据流、存储与不变量。
- `AGENTS.md`：维护工作流和验证要求。
- `meta_data_backup/`：版本化默认配置与研究证据。它不是普通临时目录，修改时要能解释研究依据。

### 本地状态与生成物

- `data/*` 默认被 Git 忽略，是数据库、缓存和运行报告；不要手工编辑来伪造成功结果。
- `reports/`、`outputs/` 常含用户本地分析产物。除非任务明确要求，不要重写、删除或纳入代码改动。
- `.scrapy/`、`__pycache__/`、测试缓存均为生成物。
- `.gitignore` 默认忽略 `research_*.py`，仅显式放行了 `research_p15_optimization.py`。新增要提交的研究脚本时必须有意识地调整例外，而不是依赖强制添加。
- 仓库没有统一 formatter/linter；不要借功能改动做全文件格式化。

## 6. 代码组织约定

### Web

- `app/routes/`：HTTP 参数解析、状态码和序列化。
- `app/services/`：读取、缓存、任务启动与业务编排。
- `fund/` 与顶层量化模块：可测试的策略、形态、存储和抓取逻辑。
- `app/templates/`、`app/static/js/`、`app/static/css/`：页面结构、行为和样式。

不要在路由中复制大段策略逻辑，也不要让模板直接理解 SQLite schema。新的长任务应接入 `job_service.start_command_job()`；会读写同一股票数据库或雷达产物的任务必须复用 `stock-data-refresh` 资源锁。

### 存储

- 通过 `fund/fund_storage.py`、`stock_storage.py`、`plate_storage.py` 提供的 API 访问主库。
- Schema 改动要在 `ensure_schema()` 中提供幂等升级，并递增对应 schema 版本；不能要求用户删库重建。
- 证券代码统一为六位字符串。名称可更新，代码才是稳定主键。
- `stock_meta.history_coverage_start_date` 只有在响应同时覆盖请求起点和已知交易日锚点时才可写入；普通增量更新只能保留或向更早日期推进，全量历史替换必须重新验证并允许清空旧边界。
- JSON/SVG 写入应使用现有原子写或公共 helper，失败时保留上一份完整产物。
- 避免在路由、前端服务和研究脚本中新增散落的生产写 SQL。

### 抓取器

- 保留超时、重试、fallback、代理绕过和可见失败。
- 允许依赖注入 fetcher，保证测试不联网。
- 空响应不能默认当作成功，也不能轻易用空集合覆盖上一次有效股票池或完整快照。
- 新字段要同时检查单位、日期格式、前复权口径、缺失值和多源字段名差异。
- 当前股票历史成交量统一按“手”理解；接入新源时必须先核实并转换单位。

### 前端

- 保持 API 字段与模板/JS 同步；前端契约通常由 `tests/test_flask_app.py` 做静态或响应断言。
- 没有 bundler，修改的 `.js`/`.css` 会直接被浏览器加载。
- 异步请求必须处理任务切换、旧响应覆盖新状态、按钮 busy 状态和错误提示。
- 图表计算与生产信号口径应复用后端返回的数据，不能在 JS 中悄悄重写另一套模型。

## 7. 金融与数据不变量

以下约束属于高风险区，改动前必须定位现有测试并补充回归：

1. **PIT**：日期 `t` 的特征只能使用 `t` 及以前可见信息；未来收益只用于评估。
2. **前复权**：策略历史与形态基于前复权日线。盘中原始现价必须投影到本地前复权尺度后再参与形态计算。
3. **候选池**：`leader`、`hotmoney`、`etf` 的成员来源不同，不得因数据缺失互相回退。
4. **ETF 隔离**：ETF 与股票可共用 `stock_history`，但 `stock_meta.instrument_type` 必须正确；ETF 不得进入财报、股东、质押、回购、龙虎榜或股票策略全库扫描。
5. **评分语义**：机会分和反转分是池内相对排序，不是上涨概率；阶段标签是结构解释，不应重复参与加分。
6. **验证范围**：股票池验证过的形态不能自动标记为 ETF 有效；研究标签要注明池、样本期和基准。
7. **刷新完整性**：多步骤刷新依赖失败时，保留上一份完整策略结果，不写新旧混合快照。
8. **实时只读**：雷达盘中重算和页面形态回放默认不写离线 JSON/SQLite。
9. **配置安全**：基金页面只接受 AST 校验后的静态六位代码列表，不得恢复为执行任意 Python。
10. **并发写**：刷新、优化、雷达重算等写任务必须串行协调。
11. **SW3 热度完整性**：行业热度只比较同一截止日的 20 个共同交易日；缺失不补零，有效行业低于 80%、热门排名未覆盖全部有效行业、升温/降温排名与正向/负向候选集合不一致，或合并趋势排名未覆盖两者并集时保留旧原子报告。行业市值必须同时输出成员覆盖率和估算标记。

## 8. 按改动类型选择测试

| 改动范围 | 优先测试 |
| --- | --- |
| 基金抓取、存储、配置或报告 | `test_core_logic.py`、`test_data_integrity_fixes.py`、`test_flask_app.py` |
| 股票存储与通用抓取 | `test_core_logic.py`、`test_data_integrity_fixes.py`、`test_etf_radar.py` |
| 长线/小盘/短线策略 | `test_stock_advanced_strategies.py`、`test_stock_strategy_optimizer.py`、`test_short_universe_integration.py` |
| 主力资金雷达或形态 | `test_hot_money_radar.py`、`test_flask_app.py`；涉及 ETF 时再加 `test_etf_radar.py` |
| 独立短线雷达 | `test_stock_short_term_radar.py` |
| 行业周期/板块存储 | `test_industry_cycle_engine.py`、`test_data_integrity_fixes.py` |
| SW3 归属、细分龙头或三级行业热度 | `test_core_logic.py`、`test_sw3_industry_heat.py`、`test_data_integrity_fixes.py`；接入页面再加 `test_flask_app.py` |
| 后台任务 | `test_job_service.py`、对应 Flask API 测试 |
| 只读研究脚本 | 使用脚本自身的离线/临时验证；不要让通用 `tests/` 依赖被忽略的 `research_*.py`；确认真实数据库不被创建或迁移 |
| 模板、JS、CSS | `test_flask_app.py`，并在相关页面做人工交互检查 |

测试新增要求：

- 正常路径和失败路径都要覆盖。
- SQLite 用临时库，至少验证新 schema、旧 schema 升级和默认查询过滤。
- 时间序列测试明确 as-of 日期，构造能暴露前视偏差的未来数据。
- 外部接口测试固定字段样本，不依赖当天网络结果。
- 算法口径改变时同时断言数值、排序和解释元数据，避免前端显示与后端模型漂移。

## 9. 当前测试基线

截至 2026-07-20，最近一次从仓库根运行共发现 387 个测试，全部通过，无模块导入错误。

后续 agent 应以这份全绿结果为基线；出现任何新失败时都要定位原因，并在测试数量或基线状态变化后同步更新本节。

## 10. 文档同步规则

- 用户可见功能、CLI、默认值、路径或风险边界变化：更新 `README.md`。
- 组件职责、数据流、数据库表、候选池或并发模型变化：更新 `ARCHITECTURE.md`。
- 开发命令、测试基线、生成物规则或高风险约束变化：更新 `AGENTS.md`。
- 因子公式、形态判据或研究有效性变化：更新代码 catalog、测试、README 说明和对应 `meta_data_backup/` 研究记录。
- 未发布工作写“未发布变更”或 “Unreleased”，不要在没有 tag/发布决定时自行发明版本号。

## 11. 交付检查清单

- [ ] 已查看并保留用户原有修改。
- [ ] 改动位于正确层，没有复制核心逻辑。
- [ ] 未把本地 `data/`、`reports/` 或 `outputs/` 当源码改写。
- [ ] 相关测试已跑，失败与任务前基线已区分。
- [ ] `git diff --check` 通过。
- [ ] CLI、API、数据结构和文档互相一致。
- [ ] 涉及 PIT、复权、ETF 隔离、候选池或并发写时已有专项回归。
- [ ] 最终说明列出改动、验证和仍存在的已知问题。
