# 股票策略框架运行教程

本文档说明如何运行 A 股长线/短线策略框架、刷新数据、启动可视化配置页面，以及重新搜索默认参数。

## 1. 进入项目目录

```bash
cd /Users/sorahjy/Downloads/SimCSE-Chinese/financial-analysis
```

## 2. 首次准备环境

```bash
pip install -r requirements.txt
```

## 3. 启动可视化策略页面

默认启动会先刷新数据，再启动 server。刷新会访问行情、指数、龙虎榜等外部数据源，耗时可能较长。

```bash
python stock_strategy_dashboard.py --port 8765
```

启动后打开：

```text
http://127.0.0.1:8765
```

## 4. 已经刷新过数据时快速启动

如果刚刚已经刷新过数据，只想立刻打开页面：

```bash
python stock_strategy_dashboard.py --port 8765 --skip-refresh
```

## 5. 单独刷新数据

全量刷新：

```bash
python stock_data_refresh.py --mode full --timeout 1800
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

## 6. 重新搜索默认参数

刷新完数据后，可以重新跑 100 次参数搜索，更新默认配置：

```bash
python stock_strategy_optimizer.py --iterations 100
```

输出文件：

```text
data/stock_strategy_optimization.json
data/stock_strategy_optimized_config.json
```

## 7. 单独生成策略结果

如果只想按当前默认配置生成最新长线/短线选股结果：

```bash
python stock_advanced_strategies.py --persist
```

输出文件：

```text
data/stock_advanced_strategy_results.json
```

## 8. 常用验证命令

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

## 9. 运行提示

- 日常打开页面优先用 `--skip-refresh`，避免每次启动都等待外部数据源。
- 需要最新数据时再运行 full/quick/capital-only 刷新。
- `stock_crawl_top_800_data.py` 的外部接口可能很慢或超时；刷新器会记录失败，并用本地 `stock_data + market_snapshot` 兜底补齐 `data/CN_stock`。
- 短线策略依赖龙虎榜数据，刷新后建议重新跑一次参数搜索。
- 页面默认读取 `data/stock_strategy_optimized_config.json` 作为优化后的默认参数。
