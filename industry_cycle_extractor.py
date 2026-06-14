"""
行业指数周期提取能力 - 空框架

目标：
  1. 把 README 第 7 节中的「行业周期」能力落成独立模块。
  2. 支持周期位置、聪明资金行为、景气度三类数据。
  3. 先定义数据范围、输出契约和处理流程；真实抓取与模型计算后续再实现。

待支持范围：
  - 周期位置（交易日）：万得全A（除科创板）、上证指数、沪深300除金融、
    上证50、中证500、中证1000、国证2000、创业板指、创业板50、港股、
    短债、长债。
  - 周期位置（交易日）：申万一级行业指数周期位置（31 个）。
  - 周期位置（交易日）：热门行业指数周期位置（10+ 个，后续配置维护）。
  - 周期位置（交易日）：大宗商品周期位置，包括黄金、白银、原油。
  - 周期位置（每周）：行业强势模型数据。
  - 聪明资金行为（交易日）：进场动作模型，只在周期底部有效并公布。
  - 景气度（交易日）：行业景气度。

当前文件只搭架子，不做真实数据拉取，不产出正式信号。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INDUSTRY_CYCLE_DIR = DATA_DIR / "industry_cycle"

CYCLE_POSITION_FILE = INDUSTRY_CYCLE_DIR / "cycle_position_latest.json"
SMART_MONEY_FILE = INDUSTRY_CYCLE_DIR / "smart_money_latest.json"
PROSPERITY_FILE = INDUSTRY_CYCLE_DIR / "industry_prosperity_latest.json"
RUN_REPORT_FILE = INDUSTRY_CYCLE_DIR / "industry_cycle_run_report.json"


class Frequency(str, Enum):
    """数据刷新频率。"""

    TRADING_DAY = "trading_day"
    WEEKLY = "weekly"


class Capability(str, Enum):
    """模块能力分组。"""

    CYCLE_POSITION = "cycle_position"
    SMART_MONEY = "smart_money"
    PROSPERITY = "prosperity"


@dataclass(frozen=True)
class TargetSpec:
    """一个待支持的数据目标。"""

    capability: Capability
    name: str
    symbols: List[str]
    frequency: Frequency
    enabled: bool = True
    note: str = ""


@dataclass
class CycleRecord:
    """单个指数/资产/行业的周期位置结果占位。"""

    symbol: str
    name: str
    as_of: str
    percentile: Optional[float] = None
    phase: str = "unknown"
    score: Optional[float] = None
    source: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SmartMoneyRecord:
    """聪明资金进场动作模型结果占位。"""

    industry: str
    as_of: str
    active: bool = False
    score: Optional[float] = None
    reason: str = ""
    only_valid_near_cycle_bottom: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProsperityRecord:
    """行业景气度结果占位。"""

    industry: str
    as_of: str
    score: Optional[float] = None
    trend: str = "unknown"
    source: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


# 待支持能力清单。后续实现时逐项替换 symbol/source 约定。
TARGET_SPECS: List[TargetSpec] = [
    TargetSpec(
        capability=Capability.CYCLE_POSITION,
        name="宽基与跨资产周期位置",
        symbols=[
            "wind_all_a_ex_star",
            "shanghai_composite",
            "csi300_ex_financial",
            "sse50",
            "csi500",
            "csi1000",
            "cnindex2000",
            "chinext",
            "chinext50",
            "hong_kong_equity",
            "short_bond",
            "long_bond",
        ],
        frequency=Frequency.TRADING_DAY,
        note="万得全A(除科创板)、上证指数、沪深300除金融、上证50、中证500、中证1000、国证2000、创业板指、创业板50、港股、短债、长债。",
    ),
    TargetSpec(
        capability=Capability.CYCLE_POSITION,
        name="申万一级行业指数周期位置",
        symbols=["sw_l1_industries"],
        frequency=Frequency.TRADING_DAY,
        note="31 个申万一级行业指数，后续展开为独立行业代码。",
    ),
    TargetSpec(
        capability=Capability.CYCLE_POSITION,
        name="热门行业指数周期位置",
        symbols=["hot_industry_indices"],
        frequency=Frequency.TRADING_DAY,
        note="10+ 热门行业指数，后续由配置文件维护。",
    ),
    TargetSpec(
        capability=Capability.CYCLE_POSITION,
        name="大宗商品周期位置",
        symbols=["gold", "silver", "crude_oil"],
        frequency=Frequency.TRADING_DAY,
        note="黄金、白银、原油。",
    ),
    TargetSpec(
        capability=Capability.CYCLE_POSITION,
        name="行业强势模型数据",
        symbols=["industry_strength_model"],
        frequency=Frequency.WEEKLY,
        note="每周更新，用于识别相对强势行业。",
    ),
    TargetSpec(
        capability=Capability.SMART_MONEY,
        name="聪明资金行为模型",
        symbols=["smart_money_entry_action"],
        frequency=Frequency.TRADING_DAY,
        note="进场动作模型，只在周期底部有效并公布。",
    ),
    TargetSpec(
        capability=Capability.PROSPERITY,
        name="行业景气度",
        symbols=["industry_prosperity"],
        frequency=Frequency.TRADING_DAY,
        note="行业景气度日频更新，先定义接口，后续接入真实指标。",
    ),
]


class IndustryCycleDataProvider:
    """数据源适配层占位。

    后续要做：
      - 接入指数历史行情：宽基、港股、债券、商品、申万行业、热门行业。
      - 统一交易日历和周频数据对齐。
      - 为聪明资金模型准备资金流、成交额、行业轮动、底部区间等输入。
      - 为行业景气度准备价格、盈利、库存、订单、开工率等指标。
    """

    def fetch_history(self, spec: TargetSpec) -> List[Dict[str, Any]]:
        raise NotImplementedError("后续实现：按 TargetSpec 拉取历史数据")

    def fetch_latest(self, spec: TargetSpec) -> Dict[str, Any]:
        raise NotImplementedError("后续实现：按 TargetSpec 拉取最新数据")


class CyclePositionCalculator:
    """周期位置计算层占位。

    后续要做：
      - 定义周期位置算法：历史分位、滚动高低点、均线偏离、波动状态等。
      - 输出统一 phase：bottom / recovery / middle / overheated / top / unknown。
      - 为不同资产配置不同窗口，例如 1Y/3Y/5Y/10Y。
      - 保留 explain 字段，方便页面展示为什么处于该周期位置。
    """

    def calculate(self, spec: TargetSpec, rows: Iterable[Dict[str, Any]]) -> List[CycleRecord]:
        raise NotImplementedError("后续实现：计算周期位置")


class SmartMoneyModel:
    """聪明资金行为模型占位。

    后续要做：
      - 只在周期底部或底部附近运行/公布模型结果。
      - 识别进场动作：资金持续净流入、放量不大涨、回撤承接、行业内扩散。
      - 明确过滤条件：非底部区间不输出强提示。
    """

    def detect(self, cycle_records: List[CycleRecord]) -> List[SmartMoneyRecord]:
        raise NotImplementedError("后续实现：检测聪明资金进场动作")


class IndustryProsperityModel:
    """行业景气度模型占位。

    后续要做：
      - 定义行业景气度指标清单。
      - 做日频或可用频率的 nowcast。
      - 输出 score/trend，以及指标明细。
    """

    def evaluate(self) -> List[ProsperityRecord]:
        raise NotImplementedError("后续实现：计算行业景气度")


class IndustryCycleExporter:
    """结果输出层占位。"""

    def write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

    def export_plan(self, specs: List[TargetSpec], path: Path = RUN_REPORT_FILE) -> None:
        payload = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "skeleton_only",
            "message": "行业周期模块空框架已建立，真实数据抓取与计算后续实现。",
            "targets": [asdict(spec) for spec in specs],
            "planned_outputs": {
                "cycle_position": str(CYCLE_POSITION_FILE.relative_to(ROOT)),
                "smart_money": str(SMART_MONEY_FILE.relative_to(ROOT)),
                "prosperity": str(PROSPERITY_FILE.relative_to(ROOT)),
                "run_report": str(RUN_REPORT_FILE.relative_to(ROOT)),
            },
        }
        self.write_json(path, payload)


class IndustryCycleExtractor:
    """行业周期提取总控占位。"""

    def __init__(self, specs: Optional[List[TargetSpec]] = None) -> None:
        self.specs = specs or TARGET_SPECS
        self.provider = IndustryCycleDataProvider()
        self.cycle_calculator = CyclePositionCalculator()
        self.smart_money_model = SmartMoneyModel()
        self.prosperity_model = IndustryProsperityModel()
        self.exporter = IndustryCycleExporter()

    def print_plan(self) -> None:
        print("行业指数周期提取能力 - 待实现清单")
        for index, spec in enumerate(self.specs, start=1):
            status = "enabled" if spec.enabled else "disabled"
            print(f"{index}. [{spec.capability.value}] {spec.name} ({spec.frequency.value}, {status})")
            print(f"   symbols: {', '.join(spec.symbols)}")
            if spec.note:
                print(f"   note: {spec.note}")

    def write_plan_report(self) -> None:
        self.exporter.export_plan(self.specs)
        print(f"已写出占位运行报告: {RUN_REPORT_FILE.relative_to(ROOT)}")

    def run(self) -> None:
        raise NotImplementedError("后续实现：串联抓取、计算、模型和导出")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="行业指数周期提取能力空框架")
    parser.add_argument("--plan", action="store_true", help="打印待实现能力清单")
    parser.add_argument("--write-plan", action="store_true", help="写出占位运行报告 JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extractor = IndustryCycleExtractor()

    if args.plan:
        extractor.print_plan()
    if args.write_plan:
        extractor.write_plan_report()
    if not args.plan and not args.write_plan:
        extractor.print_plan()


if __name__ == "__main__":
    main()
