from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request

from app.config import ROOT_DIR
from app.services.fund_report_service import load_fund_report_view
from app.services.job_service import get_job_state, is_resource_running, start_command_job
from app.utils import file_status


bp = Blueprint("fund", __name__)
FUND_EDITOR_FILE = ROOT_DIR / "funds.py"
FUND_CODES_FILE = ROOT_DIR / "data" / "fund_codes.json"

_FUND_LIST_NAMES = (
    "compare_index",
    "compare_index_bond",
    "hold_index",
    "fund_index",
    "fund_stock",
    "fund_bond",
)
_MAX_CODES_PER_LIST = 1000
_MAX_EDITOR_BYTES = 256 * 1024
_FUND_RESOURCE_KEY = "fund-data-refresh"
_FUND_CONFIG_LOCK = threading.RLock()


class FundConfigError(ValueError):
    """Raised when the fund editor payload is not a pure fund-code config."""


def _node_dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


_LEGACY_BOILERPLATE = ast.parse(
    """
import json
import os

def get_funds():
    return {'compare_index': compare_index, 'fund': fund, 'hold_index': hold_index}

def get_funds_bond():
    return {'compare_index': compare_index_bond, 'fund': fund_bond, 'hold_index': hold_index}

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    with open('data/fund_codes.json', 'w', encoding='utf-8') as fin:
        json.dump(list(set(compare_index + compare_index_bond + fund + fund_bond)), fin)
"""
)
_ALLOWED_BOILERPLATE_NODES = {_node_dump(node) for node in _LEGACY_BOILERPLATE.body}
_EXPECTED_FUND_EXPRESSION = _node_dump(ast.parse("fund_index + fund_stock", mode="eval").body)


def _validate_fund_code_list(name: str, value: object) -> list[str]:
    if not isinstance(value, list):
        raise FundConfigError(f"{name} 必须是基金代码列表")
    if len(value) > _MAX_CODES_PER_LIST:
        raise FundConfigError(f"{name} 最多允许 {_MAX_CODES_PER_LIST} 个基金代码")

    codes: list[str] = []
    seen: set[str] = set()
    for code in value:
        if not isinstance(code, str) or len(code) != 6 or not code.isascii() or not code.isdigit():
            raise FundConfigError(f"{name} 包含非法基金代码: {code!r}")
        if code in seen:
            raise FundConfigError(f"{name} 包含重复基金代码: {code}")
        seen.add(code)
        codes.append(code)
    return codes


def _validate_fund_config(config: Mapping[str, object]) -> dict[str, list[str]]:
    unknown = sorted(set(config) - set(_FUND_LIST_NAMES))
    if unknown:
        raise FundConfigError(f"不支持的配置项: {', '.join(unknown)}")

    missing = [name for name in _FUND_LIST_NAMES if name not in config]
    if missing:
        raise FundConfigError(f"缺少配置项: {', '.join(missing)}")

    return {
        name: _validate_fund_code_list(name, config[name])
        for name in _FUND_LIST_NAMES
    }


def _parse_fund_config_source(content: str) -> dict[str, list[str]]:
    if len(content.encode("utf-8")) > _MAX_EDITOR_BYTES:
        raise FundConfigError(f"基金配置文件不能超过 {_MAX_EDITOR_BYTES // 1024} KiB")
    try:
        module = ast.parse(content, filename=str(FUND_EDITOR_FILE), mode="exec")
    except SyntaxError as exc:
        raise FundConfigError(f"Python 语法错误: line {exc.lineno}, {exc.msg}") from exc
    except (ValueError, RecursionError) as exc:
        raise FundConfigError(f"无法解析基金配置: {exc}") from exc

    config: dict[str, object] = {}
    has_derived_fund = False
    required_functions = {"get_funds", "get_funds_bond"}
    found_functions: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in _FUND_LIST_NAMES:
                if name in config:
                    raise FundConfigError(f"配置项重复定义: {name}")
                try:
                    config[name] = ast.literal_eval(node.value)
                except (ValueError, TypeError, RecursionError) as exc:
                    raise FundConfigError(f"{name} 只能包含静态基金代码列表") from exc
                continue
            if name == "fund" and _node_dump(node.value) == _EXPECTED_FUND_EXPRESSION:
                has_derived_fund = True
                continue

        if _node_dump(node) in _ALLOWED_BOILERPLATE_NODES:
            if isinstance(node, ast.FunctionDef):
                found_functions.add(node.name)
            continue

        line = getattr(node, "lineno", "?")
        raise FundConfigError(f"line {line} 包含不允许执行的 Python 语句")

    if not has_derived_fund:
        raise FundConfigError("缺少固定定义: fund = fund_index + fund_stock")
    missing_functions = sorted(required_functions - found_functions)
    if missing_functions:
        raise FundConfigError(f"缺少固定函数: {', '.join(missing_functions)}")
    return _validate_fund_config(config)


def _serialize_fund_config(config: Mapping[str, list[str]]) -> str:
    lines = [
        "# 此文件由基金配置编辑器生成；只包含经过校验的基金代码。",
        "# 请通过页面编辑器修改，任意 Python 语句会被拒绝。",
        "",
    ]
    for name in _FUND_LIST_NAMES:
        value = json.dumps(config[name], ensure_ascii=False, indent=4)
        lines.extend((f"{name} = {value}", ""))
        if name == "fund_stock":
            lines.extend(("fund = fund_index + fund_stock", ""))

    lines.extend(
        (
            "def get_funds():",
            "    return {'compare_index': compare_index, 'fund': fund, 'hold_index': hold_index}",
            "",
            "",
            "def get_funds_bond():",
            "    return {'compare_index': compare_index_bond, 'fund': fund_bond, 'hold_index': hold_index}",
            "",
        )
    )
    return "\n".join(lines)


def _configured_codes(config: Mapping[str, list[str]]) -> list[str]:
    ordered = (
        config["compare_index"]
        + config["compare_index_bond"]
        + config["fund_index"]
        + config["fund_stock"]
        + config["fund_bond"]
    )
    return list(dict.fromkeys(ordered))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as tmp:
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_name = tmp.name
        os.chmod(temp_name, mode)
        os.replace(temp_name, path)
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)


def _write_fund_config(
    config: Mapping[str, list[str]],
    *,
    validated_source: str | None = None,
) -> str:
    source = validated_source if validated_source is not None else _serialize_fund_config(config)
    codes_json = json.dumps(_configured_codes(config), ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(FUND_EDITOR_FILE, source)
    _atomic_write_text(FUND_CODES_FILE, codes_json)
    return source


def _load_validated_fund_config() -> dict[str, list[str]]:
    content = FUND_EDITOR_FILE.read_text(encoding="utf-8")
    return _parse_fund_config_source(content)


def _sync_fund_codes(config: Mapping[str, list[str]]) -> None:
    _atomic_write_text(
        FUND_CODES_FILE,
        json.dumps(_configured_codes(config), ensure_ascii=False, indent=2) + "\n",
    )


@bp.get("/fund")
def fund_page():
    report_data_file = current_app.config["FUND_REPORT_DATA_FILE"]
    report_view = load_fund_report_view(report_data_file)
    return render_template(
        "fund/report.html",
        report=file_status(report_data_file),
        report_view=report_view,
        signals=file_status(current_app.config["FUND_SIGNALS_FILE"]),
    )


@bp.get("/fund/report")
def fund_report():
    return fund_page()


@bp.get("/api/fund/status")
def fund_status():
    report_data = file_status(current_app.config["FUND_REPORT_DATA_FILE"])
    return jsonify(
        {
            "report": report_data,
            "report_data": report_data,
            "signals": file_status(current_app.config["FUND_SIGNALS_FILE"]),
            "generate": get_job_state("fund-report-generate"),
            "refresh": get_job_state("fund-refresh"),
        }
    )


@bp.get("/api/fund/editor")
def read_fund_editor_file():
    try:
        content = FUND_EDITOR_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        return jsonify({"error": str(exc)}), 500
    try:
        config = _parse_fund_config_source(content)
        validation_error = None
    except FundConfigError as exc:
        config = None
        validation_error = str(exc)
    return jsonify(
        {
            "path": FUND_EDITOR_FILE.relative_to(ROOT_DIR).as_posix(),
            "content": content,
            "funds": config,
            "validation_error": validation_error,
        }
    )


@bp.put("/api/fund/editor")
def save_fund_editor_file():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, Mapping):
        return jsonify({"error": "请求体必须是 JSON 对象"}), 400
    structured = payload.get("funds")
    validated_source = None
    try:
        if structured is not None:
            if not isinstance(structured, Mapping):
                raise FundConfigError("funds 必须是结构化基金配置")
            config = _validate_fund_config(structured)
        else:
            content = payload.get("content")
            if not isinstance(content, str):
                raise FundConfigError("缺少可保存的基金配置")
            config = _parse_fund_config_source(content)
            validated_source = content
    except FundConfigError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        with _FUND_CONFIG_LOCK:
            if is_resource_running(_FUND_RESOURCE_KEY):
                return jsonify({"error": "基金刷新/报告生成运行中，暂不能修改配置"}), 409
            source = _write_fund_config(config, validated_source=validated_source)
    except OSError as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(
        {
            "saved": True,
            "path": FUND_EDITOR_FILE.relative_to(ROOT_DIR).as_posix(),
            "size": len(source.encode("utf-8")),
            "content": source,
            "funds": config,
        }
    )


@bp.post("/api/fund/report/generate")
def generate_fund_report():
    with _FUND_CONFIG_LOCK:
        try:
            _load_validated_fund_config()
        except (OSError, FundConfigError) as exc:
            return jsonify({"error": f"基金配置校验失败，未启动报告生成: {exc}"}), 409

        started = start_command_job(
            "fund-report-generate",
            [sys.executable, "-B", "fund_generate_output.py"],
            cwd=ROOT_DIR,
            timeout=300,
            resource_key=_FUND_RESOURCE_KEY,
        )
    status = 202 if started else 409
    payload = {"started": started} if started else {"error": "基金报告生成已在运行中"}
    return jsonify(payload), status


@bp.post("/api/fund/run")
def refresh_fund_report():
    with _FUND_CONFIG_LOCK:
        try:
            config = _load_validated_fund_config()
            _sync_fund_codes(config)
        except (OSError, FundConfigError) as exc:
            return jsonify({"error": f"基金配置校验失败，未启动刷新: {exc}"}), 409

        started = start_command_job(
            "fund-refresh",
            [sys.executable, "-B", "fund_data_refresh.py"],
            cwd=ROOT_DIR,
            timeout=1800,
            resource_key=_FUND_RESOURCE_KEY,
        )
    status = 202 if started else 409
    payload = {"started": started} if started else {"error": "基金刷新已在运行中"}
    return jsonify(payload), status
