from __future__ import annotations

import sys

from flask import Blueprint, current_app, jsonify, render_template, request

from app.config import ROOT_DIR
from app.services.fund_report_service import load_fund_report_view
from app.services.job_service import get_job_state, start_command_job
from app.utils import file_status


bp = Blueprint("fund", __name__)
FUND_EDITOR_FILE = ROOT_DIR / "funds.py"


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
    return jsonify(
        {
            "path": FUND_EDITOR_FILE.relative_to(ROOT_DIR).as_posix(),
            "content": content,
        }
    )


@bp.put("/api/fund/editor")
def save_fund_editor_file():
    payload = request.get_json(silent=True) or {}
    content = payload.get("content")
    if not isinstance(content, str):
        return jsonify({"error": "缺少可保存的文本内容"}), 400
    try:
        compile(content, str(FUND_EDITOR_FILE), "exec")
    except SyntaxError as exc:
        return jsonify({"error": f"Python 语法错误: line {exc.lineno}, {exc.msg}"}), 400
    try:
        FUND_EDITOR_FILE.write_text(content, encoding="utf-8")
    except OSError as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(
        {
            "saved": True,
            "path": FUND_EDITOR_FILE.relative_to(ROOT_DIR).as_posix(),
            "size": len(content.encode("utf-8")),
        }
    )


@bp.post("/api/fund/report/generate")
def generate_fund_report():
    started = start_command_job(
        "fund-report-generate",
        [sys.executable, "-B", "fund_generate_output.py"],
        cwd=ROOT_DIR,
        timeout=300,
    )
    status = 202 if started else 409
    payload = {"started": started} if started else {"error": "基金报告生成已在运行中"}
    return jsonify(payload), status


@bp.post("/api/fund/run")
def refresh_fund_report():
    started = start_command_job(
        "fund-refresh",
        ["bash", "fund_run.sh"],
        cwd=ROOT_DIR,
        timeout=1800,
    )
    status = 202 if started else 409
    payload = {"started": started} if started else {"error": "基金刷新已在运行中"}
    return jsonify(payload), status
