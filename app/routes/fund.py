from __future__ import annotations

import sys

from flask import Blueprint, current_app, jsonify, render_template

from app.config import ROOT_DIR
from app.services.fund_report_service import load_fund_report_view
from app.services.job_service import get_job_state, start_command_job
from app.utils import file_status


bp = Blueprint("fund", __name__)


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
