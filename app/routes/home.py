from __future__ import annotations

from flask import Blueprint, current_app, render_template

from app.utils import file_status


bp = Blueprint("home", __name__)


@bp.get("/")
def index():
    return render_template(
        "index.html",
        fund_report=file_status(current_app.config["FUND_REPORT_DATA_FILE"]),
        stock_result=file_status(current_app.config["STOCK_RESULT_FILE"]),
        optimized_config=file_status(current_app.config["STOCK_OPTIMIZED_CONFIG_FILE"]),
    )
