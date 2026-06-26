from __future__ import annotations

from flask import Blueprint, current_app, render_template

from app.utils import file_status
from stock_hot_money_radar import AMBUSH_RESULT_FILE


bp = Blueprint("home", __name__)


def fallback_file_status(primary, backup):
    primary_status = file_status(primary)
    if primary_status["exists"]:
        return primary_status
    backup_status = file_status(backup)
    if backup_status["exists"]:
        return backup_status
    return primary_status


@bp.get("/")
def index():
    return render_template(
        "index.html",
        fund_report=file_status(current_app.config["FUND_REPORT_DATA_FILE"]),
        stock_result=file_status(current_app.config["STOCK_RESULT_FILE"]),
        optimized_config=fallback_file_status(
            current_app.config["STOCK_OPTIMIZED_CONFIG_FILE"],
            current_app.config["STOCK_OPTIMIZED_CONFIG_BACKUP_FILE"],
        ),
        radar_result=file_status(AMBUSH_RESULT_FILE),
    )
