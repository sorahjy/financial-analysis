from __future__ import annotations

from typing import Any, Dict

from flask import Blueprint, jsonify, render_template, request

from stock_advanced_strategies import (
    get_default_config,
    get_factor_registry,
)
from stock_crawl_common import load_json_file
from stock_data_refresh import REFRESH_REPORT_FILE

from app.services.stock_strategy_service import (
    optimizer_state_snapshot,
    run_stock_strategies,
    start_stock_data_refresh,
    start_optimizer_job,
    stock_data_refresh_state,
)


bp = Blueprint("stock", __name__)


@bp.get("/stock")
def stock_page():
    return render_template("stock/dashboard.html")


def _stock_config_payload() -> Dict[str, Any]:
    return {"config": get_default_config(), "factors": get_factor_registry()}


@bp.get("/api/config")
@bp.get("/api/stock/config")
def stock_config():
    return jsonify(_stock_config_payload())


@bp.get("/api/health")
@bp.get("/api/stock/health")
def stock_health():
    return jsonify(
        {
            "ok": True,
            "refresh": load_json_file(REFRESH_REPORT_FILE, {}),
            "refresh_job": stock_data_refresh_state(),
        }
    )


@bp.post("/api/stock/refresh")
def stock_refresh():
    if start_stock_data_refresh():
        return jsonify({"started": True}), 202
    return jsonify({"error": "数据刷新已在运行中"}), 409


@bp.get("/api/optimize/status")
@bp.get("/api/stock/optimize/status")
def stock_optimize_status():
    return jsonify(optimizer_state_snapshot())


@bp.post("/api/optimize")
@bp.post("/api/stock/optimize")
def stock_optimize():
    if start_optimizer_job():
        return jsonify({"started": True}), 202
    return jsonify({"error": "参数搜索已在运行中"}), 409


@bp.post("/api/run")
@bp.post("/api/stock/run")
def stock_run():
    try:
        payload = request.get_json(silent=True) or {}
        result = run_stock_strategies(payload.get("config", {}), persist=False, invalidate=False)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
