from __future__ import annotations

from typing import Any, Dict

from flask import Blueprint, abort, jsonify, render_template, request, send_file

from stock_advanced_strategies import (
    get_default_config,
    get_factor_registry,
)
from stock_crawl_common import load_json_file
from stock_data_refresh import REFRESH_REPORT_FILE

from app.services.stock_strategy_service import (
    build_long_backtest_chart,
    build_strategy_backtest_chart,
    optimizer_state_snapshot,
    resolve_long_backtest_chart_file,
    run_stock_strategies,
    start_stock_data_refresh,
    start_optimizer_job,
    stock_data_refresh_state,
)
from app.services.radar_service import kline_bars, radar_stock_context


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
        result = run_stock_strategies(
            payload.get("config", {}),
            persist=False,
            invalidate=False,
            include_search_pool=True,
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@bp.get("/api/stock/kline")
def stock_kline():
    code = request.args.get("code", "")
    limit = request.args.get("limit", 640, type=int)
    period = request.args.get("period", "week")
    return jsonify(kline_bars(code, limit, period))


@bp.get("/api/stock/radar-context")
def stock_radar_context():
    return jsonify(radar_stock_context())


@bp.post("/api/stock/long-backtest-chart")
def stock_long_backtest_chart():
    try:
        payload = request.get_json(silent=True) or {}
        strategy = payload.get("strategy", "long")
        return jsonify(build_strategy_backtest_chart(payload.get("config", {}), strategy=strategy))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@bp.get("/api/stock/long-backtest-chart/file/<path:filename>")
def stock_long_backtest_chart_file(filename: str):
    try:
        path = resolve_long_backtest_chart_file(filename)
    except FileNotFoundError:
        abort(404)
    return send_file(path, mimetype="image/svg+xml", max_age=0)
