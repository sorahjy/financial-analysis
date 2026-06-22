from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from app.services.radar_service import (
    kline_bars,
    radar_data_state,
    radar_pattern_catalog,
    radar_payload,
    radar_run_state,
    start_radar_data_refresh,
    start_radar_run,
)


bp = Blueprint("radar", __name__)


@bp.get("/radar")
def radar_page():
    return render_template("radar/dashboard.html")


@bp.get("/api/radar/data")
def radar_data():
    return jsonify(
        {
            "payload": radar_payload(),
            "run_job": radar_run_state(),
            "data_job": radar_data_state(),
        }
    )


@bp.get("/api/radar/patterns")
def radar_patterns():
    return jsonify({"patterns": radar_pattern_catalog()})


@bp.get("/api/radar/kline")
def radar_kline():
    code = request.args.get("code", "")
    limit = request.args.get("limit", 3600, type=int)
    period = request.args.get("period", "day")
    return jsonify(kline_bars(code, limit, period))


@bp.post("/api/radar/run")
def radar_run():
    body = request.get_json(silent=True) or {}
    if start_radar_run(include_large_cap=bool(body.get("include_large_cap", True)),
                       pool=str(body.get("pool") or "leader")):
        return jsonify({"started": True}), 202
    return jsonify({"error": "雷达运行已在进行中"}), 409


@bp.post("/api/radar/refresh-data")
def radar_refresh_data():
    if start_radar_data_refresh():
        return jsonify({"started": True}), 202
    return jsonify({"error": "数据刷新已在运行中"}), 409
