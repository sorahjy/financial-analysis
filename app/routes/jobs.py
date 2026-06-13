from __future__ import annotations

from flask import Blueprint, jsonify

from app.services.job_service import get_job_state


bp = Blueprint("jobs", __name__)


@bp.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    return jsonify(get_job_state(job_id))
