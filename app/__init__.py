from __future__ import annotations

from flask import Flask


def create_app(*, warmup_stock_strategy: bool = False) -> Flask:
    app = Flask(__name__)
    app.config.from_object("app.config.AppConfig")
    app.json.ensure_ascii = False

    from app.routes.fund import bp as fund_bp
    from app.routes.home import bp as home_bp
    from app.routes.jobs import bp as jobs_bp
    from app.routes.stock import bp as stock_bp

    app.register_blueprint(home_bp)
    app.register_blueprint(fund_bp)
    app.register_blueprint(stock_bp)
    app.register_blueprint(jobs_bp)
    if warmup_stock_strategy:
        from app.services.stock_strategy_service import start_stock_strategy_warmup

        start_stock_strategy_warmup()
    return app
