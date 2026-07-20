import json
import sys
import time
import unittest
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

from app import create_app
from app.config import ROOT_DIR
from app.services.job_service import get_job_state, start_command_job
from app.services.fund_report_service import load_fund_report_view
from app.services import radar_service, stock_strategy_service


VALID_FUND_EDITOR_CONTENT = """\
# 用户可以保留基金分类和候选代码注释。
compare_index = ["510310", "510580"]
compare_index_bond = ["161119"]
hold_index = ["510310"]
fund_index = ["008115"]
fund_stock = ["040046"]
fund = fund_index + fund_stock
fund_bond = ["011555"]

def get_funds():
    return {'compare_index': compare_index, 'fund': fund, 'hold_index': hold_index}

def get_funds_bond():
    return {'compare_index': compare_index_bond, 'fund': fund_bond, 'hold_index': hold_index}
"""


class FlaskAppTest(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_workspace_pages_load(self):
        for path in ("/", "/fund", "/fund/report", "/stock", "/radar"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/html", response.content_type)

    def test_app_shell_loads_shared_scripts_once_per_page(self):
        for path in ("/", "/fund", "/stock", "/radar"):
            with self.subTest(path=path):
                html = self.client.get(path).get_data(as_text=True)
                self.assertEqual(html.count("js/fund-report.js"), 1)
                self.assertEqual(html.count("js/stock-dashboard.js"), 1)
                self.assertEqual(html.count("js/radar.js"), 1)
                self.assertEqual(html.count("js/app-navigation.js"), 1)
                self.assertEqual(html.count("vendor/live2d-widget/autoload.js"), 1)

    def test_app_shell_exposes_refreshed_design_and_active_navigation(self):
        css = (ROOT_DIR / "app/static/css/ui-refresh.css").read_text(encoding="utf-8")
        self.assertIn("--topbar-height", css)
        self.assertIn(":focus-visible", css)
        self.assertIn("prefers-reduced-motion", css)

        home = self.client.get("/").get_data(as_text=True)
        self.assertIn('class="page-home"', home)
        self.assertIn('class="skip-link"', home)
        self.assertEqual(home.count("css/ui-refresh.css"), 1)
        self.assertIn("今天想看什么？", home)
        self.assertNotIn('class="home-hero"', home)
        self.assertNotIn("把复杂数据", home)
        self.assertNotIn(".home-hero", css)
        self.assertNotIn(".hero-visual", css)

        for path, body_class, nav_path in (
            ("/fund", "page-fund", "/fund"),
            ("/stock", "page-stock", "/stock"),
            ("/radar", "page-radar", "/radar"),
        ):
            with self.subTest(path=path):
                html = self.client.get(path).get_data(as_text=True)
                self.assertIn(f'class="{body_class}"', html)
                self.assertRegex(
                    html,
                    rf'data-nav-path="{re.escape(nav_path)}"[^>]*aria-current="page"',
                )

    def test_fund_toolbars_scroll_with_content_instead_of_covering_rows(self):
        app_css = (ROOT_DIR / "app/static/css/app.css").read_text(encoding="utf-8")
        report_css = (ROOT_DIR / "app/static/css/fund-report.css").read_text(encoding="utf-8")
        refresh_css = (ROOT_DIR / "app/static/css/ui-refresh.css").read_text(encoding="utf-8")

        self.assertRegex(app_css, r"\.fund-command-bar\s*\{\s*position:\s*static;")
        self.assertRegex(report_css, r"\.fund-native \.control-panel\s*\{\s*position:\s*static;")
        self.assertRegex(refresh_css, r"\.fund-command-bar\s*\{\s*position:\s*static;")
        self.assertRegex(refresh_css, r"\.fund-native \.control-panel\s*\{\s*position:\s*static;")

    def test_radar_filter_bar_scrolls_with_content(self):
        refresh_css = (ROOT_DIR / "app/static/css/ui-refresh.css").read_text(encoding="utf-8")

        self.assertRegex(refresh_css, r"\.radar-filters\s*\{\s*position:\s*relative;")
        self.assertNotRegex(refresh_css, r"\.radar-filters\s*\{\s*position:\s*sticky;")

    def test_internal_navigation_replaces_main_without_reloading_shell(self):
        script = (ROOT_DIR / "app/static/js/app-navigation.js").read_text(encoding="utf-8")

        self.assertIn('new Set(["/", "/fund", "/fund/report", "/stock", "/radar"])', script)
        self.assertIn('headers: {"X-Requested-With": "fetch"}', script)
        self.assertIn("FinancialAnalysisPages.bootId", script)
        self.assertIn('querySelector(".app-main")', script)
        self.assertIn("currentMain.innerHTML = nextMain.innerHTML", script)
        self.assertIn("window.history.pushState", script)
        self.assertIn("function updateShellState", script)
        self.assertIn('name.startsWith("page-")', script)
        self.assertIn('link.setAttribute("aria-current", "page")', script)
        self.assertIn("pages.cleanup()", script)
        self.assertIn("pages.stock && pages.stock()", script)
        self.assertIn("pages.fund && pages.fund()", script)

    def test_page_scripts_support_partial_navigation_reinit(self):
        fund_script = (ROOT_DIR / "app/static/js/fund-report.js").read_text(encoding="utf-8")
        stock_script = (ROOT_DIR / "app/static/js/stock-dashboard.js").read_text(encoding="utf-8")
        radar_script = (ROOT_DIR / "app/static/js/radar.js").read_text(encoding="utf-8")

        self.assertIn("FinancialAnalysisPages.fund = initFundPage", fund_script)
        self.assertIn('statusRoot.dataset.fundPageInitialized === "1"', fund_script)
        self.assertIn('statusRoot.dataset.fundPageInitialized = "1"', fund_script)
        self.assertIn("FinancialAnalysisPages.cleanup", fund_script)
        self.assertIn("disposed = true", fund_script)

        self.assertIn("FinancialAnalysisPages.stock = initStockDashboard", stock_script)
        self.assertIn('dashboardRoot.dataset.stockDashboardInitialized === "1"', stock_script)
        self.assertIn('dashboardRoot.dataset.stockDashboardInitialized = "1"', stock_script)
        self.assertIn("FinancialAnalysisPages.cleanup", stock_script)
        self.assertIn("clearTimeout(runTimer)", stock_script)
        self.assertIn("clearInterval(optimizeTimer)", stock_script)
        self.assertIn("clearInterval(refreshTimer)", stock_script)

        self.assertIn("FinancialAnalysisPages.radar = initRadar", radar_script)
        self.assertIn('root.dataset.radarInit === "1"', radar_script)
        self.assertIn('root.dataset.radarInit = "1"', radar_script)
        self.assertIn("FinancialAnalysisPages.cleanup", radar_script)
        self.assertIn("clearInterval(pollTimer)", radar_script)
        self.assertIn("clearInterval(realtimeTimer)", radar_script)

    def test_radar_page_exposes_filter_and_sort_contracts(self):
        body = (ROOT_DIR / "app/templates/radar/dashboard.html").read_text(encoding="utf-8")
        script = (ROOT_DIR / "app/static/js/radar.js").read_text(encoding="utf-8")

        self.assertIn("fmtMarketCap(s.market_cap_yi)", script)
        for key, label in (
            ("market_cap_yi", "市值"),
            ("realtime_price", "现价"),
            ("realtime_change_pct", "涨幅"),
            ("opportunity_score", "机会分"),
        ):
            with self.subTest(sort_key=key):
                self.assertIn(f'sortTh("{key}", "{label}")', script)

        for element_id in (
            "radar-live-quote",
            "radar-live-status",
            "radar-phase-filter",
            "radar-pattern-filter",
            "radar-sw2-filter",
            "radar-min-score",
            "radar-export-limit",
            "radar-export",
        ):
            with self.subTest(element_id=element_id):
                self.assertIn(f'id="{element_id}"', body)

        self.assertIn('fetch("/api/radar/realtime")', script)
        self.assertIn('fetch("/api/radar/industry-heat")', script)
        self.assertIn('data-key="${key}"', script)
        self.assertIn('aria-sort="${ariaSort}"', script)
        self.assertIn('key: "opportunity_score"', script)

    def test_radar_industry_heat_lists_falling_trends_with_negative_scores(self):
        body = (ROOT_DIR / "app/templates/radar/dashboard.html").read_text(encoding="utf-8")
        script = (ROOT_DIR / "app/static/js/radar.js").read_text(encoding="utf-8")
        styles = (ROOT_DIR / "app/static/css/radar.css").read_text(encoding="utf-8")

        self.assertIn("成交额活跃度、升降趋势与数据来源", body)
        self.assertIn("function industryHeatTrendRows(report)", script)
        self.assertIn('industryHeatRankedRows(report, "trend", "trend_rank")', script)
        self.assertIn("growth < 0", script)
        self.assertIn("correlation < 0", script)
        self.assertIn("下降行业显示负分", script)
        self.assertIn("趋势分 ${fmtSigned(trendScore, 1, falling)}", script)
        self.assertIn('rowClass = hot ? "is-hot" : (falling ? "is-falling" : "is-rising")', script)
        self.assertIn(".industry-heat-row.is-falling", styles)

    def test_radar_payload_refreshes_legacy_lhb_warning_copy(self):
        legacy = {
            "stocks": [{
                "code": "000933",
                "evidence": [{"label": "近期上龙虎榜(避雷)", "kind": "bearish"}],
            }],
        }
        with patch("app.services.radar_service.load_json_file", return_value=legacy):
            payload = radar_service.radar_payload()

        self.assertEqual(
            payload["stocks"][0]["evidence"][0],
            {"label": "近期上龙虎榜，波动加剧", "kind": "bearish"},
        )

    def test_radar_pattern_colors_follow_backend_effectiveness_catalog(self):
        response = self.client.get("/api/radar/patterns")
        self.assertEqual(response.status_code, 200)
        catalog = {item["code"]: item for item in response.get_json()["patterns"]}
        self.assertEqual(catalog["P3"]["effective_style"], "bullish")
        self.assertEqual(catalog["P24"]["effective_style"], "bullish")
        self.assertEqual(catalog["P12"]["effective_style"], "momentum")
        self.assertEqual(catalog["P13"]["effective_style"], "momentum")
        self.assertEqual(catalog["P11"]["effective_style"], "risk")
        self.assertFalse(catalog["P6"]["effective"])
        self.assertTrue(catalog["P23"]["effective"])
        self.assertTrue(catalog["P25"]["effective"])
        self.assertEqual(catalog["P6"]["effective_style"], "neutral")
        self.assertEqual(catalog["P23"]["effective_style"], "bullish")
        self.assertEqual(catalog["P25"]["effective_style"], "bullish")
        self.assertEqual(catalog["P6"]["display_style"], "neutral")
        self.assertEqual(catalog["P23"]["display_style"], "bullish")
        self.assertEqual(catalog["P25"]["display_style"], "bullish")
        self.assertEqual(len(catalog), 26)
        self.assertTrue(all(item["production"] for item in catalog.values()))
        self.assertEqual(catalog["P3"]["score_usage"], "吸筹分 10%")
        self.assertEqual(catalog["P1"]["score_usage"], "吸筹分 10%")
        self.assertEqual(catalog["P25"]["score_usage"], "吸筹分 5%")
        self.assertEqual(catalog["P14"]["score_usage"], "出货分 5%")
        self.assertEqual(catalog["P15"]["score_usage"], "出货分 5%")
        self.assertEqual(catalog["P22"]["score_usage"], "出货分 5%")
        self.assertEqual(catalog["P26"]["score_usage"], "出货分 10%")
        self.assertEqual(catalog["P23"]["validation_status"], "core")
        self.assertEqual(catalog["P25"]["validation_status"], "core")
        self.assertEqual(catalog["P1"]["validation_label"], "核心有效")
        self.assertEqual(catalog["P23"]["validation_label"], "核心有效")
        self.assertEqual(catalog["P25"]["validation_label"], "核心有效")
        self.assertEqual(catalog["P6"]["validation_status"], "observation")

        script = (ROOT_DIR / "app/static/js/radar.js").read_text(encoding="utf-8")
        css = (ROOT_DIR / "app/static/css/radar.css").read_text(encoding="utf-8")
        self.assertIn('const style = meta.display_style || meta.effective_style || "neutral"', script)
        self.assertIn('style === "bullish"', script)
        self.assertIn('style === "momentum"', script)
        self.assertIn('style === "risk"', script)
        self.assertNotIn('if (!meta || !meta.effective || isEtfMode()) return "pd"', script)
        self.assertIn(
            "Promise.all([fetchPatternCatalog(), fetchScoringModel()]).finally(fetchData)",
            script,
        )
        self.assertNotIn("EFFECTIVE_RISK_PATTERNS", script)
        self.assertNotIn("PAT_TIP", script)
        for css_class in (".pat.pg", ".pat.po", ".pat.pr", ".pat.pd"):
            self.assertIn(css_class, css)

        stock_script = (ROOT_DIR / "app/static/js/stock-dashboard.js").read_text(encoding="utf-8")
        stock_css = (ROOT_DIR / "app/static/css/stock-dashboard.css").read_text(encoding="utf-8")
        self.assertIn('fetch("/api/radar/patterns")', stock_script)
        self.assertIn("function acceptRadarPatternCatalog(patterns)", stock_script)
        self.assertIn('const style = meta.display_style || meta.effective_style || "neutral"', stock_script)
        self.assertIn('style === "bullish"', stock_script)
        self.assertIn('style === "momentum"', stock_script)
        self.assertIn('style === "risk"', stock_script)
        self.assertNotIn('if (!meta || !meta.effective || radarContextMeta.pool === "etf") return "neutral"', stock_script)
        self.assertNotIn("RADAR_BUY_PATTERNS", stock_script)
        self.assertNotIn("RADAR_RISK_PATTERNS", stock_script)
        self.assertIn(".stock-radar-factor.momentum", stock_css)
        self.assertIn("background: var(--panel-soft)", stock_css)
        self.assertIn("color: var(--muted)", stock_css)
        for color in ("#eaf7ef", "#14804a", "#fdeede", "#b4530e", "#fdebe4", "#b3340f"):
            self.assertIn(color, stock_css)

    def test_radar_job_polling_uses_lightweight_endpoint(self):
        response = self.client.get("/api/radar/jobs")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.get_json()), {"run_job", "data_job"})

        script = (ROOT_DIR / "app/static/js/radar.js").read_text(encoding="utf-8")
        self.assertIn('fetch("/api/radar/jobs")', script)
        self.assertIn("setInterval(fetchJobState, 2000)", script)
        self.assertNotIn("setInterval(fetchData, 2000)", script)

    def test_radar_kline_card_shows_selected_stock_basic_info(self):
        body = (ROOT_DIR / "app/templates/radar/dashboard.html").read_text(encoding="utf-8")
        script = (ROOT_DIR / "app/static/js/radar.js").read_text(encoding="utf-8")
        css = (ROOT_DIR / "app/static/css/radar.css").read_text(encoding="utf-8")

        self.assertIn('id="radar-stock-info"', body)
        self.assertIn('aria-label="股票基本信息"', body)
        self.assertLess(body.index('id="radar-kline-title"'), body.index('id="radar-stock-info"'))
        self.assertLess(body.index('id="radar-stock-info"'), body.index('id="radar-kline-periods"'))
        for field in ("price", "change", "cap", "sw2", "sw3", "theme", "date"):
            self.assertIn(f'id="radar-stock-{field}"', body)

        self.assertIn("function renderStockInfo(s, bars = [])", script)
        self.assertIn("renderStockInfo(s, []);", script)
        self.assertIn("renderStockInfo(selected, klineBars);", script)
        self.assertIn("s.realtime_price", script)
        self.assertIn("s.parent_segment", script)
        self.assertIn("s.segment_name", script)
        self.assertIn("s.tracking_theme", script)
        self.assertIn("s.last_date", script)
        self.assertIn(".radar-stock-info", css)
        self.assertIn(".radar-stock-facts", css)

    def test_radar_kline_date_range_uses_slider_without_mouse_wheel_zoom(self):
        script = (ROOT_DIR / "app/static/js/radar.js").read_text(encoding="utf-8")

        self.assertNotIn("function onCanvasWheel", script)
        self.assertNotIn('canvas.addEventListener("wheel"', script)
        self.assertIn('klineDrag = { type: "range"', script)
        self.assertIn('klineDrag = { type: "start"', script)
        self.assertIn('klineDrag = { type: "end"', script)

    def test_radar_kline_panel_scrolls_to_reveal_pattern_details(self):
        refresh_css = (ROOT_DIR / "app/static/css/ui-refresh.css").read_text(encoding="utf-8")

        self.assertRegex(
            refresh_css,
            r"\.radar-kline\s*\{\s*position:\s*sticky;\s*align-self:\s*start;\s*"
            r"top:\s*calc\(var\(--topbar-height\) \+ 16px\);\s*"
            r"max-height:\s*calc\(100vh - var\(--topbar-height\) - 32px\);\s*"
            r"overflow-y:\s*auto;",
        )
        self.assertRegex(
            refresh_css,
            r"@media \(max-width:\s*1050px\)[\s\S]*?\.radar-kline\s*\{[\s\S]*?"
            r"max-height:\s*none;[\s\S]*?overflow-y:\s*visible;",
        )

    def test_radar_header_uses_score_and_factor_help_modals(self):
        body = (ROOT_DIR / "app/templates/radar/dashboard.html").read_text(encoding="utf-8")
        css = (ROOT_DIR / "app/static/css/radar.css").read_text(encoding="utf-8")

        self.assertIn('id="radar-score-help"', body)
        self.assertIn('id="radar-factor-help"', body)
        self.assertIn('id="radar-score-modal"', body)
        self.assertIn('id="radar-factor-modal"', body)
        self.assertIn("机会分 = 吸筹百分位 ×（1 − 0.5 × 出货百分位 ÷ 100）", body)
        self.assertIn("P3 缩量打压首次收复", body)
        self.assertIn("P24 OBV底背离", body)
        self.assertIn("股东户数变化", body)
        self.assertIn("13项原始特征直接加权", body)
        self.assertIn("P1 超额优先复合确认", body)
        self.assertIn("P25 缩量平台转强", body)
        self.assertNotIn("P25 缩量平台转强（双池实验）", body)
        self.assertIn('id="radar-effective-pattern-count">19<', body)
        self.assertIn('id="radar-scoring-factor-count">29<', body)
        self.assertIn("11项风险特征直接加权", body)
        self.assertIn("P15 新高量背离放量回撤", body)
        self.assertIn("<b>强风险因子</b><span>P16、P17、P19、P26</span>", body)
        self.assertIn("<b>风险因子</b><span>P14、P15、P20、P22</span>", body)
        self.assertIn("近90日龙虎榜", body)
        self.assertIn("游资小盘池按三日平滑反转分排序", body)
        self.assertIn('id="radar-pattern-factor-groups"', body)
        self.assertIn('id="radar-auxiliary-factors"', body)
        self.assertIn("P1–P26 全部生产形态", body)
        self.assertIn(".radar-insight-modal[hidden]", css)
        self.assertIn(".radar-factor-grid", css)
        self.assertIn(".radar-pattern-grid", css)

    def test_radar_scoring_model_endpoint_matches_production_weights(self):
        import stock_hot_money_radar as radar

        response = self.client.get("/api/radar/model")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        accumulation = {row["key"]: row["weight"] for row in payload["accumulation"]["factors"]}
        distribution = {row["key"]: row["weight"] for row in payload["distribution"]["factors"]}
        reversal = {row["key"]: row["weight"] for row in payload["reversal"]["factors"]}
        self.assertEqual(accumulation, radar.ACCUM_MODEL_WEIGHTS)
        self.assertEqual(len(accumulation), 13)
        self.assertEqual(distribution, radar.DIST_MODEL_WEIGHTS)
        self.assertEqual(len(distribution), 11)
        self.assertEqual(distribution["p15"], 0.05)
        self.assertEqual(distribution["p22"], 0.05)
        self.assertIn("P15", payload["distribution"]["warning_rule"]["pattern_codes"])
        self.assertEqual(reversal, radar.REVERSAL_WEIGHTS)
        self.assertEqual(len(payload["auxiliary"]), 4)
        self.assertEqual(
            payload["opportunity"]["distribution_penalty"],
            radar.OPPORTUNITY_DISTRIBUTION_PENALTY,
        )

        script = (ROOT_DIR / "app/static/js/radar.js").read_text(encoding="utf-8")
        self.assertIn('fetch("/api/radar/model")', script)
        self.assertIn("renderScoringModel", script)
        self.assertIn("renderProductionPatterns", script)
        self.assertIn("radar-pattern-factor-groups", script)

    def test_pattern_backtest_marks_p15_as_sell_point(self):
        import stock_hot_money_radar as radar

        bars = [
            {
                "date": f"d{index:03d}",
                "open": 10.0,
                "high": 10.2,
                "low": 9.8,
                "close": 10.0,
                "volume": 1000.0,
            }
            for index in range(radar.MIN_BARS)
        ]
        p15 = {
            "code": "P15",
            "name": "新高量背离放量回撤",
            "phase": "出货",
            "signal": "sell",
        }
        raw_features = {key: 0.0 for key in radar.ACCUM_MODEL_WEIGHTS}
        with (
            patch.object(radar, "_build_pattern_context", return_value={}),
            patch.object(radar, "_p1_rule_state", return_value={"active": False}),
            patch.object(radar, "match_patterns", return_value=[p15]),
            patch.object(radar, "_ensure_pattern_chip_context"),
            patch.object(radar, "_score_bars", return_value={"sub_scores": {}}),
            patch.object(radar, "_accumulation_raw_features", return_value=raw_features),
        ):
            events = radar_service._scan_effective_pattern_events(
                "600000",
                bars,
                radar.MIN_BARS - 1,
                "leader",
                {},
                {},
            )

        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["is_sell_point"])
        self.assertEqual(events[0]["distribution_warning_points"], 1)
        self.assertEqual(events[0]["effective_sell_pattern_count"], 1)
        self.assertEqual(events[0]["patterns"][0]["code"], "P15")
        self.assertEqual(events[0]["patterns"][0]["effective_style"], "risk")

        radar_script = (ROOT_DIR / "app/static/js/radar.js").read_text(encoding="utf-8")
        stock_script = (ROOT_DIR / "app/static/js/stock-dashboard.js").read_text(encoding="utf-8")
        expected_codes = '["P14", "P15", "P16", "P17", "P19", "P20", "P22", "P26"]'
        self.assertIn(expected_codes, radar_script)
        self.assertIn('fetch("/api/radar/patterns")', stock_script)
        self.assertIn("radarPatternCatalogByCode", stock_script)

    def test_live2d_widget_script_is_vendored_locally(self):
        html = self.client.get("/fund").get_data(as_text=True)
        self.assertIn('type="module"', html)
        self.assertIn("vendor/live2d-widget/autoload.js", html)
        self.assertNotIn("fastly.jsdelivr.net", html)

        vendor_dir = ROOT_DIR / "app/static/vendor/live2d-widget"
        required_assets = (
            "autoload.js",
            "waifu.css",
            "waifu-tips.js",
            "waifu-tips.json",
            "live2d.min.js",
            "chunk/index.js",
            "chunk/index2.js",
            "LICENSE",
        )
        for asset in required_assets:
            with self.subTest(asset=asset):
                self.assertTrue((vendor_dir / asset).is_file())
                response = self.client.get(f"/static/vendor/live2d-widget/{asset}")
                try:
                    self.assertEqual(response.status_code, 200)
                finally:
                    response.close()

        autoload = (vendor_dir / "autoload.js").read_text(encoding="utf-8")
        self.assertIn("new URL('./', import.meta.url).href", autoload)
        self.assertNotIn("const live2d_path = 'https://fastly.jsdelivr.net", autoload)

    def test_stock_config_api_loads(self):
        response = self.client.get("/api/config")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("config", payload)
        self.assertIn("factors", payload)
        self.assertIn("smallcap", payload["config"])
        self.assertIn("smallcap", payload["factors"])

    def test_stock_health_reports_refresh_state(self):
        response = self.client.get("/api/stock/health")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("refresh", payload)
        self.assertIn("refresh_job", payload)
        self.assertNotIn("warmup", payload)
        self.assertIn("log_lines", payload["refresh_job"])

    def test_stock_refresh_endpoint_starts_job(self):
        with patch("app.routes.stock.start_stock_data_refresh", return_value=True):
            response = self.client.post("/api/stock/refresh")

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json(), {"started": True})

    def test_stock_refresh_uses_cross_platform_python_entrypoint(self):
        with patch("app.services.stock_strategy_service.start_command_job", return_value=True) as start_job:
            self.assertTrue(stock_strategy_service.start_stock_data_refresh())

        command = start_job.call_args.args[1]
        self.assertEqual(command, [sys.executable, "-B", "stock_radar_fresh_data.py"])
        self.assertFalse(any(part.endswith(".sh") for part in command))
        self.assertNotIn("--strict", command)
        self.assertNotIn("--timeout", command)

    def test_radar_jobs_use_current_python_and_share_refresh_lock(self):
        with patch("app.services.radar_service.start_command_job", return_value=True) as start_job:
            self.assertTrue(radar_service.start_radar_data_refresh())

        self.assertEqual(
            start_job.call_args.args[1],
            [sys.executable, "-B", "stock_radar_fresh_data.py"],
        )
        self.assertEqual(start_job.call_args.kwargs["resource_key"], "stock-data-refresh")

        with patch("app.services.radar_service.start_command_job", return_value=True) as start_job:
            self.assertTrue(radar_service.start_radar_run(pool="leader"))

        self.assertEqual(start_job.call_args.args[1][:3], [sys.executable, "-B", "stock_hot_money_radar.py"])
        self.assertEqual(start_job.call_args.kwargs["resource_key"], "stock-data-refresh")

    def test_stock_optimizer_shares_refresh_resource_lock(self):
        with patch(
            "app.services.stock_strategy_service.start_command_job", return_value=True
        ) as start_job, patch(
            "stock_data_refresh.resolve_python", return_value=sys.executable
        ):
            self.assertTrue(stock_strategy_service.start_optimizer_job())

        self.assertEqual(start_job.call_args.args[0], "stock-optimizer")
        self.assertEqual(start_job.call_args.kwargs["resource_key"], "stock-data-refresh")
        self.assertIsNotNone(start_job.call_args.kwargs["on_success"])

    def test_stock_refresh_requires_full_fetch_confirmation(self):
        script = (ROOT_DIR / "app/static/js/stock-dashboard.js").read_text(encoding="utf-8")

        self.assertIn('if ($("refresh-data").disabled) return;', script)
        self.assertIn("window.confirm(msg)", script)
        self.assertIn("全量拉取", script)
        self.assertIn("no-proxy", script)
        self.assertIn("数据刷新时间会很久", script)

    def test_stock_long_backtest_chart_endpoint_returns_payload(self):
        payload = {
            "is_default": True,
            "url": "/api/stock/long-backtest-chart/file/stock_strategy_best_fold_paths.svg",
            "chart": {"chart_type": "existing_default"},
        }
        with patch("app.routes.stock.build_strategy_backtest_chart", return_value=payload) as build_chart:
            response = self.client.post("/api/stock/long-backtest-chart", json={"config": {"long": {"top_n": 10}}})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), payload)
        build_chart.assert_called_once_with({"long": {"top_n": 10}}, strategy="long")

    def test_stock_smallcap_backtest_chart_endpoint_uses_smallcap_strategy(self):
        payload = {
            "strategy": "smallcap",
            "is_default": False,
            "url": "/api/stock/long-backtest-chart/file/stock_strategy_smallcap_fold_paths_custom_test.svg",
            "chart": {"chart_type": "small_multiples"},
        }
        with patch("app.routes.stock.build_strategy_backtest_chart", return_value=payload) as build_chart:
            response = self.client.post(
                "/api/stock/long-backtest-chart",
                json={"strategy": "smallcap", "config": {"smallcap": {"top_n": 12}}},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), payload)
        build_chart.assert_called_once_with(
            {"smallcap": {"top_n": 12}}, strategy="smallcap"
        )

    def test_stock_config_endpoint_falls_back_to_backup_optimized_config(self):
        payload = {
            "generated_at": "2026-06-25 14:00:00",
            "iterations_per_strategy": 1500,
            "seed": 42,
            "short_universe_version": "hotmoney_small_cap_v1",
            "config": {
                "long": {"top_n": 31},
                "short": {"top_n": 8},
            },
            "caveat": "test",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            primary = Path(tmpdir) / "data" / "stock_strategy_optimized_config.json"
            backup = Path(tmpdir) / "meta_data_backup" / "stock_strategy_optimized_config.json"
            backup.parent.mkdir(parents=True)
            backup.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with patch("stock_advanced_strategies.OPTIMIZED_CONFIG_FILE", primary), \
                 patch("stock_advanced_strategies.OPTIMIZED_CONFIG_BACKUP_FILE", backup):
                response = self.client.get("/api/stock/config")

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["config"]["long"]["top_n"], 31)
        self.assertEqual(data["config"]["short"]["top_n"], 8)
        self.assertEqual(data["config"]["_optimized_defaults"]["source"], str(backup))

    def test_stock_long_backtest_chart_rebuilds_existing_default_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "stock_strategy_best_fold_paths.svg"
            output.write_text("<svg>Top10替换</svg>", encoding="utf-8")
            chart_payload = {"file": str(output), "replacement_top_n": 20}
            with patch("stock_strategy_optimizer.LONG_FOLD_PATH_CHART_FILE", output), \
                 patch("stock_strategy_optimizer.create_long_fold_path_chart", return_value=chart_payload) as build_chart:
                payload = stock_strategy_service.build_long_backtest_chart({})

        self.assertTrue(payload["is_default"])
        self.assertEqual(payload["chart"], chart_payload)
        build_chart.assert_called_once()
        self.assertEqual(build_chart.call_args.args[1], output)

    def test_stock_long_backtest_chart_file_serves_svg_only(self):
        with tempfile.NamedTemporaryFile(suffix=".svg") as tmp:
            Path(tmp.name).write_text("<svg></svg>", encoding="utf-8")
            with patch("app.routes.stock.resolve_long_backtest_chart_file", return_value=Path(tmp.name)):
                response = self.client.get("/api/stock/long-backtest-chart/file/stock_strategy_best_fold_paths.svg")
                self.assertEqual(response.status_code, 200)
                self.assertIn("image/svg+xml", response.content_type)

                response.close()

    def test_stock_dashboard_has_long_backtest_chart_button(self):
        body = (ROOT_DIR / "app/templates/stock/dashboard_body.html").read_text(encoding="utf-8")
        script = (ROOT_DIR / "app/static/js/stock-dashboard.js").read_text(encoding="utf-8")

        for element_id in (
            "long-chart-show",
            "long-chart-modal",
            "long-chart-status",
            "long-chart-image",
            "long-chart-open",
            "long-chart-close",
        ):
            with self.subTest(element_id=element_id):
                self.assertIn(f'id="{element_id}"', body)

        self.assertIn('role="dialog"', body)
        self.assertIn('fetch("/api/stock/long-backtest-chart"', script)
        self.assertIn('method: "POST"', script)
        self.assertIn('body: JSON.stringify({config, strategy: chartStrategy})', script)
        self.assertIn('const visible = active === "long" || active === "smallcap"', script)

    def test_stock_dashboard_refreshes_legacy_lhb_warning_copy(self):
        script = (ROOT_DIR / "app/static/js/stock-dashboard.js").read_text(encoding="utf-8")

        self.assertIn('["近期上龙虎榜，长线按避雷处理", "近期上龙虎榜，波动加剧"]', script)
        self.assertIn("esc(warningCopy(r))", script)

    def test_stock_kline_endpoint_returns_weekly_bars(self):
        payload = {"code": "002511", "period": "week", "bars": [{"date": "2026-06-19", "close": 10.0}]}
        with patch("app.routes.stock.kline_bars", return_value=payload) as kline:
            response = self.client.get("/api/stock/kline?code=002511&period=week&limit=640")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), payload)
        kline.assert_called_once_with("002511", 640, "week")

    def test_stock_page_renders_native_dashboard_without_iframe(self):
        response = self.client.get("/stock")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('id="stock-native-dashboard"', html)
        self.assertIn("A股策略配置台", html)
        self.assertEqual(html.count("A股策略配置台"), 1)
        self.assertIn('id="refresh-data"', html)
        self.assertIn("刷新数据", html)
        self.assertIn('id="data-refresh-time"', html)
        self.assertIn('id="data-refresh-log"', html)
        self.assertNotIn("独立打开", html)
        self.assertNotIn("<iframe", html)
        self.assertNotIn("/stock/app", html)

    def test_stock_dashboard_css_is_scoped(self):
        css = (ROOT_DIR / "app/static/css/stock-dashboard.css").read_text(encoding="utf-8")
        body = (ROOT_DIR / "app/templates/stock/dashboard_body.html").read_text(encoding="utf-8")
        script = (ROOT_DIR / "app/static/js/stock-dashboard.js").read_text(encoding="utf-8")

        self.assertIn(".stock-native", css)
        self.assertIn("A股策略配置台", body)
        self.assertIn('id="refresh-data"', body)
        self.assertIn('id="data-refresh-log"', body)
        self.assertIn("/api/stock/refresh", script)
        self.assertNotIn("<script>", body)
        self.assertNotIn(":root", css)
        self.assertIsNone(re.search(r"(^|[{}])\s*body\s*\{", css))

    def test_stock_refresh_log_scrolls_with_parameter_panel(self):
        body = (ROOT_DIR / "app/templates/stock/dashboard_body.html").read_text(encoding="utf-8")

        sticky = body.split('<div class="side-body">', 1)[0]
        side_body = body.split('<div class="side-body">', 1)[1]
        self.assertNotIn('id="data-refresh-log-panel"', sticky)
        self.assertIn('id="data-refresh-log-panel"', side_body)
        self.assertLess(side_body.index('id="data-refresh-log-panel"'), side_body.index('id="params"'))

    def test_stock_sidebar_scroll_does_not_cover_factor_sliders(self):
        css = (ROOT_DIR / "app/static/css/stock-dashboard.css").read_text(encoding="utf-8")

        self.assertIn("grid-template-rows: auto minmax(0, 1fr)", css)
        self.assertIn("overflow: hidden", css)
        self.assertIn("position: relative", css)
        self.assertIn("min-height: 0", css)
        self.assertIn("overflow: auto", css)

    def test_stock_factor_sliders_use_custom_pointer_drag(self):
        css = (ROOT_DIR / "app/static/css/stock-dashboard.css").read_text(encoding="utf-8")
        script = (ROOT_DIR / "app/static/js/stock-dashboard.js").read_text(encoding="utf-8")

        self.assertIn("touch-action: none", css)
        self.assertIn("cursor: ew-resize", css)
        self.assertIn("function bindRangeDrag(range, onValue)", script)
        self.assertIn('range.addEventListener("pointerdown"', script)
        self.assertIn('range.addEventListener("pointermove"', script)
        self.assertIn("setPointerCapture", script)
        self.assertIn("valueAt(event.clientX)", script)

    def test_fund_status_api_loads(self):
        response = self.client.get("/api/fund/status")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("report", payload)
        self.assertIn("report_data", payload)
        self.assertIn("signals", payload)
        self.assertIn("generate", payload)
        self.assertIn("refresh", payload)
        self.assertIn("command_text", payload["refresh"])
        self.assertIn("log_lines", payload["refresh"])

    def test_fund_page_renders_native_report_without_iframe(self):
        response = self.client.get("/fund")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('id="fund-native-report"', html)
        self.assertIn("数据基于", html)
        self.assertIn("刷新数据", html)
        self.assertIn("编辑 fund/funds.py", html)
        self.assertIn("Fund Quantitative Metrics Report", html)
        self.assertIn("保存编辑", html)
        self.assertIn("关闭", html)
        self.assertIn("取消编辑", html)
        self.assertIn('data-fund-editor-modal', html)
        self.assertIn('data-fund-log', html)
        self.assertIn('data-field="job-log"', html)
        self.assertNotIn('id="fund-report-frame"', html)
        self.assertNotIn(">生成报告<", html)
        self.assertNotIn(">HTML<", html)
        self.assertNotIn('id="theme-toggle"', html)
        self.assertNotIn("深色", html)
        self.assertNotIn("浅色", html)

    def test_metric_rows_include_signal_state_for_quick_filters(self):
        payload = {
            "generated_at": "2026-06-19 12:00",
            "period_labels": ["近一周"],
            "summary": {},
            "sections": {
                "equity": {
                    "id": "equity",
                    "title": "股票型基金（中高风险）",
                    "benchmark_names": ["沪深300"],
                    "rows": [
                        {
                            "code": "008115",
                            "name": "天弘中证红利低波动100联接C",
                            "total_asset": "143.09",
                            "is_held": True,
                            "asset_class": "",
                            "benchmarks": [[("1.23", "red")]],
                        }
                    ],
                }
            },
            "signal_rows": [
                {
                    "code": "008115",
                    "name": "天弘中证红利低波动100联接C",
                    "is_held": True,
                    "signal_state": "买入",
                    "signal": {"overall": "买入", "recent_navs": []},
                }
            ],
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", dir=ROOT_DIR, delete=False) as tmp:
            tmp.write(json.dumps(payload, ensure_ascii=False))
            report_path = Path(tmp.name)

        try:
            app = create_app()
            app.config["FUND_REPORT_DATA_FILE"] = report_path
            html = app.test_client().get("/fund").get_data(as_text=True)
        finally:
            report_path.unlink(missing_ok=True)

        self.assertRegex(
            html,
            r'<tr data-code="008115"[^>]*data-section="equity"[^>]*data-signal="买入"',
        )

    def test_fund_report_css_is_scoped(self):
        payload = {
            "generated_at": "2026-07-19 12:00",
            "period_labels": ["近一周"],
            "sections": {
                "equity": {
                    "id": "equity",
                    "title": "股票型基金",
                    "benchmark_names": ["沪深300"],
                    "rows": [],
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_file = Path(tmp_dir) / "fund_report_data.json"
            report_file.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            view = load_fund_report_view(report_file)

        css = (ROOT_DIR / "app/static/css/fund-report.css").read_text(encoding="utf-8")

        self.assertTrue(view.exists)
        self.assertEqual([section["id"] for section in view.sections], ["equity"])
        self.assertIn(".fund-native", css)
        self.assertNotIn(":root", css)
        self.assertIsNone(re.search(r"(^|[{}])\s*body\s*\{", css))

    def test_refresh_button_has_busy_guard(self):
        script = (ROOT_DIR / "app/static/js/fund-report.js").read_text(encoding="utf-8")

        self.assertIn("button.disabled = busy", script)
        self.assertIn("setRefreshBusy(true)", script)
        self.assertIn("data.refresh && data.refresh.running", script)
        self.assertIn("updateRefreshLog(data.refresh)", script)
        self.assertIn("完成 ${state.elapsed_sec || 0}s", script)

    def test_fund_editor_highlighter_has_escape_helper(self):
        script = (ROOT_DIR / "app/static/js/fund-report.js").read_text(encoding="utf-8")

        self.assertIn("const esc = (text)", script)
        self.assertIn("function highlightPython(source)", script)
        self.assertIn("esc(source.slice", script)

    def test_fund_editor_api_loads_funds_file(self):
        response = self.client.get("/api/fund/editor")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["path"], "fund/funds.py")
        self.assertIn("get_funds", payload["content"])

    def test_fund_editor_api_saves_only_validated_fund_config(self):
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as tmp_dir:
            target = Path(tmp_dir) / "fund.py"
            codes_target = Path(tmp_dir) / "fund_codes.json"
            target.write_text(VALID_FUND_EDITOR_CONTENT, encoding="utf-8")

            with patch("app.routes.fund.FUND_EDITOR_FILE", target), \
                 patch("app.routes.fund.FUND_CODES_FILE", codes_target):
                response = self.client.put(
                    "/api/fund/editor",
                    json={"content": VALID_FUND_EDITOR_CONTENT},
                )

            self.assertEqual(response.status_code, 200)
            saved = target.read_text(encoding="utf-8")
            self.assertEqual(saved, VALID_FUND_EDITOR_CONTENT)
            self.assertIn("# 用户可以保留基金分类和候选代码注释。", saved)
            self.assertNotIn("import os", saved)
            self.assertEqual(
                json.loads(codes_target.read_text(encoding="utf-8")),
                ["510310", "510580", "161119", "008115", "040046", "011555"],
            )

    def test_fund_editor_api_accepts_structured_config(self):
        config = {
            "compare_index": ["510310"],
            "compare_index_bond": ["161119"],
            "hold_index": ["510310"],
            "fund_index": ["008115"],
            "fund_stock": ["040046"],
            "fund_bond": ["011555"],
        }
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as tmp_dir:
            target = Path(tmp_dir) / "fund.py"
            codes_target = Path(tmp_dir) / "fund_codes.json"
            target.write_text(VALID_FUND_EDITOR_CONTENT, encoding="utf-8")

            with patch("app.routes.fund.FUND_EDITOR_FILE", target), \
                 patch("app.routes.fund.FUND_CODES_FILE", codes_target):
                response = self.client.put("/api/fund/editor", json={"funds": config})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["funds"], config)

    def test_fund_editor_api_rejects_invalid_python(self):
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as tmp_dir:
            target = Path(tmp_dir) / "fund.py"
            codes_target = Path(tmp_dir) / "fund_codes.json"
            target.write_text(VALID_FUND_EDITOR_CONTENT, encoding="utf-8")

            with patch("app.routes.fund.FUND_EDITOR_FILE", target), \
                 patch("app.routes.fund.FUND_CODES_FILE", codes_target):
                response = self.client.put("/api/fund/editor", json={"content": "def broken(:\n"})

            self.assertEqual(response.status_code, 400)
            self.assertEqual(target.read_text(encoding="utf-8"), VALID_FUND_EDITOR_CONTENT)

    def test_fund_editor_api_rejects_executable_python(self):
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as tmp_dir:
            target = Path(tmp_dir) / "fund.py"
            codes_target = Path(tmp_dir) / "fund_codes.json"
            target.write_text(VALID_FUND_EDITOR_CONTENT, encoding="utf-8")
            malicious = VALID_FUND_EDITOR_CONTENT + "\n__import__('os').system('echo unsafe')\n"

            with patch("app.routes.fund.FUND_EDITOR_FILE", target), \
                 patch("app.routes.fund.FUND_CODES_FILE", codes_target):
                response = self.client.put("/api/fund/editor", json={"content": malicious})

            self.assertEqual(response.status_code, 400)
            self.assertIn("不允许执行", response.get_json()["error"])
            self.assertEqual(target.read_text(encoding="utf-8"), VALID_FUND_EDITOR_CONTENT)
            self.assertFalse(codes_target.exists())

    def test_fund_editor_api_rejects_tuple_that_would_break_list_concatenation(self):
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as tmp_dir:
            target = Path(tmp_dir) / "fund.py"
            codes_target = Path(tmp_dir) / "fund_codes.json"
            target.write_text(VALID_FUND_EDITOR_CONTENT, encoding="utf-8")
            invalid = VALID_FUND_EDITOR_CONTENT.replace(
                'fund_index = ["008115"]', 'fund_index = ("008115",)'
            )

            with patch("app.routes.fund.FUND_EDITOR_FILE", target), \
                 patch("app.routes.fund.FUND_CODES_FILE", codes_target):
                response = self.client.put("/api/fund/editor", json={"content": invalid})

            self.assertEqual(response.status_code, 400)
            self.assertIn("必须是基金代码列表", response.get_json()["error"])
            self.assertEqual(target.read_text(encoding="utf-8"), VALID_FUND_EDITOR_CONTENT)

    def test_fund_editor_api_rejects_oversized_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as tmp_dir:
            target = Path(tmp_dir) / "fund.py"
            codes_target = Path(tmp_dir) / "fund_codes.json"
            target.write_text(VALID_FUND_EDITOR_CONTENT, encoding="utf-8")
            oversized = VALID_FUND_EDITOR_CONTENT + "#" + ("x" * (256 * 1024))

            with patch("app.routes.fund.FUND_EDITOR_FILE", target), \
                 patch("app.routes.fund.FUND_CODES_FILE", codes_target):
                response = self.client.put("/api/fund/editor", json={"content": oversized})

            self.assertEqual(response.status_code, 400)
            self.assertIn("256 KiB", response.get_json()["error"])
            self.assertEqual(target.read_text(encoding="utf-8"), VALID_FUND_EDITOR_CONTENT)

    def test_fund_run_refuses_unvalidated_python_without_starting_job(self):
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as tmp_dir:
            target = Path(tmp_dir) / "fund.py"
            codes_target = Path(tmp_dir) / "fund_codes.json"
            target.write_text("raise SystemExit('unsafe')\n", encoding="utf-8")

            with patch("app.routes.fund.FUND_EDITOR_FILE", target), \
                 patch("app.routes.fund.FUND_CODES_FILE", codes_target), \
                 patch("app.routes.fund.start_command_job") as start_mock:
                response = self.client.post("/api/fund/run")

            self.assertEqual(response.status_code, 409)
            start_mock.assert_not_called()
            self.assertFalse(codes_target.exists())

    def test_fund_report_generate_refuses_unvalidated_python(self):
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as tmp_dir:
            target = Path(tmp_dir) / "fund.py"
            target.write_text("raise SystemExit('unsafe')\n", encoding="utf-8")

            with patch("app.routes.fund.FUND_EDITOR_FILE", target), \
                 patch("app.routes.fund.start_command_job") as start_mock:
                response = self.client.post("/api/fund/report/generate")

            self.assertEqual(response.status_code, 409)
            start_mock.assert_not_called()

    def test_fund_run_syncs_codes_from_validated_config_before_start(self):
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as tmp_dir:
            target = Path(tmp_dir) / "fund.py"
            codes_target = Path(tmp_dir) / "fund_codes.json"
            target.write_text(VALID_FUND_EDITOR_CONTENT, encoding="utf-8")

            with patch("app.routes.fund.FUND_EDITOR_FILE", target), \
                 patch("app.routes.fund.FUND_CODES_FILE", codes_target), \
                 patch("app.routes.fund.start_command_job", return_value=True) as start_mock:
                response = self.client.post("/api/fund/run")

            self.assertEqual(response.status_code, 202)
            start_mock.assert_called_once_with(
                "fund-refresh",
                [sys.executable, "-B", "fund_data_refresh.py", "--no-proxy"],
                cwd=ROOT_DIR,
                timeout=1800,
                resource_key="fund-data-refresh",
            )
            self.assertEqual(
                json.loads(codes_target.read_text(encoding="utf-8")),
                ["510310", "510580", "161119", "008115", "040046", "011555"],
            )

    def test_fund_refresh_entrypoint_does_not_execute_funds_module(self):
        script = (ROOT_DIR / "fund_data_refresh.py").read_text(encoding="utf-8")

        self.assertNotIn("import funds", script)
        self.assertIn("_load_validated_fund_config", script)
        self.assertIn("_sync_fund_codes", script)

    def test_fund_editor_refuses_save_while_fund_resource_is_running(self):
        with patch("app.routes.fund.is_resource_running", return_value=True):
            response = self.client.put(
                "/api/fund/editor", json={"content": VALID_FUND_EDITOR_CONTENT}
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("运行中", response.get_json()["error"])

    def test_command_job_captures_recent_logs(self):
        job_id = f"test-log-{time.time_ns()}"

        started = start_command_job(
            job_id,
            [sys.executable, "-c", "print('hello from job log')"],
            cwd=ROOT_DIR,
            timeout=5,
        )
        self.assertTrue(started)

        state = get_job_state(job_id)
        for _ in range(100):
            if not state["running"]:
                break
            time.sleep(0.05)
            state = get_job_state(job_id)

        self.assertFalse(state["running"])
        self.assertTrue(state["ok"])
        self.assertIn("command_text", state)
        self.assertIn("hello from job log", "\n".join(state["log_lines"]))


if __name__ == "__main__":
    unittest.main()
