import json
import sys
import time
import unittest
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

from app import create_app
from app.config import FUND_REPORT_DATA_FILE, ROOT_DIR
from app.services.job_service import get_job_state, start_command_job
from app.services.fund_report_service import load_fund_report_view
from app.services import stock_strategy_service
from funds import get_funds, get_funds_bond


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

        self.assertRegex(refresh_css, r"\.radar-filters\s*\{\s*position:\s*static;")
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

        self.assertIn("FinancialAnalysisPages.fund = initFundPage", fund_script)
        self.assertIn("dataset.fundPageInitialized", fund_script)
        self.assertIn("reloadCurrentPageContent", fund_script)
        self.assertIn("await reloadCurrentPageContent()", fund_script)
        self.assertLess(
            fund_script.index('if (state.running) return "运行中";'),
            fund_script.index('if (state.ok === null) return "空闲";'),
        )
        self.assertIn("let pollTask = null;", fund_script)
        self.assertIn("if (pollTask) return pollTask;", fund_script)
        self.assertIn("if (!disposed && data.refresh && data.refresh.running)", fund_script)
        self.assertIn("FinancialAnalysisPages.stock = initStockDashboard", stock_script)
        self.assertIn("dataset.stockDashboardInitialized", stock_script)
        self.assertIn("将对长线/短线各运行 1500 次 Optuna/TPE 参数搜索回测", stock_script)
        self.assertIn("python stock_strategy_optimizer.py --iterations 1500", stock_script)
        self.assertIn("长线和短线会以独立进程并行搜索", stock_script)
        self.assertIn("约需 10 分钟左右", stock_script)
        self.assertNotIn("约需 2 分钟左右", stock_script)
        self.assertIn("clearTimeout(runTimer)", stock_script)
        self.assertIn("clearInterval(optimizeTimer)", stock_script)
        self.assertIn("clearInterval(refreshTimer)", stock_script)

        stock_html = self.client.get("/stock").get_data(as_text=True)
        self.assertIn("长线/短线各 1500 次 Optuna/TPE 参数搜索回测", stock_html)
        self.assertIn("两个独立进程并行", stock_html)
        self.assertIn("约需 10 分钟左右", stock_html)

    def test_radar_script_shows_market_cap_and_watch_label(self):
        body = (ROOT_DIR / "app/templates/radar/dashboard.html").read_text(encoding="utf-8")
        script = (ROOT_DIR / "app/static/js/radar.js").read_text(encoding="utf-8")
        css = (ROOT_DIR / "app/static/css/radar.css").read_text(encoding="utf-8")
        refresh_css = (ROOT_DIR / "app/static/css/ui-refresh.css").read_text(encoding="utf-8")

        self.assertIn('"观望⚪"', script)
        self.assertIn('replace("空仓观望", "观望")', script)
        self.assertIn('sortTh("market_cap_yi", "市值")', script)
        self.assertIn('sortTh("realtime_price", "现价")', script)
        self.assertIn('sortTh("realtime_change_pct", "涨幅")', script)
        self.assertIn('sortTh("opportunity_score", "机会分")', script)
        self.assertNotIn('sortTh("ambush_score", "吸筹分")', script)
        self.assertNotIn('sortTh("distribution_score", "出货分")', script)
        self.assertIn('key: "opportunity_score"', script)
        self.assertIn("户数/回购已并入吸筹总分", script)
        self.assertIn("fmtMarketCap(s.market_cap_yi)", script)
        self.assertIn("pi-stock-meta", script)
        self.assertIn(".pi-stock-meta", css)
        self.assertIn('id="radar-export-limit"', body)
        self.assertIn('value="10"', body)
        self.assertIn('id="radar-export"', body)
        self.assertIn('id="radar-live-quote"', body)
        self.assertIn('id="radar-live-status"', body)
        self.assertIn("实时行情", body)
        self.assertIn('id="radar-sw2-filter"', body)
        self.assertIn('id="radar-sw2-search"', body)
        self.assertIn('id="radar-sw2-min-heat"', body)
        self.assertIn('id="radar-sw2-min-heat" type="number" min="0" max="100" step="1" value="0"', body)
        self.assertNotIn('id="radar-sw2-min-heat" type="number" min="0" max="100" step="1" value="50"', body)
        self.assertIn('id="radar-sw2-options"', body)
        self.assertIn("热度Top10", body)
        self.assertIn("全选", body)
        self.assertIn("搜索二级行业", body)
        self.assertNotIn("搜索二级行业 / 热度", body)
        self.assertLess(body.index('id="radar-sw2-hot"'), body.index('id="radar-sw2-select-all"'))
        self.assertLess(body.index('id="radar-sw2-select-all"'), body.index('id="radar-sw2-clear"'))
        self.assertLess(body.index('id="radar-phase-filter"'), body.index('id="radar-sw2-filter"'))
        self.assertLess(body.index('id="radar-sw2-filter"'), body.index('id="radar-min-score"'))
        self.assertLess(body.index('id="radar-hide-dist"'), body.index('id="radar-export-limit"'))
        self.assertLess(body.index('id="radar-export-limit"'), body.index('id="radar-export"'))
        self.assertIn("function buildSw2Options()", script)
        self.assertIn("selectedSw2Industries", script)
        self.assertIn("function renderSw2Filter()", script)
        self.assertIn("function setSw2PanelOpen(open)", script)
        self.assertIn("function visibleSw2Options()", script)
        self.assertIn("new Set(selectedSw2Industries)", script)
        self.assertIn("!f.sw2.has(stockSw2Name(s))", script)
        self.assertIn('$("radar-sw2-min-heat").addEventListener("input", renderSw2Filter)', script)
        self.assertIn('$("radar-sw2-select-all").onclick', script)
        self.assertIn("new Set(visibleSw2Options().map((o) => o.name))", script)
        self.assertIn("o.heat === null || o.heat <= minHeat", script)
        self.assertIn("o.name.toLowerCase().includes(q)", script)
        self.assertNotIn("heatText(o.heat)} ${o.count}只`.toLowerCase()", script)
        self.assertIn("o.heat !== null).slice(0, 10)", script)
        self.assertIn(".industry-filter-panel", css)
        self.assertIn(".industry-heat-filter", css)
        self.assertIn(".industry-option-heat", css)
        self.assertIn("function exportTopOpportunityStocks()", script)
        self.assertIn('aria-sort="${ariaSort}"', script)
        self.assertIn('event.key !== "Enter" && event.key !== " "', script)
        self.assertIn("REALTIME_REFRESH_MS = 120000", script)
        self.assertIn("REALTIME_SOURCE_LABEL", script)
        self.assertIn('tencent_batch: "腾讯"', script)
        self.assertIn('sina_batch: "新浪"', script)
        self.assertIn('fetch("/api/radar/realtime")', script)
        self.assertIn("function setRealtimeEnabled(enabled)", script)
        self.assertIn("async function fetchRealtimeData(queueIfBusy = false)", script)
        self.assertIn("let realtimeRefreshPending = false", script)
        self.assertIn("let realtimeRefreshQueued = false", script)
        self.assertIn('updateRealtimeStatus("等待新股票池…", "live")', script)
        self.assertIn("if (wasRadarRunBusy && !radarRunBusy && realtimeRefreshPending)", script)
        self.assertIn("fetchRealtimeData(true)", script)
        self.assertIn("realtime_quote", script)
        self.assertIn("text/plain;charset=utf-8", script)
        self.assertIn("hot_money_radar_opportunity_top", script)
        self.assertIn("Number(b.opportunity_score) - Number(a.opportunity_score)", script)
        self.assertIn("Number(b.ambush_score) - Number(a.ambush_score)", script)
        self.assertIn("已导出机会分Top", script)
        self.assertIn('${cleanField(s.code)},${cleanField(s.name)},', script)
        self.assertNotIn('sortTh("capital_score", "资金分")', script)
        self.assertIn(".radar-export-limit", css)
        self.assertIn(".radar-live-status", css)
        self.assertIn('title="按筹码成本估算，当前价以下的筹码占比">获利盘</th>', script)
        self.assertIn("fmtRatioPct(sig.chip_winner)", script)
        for hidden_header in ("量比", "价分位", "换手分位", "CMF", "筹码", "连板", "走势相似行业"):
            self.assertNotIn(f'<th class="num">{hidden_header}</th>', script)
            self.assertNotIn(f'<th>{hidden_header}</th>', script)
        self.assertNotIn("themeCell(s)", script)
        self.assertRegex(
            css,
            r"\.radar-table\s*\{\s*max-height:\s*none;\s*overflow-x:\s*auto;\s*overflow-y:\s*hidden;",
        )
        self.assertRegex(
            refresh_css,
            r"\.radar-kline\s*\{\s*position:\s*sticky;\s*align-self:\s*start;",
        )
        self.assertIn("走势相似行业", body)
        self.assertIn("二级行业", script)
        self.assertIn("function sw2Cell(s)", script)
        self.assertIn("s.sw2_heat_pctile", script)
        self.assertIn("esc(industry) + heat", script)
        self.assertNotIn("三级行业", script)
        self.assertNotIn("跟踪二级行业", script)

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

    def test_live2d_widget_script_is_vendored_locally(self):
        html = self.client.get("/fund").get_data(as_text=True)
        self.assertIn('type="module"', html)
        self.assertIn("vendor/live2d-widget/autoload.js", html)
        self.assertNotIn("fastly.jsdelivr.net/npm/live2d-widgets", html)

        vendor_dir = ROOT_DIR / "app/static/vendor/live2d-widget"
        for asset in (
            "autoload.js",
            "waifu.css",
            "waifu-tips.js",
            "waifu-tips.json",
            "live2d.min.js",
            "chunk/index.js",
            "chunk/index2.js",
            "LICENSE",
        ):
            with self.subTest(asset=asset):
                self.assertTrue((vendor_dir / asset).exists())

        autoload = (vendor_dir / "autoload.js").read_text(encoding="utf-8")
        waifu_css = (vendor_dir / "waifu.css").read_text(encoding="utf-8")
        self.assertIn("new URL('./', import.meta.url).href", autoload)
        self.assertIn("const defaultModelId = 2;", autoload)
        self.assertIn("const defaultTextureId = 15;", autoload)
        self.assertIn("const switchTextureIds = [7, 8, 13, 14, 15, 16, 17];", autoload)
        self.assertIn("forceNextSwitchTexture(defaultModelId, switchTextureIds, defaultModelTextureCount);", autoload)
        self.assertIn("event.target.closest('#waifu-tool-switch-texture')", autoload)
        self.assertIn("localStorage.setItem('modelId', String(defaultModelId));", autoload)
        self.assertIn("localStorage.setItem('modelTexturesId', String(defaultTextureId));", autoload)
        self.assertIn("drag: true", autoload)
        self.assertNotIn("const live2d_path = 'https://fastly.jsdelivr.net", autoload)
        self.assertIn("z-index: 40;", waifu_css)

        for asset in ("autoload.js", "waifu.css", "waifu-tips.js", "waifu-tips.json", "live2d.min.js"):
            with self.subTest(static_asset=asset):
                response = self.client.get(f"/static/vendor/live2d-widget/{asset}")
                try:
                    self.assertEqual(response.status_code, 200)
                    self.assertGreater(len(response.data), 100)
                finally:
                    response.close()

    def test_stock_config_api_loads(self):
        response = self.client.get("/api/config")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("config", payload)
        self.assertIn("factors", payload)

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

    def test_stock_refresh_uses_radar_refresh_script(self):
        with patch("app.services.stock_strategy_service.start_command_job", return_value=True) as start_job:
            self.assertTrue(stock_strategy_service.start_stock_data_refresh())

        command = start_job.call_args.args[1]
        self.assertEqual(command, ["bash", "stock_radar_fresh_data.sh"])
        self.assertNotIn("--strict", command)
        self.assertNotIn("--timeout", command)

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
        with patch("app.routes.stock.build_long_backtest_chart", return_value=payload) as build_chart:
            response = self.client.post("/api/stock/long-backtest-chart", json={"config": {"long": {"top_n": 10}}})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), payload)
        build_chart.assert_called_once_with({"long": {"top_n": 10}})

    def test_stock_config_endpoint_falls_back_to_backup_optimized_config(self):
        payload = {
            "generated_at": "2026-06-25 14:00:00",
            "iterations_per_strategy": 1500,
            "seed": 42,
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
        css = (ROOT_DIR / "app/static/css/stock-dashboard.css").read_text(encoding="utf-8")
        script = (ROOT_DIR / "app/static/js/stock-dashboard.js").read_text(encoding="utf-8")

        self.assertIn('id="long-chart-modal"', body)
        self.assertIn('id="long-chart-show"', body)
        self.assertIn('id="long-chart-status"', body)
        self.assertIn(">策略走势</button>", body)
        self.assertIn('id="export-picks"', body)
        self.assertIn(">导出</button>", body)
        self.assertLess(body.index('id="refresh-data"'), body.index('id="long-chart-show"'))
        self.assertLess(body.index('id="long-chart-show"'), body.index('id="export-picks"'))
        self.assertIn("/api/stock/long-backtest-chart", script)
        self.assertIn("正在回测策略走势，请稍后...", script)
        self.assertIn("function showLongChartStatus", script)
        self.assertIn(".chart-modal-status[hidden]", css)
        self.assertIn(".chart-modal-body img[hidden]", css)
        self.assertIn(".chart-modal-actions a[hidden]", css)
        self.assertIn('$("long-chart-status").textContent = "";', script)
        self.assertIn("button.hidden = !visible", script)
        self.assertIn('const visible = active === "long"', script)
        self.assertIn("function exportCurrentPicks()", script)
        self.assertIn("text/plain;charset=utf-8", script)
        self.assertIn("anchor.download", script)
        self.assertIn("slice(0, limit", script)
        self.assertNotIn("picks.slice(0, 12)", script)
        self.assertIn('${cleanField(pick.code)},${cleanField(pick.name)},', script)
        self.assertIn("/api/stock/kline", script)
        self.assertIn("period=week", script)
        self.assertIn("function drawMiniKline", script)
        self.assertIn('class="stock-mini-kline"', script)
        self.assertIn(".stock-mini-kline", css)
        self.assertIn(".chart-modal", css)
        self.assertIn(".chart-modal-status", css)
        self.assertNotIn("long-chart-card", css)
        self.assertNotIn("查看走势", script)
        self.assertIn('data-stock-view="results"', body)
        self.assertIn('data-stock-view="settings"', body)
        self.assertIn("function setMobileView(view)", script)
        self.assertIn('event.key === "Escape"', script)

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
        self.assertIn("编辑funds.py", html)
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

    def test_bond_fund_config_carries_shared_holdings(self):
        equity_config = get_funds()
        bond_config = get_funds_bond()

        held_bond_codes = set(equity_config["hold_index"]) & set(bond_config["fund"])

        self.assertTrue(held_bond_codes)
        self.assertLessEqual(held_bond_codes, set(bond_config.get("hold_index", [])))

    def test_fund_report_css_is_scoped(self):
        view = load_fund_report_view(FUND_REPORT_DATA_FILE)
        css = (ROOT_DIR / "app/static/css/fund-report.css").read_text(encoding="utf-8")

        self.assertTrue(view.exists)
        self.assertGreater(len(view.sections), 0)
        self.assertIn(".fund-native", css)
        self.assertNotIn(":root", css)

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
        self.assertEqual(payload["path"], "funds.py")
        self.assertIn("get_funds", payload["content"])

    def test_fund_editor_api_saves_patched_target_file(self):
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as tmp_dir:
            target = Path(tmp_dir) / "fund.py"
            target.write_text("value = 1\n", encoding="utf-8")

            with patch("app.routes.fund.FUND_EDITOR_FILE", target):
                response = self.client.put("/api/fund/editor", json={"content": "value = 2\n"})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 2\n")

    def test_fund_editor_api_rejects_invalid_python(self):
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as tmp_dir:
            target = Path(tmp_dir) / "fund.py"
            target.write_text("value = 1\n", encoding="utf-8")

            with patch("app.routes.fund.FUND_EDITOR_FILE", target):
                response = self.client.put("/api/fund/editor", json={"content": "def broken(:\n"})

            self.assertEqual(response.status_code, 400)
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")

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
