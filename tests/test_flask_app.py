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
        for path in ("/", "/fund", "/stock"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/html", response.content_type)

    def test_app_shell_loads_shared_scripts_once_per_page(self):
        for path in ("/", "/fund", "/stock"):
            with self.subTest(path=path):
                html = self.client.get(path).get_data(as_text=True)
                self.assertEqual(html.count("js/fund-report.js"), 1)
                self.assertEqual(html.count("js/stock-dashboard.js"), 1)
                self.assertEqual(html.count("js/app-navigation.js"), 1)
                self.assertEqual(html.count("vendor/live2d-widget/autoload.js"), 1)

    def test_internal_navigation_replaces_main_without_reloading_shell(self):
        script = (ROOT_DIR / "app/static/js/app-navigation.js").read_text(encoding="utf-8")

        self.assertIn('new Set(["/", "/fund", "/fund/report", "/stock"])', script)
        self.assertIn('headers: {"X-Requested-With": "fetch"}', script)
        self.assertIn("FinancialAnalysisPages.bootId", script)
        self.assertIn('querySelector(".app-main")', script)
        self.assertIn("currentMain.innerHTML = nextMain.innerHTML", script)
        self.assertIn("window.history.pushState", script)
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
        self.assertIn("将对长线/短线各运行 300 次参数搜索回测", stock_script)
        self.assertIn("python stock_strategy_optimizer.py --iterations 300", stock_script)
        self.assertIn("约需 3 分钟左右", stock_script)
        self.assertNotIn("约需 2 分钟左右", stock_script)
        self.assertIn("clearTimeout(runTimer)", stock_script)
        self.assertIn("clearInterval(optimizeTimer)", stock_script)
        self.assertIn("clearInterval(refreshTimer)", stock_script)

        stock_html = self.client.get("/stock").get_data(as_text=True)
        self.assertIn("长线/短线各 300 次随机搜索回测", stock_html)
        self.assertIn("约需 3 分钟左右", stock_html)

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

    def test_stock_refresh_uses_full_no_proxy_command(self):
        with patch("app.services.stock_strategy_service.start_command_job", return_value=True) as start_job:
            self.assertTrue(stock_strategy_service.start_stock_data_refresh())

        command = start_job.call_args.args[1]
        self.assertEqual(command, ["python", "stock_data_refresh.py", "--mode", "full", "--no-proxy"])
        self.assertNotIn("--strict", command)
        self.assertNotIn("--timeout", command)

    def test_stock_refresh_requires_full_fetch_confirmation(self):
        script = (ROOT_DIR / "app/static/js/stock-dashboard.js").read_text(encoding="utf-8")

        self.assertIn('if ($("refresh-data").disabled) return;', script)
        self.assertIn("window.confirm(msg)", script)
        self.assertIn("全量拉取", script)
        self.assertIn("no-proxy", script)
        self.assertIn("数据刷新时间会很久", script)

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
