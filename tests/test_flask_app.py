import sys
import time
import unittest
import re
from unittest.mock import patch

from app import create_app
from app.config import FUND_REPORT_DATA_FILE, ROOT_DIR
from app.services.job_service import get_job_state, start_command_job
from app.services.fund_report_service import load_fund_report_view
from app.services import stock_strategy_service


class FlaskAppTest(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_workspace_pages_load(self):
        for path in ("/", "/fund", "/stock"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/html", response.content_type)

    def test_stock_config_api_loads(self):
        response = self.client.get("/api/config")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("config", payload)
        self.assertIn("factors", payload)

    def test_stock_health_reports_warmup_state(self):
        response = self.client.get("/api/stock/health")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("refresh", payload)
        self.assertIn("refresh_job", payload)
        self.assertIn("warmup", payload)
        self.assertIn("log_lines", payload["refresh_job"])
        self.assertIn("running", payload["warmup"])
        self.assertIn("elapsed_sec", payload["warmup"])

    def test_stock_refresh_endpoint_starts_job(self):
        with patch("app.routes.stock.start_stock_data_refresh", return_value=True):
            response = self.client.post("/api/stock/refresh")

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json(), {"started": True})

    def test_stock_refresh_uses_full_non_strict_command(self):
        with patch("app.services.stock_strategy_service.start_command_job", return_value=True) as start_job:
            self.assertTrue(stock_strategy_service.start_stock_data_refresh())

        command = start_job.call_args.args[1]
        self.assertIn("--mode", command)
        self.assertEqual(command[command.index("--mode") + 1], "full")
        self.assertNotIn("--strict", command)

    def test_stock_refresh_requires_full_fetch_confirmation(self):
        script = (ROOT_DIR / "app/static/js/stock-dashboard.js").read_text(encoding="utf-8")

        self.assertIn('if ($("refresh-data").disabled) return;', script)
        self.assertIn("window.confirm(msg)", script)
        self.assertIn("全量拉取", script)
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
        self.assertIn('data-fund-log', html)
        self.assertIn('data-field="job-log"', html)
        self.assertNotIn('id="fund-report-frame"', html)
        self.assertNotIn(">生成报告<", html)
        self.assertNotIn(">HTML<", html)
        self.assertNotIn('id="theme-toggle"', html)
        self.assertNotIn("深色", html)
        self.assertNotIn("浅色", html)

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
