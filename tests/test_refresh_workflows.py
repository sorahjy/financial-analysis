import os
import subprocess
import tempfile
import unittest
from datetime import date, datetime, time, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import requests

import fund_data_refresh
import stock_crawl_common
import stock_data_refresh
import stock_radar_fresh_data
from refresh_workflow import (
    build_child_environment,
    run_steps,
    strip_proxy_environment,
)
from stock_data_refresh import resolve_python


class RefreshWorkflowTest(unittest.TestCase):
    def test_no_proxy_environment_is_case_insensitive_and_forces_utf8(self):
        env = build_child_environment(
            {
                "Path": "C:\\Windows",
                "HTTP_PROXY": "http://upper",
                "https_proxy": "http://lower",
                "All_Proxy": "socks5://mixed",
                "No_Proxy": "localhost",
            },
            no_proxy=True,
            no_proxy_marker="STOCK_CRAWL_NO_PROXY",
        )

        self.assertEqual(env["Path"], "C:\\Windows")
        self.assertFalse(any(key.lower() in {"http_proxy", "https_proxy", "all_proxy"} for key in env))
        self.assertEqual(env["NO_PROXY"], "*")
        self.assertEqual(env["no_proxy"], "*")
        self.assertEqual(env["STOCK_CRAWL_NO_PROXY"], "1")
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(env["PYTHONUTF8"], "1")
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")

    def test_proxy_environment_cleanup_mutates_current_environment(self):
        with patch.dict(
            os.environ,
            {
                "Https_Proxy": "http://mixed-http",
                "All_Proxy": "socks5://mixed-all",
                "No_Proxy": "localhost",
            },
            clear=True,
        ):
            result = strip_proxy_environment(
                no_proxy_marker="STOCK_CRAWL_NO_PROXY",
            )

            self.assertIs(result, os.environ)
            self.assertFalse(
                any(
                    key.lower() in {"http_proxy", "https_proxy", "all_proxy"}
                    for key in os.environ
                )
            )
            self.assertEqual(os.environ["NO_PROXY"], "*")
            self.assertEqual(os.environ["no_proxy"], "*")
            self.assertEqual(os.environ["STOCK_CRAWL_NO_PROXY"], "1")

    def test_clean_child_environment_resolves_no_requests_proxy(self):
        env = build_child_environment(
            {
                "Https_Proxy": "http://proxy.invalid",
                "All_Proxy": "socks5://proxy.invalid",
            },
            no_proxy=True,
            no_proxy_marker="STOCK_CRAWL_NO_PROXY",
        )
        with patch.dict(os.environ, env, clear=True):
            proxies = requests.utils.get_environ_proxies(
                "https://finance.sina.com.cn/"
            )

        self.assertEqual(proxies, {})

    def test_stock_common_honors_case_insensitive_marker_and_proxy_keys(self):
        with patch.dict(
            os.environ,
            {
                "stock_crawl_no_proxy": "ON",
                "Https_Proxy": "http://mixed-http",
                "All_Proxy": "socks5://mixed-all",
            },
            clear=True,
        ):
            stock_crawl_common.strip_proxy_env()

            self.assertNotIn("Https_Proxy", os.environ)
            self.assertNotIn("All_Proxy", os.environ)
            self.assertEqual(os.environ["NO_PROXY"], "*")
            self.assertEqual(os.environ["STOCK_CRAWL_NO_PROXY"], "1")

    def test_runner_stops_at_first_failure_without_using_a_shell(self):
        calls = []

        def fake_runner(command, **kwargs):
            calls.append((command, kwargs))
            returncode = 9 if command[-1] == "second.py" else 0
            return subprocess.CompletedProcess(command, returncode)

        result = run_steps(
            [
                ("first", ["python", "first.py"]),
                ("second", ["python", "second.py"]),
                ("third", ["python", "third.py"]),
            ],
            cwd=Path("."),
            env={},
            runner=fake_runner,
        )

        self.assertEqual(result, 9)
        self.assertEqual([call[0][-1] for call in calls], ["first.py", "second.py"])
        self.assertTrue(all(call[1]["shell"] is False for call in calls))

    def test_fund_refresh_uses_one_python_for_all_steps(self):
        python = r"C:\repo\.venv\Scripts\python.exe"
        commands = [command for _, command in fund_data_refresh.build_refresh_steps(python)]

        self.assertEqual([command[0] for command in commands], [python] * 4)
        self.assertEqual(commands[0], [python, "-B", "-m", "scrapy", "crawl", "jijin"])
        self.assertEqual(
            commands[1:],
            [
                [python, "-B", "-m", "fund.fund_fetch_data"],
                [python, "-B", "-m", "fund.fund_technical_analysis"],
                [python, "-B", "-m", "fund.fund_generate_output"],
            ],
        )

    def test_fund_refresh_syncs_config_and_passes_no_proxy_environment(self):
        calls = []
        sync_calls = []

        def fake_runner(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fund_data_refresh.refresh_fund_data(
                no_proxy=True,
                python_executable="python-test",
                runner=fake_runner,
                syncer=lambda: sync_calls.append(True),
                cache_dir=Path(tmpdir) / "missing-cache",
                base_env={"HTTP_PROXY": "http://proxy"},
            )

        self.assertEqual(result, 0)
        self.assertEqual(sync_calls, [True])
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0][1]["env"]["FUND_CRAWL_NO_PROXY"], "1")
        self.assertNotIn("HTTP_PROXY", calls[0][1]["env"])

    def test_fund_refresh_honors_legacy_no_proxy_environment_flag(self):
        with patch.dict(os.environ, {"fund_crawl_no_proxy": "ON"}), patch.object(
            fund_data_refresh,
            "refresh_fund_data",
            return_value=0,
        ) as refresh:
            self.assertEqual(fund_data_refresh.main([]), 0)

        refresh.assert_called_once_with(no_proxy=True)

    def test_invalid_fund_config_stops_before_any_child_process(self):
        calls = []

        def reject_config():
            raise ValueError("unsafe config")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "jijin"
            cache_dir.mkdir()
            marker = cache_dir / "keep-me"
            marker.write_text("safe", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe config"):
                fund_data_refresh.refresh_fund_data(
                    runner=lambda command, **kwargs: calls.append((command, kwargs)),
                    syncer=reject_config,
                    cache_dir=cache_dir,
                )

            self.assertTrue(marker.exists())

        self.assertEqual(calls, [])

    def test_fund_scrapy_cache_expires_by_local_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "jijin"
            cache_dir.mkdir()
            stale = datetime.combine(date.today() - timedelta(days=1), time(hour=12)).timestamp()
            os.utime(cache_dir, (stale, stale))

            self.assertTrue(fund_data_refresh.prepare_scrapy_cache(cache_dir))
            self.assertFalse(cache_dir.exists())

            cache_dir.mkdir()
            current = datetime.combine(date.today(), time(hour=12)).timestamp()
            os.utime(cache_dir, (current, current))
            self.assertFalse(fund_data_refresh.prepare_scrapy_cache(cache_dir))
            self.assertTrue(cache_dir.exists())

    def test_stock_refresh_preserves_the_existing_five_step_order(self):
        python = r"C:\repo\.venv\Scripts\python.exe"
        commands = [command for _, command in stock_radar_fresh_data.build_refresh_steps(python)]

        self.assertEqual([command[0] for command in commands], [python] * 5)
        self.assertEqual(
            [command[2] for command in commands],
            [
                "stock_crawl_etf_pool.py",
                "stock_data_refresh.py",
                "plate_crawl_history.py",
                "stock_theme_candidates.py",
                "stock_hot_money_radar.py",
            ],
        )
        self.assertEqual(commands[1][-3:], ["--mode", "full", "--no-proxy"])
        self.assertEqual(commands[-1][-2:], ["--pool", "leader"])

    def test_stock_refresh_passes_no_proxy_and_utf8_to_every_step(self):
        calls = []

        def fake_runner(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0)

        result = stock_radar_fresh_data.refresh_stock_data(
            python_executable="python-test",
            runner=fake_runner,
            base_env={"Https_Proxy": "http://proxy"},
        )

        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 5)
        for _, kwargs in calls:
            env = kwargs["env"]
            self.assertFalse(any(key.lower() == "https_proxy" for key in env))
            self.assertEqual(env["STOCK_CRAWL_NO_PROXY"], "1")
            self.assertEqual(env["PYTHONIOENCODING"], "utf-8")

    def test_stock_data_refresh_applies_marker_to_local_and_child_steps(self):
        child_environments = []
        local_environments = []

        def fake_run_step(name, command, *, timeout, env=None, skip=False):
            child_environments.append(dict(env or {}))
            return stock_data_refresh.StepResult(
                name,
                " ".join(command),
                True,
                0,
                0.0,
                skipped=skip,
            )

        def fake_local_step(name, command, func):
            local_environments.append(dict(os.environ))
            return stock_data_refresh.StepResult(name, command, True, 0, 0.0)

        health = {
            "hot_money_history_stale_count": 0,
            "hot_money_history_stale_details": [],
        }
        with patch.dict(
            os.environ,
            {
                "STOCK_CRAWL_NO_PROXY": "yes",
                "Https_Proxy": "http://mixed-http",
                "All_Proxy": "socks5://mixed-all",
            },
            clear=True,
        ), patch.object(
            stock_data_refresh,
            "run_step",
            side_effect=fake_run_step,
        ), patch.object(
            stock_data_refresh,
            "local_step_result",
            side_effect=fake_local_step,
        ), patch.object(
            stock_data_refresh,
            "collect_data_health",
            return_value=health,
        ), patch.object(
            stock_data_refresh,
            "write_json_file",
        ):
            report = stock_data_refresh.refresh_before_server(mode="quick")

        self.assertTrue(report["ok"])
        self.assertTrue(child_environments)
        self.assertTrue(local_environments)
        for env in child_environments + local_environments:
            self.assertFalse(
                any(
                    key.lower() in {"http_proxy", "https_proxy", "all_proxy"}
                    for key in env
                )
            )
            self.assertEqual(env["NO_PROXY"], "*")
            self.assertEqual(env["STOCK_CRAWL_NO_PROXY"], "1")

    def test_stock_data_refresh_main_honors_no_proxy_environment_marker(self):
        with patch.dict(
            os.environ,
            {"stock_crawl_no_proxy": "true"},
            clear=True,
        ), patch.object(
            stock_data_refresh,
            "refresh_before_server",
            return_value={"ok": True},
        ) as refresh, patch.object(
            stock_data_refresh.sys,
            "argv",
            ["stock_data_refresh.py", "--mode", "quick"],
        ), patch("builtins.print"):
            stock_data_refresh.main()

        refresh.assert_called_once_with(
            mode="quick",
            timeout=None,
            no_proxy=True,
        )

    def test_resolve_python_keeps_the_active_interpreter_across_platforms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".venv" / "Scripts").mkdir(parents=True)
            (root / ".venv" / "Scripts" / "python.exe").touch()
            (root / ".venv" / "bin").mkdir(parents=True)
            (root / ".venv" / "bin" / "python").touch()
            active_python = r"C:\active-venv\Scripts\python.exe"

            with patch.object(stock_data_refresh, "ROOT", root), patch.dict(
                os.environ, {}, clear=True
            ), patch.object(
                stock_data_refresh.sys,
                "executable",
                active_python,
            ):
                self.assertEqual(resolve_python(), active_python)

    def test_stock_inner_step_inherits_web_job_group_without_local_timeout(self):
        process = Mock()
        process.wait.return_value = 0

        with patch.object(stock_data_refresh.subprocess, "Popen", return_value=process) as popen:
            result = stock_data_refresh.run_step(
                "inherit-group",
                ["python-test", "child.py"],
                timeout=None,
            )

        self.assertTrue(result.ok)
        self.assertNotIn("start_new_session", popen.call_args.kwargs)
        self.assertNotIn("creationflags", popen.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
