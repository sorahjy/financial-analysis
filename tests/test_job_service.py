import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.services.job_service import (
    _run_command,
    _job_environment,
    _popen_group_kwargs,
    get_job_state,
    start_command_job,
    terminate_process_tree,
)


def wait_for_job(job_id: str, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = get_job_state(job_id)
        if not state["running"]:
            return state
        time.sleep(0.02)
    raise AssertionError(f"job did not finish: {job_id}")


class JobServiceTest(unittest.TestCase):
    def test_child_environment_forces_utf8_on_windows(self):
        env = _job_environment({"Path": r"C:\Windows"})

        self.assertEqual(env["Path"], r"C:\Windows")
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(env["PYTHONUTF8"], "1")
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")

    def test_popen_group_options_are_platform_specific(self):
        windows = _popen_group_kwargs("nt")
        posix = _popen_group_kwargs("posix")

        self.assertIn("creationflags", windows)
        self.assertNotIn("start_new_session", windows)
        self.assertEqual(posix, {"start_new_session": True})

    def test_windows_timeout_uses_taskkill_for_the_whole_tree(self):
        process = Mock()
        process.pid = 4321
        process.poll.return_value = None
        process.wait.return_value = 0
        completed = subprocess.CompletedProcess([], 0)

        with patch("app.services.job_service.subprocess.run", return_value=completed) as run:
            terminate_process_tree(process, grace_seconds=0, platform_name="nt")

        self.assertEqual(
            run.call_args.args[0],
            ["taskkill", "/PID", "4321", "/T", "/F"],
        )
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_run_command_wires_utf8_environment_and_group_options_to_popen(self):
        process = Mock()
        process.stdout = None
        process.wait.return_value = 0
        expected_env = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
        }

        with patch(
            "app.services.job_service._job_environment",
            return_value=expected_env,
        ), patch(
            "app.services.job_service._popen_group_kwargs",
            return_value={"creationflags": 0x00000200},
        ), patch("app.services.job_service.subprocess.Popen", return_value=process) as popen:
            _run_command(
                "test-windows-popen-wiring",
                ["python-test", "child.py"],
                Path("."),
                1,
                None,
            )

        self.assertEqual(popen.call_args.kwargs["env"], expected_env)
        self.assertEqual(popen.call_args.kwargs["creationflags"], 0x00000200)
        self.assertNotIn("start_new_session", popen.call_args.kwargs)

    def test_windows_taskkill_failure_falls_back_to_direct_termination(self):
        process = Mock()
        process.pid = 9876
        process.poll.return_value = None
        process.wait.side_effect = [
            subprocess.TimeoutExpired("taskkill", 1),
            0,
        ]
        completed = subprocess.CompletedProcess([], 1)

        with patch("app.services.job_service.subprocess.run", return_value=completed):
            terminate_process_tree(process, grace_seconds=0, platform_name="nt")

        process.terminate.assert_called_once_with()
        process.kill.assert_not_called()

    def test_timeout_terminates_descendant_processes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            marker = Path(tmpdir) / "orphan-finished"
            child = f"import time; time.sleep(.5); open({str(marker)!r}, 'w').write('bad')"
            parent = (
                "import subprocess, sys, time; "
                f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
                "time.sleep(5)"
            )
            job_id = "test-process-tree-timeout"
            self.assertTrue(start_command_job(job_id, [sys.executable, "-c", parent], timeout=0.1))
            state = wait_for_job(job_id)
            time.sleep(0.6)

            self.assertFalse(state["ok"])
            self.assertIn("任务超时", state["error"])
            self.assertFalse(marker.exists())

    def test_resource_key_blocks_same_underlying_refresh(self):
        first = "test-resource-lock-first"
        second = "test-resource-lock-second"
        command = [sys.executable, "-c", "import time; time.sleep(.25)"]
        self.assertTrue(start_command_job(first, command, timeout=2, resource_key="shared-test-resource"))
        self.assertFalse(start_command_job(second, command, timeout=2, resource_key="shared-test-resource"))
        self.assertTrue(wait_for_job(first)["ok"])
        self.assertTrue(start_command_job(second, command, timeout=2, resource_key="shared-test-resource"))
        self.assertTrue(wait_for_job(second)["ok"])


if __name__ == "__main__":
    unittest.main()
