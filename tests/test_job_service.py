import sys
import tempfile
import time
import unittest
from pathlib import Path

from app.services.job_service import get_job_state, start_command_job


def wait_for_job(job_id: str, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = get_job_state(job_id)
        if not state["running"]:
            return state
        time.sleep(0.02)
    raise AssertionError(f"job did not finish: {job_id}")


class JobServiceTest(unittest.TestCase):
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
