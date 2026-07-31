import sys
import tempfile
import unittest
from pathlib import Path

from run_with_timeout import run_command


class TimeoutRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_captures_successful_output(self) -> None:
        log = self.root / "success.log"
        status = run_command([sys.executable, "-c", "print('validated')"], 5, log)

        self.assertEqual(status, 0)
        self.assertIn("validated", log.read_text(encoding="utf-8"))
        self.assertIn("exit code 0", log.read_text(encoding="utf-8"))

    def test_preserves_failure_status(self) -> None:
        log = self.root / "failure.log"
        status = run_command([sys.executable, "-c", "raise SystemExit(7)"], 5, log)

        self.assertEqual(status, 7)
        self.assertIn("exit code 7", log.read_text(encoding="utf-8"))

    def test_terminates_timed_out_command(self) -> None:
        log = self.root / "timeout.log"
        status = run_command([sys.executable, "-c", "import time; time.sleep(5)"], 0.1, log)

        self.assertEqual(status, 124)
        self.assertIn("timed out", log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
