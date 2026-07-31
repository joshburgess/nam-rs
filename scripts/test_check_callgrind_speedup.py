import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from check_callgrind_speedup import check_speedup, read_instruction_count


class CallgrindSpeedupTests(unittest.TestCase):
    def test_reads_a_callgrind_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "callgrind.out"
            profile.write_text(
                "events: Ir\nsummary: 123456\n",
                encoding="utf-8",
            )

            self.assertEqual(read_instruction_count(profile), 123_456)

    def test_rejects_missing_and_duplicate_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "callgrind.out"
            profile.write_text("events: Ir\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "0 Callgrind instruction summaries"):
                read_instruction_count(profile)

            profile.write_text("summary: 1\nsummary: 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "2 Callgrind instruction summaries"):
                read_instruction_count(profile)

    def test_requires_a_ten_percent_instruction_speedup(self) -> None:
        with redirect_stdout(StringIO()):
            self.assertEqual(check_speedup(100, 90, 10.0), [])
            errors = check_speedup(100, 91, 10.0)

        self.assertEqual(len(errors), 1)
        self.assertIn("missed the instruction speedup gate", errors[0])


if __name__ == "__main__":
    unittest.main()
