import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from check_instruction_budget import check_budget, read_instruction_counts


def summary(benchmark_id: str, instructions: int) -> dict:
    return {
        "id": benchmark_id,
        "callgrind_summary": {
            "callgrind_run": {
                "total": {
                    "summary": {
                        "Ir": {
                            "metrics": {"Left": instructions},
                        }
                    }
                }
            }
        },
    }


class InstructionBudgetTests(unittest.TestCase):
    def test_reads_instruction_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for benchmark_id, instructions in (("a1", 100), ("a2", 200)):
                output = root / benchmark_id
                output.mkdir()
                (output / "summary.json").write_text(
                    json.dumps(summary(benchmark_id, instructions)),
                    encoding="utf-8",
                )

            self.assertEqual(read_instruction_counts(root), {"a1": 100, "a2": 200})

    def test_rejects_regression_above_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.json"
            baseline_path.write_text(
                json.dumps(
                    {
                        "tolerance_percent": 3.0,
                        "configurations": {"default": {"a1": 100}},
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(check_budget("default", {"a1": 103}, baseline_path), [])
            errors = check_budget("default", {"a1": 104}, baseline_path)
            self.assertEqual(len(errors), 1)
            self.assertIn("exceeded its instruction budget", errors[0])

    def test_requires_exact_benchmark_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.json"
            baseline_path.write_text(
                json.dumps(
                    {
                        "tolerance_percent": 3.0,
                        "configurations": {"default": {"a1": 100, "a2": 200}},
                    }
                ),
                encoding="utf-8",
            )

            errors = check_budget("default", {"a1": 100, "extra": 1}, baseline_path)
            self.assertIn("missing benchmark measurement: a2", errors)
            self.assertIn("unexpected benchmark measurement: extra", errors)

    def test_rejects_a_different_baseline_platform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.json"
            baseline_path.write_text(
                json.dumps(
                    {
                        "platform": {"system": "Linux", "machine": "x86_64"},
                        "tolerance_percent": 3.0,
                        "configurations": {"default": {"a1": 100}},
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("check_instruction_budget.platform.system", return_value="Darwin"),
                patch("check_instruction_budget.platform.machine", return_value="arm64"),
            ):
                errors = check_budget("default", {"a1": 100}, baseline_path)
            self.assertEqual(
                errors,
                [
                    "instruction baseline platform mismatch: "
                    "expected Linux/x86_64, found Darwin/arm64"
                ],
            )

    def test_enforces_scaling_envelopes_and_absolute_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.json"
            baseline_path.write_text(
                json.dumps(
                    {
                        "tolerance_percent": 3.0,
                        "configurations": {"default": {"a1_64": 100}},
                        "scaling_envelopes": [
                            {
                                "reference": "a1_64",
                                "reference_samples": 64,
                                "benchmark": "a1_256",
                                "benchmark_samples": 256,
                                "max_relative_instructions_per_sample": 1.1,
                            }
                        ],
                        "absolute_limits": {"lstm_64": 500},
                    }
                ),
                encoding="utf-8",
            )

            counts = {"a1_64": 100, "a1_256": 441, "lstm_64": 501}
            errors = check_budget("default", counts, baseline_path)
            self.assertEqual(len(errors), 2)
            self.assertTrue(any("scaling envelope" in error for error in errors))
            self.assertTrue(any("absolute instruction limit" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
