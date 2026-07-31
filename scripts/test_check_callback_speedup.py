import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from check_callback_speedup import Estimate, check_speedup, read_estimates


def write_estimate(root: Path, benchmark_id: str, estimate: Estimate) -> None:
    output = root / benchmark_id / "new"
    output.mkdir(parents=True)
    (output / "estimates.json").write_text(
        json.dumps(
            {
                "mean": {
                    "confidence_interval": {
                        "confidence_level": 0.95,
                        "lower_bound": estimate.lower,
                        "upper_bound": estimate.upper,
                    },
                    "point_estimate": estimate.point,
                }
            }
        ),
        encoding="utf-8",
    )


class CallbackSpeedupTests(unittest.TestCase):
    def check(
        self,
        scalar: dict[str, Estimate],
        vector: dict[str, Estimate],
        minimum_percent: float,
    ) -> list[str]:
        with redirect_stdout(StringIO()):
            return check_speedup(scalar, vector, minimum_percent)

    def test_reads_criterion_mean_estimates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            estimate = Estimate(lower=99.0, point=100.0, upper=101.0)
            write_estimate(root, "64", estimate)

            self.assertEqual(read_estimates(root), {"64": estimate})

    def test_requires_a_conservative_ten_percent_speedup(self) -> None:
        scalar = {
            benchmark_id: Estimate(lower=100.0, point=101.0, upper=102.0)
            for benchmark_id in ("16", "32", "64", "128", "256")
        }
        passing_vector = {
            benchmark_id: Estimate(lower=88.0, point=89.0, upper=90.0)
            for benchmark_id in scalar
        }
        failing_vector = dict(passing_vector)
        failing_vector["64"] = Estimate(lower=89.0, point=90.0, upper=91.0)

        self.assertEqual(self.check(scalar, passing_vector, 10.0), [])
        errors = self.check(scalar, failing_vector, 10.0)
        self.assertEqual(len(errors), 1)
        self.assertIn("64 samples missed", errors[0])

    def test_requires_the_exact_benchmark_set(self) -> None:
        estimate = Estimate(lower=80.0, point=85.0, upper=89.0)
        scalar = {benchmark_id: estimate for benchmark_id in ("16", "32", "64")}
        vector = dict(scalar)
        vector["other"] = estimate

        errors = self.check(scalar, vector, 10.0)

        self.assertIn("missing scalar benchmark measurement: 128", errors)
        self.assertIn("missing vector benchmark measurement: 256", errors)
        self.assertIn("unexpected vector benchmark measurement: other", errors)

    def test_rejects_an_invalid_confidence_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_estimate(root, "64", Estimate(lower=101.0, point=100.0, upper=102.0))

            with self.assertRaisesRegex(ValueError, "invalid confidence interval"):
                read_estimates(root)


if __name__ == "__main__":
    unittest.main()
