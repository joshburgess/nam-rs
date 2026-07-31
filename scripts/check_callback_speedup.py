#!/usr/bin/env python3

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


EXPECTED_BENCHMARKS = {"16", "32", "64", "128", "256"}


@dataclass(frozen=True)
class Estimate:
    lower: float
    point: float
    upper: float


def read_estimates(profile_root: Path) -> dict[str, Estimate]:
    estimates: dict[str, Estimate] = {}
    for estimates_path in profile_root.rglob("new/estimates.json"):
        benchmark_id = estimates_path.parent.parent.name
        if benchmark_id in estimates:
            raise ValueError(f"duplicate benchmark measurement: {benchmark_id}")

        with estimates_path.open(encoding="utf-8") as estimates_file:
            data = json.load(estimates_file)
        try:
            mean = data["mean"]
            confidence_interval = mean["confidence_interval"]
            confidence_level = float(confidence_interval["confidence_level"])
            estimate = Estimate(
                lower=float(confidence_interval["lower_bound"]),
                point=float(mean["point_estimate"]),
                upper=float(confidence_interval["upper_bound"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{estimates_path} has no valid mean estimate") from error

        if confidence_level != 0.95:
            raise ValueError(f"{estimates_path} does not use a 95% confidence interval")
        values = (estimate.lower, estimate.point, estimate.upper)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError(f"{estimates_path} has a non-positive or non-finite mean")
        if not estimate.lower <= estimate.point <= estimate.upper:
            raise ValueError(f"{estimates_path} has an invalid confidence interval")
        estimates[benchmark_id] = estimate

    if not estimates:
        raise ValueError(f"no Criterion estimates found below {profile_root}")
    return estimates


def check_speedup(
    scalar: dict[str, Estimate],
    vector: dict[str, Estimate],
    minimum_percent: float,
) -> list[str]:
    errors: list[str] = []
    for label, measurements in (("scalar", scalar), ("vector", vector)):
        for missing in sorted(EXPECTED_BENCHMARKS - measurements.keys(), key=int):
            errors.append(f"missing {label} benchmark measurement: {missing}")
        for unexpected in sorted(measurements.keys() - EXPECTED_BENCHMARKS):
            errors.append(f"unexpected {label} benchmark measurement: {unexpected}")

    for benchmark_id in sorted(
        EXPECTED_BENCHMARKS & scalar.keys() & vector.keys(), key=int
    ):
        scalar_estimate = scalar[benchmark_id]
        vector_estimate = vector[benchmark_id]
        point_speedup = (
            (scalar_estimate.point - vector_estimate.point)
            * 100.0
            / scalar_estimate.point
        )
        conservative_speedup = (
            (scalar_estimate.lower - vector_estimate.upper)
            * 100.0
            / scalar_estimate.lower
        )
        print(
            f"{benchmark_id} samples: {point_speedup:.2f}% point speedup, "
            f"{conservative_speedup:.2f}% conservative speedup"
        )
        if conservative_speedup < minimum_percent:
            errors.append(
                f"{benchmark_id} samples missed the callback speedup gate: "
                f"{conservative_speedup:.2f}% < {minimum_percent:.2f}%"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare paired Criterion callback profiles"
    )
    parser.add_argument("scalar_profile", type=Path)
    parser.add_argument("vector_profile", type=Path)
    parser.add_argument("--minimum-percent", type=float, default=10.0)
    args = parser.parse_args()

    if not math.isfinite(args.minimum_percent) or args.minimum_percent < 0.0:
        parser.error("--minimum-percent must be a non-negative finite number")

    try:
        scalar = read_estimates(args.scalar_profile)
        vector = read_estimates(args.vector_profile)
        errors = check_speedup(scalar, vector, args.minimum_percent)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 2

    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
