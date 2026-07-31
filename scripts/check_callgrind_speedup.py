#!/usr/bin/env python3

import argparse
import math
import sys
from pathlib import Path


def read_instruction_count(profile_path: Path) -> int:
    summaries = []
    with profile_path.open(encoding="utf-8") as profile_file:
        for line in profile_file:
            if line.startswith("summary:"):
                try:
                    summaries.append(int(line.removeprefix("summary:").strip()))
                except ValueError as error:
                    raise ValueError(
                        f"{profile_path} has an invalid Callgrind summary"
                    ) from error

    if len(summaries) != 1:
        raise ValueError(
            f"{profile_path} has {len(summaries)} Callgrind instruction summaries"
        )
    if summaries[0] <= 0:
        raise ValueError(f"{profile_path} has a non-positive instruction count")
    return summaries[0]


def check_speedup(scalar: int, vector: int, minimum_percent: float) -> list[str]:
    speedup = (scalar - vector) * 100.0 / scalar
    print(
        f"scalar: {scalar:,} instructions, vector: {vector:,} instructions, "
        f"speedup: {speedup:.2f}%"
    )
    if speedup < minimum_percent:
        return [
            "portable vector callback missed the instruction speedup gate: "
            f"{speedup:.2f}% < {minimum_percent:.2f}%"
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare paired complete-callback Callgrind profiles"
    )
    parser.add_argument("scalar_profile", type=Path)
    parser.add_argument("vector_profile", type=Path)
    parser.add_argument("--minimum-percent", type=float, default=10.0)
    args = parser.parse_args()

    if not math.isfinite(args.minimum_percent) or args.minimum_percent < 0.0:
        parser.error("--minimum-percent must be a non-negative finite number")

    try:
        scalar = read_instruction_count(args.scalar_profile)
        vector = read_instruction_count(args.vector_profile)
        errors = check_speedup(scalar, vector, args.minimum_percent)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
