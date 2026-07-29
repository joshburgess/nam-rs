#!/usr/bin/env python3

import argparse
import json
import platform
import sys
from pathlib import Path


def read_instruction_counts(summary_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for summary_path in summary_root.rglob("summary.json"):
        with summary_path.open(encoding="utf-8") as summary_file:
            summary = json.load(summary_file)
        benchmark_id = summary.get("id")
        if not benchmark_id:
            continue
        try:
            metrics = summary["callgrind_summary"]["callgrind_run"]["total"]["summary"]["Ir"][
                "metrics"
            ]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"{summary_path} has no total Callgrind instruction count"
            ) from error

        if "Left" in metrics:
            instruction_count = metrics["Left"]
        elif "Both" in metrics:
            instruction_count = metrics["Both"][0]
        else:
            raise ValueError(f"{summary_path} has no new Callgrind instruction count")

        if benchmark_id in counts:
            raise ValueError(f"duplicate benchmark id in summaries: {benchmark_id}")
        counts[benchmark_id] = instruction_count

    if not counts:
        raise ValueError(f"no benchmark summaries found below {summary_root}")
    return counts


def check_budget(
    configuration: str,
    counts: dict[str, int],
    baseline_path: Path,
) -> list[str]:
    with baseline_path.open(encoding="utf-8") as baseline_file:
        baseline = json.load(baseline_file)

    expected_platform = baseline.get("platform")
    if expected_platform is not None:
        actual_system = platform.system()
        actual_machine = platform.machine()
        expected_system = expected_platform["system"]
        expected_machine = expected_platform["machine"]
        if actual_system != expected_system or actual_machine != expected_machine:
            return [
                "instruction baseline platform mismatch: "
                f"expected {expected_system}/{expected_machine}, "
                f"found {actual_system}/{actual_machine}"
            ]

    tolerance_percent = baseline["tolerance_percent"]
    expected = baseline["configurations"].get(configuration)
    if expected is None:
        return [f"missing baseline configuration: {configuration}"]
    envelopes = baseline.get("scaling_envelopes", [])
    absolute_limits = baseline.get("absolute_limits", {})

    errors: list[str] = []
    envelope_ids = {
        benchmark_id
        for envelope in envelopes
        for benchmark_id in (envelope["reference"], envelope["benchmark"])
    }
    expected_ids = set(expected) | envelope_ids | set(absolute_limits)
    measured_ids = set(counts)
    for missing in sorted(expected_ids - measured_ids):
        errors.append(f"missing benchmark measurement: {missing}")
    for unexpected in sorted(measured_ids - expected_ids):
        errors.append(f"unexpected benchmark measurement: {unexpected}")

    for benchmark_id in sorted(set(expected) & measured_ids):
        baseline_count = expected[benchmark_id]
        measured_count = counts[benchmark_id]
        limit = baseline_count * (1.0 + tolerance_percent / 100.0)
        change_percent = (measured_count - baseline_count) * 100.0 / baseline_count
        print(
            f"{configuration}/{benchmark_id}: {measured_count:,} instructions "
            f"({change_percent:+.2f}%, limit {limit:,.0f})"
        )
        if measured_count > limit:
            errors.append(
                f"{configuration}/{benchmark_id} exceeded its instruction budget: "
                f"{measured_count:,} > {limit:,.0f}"
            )

    for envelope in envelopes:
        reference_id = envelope["reference"]
        benchmark_id = envelope["benchmark"]
        if reference_id not in counts or benchmark_id not in counts:
            continue
        reference_samples = envelope["reference_samples"]
        benchmark_samples = envelope["benchmark_samples"]
        per_sample_ratio = envelope["max_relative_instructions_per_sample"]
        reference_count = counts[reference_id]
        measured_count = counts[benchmark_id]
        limit = (
            reference_count
            * benchmark_samples
            / reference_samples
            * per_sample_ratio
        )
        print(
            f"{configuration}/{benchmark_id}: {measured_count:,} instructions "
            f"(scaling limit {limit:,.0f} from {reference_id})"
        )
        if measured_count > limit:
            errors.append(
                f"{configuration}/{benchmark_id} exceeded its scaling envelope: "
                f"{measured_count:,} > {limit:,.0f}"
            )

    for benchmark_id, limit in absolute_limits.items():
        measured_count = counts.get(benchmark_id)
        if measured_count is None:
            continue
        print(
            f"{configuration}/{benchmark_id}: {measured_count:,} instructions "
            f"(absolute limit {limit:,})"
        )
        if measured_count > limit:
            errors.append(
                f"{configuration}/{benchmark_id} exceeded its absolute instruction limit: "
                f"{measured_count:,} > {limit:,}"
            )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("configuration")
    parser.add_argument("summary_root", type=Path)
    parser.add_argument("baseline", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        counts = read_instruction_counts(args.summary_root)
        errors = check_budget(args.configuration, counts, args.baseline)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"instruction budget check failed: {error}", file=sys.stderr)
        return 1

    if errors:
        for error in errors:
            print(f"instruction budget check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
