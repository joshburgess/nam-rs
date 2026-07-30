#!/usr/bin/env python3

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("upstream_compatibility.json")


@dataclass(frozen=True)
class Comparison:
    sample_count: int
    differing_samples: int
    max_abs_error: float
    rms_error: float


def read_float_wav(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF WAVE file")

    position = 12
    format_code = None
    sample_data = None
    while position + 8 <= len(data):
        chunk_id = data[position : position + 4]
        chunk_size = struct.unpack_from("<I", data, position + 4)[0]
        chunk_start = position + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(data):
            raise ValueError(f"{path} has a truncated {chunk_id!r} chunk")
        chunk = data[chunk_start:chunk_end]
        if chunk_id == b"fmt ":
            if len(chunk) < 16:
                raise ValueError(f"{path} has a truncated format chunk")
            format_code = struct.unpack_from("<H", chunk)[0]
            if format_code == 0xFFFE and len(chunk) >= 26:
                format_code = struct.unpack_from("<H", chunk, 24)[0]
        elif chunk_id == b"data":
            sample_data = chunk
        position = chunk_end + (chunk_size & 1)

    if format_code != 3:
        raise ValueError(f"{path} is not a 32-bit float WAVE file")
    if sample_data is None or len(sample_data) % 4:
        raise ValueError(f"{path} has invalid float sample data")
    return list(struct.unpack(f"<{len(sample_data) // 4}f", sample_data))


def compare_samples(actual: list[float], expected: list[float]) -> Comparison:
    if len(actual) != len(expected):
        raise ValueError(
            f"sample-count mismatch: rendered {len(actual)}, reference {len(expected)}"
        )
    differing = 0
    max_error = 0.0
    squared_error = 0.0
    for actual_sample, expected_sample in zip(actual, expected):
        if struct.pack("<f", actual_sample) != struct.pack("<f", expected_sample):
            differing += 1
        error = abs(actual_sample - expected_sample)
        max_error = max(max_error, error)
        squared_error += error * error
    rms_error = math.sqrt(squared_error / len(actual)) if actual else 0.0
    return Comparison(len(actual), differing, max_error, rms_error)


def git_head(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not manifest.get("fixtures"):
        raise ValueError(f"{path} contains no compatibility fixtures")
    return manifest


def fixture_failure(fixture: dict, comparison: Comparison) -> str | None:
    name = fixture["name"]
    if fixture.get("require_bit_exact") and comparison.differing_samples:
        return (
            f"{name} is not bit-exact "
            f"({comparison.differing_samples} samples differ)"
        )
    if "max_abs_error" in fixture:
        tolerance = float(fixture["max_abs_error"])
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"{name} has invalid max_abs_error {tolerance}")
        if comparison.max_abs_error > tolerance:
            return (
                f"{name} exceeds max_abs_error {tolerance:.9g} "
                f"({comparison.max_abs_error:.9g})"
            )
    return None


def audit(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    expected_commit = manifest["neural_amp_modeler_core"]["commit"]
    actual_commit = git_head(args.core)
    if actual_commit != expected_commit:
        raise ValueError(
            f"Core checkout is {actual_commit}; expected pinned commit {expected_commit}"
        )

    renderer = args.renderer or args.core / "build-parity" / "tools" / "render"
    if not renderer.is_file():
        raise ValueError(f"upstream renderer not found: {renderer}")

    failures = []
    with tempfile.TemporaryDirectory(prefix="nam-rs-upstream-audit-") as temp_dir:
        temp_root = Path(temp_dir)
        for fixture in manifest["fixtures"]:
            name = fixture["name"]
            model = REPO_ROOT / fixture["model"]
            input_path = REPO_ROOT / fixture["input"]
            reference = REPO_ROOT / fixture["reference"]
            rendered = temp_root / f"{name}.wav"
            subprocess.run(
                [str(renderer), str(model), str(input_path), str(rendered)],
                check=True,
            )

            if args.update:
                shutil.copyfile(rendered, reference)
                print(f"{name}: updated {reference.relative_to(REPO_ROOT)}")
                continue

            comparison = compare_samples(
                read_float_wav(rendered),
                read_float_wav(reference),
            )
            print(
                f"{name}: {comparison.sample_count} samples, "
                f"{comparison.differing_samples} bit differences, "
                f"max {comparison.max_abs_error:.9g}, "
                f"RMS {comparison.rms_error:.9g}"
            )
            failure = fixture_failure(fixture, comparison)
            if failure is not None:
                failures.append(failure)

    if failures:
        for failure in failures:
            print(f"compatibility audit failed: {failure}", file=sys.stderr)
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render pinned upstream fixtures and compare their float samples."
    )
    parser.add_argument(
        "--core",
        type=Path,
        required=True,
        help="NeuralAmpModelerCore checkout at the manifest's pinned commit",
    )
    parser.add_argument(
        "--renderer",
        type=Path,
        help="renderer executable (default: CORE/build-parity/tools/render)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--update",
        action="store_true",
        help="replace checked-in references with the pinned renderer output",
    )
    return parser.parse_args()


def main() -> int:
    try:
        return audit(parse_args())
    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"compatibility audit failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
