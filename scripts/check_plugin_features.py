#!/usr/bin/env python3

import subprocess
import sys


def nam_core_features(extra_args: list[str]) -> set[str]:
    command = [
        "cargo",
        "tree",
        "--prefix",
        "none",
        "--format",
        "{p}|{f}",
        "-p",
        "nam-plugin",
        *extra_args,
    ]
    output = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    lines = [line for line in output.splitlines() if line.startswith("nam-core v")]
    if len(lines) != 1:
        raise ValueError(f"expected one nam-core package entry, found {len(lines)}")
    _, separator, features = lines[0].partition("|")
    if not separator:
        raise ValueError("nam-core package entry has no feature list")
    return {feature for feature in features.split(",") if feature}


def main() -> int:
    try:
        release_features = nam_core_features([])
        portable_features = nam_core_features(["--no-default-features"])
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"plugin feature check failed: {error}", file=sys.stderr)
        return 1

    errors = []
    if "fast-kernels" not in release_features:
        errors.append("the plugin default does not enable nam-core/fast-kernels")
    if "fast-kernels" in portable_features:
        errors.append("--no-default-features does not disable nam-core/fast-kernels")

    for error in errors:
        print(f"plugin feature check failed: {error}", file=sys.stderr)
    if not errors:
        print("plugin default enables fast-kernels; portable fallback disables it")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
