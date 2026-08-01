#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys


def workspace_versions() -> dict[str, str]:
    output = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    metadata = json.loads(output)
    return {
        package["name"]: package["version"]
        for package in metadata["packages"]
        if package["name"].startswith("nam-")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True)
    parser.add_argument("--tag")
    args = parser.parse_args()
    try:
        versions = workspace_versions()
        mismatches = {
            name: version for name, version in versions.items() if version != args.expected
        }
        if mismatches:
            raise ValueError(f"workspace version mismatch: {mismatches}")
        if args.tag and args.tag != f"v{args.expected}":
            raise ValueError(f"release tag {args.tag} does not match v{args.expected}")
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"release version check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
