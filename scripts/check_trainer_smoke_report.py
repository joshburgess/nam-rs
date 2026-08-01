#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    try:
        value = json.loads(args.report.read_text(encoding="utf-8"))
        expected = {
            "version": args.version,
            "git_commit": args.commit,
            "target": args.target,
            "profile": "release",
        }
        for name, expected_value in expected.items():
            if value.get(name) != expected_value:
                raise ValueError(
                    f"smoke report {name} is {value.get(name)!r}, expected {expected_value!r}"
                )
        features = value.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError("smoke report has no build features")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"trainer smoke report check failed: {error}", file=sys.stderr)
        return 1
    print(f"verified {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
