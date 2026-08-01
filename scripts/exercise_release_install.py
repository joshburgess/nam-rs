#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

from install_release import install


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    try:
        install(args.previous, root=args.root)
        try:
            install(args.current, root=args.root)
        except ValueError as error:
            if "overwrite" not in str(error):
                raise
        else:
            raise ValueError("installing over an existing release did not require --overwrite")
        paths = install(args.current, root=args.root, overwrite=True)
        state = json.loads(paths.state.read_text(encoding="utf-8"))
        if state.get("version") != "0.2.0":
            raise ValueError("upgrade did not install v0.2.0")
        output = os.environ.get("GITHUB_OUTPUT")
        values = {
            "vst3": paths.vst3,
            "clap": paths.clap,
            "trainer": paths.trainer,
            "state": paths.state,
        }
        if output:
            with Path(output).open("a", encoding="utf-8") as destination:
                for name, value in values.items():
                    destination.write(f"{name}={value}\n")
        else:
            print(json.dumps({name: str(value) for name, value in values.items()}))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"release install exercise failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
