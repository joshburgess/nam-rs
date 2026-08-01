#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from package_release import sha256


def generate(directory: Path, output_name: str = "SHA256SUMS") -> Path:
    output = directory / output_name
    artifacts = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name != output_name
    )
    if not artifacts:
        raise ValueError(f"no release artifacts found in {directory}")
    output.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
    )
    return output


def verify(directory: Path, checksums: Path) -> None:
    seen = set()
    for line in checksums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or not name or Path(name).name != name:
            raise ValueError(f"invalid checksum line: {line}")
        artifact = directory / name
        if not artifact.is_file() or sha256(artifact) != digest:
            raise ValueError(f"checksum verification failed: {name}")
        seen.add(name)
    expected = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path != checksums
    }
    if seen != expected:
        raise ValueError("checksum manifest does not cover every release artifact")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    try:
        if args.verify:
            verify(args.directory, args.verify)
            print(f"verified {args.verify}")
        else:
            print(generate(args.directory))
    except (OSError, ValueError) as error:
        print(f"release checksum operation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
