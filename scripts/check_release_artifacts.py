#!/usr/bin/env python3

import argparse
import os
import platform
import re
import struct
import subprocess
import sys
from pathlib import Path


def binary_format(data: bytes) -> str | None:
    if data.startswith(b"\x7fELF"):
        return "elf"
    if data.startswith(b"MZ"):
        return "pe"
    if data.startswith(b"\xcf\xfa\xed\xfe"):
        return "mach-o"
    return None


def read_header(path: Path) -> bytes:
    with path.open("rb") as binary:
        return binary.read(4096)


def validate_architecture(path: Path, expected_format: str) -> None:
    data = read_header(path)
    actual_format = binary_format(data)
    if actual_format != expected_format:
        raise ValueError(f"{path} is {actual_format or 'not a recognized binary'}, expected {expected_format}")

    if actual_format == "elf":
        machine = struct.unpack_from("<H", data, 18)[0]
        if machine != 62:
            raise ValueError(f"{path} has ELF machine {machine}, expected x86-64")
    elif actual_format == "pe":
        pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe_offset : pe_offset + 4] != b"PE\0\0":
            raise ValueError(f"{path} has an invalid PE header")
        machine = struct.unpack_from("<H", data, pe_offset + 4)[0]
        if machine != 0x8664:
            raise ValueError(f"{path} has PE machine 0x{machine:04x}, expected x86-64")
    else:
        cpu_type = struct.unpack_from("<I", data, 4)[0]
        if cpu_type != 0x0100000C:
            raise ValueError(f"{path} has Mach-O CPU type 0x{cpu_type:08x}, expected arm64")


def validate_build_metadata(
    path: Path, version: str, commit: str, target: str, features: str
) -> None:
    data = path.read_bytes()
    expected = (version, commit[:12], target, "release", features)
    missing = [value for value in expected if value.encode() not in data]
    if missing:
        raise ValueError(f"{path} is missing embedded build metadata: {', '.join(missing)}")


def bundled_binaries(bundle_root: Path, expected_format: str) -> list[Path]:
    binaries = []
    for path in bundle_root.rglob("*"):
        if path.is_file() and binary_format(read_header(path)) == expected_format:
            binaries.append(path)
    if len(binaries) != 2:
        raise ValueError(f"expected two plugin binaries below {bundle_root}, found {len(binaries)}")
    return sorted(binaries)


def validate_linux_dependencies(path: Path) -> None:
    notes = subprocess.run(
        ["readelf", "--notes", path], check=True, capture_output=True, text=True
    ).stdout
    if re.search(r"x86-64-v[234].*needed", notes, re.IGNORECASE):
        raise ValueError(f"{path} requires a non-baseline x86-64 ISA")

    output = subprocess.run(
        ["ldd", path], check=True, capture_output=True, text=True
    ).stdout
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("linux-vdso.so"):
            continue
        if "not found" in line:
            raise ValueError(f"{path} has an unresolved dependency: {line}")
        resolved = line.split("=>", 1)[1].strip().split(" ", 1)[0] if "=>" in line else line.split(" ", 1)[0]
        if not resolved.startswith(("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/")):
            raise ValueError(f"{path} has a non-system dependency: {line}")


def validate_macos_dependencies(path: Path) -> None:
    output = subprocess.run(
        ["otool", "-L", path], check=True, capture_output=True, text=True
    ).stdout
    for line in output.splitlines()[2:]:
        dependency = line.strip().split(" ", 1)[0]
        if not dependency.startswith(("/System/Library/", "/usr/lib/")):
            raise ValueError(f"{path} has a non-system dependency: {dependency}")


def find_dumpbin() -> Path:
    installer = (
        Path(os.environ["ProgramFiles(x86)"])
        / "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    output = subprocess.run(
        [
            installer,
            "-latest",
            "-products",
            "*",
            "-find",
            r"VC\Tools\MSVC\**\bin\Hostx64\x64\dumpbin.exe",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    paths = [Path(line.strip()) for line in output.splitlines() if line.strip()]
    if not paths:
        raise ValueError("dumpbin.exe was not found")
    return paths[-1]


def validate_windows_dependencies(path: Path, dumpbin: Path) -> None:
    output = subprocess.run(
        [dumpbin, "/dependents", path], check=True, capture_output=True, text=True
    ).stdout
    dependencies = sorted(set(re.findall(r"(?im)^\s*([^\\/:\s]+\.dll)\s*$", output)))
    if not dependencies:
        raise ValueError(f"{path} has no reported PE dependencies")

    system32 = Path(os.environ["SystemRoot"]) / "System32"
    for dependency in dependencies:
        lower = dependency.lower()
        if lower.startswith(("api-ms-win-", "ext-ms-")):
            continue
        if not (system32 / dependency).is_file():
            raise ValueError(f"{path} has a dependency outside System32: {dependency}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--commit")
    parser.add_argument("--target")
    parser.add_argument("--features")
    args = parser.parse_args()

    metadata_values = (args.version, args.commit, args.target, args.features)
    if any(metadata_values) and not all(metadata_values):
        parser.error("--version, --commit, --target, and --features must be used together")

    system = platform.system()
    expected_formats = {"Linux": "elf", "Darwin": "mach-o", "Windows": "pe"}
    expected_format = expected_formats.get(system)
    if expected_format is None:
        parser.error(f"unsupported platform: {system}")

    try:
        binaries = bundled_binaries(args.bundle_root, expected_format)
        dumpbin = find_dumpbin() if system == "Windows" else None
        for path in binaries:
            validate_architecture(path, expected_format)
            if all(metadata_values):
                validate_build_metadata(
                    path, args.version, args.commit, args.target, args.features
                )
            if system == "Linux":
                validate_linux_dependencies(path)
            elif system == "Darwin":
                validate_macos_dependencies(path)
            elif dumpbin is not None:
                validate_windows_dependencies(path, dumpbin)
            print(f"validated {path}")
    except (KeyError, OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"release artifact check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
