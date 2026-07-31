#!/usr/bin/env python3

import argparse
import hashlib
import json
import shutil
import stat
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO


def checked_relative_path(name: str) -> Path:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"archive member escapes its destination: {name}")
    return Path(*path.parts)


def copy_stream(source: BinaryIO, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        shutil.copyfileobj(source, output)


def extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            path = destination / checked_relative_path(member.filename)
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            with package.open(member) as source:
                copy_stream(source, path)


def extract_tar_gz(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as package:
        for member in package.getmembers():
            path = destination / checked_relative_path(member.name)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unsupported archive member: {member.name}")
            source = package.extractfile(member)
            if source is None:
                raise ValueError(f"could not read archive member: {member.name}")
            with source:
                copy_stream(source, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "nam-rs-ci"})
    with urllib.request.urlopen(request, timeout=60) as source:
        copy_stream(source, destination)


def install_validator(name: str, settings: dict[str, str], destination: Path) -> Path:
    tool_root = destination / name
    tool_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{name}-", dir=destination) as temporary:
        archive = Path(temporary) / "validator.zip"
        download(settings["url"], archive)
        actual_digest = sha256(archive)
        if actual_digest != settings["sha256"]:
            raise ValueError(
                f"{name} archive has SHA-256 {actual_digest}, expected {settings['sha256']}"
            )
        extract_zip(archive, tool_root)

    for nested_archive in tool_root.glob("*.tar.gz"):
        extract_tar_gz(nested_archive, tool_root)
        nested_archive.unlink()

    executable = tool_root / settings["executable"]
    if not executable.is_file():
        raise ValueError(f"{name} executable was not found at {executable}")
    executable.chmod(
        executable.stat().st_mode
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH
    )
    return executable


def install_validators(manifest_path: Path, platform: str, destination: Path) -> dict[str, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    installed = {}
    for name, validator in manifest.items():
        try:
            settings = validator["platforms"][platform]
        except KeyError as error:
            raise ValueError(f"no {name} archive is configured for {platform}") from error
        installed[name] = install_validator(name, settings, destination)
    return installed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("linux", "macos", "windows"), required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("plugin_validators.json"),
    )
    args = parser.parse_args()

    try:
        installed = install_validators(args.manifest, args.platform, args.destination)
    except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as error:
        print(f"validator installation failed: {error}")
        return 1

    for name, executable in installed.items():
        print(f"installed {name}: {executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
