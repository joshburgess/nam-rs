#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from package_release import checked_archive_name, sha256, verify_archive


@dataclass(frozen=True)
class InstallPaths:
    vst3: Path
    clap: Path
    trainer: Path
    state: Path


def install_paths(platform: str, root: Path | None = None) -> InstallPaths:
    if root is not None:
        home = root / "home"
        local_app_data = root / "local-app-data"
    else:
        home = Path.home()
        local_app_data = Path(os.environ.get("LOCALAPPDATA", home / "AppData/Local"))

    if platform.startswith("macos-"):
        return InstallPaths(
            home / "Library/Audio/Plug-Ins/VST3/nam-plugin.vst3",
            home / "Library/Audio/Plug-Ins/CLAP/nam-plugin.clap",
            home / "Applications/nam-trainer",
            home / "Library/Application Support/nam-rs/install-state.json",
        )
    if platform.startswith("windows-"):
        common = local_app_data / "Programs/Common"
        return InstallPaths(
            common / "VST3/nam-plugin.vst3",
            common / "CLAP/nam-plugin.clap",
            local_app_data / "Programs/nam-rs/nam-trainer.exe",
            local_app_data / "nam-rs/install-state.json",
        )
    if platform.startswith("linux-"):
        return InstallPaths(
            home / ".vst3/nam-plugin.vst3",
            home / ".clap/nam-plugin.clap",
            home / ".local/bin/nam-trainer",
            home / ".local/share/nam-rs/install-state.json",
        )
    raise ValueError(f"unsupported release platform: {platform}")


def tree_hash(path: Path) -> str:
    if path.is_file():
        return sha256(path)
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(file)))
    return digest.hexdigest()


def copy_installed(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise ValueError(f"installation would overwrite {destination}; pass --overwrite to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.nam-rs-installing")
    if temporary.exists():
        if temporary.is_dir():
            shutil.rmtree(temporary)
        else:
            temporary.unlink()
    if source.is_dir():
        shutil.copytree(source, temporary)
    else:
        shutil.copy2(source, temporary)
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    temporary.replace(destination)


def extracted_release(archive: Path, destination: Path) -> tuple[Path, dict[str, object]]:
    metadata = verify_archive(archive)
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        roots = {checked_archive_name(member.filename).parts[0] for member in members}
        source.extractall(destination)
        if os.name != "nt":
            for member in members:
                mode = member.external_attr >> 16
                extracted = destination.joinpath(*checked_archive_name(member.filename).parts)
                if mode and extracted.exists():
                    extracted.chmod(mode & 0o777)
    if len(roots) != 1:
        raise ValueError("release archive must contain one root directory")
    return destination / roots.pop(), metadata


def install(archive: Path, root: Path | None = None, overwrite: bool = False) -> InstallPaths:
    with tempfile.TemporaryDirectory(prefix="nam-release-install-") as temporary:
        release_root, metadata = extracted_release(archive, Path(temporary))
        platform = metadata.get("platform")
        if not isinstance(platform, str):
            raise ValueError("release metadata is missing its platform")
        paths = install_paths(platform, root)
        sources = {
            paths.vst3: release_root / "plugins/nam-plugin.vst3",
            paths.clap: release_root / "plugins/nam-plugin.clap",
            paths.trainer: release_root
            / "bin"
            / ("nam-trainer.exe" if platform.startswith("windows-") else "nam-trainer"),
        }
        for destination, source in sources.items():
            copy_installed(source, destination, overwrite)
        state = {
            "schema_version": 1,
            "version": metadata.get("version"),
            "git_commit": metadata.get("git_commit"),
            "platform": platform,
            "package_sha256": sha256(archive),
            "installed": [
                {"path": str(path), "sha256": tree_hash(path)} for path in sources
            ],
        }
        paths.state.parent.mkdir(parents=True, exist_ok=True)
        temporary_state = paths.state.with_suffix(".tmp")
        temporary_state.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_state.replace(paths.state)
        return paths


def validate_install_state(
    state: object, paths: InstallPaths, platform: str
) -> list[tuple[Path, str]]:
    if not isinstance(state, dict):
        raise ValueError("installation state must be a JSON object")
    if state.get("schema_version") != 1:
        raise ValueError("installation state has an unsupported schema version")
    if state.get("platform") != platform:
        raise ValueError("installation state platform does not match the uninstall target")
    entries = state.get("installed")
    if not isinstance(entries, list):
        raise ValueError("installation state is missing its file manifest")

    expected_paths = {paths.vst3, paths.clap, paths.trainer}
    validated: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("installation state contains an invalid entry")
        raw_path = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(digest, str):
            raise ValueError("installation state contains an invalid entry")
        path = Path(raw_path)
        if path not in expected_paths:
            raise ValueError(f"installation state contains an unexpected path: {path}")
        if path in seen:
            raise ValueError(f"installation state contains a duplicate path: {path}")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"installation state contains an invalid checksum for {path}")
        seen.add(path)
        validated.append((path, digest))

    missing = expected_paths - seen
    if missing:
        missing_paths = ", ".join(str(path) for path in sorted(missing))
        raise ValueError(f"installation state is missing expected paths: {missing_paths}")
    return validated


def uninstall(platform: str, root: Path | None = None) -> InstallPaths:
    paths = install_paths(platform, root)
    if not paths.state.is_file():
        raise ValueError(f"installation state does not exist: {paths.state}")
    state = json.loads(paths.state.read_text(encoding="utf-8"))
    entries = validate_install_state(state, paths, platform)
    for path, digest in entries:
        if path.is_symlink():
            raise ValueError(f"refusing to remove symbolic link installation path: {path}")
        if path.exists() and tree_hash(path) != digest:
            raise ValueError(f"refusing to remove modified installation path: {path}")
    for path, _ in entries:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    paths.state.unlink()
    return paths


def report_paths(paths: InstallPaths, github_output: bool) -> None:
    values = {
        "vst3": str(paths.vst3),
        "clap": str(paths.clap),
        "trainer": str(paths.trainer),
        "state": str(paths.state),
    }
    if github_output:
        output = os.environ.get("GITHUB_OUTPUT")
        if not output:
            raise ValueError("GITHUB_OUTPUT is not set")
        with Path(output).open("a", encoding="utf-8") as destination:
            for name, value in values.items():
                destination.write(f"{name}={value}\n")
    else:
        print(json.dumps(values, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    install_parser = subcommands.add_parser("install")
    install_parser.add_argument("archive", type=Path)
    install_parser.add_argument("--root", type=Path)
    install_parser.add_argument("--overwrite", action="store_true")
    install_parser.add_argument("--github-output", action="store_true")
    uninstall_parser = subcommands.add_parser("uninstall")
    uninstall_parser.add_argument("--platform", required=True)
    uninstall_parser.add_argument("--root", type=Path)
    paths_parser = subcommands.add_parser("paths")
    paths_parser.add_argument("--platform", required=True)
    paths_parser.add_argument("--root", type=Path)
    paths_parser.add_argument("--github-output", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "install":
            paths = install(args.archive, args.root, args.overwrite)
            report_paths(paths, args.github_output)
        elif args.command == "uninstall":
            paths = uninstall(args.platform, args.root)
            print(paths.state)
        else:
            paths = install_paths(args.platform, args.root)
            report_paths(paths, args.github_output)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(f"release installation failed: {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
