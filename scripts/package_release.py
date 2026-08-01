#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 1
ZIP_EPOCH = 315532800


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packaged_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"release packages cannot contain symbolic links: {path}")
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def release_metadata(
    root: Path,
    version: str,
    commit: str,
    platform: str,
    target: str,
    features: list[str],
) -> dict[str, object]:
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "size": path.stat().st_size,
        }
        for path in packaged_files(root)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "version": version,
        "git_commit": commit,
        "platform": platform,
        "target": target,
        "features": features,
        "publisher_signing": "none",
        "notarized": False,
        "files": files,
    }


def zip_timestamp(source_date_epoch: int) -> tuple[int, int, int, int, int, int]:
    timestamp = max(source_date_epoch, ZIP_EPOCH)
    value = time.gmtime(timestamp)[:6]
    return value[0], value[1], value[2], value[3], value[4], value[5]


def write_zip(source_root: Path, destination: Path, source_date_epoch: int) -> None:
    timestamp = zip_timestamp(source_date_epoch)
    archive_root = source_root.name
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        directories = [
            source_root,
            *sorted(path for path in source_root.rglob("*") if path.is_dir()),
        ]
        for directory in directories:
            relative = directory.relative_to(source_root).as_posix()
            name = f"{archive_root}/{relative}/" if relative != "." else f"{archive_root}/"
            info = zipfile.ZipInfo(name, timestamp)
            info.create_system = 3
            info.external_attr = (stat.S_IFDIR | 0o755) << 16
            archive.writestr(info, b"")
        for path in packaged_files(source_root):
            relative = path.relative_to(source_root).as_posix()
            info = zipfile.ZipInfo(f"{archive_root}/{relative}", timestamp)
            info.create_system = 3
            mode = 0o755 if os.access(path, os.X_OK) else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            with path.open("rb") as source:
                archive.writestr(
                    info,
                    source.read(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )


def checked_archive_name(name: str) -> PurePosixPath:
    if "\\" in name or "\0" in name:
        raise ValueError(f"unsafe archive member: {name}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive member: {name}")
    return path


def verify_archive(path: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="nam-release-verify-") as temporary:
        destination = Path(temporary)
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [checked_archive_name(member.filename) for member in members]
            if len(names) != len(set(names)):
                raise ValueError("release archive contains duplicate members")
            roots = {name.parts[0] for name in names}
            if len(roots) != 1:
                raise ValueError("release archive must contain one root directory")
            archive.extractall(destination)
        root = destination / roots.pop()
        metadata_path = root / "BUILD-METADATA.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported release metadata schema")
        entries = metadata.get("files")
        if not isinstance(entries, list):
            raise ValueError("release metadata is missing its file manifest")
        expected = set()
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError("release metadata contains an invalid file entry")
            relative = checked_archive_name(entry["path"])
            if relative.as_posix() in expected:
                raise ValueError(f"release metadata contains a duplicate entry: {relative}")
            source = root.joinpath(*relative.parts)
            if not source.is_file() or sha256(source) != entry.get("sha256"):
                raise ValueError(f"release file failed checksum validation: {relative}")
            expected.add(relative.as_posix())
        actual = {
            file.relative_to(root).as_posix()
            for file in packaged_files(root)
            if file.name != "BUILD-METADATA.json"
        }
        if actual != expected:
            raise ValueError("release metadata does not cover every packaged file")
        return metadata


def package_release(
    plugin_root: Path,
    trainer: Path,
    output_dir: Path,
    version: str,
    commit: str,
    platform: str,
    target: str,
    features: list[str],
    source_date_epoch: int,
) -> Path:
    if not plugin_root.is_dir():
        raise ValueError(f"plugin bundle directory does not exist: {plugin_root}")
    if not trainer.is_file():
        raise ValueError(f"trainer executable does not exist: {trainer}")
    for name in ("nam-plugin.vst3", "nam-plugin.clap"):
        if not (plugin_root / name).exists():
            raise ValueError(f"missing plugin bundle: {plugin_root / name}")

    archive_name = f"nam-rs-v{version}-{platform}.zip"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / archive_name
    with tempfile.TemporaryDirectory(prefix="nam-release-package-") as temporary:
        root = Path(temporary) / destination.stem
        plugins = root / "plugins"
        binaries = root / "bin"
        plugins.mkdir(parents=True)
        binaries.mkdir()
        for name in ("nam-plugin.vst3", "nam-plugin.clap"):
            source = plugin_root / name
            target_path = plugins / name
            if source.is_dir():
                shutil.copytree(source, target_path)
            else:
                shutil.copy2(source, target_path)
        trainer_name = "nam-trainer.exe" if platform.startswith("windows-") else "nam-trainer"
        shutil.copy2(trainer, binaries / trainer_name)
        shutil.copy2("LICENSE", root / "LICENSE")
        shutil.copy2("docs/installing-releases.md", root / "INSTALL.md")
        metadata = release_metadata(root, version, commit, platform, target, features)
        (root / "BUILD-METADATA.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_zip(root, destination, source_date_epoch)
    verify_archive(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-root", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--features", default="fast-kernels")
    parser.add_argument(
        "--source-date-epoch",
        type=int,
        default=int(os.environ.get("SOURCE_DATE_EPOCH", ZIP_EPOCH)),
    )
    args = parser.parse_args()
    try:
        archive = package_release(
            args.plugin_root,
            args.trainer,
            args.output_dir,
            args.version,
            args.commit,
            args.platform,
            args.target,
            [feature for feature in args.features.split(",") if feature],
            args.source_date_epoch,
        )
        print(archive)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(f"release packaging failed: {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
