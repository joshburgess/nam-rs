import importlib.util
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path


def load_module(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    os.sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package_release = load_module("package_release")
install_release = load_module("install_release")
generate_release_checksums = load_module("generate_release_checksums")


class ReleasePackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plugin_root = self.root / "bundled"
        (self.plugin_root / "nam-plugin.vst3/Contents/x86_64-linux").mkdir(parents=True)
        (self.plugin_root / "nam-plugin.vst3/Contents/x86_64-linux/nam-plugin.so").write_bytes(b"vst3")
        (self.plugin_root / "nam-plugin.clap").write_bytes(b"clap")
        self.trainer = self.root / "nam-trainer"
        self.trainer.write_bytes(b"trainer")
        self.trainer.chmod(0o755)
        self.output = self.root / "release"

    def tearDown(self):
        self.temporary.cleanup()

    def package(self, version="0.2.0", commit="a" * 40):
        return package_release.package_release(
            self.plugin_root,
            self.trainer,
            self.output,
            version,
            commit,
            "linux-x86_64",
            "x86_64-unknown-linux-gnu",
            ["fast-kernels"],
            package_release.ZIP_EPOCH,
        )

    def test_archive_is_deterministic_and_manifest_covers_every_file(self):
        first = self.package()
        first_bytes = first.read_bytes()
        second = self.package()
        self.assertEqual(first_bytes, second.read_bytes())
        metadata = package_release.verify_archive(second)
        self.assertEqual(metadata["version"], "0.2.0")
        self.assertEqual(metadata["publisher_signing"], "none")
        self.assertIs(metadata["notarized"], False)
        self.assertEqual(metadata["features"], ["fast-kernels"])
        self.assertEqual(len(metadata["files"]), 5)

    def test_archive_tampering_is_rejected(self):
        archive = self.package()
        tampered = self.root / "tampered.zip"
        with zipfile.ZipFile(archive) as source, zipfile.ZipFile(tampered, "w") as destination:
            for member in source.infolist():
                contents = source.read(member)
                if member.filename.endswith("nam-plugin.clap"):
                    contents = b"tampered"
                destination.writestr(member, contents)
        with self.assertRaisesRegex(ValueError, "checksum"):
            package_release.verify_archive(tampered)

    def test_archive_duplicate_members_are_rejected(self):
        archive = self.package()
        with zipfile.ZipFile(archive, "a") as destination:
            destination.writestr(f"{archive.stem}/LICENSE", b"duplicate")
        with self.assertRaisesRegex(ValueError, "duplicate members"):
            package_release.verify_archive(archive)

    def test_archive_backslash_paths_are_rejected(self):
        archive = self.package()
        with zipfile.ZipFile(archive, "a") as destination:
            destination.writestr("..\\outside", b"unsafe")
        with self.assertRaisesRegex(ValueError, "unsafe archive member"):
            package_release.verify_archive(archive)

    def test_install_overwrite_upgrade_and_uninstall(self):
        old_archive = self.package(version="0.1.0", commit="b" * 40)
        install_root = self.root / "installed"
        old_paths = install_release.install(old_archive, install_root)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            install_release.install(old_archive, install_root)

        new_archive = self.package(version="0.2.0", commit="c" * 40)
        paths = install_release.install(new_archive, install_root, overwrite=True)
        state = json.loads(paths.state.read_text(encoding="utf-8"))
        self.assertEqual(state["version"], "0.2.0")
        self.assertEqual(state["git_commit"], "c" * 40)
        self.assertTrue(paths.vst3.exists())
        self.assertTrue(paths.clap.exists())
        self.assertTrue(paths.trainer.exists())

        install_release.uninstall("linux-x86_64", install_root)
        self.assertFalse(paths.vst3.exists())
        self.assertFalse(paths.clap.exists())
        self.assertFalse(paths.trainer.exists())
        self.assertFalse(paths.state.exists())
        self.assertEqual(old_paths, paths)

    def test_uninstall_refuses_to_remove_modified_files(self):
        archive = self.package()
        install_root = self.root / "installed"
        paths = install_release.install(archive, install_root)
        paths.clap.write_bytes(b"locally modified")
        with self.assertRaisesRegex(ValueError, "modified"):
            install_release.uninstall("linux-x86_64", install_root)
        self.assertTrue(paths.clap.exists())
        self.assertTrue(paths.state.exists())

    def test_checksum_manifest_covers_all_release_artifacts(self):
        archive = self.package()
        sbom = self.output / "nam-rs.spdx.json"
        sbom.write_text("{}\n", encoding="utf-8")
        checksums = generate_release_checksums.generate(self.output)
        generate_release_checksums.verify(self.output, checksums)
        lines = checksums.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(any(archive.name in line for line in lines))
        self.assertTrue(any(sbom.name in line for line in lines))

        archive.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            generate_release_checksums.verify(self.output, checksums)


if __name__ == "__main__":
    unittest.main()
