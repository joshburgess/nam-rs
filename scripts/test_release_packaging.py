import importlib.util
import json
import os
import random
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

    def test_generated_archive_member_names_obey_safety_contract(self):
        generator = random.Random(0x4E414D)
        alphabet = "abcXYZ019./\\\0_-"
        accepted = 0
        rejected = 0
        for _ in range(2_000):
            name = "".join(
                generator.choice(alphabet) for _ in range(generator.randrange(0, 96))
            )
            try:
                path = package_release.checked_archive_name(name)
            except ValueError:
                rejected += 1
                continue
            accepted += 1
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)
            self.assertNotIn("\\", name)
            self.assertNotIn("\0", name)
            self.assertTrue(path.parts)
        self.assertGreater(accepted, 0)
        self.assertGreater(rejected, 0)

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

    def test_install_state_rejects_redirects_duplicates_and_malformed_metadata(self):
        archive = self.package()
        install_root = self.root / "installed"
        paths = install_release.install(archive, install_root)
        original = json.loads(paths.state.read_text(encoding="utf-8"))
        outside = self.root / "outside"
        outside.write_bytes(b"must survive")

        mutations = {
            "redirected path": lambda state: state["installed"][0].update(
                path=str(outside), sha256=install_release.tree_hash(outside)
            ),
            "duplicate path": lambda state: state["installed"][1].update(
                path=state["installed"][0]["path"]
            ),
            "wrong platform": lambda state: state.update(platform="windows-x86_64"),
            "wrong schema": lambda state: state.update(schema_version=2),
            "invalid checksum": lambda state: state["installed"][0].update(sha256="xyz"),
            "missing entry": lambda state: state["installed"].pop(),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                state = json.loads(json.dumps(original))
                mutate(state)
                with self.assertRaises(ValueError):
                    install_release.validate_install_state(
                        state, paths, "linux-x86_64"
                    )
        self.assertEqual(outside.read_bytes(), b"must survive")

    def test_generated_install_state_mutations_never_escape_expected_targets(self):
        archive = self.package()
        install_root = self.root / "generated-state"
        paths = install_release.install(archive, install_root)
        original = json.loads(paths.state.read_text(encoding="utf-8"))
        expected = {paths.vst3, paths.clap, paths.trainer}
        generator = random.Random(0x4132)

        for _ in range(64):
            state = json.loads(json.dumps(original))
            generator.shuffle(state["installed"])
            validated = install_release.validate_install_state(
                state, paths, "linux-x86_64"
            )
            self.assertEqual({path for path, _ in validated}, expected)

        for case in range(256):
            state = json.loads(json.dumps(original))
            entry = generator.randrange(len(state["installed"]))
            mutation = case % 6
            if mutation == 0:
                state["installed"][entry]["path"] = str(
                    self.root / f"outside-{case}"
                )
            elif mutation == 1:
                state["installed"][entry]["sha256"] = "".join(
                    generator.choice("0123456789abcdefXYZ") for _ in range(64)
                )
                if all(
                    character in "0123456789abcdef"
                    for character in state["installed"][entry]["sha256"]
                ):
                    state["installed"][entry]["sha256"] = "g" * 64
            elif mutation == 2:
                state["installed"][entry] = generator.choice(
                    [None, [], "entry", 7]
                )
            elif mutation == 3:
                state["installed"].append(state["installed"][entry].copy())
            elif mutation == 4:
                state["installed"].pop(entry)
            else:
                state["platform"] = generator.choice(
                    [None, "", "windows-x86_64", 7]
                )

            with self.subTest(case=case), self.assertRaises(ValueError):
                install_release.validate_install_state(
                    state, paths, "linux-x86_64"
                )

    def test_uninstall_rejects_replaced_installation_symlink(self):
        archive = self.package()
        install_root = self.root / "installed-symlink"
        paths = install_release.install(archive, install_root)
        outside = self.root / "outside-plugin"
        outside.write_bytes(paths.clap.read_bytes())
        paths.clap.unlink()
        try:
            paths.clap.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"symbolic links are unavailable: {error}")

        with self.assertRaisesRegex(ValueError, "symbolic link"):
            install_release.uninstall("linux-x86_64", install_root)
        self.assertTrue(outside.exists())
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
