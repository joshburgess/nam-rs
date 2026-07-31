import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from install_plugin_validators import checked_relative_path, install_validators


class PluginValidatorInstallerTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write_zip(self, name: str, members: dict[str, bytes]) -> Path:
        path = self.root / name
        with zipfile.ZipFile(path, "w") as archive:
            for member, contents in members.items():
                archive.writestr(member, contents)
        return path

    def write_nested_tar_zip(self, name: str) -> Path:
        tar_contents = io.BytesIO()
        with tarfile.open(fileobj=tar_contents, mode="w:gz") as archive:
            executable = b"clap validator"
            info = tarfile.TarInfo("clap-validator")
            info.size = len(executable)
            archive.addfile(info, io.BytesIO(executable))
        return self.write_zip(name, {"clap-validator.tar.gz": tar_contents.getvalue()})

    def test_installs_zip_and_nested_tar_archives(self) -> None:
        pluginval = self.write_zip("pluginval.zip", {"pluginval": b"plugin validator"})
        clap_validator = self.write_nested_tar_zip("clap.zip")
        manifest = {
            "pluginval": {
                "platforms": {
                    "linux": {
                        "url": pluginval.as_uri(),
                        "sha256": hashlib.sha256(pluginval.read_bytes()).hexdigest(),
                        "executable": "pluginval",
                    }
                }
            },
            "clap-validator": {
                "platforms": {
                    "linux": {
                        "url": clap_validator.as_uri(),
                        "sha256": hashlib.sha256(clap_validator.read_bytes()).hexdigest(),
                        "executable": "clap-validator",
                    }
                }
            },
        }
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        installed = install_validators(manifest_path, "linux", self.root / "tools")

        self.assertEqual(installed["pluginval"].read_bytes(), b"plugin validator")
        self.assertEqual(installed["clap-validator"].read_bytes(), b"clap validator")

    def test_rejects_checksum_mismatch(self) -> None:
        archive = self.write_zip("pluginval.zip", {"pluginval": b"validator"})
        manifest = {
            "pluginval": {
                "platforms": {
                    "linux": {
                        "url": archive.as_uri(),
                        "sha256": "0" * 64,
                        "executable": "pluginval",
                    }
                }
            }
        }
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "expected"):
            install_validators(manifest_path, "linux", self.root / "tools")

    def test_rejects_archive_path_traversal(self) -> None:
        with self.assertRaisesRegex(ValueError, "escapes"):
            checked_relative_path("../validator")


if __name__ == "__main__":
    unittest.main()
