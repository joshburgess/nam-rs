import struct
import tempfile
import unittest
from pathlib import Path

from check_release_artifacts import binary_format, validate_architecture


class ReleaseArtifactTests(unittest.TestCase):
    def write_binary(self, data: bytes) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "plugin"
        path.write_bytes(data)
        return path

    def test_validates_x86_64_elf(self) -> None:
        data = bytearray(64)
        data[:4] = b"\x7fELF"
        struct.pack_into("<H", data, 18, 62)
        validate_architecture(self.write_binary(data), "elf")

    def test_validates_x86_64_pe(self) -> None:
        data = bytearray(256)
        data[:2] = b"MZ"
        struct.pack_into("<I", data, 0x3C, 128)
        data[128:132] = b"PE\0\0"
        struct.pack_into("<H", data, 132, 0x8664)
        validate_architecture(self.write_binary(data), "pe")

    def test_validates_arm64_mach_o(self) -> None:
        data = bytearray(64)
        data[:4] = b"\xcf\xfa\xed\xfe"
        struct.pack_into("<I", data, 4, 0x0100000C)
        validate_architecture(self.write_binary(data), "mach-o")

    def test_rejects_wrong_architecture(self) -> None:
        data = bytearray(64)
        data[:4] = b"\x7fELF"
        struct.pack_into("<H", data, 18, 183)

        with self.assertRaisesRegex(ValueError, "expected x86-64"):
            validate_architecture(self.write_binary(data), "elf")

    def test_recognizes_supported_formats(self) -> None:
        self.assertEqual(binary_format(b"\x7fELF"), "elf")
        self.assertEqual(binary_format(b"MZ"), "pe")
        self.assertEqual(binary_format(b"\xcf\xfa\xed\xfe"), "mach-o")
        self.assertIsNone(binary_format(b"text"))


if __name__ == "__main__":
    unittest.main()
