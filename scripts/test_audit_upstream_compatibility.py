import struct
import tempfile
import unittest
from pathlib import Path

import audit_upstream_compatibility as audit


def write_float_wav(path: Path, samples: list[float]) -> None:
    sample_data = struct.pack(f"<{len(samples)}f", *samples)
    format_chunk = struct.pack("<HHIIHH", 3, 1, 48_000, 192_000, 4, 32)
    riff_size = 4 + 8 + len(format_chunk) + 8 + len(sample_data)
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVEfmt "
        + struct.pack("<I", len(format_chunk))
        + format_chunk
        + b"data"
        + struct.pack("<I", len(sample_data))
        + sample_data
    )


class CompatibilityAuditTests(unittest.TestCase):
    def test_reads_float_wav_and_reports_bit_exact_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "samples.wav"
            samples = [0.0, -0.25, 0.5, 1.0]
            write_float_wav(wav_path, samples)

            actual = audit.read_float_wav(wav_path)
            comparison = audit.compare_samples(actual, samples)

        self.assertEqual(actual, samples)
        self.assertEqual(comparison.differing_samples, 0)
        self.assertEqual(comparison.max_abs_error, 0.0)
        self.assertEqual(comparison.rms_error, 0.0)

    def test_reports_float_bit_differences(self) -> None:
        expected = [0.0, 1.0]
        actual = [0.0, f32_from_bits(0x3F80_0001)]

        comparison = audit.compare_samples(actual, expected)

        self.assertEqual(comparison.differing_samples, 1)
        self.assertGreater(comparison.max_abs_error, 0.0)

    def test_rejects_sample_count_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample-count mismatch"):
            audit.compare_samples([0.0], [0.0, 1.0])


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


if __name__ == "__main__":
    unittest.main()
