import unittest
from unittest.mock import Mock, patch

from check_plugin_features import nam_core_features


class PluginFeatureTests(unittest.TestCase):
    @patch("check_plugin_features.subprocess.run")
    def test_reads_nam_core_features(self, run: Mock) -> None:
        run.return_value.stdout = (
            "nam-plugin v0.1.0|default,fast-kernels\n"
            "nam-core v0.1.0 (/workspace/nam-core)|default,fast-kernels\n"
        )

        self.assertEqual(nam_core_features([]), {"default", "fast-kernels"})

    @patch("check_plugin_features.subprocess.run")
    def test_rejects_missing_nam_core_entry(self, run: Mock) -> None:
        run.return_value.stdout = "nam-plugin v0.1.0|default\n"

        with self.assertRaisesRegex(ValueError, "expected one nam-core"):
            nam_core_features([])


if __name__ == "__main__":
    unittest.main()
