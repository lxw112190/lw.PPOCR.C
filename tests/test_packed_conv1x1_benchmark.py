from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import unittest


class PackedConv1x1BenchmarkTest(unittest.TestCase):
    def test_rec_and_medium_geometries_are_correct_and_machine_readable(self) -> None:
        for target_width, expected_width in ((320, 80), (960, 240)):
            completed = subprocess.run(
                [ARGS.driver, str(target_width), "1"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["target_width"], target_width)
            self.assertEqual(report["iterations"], 1)
            self.assertTrue(report["backend"])
            cases = report["cases"]
            self.assertEqual(
                [(item["input_channels"], item["output_channels"], item["height"], item["width"])
                 for item in cases],
                [
                    (96, 192, 12, expected_width),
                    (48, 96, 12, expected_width),
                    (96, 48, 12, expected_width),
                    (96, 192, 6, expected_width),
                    (192, 96, 6, expected_width),
                    (160, 320, 3, expected_width),
                    (320, 160, 3, expected_width),
                    (192, 384, 6, expected_width),
                    (384, 192, 6, expected_width),
                    (384, 768, 3, expected_width),
                    (768, 384, 3, expected_width),
                    (512, 1024, 6, expected_width),
                    (1024, 512, 6, expected_width),
                    (1536, 768, 3, expected_width),
                ],
            )
            for item in cases:
                for field in ("scalar_ms", "avx2_ms", "avx2_min_ms", "avx2_max_ms", "avx2_p90_ms", "avx2_speedup", "dispatched_ms", "speedup"):
                    self.assertTrue(math.isfinite(item[field]), (field, item))
                    self.assertGreater(item[field], 0.0, (field, item))
                self.assertRegex(item["checksum"], re.compile(r"^0x[0-9a-f]{16}$"))
                if "fma_ms" in item:
                    for field in ("fma_ms", "fma_min_ms", "fma_max_ms", "fma_p90_ms", "fma_speedup", "fma_vs_avx2", "fma_max_abs_error", "fma_max_relative_error"):
                        self.assertTrue(math.isfinite(item[field]), (field, item))
                    self.assertGreater(item["fma_ms"], 0.0, item)
                    self.assertGreater(item["fma_speedup"], 0.0, item)
                    self.assertGreater(item["fma_vs_avx2"], 0.0, item)
                    self.assertLessEqual(item["fma_max_abs_error"], 1.0e-2, item)
                    self.assertRegex(item["fma_checksum"], re.compile(r"^0x[0-9a-f]{16}$"))
                    if "fma8_ms" in item:
                        for field in ("fma8_ms", "fma8_vs_fma", "fma8_max_abs_error"):
                            self.assertTrue(math.isfinite(item[field]), (field, item))
                        self.assertGreater(item["fma8_ms"], 0.0, item)
                        self.assertGreater(item["fma8_vs_fma"], 0.0, item)
                        self.assertLessEqual(item["fma8_max_abs_error"], 1.0e-2, item)
                        self.assertRegex(item["fma8_checksum"], re.compile(r"^0x[0-9a-f]{16}$"))
                        self.assertEqual(item["fma8_checksum"], item["fma_checksum"], item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    unittest.main(argv=[__file__], verbosity=2)
