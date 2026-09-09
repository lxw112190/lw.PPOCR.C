from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import unittest


class ErfBenchmarkTest(unittest.TestCase):
    def test_rec_geometries_and_error_contract(self) -> None:
        for target_width, width_by_stage in ((320, (160, 80)), (960, (480, 240))):
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
            if not report["supported"]:
                continue
            cases = report["cases"]
            self.assertEqual(
                [(item["channels"], item["height"], item["width"]) for item in cases],
                [
                    (24, 24, width_by_stage[0]),
                    (96, 12, width_by_stage[1]),
                    (96, 6, width_by_stage[1]),
                    (192, 6, width_by_stage[1]),
                    (192, 3, width_by_stage[1]),
                    (320, 3, width_by_stage[1]),
                ],
            )
            for item in cases:
                for field in ("scalar_ms", "avx2_ms", "speedup"):
                    self.assertTrue(math.isfinite(item[field]), (field, item))
                    self.assertGreater(item[field], 0.0, (field, item))
                self.assertLessEqual(item["max_abs_error"], 5.0e-7, item)
                self.assertRegex(item["checksum"], re.compile(r"^0x[0-9a-f]{16}$"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    unittest.main(argv=[__file__], verbosity=2)
