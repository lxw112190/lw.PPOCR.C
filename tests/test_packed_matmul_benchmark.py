from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import unittest


class PackedMatMulBenchmarkTest(unittest.TestCase):
    def test_terminal_geometry_and_result_contract(self) -> None:
        completed = subprocess.run(
            [ARGS.driver, "1"],
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
        self.assertIn(report["backend"], {"scalar", "sse2", "avx2", "neon", "lsx", "lasx"})
        self.assertEqual(report["rows"], 40)
        self.assertEqual(report["inner_dimension"], 80)
        self.assertEqual(report["columns"], 6906)
        self.assertEqual(report["iterations"], 1)
        if not report["supported"]:
            return
        for field in ("scalar_ms", "avx2_ms", "speedup"):
            self.assertTrue(math.isfinite(report[field]), (field, report))
            self.assertGreater(report[field], 0.0, (field, report))
        self.assertRegex(report["output_checksum"], re.compile(r"^0x[0-9a-f]{16}$"))
        self.assertRegex(report["argmax_checksum"], re.compile(r"^0x[0-9a-f]{16}$"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    unittest.main(argv=[__file__], verbosity=2)
