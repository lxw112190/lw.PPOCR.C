from __future__ import annotations

import argparse
import json
import subprocess
import unittest
from pathlib import Path


class AbiLayoutTest(unittest.TestCase):
    def test_layout_matches_manifest(self) -> None:
        expected = json.loads(ARGUMENTS.manifest.read_text(encoding="utf-8"))
        completed = subprocess.run(
            [str(ARGUMENTS.driver)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        actual = json.loads(completed.stdout)
        self.assertEqual(actual["structures"], expected["structures"])
        self.assertEqual(actual["enums"], expected["enums"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


ARGUMENTS = parse_args()

if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)