from __future__ import annotations

import argparse
import subprocess
import unittest
from pathlib import Path


class RecognizePpmDemoTest(unittest.TestCase):
    def test_demo_recognizes_packaged_fixture(self) -> None:
        completed = subprocess.run(
            [
                str(ARGUMENTS.demo),
                str(ARGUMENTS.model),
                str(ARGUMENTS.dictionary),
                str(ARGUMENTS.image),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("text=纯臻营养护发素", completed.stdout)
        self.assertIn("chars=7", completed.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    return parser.parse_args()


ARGUMENTS = parse_args()

if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
