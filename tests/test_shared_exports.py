from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unittest
from pathlib import Path


def exported_symbols(library: Path, tool: Path) -> set[str]:
    if sys.platform == "win32":
        command = [str(tool), "/nologo", "/exports", str(library)]
        pattern = re.compile(
            r"^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(lw_[A-Za-z0-9_]+)\s*$"
        )
    elif sys.platform == "darwin":
        command = [str(tool), "-gU", str(library)]
        pattern = re.compile(r"\b_?(lw_[A-Za-z0-9_]+)$")
    else:
        command = [str(tool), "-D", "--defined-only", str(library)]
        pattern = re.compile(r"\b(lw_[A-Za-z0-9_]+)(?:@@?[^ ]+)?$")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stdout + completed.stderr)
    return {
        match.group(1)
        for line in completed.stdout.splitlines()
        if (match := pattern.search(line)) is not None
    }


class SharedExportTest(unittest.TestCase):
    def test_exports_match_allowlist(self) -> None:
        expected = {
            line.strip()
            for line in ARGUMENTS.allowlist.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        actual = exported_symbols(ARGUMENTS.library, ARGUMENTS.tool)
        self.assertEqual(actual, expected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--tool", type=Path, required=True)
    return parser.parse_args()


ARGUMENTS = parse_args()

if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
