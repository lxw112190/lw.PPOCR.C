from __future__ import annotations

import argparse
import json
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


class AbiV1CandidateTest(unittest.TestCase):
    def test_manifest_matches_candidate_scope(self) -> None:
        manifest = json.loads(ARGUMENTS.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["abi_name"], "lw.PPOCR.C")
        self.assertEqual(manifest["abi_version"], 1)
        self.assertEqual(manifest["status"], "freeze-candidate")
        self.assertEqual(manifest["stable_symbols"], "abi/exports-v1-candidate.txt")
        self.assertIn("lw_model_*", manifest["experimental_scope"])
        self.assertIn("lw_session_*", manifest["experimental_scope"])

    def test_candidate_symbols_are_declared_and_exported(self) -> None:
        symbols = {
            line.strip()
            for line in ARGUMENTS.allowlist.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        header = ARGUMENTS.header.read_text(encoding="utf-8")
        actual = exported_symbols(ARGUMENTS.library, ARGUMENTS.tool)
        for symbol in sorted(symbols):
            self.assertRegex(header, rf"\b{re.escape(symbol)}\s*\(", symbol)
        self.assertTrue(symbols <= actual, sorted(symbols - actual))

    def test_low_level_api_is_not_in_candidate_scope(self) -> None:
        symbols = {
            line.strip()
            for line in ARGUMENTS.allowlist.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        experimental = sorted(
            symbol
            for symbol in symbols
            if symbol.startswith("lw_model_") or symbol.startswith("lw_session_")
        )
        self.assertEqual(experimental, [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--tool", type=Path, required=True)
    return parser.parse_args()


ARGUMENTS = parse_args()

if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
