from __future__ import annotations

import argparse
import json
import subprocess
import sys
import unittest
from pathlib import Path


class StagedPackageTest(unittest.TestCase):
    def test_required_files_and_live_demo(self) -> None:
        root = ARGUMENTS.root.resolve()
        executable = (
            "lw-recognize-ppm.exe"
            if sys.platform == "win32"
            else "lw-recognize-ppm"
        )
        benchmark = (
            "lw-rec-benchmark.exe"
            if sys.platform == "win32"
            else "lw-rec-benchmark"
        )
        shared = "lw_ppocr_c.dll" if sys.platform == "win32" else "liblw_ppocr_c.so"
        required = [
            root / "bin" / executable,
            root / "bin" / benchmark,
            root / ("bin" if sys.platform == "win32" else "lib") / shared,
            root / "include" / "lw_infer.h",
            root / "models" / "rec.lwm",
            root / "models" / "ppocr_keys.txt",
            root / "models" / "sample-crop.ppm",
            root / "lib" / "cmake" / "lw.PPOCR.C" / "lw.PPOCR.CConfig.cmake",
            root / "LICENSE",
            root / "README.md",
            root / "THIRD-PARTY-NOTICES.md",
            root / "sbom.cdx.json",
        ]
        if sys.platform == "win32":
            required.extend(
                [
                    root / "lib" / "lw_ppocr_c.lib",
                    root / "lib" / "lw_ppocr_c_static.lib",
                ]
            )
        else:
            required.extend(
                [
                    root / "lib" / "liblw_ppocr_c.so",
                    root / "lib" / "liblw_ppocr_c_static.a",
                ]
            )
        missing = [str(path) for path in required if not path.is_file()]
        self.assertFalse(missing, f"missing staged files: {missing}")
        completed = subprocess.run(
            [
                str(root / "bin" / executable),
                str(root / "models" / "rec.lwm"),
                str(root / "models" / "ppocr_keys.txt"),
                str(root / "models" / "sample-crop.ppm"),
            ],
            cwd=root,
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
        measured = subprocess.run(
            [
                str(root / "bin" / benchmark),
                str(root / "models" / "rec.lwm"),
                str(root / "models" / "ppocr_keys.txt"),
                str(root / "models" / "sample-crop.ppm"),
                "1",
                "2",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        self.assertEqual(measured.returncode, 0, measured.stdout + measured.stderr)
        report = json.loads(measured.stdout)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["text"], "纯臻营养护发素")
        self.assertEqual(report["iterations"], 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


ARGUMENTS = parse_args()

if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
