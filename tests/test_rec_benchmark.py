from __future__ import annotations

import argparse
import json
import subprocess
import unittest
from pathlib import Path


class RecBenchmarkTest(unittest.TestCase):
    def test_runtime_benchmark_schema_and_stability(self) -> None:
        completed = subprocess.run(
            [
                str(ARGUMENTS.benchmark),
                str(ARGUMENTS.model),
                str(ARGUMENTS.dictionary),
                str(ARGUMENTS.image),
                "2",
                "8",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["schema_version"], 1)
        self.assertIn(report["backend"], {"scalar", "sse2", "avx2"})
        self.assertEqual(report["text"], "纯臻营养护发素")
        self.assertEqual(report["characters"], 7)
        self.assertEqual(report["warmup"], 2)
        self.assertEqual(report["iterations"], 8)
        self.assertGreater(report["model_file_bytes"], 0)
        self.assertGreater(report["workspace_bytes"], 0)
        self.assertGreater(report["preallocated_io_bytes"], 0)
        self.assertGreater(report["recognizer_create_ms"], 0.0)
        latency = report["latency_ms"]
        self.assertGreater(latency["min"], 0.0)
        self.assertLessEqual(latency["min"], latency["median"])
        self.assertLessEqual(latency["median"], latency["max"])
        self.assertLessEqual(latency["p95"], latency["max"])
        self.assertGreater(report["throughput_per_second"], 0.0)
        if report["rss_after_warmup_bytes"] > 0:
            self.assertGreater(report["rss_final_bytes"], 0)
            self.assertLessEqual(report["rss_growth_bytes"], 16 * 1024 * 1024)
        if report["peak_rss_bytes"] > 0 and report["rss_final_bytes"] > 0:
            self.assertGreaterEqual(
                report["peak_rss_bytes"], report["rss_final_bytes"]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    return parser.parse_args()


ARGUMENTS = parse_args()

if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
