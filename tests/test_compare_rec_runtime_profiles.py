from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from tools.compare_rec_runtime_profiles import render_markdown, summarize


class CompareRecRuntimeProfilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.compact = {
            "ocr_ms": {"mean": 124.5, "p95": 144.2},
            "peak_rss_bytes": 128 * 1024 * 1024,
            "output_checksum": "abc",
            "lines": 16,
        }
        self.performance = {
            "ocr_ms": {"mean": 111.1, "p95": 117.4},
            "peak_rss_bytes": 175 * 1024 * 1024,
            "output_checksum": "abc",
            "lines": 16,
        }

    def test_summary_calculates_speed_and_memory_delta(self) -> None:
        summary = summarize(self.compact, self.performance)
        self.assertEqual(summary["schema_version"], 1)
        self.assertAlmostEqual(summary["comparison"]["mean_speedup"], 124.5 / 111.1)
        self.assertAlmostEqual(summary["comparison"]["rss_delta_mib"], 47.0)
        self.assertTrue(summary["contract"]["match"])
        self.assertEqual(summary["contract"]["compact_checksum"], "abc")

    def test_contract_mismatch_is_rejected(self) -> None:
        different = dict(self.performance, output_checksum="different")
        with self.assertRaisesRegex(RuntimeError, "OCR output contract differs"):
            summarize(self.compact, different)

    def test_markdown_contains_contract_and_metrics(self) -> None:
        markdown = render_markdown(summarize(self.compact, self.performance))
        self.assertIn("Compact vs Performance OCR runtime", markdown)
        self.assertIn("1.121x", markdown)
        self.assertIn("abc` vs `abc", markdown)

    def test_json_round_trip_is_stable(self) -> None:
        summary = summarize(self.compact, self.performance)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "summary.json"
            path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), summary)


if __name__ == "__main__":
    unittest.main()
