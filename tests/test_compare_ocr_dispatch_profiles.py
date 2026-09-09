import unittest

from tools.compare_ocr_dispatch_profiles import render_markdown, summarize


def report(mean, p95, rss=80 * 1024 * 1024):
    return {
        "ocr_ms": {"mean": mean, "p95": p95},
        "peak_rss_bytes": rss,
        "output_checksum": "0ebf8b448ab7df47",
        "lines": 16,
    }


class CompareOcrDispatchProfilesTest(unittest.TestCase):
    def test_summary_uses_median_and_preserves_contract(self):
        summary = summarize(
            [report(100.0, 110.0), report(120.0, 130.0), report(110.0, 120.0)],
            [report(80.0, 90.0), report(100.0, 110.0), report(90.0, 100.0)],
        )
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["repeats"], 3)
        self.assertEqual(summary["baseline"]["ocr_mean_ms"], 110.0)
        self.assertEqual(summary["candidate"]["ocr_mean_ms"], 90.0)
        self.assertAlmostEqual(summary["comparison"]["mean_speedup"], 110.0 / 90.0)
        self.assertTrue(summary["contract"]["match"])

    def test_summary_rejects_text_contract_change(self):
        changed = report(90.0, 100.0)
        changed["output_checksum"] = "different"
        with self.assertRaisesRegex(RuntimeError, "OCR output contract differs"):
            summarize([report(100.0, 110.0)], [changed])

    def test_markdown_names_both_dispatches(self):
        summary = summarize([report(100.0, 110.0)], [report(90.0, 100.0)])
        markdown = render_markdown(summary, "FMA experiment")
        self.assertIn("Default AVX2", markdown)
        self.assertIn("Experimental AVX2+FMA", markdown)
        self.assertIn("Checksum:", markdown)


if __name__ == "__main__":
    unittest.main()