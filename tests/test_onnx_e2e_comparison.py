"""The comparison harness must not turn text drift into a speed claim."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import compare_onnx_e2e as comparison


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "models").mkdir()
        for name in ("det", "cls", "rec", "dict"):
            (self.root / "models" / name).write_bytes(name.encode())
        self.old = self.root / "old.exe"
        self.new = self.root / "new.exe"
        self.old.write_bytes(b"old")
        self.new.write_bytes(b"new")
        self.sample = self.root / "sample.ppm"
        self.sample.write_bytes(b"sample")
        self.calls = []
        self.returncode = 0
        self.mutate = lambda label, workers, result: None

    def run_process(self, args, **kwargs):
        label = "baseline" if Path(args[0]) == self.old else "candidate"
        workers = int(args[-2])
        self.calls.append((label, workers, kwargs["env"]))
        result = {"workers": workers, "rec_target_width": 960, "lines": 16,
                  "image_width": 500, "image_height": 500, "warmup": 2,
                  "iterations": int(args[-3]), "ocr_ms": {"mean": 20 if label == "baseline" else 10},
                  "peak_rss_bytes": 100 * 1048576, "output_checksum": "0000000000000000"}
        self.mutate(label, workers, result)
        return type("Result", (), {"stdout": json.dumps(result), "stderr": "diagnostic",
                                   "returncode": self.returncode})()

    def evaluate(self, **options):
        catalog = {"resolved": {"tiny": {"det": "det", "cls": "cls", "rec": "rec",
                                          "dictionary": "dict"}}}
        with patch.object(comparison, "ROOT", self.root), \
             patch("tools.validate_model_catalog.validate_catalog", return_value=catalog), \
             patch.object(comparison.subprocess, "run", side_effect=self.run_process), \
             patch.dict(comparison.os.environ, {"LW_X64REC_PROFILE": "1", "LW_WASM_DET_DEBUG": "1"}), \
             contextlib.redirect_stdout(io.StringIO()):
            return comparison.compare(self.old, self.new, self.sample, 2, 3,
                                      self.root / "report", ["tiny"], **options)

    def test_pairing_identity_and_environment(self):
        report = self.evaluate()
        self.assertEqual([x[0] for x in self.calls[:4]],
                         ["baseline", "candidate", "candidate", "baseline"])
        self.assertTrue(report["text_contract_pass"])
        for case in report["cases"]:
            self.assertEqual(case["paired_speedup"], 2)
            self.assertEqual(case["baseline_peak_mib"], 100)
        self.assertNotEqual(report["binaries"]["baseline"]["sha256"],
                            report["binaries"]["candidate"]["sha256"])
        self.assertNotIn("LW_X64REC_PROFILE", self.calls[0][2])
        self.assertNotIn("LW_WASM_DET_DEBUG", self.calls[0][2])

    def test_text_diff_is_explicit_and_visible(self):
        def mutate(label, workers, result):
            result["output_checksum"] = "0000000000000001" if label == "baseline" else "0000000000000002"
        self.mutate = mutate
        with self.assertRaisesRegex(ValueError, "full text differs"):
            self.evaluate()
        report = self.evaluate(report_text_differences=True)
        self.assertFalse(report["text_contract_pass"])
        self.assertIn("NOT an output-equivalent", (self.root / "report/report.md").read_text())

    def test_worker_nondeterminism_never_allowed(self):
        def mutate(label, workers, result):
            result["output_checksum"] = f"{workers:016x}"
        self.mutate = mutate
        with self.assertRaisesRegex(ValueError, "non-deterministic"):
            self.evaluate(report_text_differences=True)

    def test_contract_rejections(self):
        for key, value in (("workers", 8), ("rec_target_width", 320), ("lines", 0),
                           ("warmup", 0), ("iterations", 1), ("image_width", 512),
                           ("output_checksum", "bad"),
                           ("peak_rss_bytes", 0), ("ocr_ms", {"mean": float("nan")}),
                           ("ocr_ms", {"mean": 0})):
            with self.subTest(key=key):
                self.mutate = lambda label, workers, result: result.update({key: value})
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    self.evaluate()

    def test_failed_process_diagnostics_are_saved(self):
        self.returncode = 1
        with self.assertRaisesRegex(RuntimeError, "stderr.log"):
            self.evaluate()
        self.assertEqual((self.root / "report/tiny-1-1-baseline.stderr.log").read_text(),
                         "diagnostic")

    def test_sideband_rejects_wrong_text(self):
        result = type("Result", (), {"stdout": "0 text=中文 rec=0.99\n"})()
        with patch.object(comparison.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(ValueError, "measured text buffer"):
                comparison.collect_text(self.old, [], self.sample, {}, "bad")

    def test_sideband_full_utf8_text(self):
        import hashlib
        texts = [f"中文 {index}" for index in range(16)]
        checksum = 14695981039346656037
        for byte in ("\0".join(texts) + "\0").encode("utf-8"):
            checksum = ((checksum ^ byte) * 1099511628211) & ((1 << 64) - 1)
        result = type("Result", (), {"stdout": "\n".join(
            f"{index} text={text} rec=0.99" for index, text in enumerate(texts))})()
        with patch.object(comparison.subprocess, "run", return_value=result):
            detail = comparison.collect_text(self.old, [], self.sample, {}, f"{checksum:016x}")
        self.assertEqual(detail["text_lines"], texts)
        self.assertEqual(detail["text_sha256"],
                         hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest())


if __name__ == "__main__":
    unittest.main()
