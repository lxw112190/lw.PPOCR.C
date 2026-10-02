import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_compare_ocr_dataset_reports import make_report
from tools.compare_native_rec_quality import compare_quality
from tools import compare_native_rec_quality as comparison


def fixture():
    result = make_report("same-model", 1.0, 0.03)
    result["images"] = [{"file": "one.jpg"}, {"file": "two.jpg"}]
    result["metrics"]["ground_truth_lines"] = 10
    result["dataset"]["selected_images"] = 2
    return result


class NativeRecQualityTest(unittest.TestCase):
    def test_runner_pins_child_utf8_and_saves_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models").mkdir()
            for asset in ("det", "cls", "rec", "dict"):
                (root / "models" / asset).write_bytes(asset.encode())
            args = SimpleNamespace(baseline=root / "old.exe", candidate=root / "new.exe",
                                   dataset=root / "metadata.json", output=root / "output",
                                   models=["tiny"], limit=None, report_quality_differences=False)
            catalog = {"resolved": {"tiny": {"det": "det", "cls": "cls",
                                              "rec": "rec", "dictionary": "dict"}}}
            def process(command, **kwargs):
                self.assertTrue(kwargs["check"])
                self.assertEqual(kwargs["env"]["PYTHONIOENCODING"], "utf-8")
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps(fixture()), encoding="utf-8")
            with patch.object(comparison, "ROOT", root), \
                    patch.object(comparison, "validate_catalog", return_value=catalog), \
                    patch.object(comparison, "binary_identity", return_value={"sha256": "same"}), \
                    patch.object(comparison.subprocess, "run", side_effect=process), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = comparison.run(args)
            self.assertTrue(result["quality_pass"])
            self.assertEqual(result["accuracy_gate_mode"], "strict")
            self.assertTrue((args.output / "tiny-baseline.json").is_file())
            self.assertIn("PASS", (args.output / "quality.md").read_text(encoding="utf-8"))

    def test_cli_quality_regressions_fail_by_default(self):
        arguments = ["quality", "--baseline", "old.exe", "--candidate", "new.exe",
                     "--dataset", "metadata.json", "--output", "reports"]
        with patch("sys.argv", arguments), patch.object(comparison, "run",
                return_value={"quality_pass": False}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(comparison.main(), 1)
        with patch("sys.argv", arguments + ["--report-quality-differences"]), \
                patch.object(comparison, "run", return_value={"quality_pass": False}), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(comparison.main(), 0)
            self.assertIn("do not promote", output.getvalue())

    def test_report_only_does_not_waive_broken_contract(self):
        arguments = ["quality", "--baseline", "old.exe", "--candidate", "new.exe",
                     "--dataset", "metadata.json", "--output", "reports",
                     "--report-quality-differences"]
        with patch("sys.argv", arguments), patch.object(comparison, "run",
                side_effect=ValueError("incompatible report")):
            with self.assertRaisesRegex(ValueError, "incompatible"):
                comparison.main()

    def test_accept_correct_text_improvement(self):
        old = fixture()
        new = copy.deepcopy(old)
        new["metrics"]["cer_on_matched_lines"] = 0.02
        self.assertTrue(compare_quality(old, new)["non_regression"])

    def test_no_quality_regression_budget(self):
        for key, value in (("cer_on_matched_lines", 0.031),
                           ("exact_reference_line_rate", 0.39), ("detection_f1", 0.99)):
            with self.subTest(metric=key):
                old = fixture()
                new = copy.deepcopy(old)
                new["metrics"][key] = value
                self.assertFalse(compare_quality(old, new)["non_regression"])

    def test_identical_selection_required(self):
        old = fixture()
        new = copy.deepcopy(old)
        new["images"].reverse()
        with self.assertRaisesRegex(ValueError, "identical ordered"):
            compare_quality(old, new)

    def test_invalid_or_incompatible_reports_rejected(self):
        for mutate in (
            lambda r: r["metrics"].update(cer_on_matched_lines=float("nan")),
            lambda r: r["metrics"].update(detection_f1=1.01),
            lambda r: r["metrics"].update(ground_truth_lines=0),
            lambda r: r["dataset"].update(manifest_sha256="wrong"),
            lambda r: r.update(rec_max_width=320),
            lambda r: r.update(status="failed"),
            lambda r: r.update(schema_version=2),
            lambda r: r["dataset"].update(selected_images=1),
            lambda r: r["dataset"].update(manifest_sha256=""),
            lambda r: r["images"].append({"file": "one.jpg"}),
        ):
            new = fixture()
            mutate(new)
            with self.assertRaises(ValueError):
                compare_quality(fixture(), new)


if __name__ == "__main__":
    unittest.main()
