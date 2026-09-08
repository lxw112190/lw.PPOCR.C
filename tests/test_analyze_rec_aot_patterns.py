from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from converter.analyze_rec_aot_patterns import (
    analyze_rec_patterns,
    main,
    render_markdown,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "ppocrv6-tiny" / "rec.onnx"


class RecAotPatternAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = analyze_rec_patterns(MODEL, [1, 3, 48, 320])

    def test_report_identity_and_shape(self) -> None:
        self.assertEqual(self.report["schema_version"], 1)
        self.assertEqual(self.report["onnx_node_count"], 219)
        self.assertEqual(self.report["representative_input_shape"], [1, 3, 48, 320])
        self.assertEqual(len(self.report["heavy_nodes"]), 39)

    def test_terminal_matmul_is_explicit_candidate(self) -> None:
        matmuls = [node for node in self.report["heavy_nodes"] if node["op"] == "MatMul"]
        self.assertEqual(len(matmuls), 2)
        terminal = matmuls[-1]
        self.assertEqual(terminal["input_shape"], [1, 40, 80])
        self.assertEqual(terminal["right_shape"], [80, 6906])
        self.assertTrue(any(candidate["pattern"] == "terminal_matmul" for candidate in self.report["aot_candidates"]))

    def test_repeated_pointwise_family_is_reported(self) -> None:
        families = self.report["repeated_families"]
        pointwise = [family for family in families if family["kind"] == "pointwise"]
        self.assertTrue(pointwise)
        self.assertTrue(any(family["count"] >= 3 for family in pointwise))
        self.assertTrue(
            any(candidate["pattern"] == "repeated_pointwise_family" for candidate in self.report["aot_candidates"])
        )

    def test_markdown_is_deterministic_and_safe(self) -> None:
        markdown = render_markdown(self.report)
        self.assertIn("## AOT candidates", markdown)
        self.assertIn("Repeated Conv families", markdown)
        self.assertIn("not LWM indexes", markdown)
        self.assertEqual(markdown, render_markdown(json.loads(json.dumps(self.report))))

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            json_path = root / "patterns.json"
            markdown_path = root / "patterns.md"
            status = main(
                [
                    "--model",
                    str(MODEL),
                    "--json-output",
                    str(json_path),
                    "--markdown-output",
                    str(markdown_path),
                ]
            )
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["schema_version"], 1)
            self.assertIn("AOT candidates", markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()