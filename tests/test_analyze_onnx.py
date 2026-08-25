from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from converter import analyze_onnx


class AnalyzeBundledModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = {
            label: analyze_onnx.analyze_model(
                label,
                path,
                analyze_onnx.DEFAULT_REPRESENTATIVE_SHAPES[label],
            )
            for label, path in analyze_onnx.DEFAULT_MODELS.items()
        }
        cls.report = analyze_onnx.build_report(list(cls.models.values()))

    def test_exact_model_identity(self) -> None:
        expected = {
            "det": ("193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8", 242),
            "cls": ("dd8b2b61983d76ab230a58da9e0e0e84956b71c3877f2ce6e438fe22d74d2cf2", 143),
            "rec": ("9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6", 219),
        }
        for label, (sha256, nodes) in expected.items():
            with self.subTest(model=label):
                self.assertEqual(self.models[label]["sha256"], sha256)
                self.assertEqual(self.models[label]["node_count"], nodes)

    def test_operator_counts_are_locked_to_bundled_models(self) -> None:
        self.assertEqual(self.models["det"]["operator_counts"]["Conv"], 83)
        self.assertEqual(self.models["cls"]["operator_counts"]["BatchNormalization"], 27)
        self.assertEqual(self.models["rec"]["operator_counts"]["Identity"], 58)
        self.assertEqual(self.models["rec"]["operator_counts"]["MatMul"], 2)
        self.assertEqual(self.models["rec"]["operator_counts"]["Softmax"], 1)

    def test_dynamic_rec_width_is_preserved(self) -> None:
        rec = self.models["rec"]
        self.assertEqual(rec["inputs"][0]["shape"][:3], ["DynamicDimension.0", 3, 48])
        self.assertEqual(rec["inputs"][0]["shape"][3], "DynamicDimension.1")
        self.assertIn("DynamicDimension.1", str(rec["outputs"][0]["shape"]))
        self.assertGreater(len(rec["dynamic_values"]), 0)

    def test_shape_and_compute_analysis_are_nonempty(self) -> None:
        self.assertGreaterEqual(len(self.models["cls"]["shape_only_nodes"]), 1)
        for label, model in self.models.items():
            with self.subTest(model=label):
                self.assertGreater(model["flops"]["total"], 0)
                self.assertIsNone(model["shape_inference_error"])

    def test_report_is_deterministic_and_answers_required_sections(self) -> None:
        first = json.dumps(self.report, ensure_ascii=False, sort_keys=True)
        second = json.dumps(
            analyze_onnx.build_report(list(self.models.values())),
            ensure_ascii=False,
            sort_keys=True,
        )
        self.assertEqual(first, second)
        markdown = analyze_onnx.render_markdown(self.report)
        for heading in (
            "Operator union",
            "REC-first runtime surface",
            "REC dynamic-width propagation",
            "Shape and converter-removable work",
            "Main compute conclusion",
        ):
            self.assertIn(heading, markdown)

    def test_cli_writes_machine_and_human_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            json_path = root / "analysis.json"
            markdown_path = root / "analysis.md"
            status = analyze_onnx.main([
                "--json-output", str(json_path),
                "--markdown-output", str(markdown_path),
            ])
            self.assertEqual(status, 0)
            parsed = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["analysis_schema_version"], 1)
            self.assertEqual(len(parsed["models"]), 3)
            self.assertIn("REC dynamic-width propagation", markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
