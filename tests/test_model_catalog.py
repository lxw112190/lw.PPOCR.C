from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import onnx

from converter.ppocr_contracts import (
    PP_OCRV6_MEDIUM_DET_SHA256,
    PP_OCRV6_MEDIUM_REC_SHA256,
)
from tools.validate_model_catalog import validate_catalog

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "models" / "ppocrv6-models.json"
MEDIUM_REPORT = ROOT / "docs" / "ppocrv6-medium-analysis.json"


class ModelCatalogTests(unittest.TestCase):
    def test_catalog_resolves_shared_cls_and_dictionary(self) -> None:
        report = validate_catalog(CATALOG)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["resolved"]["tiny"]["cls"], report["resolved"]["small"]["cls"])
        self.assertEqual(report["resolved"]["small"]["cls"], report["resolved"]["medium"]["cls"])
        self.assertEqual(report["resolved"]["small"]["dictionary"], report["resolved"]["medium"]["dictionary"])
        self.assertEqual(
            report["runtime_status"],
            {"tiny": "primary", "small": "supported", "medium": "supported"},
        )

        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        self.assertEqual(set(catalog["shared_assets"]["cls"]["used_by"]), {"tiny", "small", "medium"})
        self.assertEqual(set(catalog["shared_assets"]["small_rec_dictionary"]["used_by"]), {"small", "medium"})

    def test_catalog_hashes_are_real_files(self) -> None:
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        for entry in catalog["variants"].values():
            for role in ("det", "rec"):
                asset = ROOT / "models" / entry[role]["path"]
                self.assertEqual(hashlib.sha256(asset.read_bytes()).hexdigest(), entry[role]["sha256"])

    def test_small_and_medium_rec_match_the_shared_dictionary(self) -> None:
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        dictionary = ROOT / "models" / "ppocrv6-shared" / "PP-OCRv6_small_rec_dict.txt"
        entries = len(dictionary.read_text(encoding="utf-8").splitlines())
        for variant in ("small", "medium"):
            rec = ROOT / "models" / catalog["variants"][variant]["rec"]["path"]
            model = onnx.load(str(rec), load_external_data=True)
            onnx.checker.check_model(model)
            classes = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
            self.assertEqual(classes, entries + 2, variant)

    def test_medium_analysis_report_matches_catalog(self) -> None:
        report = json.loads(MEDIUM_REPORT.read_text(encoding="utf-8"))
        models = {entry["label"]: entry for entry in report["models"]}
        self.assertEqual(set(models), {"det", "rec"})
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        for role in ("det", "rec"):
            self.assertEqual(models[role]["sha256"], catalog["variants"]["medium"][role]["sha256"])
            self.assertIsNone(models[role]["shape_inference_error"])

    def test_medium_conversion_contract_hashes_match_assets(self) -> None:
        self.assertEqual(
            hashlib.sha256((ROOT / "models/ppocrv6-medium/det.onnx").read_bytes()).hexdigest(),
            PP_OCRV6_MEDIUM_DET_SHA256,
        )
        self.assertEqual(
            hashlib.sha256((ROOT / "models/ppocrv6-medium/rec.onnx").read_bytes()).hexdigest(),
            PP_OCRV6_MEDIUM_REC_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
