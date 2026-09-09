from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"[0-9a-f]{64}")
VARIANTS = ("tiny", "small", "medium")


class WebOcrGoldenContractTest(unittest.TestCase):
    def load_web_contract(self, variant: str) -> dict:
        path = ROOT / "ci" / f"web-ppocrv6-{variant}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_contracts_are_complete_and_use_the_bundled_sample(self) -> None:
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                contract = self.load_web_contract(variant)
                self.assertEqual(
                    set(contract),
                    {
                        "schema_version",
                        "family",
                        "variant",
                        "sample",
                        "expected_line_count",
                        "expected_text_sha256",
                    },
                )
                self.assertEqual(contract["schema_version"], 1)
                self.assertEqual(contract["family"], "PP-OCRv6")
                self.assertEqual(contract["variant"], variant)
                self.assertEqual(contract["expected_line_count"], 16)
                self.assertRegex(contract["expected_text_sha256"], SHA256_RE)
                self.assertTrue((ROOT / contract["sample"]).is_file())

    def test_larger_web_goldens_match_native_validation(self) -> None:
        for variant in ("small", "medium"):
            with self.subTest(variant=variant):
                web = self.load_web_contract(variant)
                native_path = ROOT / "ci" / f"ppocrv6-{variant}-validation.json"
                native = json.loads(native_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    web["expected_line_count"], native["full_ocr"]["expected_lines"]
                )
                self.assertEqual(
                    web["expected_text_sha256"],
                    native["full_ocr"]["expected_text_sha256"],
                )

    def test_documentation_and_workflows_describe_the_preview_boundary(self) -> None:
        supported = (ROOT / "docs" / "supported-models.md").read_text(encoding="utf-8")
        web_sdk = (ROOT / "docs" / "web-sdk.md").read_text(encoding="utf-8")
        model_packs = (ROOT / "docs" / "model-packs.md").read_text(encoding="utf-8")
        workflow = (ROOT / ".github" / "workflows" / "wasm-html.yml").read_text(
            encoding="utf-8"
        )
        release = (ROOT / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8"
        )

        for variant in VARIANTS:
            title = variant.title()
            self.assertIn(f"| {title} |", supported)
            self.assertIn(f"ci/web-ppocrv6-{variant}.json", workflow)
        self.assertIn("| Small | ✅ Windows/Linux golden", supported)
        self.assertIn("| Medium | ✅ Windows/Linux golden", supported)
        self.assertIn("desktop-first preview", supported)
        self.assertIn("lw-ppocr-v6-small.js", web_sdk)
        self.assertIn("lw-ppocr-v6-medium.js", web_sdk)
        self.assertIn("Small and Medium are opt-in preview assets", model_packs)
        self.assertIn("build_web_variants: true", release)
    def test_checksum_contract_joins_lines_with_lf(self) -> None:
        texts = ["第一行", "second", ""]
        expected = hashlib.sha256("第一行\nsecond\n".encode("utf-8")).hexdigest()
        actual = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
