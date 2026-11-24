from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WebAbiContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(
            (ROOT / "abi" / "web-abi-v1-candidate.json").read_text(encoding="utf-8")
        )
        self.source = (ROOT / "web" / "lw_web_api.c").read_text(encoding="utf-8")
        self.sdk = (ROOT / "web" / "lw_ppocr_sdk.template.js").read_text(encoding="utf-8")
        self.cmake = (ROOT / "cmake" / "WebTargets.cmake").read_text(encoding="utf-8")

    def test_manifest_identity_and_status(self) -> None:
        self.assertEqual(self.manifest["schema_version"], 1)
        self.assertEqual(self.manifest["abi_name"], "lw.PPOCR.C.wasm-host")
        self.assertEqual(self.manifest["abi_version"], 1)
        self.assertEqual(self.manifest["status"], "frozen")

    def test_c_layout_matches_manifest(self) -> None:
        structures = self.manifest["structures"]
        for name, contract in structures.items():
            self.assertRegex(
                self.source,
                rf"_Static_assert\(sizeof\({re.escape(name)}\) == {contract['size']}u",
                name,
            )
            for field, offset in contract["fields"].items():
                self.assertRegex(
                    self.source,
                    rf"offsetof\({re.escape(name)}, {re.escape(field)}\) == {offset}u",
                    f"{name}.{field}",
                )

    def test_js_adapter_pins_version_and_sizes(self) -> None:
        self.assertRegex(self.sdk, r"const WEB_ABI_VERSION = 1\s*;")
        self.assertEqual(self.sdk.count("infoSize !== 20"), 2)
        self.assertEqual(self.sdk.count("lineSize !== 60"), 2)
        self.assertEqual(self.sdk.count("resultSize !== 16"), 2)

    def test_exported_symbol_contract_is_present_in_cmake(self) -> None:
        for symbol in self.manifest["stable_symbols"]:
            self.assertIn(f"'_{symbol}'", self.cmake, symbol)
            self.assertIn(symbol, self.source, symbol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
