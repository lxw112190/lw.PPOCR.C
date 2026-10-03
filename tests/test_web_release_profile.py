from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.stage_web_release_artifacts import sha256, stage


ROOT = Path(__file__).resolve().parents[1]


class WebReleaseProfileTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.primary = self.root / "primary"
        self.fast = self.root / "fast"
        self.output = self.root / "distribution"
        for directory, compiled in ((self.primary, False), (self.fast, True)):
            directory.mkdir()
            values = {"CMAKE_PROJECT_VERSION": "1.2.0", "LW_DEPLOY_ONNX_MODELS": "ON",
                      "LW_WASM_SIMD128": "ON", "LW_WEB_PDF": "ON", "LW_WEB_LEGACY": "OFF",
                      "LW_WASM_COMPILED_REC": "ON" if compiled else "OFF",
                      "LW_WASM_REC_LAZY_FALLBACK": "ON", "LW_WASM_REC_SIMD_KERNELS": "ON",
                      "LW_WASM_COMPILED_DET": "ON", "LW_WASM_POINTWISE_ROWS": "2",
                      "LW_WASM_FINE_REC_WIDTHS": "OFF",
                      "LW_EXPERIMENTAL_REC_FFN_TILING": "OFF",
                      "LW_EXPERIMENTAL_CTC_TILED": "OFF"}
            (directory / "CMakeCache.txt").write_text(
                "".join(f"{name}:STRING={value}\n" for name, value in values.items()), encoding="utf-8")
            for name in ("lw-ppocr.js", "ocr-demo.html", "lw-ppocr-v6-small.js",
                         "ocr-demo-small.html", "lw-ppocr-v6-medium.js", "ocr-demo-medium.html"):
                (directory / name).write_bytes(directory.name.encode())
        self.archive = self.primary / "lw.PPOCR.C-1.2.0-node-wasm.zip"
        self.archive.write_bytes(b"canonical Node package")
        self.sidecar = Path(str(self.archive) + ".sha256")
        self.sidecar.write_text(f"{sha256(self.archive)}  {self.archive.name}\n", encoding="utf-8")

    def stage(self, variants=True):
        return stage(self.primary, self.output, "1.2.0", self.fast, variants)

    def test_exact_profile_selection_and_flat_artifact(self):
        report = self.stage()
        self.assertEqual(len(report["files"]), 8)
        self.assertEqual(report["configurations"]["compiled"]["LW_WASM_POINTWISE_ROWS"], "2")
        for name, record in report["files"].items():
            fast = "medium" not in name and "node-wasm" not in name
            self.assertEqual(record["profile"], "compiled" if fast else "primary")
            self.assertEqual(record["sha256"], sha256(self.output / name))
        self.assertEqual((self.output / "ocr-demo-small.html").read_bytes(), b"fast")
        self.assertEqual((self.output / "ocr-demo-medium.html").read_bytes(), b"primary")

    def test_push_profile_omits_medium(self):
        report = self.stage(False)
        self.assertEqual(len(report["files"]), 6)
        self.assertNotIn("ocr-demo-medium.html", report["files"])

    def test_wrong_configuration_rejected_before_copy(self):
        original = (self.fast / "CMakeCache.txt").read_text(encoding="utf-8")
        for name in ("LW_WASM_COMPILED_REC", "LW_WASM_REC_LAZY_FALLBACK",
                     "LW_WASM_REC_SIMD_KERNELS", "LW_WASM_COMPILED_DET",
                     "LW_WASM_SIMD128", "LW_WASM_FINE_REC_WIDTHS", "LW_WEB_LEGACY",
                     "LW_EXPERIMENTAL_REC_FFN_TILING", "LW_EXPERIMENTAL_CTC_TILED"):
            with self.subTest(name=name):
                text = original.replace(f"{name}:STRING=ON", f"{name}:STRING=OFF") if f"{name}:STRING=ON" in original else original.replace(f"{name}:STRING=OFF", f"{name}:STRING=ON")
                (self.fast / "CMakeCache.txt").write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, name):
                    self.stage()
                self.assertFalse(self.output.exists())
        (self.fast / "CMakeCache.txt").write_text(original, encoding="utf-8")

    def test_wrong_tile_version_and_canonical_profile_rejected(self):
        path = self.fast / "CMakeCache.txt"
        original = path.read_text(encoding="utf-8")
        for old, new, message in (("ROWS:STRING=2", "ROWS:STRING=4", "2x16"),
                                  ("VERSION:STRING=1.2.0", "VERSION:STRING=1.1.0", "version mismatch")):
            path.write_text(original.replace(old, new), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                self.stage()
        path.write_text(original, encoding="utf-8")
        path = self.primary / "CMakeCache.txt"
        original = path.read_text(encoding="utf-8")
        path.write_text(original.replace("COMPILED_REC:STRING=OFF", "COMPILED_REC:STRING=ON"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "LW_WASM_COMPILED_REC"):
            self.stage()

    def test_missing_empty_stale_and_tampered_files_rejected(self):
        (self.fast / "ocr-demo.html").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "missing or empty"):
            self.stage()
        (self.fast / "ocr-demo.html").write_bytes(b"fast")
        self.archive.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.stage()
        (self.primary / "lw.PPOCR.C-1.1.0-node-wasm.zip").write_bytes(b"stale")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.stage()

    def test_nonempty_destination_never_overwritten(self):
        self.output.mkdir()
        (self.output / "user-file").write_bytes(b"preserve")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            self.stage()
        self.assertEqual((self.output / "user-file").read_bytes(), b"preserve")

    def test_release_workflow_uses_model_specific_profile(self):
        release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        options = release.split("  wasm:", 1)[1].split("  android:", 1)[0]
        self.assertIn("release_web_profile: true", options)
        self.assertIn("build_web_variants: true", options)
        self.assertNotIn("compiled_rec:", options)
        workflow = (ROOT / ".github/workflows/wasm-html.yml").read_text(encoding="utf-8")
        self.assertIn("--target lw-ocr-html lw-ocr-html-small", workflow)
        self.assertIn("build/web-distribution/ocr-demo-medium.html", workflow)
        self.assertIn("--fast-dir build-wasm-compiled", workflow)
        self.assertIn("--golden ci/web-ppocrv6-small.json", workflow)


if __name__ == "__main__":
    unittest.main()
