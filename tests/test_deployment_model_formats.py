"""Regression tests for explicit ONNX/LWM deployment asset names."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from tools.prepare_ppocrv6_runtime_variant import prepare

ROOT = Path(__file__).resolve().parents[1]


class DeploymentModelFormatTests(unittest.TestCase):
    def test_onnx_staging_needs_no_converted_build_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_build = root / "no-build-models"
            with patch("tools.prepare_ppocrv6_runtime_variant.executable_model") as lookup, \
                    patch("tools.prepare_ppocrv6_runtime_variant.run_conversion") as convert:
                for variant in ("tiny", "small"):
                    with self.subTest(variant=variant):
                        output = root / variant
                        report = prepare(variant, missing_build, output, ROOT,
                                         model_format="onnx")
                        self.assertEqual(report["model_format"], "onnx")
                        self.assertEqual(set(report["assets"]),
                                         {"det.onnx", "cls.onnx", "rec.onnx", "ppocr_keys.txt"})
                        for name, digest in report["assets"].items():
                            self.assertEqual(digest, hashlib.sha256((output / name).read_bytes())
                                             .hexdigest())
                        self.assertFalse((output / "cls.lwm").exists())
                lookup.assert_not_called()
                convert.assert_not_called()
            self.assertFalse(missing_build.exists())

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for asset selection regression")
    def test_node_manifest_wins_over_stale_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime.cjs").write_text("""
module.exports = async function () {
  const heap = new Uint32Array(32);
  return {
    HEAPU32: heap,
    FS: {mkdir() {}, writeFile(name) {global.__modelFiles.push(name);}},
    _lw_web_init() {return 0;},
    _lw_web_malloc() {return 4;}, _lw_web_free() {},
    _lw_web_get_info() {heap.set([1, 1000, 361000, 60, 16], 1); return 0;}
  };
};
""", encoding="utf-8")
            for component in ("det", "cls", "rec"):
                for extension in ("onnx", "lwm"):
                    (root / (component + "." + extension)).write_bytes(b"fixture")
            (root / "ppocr_keys.txt").write_bytes(b"fixture")
            script = """
global.__modelFiles = [];
const smoke = require(process.argv[1]);
smoke.createRuntime(process.argv[2], true).then(() => {
  console.log(JSON.stringify(global.__modelFiles));
}).catch(error => {console.error(error); process.exitCode = 1;});
"""
            for extension in ("lwm", "onnx"):
                (root / "manifest.json").write_text(json.dumps({"assets": {
                    component: {"path": component + "." + extension}
                    for component in ("det", "cls", "rec")
                }}), encoding="utf-8")
                result = subprocess.run(["node", "-e", script,
                                         str(ROOT / "tests/node/smoke.cjs"), str(root)],
                                        check=True, capture_output=True, text=True)
                self.assertEqual(json.loads(result.stdout),
                                 ["/models/" + component + "." + extension
                                  for component in ("det", "cls", "rec")] +
                                 ["/models/ppocr_keys.txt"])
            # A staged runtime manifest takes priority over an old packaged manifest.
            (root / "runtime-assets.json").write_text(json.dumps({"assets": {
                component + ".lwm": {} for component in ("det", "cls", "rec")
            }}), encoding="utf-8")
            result = subprocess.run(["node", "-e", script,
                                     str(ROOT / "tests/node/smoke.cjs"), str(root)],
                                    check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout),
                             ["/models/" + component + ".lwm"
                              for component in ("det", "cls", "rec")] +
                             ["/models/ppocr_keys.txt"])

    def test_browser_and_node_pack_both_formats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime.cjs"
            runtime.write_text("var LwPpocrModule = function() {};", encoding="utf-8")
            dictionary = root / "keys.txt"
            dictionary.write_text("fixture", encoding="utf-8")
            for extension in ("onnx", "lwm"):
                models = {}
                for component in ("det", "cls", "rec"):
                    models[component] = root / (component + "." + extension)
                    models[component].write_bytes((component + extension).encode("ascii"))
                shared = [item for component in ("det", "cls", "rec")
                          for item in ("--" + component, str(models[component]))]
                sdk = root / (extension + ".js")
                subprocess.run([
                    sys.executable, str(ROOT / "web/package_ocr_sdk.py"),
                    "--template", str(ROOT / "web/lw_ppocr_sdk.template.js"),
                    "--runtime", str(runtime), *shared,
                    "--dictionary", str(dictionary), "--version", "test", "--output", str(sdk),
                ], check=True, capture_output=True)
                wrapper = sdk.read_text(encoding="utf-8")
                self.assertNotIn("__LW_", wrapper)
                self.assertEqual(wrapper.count('"det.' + extension + '"'), 2)
                output = root / ("node-" + extension)
                subprocess.run([
                    sys.executable, str(ROOT / "tools/package_node_wasm.py"),
                    "--runtime", str(runtime), *shared,
                    "--dictionary", str(dictionary), "--version", "test",
                    "--wasm-backend", "scalar", "--wasm-host-abi-version", "1",
                    "--lwm-version", "0.1", "--license", str(ROOT / "LICENSE"),
                    "--notices", str(ROOT / "THIRD-PARTY-NOTICES.md"),
                    "--model-license", str(ROOT / "licenses/PaddleOCR-models-APACHE-2.0.txt"),
                    "--output-dir", str(output), "--archive", str(root / (extension + ".zip")),
                ], check=True, capture_output=True)
                manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["runtime"]["modelFormat"], extension)
                for component in models:
                    asset = manifest["assets"][component]
                    self.assertEqual(asset["path"], component + "." + extension)
                    self.assertEqual(asset["sha256"],
                                     hashlib.sha256(models[component].read_bytes()).hexdigest())
                self.assertIn("det." + extension,
                              (output / "README.md").read_text(encoding="utf-8"))
                for line in (output / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
                    digest, name = line.split("  ", 1)
                    self.assertEqual(digest, hashlib.sha256((output / name).read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
