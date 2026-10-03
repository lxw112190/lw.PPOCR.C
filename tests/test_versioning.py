from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.check_release_readiness import check, load_json


ROOT = Path(__file__).resolve().parents[1]


class VersionConsistencyTest(unittest.TestCase):
    def project_version(self) -> str:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        match = re.search(
            r"^project\(lw\.PPOCR\.C VERSION ([0-9]+\.[0-9]+\.[0-9]+) LANGUAGES C\)$",
            cmake,
            re.MULTILINE,
        )
        self.assertIsNotNone(match)
        return match.group(1)

    def test_product_metadata_uses_the_cmake_version(self) -> None:
        version = self.project_version()
        self.assertEqual(version, "1.2.0")

        assembly = (
            ROOT / "examples" / "csharp-winforms" / "Properties" / "AssemblyInfo.cs"
        ).read_text(encoding="utf-8-sig")
        self.assertIn(f'AssemblyVersion("{version}.0")', assembly)
        self.assertIn(f'AssemblyFileVersion("{version}.0")', assembly)

        for relative in (
            "android/demo/build.gradle.kts",
            "android/demo-java/build.gradle.kts",
        ):
            gradle = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("versionCode = 4", gradle)
            self.assertIn(f'versionName = "{version}-preview.1"', gradle)

        sbom = json.loads((ROOT / "sbom.cdx.json").read_text(encoding="utf-8"))
        component = sbom["metadata"]["component"]
        self.assertEqual(component["version"], version)
        self.assertEqual(component["purl"], f"pkg:generic/lw.PPOCR.C@{version}")
        self.assertEqual(component["bom-ref"], component["purl"])
        dependency_refs = {item["ref"] for item in sbom["dependencies"]}
        self.assertIn(component["bom-ref"], dependency_refs)

    def test_android_workflow_does_not_request_removed_tools_package(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "android.yml").read_text(
            encoding="utf-8"
        )
        setup_start = workflow.index("uses: android-actions/setup-android@v3")
        pinned_start = workflow.index("- name: Install pinned SDK components")
        setup_block = workflow[setup_start:pinned_start]
        self.assertIn('packages: "platform-tools"', setup_block)
        self.assertNotIn('packages: "tools platform-tools"', setup_block)
        self.assertNotIn('"tools"', setup_block)

    def test_sbom_covers_official_onnx_and_shared_dictionary(self) -> None:
        sbom = json.loads((ROOT / "sbom.cdx.json").read_text(encoding="utf-8"))
        catalog = json.loads((ROOT / "models/ppocrv6-models.json").read_text(encoding="utf-8"))
        components = {item["bom-ref"]: item for item in sbom["components"]}
        dependencies = set(sbom["dependencies"][0]["dependsOn"])
        assets = [(f"lw-model:ppocrv6-{variant}-{kind}:official-onnx", value[kind])
                  for variant, value in catalog["variants"].items() for kind in ("det", "rec")]
        assets += [("lw-model:ppocrv6-shared-cls:official-onnx", catalog["shared_assets"]["cls"]),
                   ("lw-file:ppocrv6-small-medium-dictionary", catalog["shared_assets"]["small_rec_dictionary"])]
        for identity, asset in assets:
            with self.subTest(identity=identity):
                self.assertIn(identity, dependencies)
                component = components[identity]
                self.assertEqual(component["hashes"], [{"alg": "SHA-256", "content": asset["sha256"]}])
                self.assertEqual(component["licenses"][0]["license"]["id"], "Apache-2.0")
                self.assertIn({"name": "lw.asset.path", "value": "models/" + asset["path"]}, component["properties"])

    def test_runtime_contract_snapshot_matches_sources(self) -> None:
        snapshot = json.loads(
            (ROOT / "abi" / "runtime-contract-v1.json").read_text(encoding="utf-8")
        )
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        version = self.project_version()
        self.assertEqual(snapshot["schema_version"], 1)
        self.assertEqual(snapshot["product_version"], version)
        wasm_match = re.search(
            r'set\(LW_WASM_HOST_ABI_VERSION "([0-9]+)" CACHE STRING', cmake
        )
        lwm_match = re.search(
            r'set\(LW_LWM_FORMAT_VERSION "([0-9]+\.[0-9]+)" CACHE STRING', cmake
        )
        self.assertIsNotNone(wasm_match)
        self.assertIsNotNone(lwm_match)
        assert wasm_match is not None
        assert lwm_match is not None
        self.assertEqual(snapshot["wasm_host_abi_version"], int(wasm_match.group(1)))
        self.assertEqual(
            snapshot["wasm_host_abi_manifest"], "abi/web-abi-v1-candidate.json"
        )
        self.assertTrue((ROOT / snapshot["wasm_host_abi_manifest"]).is_file())
        self.assertEqual(snapshot["lwm_format_version"], lwm_match.group(1))

        candidate = json.loads(
            (ROOT / "abi" / "c-abi-v1-candidate.json").read_text(encoding="utf-8")
        )
        self.assertEqual(snapshot["c_abi"]["version"], candidate["abi_version"])
        self.assertEqual(snapshot["c_abi"]["status"], candidate["status"])
        self.assertEqual(
            snapshot["c_abi"]["candidate_manifest"], "abi/c-abi-v1-candidate.json"
        )
        self.assertEqual(snapshot["c_abi"]["layout_manifest"], candidate["layout_manifest"])
        self.assertEqual(snapshot["lwm_layout_manifest"], candidate["lwm_layout_manifest"])
        self.assertEqual(snapshot["c_abi"]["symbol_allowlist"], candidate["stable_symbols"])
        self.assertTrue((ROOT / snapshot["orientation_contract"]).is_file())
        self.assertTrue((ROOT / snapshot["lwm_layout_manifest"]).is_file())
        lwm_layout = json.loads((ROOT / "abi" / "lwm-v0.1-layout.json").read_text(encoding="utf-8"))
        self.assertEqual(lwm_layout["format"], "LWM")
        self.assertEqual(
            f'{lwm_layout["format_major"]}.{lwm_layout["format_minor"]}',
            snapshot["lwm_format_version"],
        )

    def test_stable_release_scope_is_explicit(self) -> None:
        scope = json.loads(
            (ROOT / "ci" / "stable-release-scope.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(scope["schema_version"], 1)
        self.assertEqual(scope["release_version"], self.project_version())
        self.assertEqual(scope["status"], "approved")
        self.assertEqual(scope["lwm_policy"], "internal-preview")
        self.assertEqual(scope["stable_models"], ["tiny"])
        self.assertEqual(scope["preview_only_models"], ["small", "medium"])

    def test_stable_gate_only_requires_models_in_stable_scope(self) -> None:
        report = check("stable", self.project_version())
        self.assertEqual(report["blockers"], [])
        self.assertNotIn("stable release scope is not approved", report["blockers"])
        self.assertNotIn(
            "analysis-only model variants remain: medium, small",
            report["blockers"],
        )

    def test_preview_tools_and_release_example_share_the_version_base(self) -> None:
        version = self.project_version()
        packager = (ROOT / "tools" / "package_ppocrv6_runtime.py").read_text(
            encoding="utf-8"
        )
        package_doc = (ROOT / "docs" / "package.md").read_text(encoding="utf-8")
        release_notes = (ROOT / "docs" / f"release-notes-v{version}.md").read_text(
            encoding="utf-8"
        )
        web_doc = (ROOT / "docs" / "web-sdk.md").read_text(encoding="utf-8")
        node_packager = (ROOT / "tools" / "package_node_wasm.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(f'DEFAULT_RUNTIME_VERSION = "{version}"', packager)
        self.assertIn('DEFAULT_MINIMUM_RUNTIME_VERSION = "1.0.0"', packager)
        self.assertIn("`v1.0.0` archive is the first ABI-frozen stable release", package_doc)
        self.assertIn(f"git tag -a v{version}", package_doc)
        self.assertNotIn(f"git tag -s v{version}", package_doc)
        self.assertIn("gh attestation verify", package_doc)
        self.assertIn(f"# lw.PPOCR.C v{version}", release_notes)
        self.assertIn("LW_WASM_COMPILED_REC", release_notes)
        self.assertIn("Small and Medium remain Preview", release_notes)
        self.assertIn(f'for example "{version}"', web_doc)
        self.assertIn("frozen contract", node_packager)

        for variant in ("tiny", "small", "medium"):
            validation = (ROOT / "tools" / f"run_{variant}_validation.py").read_text(
                encoding="utf-8"
            )
            self.assertIn(f'--runtime-version", default="{version}"', validation)

    def test_stale_scope_and_expected_version_are_rejected(self) -> None:
        def stale_scope(path, blockers):
            value = load_json(path, blockers)
            if path.name == "stable-release-scope.json":
                value["release_version"] = "0.0.0"
            return value

        with patch("tools.check_release_readiness.load_json", side_effect=stale_scope):
            report = check("stable", self.project_version())
        self.assertIn(
            "stable release scope version does not match CMake", report["blockers"]
        )
        self.assertEqual(check("stable", "0.0.0")["status"], "blocked")

    def test_release_keeps_compiled_wasm_opt_in(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertRegex(
            cmake, r'option\(LW_WASM_COMPILED_REC\s+"[^"]+" OFF\)'
        )
        workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        wasm_job = workflow.split("  wasm:", 1)[1].split("  android:", 1)[0]
        self.assertNotIn("compiled_rec: true", wasm_job)
        self.assertIn("release_web_profile: true", wasm_job)

    def test_preview_release_documentation_matches_supported_outputs(self) -> None:
        version = self.project_version()
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        readme_zh = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
        platform_matrix = (ROOT / "docs" / "platform-matrix.md").read_text(
            encoding="utf-8"
        )
        c_api_doc = (ROOT / "docs" / "c-api.md").read_text(encoding="utf-8")
        abi_candidate_doc = (ROOT / "docs" / "c-abi-v1-candidate.md").read_text(
            encoding="utf-8"
        )
        java_readme = (ROOT / "examples" / "java-jni" / "README.md").read_text(
            encoding="utf-8"
        )
        java_readme_zh = (
            ROOT / "examples" / "java-jni" / "README.zh-CN.md"
        ).read_text(encoding="utf-8")

        self.assertIn(f"## Preparing stable release: v{version}", readme)
        self.assertIn(f"## 准备发布稳定版：v{version}", readme_zh)
        for token in ("Tiny", "Small", "Medium", "android-arm64.aar"):
            self.assertIn(token, readme)
        for token in ("Tiny", "Small", "Medium", "android-arm64.aar"):
            self.assertIn(token, readme_zh)
        self.assertIn("Do not mix binaries", readme)
        self.assertIn(f"The v{version} release preserves", c_api_doc)
        self.assertIn(f"v{version} preserves it unchanged", abi_candidate_doc)
        self.assertIn("不要混用不同 Release", readme_zh)

        for artifact in (
            "lw-ppocr-java-jni-windows-x64",
            "lw-ppocr-java-jni-linux-x64",
            "lw-ppocr-java-jni-macos-arm64",
        ):
            self.assertIn(artifact, java_readme)
            self.assertIn(artifact, java_readme_zh)
        self.assertIn("Desktop Java/JNI macOS ARM64", platform_matrix)


if __name__ == "__main__":
    unittest.main()
