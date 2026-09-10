from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


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
        self.assertEqual(version, "0.2.0")

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
            self.assertIn("versionCode = 2", gradle)
            self.assertIn(f'versionName = "{version}-preview.1"', gradle)

        sbom = json.loads((ROOT / "sbom.cdx.json").read_text(encoding="utf-8"))
        component = sbom["metadata"]["component"]
        self.assertEqual(component["version"], version)
        self.assertEqual(component["purl"], f"pkg:generic/lw.PPOCR.C@{version}")
        self.assertEqual(component["bom-ref"], component["purl"])
        dependency_refs = {item["ref"] for item in sbom["dependencies"]}
        self.assertIn(component["bom-ref"], dependency_refs)

    def test_preview_tools_and_release_example_share_the_version_base(self) -> None:
        version = self.project_version()
        packager = (ROOT / "tools" / "package_ppocrv6_runtime.py").read_text(
            encoding="utf-8"
        )
        package_doc = (ROOT / "docs" / "package.md").read_text(encoding="utf-8")
        web_doc = (ROOT / "docs" / "web-sdk.md").read_text(encoding="utf-8")
        self.assertIn(f'DEFAULT_RUNTIME_VERSION = "{version}-preview.1"', packager)
        self.assertIn(f'DEFAULT_MINIMUM_RUNTIME_VERSION = "{version}"', packager)
        self.assertIn(f"`v{version}-preview.1` has already been published", package_doc)
        self.assertIn(f"git tag -s v{version}-preview.2", package_doc)
        self.assertIn(f"git verify-tag v{version}-preview.2", package_doc)
        self.assertIn("gh attestation verify", package_doc)
        self.assertIn(f'for example "{version}"', web_doc)

    def test_preview_release_documentation_matches_supported_outputs(self) -> None:
        version = self.project_version()
        preview = f"{version}-preview.1"
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        readme_zh = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
        platform_matrix = (ROOT / "docs" / "platform-matrix.md").read_text(
            encoding="utf-8"
        )
        java_readme = (ROOT / "examples" / "java-jni" / "README.md").read_text(
            encoding="utf-8"
        )
        java_readme_zh = (
            ROOT / "examples" / "java-jni" / "README.zh-CN.md"
        ).read_text(encoding="utf-8")

        self.assertIn(f"## Current preview: v{preview}", readme)
        self.assertIn(f"## 当前预览版：v{preview}", readme_zh)
        for token in ("Tiny", "Small", "Medium", "android-arm64.aar"):
            self.assertIn(token, readme)
        for token in ("Tiny", "Small", "Medium", "android-arm64.aar"):
            self.assertIn(token, readme_zh)
        self.assertIn("Do not mix binaries", readme)
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
