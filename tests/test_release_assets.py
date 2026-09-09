from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.verify_release_assets import (
    ReleaseAssetError,
    release_plan,
    verify_release_assets,
)


class ReleaseAssetContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.assets = self.root / "assets"
        self.assets.mkdir()
        self.manifest = self.root / "release-assets.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_manifest(self, required_assets: list[dict[str, object]]) -> None:
        self.manifest.write_text(
            json.dumps(
                {"schema_version": 1, "required_assets": required_assets},
                indent=2,
            ),
            encoding="utf-8",
        )

    def write_asset(self, name: str, data: bytes = b"release") -> str:
        (self.assets / name).write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def write_sidecar(self, name: str, digest: str) -> None:
        (self.assets / f"{name}.sha256").write_text(
            f"{digest}  {name}\n", encoding="ascii"
        )

    def prepare_sidecar_contract(self, version: str = "0.2.0-preview.1") -> str:
        self.write_manifest(
            [{"name": "runtime-{version}.zip", "checksum": "sidecar"}]
        )
        name = f"runtime-{version}.zip"
        self.write_sidecar(name, self.write_asset(name))
        return name

    def test_valid_sidecar_and_version_placeholder(self) -> None:
        self.prepare_sidecar_contract()
        report = verify_release_assets(
            self.manifest, self.assets, "0.2.0-preview.1", strict=True
        )
        self.assertEqual(report["required_asset_count"], 1)
        self.assertEqual(report["verified_file_count"], 2)
        self.assertEqual(report["warnings"], [])

    def test_missing_asset_is_rejected(self) -> None:
        self.write_manifest(
            [{"name": "runtime-{version}.zip", "checksum": "sidecar"}]
        )
        with self.assertRaisesRegex(ReleaseAssetError, "missing"):
            verify_release_assets(self.manifest, self.assets, "0.2.0")

    def test_bad_checksum_is_rejected(self) -> None:
        name = self.prepare_sidecar_contract("0.2.0")
        self.write_sidecar(name, "0" * 64)
        with self.assertRaisesRegex(ReleaseAssetError, "SHA-256 mismatch"):
            verify_release_assets(self.manifest, self.assets, "0.2.0")

    def test_duplicate_manifest_name_is_rejected(self) -> None:
        entry = {"name": "runtime-{version}.zip", "checksum": "sidecar"}
        self.write_manifest([entry, entry])
        with self.assertRaisesRegex(ReleaseAssetError, "duplicate"):
            verify_release_assets(self.manifest, self.assets, "0.2.0")

    def test_extra_file_warns_in_non_strict_mode(self) -> None:
        self.prepare_sidecar_contract("0.2.0")
        (self.assets / "notes.txt").write_text("extra", encoding="utf-8")
        report = verify_release_assets(self.manifest, self.assets, "0.2.0")
        self.assertEqual(report["warnings"], ["unknown release file: notes.txt"])

    def test_extra_file_is_rejected_in_strict_mode(self) -> None:
        self.prepare_sidecar_contract("0.2.0")
        (self.assets / "notes.txt").write_text("extra", encoding="utf-8")
        with self.assertRaisesRegex(ReleaseAssetError, "unknown release files"):
            verify_release_assets(self.manifest, self.assets, "0.2.0", strict=True)

    def test_unknown_or_missing_placeholder_is_rejected(self) -> None:
        for template in (
            "runtime.zip",
            "runtime-{revision}.zip",
            "runtime-{version.zip",
            "runtime-{version!r}.zip",
        ):
            with self.subTest(template=template):
                self.write_manifest([{"name": template, "checksum": "sidecar"}])
                with self.assertRaisesRegex(ReleaseAssetError, "placeholder"):
                    verify_release_assets(self.manifest, self.assets, "0.2.0")

    def test_empty_asset_is_rejected(self) -> None:
        self.write_manifest(
            [{"name": "runtime-{version}.zip", "checksum": "sidecar"}]
        )
        name = "runtime-0.2.0.zip"
        digest = self.write_asset(name, b"")
        self.write_sidecar(name, digest)
        with self.assertRaisesRegex(ReleaseAssetError, "empty"):
            verify_release_assets(self.manifest, self.assets, "0.2.0")

    def test_repository_manifest_is_self_consistent(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        manifest = repository / "ci/release-assets.json"
        version = "0.2.0-preview.1"
        assets, checksum_groups = release_plan(manifest, version)
        for name in assets:
            self.write_asset(name, name.encode("utf-8"))
        for checksum_name, names in checksum_groups.items():
            content = "".join(
                f"{hashlib.sha256(name.encode('utf-8')).hexdigest()}  {name}\n"
                for name in sorted(names)
            )
            (self.assets / checksum_name).write_text(content, encoding="ascii")
        report = verify_release_assets(manifest, self.assets, version, strict=True)
        self.assertEqual(report["required_asset_count"], len(assets))
        self.assertEqual(
            report["verified_file_count"], len(assets | set(checksum_groups))
        )

    def test_repository_workflow_uses_manifest_and_full_tag_version(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        workflow = (repository / ".github/workflows/release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("tools/verify_release_assets.py", workflow)
        self.assertIn("--manifest ci/release-assets.json", workflow)
        self.assertIn("--strict", workflow)
        self.assertIn("BUILD_VERSION: ${{ steps.version.outputs.base_version }}", workflow)
        self.assertIn("VERSION: ${{ steps.version.outputs.version }}", workflow)
        self.assertNotIn(
            'if [[ "${VERSION}" == 0.* || "${VERSION}" == *-* ]]', workflow
        )
        self.assertNotIn('-ne 35', workflow)

    def test_shared_checksum_file_is_verified(self) -> None:
        checksum_template = "android-{version}.SHA256SUMS.txt"
        self.write_manifest(
            [
                {
                    "name": "android-{version}.aar",
                    "checksum": {"file": checksum_template},
                },
                {
                    "name": "android-{version}.apk",
                    "checksum": {"file": checksum_template},
                },
            ]
        )
        version = "0.2.0"
        aar = f"android-{version}.aar"
        apk = f"android-{version}.apk"
        aar_hash = self.write_asset(aar, b"aar")
        apk_hash = self.write_asset(apk, b"apk")
        checksum_name = checksum_template.format(version=version)
        (self.assets / checksum_name).write_text(
            f"{aar_hash}  {aar}\n{apk_hash}  {apk}\n", encoding="ascii"
        )
        report = verify_release_assets(
            self.manifest, self.assets, version, strict=True
        )
        self.assertEqual(report["verified_file_count"], 3)


if __name__ == "__main__":
    unittest.main()
