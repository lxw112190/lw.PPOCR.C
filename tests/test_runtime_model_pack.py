from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

from tools.package_ppocrv6_runtime import (
    build_manifest,
    normalize_runtime_version,
    package,
    resolve_runtime_status,
)
from tools.validate_runtime_model_pack import validate_pack


class RuntimeModelPackTests(unittest.TestCase):
    def test_pack_is_self_contained_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "models"
            source.mkdir()
            for name, data in {
                "det.lwm": b"det",
                "cls.lwm": b"cls",
                "rec.lwm": b"rec",
                "ppocr_keys.txt": "中\n文\n".encode("utf-8"),
            }.items():
                (source / name).write_bytes(data)
            first, second = root / "first.zip", root / "second.zip"
            report = package(source, first, "small")
            package(source, second, "small")
            validation = validate_pack(first)
            self.assertEqual(validation["status"], "ok")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(report["variant"], "small")
            with zipfile.ZipFile(first) as archive:
                manifest = json.loads(archive.read("ppocrv6-small/manifest.json"))
                self.assertEqual(manifest["model_id"], "ppocrv6-small")
                self.assertEqual(manifest["model_revision"], "1.2.0")
                self.assertEqual(manifest["runtime_status"], "production")
                self.assertEqual(manifest["minimum_runtime_version"], "1.0.0")
                self.assertEqual(manifest["lwm_format_version"], "0.1")
                self.assertEqual(
                    sorted(archive.namelist()),
                    [
                        "ppocrv6-small/SHA256SUMS",
                        "ppocrv6-small/cls.lwm",
                        "ppocrv6-small/det.lwm",
                        "ppocrv6-small/manifest.json",
                        "ppocrv6-small/ppocr_keys.txt",
                        "ppocrv6-small/rec.lwm",
                    ],
                )
                manifest_bytes = archive.read("ppocrv6-small/manifest.json")
                self.assertEqual(
                    validation["manifest_sha256"],
                    hashlib.sha256(manifest_bytes).hexdigest(),
                )
                self.assertTrue(
                    archive.read("ppocrv6-small/SHA256SUMS")
                    .decode("ascii")
                    .endswith("  manifest.json\n")
                )

    def test_unsafe_member_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pack_path = Path(directory) / "unsafe.zip"
            with zipfile.ZipFile(pack_path, "w") as archive:
                archive.writestr("../manifest.json", b"{}")
            with self.assertRaisesRegex(ValueError, "unsafe member path"):
                validate_pack(pack_path)

    def test_duplicate_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pack_path = Path(directory) / "duplicate.zip"
            with zipfile.ZipFile(pack_path, "w") as archive:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    archive.writestr("ppocrv6-small/manifest.json", b"{}")
                    archive.writestr("ppocrv6-small/manifest.json", b"{}")
            with self.assertRaisesRegex(ValueError, "duplicate members"):
                validate_pack(pack_path)

    def test_production_revision_is_marked_production(self) -> None:
        hashes = {
            name: "0" * 64
            for name in ("det.lwm", "cls.lwm", "rec.lwm", "ppocr_keys.txt")
        }
        manifest = build_manifest("tiny", "v0.2.0", "0.1", hashes)
        self.assertEqual(manifest["model_revision"], "0.2.0")
        self.assertEqual(manifest["runtime_status"], "production")

    def test_analysis_only_status_is_preserved_on_stable_revision(self) -> None:
        hashes = {
            name: "0" * 64
            for name in ("det.lwm", "cls.lwm", "rec.lwm", "ppocr_keys.txt")
        }
        manifest = build_manifest(
            "small", "v1.0.0", "0.1", hashes, runtime_status_override="analysis-only"
        )
        self.assertEqual(manifest["model_revision"], "1.0.0")
        self.assertEqual(manifest["runtime_status"], "analysis-only")
        self.assertEqual(resolve_runtime_status("1.0.0", "analysis-only"), "analysis-only")

    def test_status_and_revision_must_agree(self) -> None:
        with self.assertRaises(ValueError):
            resolve_runtime_status("1.0.0-preview.1", "production")
        with self.assertRaises(ValueError):
            resolve_runtime_status("1.0.0", "preview")

    def test_asset_set_id_normalizes_revision_and_rejects_bad_hashes(self) -> None:
        hashes = {
            name: "a" * 64
            for name in ("det.lwm", "cls.lwm", "rec.lwm", "ppocr_keys.txt")
        }
        from tools.package_ppocrv6_runtime import asset_set_id

        self.assertEqual(asset_set_id("tiny", "v0.2.0", hashes), asset_set_id("tiny", "0.2.0", hashes))
        with self.assertRaises(ValueError):
            asset_set_id("tiny", "0.2.0", {**hashes, "det.lwm": "bad"})

    def test_runtime_version_rejects_unstable_labels(self) -> None:
        for value in ("", "latest", "main", "release", "foo", "vfoo", "1.2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_runtime_version(value)

    def test_missing_runtime_asset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for name in ("det.lwm", "cls.lwm", "rec.lwm"):
                (source / name).write_bytes(b"x")
            with self.assertRaises(ValueError):
                package(source, source / "model.zip", "medium")


if __name__ == "__main__":
    unittest.main()
