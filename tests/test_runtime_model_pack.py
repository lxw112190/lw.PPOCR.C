from __future__ import annotations
import tempfile
import unittest
import zipfile
from pathlib import Path
from tools.package_ppocrv6_runtime import package
from tools.validate_runtime_model_pack import validate_pack

class RuntimeModelPackTests(unittest.TestCase):
    def test_pack_is_self_contained_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "models"
            source.mkdir()
            for name, data in {"det.lwm": b"det", "cls.lwm": b"cls", "rec.lwm": b"rec", "ppocr_keys.txt": "中\n文\n".encode("utf-8")}.items():
                (source / name).write_bytes(data)
            first, second = root / "first.zip", root / "second.zip"
            report = package(source, first, "small")
            package(source, second, "small")
            self.assertEqual(validate_pack(first)["status"], "ok")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(report["variant"], "small")
            with zipfile.ZipFile(first) as archive:
                self.assertEqual(sorted(archive.namelist()), ["SHA256SUMS", "cls.lwm", "det.lwm", "manifest.json", "ppocr_keys.txt", "rec.lwm"])

    def test_missing_runtime_asset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for name in ("det.lwm", "cls.lwm", "rec.lwm"):
                (source / name).write_bytes(b"x")
            with self.assertRaises(ValueError):
                package(source, source / "model.zip", "medium")

if __name__ == "__main__":
    unittest.main()