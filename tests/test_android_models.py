#!/usr/bin/env python3
"""Regression tests for the deterministic Android model manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "tools" / "prepare_android_models.py"
MODEL_NAMES = ("det.lwm", "cls.lwm", "rec.lwm", "ppocr_keys.txt")
MODEL_NAME = "PP-OCRv6 tiny"
RUNTIME_FORMAT = "LWM 0.1"


def expected_asset_set_id(files: dict[str, dict[str, object]]) -> str:
    canonical = {
        "model": MODEL_NAME,
        "runtime_format": RUNTIME_FORMAT,
        "files": {name: files[name] for name in sorted(files)},
    }
    payload = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_metadata(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


class AndroidModelManifestTest(unittest.TestCase):
    def prepare(self, build_models: Path, dictionary: Path, output: Path) -> dict:
        subprocess.run(
            [
                sys.executable,
                str(PREPARE),
                "--build-models",
                str(build_models),
                "--dictionary",
                str(dictionary),
                "--output",
                str(output),
            ],
            check=True,
            cwd=ROOT,
        )
        return json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    def test_manifest_identity_is_deterministic_and_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_models = root / "build-models"
            model_dir = build_models / "models"
            model_dir.mkdir(parents=True)
            dictionary = root / "ppocr_keys.txt"
            for index, name in enumerate(MODEL_NAMES):
                target = dictionary if name == "ppocr_keys.txt" else model_dir / name
                target.write_bytes(("fixture-" + str(index)).encode("ascii"))

            first = self.prepare(build_models, dictionary, root / "output-a")
            second = self.prepare(build_models, dictionary, root / "output-b")
            self.assertEqual(first, second)
            self.assertRegex(first["asset_set_id"], r"^[0-9a-f]{64}$")

            actual_files = {
                name: file_metadata(
                    dictionary if name == "ppocr_keys.txt" else model_dir / name
                )
                for name in MODEL_NAMES
            }
            self.assertEqual(first["files"], actual_files)
            self.assertEqual(first["asset_set_id"], expected_asset_set_id(actual_files))

            (model_dir / "det.lwm").write_bytes(b"fixture-mutated")
            mutated = self.prepare(build_models, dictionary, root / "output-c")
            self.assertNotEqual(first["asset_set_id"], mutated["asset_set_id"])


if __name__ == "__main__":
    unittest.main()
