#!/usr/bin/env python3
"""Validate a self-contained PP-OCRv6 LWM model pack."""
from __future__ import annotations
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.package_ppocrv6_runtime import ASSET_NAMES, SCHEMA_VERSION, asset_set_id

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def validate_pack(pack_path: Path) -> dict[str, Any]:
    if not pack_path.is_file():
        raise ValueError(f"model pack does not exist: {pack_path}")
    with zipfile.ZipFile(pack_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("model pack contains duplicate members")
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError("model pack contains an unsafe member path")
        required = set(ASSET_NAMES) | {"manifest.json", "SHA256SUMS"}
        if set(names) != required:
            raise ValueError(f"model pack members must be exactly {sorted(required)}")
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported model pack schema_version")
        if manifest.get("family") != "PP-OCRv6":
            raise ValueError("model pack family must be PP-OCRv6")
        variant = manifest.get("variant")
        if variant not in {"tiny", "small", "medium"}:
            raise ValueError("model pack variant is invalid")
        if manifest.get("runtime_status") != "preview":
            raise ValueError("v0.2.0 model packs must be marked preview")
        if manifest.get("models") != {"det": "det.lwm", "cls": "cls.lwm", "rec": "rec.lwm"}:
            raise ValueError("model pack models mapping is invalid")
        if manifest.get("dictionary") != "ppocr_keys.txt":
            raise ValueError("model pack dictionary mapping is invalid")
        checksums = manifest.get("checksums")
        if not isinstance(checksums, dict) or set(checksums) != set(ASSET_NAMES):
            raise ValueError("manifest checksums must cover all runtime assets")
        for name in ASSET_NAMES:
            actual = _sha256(archive.read(name))
            if checksums[name] != actual:
                raise ValueError(f"manifest checksum mismatch for {name}")
        expected_id = asset_set_id(variant, manifest["model_revision"], checksums)
        if manifest.get("asset_set_id") != expected_id:
            raise ValueError("manifest asset_set_id does not match the assets")
        checksum_lines = archive.read("SHA256SUMS").decode("ascii").splitlines()
        expected_lines = [f"{checksums[name]}  {name}" for name in ASSET_NAMES]
        expected_lines.append(f"{_sha256(archive.read('manifest.json'))}  manifest.json")
        if checksum_lines != expected_lines:
            raise ValueError("SHA256SUMS does not match manifest and assets")
        return {"status": "ok", "variant": variant, "asset_set_id": manifest["asset_set_id"], "members": names}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(validate_pack(args.pack), ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())