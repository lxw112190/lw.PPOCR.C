#!/usr/bin/env python3
"""Validate a self-contained PP-OCRv6 LWM model pack."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.package_ppocrv6_runtime import (
    ASSET_NAMES,
    DEFAULT_MINIMUM_RUNTIME_VERSION,
    LWM_VERSION_RE,
    SCHEMA_VERSION,
    asset_set_id,
    normalize_runtime_version,
    runtime_status,
)


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
        if manifest.get("model_id") != f"ppocrv6-{variant}":
            raise ValueError("model pack model_id is invalid")
        try:
            model_revision = normalize_runtime_version(manifest.get("model_revision"))
            minimum_runtime_version = normalize_runtime_version(
                manifest.get("minimum_runtime_version", DEFAULT_MINIMUM_RUNTIME_VERSION)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"model pack version metadata is invalid: {error}") from error
        if manifest.get("model_revision") != model_revision:
            raise ValueError("model_revision must be normalized without a leading v")
        if manifest.get("minimum_runtime_version") != minimum_runtime_version:
            raise ValueError("minimum_runtime_version must be normalized without a leading v")
        if manifest.get("runtime_status") != runtime_status(model_revision):
            raise ValueError("runtime_status does not match model_revision")
        if not LWM_VERSION_RE.fullmatch(str(manifest.get("lwm_format_version", ""))):
            raise ValueError("lwm_format_version must use the form major.minor")
        if manifest.get("models") != {"det": "det.lwm", "cls": "cls.lwm", "rec": "rec.lwm"}:
            raise ValueError("model pack models mapping is invalid")
        if manifest.get("dictionary") != "ppocr_keys.txt":
            raise ValueError("model pack dictionary mapping is invalid")
        recommended = manifest.get("recommended")
        if not isinstance(recommended, dict):
            raise ValueError("model pack recommended metadata is missing")
        rec = recommended.get("rec")
        if not isinstance(rec, dict) or rec.get("adaptive_width") is not True:
            raise ValueError("model pack recommended.rec.adaptive_width must be true")
        if not isinstance(rec.get("max_width"), int) or rec["max_width"] not in {192, 320, 480, 640, 960}:
            raise ValueError("model pack recommended.rec.max_width is invalid")
        checksums = manifest.get("checksums")
        if not isinstance(checksums, dict) or set(checksums) != set(ASSET_NAMES):
            raise ValueError("manifest checksums must cover all runtime assets")
        for name in ASSET_NAMES:
            actual = _sha256(archive.read(name))
            if checksums[name] != actual:
                raise ValueError(f"manifest checksum mismatch for {name}")
        expected_id = asset_set_id(variant, model_revision, checksums)
        if manifest.get("asset_set_id") != expected_id:
            raise ValueError("manifest asset_set_id does not match the assets")
        checksum_lines = archive.read("SHA256SUMS").decode("ascii").splitlines()
        expected_lines = [f"{checksums[name]}  {name}" for name in ASSET_NAMES]
        expected_lines.append(f"{_sha256(archive.read('manifest.json'))}  manifest.json")
        if checksum_lines != expected_lines:
            raise ValueError("SHA256SUMS does not match manifest and assets")
        return {
            "status": "ok",
            "variant": variant,
            "model_revision": model_revision,
            "runtime_status": manifest["runtime_status"],
            "asset_set_id": manifest["asset_set_id"],
            "members": names,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(validate_pack(args.pack), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
