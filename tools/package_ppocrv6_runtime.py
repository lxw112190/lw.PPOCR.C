#!/usr/bin/env python3
"""Create a deterministic, self-contained PP-OCRv6 LWM model pack."""
from __future__ import annotations
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

ASSET_NAMES = ("det.lwm", "cls.lwm", "rec.lwm", "ppocr_keys.txt")
SCHEMA_VERSION = 1
DEFAULT_LWM_VERSION = "0.1"
DEFAULT_RUNTIME_VERSION = "0.2.0-preview.1"
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def normalize_runtime_version(value: str) -> str:
    return value[1:] if value.startswith("v") else value
def asset_set_id(variant: str, runtime_version: str, hashes: dict[str, str]) -> str:
    payload = {"variant": variant, "runtime_version": runtime_version, "assets": {name: hashes[name] for name in ASSET_NAMES}}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]

def build_manifest(variant: str, runtime_version: str, lwm_version: str, hashes: dict[str, str]) -> dict[str, Any]:
    runtime_version = normalize_runtime_version(runtime_version)
    return {
        "schema_version": SCHEMA_VERSION,
        "family": "PP-OCRv6",
        "variant": variant,
        "model_id": f"ppocrv6-{variant}",
        "model_revision": runtime_version,
        "asset_set_id": asset_set_id(variant, runtime_version, hashes),
        "runtime_status": "preview",
        "lwm_format_version": lwm_version,
        "minimum_runtime_version": runtime_version,
        "models": {"det": "det.lwm", "cls": "cls.lwm", "rec": "rec.lwm"},
        "dictionary": "ppocr_keys.txt",
        "checksums": hashes,
        "recommended": {"rec": {"adaptive_width": True, "max_width": 960}},
    }

def _zip_member(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    info.extra = b""
    return info

def package(input_dir: Path, output_path: Path, variant: str, runtime_version: str = DEFAULT_RUNTIME_VERSION, lwm_version: str = DEFAULT_LWM_VERSION) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"model input directory does not exist: {input_dir}")
    assets: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    for name in ASSET_NAMES:
        path = input_dir / name
        if not path.is_file():
            raise ValueError(f"model pack is missing required asset: {path}")
        assets[name] = path.read_bytes()
        hashes[name] = sha256(path)
    manifest = build_manifest(variant, runtime_version, lwm_version, hashes)
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    checksum_lines = [f"{hashes[name]}  {name}" for name in ASSET_NAMES]
    checksum_lines.append(f"{sha256_bytes(manifest_bytes)}  manifest.json")
    checksums_bytes = ("\n".join(checksum_lines) + "\n").encode("ascii")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    members = {**assets, "manifest.json": manifest_bytes, "SHA256SUMS": checksums_bytes}
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(members):
            archive.writestr(_zip_member(name), members[name])
    return {"status": "ok", "variant": variant, "asset_set_id": manifest["asset_set_id"], "output": str(output_path), "members": sorted(members)}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("tiny", "small", "medium"), required=True)
    parser.add_argument("--runtime-version", default=DEFAULT_RUNTIME_VERSION)
    parser.add_argument("--lwm-version", default=DEFAULT_LWM_VERSION)
    args = parser.parse_args(argv)
    print(json.dumps(package(args.input_dir, args.output, args.variant, args.runtime_version, args.lwm_version), ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())