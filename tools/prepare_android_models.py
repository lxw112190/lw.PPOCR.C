#!/usr/bin/env python3
"""Stage one deterministic LWM model set for the Android AAR."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

MODEL_FILES = ("det.lwm", "cls.lwm", "rec.lwm")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-models", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.build_models / "models"
    args.output.mkdir(parents=True, exist_ok=True)
    files = list(MODEL_FILES) + ["ppocr_keys.txt"]
    for name in files:
        source_path = source / name if name != "ppocr_keys.txt" else args.dictionary
        if not source_path.is_file():
            raise SystemExit(f"missing Android model asset: {source_path}")
        shutil.copyfile(source_path, args.output / name)
    manifest = {
        "schema_version": 1,
        "model": "PP-OCRv6 tiny",
        "runtime_format": "LWM 0.1",
        "files": {
            name: {
                "bytes": (args.output / name).stat().st_size,
                "sha256": sha256(args.output / name),
            }
            for name in files
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
