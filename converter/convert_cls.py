#!/usr/bin/env python3
"""Convert the bundled PP-OCRv6 tiny direction classifier to LWM v0.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from converter.lwm_v0 import convert_cls_model
except ModuleNotFoundError:  # Direct execution: python converter/convert_cls.py
    from lwm_v0 import convert_cls_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="input CLS ONNX model")
    parser.add_argument("--output", type=Path, required=True, help="output LWM file")
    parser.add_argument("--metadata-output", type=Path, help="optional conversion metadata JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    info = convert_cls_model(args.input, args.output)
    metadata = {
        "format": "LWM",
        "format_version": "0.1",
        "input": str(args.input),
        "output": str(args.output),
        **info.__dict__,
        "checksum": f"0x{info.checksum:016x}",
    }
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
