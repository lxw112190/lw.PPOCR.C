#!/usr/bin/env python3
"""Stage exact tested browser files in one flat artifact; keep Node/Medium canonical."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def cache(directory: Path) -> dict[str, str]:
    values = {}
    for line in (directory / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"([^#/:][^:]*):[^=]+=(.*)$", line)
        if match:
            values[match[1]] = match[2]
    return values


def enabled(value: str) -> bool:
    return value.upper() in {"ON", "TRUE", "YES", "1"}


def validate(directory: Path, compiled: bool, version: str) -> dict[str, str]:
    values = cache(directory)
    if values.get("CMAKE_PROJECT_VERSION") != version:
        raise ValueError(f"{directory}: runtime version mismatch")
    required = {"LW_DEPLOY_ONNX_MODELS": True, "LW_WASM_SIMD128": True,
                "LW_WEB_PDF": True, "LW_WEB_LEGACY": False,
                "LW_WASM_COMPILED_REC": compiled,
                "LW_WASM_FINE_REC_WIDTHS": False,
                "LW_EXPERIMENTAL_REC_FFN_TILING": False,
                "LW_EXPERIMENTAL_CTC_TILED": False}
    if compiled:
        required.update(LW_WASM_REC_LAZY_FALLBACK=True,
                        LW_WASM_REC_SIMD_KERNELS=True, LW_WASM_COMPILED_DET=True)
        if values.get("LW_WASM_POINTWISE_ROWS") != "2":
            raise ValueError(f"{directory}: release requires Pointwise 2x16")
    for name, expected in required.items():
        if name not in values or enabled(values[name]) != expected:
            raise ValueError(f"{directory}: {name} must be {expected}")
    result = {name: values[name] for name in sorted(required)}
    if compiled:
        result["LW_WASM_POINTWISE_ROWS"] = values["LW_WASM_POINTWISE_ROWS"]
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage(primary: Path, output: Path, version: str, fast: Path | None = None,
          include_variants: bool = False) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty (no stale artifacts)")
    configurations = {}
    if fast is not None:
        configurations["canonical"] = validate(primary, False, version)
        configurations["compiled"] = validate(fast, True, version)
    browser = fast or primary
    selected = {"lw-ppocr.js": browser, "ocr-demo.html": browser}
    if fast is not None or include_variants:
        selected.update({"lw-ppocr-v6-small.js": browser, "ocr-demo-small.html": browser})
    if include_variants:
        selected.update({"lw-ppocr-v6-medium.js": primary, "ocr-demo-medium.html": primary})
    archives = list(primary.glob("lw.PPOCR.C-*-node-wasm.zip"))
    expected_archive = f"lw.PPOCR.C-{version}-node-wasm.zip"
    if len(archives) != 1 or archives[0].name != expected_archive:
        raise ValueError("exactly one current-version Node archive is required")
    selected[expected_archive] = primary
    selected[expected_archive + ".sha256"] = primary
    for name, directory in selected.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty artifact: {path}")
    checksum = (primary / (expected_archive + ".sha256")).read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{64} [ *]" + re.escape(expected_archive), checksum):
        raise ValueError("invalid Node archive checksum sidecar")
    if checksum[:64].lower() != sha256(primary / expected_archive):
        raise ValueError("Node archive checksum mismatch")
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, directory in selected.items():
        source = directory / name
        digest = sha256(source)
        shutil.copyfile(source, output / name)
        if sha256(output / name) != digest:
            raise ValueError(f"staged bytes differ: {name}")
        records[name] = {"profile": "compiled" if directory == fast else "primary",
                         "sha256": digest, "bytes": source.stat().st_size}
    return {"schema_version": 1, "version": version,
            "configurations": configurations, "files": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", required=True, type=Path)
    parser.add_argument("--fast-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--include-variants", action="store_true")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = stage(args.primary_dir, args.output_dir, args.version,
                   args.fast_dir, args.include_variants)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
