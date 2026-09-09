#!/usr/bin/env python3
"""Prepare one canonical PP-OCRv6 runtime model directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.validate_model_catalog import validate_catalog

ASSET_NAMES = ("det.lwm", "cls.lwm", "rec.lwm", "ppocr_keys.txt")
VARIANTS = ("tiny", "small", "medium")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def executable_model(build_dir: Path, name: str) -> Path:
    candidates = [build_dir / "models" / name]
    if sys.platform == "win32":
        candidates.insert(0, build_dir / "Release" / "models" / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"build model not found: {name} under {build_dir}")


def run_conversion(root: Path, command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "runtime model conversion failed:\n"
            + completed.stdout
            + completed.stderr
        )


def prepare(
    variant: str,
    build_dir: Path,
    output_dir: Path,
    repository_root: Path,
    converted_dir: Path | None = None,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported PP-OCRv6 variant: {variant}")
    build_dir = build_dir.resolve()
    output_dir = output_dir.resolve()
    catalog_path = repository_root / "models" / "ppocrv6-models.json"
    catalog = validate_catalog(catalog_path)
    assets = catalog["resolved"][variant]
    model_root = catalog_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = converted_dir.resolve() if converted_dir else None

    if source_dir is not None:
        source = {name: source_dir / name for name in ASSET_NAMES}
    elif variant == "tiny":
        source = {
            "det.lwm": executable_model(build_dir, "det.lwm"),
            "cls.lwm": executable_model(build_dir, "cls.lwm"),
            "rec.lwm": executable_model(build_dir, "rec.lwm"),
            "ppocr_keys.txt": model_root / assets["dictionary"],
        }
    else:
        source = {
            "cls.lwm": executable_model(build_dir, "cls.lwm"),
            "ppocr_keys.txt": model_root / assets["dictionary"],
        }
        detector = (model_root / assets["det"]).resolve()
        recognizer = (model_root / assets["rec"]).resolve()
        if variant == "small":
            run_conversion(
                repository_root,
                [
                    sys.executable,
                    "tools/convert_small_det_experimental.py",
                    "--model",
                    str(detector),
                    "--height",
                    "640",
                    "--width",
                    "640",
                    "--dynamic",
                    "--output",
                    str(output_dir / "det.lwm"),
                ],
            )
            run_conversion(
                repository_root,
                [
                    sys.executable,
                    "tools/convert_small_rec_experimental.py",
                    "--model",
                    str(recognizer),
                    "--dynamic",
                    "--output",
                    str(output_dir / "rec.lwm"),
                ],
            )
        else:
            run_conversion(
                repository_root,
                [
                    sys.executable,
                    "tools/convert_medium_det_experimental.py",
                    "--model",
                    str(detector),
                    "--dynamic",
                    "--output",
                    str(output_dir / "det.lwm"),
                ],
            )
            run_conversion(
                repository_root,
                [
                    sys.executable,
                    "tools/convert_medium_rec_experimental.py",
                    "--model",
                    str(recognizer),
                    "--dynamic",
                    "--output",
                    str(output_dir / "rec.lwm"),
                ],
            )
        source["det.lwm"] = output_dir / "det.lwm"
        source["rec.lwm"] = output_dir / "rec.lwm"

    for name in ASSET_NAMES:
        path = source[name]
        if not path.is_file():
            raise FileNotFoundError(f"runtime asset is missing: {path}")
        if path.resolve() != (output_dir / name).resolve():
            shutil.copyfile(path, output_dir / name)

    checksums = {name: sha256(output_dir / name) for name in ASSET_NAMES}
    report = {
        "schema_version": 1,
        "status": "ok",
        "family": "PP-OCRv6",
        "variant": variant,
        "output_dir": str(output_dir),
        "assets": checksums,
        "sources": {name: str(source[name]) for name in ASSET_NAMES},
    }
    (output_dir / "runtime-assets.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--converted-dir",
        type=Path,
        help="reuse an existing converted directory instead of running converters",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    report = prepare(args.variant, args.build_dir, args.output_dir, root, args.converted_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
