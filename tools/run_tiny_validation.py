#!/usr/bin/env python3
"""Run the Tiny runtime model-pack validation pipeline on one native build."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

LINE_RE = re.compile(r"^\d+ text=(?P<text>.*?) rec=")


def configure_utf8_output() -> None:
    """Keep Windows CI logs from failing on UTF-8 OCR text."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def run(label: str, command: list[str], cwd: Path) -> str:
    print(f"[tiny] {label}: {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return completed.stdout


def executable(build_dir: Path, name: str) -> Path:
    candidates = [build_dir / name]
    if sys.platform == "win32":
        candidates = [build_dir / f"{name}.exe", build_dir / "Release" / f"{name}.exe", *candidates]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"build executable not found: {name} under {build_dir}")


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-version", default="0.2.0-preview.1")
    parser.add_argument("--expected-lines", type=int, default=16)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    build = args.build_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    models = build / "models"
    required = {
        "det.lwm": models / "det.lwm",
        "cls.lwm": models / "cls.lwm",
        "rec.lwm": models / "rec.lwm",
        "ppocr_keys.txt": root / "models" / "ppocrv6-tiny" / "ppocr_keys.txt",
        "sample.ppm": models / "sample.ppm",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Tiny validation assets are missing: " + ", ".join(missing))

    ocr = executable(build, "lw-ocr-ppm")
    stdout = run(
        "full OCR sample",
        [
            str(ocr),
            str(required["det.lwm"]),
            str(required["cls.lwm"]),
            str(required["rec.lwm"]),
            str(required["ppocr_keys.txt"]),
            str(required["sample.ppm"]),
        ],
        root,
    )
    line_match = re.search(r"(?m)^lines=(\d+)\s", stdout)
    if line_match is None or int(line_match.group(1)) != args.expected_lines:
        raise RuntimeError(
            f"Tiny full OCR sample did not return {args.expected_lines} lines:\n{stdout}"
        )
    text_lines = [
        match.group("text")
        for line in stdout.splitlines()
        if (match := LINE_RE.match(line))
    ]
    if len(text_lines) != args.expected_lines:
        raise RuntimeError(
            f"Tiny full OCR sample returned {len(text_lines)} parsed lines, "
            f"expected {args.expected_lines}"
        )
    if not text_lines or text_lines[0] != "纯臻营养护发素":
        raise RuntimeError(
            f"Tiny full OCR first line changed: {text_lines[0] if text_lines else '<missing>'}"
        )
    (output / "full-ocr.txt").write_text(stdout, encoding="utf-8", newline="\n")

    pack_dir = output / "runtime-model"
    run("canonical runtime staging", [
        sys.executable,
        "tools/prepare_ppocrv6_runtime_variant.py",
        "--variant",
        "tiny",
        "--build-dir",
        str(build),
        "--output-dir",
        str(pack_dir),
    ], root)
    pack = output / "ppocrv6-tiny-runtime.zip"
    run(
        "runtime model pack",
        [
            sys.executable,
            "tools/package_ppocrv6_runtime.py",
            "--input-dir",
            str(pack_dir),
            "--variant",
            "tiny",
            "--runtime-version",
            args.runtime_version,
            "--output",
            str(pack),
        ],
        root,
    )
    validation = run(
        "runtime model pack validation",
        [sys.executable, "tools/validate_runtime_model_pack.py", str(pack)],
        root,
    )
    (output / "pack-validation.json").write_text(validation, encoding="utf-8", newline="\n")
    summary = {
        "status": "ok",
        "variant": "tiny",
        "full_ocr_lines": args.expected_lines,
        "full_ocr_first_text": text_lines[0],
        "runtime_model_pack": pack.name,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
