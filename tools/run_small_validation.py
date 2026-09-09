#!/usr/bin/env python3
"""Run the checked-in Small validation pipeline on one native build."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.stage_small_validation_bundle import create_bundle


LINE_RE = re.compile(r"^\d+ text=(?P<text>.*?) rec=")


def recognized_text(stdout: str) -> list[str]:
    return [match.group("text") for line in stdout.splitlines() if (match := LINE_RE.match(line))]


def recognized_text_sha256(stdout: str) -> str:
    return hashlib.sha256("\n".join(recognized_text(stdout)).encode("utf-8")).hexdigest()


def run(label: str, command: list[str], cwd: Path) -> str:
    print(f"[small] {label}: {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{completed.stdout}\n{completed.stderr}")
    return completed.stdout


def executable(build_dir: Path, name: str) -> Path:
    candidates = [build_dir / name]
    if sys.platform == "win32":
        candidates.insert(0, build_dir / "Release" / f"{name}.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"build executable not found: {name} under {build_dir}")


def repository_path(root: Path, path: Path) -> Path:
    resolved = path if path.is_absolute() else root / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"validation asset not found: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-version", default="0.2.0-preview.1")
    parser.add_argument(
        "--rec-max-width",
        type=int,
        choices=(192, 320, 480, 640, 960),
        default=960,
        help="REC width limit used by the complete OCR gate",
    )
    parser.add_argument(
        "--expected-full-text-sha256",
        help="optional SHA-256 of newline-joined recognized text",
    )
    parser.add_argument(
        "--expected-lines",
        type=int,
        default=16,
        help="expected number of lines in the complete OCR sample",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = args.build_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = create_bundle(
        repository_path(root, args.detector),
        repository_path(root, args.classifier),
        repository_path(root, args.recognizer),
        repository_path(root, args.dictionary),
        output / "model",
    )
    (output / "contract.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    run("model contract", [sys.executable, "tools/validate_model_contract.py", str(output / "model")], root)
    metadata = output / "small-rec-metadata.json"
    run(
        "REC metadata probe",
        [sys.executable, "tools/probe_rec_shape_metadata.py", "--model", str(output / "model/rec.onnx"), "--json-output", str(metadata)],
        root,
    )
    run("REC metadata contract", [sys.executable, "tools/validate_small_rec_dynamic_rule.py", str(metadata)], root)
    det_lwm = output / "small-det-dynamic.lwm"
    rec_lwm = output / "small-rec-dynamic.lwm"
    run(
        "DET conversion",
        [sys.executable, "tools/convert_small_det_experimental.py", "--model", str(output / "model/det.onnx"), "--height", "640", "--width", "640", "--dynamic", "--output", str(det_lwm)],
        root,
    )
    run(
        "REC conversion",
        [sys.executable, "tools/convert_small_rec_experimental.py", "--model", str(output / "model/rec.onnx"), "--dynamic", "--output", str(rec_lwm)],
        root,
    )
    det_outputs = output / "det-outputs"
    det_outputs.mkdir(exist_ok=True)
    det_driver = executable(build, "det-graph-driver")
    for height, width in ((320, 320), (640, 640), (640, 960)):
        run(
            f"DET graph {height}x{width}",
            [str(det_driver), str(det_lwm), str(height), str(width), str(det_outputs / f"small-det-lwm-{height}x{width}.f32")],
            root,
        )
    run(
        "DET numerical gate",
        [sys.executable, "tools/compare_small_det_execution.py", "--model", str(output / "model/det.onnx"), "--output-dir", str(det_outputs), "--height", "320", "--width", "320", "--height", "640", "--width", "640", "--height", "640", "--width", "960", "--error-threshold", "1e-4", "--max-abs-error", "1e-4", "--max-mean-abs-error", "1e-6", "--max-fraction-over", "1e-4"],
        root,
    )
    rec_outputs = output / "rec-outputs"
    rec_outputs.mkdir(exist_ok=True)
    rec_driver = executable(build, "rec-graph-driver")
    widths = (192, 320, 480, 640, 960)
    for width in widths:
        run(
            f"REC graph width {width}",
            [str(rec_driver), str(rec_lwm), str(width), str(rec_outputs / f"small-lwm-w{width}.f32")],
            root,
        )
    run(
        "REC numerical gate",
        [sys.executable, "tools/compare_small_rec_execution.py", "--model", str(output / "model/rec.onnx"), "--output-dir", str(rec_outputs), "--error-threshold", "1e-4", "--max-mean-abs-error", "1e-6", "--max-fraction-over", "1e-4"],
        root,
    )
    ocr = executable(build, "lw-ocr-ppm")
    sample = build / "models" / "sample.ppm"
    stdout = run(
        "full OCR sample",
        [
            str(ocr),
            str(det_lwm),
            str(build / "models" / "cls.lwm"),
            str(rec_lwm),
            str(output / "model/ppocr_keys.txt"),
            str(sample),
            str(args.rec_max_width),
        ],
        root,
    )
    match = re.search(r"(?m)^lines=(\d+)\s", stdout)
    if match is None or int(match.group(1)) != args.expected_lines:
        raise RuntimeError(
            f"full OCR sample did not return {args.expected_lines} lines:\n{stdout}"
        )
    text_lines = recognized_text(stdout)
    if len(text_lines) != args.expected_lines:
        raise RuntimeError(
            f"full OCR sample returned {len(text_lines)} parsed lines, "
            f"expected {args.expected_lines}"
        )
    text_sha256 = recognized_text_sha256(stdout)
    if args.expected_full_text_sha256 and text_sha256.lower() != args.expected_full_text_sha256.lower():
        raise RuntimeError(
            "full OCR text SHA-256 mismatch: "
            f"{text_sha256} != {args.expected_full_text_sha256}"
        )
    (output / "full-ocr.txt").write_text(stdout, encoding="utf-8", newline="\n")

    pack_dir = output / "runtime-model"
    pack_dir.mkdir(exist_ok=True)
    shutil.copyfile(det_lwm, pack_dir / "det.lwm")
    shutil.copyfile(build / "models" / "cls.lwm", pack_dir / "cls.lwm")
    shutil.copyfile(rec_lwm, pack_dir / "rec.lwm")
    shutil.copyfile(output / "model" / "ppocr_keys.txt", pack_dir / "ppocr_keys.txt")
    pack = output / "ppocrv6-small-runtime.zip"
    run("runtime model pack", [sys.executable, "tools/package_ppocrv6_runtime.py", "--input-dir", str(pack_dir), "--variant", "small", "--runtime-version", args.runtime_version, "--output", str(pack)], root)
    run("runtime model pack validation", [sys.executable, "tools/validate_runtime_model_pack.py", str(pack)], root)
    summary = {
        "status": "ok",
        "rec_widths": list(widths),
        "full_ocr_rec_max_width": args.rec_max_width,
        "det_shapes": [[320, 320], [640, 640], [640, 960]],
        "full_ocr_lines": args.expected_lines,
        "full_ocr_text_sha256": text_sha256,
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
