#!/usr/bin/env python3
"""Run the checked-in PP-OCRv6 Medium analysis validation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.resolve_medium_validation_contract import resolve_contract


LINE_RE = re.compile(r"^\d+ text=(?P<text>.*?) rec=")


def configure_utf8_output() -> None:
    """Keep Windows CI logs from failing when OCR output contains Unicode."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recognized_text(stdout: str) -> list[str]:
    return [
        match.group("text")
        for line in stdout.splitlines()
        if (match := LINE_RE.match(line))
    ]


def run(
    label: str,
    command: list[str],
    cwd: Path,
    report_path: Path | None = None,
) -> str:
    print(f"[medium] {label}: {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(completed.stdout, encoding="utf-8", newline="\n")
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")
    return completed.stdout


def executable(build_dir: Path, name: str) -> Path:
    candidates = [build_dir / name]
    if sys.platform == "win32":
        candidates.insert(0, build_dir / "Release" / f"{name}.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"build executable not found: {name} under {build_dir}")


def command_gate_arguments(gate: dict[str, float]) -> list[str]:
    return [
        "--error-threshold",
        format(gate["error_threshold"], ".17g"),
        "--max-abs-error",
        format(gate["max_abs_error"], ".17g"),
        "--max-mean-abs-error",
        format(gate["max_mean_abs_error"], ".17g"),
        "--max-fraction-over",
        format(gate["max_fraction_over"], ".17g"),
    ]


def asset_summary(root: Path, resolved: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for role, key in (
        ("detector", "detector_path"),
        ("classifier", "classifier_path"),
        ("recognizer", "recognizer_path"),
        ("dictionary", "dictionary_path"),
    ):
        relative = Path(resolved[key])
        path = (root / relative).resolve()
        result[role] = {
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return result


def main() -> int:
    configure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-version", default="0.2.0-preview.1")
    parser.add_argument("--expected-full-text-sha256-override")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved = resolve_contract(
        args.contract,
        args.expected_full_text_sha256_override,
    )
    build = args.build_dir.resolve()
    detector = (root / resolved["detector_path"]).resolve()
    classifier = (root / resolved["classifier_path"]).resolve()
    recognizer = (root / resolved["recognizer_path"]).resolve()
    dictionary = (root / resolved["dictionary_path"]).resolve()
    det_driver = executable(build, "det-graph-driver")
    rec_driver = executable(build, "rec-graph-driver")
    ocr_driver = executable(build, "lw-ocr-ppm")
    cls_lwm = build / "models" / "cls.lwm"
    sample = build / "models" / "sample.ppm"
    if not cls_lwm.is_file() or not sample.is_file():
        raise FileNotFoundError(
            "lw-ocr-ppm build assets are missing: expected build/models/cls.lwm "
            "and build/models/sample.ppm"
        )

    (output / "resolved-contract.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    assets = asset_summary(root, resolved)
    with tempfile.TemporaryDirectory(prefix="medium-validation-", dir=output) as directory:
        work = Path(directory)
        det_lwm = work / "medium-det-dynamic.lwm"
        rec_lwm = work / "medium-rec-dynamic.lwm"
        run(
            "DET dynamic conversion",
            [
                sys.executable,
                "tools/convert_medium_det_experimental.py",
                "--model",
                str(detector),
                "--dynamic",
                "--output",
                str(det_lwm),
                "--report",
                str(output / "medium-det-conversion.json"),
            ],
            root,
        )
        det_prefix = "medium-det-lwm-"
        for height, width in resolved["det_shapes"]:
            run(
                f"DET graph {height}x{width}",
                [
                    str(det_driver),
                    str(det_lwm),
                    str(height),
                    str(width),
                    str(work / f"{det_prefix}{height}x{width}.f32"),
                ],
                root,
            )
        det_command = [
            sys.executable,
            "tools/compare_small_det_execution.py",
            "--model",
            str(detector),
            "--output-dir",
            str(work),
            "--output-prefix",
            det_prefix,
            "--label",
            "Medium DET",
        ]
        for height, width in resolved["det_shapes"]:
            det_command.extend(["--height", str(height), "--width", str(width)])
        det_command.extend(command_gate_arguments(resolved["numerical_gates"]["det"]))
        run(
            "DET numerical gate",
            det_command,
            root,
            output / "medium-det-numerical.json",
        )

        run(
            "REC dynamic conversion",
            [
                sys.executable,
                "tools/convert_medium_rec_experimental.py",
                "--model",
                str(recognizer),
                "--dynamic",
                "--output",
                str(rec_lwm),
                "--report",
                str(output / "medium-rec-conversion.json"),
            ],
            root,
        )
        rec_prefix = "medium-rec-dynamic-w"
        for width in resolved["rec_widths"]:
            run(
                f"REC graph width {width}",
                [
                    str(rec_driver),
                    str(rec_lwm),
                    str(width),
                    str(work / f"{rec_prefix}{width}.f32"),
                ],
                root,
            )
        rec_command = [
            sys.executable,
            "tools/compare_small_rec_execution.py",
            "--model",
            str(recognizer),
            "--output-dir",
            str(work),
            "--output-prefix",
            rec_prefix,
            "--label",
            "Medium REC",
        ]
        for width in resolved["rec_widths"]:
            rec_command.extend(["--width", str(width)])
        rec_command.extend(command_gate_arguments(resolved["numerical_gates"]["rec"]))
        run(
            "REC numerical gate",
            rec_command,
            root,
            output / "medium-rec-numerical.json",
        )

        stdout = run(
            "full OCR sample",
            [
                str(ocr_driver),
                str(det_lwm),
                str(cls_lwm),
                str(rec_lwm),
                str(dictionary),
                str(sample),
                str(resolved["rec_max_width"]),
            ],
            root,
            output / "full-ocr.txt",
        )

        converted_dir = output / "converted-runtime"
        converted_dir.mkdir(exist_ok=True)
        shutil.copyfile(det_lwm, converted_dir / "det.lwm")
        shutil.copyfile(cls_lwm, converted_dir / "cls.lwm")
        shutil.copyfile(rec_lwm, converted_dir / "rec.lwm")
        shutil.copyfile(dictionary, converted_dir / "ppocr_keys.txt")
        pack_dir = output / "runtime-model"
        run("canonical runtime staging", [sys.executable, "tools/prepare_ppocrv6_runtime_variant.py", "--variant", "medium", "--build-dir", str(build), "--output-dir", str(pack_dir), "--converted-dir", str(converted_dir)], root)
        pack = output / "ppocrv6-medium-runtime.zip"
        run("runtime model pack", [sys.executable, "tools/package_ppocrv6_runtime.py", "--input-dir", str(pack_dir), "--variant", "medium", "--runtime-version", args.runtime_version, "--output", str(pack)], root)
        run("runtime model pack validation", [sys.executable, "tools/validate_runtime_model_pack.py", str(pack)], root)

    match = re.search(r"(?m)^lines=(\d+)\s", stdout)
    if match is None or int(match.group(1)) != resolved["expected_lines"]:
        raise RuntimeError(
            f"Medium full OCR returned an unexpected line count:\n{stdout}"
        )
    lines = recognized_text(stdout)
    if len(lines) != resolved["expected_lines"]:
        raise RuntimeError(
            f"Medium full OCR parsed {len(lines)} lines, "
            f"expected {resolved['expected_lines']}"
        )
    text_sha256 = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    if text_sha256 != resolved["expected_full_text_sha256"]:
        raise RuntimeError(
            "Medium full OCR text SHA-256 mismatch: "
            f"{text_sha256} != {resolved['expected_full_text_sha256']}"
        )

    summary = {
        "schema_version": 1,
        "status": "validated-analysis-only",
        "variant": "ppocrv6-medium",
        "contract_sha256": resolved["contract_sha256"],
        "model_catalog_sha256": resolved["model_catalog_sha256"],
        "assets": assets,
        "det_shapes": resolved["det_shapes"],
        "rec_widths": resolved["rec_widths"],
        "numerical_gates": resolved["numerical_gates"],
        "full_ocr": {
            "rec_max_width": resolved["rec_max_width"],
            "lines": len(lines),
            "first_text": lines[0],
            "text_sha256": text_sha256,
        },
        "temporary_tensors_uploaded": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
