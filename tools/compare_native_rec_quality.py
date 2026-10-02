#!/usr/bin/env python3
"""Compare native REC shape policies against project-generated ground truth."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.compare_onnx_e2e import binary_identity
from tools.compare_ocr_dataset_reports import compare_reports
from tools.validate_model_catalog import validate_catalog, sha256


def compare_quality(old: dict, new: dict) -> dict:
    old_files = [image["file"] for image in old["images"]]
    new_files = [image["file"] for image in new["images"]]
    if not old_files or old_files != new_files:
        raise ValueError("quality reports must cover the identical ordered image selection")
    for report in (old, new):
        if report.get("schema_version") != 1 or report.get("status") != "ok" or \
                report["rec_max_width"] != 960:
            raise ValueError("invalid quality report contract")
        if report["dataset"].get("selected_images") != len(report["images"]) or \
                len(set(image["file"] for image in report["images"])) != len(report["images"]):
            raise ValueError("quality report image selection is incomplete or duplicated")
        if not report["dataset"].get("manifest_sha256"):
            raise ValueError("missing ground truth manifest identity")
        for key in ("detection_f1", "exact_reference_line_rate", "cer_on_matched_lines"):
            value = report["metrics"][key]
            if not math.isfinite(value) or value < 0 or \
                    (key != "cer_on_matched_lines" and value > 1):
                raise ValueError("non-finite or negative quality metric")
        if report["metrics"]["ground_truth_lines"] <= 0:
            raise ValueError("empty ground truth")
    if old["metrics"]["ground_truth_lines"] != new["metrics"]["ground_truth_lines"]:
        raise ValueError("ground truth line count differs")
    result = compare_reports(old, new)
    # Zero tolerance: do not obscure a quality regression with a wider budget.
    result["non_regression"] = all(
        result["overall"][key]["delta"] >= 0 for key in
        ("detection_f1", "exact_reference_line_rate")) and \
        result["overall"]["cer_on_matched_lines"]["delta"] <= 0
    return result


def run(args) -> dict:
    catalog = validate_catalog(ROOT / "models/ppocrv6-models.json")["resolved"]
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    drivers = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    identities = {label: binary_identity(driver) for label, driver in drivers.items()}
    for variant in args.models:
        assets = {key: ROOT / "models" / catalog[variant][key]
                  for key in ("det", "cls", "rec", "dictionary")}
        reports = {}
        for label, driver in drivers.items():
            print(f"quality {variant}/{label}: START", flush=True)
            output = args.output / f"{variant}-{label}.json"
            command = [sys.executable, "-u", str(ROOT / "tools/evaluate_ocr_dataset.py"),
                "--dataset", str(args.dataset.resolve()), "--driver", str(driver),
                "--detector", str(assets["det"]), "--classifier", str(assets["cls"]),
                "--recognizer", str(assets["rec"]), "--dictionary", str(assets["dictionary"]),
                "--rec-max-width", "960", "--model-name", f"{variant}-{label}",
                "--output", str(output.resolve())]
            if args.limit is not None:
                command += ["--limit", str(args.limit)]
            # Inherit progress output; remote jobs must not appear hung for minutes.
            environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            subprocess.run(command, check=True, timeout=1800, env=environment)
            reports[label] = json.loads(output.read_text(encoding="utf-8"))
        case = compare_quality(reports["baseline"], reports["candidate"])
        case["variant"] = variant
        case["asset_sha256"] = {key: sha256(path) for key, path in assets.items()}
        cases.append(case)
        report = {"schema_version": 1, "binaries": identities, "cases": cases,
                  "quality_pass": all(case["non_regression"] for case in cases),
                  "accuracy_gate_mode": "informational" if args.report_quality_differences
                                        else "strict",
                  "scope": "generated corpus; not a real-world quality guarantee"}
        (args.output / "quality.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Native REC width policy: project corpus quality", "",
             "| Model | Images | Exact before | Exact after | CER before | CER after | Non-regression |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for case in cases:
        exact = case["overall"]["exact_reference_line_rate"]
        cer = case["overall"]["cer_on_matched_lines"]
        lines.append(f"| {case['variant']} | {case['dataset']['selected_images']} | "
            f"{exact['baseline']:.3%} | {exact['candidate']:.3%} | "
            f"{cer['baseline']:.4%} | {cer['candidate']:.4%} | {case['non_regression']} |")
    lines += ["", "Corpus accuracy non-regression: " + ("PASS" if report["quality_pass"] else "FAILED"),
              "GT-matched CER uses NFC normalization; spaces and punctuation are preserved.",
              "These are quality measurements, not timing measurements. Single-image text",
              "correction approval does not waive corpus regression checks."]
    (args.output / "quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("tiny", "small", "medium"),
                        default=["tiny", "small", "medium"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report-quality-differences", action="store_true",
                        help="report experimental accuracy regressions without failing; "
                             "invalid/incompatible OCR reports still fail")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if len(set(args.models)) != len(args.models):
        parser.error("models must be unique")
    report = run(args)
    if not report["quality_pass"]:
        print("WARNING: experimental width policy regressed corpus quality; "
              "do not promote it to the production default.", flush=True)
    return 0 if report["quality_pass"] or args.report_quality_differences else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
