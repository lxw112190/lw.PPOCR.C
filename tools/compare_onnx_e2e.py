#!/usr/bin/env python3
"""Paired, fresh-process full OCR A/B on the reviewed official ONNX assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def collect_text(driver: Path, paths: list[Path], sample: Path, environment: dict,
                 expected_checksum: str) -> dict:
    """Untimed public CLI sideband, checked against the measured text buffer."""
    cli = driver.with_name("lw-ocr-ppm" + driver.suffix)
    completed = subprocess.run(
        [str(cli.resolve()), *map(str, paths), str(sample.resolve()), "960"],
        capture_output=True, text=True, encoding="utf-8", errors="strict",
        env=environment, timeout=600, check=True)
    texts = re.findall(r"^\d+ text=(.*?) rec=", completed.stdout, re.MULTILINE)
    checksum = 14695981039346656037
    for byte in ("\0".join(texts) + "\0").encode("utf-8"):
        checksum = ((checksum ^ byte) * 1099511628211) & ((1 << 64) - 1)
    if len(texts) != 16 or f"{checksum:016x}" != expected_checksum:
        raise ValueError("untimed OCR text does not match measured text buffer")
    return {"text_lines": texts,
            "text_sha256": hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()}


def binary_identity(driver: Path) -> dict:
    from tools.validate_model_catalog import sha256
    library = driver.parent / "lw_ppocr_c.dll"
    result = {"path": str(driver.resolve()), "sha256": sha256(driver)}
    if library.is_file():
        result["library_sha256"] = sha256(library)
    cache = driver.parent.parent / "CMakeCache.txt"
    if not cache.is_file():
        cache = driver.parent / "CMakeCache.txt"
    if cache.is_file():
        keys = {"LW_EXPERIMENTAL_AVX2_FAST_PATH", "LW_NATIVE_FINE_REC_WIDTHS",
                "LW_EXPERIMENTAL_REC_FFN_TILING", "LW_REC_FFN_TILE_PIXELS",
                "LW_REC_RESIDENT_WIDTHS", "CMAKE_GENERATOR", "CMAKE_BUILD_TYPE"}
        result["cmake"] = {line.split(":", 1)[0]: line.split("=", 1)[1]
                           for line in cache.read_text(encoding="utf-8").splitlines()
                           if ":" in line and "=" in line and line.split(":", 1)[0] in keys}
    return result


def compare(baseline: Path, candidate: Path, sample: Path, rounds: int,
            iterations: int, output: Path, models: list[str],
            report_text_differences: bool = False, include_text: bool = False) -> dict:
    if rounds < 1 or iterations < 1 or not models or len(set(models)) != len(models):
        raise ValueError("invalid comparison workload")
    # Resolve model paths through the same reviewed catalog used by packaging.
    from tools.validate_model_catalog import validate_catalog
    resolved = validate_catalog(ROOT / "models/ppocrv6-models.json")["resolved"]
    environment = {key: value for key, value in os.environ.items()
                   if not key.endswith("_PROFILE") and not key.endswith("_DEBUG")}
    cases = []
    identities = {label: binary_identity(driver)
                  for label, driver in (("baseline", baseline), ("candidate", candidate))}
    observed_text = {}
    quality = {}
    output.mkdir(parents=True, exist_ok=True)
    for variant in models:
        assets = resolved[variant]
        paths = [ROOT / "models" / assets[key] for key in ("det", "cls", "rec", "dictionary")]
        for workers in (1, 4):
            runs = {"baseline": [], "candidate": []}
            for replica in range(rounds):
                order = ("baseline", "candidate") if replica % 2 == 0 else ("candidate", "baseline")
                for label in order:
                    driver = baseline if label == "baseline" else candidate
                    print(f"{variant} workers={workers} pair={replica + 1}/{rounds} {label}",
                          flush=True)
                    completed = subprocess.run(
                        [str(driver.resolve()), *map(str, paths), str(sample.resolve()),
                         "2", str(iterations), str(workers), "960"],
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", env=environment, timeout=600, check=False)
                    stem = output / f"{variant}-{workers}-{replica + 1}-{label}"
                    stem.with_suffix(".stdout.log").write_text(completed.stdout, encoding="utf-8")
                    stem.with_suffix(".stderr.log").write_text(completed.stderr, encoding="utf-8")
                    if completed.returncode:
                        raise RuntimeError(f"OCR benchmark failed; see {stem}.stderr.log")
                    report = json.loads(completed.stdout.strip().splitlines()[-1])
                    if (report["workers"] != workers or report["rec_target_width"] != 960 or
                            report["lines"] != 16 or report["image_width"] != 500 or
                            report["image_height"] != 500 or report["warmup"] != 2 or
                            report["iterations"] != iterations or
                            not re.fullmatch(r"[0-9a-f]{16}", report["output_checksum"]) or
                            not math.isfinite(report["ocr_ms"]["mean"]) or
                            report["ocr_ms"]["mean"] <= 0 or report["peak_rss_bytes"] <= 0):
                        raise ValueError("full OCR benchmark contract mismatch")
                    identity = (variant, label)
                    previous = observed_text.setdefault(identity, report["output_checksum"])
                    if previous != report["output_checksum"]:
                        raise ValueError(f"{variant}/{label}: non-deterministic OCR output")
                    runs[label].append(report)
            checksums = {run["output_checksum"] for group in runs.values() for run in group}
            text_matches = len(checksums) == 1
            if not text_matches and not report_text_differences:
                raise ValueError(f"{variant}/{workers}: full text differs: {checksums}")
            base_ms = [run["ocr_ms"]["mean"] for run in runs["baseline"]]
            new_ms = [run["ocr_ms"]["mean"] for run in runs["candidate"]]
            ratios = [a / b for a, b in zip(base_ms, new_ms)]
            case = {
                "variant": variant, "workers": workers,
                "baseline_ms": statistics.median(base_ms),
                "candidate_ms": statistics.median(new_ms),
                "paired_speedup": statistics.median(ratios),
                "ratio_range": [min(ratios), max(ratios)],
                "baseline_peak_mib": statistics.median(
                    run["peak_rss_bytes"] / 1048576 for run in runs["baseline"]),
                "candidate_peak_mib": statistics.median(
                    run["peak_rss_bytes"] / 1048576 for run in runs["candidate"]),
                "text_matches": text_matches,
                "output_checksums": {label: group[0]["output_checksum"]
                                     for label, group in runs.items()}, "runs": runs,
                "asset_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for key, path in zip(("det", "cls", "rec", "dictionary"), paths)},
            }
            cases.append(case)
            # Preserve finished cases if a later model fails.
            (output / "report.json").write_text(json.dumps(
                {"schema_version": 1, "cases": cases}, indent=2) + "\n", encoding="utf-8")
        if include_text:
            quality[variant] = {
                label: collect_text(driver, paths, sample, environment,
                                    observed_text[(variant, label)])
                for label, driver in (("baseline", baseline), ("candidate", candidate))}
            old = quality[variant]["baseline"]["text_lines"]
            new = quality[variant]["candidate"]["text_lines"]
            quality[variant]["changed_lines"] = [
                {"index": index, "baseline": a, "candidate": b}
                for index, (a, b) in enumerate(zip(old, new)) if a != b]
    report = {"schema_version": 1, "target_width": 960, "use_cls": True,
              "rounds": rounds, "iterations": iterations,
              "sample_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
              "environment": {"platform": platform.platform(),
                              "processor": platform.processor(),
                              "logical_cpus": os.cpu_count()},
              "text_contract_pass": all(case["text_matches"] for case in cases),
              "report_text_differences": report_text_differences,
              "baseline_driver": str(baseline.resolve()),
              "candidate_driver": str(candidate.resolve()), "cases": cases}
    report["binaries"] = identities
    report["quality"] = quality
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Official ONNX full OCR A/B", "",
             "Same machine; CLS enabled; REC max width 960; two warm-ups; alternating fresh processes.",
             "", "| Model | Workers | Before ms | After ms | Paired speedup (range) | Before peak MiB | After peak MiB | Text |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for case in cases:
        lines.append(f"| {case['variant']} | {case['workers']} | {case['baseline_ms']:.2f} | "
                     f"{case['candidate_ms']:.2f} | {case['paired_speedup']:.3f}x "
                     f"({case['ratio_range'][0]:.3f}-{case['ratio_range'][1]:.3f}) | "
                     f"{case['baseline_peak_mib']:.1f} | {case['candidate_peak_mib']:.1f} | "
                     f"{'same' if case['text_matches'] else 'DIFFERENT'} |")
    lines += ["", ("All paired full-text checksums match." if report["text_contract_pass"]
                   else "Text differs: accuracy experiment, NOT an output-equivalent speedup claim."),
              "Peak includes model initialization and",
              "the benchmark's extra standalone detector. Timing is informational, not a CI gate."]
    for case in cases:
        if case["ratio_range"][0] < 1.0 < case["ratio_range"][1]:
            lines += ["", f"{case['variant']}/{case['workers']} workers: paired ratios "
                      "straddle 1.0; the direction of the latency change is unresolved."]
    for variant, detail in quality.items():
        for change in detail["changed_lines"]:
            lines += ["", f"{variant} line {change['index']}: "
                      f"`{change['baseline']}` -> `{change['candidate']}`"]
    markdown = "\n".join(lines) + "\n"
    (output / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=ROOT / "build/models/sample.ppm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3, choices=range(1, 11))
    parser.add_argument("--iterations", type=int, default=5, choices=range(1, 101))
    parser.add_argument("--models", nargs="+", choices=("tiny", "small", "medium"),
                        default=["tiny", "small", "medium"])
    parser.add_argument("--report-text-differences", action="store_true",
                        help="Report accuracy experiments; nondeterminism still fails")
    parser.add_argument("--collect-text", action="store_true",
                        help="Use sibling lw-ocr-ppm for untimed, checksum-verified text diffs")
    args = parser.parse_args()
    # Text differences require an explicit experimental-report switch.
    compare(args.baseline, args.candidate, args.sample, args.rounds,
            args.iterations, args.output, args.models, args.report_text_differences,
            args.collect_text)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
