#!/usr/bin/env python3
"""Compare default and experimental OCR dispatch builds on one workload."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
from typing import Any

SCHEMA_VERSION = 1


def run_benchmark(executable: pathlib.Path, arguments: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [str(executable), *arguments], check=True, capture_output=True, text=True
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"benchmark produced no JSON: {executable}")
    try:
        report = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"benchmark output is not JSON: {executable}") from exc
    if not isinstance(report, dict) or "ocr_ms" not in report:
        raise RuntimeError(f"benchmark JSON is missing ocr_ms: {executable}")
    return report


def number(report: dict[str, Any], *path: str) -> float:
    value: Any = report
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise RuntimeError(f"benchmark JSON is missing {'.'.join(path)}")
        value = value[key]
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"benchmark value is not numeric: {'.'.join(path)}")
    return float(value)


def validate_contract(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    baseline_checksum = baseline.get("output_checksum")
    candidate_checksum = candidate.get("output_checksum")
    baseline_lines = baseline.get("lines")
    candidate_lines = candidate.get("lines")
    if not isinstance(baseline_checksum, str) or not isinstance(candidate_checksum, str):
        raise RuntimeError("both benchmark reports must include output_checksum")
    if baseline_checksum != candidate_checksum or baseline_lines != candidate_lines:
        raise RuntimeError(
            "OCR output contract differs: "
            f"lines {baseline_lines} vs {candidate_lines}, "
            f"checksum {baseline_checksum} vs {candidate_checksum}"
        )


def collect(
    baseline_driver: pathlib.Path,
    candidate_driver: pathlib.Path,
    benchmark_args: list[str],
    repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_reports: list[dict[str, Any]] = []
    candidate_reports: list[dict[str, Any]] = []
    for repeat in range(repeats):
        ordered = (
            ((baseline_driver, baseline_reports), (candidate_driver, candidate_reports))
            if repeat % 2 == 0
            else ((candidate_driver, candidate_reports), (baseline_driver, baseline_reports))
        )
        for executable, reports in ordered:
            report = run_benchmark(executable, benchmark_args)
            reports.append(report)
        validate_contract(baseline_reports[-1], candidate_reports[-1])
    return baseline_reports, candidate_reports


def median_value(reports: list[dict[str, Any]], *path: str) -> float:
    return statistics.median(number(report, *path) for report in reports)


def summarize(
    baseline_reports: list[dict[str, Any]], candidate_reports: list[dict[str, Any]]
) -> dict[str, Any]:
    if not baseline_reports or len(baseline_reports) != len(candidate_reports):
        raise RuntimeError("baseline and candidate report counts must match")
    baseline_mean = median_value(baseline_reports, "ocr_ms", "mean")
    candidate_mean = median_value(candidate_reports, "ocr_ms", "mean")
    baseline_p95 = median_value(baseline_reports, "ocr_ms", "p95")
    candidate_p95 = median_value(candidate_reports, "ocr_ms", "p95")
    baseline_rss = median_value(baseline_reports, "peak_rss_bytes")
    candidate_rss = median_value(candidate_reports, "peak_rss_bytes")
    if min(baseline_mean, candidate_mean, baseline_p95, candidate_p95) <= 0.0:
        raise RuntimeError("benchmark latency values must be positive")
    validate_contract(baseline_reports[0], candidate_reports[0])
    return {
        "schema_version": SCHEMA_VERSION,
        "repeats": len(baseline_reports),
        "baseline": {
            "ocr_mean_ms": baseline_mean,
            "ocr_p95_ms": baseline_p95,
            "peak_rss_mib": baseline_rss / (1024.0 * 1024.0),
            "samples_ms": [number(report, "ocr_ms", "mean") for report in baseline_reports],
        },
        "candidate": {
            "ocr_mean_ms": candidate_mean,
            "ocr_p95_ms": candidate_p95,
            "peak_rss_mib": candidate_rss / (1024.0 * 1024.0),
            "samples_ms": [number(report, "ocr_ms", "mean") for report in candidate_reports],
        },
        "comparison": {
            "mean_speedup": baseline_mean / candidate_mean,
            "p95_speedup": baseline_p95 / candidate_p95,
            "mean_latency_change_percent": (candidate_mean / baseline_mean - 1.0) * 100.0,
            "rss_delta_mib": (candidate_rss - baseline_rss) / (1024.0 * 1024.0),
        },
        "contract": {
            "match": True,
            "checksum": baseline_reports[0]["output_checksum"],
            "lines": baseline_reports[0]["lines"],
        },
    }


def render_markdown(summary: dict[str, Any], title: str) -> str:
    baseline = summary["baseline"]
    candidate = summary["candidate"]
    comparison = summary["comparison"]
    contract = summary["contract"]
    return "\n".join(
        [
            f"# {title}",
            "",
            "| Profile | OCR mean (ms) | OCR P95 (ms) | Peak RSS (MiB) |",
            "|---|---:|---:|---:|",
            f"| Default AVX2 | {baseline['ocr_mean_ms']:.3f} | {baseline['ocr_p95_ms']:.3f} | {baseline['peak_rss_mib']:.3f} |",
            f"| Experimental AVX2+FMA | {candidate['ocr_mean_ms']:.3f} | {candidate['ocr_p95_ms']:.3f} | {candidate['peak_rss_mib']:.3f} |",
            "",
            f"Median repeats: **{summary['repeats']}**",
            f"Mean speedup: **{comparison['mean_speedup']:.3f}x**",
            f"P95 speedup: **{comparison['p95_speedup']:.3f}x**",
            f"Mean latency change: **{comparison['mean_latency_change_percent']:+.2f}%**",
            f"RSS delta: **{comparison['rss_delta_mib']:+.3f} MiB**",
            "",
            f"Checksum: `{contract['checksum']}`",
            f"Lines: `{contract['lines']}`",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-driver", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-driver", type=pathlib.Path, required=True)
    parser.add_argument("--det", type=pathlib.Path, required=True)
    parser.add_argument("--cls", type=pathlib.Path, required=True)
    parser.add_argument("--rec", type=pathlib.Path, required=True)
    parser.add_argument("--dictionary", type=pathlib.Path, required=True)
    parser.add_argument("--image", type=pathlib.Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target-width", type=int, default=960)
    parser.add_argument("--det-threads", type=int, default=0)
    parser.add_argument("--json-output", type=pathlib.Path)
    parser.add_argument("--markdown-output", type=pathlib.Path)
    parser.add_argument("--title", default="Default vs Experimental AVX2+FMA OCR")
    args = parser.parse_args(argv)
    if min(args.warmup, args.iterations, args.repeats, args.workers) <= 0:
        parser.error("warmup, iterations, repeats and workers must be positive")
    benchmark_args = [
        str(args.det), str(args.cls), str(args.rec), str(args.dictionary), str(args.image),
        str(args.warmup), str(args.iterations), str(args.workers), str(args.target_width),
    ]
    if args.det_threads:
        benchmark_args.append(str(args.det_threads))
    baseline, candidate = collect(
        args.baseline_driver, args.candidate_driver, benchmark_args, args.repeats
    )
    summary = summarize(baseline, candidate)
    markdown = render_markdown(summary, args.title)
    if args.json_output:
        args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.write_text(markdown, encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())