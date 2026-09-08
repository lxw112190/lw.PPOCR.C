#!/usr/bin/env python3
"""Compare Compact and resident REC runtime benchmark JSON outputs."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
from typing import Any

SCHEMA_VERSION = 1


def run_benchmark(executable: pathlib.Path, arguments: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [str(executable), *arguments],
        check=True,
        capture_output=True,
        text=True,
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


def _number(report: dict[str, Any], *path: str) -> float:
    value: Any = report
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise RuntimeError(f"benchmark JSON is missing {'.'.join(path)}")
        value = value[key]
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"benchmark value is not numeric: {'.'.join(path)}")
    return float(value)


def summarize(compact: dict[str, Any], performance: dict[str, Any]) -> dict[str, Any]:
    compact_mean = _number(compact, "ocr_ms", "mean")
    performance_mean = _number(performance, "ocr_ms", "mean")
    compact_p95 = _number(compact, "ocr_ms", "p95")
    performance_p95 = _number(performance, "ocr_ms", "p95")
    compact_rss = _number(compact, "peak_rss_bytes")
    performance_rss = _number(performance, "peak_rss_bytes")
    if compact_mean <= 0.0 or performance_mean <= 0.0:
        raise RuntimeError("benchmark OCR means must be positive")

    compact_checksum = compact.get("output_checksum")
    performance_checksum = performance.get("output_checksum")
    compact_lines = compact.get("lines")
    performance_lines = performance.get("lines")
    if not isinstance(compact_checksum, str) or not isinstance(performance_checksum, str):
        raise RuntimeError("both benchmark reports must include output_checksum")
    if compact_lines != performance_lines or compact_checksum != performance_checksum:
        raise RuntimeError(
            "OCR output contract differs: "
            f"lines {compact_lines} vs {performance_lines}, "
            f"checksum {compact_checksum} vs {performance_checksum}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "compact": {
            "ocr_mean_ms": compact_mean,
            "ocr_p95_ms": compact_p95,
            "peak_rss_mib": compact_rss / (1024.0 * 1024.0),
        },
        "performance": {
            "ocr_mean_ms": performance_mean,
            "ocr_p95_ms": performance_p95,
            "peak_rss_mib": performance_rss / (1024.0 * 1024.0),
        },
        "comparison": {
            "mean_speedup": compact_mean / performance_mean,
            "mean_latency_change_percent": (performance_mean / compact_mean - 1.0) * 100.0,
            "p95_latency_change_percent": (performance_p95 / compact_p95 - 1.0) * 100.0,
            "rss_delta_mib": (performance_rss - compact_rss) / (1024.0 * 1024.0),
        },
        "contract": {
            "match": True,
            "compact_checksum": compact_checksum,
            "performance_checksum": performance_checksum,
            "compact_lines": compact_lines,
            "performance_lines": performance_lines,
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    compact = summary["compact"]
    performance = summary["performance"]
    comparison = summary["comparison"]
    contract = summary["contract"]
    return "\n".join(
        [
            "# Compact vs Performance OCR runtime",
            "",
            "| Profile | OCR mean (ms) | OCR P95 (ms) | Peak RSS (MiB) |",
            "|---|---:|---:|---:|",
            f"| Compact | {compact['ocr_mean_ms']:.3f} | {compact['ocr_p95_ms']:.3f} | {compact['peak_rss_mib']:.3f} |",
            f"| Performance | {performance['ocr_mean_ms']:.3f} | {performance['ocr_p95_ms']:.3f} | {performance['peak_rss_mib']:.3f} |",
            "",
            f"Mean speedup: **{comparison['mean_speedup']:.3f}x**",
            f"Mean latency change: **{comparison['mean_latency_change_percent']:+.2f}%**",
            f"P95 latency change: **{comparison['p95_latency_change_percent']:+.2f}%**",
            f"RSS delta: **{comparison['rss_delta_mib']:+.3f} MiB**",
            "",
            f"Checksum: `{contract['compact_checksum']}` vs `{contract['performance_checksum']}`",
            f"Lines: `{contract['compact_lines']}` vs `{contract['performance_lines']}`",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-driver", type=pathlib.Path, required=True)
    parser.add_argument("--performance-driver", type=pathlib.Path, required=True)
    parser.add_argument("--det", type=pathlib.Path, required=True)
    parser.add_argument("--cls", type=pathlib.Path, required=True)
    parser.add_argument("--rec", type=pathlib.Path, required=True)
    parser.add_argument("--dictionary", type=pathlib.Path, required=True)
    parser.add_argument("--image", type=pathlib.Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target-width", type=int, default=960)
    parser.add_argument("--det-threads", type=int, default=0)
    parser.add_argument("--json-output", type=pathlib.Path)
    parser.add_argument("--markdown-output", type=pathlib.Path)
    args = parser.parse_args(argv)
    benchmark_args = [
        str(args.det), str(args.cls), str(args.rec), str(args.dictionary), str(args.image),
        str(args.warmup), str(args.iterations), str(args.workers), str(args.target_width),
    ]
    if args.det_threads:
        benchmark_args.append(str(args.det_threads))
    summary = summarize(
        run_benchmark(args.compact_driver, benchmark_args),
        run_benchmark(args.performance_driver, benchmark_args),
    )
    markdown = render_markdown(summary)
    if args.json_output:
        args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.write_text(markdown, encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
