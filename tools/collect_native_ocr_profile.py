#!/usr/bin/env python3
"""Collect reproducible native OCR latency, operator, and RSS profiles.

The collector intentionally runs the existing benchmark and profile drivers
instead of changing the public OCR ABI.  A case is written only after both
drivers produce deterministic OCR output and compatible metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_CASES = (
    "1:1", "2:1", "4:1",
    "1:2", "2:2", "4:2",
    "1:4", "2:4", "4:4",
)


def parse_positive(value: str, name: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be a positive integer")
    return parsed


def parse_case(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("case must use workers:det_threads, for example 4:2")
    workers = parse_positive(parts[0], "workers")
    det_threads = parse_positive(parts[1], "det_threads")
    if workers > 16 or det_threads > 16:
        raise argparse.ArgumentTypeError("workers and det_threads must be <= 16")
    return workers, det_threads


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} does not exist: {path}")


def run_json(command: list[str], timeout: int, label: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{label} timed out after {timeout}s") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{label} produced no JSON output")
    try:
        report = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{label} last output line is not JSON: {lines[-1][-1000:]}"
        ) from exc
    if not isinstance(report, dict):
        raise RuntimeError(f"{label} JSON root must be an object")
    return report


def require_number(value: Any, label: str, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise RuntimeError(f"{label} is not finite and positive")
    return number


def require_int(report: dict[str, Any], key: str, label: str) -> int:
    value = report.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"{label} is missing integer field {key}")
    return value


def normalize_profile(profile: dict[str, Any], iterations: int) -> dict[str, Any]:
    scale = 1.0 / (float(iterations) * 1.0e6)

    def milliseconds(value: Any) -> float:
        return require_number(value, "profile nanoseconds") * scale

    wall = profile.get("wall_nanoseconds")
    line_work = profile.get("line_work_nanoseconds")
    if not isinstance(wall, dict) or not isinstance(line_work, dict):
        raise RuntimeError("profile is missing wall_nanoseconds or line_work_nanoseconds")

    session_cache: dict[str, float] = {}
    raw_session_cache = profile.get("rec_session_cache")
    if raw_session_cache is not None:
        if not isinstance(raw_session_cache, dict):
            raise RuntimeError("profile rec_session_cache must be an object")
        for key in ("hits", "misses", "reconfigurations"):
            session_cache[key] = (
                require_number(
                    raw_session_cache.get(key),
                    f"profile rec_session_cache.{key}",
                )
                / float(iterations)
            )

    operator_ms: dict[str, Any] = {}
    for item in profile.get("operators", []):
        if not isinstance(item, dict):
            raise RuntimeError("profile operators contains a non-object")
        operator_id = require_int(item, "id", "profile operator")
        operator_ms[str(operator_id)] = {
            "name": item.get("name"),
            "milliseconds": milliseconds(item.get("nanoseconds")),
            "invocations_per_request": require_number(
                item.get("invocations"), "profile operator invocations"
            )
            / float(iterations),
            "percentage": require_number(item.get("percentage"), "profile operator percentage"),
        }

    conv_ms: dict[str, Any] = {}
    for item in profile.get("conv_classes", []):
        if not isinstance(item, dict):
            raise RuntimeError("profile conv_classes contains a non-object")
        class_id = require_int(item, "id", "profile Conv class")
        conv_ms[str(class_id)] = {
            "name": item.get("name"),
            "milliseconds": milliseconds(item.get("nanoseconds")),
            "invocations_per_request": require_number(
                item.get("invocations"), "profile Conv class invocations"
            )
            / float(iterations),
        }

    nodes = []
    for item in profile.get("det_convolution_nodes", []):
        if not isinstance(item, dict):
            raise RuntimeError("profile det_convolution_nodes contains a non-object")
        nodes.append(
            {
                "node": require_int(item, "node", "profile DET node"),
                "operation": item.get("operation"),
                "milliseconds": milliseconds(item.get("nanoseconds")),
                "invocations_per_request": require_number(
                    item.get("invocations"), "profile DET node invocations"
                )
                / float(iterations),
                "input": item.get("input"),
                "weights": item.get("weights"),
                "output": item.get("output"),
                "group": item.get("group"),
                "kernel": item.get("kernel"),
                "strides": item.get("strides"),
                "pads": item.get("pads"),
            }
        )
    nodes.sort(key=lambda item: item["milliseconds"], reverse=True)

    implementation_paths: dict[str, dict[str, float]] = {}
    raw_paths = profile.get("implementation_paths")
    if raw_paths is not None:
        if not isinstance(raw_paths, dict):
            raise RuntimeError("profile implementation_paths must be an object")
        for component in ("detector", "classifier", "recognizer"):
            raw_component = raw_paths.get(component)
            if not isinstance(raw_component, dict):
                raise RuntimeError(
                    f"profile implementation_paths is missing {component}"
                )
            implementation_paths[component] = {}
            for key in (
                "packed_conv1x1",
                "packed_conv3x3_stride2",
                "unpacked_conv",
                "packed_matmul",
                "unpacked_matmul",
                "fused_gelu",
                "ctc_greedy",
                "ctc_packed_projection",
                "ctc_generic_projection",
            ):
                implementation_paths[component][key] = (
                    require_number(
                        raw_component.get(key),
                        f"profile implementation path {component}.{key}",
                    )
                    / float(iterations)
                )

    return {
        "iterations": iterations,
        "lines": require_int(profile, "lines", "profile"),
        "output_checksum": profile.get("output_checksum"),
        "wall_ms_per_request": {
            key: milliseconds(value) for key, value in wall.items()
        },
        "line_work_ms_per_request": {
            key: milliseconds(value) for key, value in line_work.items()
        },
        "rec_session_cache": session_cache,
        "graph_work_ms_per_request": milliseconds(profile.get("graph_work_nanoseconds")),
        "conv_ms_per_request": milliseconds(profile.get("conv_nanoseconds")),
        "implementation_paths": implementation_paths,
        "conv_invocations_per_request": require_number(
            profile.get("conv_invocations"), "profile Conv invocations"
        )
        / float(iterations),
        "operators": operator_ms,
        "conv_classes": conv_ms,
        "top_det_convolution_nodes": nodes[:20],
        "parallel": profile.get("parallel"),
        "rec_width": profile.get("rec_width"),
    }


def normalize_benchmark(
    benchmark: dict[str, Any], workers: int, det_threads: int, target_width: int,
    expected_backend: str | None, require_rss: bool,
) -> dict[str, Any]:
    if expected_backend is not None and benchmark.get("backend") != expected_backend:
        raise RuntimeError(
            f"benchmark backend {benchmark.get('backend')!r} != {expected_backend!r}"
        )
    if require_int(benchmark, "workers", "benchmark") != workers:
        raise RuntimeError("benchmark worker count does not match the requested case")
    if require_int(benchmark, "rec_target_width", "benchmark") != target_width:
        raise RuntimeError("benchmark REC target width does not match the requested case")
    if require_int(benchmark, "lines", "benchmark") <= 0:
        raise RuntimeError("benchmark returned no OCR lines")
    actual_det_threads = require_int(benchmark, "det_intra_actual", "benchmark")
    if actual_det_threads > det_threads:
        raise RuntimeError("benchmark used more DET threads than requested")
    for key in ("detector_ms", "ocr_ms"):
        value = benchmark.get(key)
        if not isinstance(value, dict):
            raise RuntimeError(f"benchmark is missing {key}")
        require_number(value.get("mean"), f"benchmark {key}.mean", positive=True)
        require_number(value.get("p95"), f"benchmark {key}.p95", positive=True)
    rss_after = require_number(
        benchmark.get("rss_after_warmup_bytes"), "benchmark rss_after_warmup_bytes"
    )
    rss_peak = require_number(benchmark.get("peak_rss_bytes"), "benchmark peak_rss_bytes")
    if require_rss and (rss_after <= 0.0 or rss_peak <= 0.0):
        raise RuntimeError("benchmark did not report positive RSS on a required-RSS target")
    return {
        "backend": benchmark.get("backend"),
        "workers": workers,
        "det_threads_requested": det_threads,
        "det_threads_actual": actual_det_threads,
        "lines": benchmark["lines"],
        "rec_target_width": target_width,
        "detector_ms": benchmark["detector_ms"],
        "ocr_ms": benchmark["ocr_ms"],
        "after_detector_ms": require_number(
            benchmark.get("after_detector_ms"), "benchmark after_detector_ms"
        ),
        "after_detector_ms_deprecated": benchmark.get(
            "after_detector_ms_deprecated", False
        ),
        "ocr_minus_standalone_detector_ms": require_number(
            benchmark.get("ocr_minus_standalone_detector_ms", benchmark.get("after_detector_ms")),
            "benchmark ocr_minus_standalone_detector_ms",
        ),
        "throughput_per_second": require_number(
            benchmark.get("throughput_per_second"), "benchmark throughput", positive=True
        ),
        "rss_after_warmup_bytes": int(rss_after),
        "rss_final_bytes": int(
            require_number(benchmark.get("rss_final_bytes"), "benchmark rss_final_bytes")
        ),
        "peak_rss_bytes": int(rss_peak),
    }


def markdown_summary(summary: dict[str, Any]) -> str:
    cases = summary["cases"]
    case_map = {
        (case["benchmark"]["workers"], case["benchmark"]["det_threads_requested"]): case
        for case in cases
    }
    baseline_case = case_map.get((1, 1))
    baseline_ocr = (
        baseline_case["benchmark"]["ocr_ms"]["mean"] if baseline_case else None
    )
    baseline_rss = (
        baseline_case["benchmark"]["peak_rss_bytes"] if baseline_case else None
    )

    def format_optional(value: float | None, suffix: str = "") -> str:
        return "n/a" if value is None else f"{value:.3f}{suffix}"

    lines = [
        "# Native OCR profile summary",
        "",
        f"- Commit: `{summary.get('commit') or 'unknown'}`",
        f"- Backend: `{summary['platform'].get('backend', 'unknown')}`",
        f"- Model: `{summary['model']['variant']}`",
        f"- REC target width: `{summary['settings']['rec_target_width']}`",
        "",
        "| Line workers | DET requested | DET actual | Standalone DET (ms) | OCR mean (ms) | OCR P95 (ms) | Speedup | RSS peak (MiB) | RSS Δ (MiB) | Lines | Checksum |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in cases:
        benchmark = case["benchmark"]
        profile = case["profile"]
        ocr_mean = benchmark["ocr_ms"]["mean"]
        speedup = baseline_ocr / ocr_mean if baseline_ocr else None
        rss_mb = benchmark["peak_rss_bytes"] / 1048576.0
        rss_delta = (
            rss_mb - baseline_rss / 1048576.0 if baseline_rss is not None else None
        )
        lines.append(
            "| {workers} | {det_threads_requested} | {det_threads_actual} | {det:.3f} | "
            "{ocr:.3f} | {p95:.3f} | {speedup} | {rss:.2f} | {rss_delta} | "
            "{lines_count} | `{checksum}` |".format(
                workers=benchmark["workers"],
                det_threads_requested=benchmark["det_threads_requested"],
                det_threads_actual=benchmark["det_threads_actual"],
                det=benchmark["detector_ms"]["mean"],
                ocr=ocr_mean,
                p95=benchmark["ocr_ms"]["p95"],
                speedup=format_optional(speedup, "x"),
                rss=rss_mb,
                rss_delta=format_optional(rss_delta, ""),
                lines_count=profile["lines"],
                checksum=profile["output_checksum"],
            )
        )
    lines.extend(["", "- `Standalone DET` is measured by the standalone detector benchmark.",
                  "- `det_threads_actual` is the actual OCR-handle DET intra-op count.",
                  "- `RSS Δ` and speedup are relative to workers=1, requested DET threads=1.", ""])
    lines.extend(["## Observed parallel scaling", ""])

    def scaling_line(label: str, keys: list[tuple[int, int]]) -> None:
        values = []
        for key in keys:
            case = case_map.get(key)
            if case is None:
                continue
            values.append(
                f"{key[0]}:{key[1]}={case['benchmark']['ocr_ms']['mean']:.3f} ms"
            )
        if values:
            lines.append(f"- {label}: " + ", ".join(values))

    scaling_line("Line workers at DET threads=1", [(1, 1), (2, 1), (4, 1)])
    scaling_line("DET intra-op at workers=1", [(1, 1), (1, 2), (1, 4)])
    combined = case_map.get((4, 4))
    if baseline_case and combined:
        combined_speedup = baseline_ocr / combined["benchmark"]["ocr_ms"]["mean"]
        combined_rss_delta = (
            combined["benchmark"]["peak_rss_bytes"] - baseline_rss
        ) / 1048576.0
        lines.append(f"- Combined 1:1 → 4:4: {combined_speedup:.3f}x, "
                     f"RSS Δ {combined_rss_delta:+.2f} MiB")
    lines.extend(["", "## Best observed configuration on this runner", ""])
    best = min(cases, key=lambda case: case["benchmark"]["ocr_ms"]["mean"])
    best_benchmark = best["benchmark"]
    lines.append(
        f"- Line workers: {best_benchmark['workers']}\n"
        f"- DET threads requested/actual: {best_benchmark['det_threads_requested']} / "
        f"{best_benchmark['det_threads_actual']}\n"
        f"- Mean OCR: {best_benchmark['ocr_ms']['mean']:.3f} ms\n"
        f"- P95 OCR: {best_benchmark['ocr_ms']['p95']:.3f} ms\n"
        f"- Peak RSS: {best_benchmark['peak_rss_bytes'] / 1048576.0:.2f} MiB"
    )
    lines.extend(
        [
            "",
            "## Stage breakdown",
            "",
        ]
    )
    wall_stages = (
        ("DET preprocess", "det_preprocess"),
        ("DET graph", "det_graph"),
        ("DET postprocess", "det_postprocess"),
        ("Crop", "crop"),
        ("Line workers critical path", "line_worker_critical"),
        ("Output", "output"),
    )
    line_stages = (
        ("CLS preprocess", "cls_preprocess"),
        ("CLS graph", "cls_graph"),
        ("CLS postprocess", "cls_postprocess"),
        ("REC preprocess", "rec_preprocess"),
        ("REC graph", "rec_graph"),
        ("REC postprocess", "rec_postprocess"),
    )
    for case in cases:
        lines.append(
            f"### workers={case['benchmark']['workers']}, det_threads={case['benchmark']['det_threads_requested']}"
        )
        lines.append("")
        wall = case["profile"]["wall_ms_per_request"]
        total = wall["total"]
        lines.append("| Stage | Profiled ms/request | % of profiled wall |")
        lines.append("|---|---:|---:|")
        for label, key in wall_stages:
            lines.append(
                f"| {label} | {wall[key]:.3f} | "
                f"{(wall[key] * 100.0 / total) if total else 0.0:.2f}% |"
            )
        lines.extend(["", "Accumulated line work (not wall time; stages must not be summed with each other):", "",
                      "| Stage | Accumulated ms/request |", "|---|---:|"])
        line_work = case["profile"]["line_work_ms_per_request"]
        for label, key in line_stages:
            lines.append(f"| {label} | {line_work[key]:.3f} |")
        lines.extend(["", "Largest profiled DET convolution nodes:", "",
                      "| Node | Operation | Input | Kernel | ms/request |",
                      "|---:|---|---|---|---:|"])
        for node in case["profile"]["top_det_convolution_nodes"][:10]:
            lines.append(
                f"| {node['node']} | {node['operation']} | `{node['input']}` | "
                f"`{node['kernel']}` | {node['milliseconds']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def collect(args: argparse.Namespace) -> int:
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "profile_driver": args.profile_driver,
        "benchmark_driver": args.benchmark_driver,
        "det": args.det,
        "cls": args.cls,
        "rec": args.rec,
        "dictionary": args.dictionary,
        "image": args.image,
    }
    for label, path in paths.items():
        require_file(path, label)

    cases = [parse_case(value) if isinstance(value, str) else value for value in args.case]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "commit": args.commit or os.environ.get("GITHUB_SHA"),
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_processors": os.cpu_count(),
            "backend": args.expected_backend,
        },
        "model": {
            "variant": args.variant,
            "det_sha256": sha256_file(args.det),
            "cls_sha256": sha256_file(args.cls),
            "rec_sha256": sha256_file(args.rec),
            "dictionary_sha256": sha256_file(args.dictionary),
            "image_sha256": sha256_file(args.image),
        },
        "settings": {
            "rec_target_width": args.target_width,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "profile_iterations": args.profile_iterations,
            "cases": [f"{workers}:{det_threads}" for workers, det_threads in cases],
        },
        "cases": [],
    }

    checksums: set[str] = set()
    for workers, det_threads in cases:
        case_name = f"w{workers}-d{det_threads}"
        benchmark = run_json(
            [
                str(args.benchmark_driver),
                str(args.det),
                str(args.cls),
                str(args.rec),
                str(args.dictionary),
                str(args.image),
                str(args.warmup),
                str(args.iterations),
                str(workers),
                str(args.target_width),
                str(det_threads),
            ],
            args.timeout,
            f"benchmark {case_name}",
        )
        profile = run_json(
            [
                str(args.profile_driver),
                str(args.det),
                str(args.cls),
                str(args.rec),
                str(args.dictionary),
                str(args.image),
                str(args.profile_iterations),
                str(workers),
                str(args.target_width),
                str(det_threads),
            ],
            args.timeout,
            f"profile {case_name}",
        )
        normalized_benchmark = normalize_benchmark(
            benchmark,
            workers,
            det_threads,
            args.target_width,
            args.expected_backend,
            args.require_rss,
        )
        normalized_profile = normalize_profile(profile, args.profile_iterations)
        if normalized_benchmark["lines"] != normalized_profile["lines"]:
            raise RuntimeError(f"{case_name} benchmark/profile line count differs")
        checksum = normalized_profile["output_checksum"]
        if not isinstance(checksum, str) or not checksum:
            raise RuntimeError(f"{case_name} profile has no output checksum")
        checksums.add(checksum)
        (output_dir / f"benchmark-{case_name}.json").write_text(
            json.dumps(benchmark, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (output_dir / f"profile-{case_name}.json").write_text(
            json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        summary["cases"].append(
            {
                "name": case_name,
                "benchmark": normalized_benchmark,
                "profile": normalized_profile,
            }
        )

    if len(checksums) != 1:
        raise RuntimeError(f"OCR checksum changed between cases: {sorted(checksums)}")
    summary["output_checksum"] = next(iter(checksums))
    (output_dir / "environment.json").write_text(
        json.dumps(summary["platform"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / args.summary_name).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "SUMMARY.md").write_text(markdown_summary(summary), encoding="utf-8")
    print(json.dumps({"output": str(output_dir), "checksum": summary["output_checksum"]}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-driver", type=Path, required=True)
    parser.add_argument("--benchmark-driver", type=Path, required=True)
    parser.add_argument("--det", type=Path, required=True)
    parser.add_argument("--cls", type=Path, required=True)
    parser.add_argument("--rec", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--variant", default="tiny")
    parser.add_argument("--target-width", type=lambda value: parse_positive(value, "target-width"), default=960)
    parser.add_argument("--warmup", type=lambda value: parse_positive(value, "warmup"), default=3)
    parser.add_argument("--iterations", type=lambda value: parse_positive(value, "iterations"), default=10)
    parser.add_argument(
        "--profile-iterations",
        type=lambda value: parse_positive(value, "profile-iterations"),
        default=3,
    )
    parser.add_argument("--timeout", type=lambda value: parse_positive(value, "timeout"), default=1800)
    parser.add_argument("--expected-backend", default="neon")
    parser.add_argument("--commit")
    parser.add_argument(
        "--summary-name",
        default="arm64-profile-summary.json",
        help="JSON summary filename; the default preserves the ARM64 artifact contract",
    )
    parser.add_argument("--require-rss", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="workers:det_threads; repeat for multiple cases",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.case is None:
        args.case = list(DEFAULT_CASES)
    try:
        return collect(args)
    except (OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
