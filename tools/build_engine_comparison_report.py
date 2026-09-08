#!/usr/bin/env python3
"""Validate paired C/C# OCR benchmark results and build a Markdown report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class ComparisonError(ValueError):
    """Raised when a benchmark artifact violates the comparison contract."""


@dataclass(frozen=True)
class Run:
    path: Path
    replica: int
    engine: str
    profile: str
    workers: int
    meta: dict[str, Any]
    summary: dict[str, Any]
    rows: list[dict[str, Any]]

    @property
    def median_ms(self) -> float:
        return float(self.summary["total_ms"]["median"])

    @property
    def peak_mb(self) -> float:
        return float(self.meta["working_set_mb_peak"])

    @property
    def loaded_mb(self) -> float:
        return float(self.meta["working_set_mb_loaded"])

    @property
    def peak_delta_mb(self) -> float:
        return self.peak_mb - self.loaded_mb

    @property
    def accuracy(self) -> dict[str, Any]:
        return self.meta["accuracy"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ComparisonError(f"JSON root must be an object: {path}")
    return value


def require_full_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise ComparisonError(f"{label} must be a 40-character Git SHA")
    if any(ch not in "0123456789abcdefABCDEF" for ch in value):
        raise ComparisonError(f"{label} must be hexadecimal")
    return value.lower()


def require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ComparisonError(f"{label} must be a SHA-256")
    if any(ch not in "0123456789abcdefABCDEF" for ch in value):
        raise ComparisonError(f"{label} must be hexadecimal")
    return value.lower()


def finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ComparisonError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ComparisonError(f"{label} is outside its valid range")
    return result


def load_contract(path: Path) -> dict[str, Any]:
    contract = load_json(path)
    if contract.get("schema_version") != 1:
        raise ComparisonError("comparison contract schema_version must be 1")
    require_full_sha(contract.get("simd_ref"), "simd_ref")
    if contract.get("simd_repository") != "lxw112190/SimdPaddleOCR":
        raise ComparisonError("comparison contract uses an unexpected harness repository")
    if contract.get("models") != ["tiny"]:
        raise ComparisonError("comparison v1 must contain only the tiny model")
    if contract.get("workers") != [1, 4]:
        raise ComparisonError("comparison v1 workers must be [1, 4]")
    if contract.get("replicas") != 3:
        raise ComparisonError("comparison v1 requires three paired replicas")
    dataset = contract.get("dataset")
    if not isinstance(dataset, dict):
        raise ComparisonError("comparison contract lacks dataset")
    expected = dataset.get("expected_images")
    warmup = dataset.get("warmup_images")
    measured = dataset.get("measured_images")
    if not all(isinstance(x, int) for x in (expected, warmup, measured)):
        raise ComparisonError("dataset image counts must be integers")
    if expected != warmup + measured or expected <= 1 or warmup != 1:
        raise ComparisonError("dataset counts must describe one warm-up plus measured images")
    if dataset.get("generator_project") != "test/Sdcb.SimdPaddleOCR.TestData":
        raise ComparisonError("comparison v1 uses an unexpected dataset generator")
    profiles = contract.get("profiles")
    expected_profiles = {
        "normalized-fixed320": {
            "requested_isa": "avx2",
            "sharp": {"rec_policy": "fixed", "rec_width": 320},
            "c": {"rec_policy": "fixed", "rec_width": 320},
        },
        "product": {
            "requested_isa": "best",
            "sharp": {"rec_policy": "native-adaptive", "rec_width": 320},
            "c": {"rec_policy": "adaptive-max", "rec_width": 960},
        },
    }
    if profiles != expected_profiles:
        raise ComparisonError("comparison v1 profile semantics have drifted")
    expected_normalized = {
        "reading_order": "horizontal",
        "detector": {
            "limit_side": 960,
            "bitmap_threshold": 0.3,
            "box_threshold": 0.6,
            "unclip_ratio": 1.6,
            "use_dilation": False,
            "max_candidates": 1000,
        },
        "classifier": {"enabled": True, "threshold": 0.9},
    }
    if contract.get("normalized_options") != expected_normalized:
        raise ComparisonError("comparison v1 normalized DET/CLS options have drifted")
    policy = contract.get("comparison_policy")
    if not isinstance(policy, dict) or policy.get("ratios") != "paired-within-replica":
        raise ComparisonError("comparison ratios must be paired within each replica")
    if policy.get("performance_gate") is not False or policy.get("accuracy_gate") is not False:
        raise ComparisonError("performance and accuracy must remain informational in v1")
    if policy.get("correctness_gate") is not True:
        raise ComparisonError("correctness must be gating in v1")
    return contract


def read_sha256s(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise ComparisonError(f"cannot read checksum file {path}: {error}") from error
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            raise ComparisonError(f"invalid checksum line in {path}: {line}")
        digest = require_hash(parts[0], f"checksum in {path}")
        name = parts[1].lstrip("*").replace("\\", "/")
        if name in result:
            raise ComparisonError(f"duplicate checksum entry: {name}")
        result[name] = digest
    return result


def validate_dataset(
    dataset_dir: Path, expected_images: int
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    metadata_path = dataset_dir / "metadata.json"
    metadata = load_json(metadata_path)
    images = metadata.get("images")
    if metadata.get("version") != 1 or not isinstance(images, list):
        raise ComparisonError("dataset metadata version/images are invalid")
    if len(images) != expected_images:
        raise ComparisonError(
            f"dataset contains {len(images)} images, expected {expected_images}"
        )
    references: dict[str, list[str]] = {}
    for entry in images:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise ComparisonError("dataset image entry is invalid")
        name = entry["file"]
        image_path = dataset_dir / name
        if not image_path.is_file():
            raise ComparisonError(f"dataset image is missing: {name}")
        if require_hash(entry.get("sha256"), f"dataset {name} sha256") != sha256_file(image_path):
            raise ComparisonError(f"dataset image SHA-256 mismatch: {name}")
        lines = entry.get("lines")
        if not isinstance(lines, list) or not all(
            isinstance(line, dict) and isinstance(line.get("text"), str) for line in lines
        ):
            raise ComparisonError(f"dataset reference lines are invalid: {name}")
        references[name] = [line["text"] for line in lines]

    sums_path = dataset_dir / "SHA256SUMS"
    if sums_path.is_file():
        for name, digest in read_sha256s(sums_path).items():
            target = dataset_dir / name
            if not target.is_file() or sha256_file(target) != digest:
                raise ComparisonError(f"dataset SHA256SUMS mismatch: {name}")
    return references, {
        "version": metadata["version"],
        "seed": metadata.get("seed"),
        "images": len(images),
        "reference_lines": sum(len(lines) for lines in references.values()),
        "manifest_sha256": sha256_file(metadata_path),
    }


def validate_run(
    path: Path,
    doc: dict[str, Any],
    contract: dict[str, Any],
    replica: int,
    lw_commit: str,
    model_hashes: dict[str, str],
) -> Run:
    meta = doc.get("meta")
    summary = doc.get("summary")
    rows = doc.get("rows")
    if not isinstance(meta, dict) or not isinstance(summary, dict) or not isinstance(rows, list):
        raise ComparisonError(f"benchmark JSON lacks meta/summary/rows: {path}")
    if meta.get("schema_version") != 1:
        raise ComparisonError(f"benchmark schema_version mismatch: {path}")
    engine = meta.get("mode")
    profile = meta.get("comparison_profile")
    workers = meta.get("workers")
    if engine not in ("sharp", "c") or profile not in contract["profiles"]:
        raise ComparisonError(f"benchmark engine/profile is invalid: {path}")
    if workers not in contract["workers"] or meta.get("model") != "tiny":
        raise ComparisonError(f"benchmark model/workers is invalid: {path}")
    if meta.get("replica") != replica:
        raise ComparisonError(f"benchmark replica mismatch: {path}")
    if meta.get("benchmark") is not True or meta.get("benchmarkKind") != "engine":
        raise ComparisonError(f"benchmark mode is not the engine benchmark: {path}")

    profile_contract = contract["profiles"][profile]
    engine_contract = profile_contract[engine]
    for key in ("rec_policy", "rec_width"):
        if meta.get(key) != engine_contract[key]:
            raise ComparisonError(f"{key} violates {profile} contract: {path}")
    if meta.get("requested_isa") != profile_contract["requested_isa"]:
        raise ComparisonError(f"requested ISA violates {profile} contract: {path}")
    effective_isa = meta.get("effective_isa")
    if not isinstance(effective_isa, str) or effective_isa in ("", "unknown"):
        raise ComparisonError(f"effective ISA is missing: {path}")
    if profile == "normalized-fixed320" and effective_isa != "avx2":
        raise ComparisonError(f"normalized benchmark is not AVX2: {path}")

    expected_images = contract["dataset"]["expected_images"]
    measured_images = contract["dataset"]["measured_images"]
    if len(rows) != expected_images or meta.get("sampleCount") != expected_images:
        raise ComparisonError(f"benchmark sample count mismatch: {path}")
    warmups = sum(row.get("warmup") is True for row in rows if isinstance(row, dict))
    if warmups != contract["dataset"]["warmup_images"] or summary.get("n") != measured_images:
        raise ComparisonError(f"benchmark warm-up/measured count mismatch: {path}")
    total_ms = summary.get("total_ms")
    if not isinstance(total_ms, dict):
        raise ComparisonError(f"benchmark timing summary is missing: {path}")
    for key in ("mean", "median", "p95"):
        finite_number(total_ms.get(key), f"{path} total_ms.{key}", positive=True)
    finite_number(meta.get("working_set_mb_loaded"), f"{path} loaded working set", positive=True)
    peak = finite_number(meta.get("working_set_mb_peak"), f"{path} peak working set", positive=True)
    loaded = float(meta["working_set_mb_loaded"])
    if peak < loaded:
        raise ComparisonError(f"peak working set is below loaded working set: {path}")

    accuracy = meta.get("accuracy")
    if not isinstance(accuracy, dict):
        raise ComparisonError(f"benchmark accuracy is missing: {path}")
    for key in ("exact_lines", "total_lines", "images", "errors", "total_chars"):
        if not isinstance(accuracy.get(key), int):
            raise ComparisonError(f"benchmark accuracy.{key} is invalid: {path}")
    if accuracy["images"] != measured_images or accuracy["total_lines"] <= 0:
        raise ComparisonError(f"benchmark accuracy counts are invalid: {path}")
    finite_number(accuracy.get("cer"), f"{path} accuracy.cer")

    if engine == "c":
        if meta.get("c_assets_mode") != "external":
            raise ComparisonError(f"C benchmark did not use external-only assets: {path}")
        if require_full_sha(meta.get("lw_commit"), f"{path} lw_commit") != lw_commit:
            raise ComparisonError(f"C benchmark lw commit mismatch: {path}")
        expected_assets = {
            "det_sha256": model_hashes["det.lwm"],
            "cls_sha256": model_hashes["cls.lwm"],
            "rec_sha256": model_hashes["rec.lwm"],
            "dictionary_sha256": model_hashes["ppocr_keys.txt"],
        }
        for key, expected in expected_assets.items():
            if require_hash(meta.get(key), f"{path} {key}") != expected:
                raise ComparisonError(f"C benchmark asset mismatch for {key}: {path}")
        require_hash(meta.get("lw_dll_sha256"), f"{path} lw_dll_sha256")

    return Run(path, replica, engine, profile, int(workers), meta, summary, rows)


def load_runs(
    root: Path,
    contract: dict[str, Any],
    lw_commit: str,
    dataset_sha: str,
    model_hashes: dict[str, str],
    include_product: bool,
) -> list[Run]:
    contract_sha = sha256_file(Path(contract["_path"]))
    harness_sha = require_full_sha(contract["simd_ref"], "simd_ref")
    manifests = sorted(root.rglob("replica-manifest.json"))
    if len(manifests) != contract["replicas"]:
        raise ComparisonError(
            f"found {len(manifests)} replica manifests, expected {contract['replicas']}"
        )
    runs: list[Run] = []
    replicas_seen: set[int] = set()
    for manifest_path in manifests:
        manifest = load_json(manifest_path)
        replica = manifest.get("replica")
        if not isinstance(replica, int) or replica in replicas_seen:
            raise ComparisonError(f"invalid or duplicate replica manifest: {manifest_path}")
        replicas_seen.add(replica)
        if manifest.get("schema_version") != 1:
            raise ComparisonError(f"replica manifest schema mismatch: {manifest_path}")
        if require_full_sha(manifest.get("lw_commit"), "replica lw_commit") != lw_commit:
            raise ComparisonError(f"replica lw commit mismatch: {manifest_path}")
        if require_full_sha(manifest.get("harness_commit"), "harness_commit") != harness_sha:
            raise ComparisonError(f"replica harness commit mismatch: {manifest_path}")
        if require_hash(manifest.get("contract_sha256"), "contract_sha256") != contract_sha:
            raise ComparisonError(f"replica contract SHA mismatch: {manifest_path}")
        if require_hash(manifest.get("dataset_manifest_sha256"), "dataset manifest SHA") != dataset_sha:
            raise ComparisonError(f"replica dataset SHA mismatch: {manifest_path}")
        if manifest.get("c_backend") != "avx2":
            raise ComparisonError(f"replica C backend must be avx2: {manifest_path}")
        cases = manifest.get("cases")
        if not isinstance(cases, list) or not all(isinstance(x, str) for x in cases):
            raise ComparisonError(f"replica case list is invalid: {manifest_path}")
        for relative in cases:
            case_path = manifest_path.parent / relative
            if not case_path.is_file():
                raise ComparisonError(f"replica case is missing: {case_path}")
            runs.append(
                validate_run(
                    case_path,
                    load_json(case_path),
                    contract,
                    replica,
                    lw_commit,
                    model_hashes,
                )
            )

    profiles = ["normalized-fixed320"] + (["product"] if include_product else [])
    expected = {
        (replica, engine, profile, workers)
        for replica in range(1, contract["replicas"] + 1)
        for engine in ("sharp", "c")
        for profile in profiles
        for workers in contract["workers"]
    }
    actual = {(r.replica, r.engine, r.profile, r.workers) for r in runs}
    if actual != expected or len(runs) != len(expected):
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ComparisonError(f"benchmark case set mismatch; missing={missing}, extra={extra}")

    for replica in range(1, contract["replicas"] + 1):
        dll_hashes = {
            run.meta["lw_dll_sha256"]
            for run in runs
            if run.replica == replica and run.engine == "c"
        }
        if len(dll_hashes) != 1:
            raise ComparisonError(f"replica {replica} used multiple C DLLs")

    for key in {(r.profile, r.workers) for r in runs}:
        c_runs = sorted(
            (r for r in runs if r.engine == "c" and (r.profile, r.workers) == key),
            key=lambda r: r.replica,
        )
        fingerprints = [
            [
                (
                    row.get("file"),
                    row.get("hash"),
                    row.get("texts"),
                    row.get("rotations"),
                )
                for row in run.rows
                if row.get("warmup") is not True
            ]
            for run in c_runs
        ]
        if any(value != fingerprints[0] for value in fingerprints[1:]):
            raise ComparisonError(
                f"C output is not deterministic across replicas for {key[0]} {key[1]}w"
            )
    return runs


def paired(runs: list[Run], profile: str, workers: int) -> list[tuple[Run, Run]]:
    result: list[tuple[Run, Run]] = []
    replicas = sorted({run.replica for run in runs})
    for replica in replicas:
        sharp = next(
            run for run in runs
            if (run.replica, run.engine, run.profile, run.workers)
            == (replica, "sharp", profile, workers)
        )
        c_run = next(
            run for run in runs
            if (run.replica, run.engine, run.profile, run.workers)
            == (replica, "c", profile, workers)
        )
        if sharp.accuracy["total_lines"] != c_run.accuracy["total_lines"]:
            raise ComparisonError(
                f"reference line count differs in replica {replica}, {profile}, {workers}w"
            )
        result.append((sharp, c_run))
    return result


def ratio_summary(values: Iterable[float]) -> str:
    numbers = list(values)
    return (
        f"{statistics.median(numbers):.3f}x "
        f"(range {min(numbers):.3f}-{max(numbers):.3f}x)"
    )


def accuracy_runs(
    runs: list[Run], engine: str, profile: str, workers: int
) -> list[Run]:
    return sorted(
        (
            run
            for run in runs
            if (run.engine, run.profile, run.workers) == (engine, profile, workers)
        ),
        key=lambda run: run.replica,
    )


def line_edit_distance(left: str, right: str) -> int:
    """Return the UTF-8 text's Unicode-code-point Levenshtein distance."""
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def normalized_line_distance(left: str, right: str) -> float:
    denominator = max(len(left), len(right), 1)
    return line_edit_distance(left, right) / denominator


def align_lines(
    expected: list[str], predicted: list[str]
) -> tuple[list[tuple[int, int | None]], list[int]]:
    """Align predicted lines to references without shifting all later lines.

    The diagonal operation wins deterministic ties, followed by deletion and
    insertion. Extra predicted lines are returned separately because they do
    not have a reference line to which their error can be attributed.
    """
    rows = len(expected)
    columns = len(predicted)
    costs = [[0.0] * (columns + 1) for _ in range(rows + 1)]
    choices = [["start"] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        costs[row][0] = float(row)
        choices[row][0] = "delete"
    for column in range(1, columns + 1):
        costs[0][column] = float(column)
        choices[0][column] = "insert"
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            candidates = (
                (
                    costs[row - 1][column - 1]
                    + normalized_line_distance(expected[row - 1], predicted[column - 1]),
                    0,
                    "match",
                ),
                (costs[row - 1][column] + 1.0, 1, "delete"),
                (costs[row][column - 1] + 1.0, 2, "insert"),
            )
            costs[row][column], _, choices[row][column] = min(candidates)

    aligned: list[tuple[int, int | None]] = []
    extras: list[int] = []
    row = rows
    column = columns
    while row > 0 or column > 0:
        choice = choices[row][column]
        if choice == "match":
            aligned.append((row - 1, column - 1))
            row -= 1
            column -= 1
        elif choice == "delete":
            aligned.append((row - 1, None))
            row -= 1
        elif choice == "insert":
            extras.append(column - 1)
            column -= 1
        else:
            raise ComparisonError("line alignment produced an invalid traceback")
    aligned.reverse()
    extras.reverse()
    return aligned, extras


def build_line_diagnostics(
    runs: list[Run],
    references: dict[str, list[str]],
    profiles: list[str],
    workers_values: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    line_cases: list[dict[str, Any]] = []
    contributors: list[dict[str, Any]] = []
    summary = [
        "## Line-level accuracy diagnostics",
        "",
        "Line alignment uses normalized character edit distance; extra predicted lines are reported separately.",
        "",
        "| Profile | Workers | GT lines | Both correct | C# only | C only | Both wrong | Missing/extra |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile in profiles:
        for workers in workers_values:
            sharp, c_run = paired(runs, profile, workers)[0]
            sharp_rows = {
                row["file"]: row for row in sharp.rows if row.get("warmup") is not True
            }
            c_rows = {
                row["file"]: row for row in c_run.rows if row.get("warmup") is not True
            }
            counts = {
                "gt": 0,
                "both_correct": 0,
                "sharp_only": 0,
                "c_only": 0,
                "both_wrong": 0,
                "missing_extra": 0,
            }
            for name in sorted(set(sharp_rows) | set(c_rows)):
                if name not in references or name not in sharp_rows or name not in c_rows:
                    raise ComparisonError(f"cannot align line diagnostic case: {name}")
                expected = references[name]
                sharp_text = sharp_rows[name].get("texts")
                c_text = c_rows[name].get("texts")
                if not isinstance(sharp_text, list) or not isinstance(c_text, list):
                    raise ComparisonError(f"invalid OCR texts in line diagnostic case: {name}")
                sharp_alignment, sharp_extras = align_lines(expected, sharp_text)
                c_alignment, c_extras = align_lines(expected, c_text)
                sharp_by_reference = dict(sharp_alignment)
                c_by_reference = dict(c_alignment)
                line_items: list[dict[str, Any]] = []
                for reference_index, expected_text in enumerate(expected):
                    sharp_index = sharp_by_reference.get(reference_index)
                    c_index = c_by_reference.get(reference_index)
                    sharp_value = None if sharp_index is None else sharp_text[sharp_index]
                    c_value = None if c_index is None else c_text[c_index]
                    sharp_edit = line_edit_distance(expected_text, sharp_value or "")
                    c_edit = line_edit_distance(expected_text, c_value or "")
                    sharp_correct = sharp_value == expected_text
                    c_correct = c_value == expected_text
                    counts["gt"] += 1
                    if sharp_correct and c_correct:
                        counts["both_correct"] += 1
                    elif sharp_correct:
                        counts["sharp_only"] += 1
                    elif c_correct:
                        counts["c_only"] += 1
                    else:
                        counts["both_wrong"] += 1
                    if sharp_value is None or c_value is None:
                        counts["missing_extra"] += 1
                    item = {
                        "profile": profile,
                        "workers": workers,
                        "file": name,
                        "line_index": reference_index,
                        "reference": expected_text,
                        "sharp": sharp_value,
                        "c": c_value,
                        "sharp_edit": sharp_edit,
                        "c_edit": c_edit,
                        "c_extra_errors": c_edit - sharp_edit,
                    }
                    line_items.append(item)
                    contributors.append(item)
                extra_item = {
                    "profile": profile,
                    "workers": workers,
                    "file": name,
                    "sharp_extra": [sharp_text[index] for index in sharp_extras],
                    "c_extra": [c_text[index] for index in c_extras],
                }
                if sharp_extras or c_extras:
                    counts["missing_extra"] += len(sharp_extras) + len(c_extras)
                line_cases.append(
                    {
                        "profile": profile,
                        "workers": workers,
                        "file": name,
                        "lines": line_items,
                        "extra": extra_item,
                    }
                )
            summary.append(
                f"| {profile} | {workers} | {counts['gt']} | {counts['both_correct']} | "
                f"{counts['sharp_only']} | {counts['c_only']} | {counts['both_wrong']} | "
                f"{counts['missing_extra']} |"
            )
    contributors.sort(
        key=lambda item: (-item["c_extra_errors"], item["file"], item["line_index"])
    )
    summary_profile = "product" if "product" in profiles else "normalized-fixed320"
    summary_workers = max(workers_values)
    summary_contributors = [
        item
        for item in contributors
        if item["profile"] == summary_profile and item["workers"] == summary_workers
    ]
    if not summary_contributors:
        summary_contributors = contributors
    summary.extend(
        [
            "",
            "### Top C-vs-C# CER contributors",
            "",
            "| Profile | Workers | Image | GT line | C# edit | C edit | C extra errors |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary_contributors[:10]:
        summary.append(
            f"| {item['profile']} | {item['workers']} | {item['file']} | "
            f"{item['line_index']} | {item['sharp_edit']} | {item['c_edit']} | "
            f"{item['c_extra_errors']} |"
        )
    summary.append("")
    return line_cases, contributors, summary


def build_disagreements(
    runs: list[Run],
    references: dict[str, list[str]],
    profiles: list[str],
    workers_values: list[int],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "sharp-correct-c-wrong": [],
        "c-correct-sharp-wrong": [],
        "both-wrong": [],
        "text-different": [],
    }
    summary = [
        "# OCR disagreement summary",
        "",
        "Detailed cases use replica 1; C determinism is checked across all replicas.",
        "",
        "| Profile | Workers | C# correct / C wrong | C correct / C# wrong | Both wrong | Text differs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile in profiles:
        for workers in workers_values:
            sharp, c_run = paired(runs, profile, workers)[0]
            sharp_rows = {
                row["file"]: row for row in sharp.rows if row.get("warmup") is not True
            }
            c_rows = {
                row["file"]: row for row in c_run.rows if row.get("warmup") is not True
            }
            counts = {name: 0 for name in buckets}
            for name in sorted(set(sharp_rows) | set(c_rows)):
                if name not in references or name not in sharp_rows or name not in c_rows:
                    raise ComparisonError(f"cannot align disagreement case: {name}")
                expected = references[name]
                sharp_text = sharp_rows[name].get("texts")
                c_text = c_rows[name].get("texts")
                if not isinstance(sharp_text, list) or not isinstance(c_text, list):
                    raise ComparisonError(f"invalid OCR texts in disagreement case: {name}")
                sharp_ok = sharp_text == expected
                c_ok = c_text == expected
                item = {
                    "profile": profile,
                    "workers": workers,
                    "file": name,
                    "reference": expected,
                    "sharp": sharp_text,
                    "c": c_text,
                }
                if sharp_ok and not c_ok:
                    category = "sharp-correct-c-wrong"
                elif c_ok and not sharp_ok:
                    category = "c-correct-sharp-wrong"
                elif not sharp_ok and not c_ok:
                    category = "both-wrong"
                else:
                    category = ""
                if category:
                    buckets[category].append(item)
                    counts[category] += 1
                if sharp_text != c_text:
                    buckets["text-different"].append(item)
                    counts["text-different"] += 1
            summary.append(
                f"| {profile} | {workers} | "
                f"{counts['sharp-correct-c-wrong']} | "
                f"{counts['c-correct-sharp-wrong']} | "
                f"{counts['both-wrong']} | {counts['text-different']} |"
            )
    summary.append("")
    return buckets, summary


def build_markdown(
    runs: list[Run],
    contract: dict[str, Any],
    lw_commit: str,
    contract_sha: str,
    dataset_info: dict[str, Any],
    profiles: list[str],
) -> str:
    lines = [
        "# lw.PPOCR.C vs SimdPaddleOCR",
        "",
        "## Reproducibility",
        "",
        f"- lw.PPOCR.C: {lw_commit}",
        f"- SimdPaddleOCR: {contract['simd_ref']}",
        f"- Comparison contract SHA-256: {contract_sha}",
        f"- Dataset manifest SHA-256: {dataset_info['manifest_sha256']}",
        f"- Dataset: {dataset_info['images']} images, "
        f"{dataset_info['reference_lines']} reference lines, "
        f"{contract['dataset']['measured_images']} measured after one warm-up",
        f"- Replicas: {contract['replicas']}",
        "",
        "> Ratios are calculated only within the same replica. Absolute latency "
        "values from different runner CPUs are not pooled.",
        "",
        "## Runners",
        "",
        "| Replica | CPU | Logical CPU | RAM | C ISA | C# ISA |",
        "| ---: | --- | ---: | ---: | --- | --- |",
    ]
    normalized_pairs = paired(runs, "normalized-fixed320", 1)
    for replica in range(1, contract["replicas"] + 1):
        sharp, c_run = normalized_pairs[replica - 1]
        if sharp.meta.get("machine") != c_run.meta.get("machine"):
            raise ComparisonError(f"replica {replica} engines did not run on the same machine")
        cpu = str(c_run.meta.get("cpuName") or sharp.meta.get("cpuName") or "unknown")
        logical = c_run.meta.get("cpu", "unknown")
        memory = c_run.meta.get("memoryMb") or sharp.meta.get("memoryMb")
        ram = f"{float(memory) / 1024:.1f} GiB" if isinstance(memory, (int, float)) else "unknown"
        lines.append(
            f"| {replica} | {cpu.replace('|', '/')} | {logical} | {ram} | "
            f"{c_run.meta['effective_isa']} | {sharp.meta['effective_isa']} |"
        )

    summary_profile = "product" if "product" in profiles else "normalized-fixed320"
    lines.extend(
        [
            "",
            "## Executive summary",
            "",
            f"The primary summary profile is `{summary_profile}`. Values are paired within each replica; "
            "performance and accuracy remain informational.",
            "",
            "| Profile | Workers | C/C# latency | C-C# median ms | C peak WS saving | Exact-line gap | CER gap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for workers in contract["workers"]:
        pairs = paired(runs, summary_profile, workers)
        ratios = [c.median_ms / sharp.median_ms for sharp, c in pairs]
        deltas = [c.median_ms - sharp.median_ms for sharp, c in pairs]
        memory_savings = [1.0 - c.peak_mb / sharp.peak_mb for sharp, c in pairs]
        exact_gaps = [
            c.accuracy["exact_lines"] / c.accuracy["total_lines"]
            - sharp.accuracy["exact_lines"] / sharp.accuracy["total_lines"]
            for sharp, c in pairs
        ]
        cer_gaps = [float(c.accuracy["cer"]) - float(sharp.accuracy["cer"]) for sharp, c in pairs]
        lines.append(
            f"| {summary_profile} | {workers} | {statistics.median(ratios):.3f}x | "
            f"{statistics.median(deltas):+.3f} | {statistics.median(memory_savings):+.2%} | "
            f"{statistics.median(exact_gaps):+.2%} | {statistics.median(cer_gaps):+.2%} |"
        )
    lines.extend(
        [
            "",
            "> `normalized-fixed320` is an implementation-normalization workload. Long lines are intentionally compressed, so its accuracy must not be presented as product OCR quality.",
        ]
    )

    for profile in profiles:
        title = (
            "Normalized fixed-320 / AVX2"
            if profile == "normalized-fixed320"
            else "Product profile"
        )
        lines.extend(["", f"## {title}", ""])
        if profile == "product":
            lines.extend([
                "> Recognition width policies intentionally differ here. This section "
                "measures each project's product strategy.",
                "",
            ])
        for workers in contract["workers"]:
            pairs = paired(runs, profile, workers)
            lines.extend([
                f"### {workers} worker{'s' if workers != 1 else ''}",
                "",
                "| Replica | C# median ms | C median ms | C/C# latency | "
                "C# peak MB | C peak MB |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ])
            ratios: list[float] = []
            for sharp, c_run in pairs:
                ratio = c_run.median_ms / sharp.median_ms
                ratios.append(ratio)
                lines.append(
                    f"| {sharp.replica} | {sharp.median_ms:.3f} | {c_run.median_ms:.3f} | "
                    f"{ratio:.3f}x | {sharp.peak_mb:.1f} | {c_run.peak_mb:.1f} |"
                )
            lines.extend([
                "",
                f"Paired C/C# latency: **{ratio_summary(ratios)}**.",
                "",
            ])

        lines.extend([
            "### Accuracy",
            "",
            "| Workers | Engine | Median exact line rate | Median CER | Replica range |",
            "| ---: | --- | ---: | ---: | ---: |",
        ])
        for workers in contract["workers"]:
            for engine, label in (("sharp", "C#"), ("c", "C")):
                values = accuracy_runs(runs, engine, profile, workers)
                exact_rates = [
                    run.accuracy["exact_lines"] / run.accuracy["total_lines"]
                    for run in values
                ]
                cer_values = [float(run.accuracy["cer"]) for run in values]
                lines.append(
                    f"| {workers} | {label} | {statistics.median(exact_rates):.2%} | "
                    f"{statistics.median(cer_values):.2%} | "
                    f"exact {min(exact_rates):.2%}-{max(exact_rates):.2%}; "
                    f"CER {min(cer_values):.2%}-{max(cer_values):.2%} |"
                )

        lines.extend([
            "",
            "### Paired memory ratios",
            "",
            "| Workers | Peak WS C/C# | Peak growth C/C# |",
            "| ---: | ---: | ---: |",
        ])
        for workers in contract["workers"]:
            pairs = paired(runs, profile, workers)
            peak_ratios = [c.peak_mb / s.peak_mb for s, c in pairs]
            growth_ratios = [
                c.peak_delta_mb / s.peak_delta_mb
                for s, c in pairs
                if s.peak_delta_mb > 0
            ]
            growth = ratio_summary(growth_ratios) if growth_ratios else "n/a"
            lines.append(
                f"| {workers} | {ratio_summary(peak_ratios)} | {growth} |"
            )

    lines.extend([
        "",
        "## Gate policy",
        "",
        "- Contract, SHA identity, case completeness, AVX2 parity, paired-runner "
        "identity, C determinism and result structure are gating.",
        "- Performance and accuracy values are informational in schema v1.",
        "",
    ])
    return chr(10).join(lines)


def build_report(
    contract_path: Path,
    input_dir: Path,
    dataset_dir: Path,
    model_sha256s: Path,
    output_dir: Path,
    lw_commit: str,
    *,
    include_product: bool,
    keep_detailed_results: bool,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    contract["_path"] = str(contract_path)
    lw_commit = require_full_sha(lw_commit, "lw_commit")
    references, dataset_info = validate_dataset(
        dataset_dir, contract["dataset"]["expected_images"]
    )
    model_hashes = read_sha256s(model_sha256s)
    for name in ("det.lwm", "cls.lwm", "rec.lwm", "ppocr_keys.txt"):
        if name not in model_hashes:
            raise ComparisonError(f"model checksum file lacks {name}")
    runs = load_runs(
        input_dir,
        contract,
        lw_commit,
        dataset_info["manifest_sha256"],
        model_hashes,
        include_product,
    )
    profiles = ["normalized-fixed320"] + (["product"] if include_product else [])
    contract_sha = sha256_file(contract_path)
    markdown = build_markdown(
        runs, contract, lw_commit, contract_sha, dataset_info, profiles
    )
    disagreements, disagreement_summary = build_disagreements(
        runs, references, profiles, contract["workers"]
    )
    line_cases, contributors, line_summary = build_line_diagnostics(
        runs, references, profiles, contract["workers"]
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ComparisonError(f"output directory must be empty: {output_dir}")
    (output_dir / "contract").mkdir(parents=True)
    (output_dir / "assets").mkdir(parents=True)
    (output_dir / "disagreements").mkdir(parents=True)
    shutil.copy2(contract_path, output_dir / "contract" / contract_path.name)
    shutil.copy2(model_sha256s, output_dir / "assets" / "lw-models-SHA256SUMS")
    dataset_sums = dataset_dir / "SHA256SUMS"
    if dataset_sums.is_file():
        shutil.copy2(dataset_sums, output_dir / "assets" / "dataset-SHA256SUMS")

    for run in runs:
        replica_dir = output_dir / f"replica-{run.replica}"
        replica_dir.mkdir(exist_ok=True)
        shutil.copy2(run.path, replica_dir / run.path.name)
    for manifest in input_dir.rglob("replica-manifest.json"):
        replica = load_json(manifest)["replica"]
        shutil.copy2(
            manifest,
            output_dir / f"replica-{replica}" / "replica-manifest.json",
        )

    (output_dir / "SUMMARY.md").write_text(
        markdown + chr(10), encoding="utf-8", newline=chr(10)
    )
    (output_dir / "disagreements" / "SUMMARY.md").write_text(
        chr(10).join(disagreement_summary + line_summary) + chr(10),
        encoding="utf-8",
        newline=chr(10),
    )
    if keep_detailed_results:
        for name, items in disagreements.items():
            (output_dir / "disagreements" / f"{name}.json").write_text(
                json.dumps(
                    {"schema_version": 1, "cases": items},
                    ensure_ascii=False,
                    indent=2,
                )
                + chr(10),
                encoding="utf-8",
                newline=chr(10),
            )
        (output_dir / "disagreements" / "line-cases.json").write_text(
            json.dumps(
                {"schema_version": 1, "alignment": "reference-line-dp", "cases": line_cases},
                ensure_ascii=False,
                indent=2,
            )
            + chr(10),
            encoding="utf-8",
            newline=chr(10),
        )
        (output_dir / "disagreements" / "cer-contributors.json").write_text(
            json.dumps(
                {"schema_version": 1, "edit_unit": "unicode-code-point", "cases": contributors},
                ensure_ascii=False,
                indent=2,
            )
            + chr(10),
            encoding="utf-8",
            newline=chr(10),
        )

    final_manifest = {
        "schema_version": 1,
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lw_commit": lw_commit,
        "benchmark_harness": {
            "repository": contract["simd_repository"],
            "commit": contract["simd_ref"],
        },
        "contract_sha256": contract_sha,
        "dataset": dataset_info,
        "model_assets": {
            name: model_hashes[name]
            for name in ("det.lwm", "cls.lwm", "rec.lwm", "ppocr_keys.txt")
        },
        "c_runtime_dlls": {
            str(replica): next(
                run.meta["lw_dll_sha256"]
                for run in runs
                if run.replica == replica and run.engine == "c"
            )
            for replica in range(1, contract["replicas"] + 1)
        },
        "replicas": contract["replicas"],
        "profiles": profiles,
        "cases": len(runs),
        "comparison_policy": contract["comparison_policy"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(final_manifest, ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8",
        newline=chr(10),
    )
    return final_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-sha256s", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lw-commit", required=True)
    parser.add_argument("--include-product", action="store_true")
    parser.add_argument("--keep-detailed-results", action="store_true")
    args = parser.parse_args()
    try:
        manifest = build_report(
            args.contract,
            args.input,
            args.dataset,
            args.model_sha256s,
            args.output,
            args.lw_commit,
            include_product=args.include_product,
            keep_detailed_results=args.keep_detailed_results,
        )
    except (ComparisonError, OSError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
