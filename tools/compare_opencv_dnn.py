#!/usr/bin/env python3
"""Compare lw.PPOCR.C and lw.PPOCR.OpenCVDNN over the same HTTP workload."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import http.client
import json
import math
import os
import platform
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, TextIO, Tuple


REFERENCE_REPOSITORY = "https://github.com/lxw112190/lw.PPOCR.OpenCVDNN"


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Sequence[float], percentage: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 3),
        "p50": round(percentile(values, 50), 3),
        "p95": round(percentile(values, 95), 3),
        "p99": round(percentile(values, 99), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def parse_cpu_list(value: str) -> List[int]:
    cpus = set()
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            first_text, last_text = token.split("-", 1)
            first = int(first_text)
            last = int(last_text)
            if first < 0 or last < first:
                raise ValueError("CPU ranges must be non-negative and ascending")
            cpus.update(range(first, last + 1))
        else:
            cpu = int(token)
            if cpu < 0:
                raise ValueError("CPU indexes must be non-negative")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("CPU affinity list must not be empty")
    logical_cpus = os.cpu_count() or 1
    if max(cpus) >= logical_cpus:
        raise ValueError(
            f"CPU affinity index {max(cpus)} exceeds host range 0..{logical_cpus - 1}"
        )
    return sorted(cpus)


def set_process_affinity(pid: int, cpus: Sequence[int]) -> None:
    if platform.system() == "Windows":
        process_set_information = 0x0200
        process_query_limited_information = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(
            process_set_information | process_query_limited_information, False, pid
        )
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            mask = 0
            for cpu in cpus:
                if cpu >= ctypes.sizeof(ctypes.c_size_t) * 8:
                    raise ValueError("CPU affinity exceeds the current Windows processor group")
                mask |= 1 << cpu
            if not kernel32.SetProcessAffinityMask(handle, mask):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel32.CloseHandle(handle)
    elif platform.system() == "Linux":
        os.sched_setaffinity(pid, set(cpus))
    else:
        raise RuntimeError("process affinity is currently supported on Windows and Linux")


def process_rss_bytes(pid: int) -> int:
    system = platform.system()
    if system == "Windows":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        process_query_information = 0x0400
        process_vm_read = 0x0010
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        handle = kernel32.OpenProcess(
            process_query_information | process_vm_read, False, pid
        )
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                raise ctypes.WinError(ctypes.get_last_error())
            return int(counters.WorkingSetSize)
        finally:
            kernel32.CloseHandle(handle)
    if system == "Linux":
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    if system == "Darwin":
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip()) * 1024
    raise RuntimeError(f"RSS measurement is unsupported on {system}")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(character for character in normalized if not character.isspace())


def ppm_to_bmp(ppm: bytes) -> bytes:
    if not ppm.startswith(b"P6"):
        raise ValueError("benchmark input must be a binary P6 PPM")
    offset = 2
    tokens = []
    while len(tokens) < 3:
        while offset < len(ppm) and chr(ppm[offset]).isspace():
            offset += 1
        if offset < len(ppm) and ppm[offset] == ord("#"):
            while offset < len(ppm) and ppm[offset] not in (10, 13):
                offset += 1
            continue
        begin = offset
        while offset < len(ppm) and not chr(ppm[offset]).isspace():
            offset += 1
        if begin == offset or offset >= len(ppm):
            raise ValueError("benchmark PPM header is incomplete")
        tokens.append(int(ppm[begin:offset]))
    if offset >= len(ppm) or not chr(ppm[offset]).isspace():
        raise ValueError("benchmark PPM header has no pixel delimiter")
    delimiter = ppm[offset]
    offset += 1
    if delimiter == 13 and offset < len(ppm) and ppm[offset] == 10:
        offset += 1
    width, height, maximum = tokens
    if width <= 0 or height <= 0 or maximum != 255:
        raise ValueError("benchmark PPM dimensions or maximum value are invalid")
    rgb = ppm[offset:]
    if len(rgb) != width * height * 3:
        raise ValueError("benchmark PPM pixel payload length is invalid")
    row_bytes = (width * 3 + 3) & ~3
    pixel_bytes = row_bytes * height
    output = bytearray(54 + pixel_bytes)
    struct.pack_into("<2sIHHI", output, 0, b"BM", len(output), 0, 0, 54)
    struct.pack_into(
        "<IiiHHIIiiII",
        output,
        14,
        40,
        width,
        height,
        1,
        24,
        0,
        pixel_bytes,
        2835,
        2835,
        0,
        0,
    )
    for y in range(height):
        source_offset = y * width * 3
        destination_offset = 54 + (height - 1 - y) * row_bytes
        for x in range(width):
            source = source_offset + x * 3
            destination = destination_offset + x * 3
            output[destination : destination + 3] = (
                rgb[source + 2],
                rgb[source + 1],
                rgb[source],
            )
    return bytes(output)


def normalize_result(document: Mapping[str, object]) -> List[Dict[str, object]]:
    raw_lines = document.get("result")
    if not isinstance(raw_lines, list):
        raise ValueError("OCR response does not contain a result array")
    lines: List[Dict[str, object]] = []
    for line_index, raw_line in enumerate(raw_lines):
        if not isinstance(raw_line, dict) or not isinstance(raw_line.get("text"), str):
            raise ValueError(f"OCR result {line_index} has an invalid text field")
        coordinates = []
        for point in range(1, 5):
            x = raw_line.get(f"x{point}")
            y = raw_line.get(f"y{point}")
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise ValueError(f"OCR result {line_index} has invalid point {point}")
            coordinates.extend((float(x), float(y)))
        lines.append(
            {
                "text": raw_line["text"],
                "normalized_text": normalize_text(str(raw_line["text"])),
                "box": coordinates,
                "bounds": [
                    min(coordinates[0::2]),
                    min(coordinates[1::2]),
                    max(coordinates[0::2]),
                    max(coordinates[1::2]),
                ],
            }
        )
    return lines


def result_checksum(lines: Sequence[Mapping[str, object]], normalized: bool) -> str:
    key = "normalized_text" if normalized else "text"
    payload = "\n".join(str(line[key]) for line in lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compare_results(
    c_lines: Sequence[Mapping[str, object]],
    opencv_lines: Sequence[Mapping[str, object]],
    coordinate_tolerance: float,
) -> Dict[str, object]:
    compared = min(len(c_lines), len(opencv_lines))
    exact_matches = 0
    normalized_matches = 0
    absolute_deltas: List[float] = []
    lines_within_tolerance = 0
    exact_mismatch_indexes = []
    normalized_mismatch_indexes = []
    coordinate_mismatch_indexes = []
    for index in range(compared):
        c_line = c_lines[index]
        opencv_line = opencv_lines[index]
        if c_line["text"] == opencv_line["text"]:
            exact_matches += 1
        else:
            exact_mismatch_indexes.append(index)
        if c_line["normalized_text"] == opencv_line["normalized_text"]:
            normalized_matches += 1
        else:
            normalized_mismatch_indexes.append(index)
        line_deltas = [
            abs(left - right)
            for left, right in zip(c_line["bounds"], opencv_line["bounds"])
        ]
        absolute_deltas.extend(line_deltas)
        if line_deltas and max(line_deltas) <= coordinate_tolerance:
            lines_within_tolerance += 1
        else:
            coordinate_mismatch_indexes.append(index)
    counts_match = len(c_lines) == len(opencv_lines)
    return {
        "c_line_count": len(c_lines),
        "opencv_line_count": len(opencv_lines),
        "line_counts_match": counts_match,
        "exact_text_matches": exact_matches,
        "normalized_text_matches": normalized_matches,
        "exact_text_match": counts_match and exact_matches == compared,
        "normalized_text_match": counts_match and normalized_matches == compared,
        "exact_text_mismatch_indexes": exact_mismatch_indexes,
        "normalized_text_mismatch_indexes": normalized_mismatch_indexes,
        "bounds_tolerance_px": coordinate_tolerance,
        "bounds_lines_within_tolerance": lines_within_tolerance,
        "bounds_within_tolerance": counts_match
        and lines_within_tolerance == compared,
        "bounds_mismatch_indexes": coordinate_mismatch_indexes,
        "bounds_mean_absolute_delta_px": round(
            statistics.fmean(absolute_deltas), 3
        )
        if absolute_deltas
        else 0.0,
        "bounds_max_absolute_delta_px": round(max(absolute_deltas), 3)
        if absolute_deltas
        else 0.0,
        "c_text_sha256": result_checksum(c_lines, normalized=False),
        "opencv_text_sha256": result_checksum(opencv_lines, normalized=False),
        "c_normalized_text_sha256": result_checksum(c_lines, normalized=True),
        "opencv_normalized_text_sha256": result_checksum(
            opencv_lines, normalized=True
        ),
    }


def server_total_ms(document: Mapping[str, object], engine: str) -> float:
    if engine == "c":
        timing = document.get("timing_ms")
        if isinstance(timing, dict) and isinstance(timing.get("server_total"), (int, float)):
            return float(timing["server_total"])
    else:
        timing = document.get("timing")
        if isinstance(timing, dict) and isinstance(
            timing.get("server_total_ms"), (int, float)
        ):
            return float(timing["server_total_ms"])
    raise ValueError(f"{engine} OCR response does not contain server total timing")


class HttpOcrClient:
    def __init__(self, port: int, timeout_seconds: float) -> None:
        self.port = port
        self.timeout_seconds = timeout_seconds
        self.connection: Optional[http.client.HTTPConnection] = None

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def request(self, image: bytes) -> Tuple[float, Dict[str, object]]:
        for attempt in range(2):
            if self.connection is None:
                self.connection = http.client.HTTPConnection(
                    "127.0.0.1", self.port, timeout=self.timeout_seconds
                )
            started = time.perf_counter()
            try:
                self.connection.request(
                    "POST",
                    "/api/ocr",
                    body=image,
                    headers={
                        "Content-Type": "image/bmp",
                        "Connection": "keep-alive",
                    },
                )
                response = self.connection.getresponse()
                payload = response.read()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                document = json.loads(payload.decode("utf-8"))
                if response.status != 200 or document.get("ok") is not True:
                    raise RuntimeError(
                        f"OCR request failed: HTTP {response.status}: {document}"
                    )
                return elapsed_ms, document
            except (ConnectionError, http.client.HTTPException, OSError):
                self.close()
                if attempt != 0:
                    raise
        raise RuntimeError("unreachable HTTP retry state")


@dataclass
class ServiceProcess:
    name: str
    process: subprocess.Popen
    log: TextIO
    client: HttpOcrClient

    def stop(self) -> None:
        self.client.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.log.close()


def wait_for_health(process: subprocess.Popen, port: int, log: TextIO) -> None:
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log.seek(0)
            raise RuntimeError(
                f"service exited before readiness with code {process.returncode}:\n{log.read()}"
            )
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            document = json.loads(response.read().decode("utf-8"))
            if response.status == 200 and document.get("ok") is True:
                return
        except (ConnectionError, http.client.HTTPException, OSError, ValueError):
            pass
        finally:
            connection.close()
        time.sleep(0.1)
    raise RuntimeError("service did not become ready within 60 seconds")


def start_service(
    name: str,
    command: Sequence[str],
    cwd: Path,
    port: int,
    environment: Optional[Mapping[str, str]],
    affinity: Sequence[int],
    timeout_seconds: float,
) -> ServiceProcess:
    log = tempfile.TemporaryFile(mode="w+t", encoding="utf-8", errors="replace")
    creation_flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=None if environment is None else dict(environment),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creation_flags,
    )
    try:
        if affinity:
            set_process_affinity(process.pid, affinity)
        wait_for_health(process, port, log)
        return ServiceProcess(
            name=name,
            process=process,
            log=log,
            client=HttpOcrClient(port, timeout_seconds),
        )
    except Exception:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        log.close()
        raise


def read_contract(
    contract_path: Path,
    source_image: Path,
    source_models: Path,
    c_models: Path,
    opencv_package: Path,
) -> Dict[str, object]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    model_sha = contract.get("model_sha256")
    expected_source = str(contract.get("source_sha256", "")).lower()
    c_source_paths = {
        "detector": source_models / "det.onnx",
        "classifier": source_models / "cls.onnx",
        "recognizer": source_models / "rec.onnx",
        "dictionary": source_models / "ppocr_keys.txt",
    }
    c_runtime_paths = {
        "detector": c_models / "det.lwm",
        "classifier": c_models / "cls.lwm",
        "recognizer": c_models / "rec.lwm",
        "dictionary": c_models / "ppocr_keys.txt",
    }
    for path in tuple(c_source_paths.values()) + tuple(c_runtime_paths.values()):
        if not path.is_file():
            raise ValueError(f"comparison model asset does not exist: {path}")
    c_source_sha = {key: sha256_file(path) for key, path in c_source_paths.items()}
    c_runtime_sha = {key: sha256_file(path) for key, path in c_runtime_paths.items()}
    manifest_path = opencv_package / "models/ppocrv6-tiny/model.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = manifest["stages"]
    opencv_sha = {
        "detector": stages["detector"]["artifacts"]["onnx"]["sha256"],
        "classifier": stages["classifier"]["artifacts"]["onnx"]["sha256"],
        "recognizer": stages["recognizer"]["artifacts"]["onnx"]["sha256"],
        "dictionary": manifest["dictionary"]["sha256"],
    }
    source_actual = sha256_file(source_image)
    expected_model_sha = model_sha if isinstance(model_sha, dict) else {}
    return {
        "contract": str(contract_path.resolve()),
        "source_image": str(source_image.resolve()),
        "source_expected_sha256": expected_source,
        "source_actual_sha256": source_actual,
        "source_sha256_match": source_actual == expected_source,
        "model_sha256": model_sha,
        "c_source_model_sha256": c_source_sha,
        "c_source_models_match_contract": all(
            str(expected_model_sha.get(key, "")).lower() == value.lower()
            for key, value in c_source_sha.items()
        ),
        "c_runtime_asset_sha256": c_runtime_sha,
        "opencv_manifest_model_sha256": opencv_sha,
        "opencv_manifest_matches_contract": all(
            str(expected_model_sha.get(key, "")).lower() == value.lower()
            for key, value in opencv_sha.items()
        ),
    }


def validate_opencv_config(config_path: Path) -> Dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "enable_classifier": True,
        "limit_side_len": 960,
        "det_db_threshold": 0.3,
        "det_db_box_threshold": 0.6,
        "det_db_unclip_ratio": 1.6,
        "det_use_dilation": False,
        "cls_threshold": 0.9,
        "cls_batch_size": 8,
        "rec_batch_size": 8,
        "rec_concurrency": 1,
    }
    actual = {key: config.get(key) for key in expected}
    return {
        "path": str(config_path.resolve()),
        "expected": expected,
        "actual": actual,
        "matches_comparison_contract": actual == expected,
    }


def measure_batch(
    service: ServiceProcess,
    engine: str,
    image: bytes,
    iterations: int,
    client_times: List[float],
    server_times: List[float],
    rss_values: List[int],
    expected_lines: Sequence[Mapping[str, object]],
) -> None:
    for _ in range(iterations):
        client_ms, document = service.client.request(image)
        lines = normalize_result(document)
        if result_checksum(lines, normalized=False) != result_checksum(
            expected_lines, normalized=False
        ):
            raise RuntimeError(f"{service.name} OCR text changed during measurement")
        client_times.append(client_ms)
        server_times.append(server_total_ms(document, engine))
    rss_values.append(process_rss_bytes(service.process.pid))


def safe_percentage(numerator: float, denominator: float) -> float:
    return round(100.0 * numerator / denominator, 3) if denominator else 0.0


def main() -> int:
    configure_utf8_output()
    parser = argparse.ArgumentParser(
        description="Launch and compare lw.PPOCR.C and lw.PPOCR.OpenCVDNN HTTP services."
    )
    parser.add_argument("--c-server", type=Path, required=True)
    parser.add_argument("--c-models", type=Path, required=True)
    parser.add_argument("--c-www", type=Path, required=True)
    parser.add_argument("--opencv-package", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, help="P6 PPM benchmark input")
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--c-source-models", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--c-workers", type=int, default=4)
    parser.add_argument("--c-rec-max-width", type=int, default=960)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations-per-round", type=int, default=20)
    parser.add_argument("--coordinate-tolerance", type=float, default=5.0)
    parser.add_argument("--cpu-affinity", default="")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    if args.c_workers < 1 or args.c_workers > 16:
        parser.error("--c-workers must be between 1 and 16")
    if args.c_rec_max_width not in (192, 320, 480, 640, 960):
        parser.error("--c-rec-max-width must be one of 192, 320, 480, 640, or 960")
    if args.warmup < 1 or args.rounds < 1 or args.iterations_per_round < 1:
        parser.error("warmup, rounds, and iterations-per-round must be positive")
    if not math.isfinite(args.coordinate_tolerance) or args.coordinate_tolerance < 0:
        parser.error("--coordinate-tolerance must be finite and non-negative")
    affinity = parse_cpu_list(args.cpu_affinity) if args.cpu_affinity else []

    required_files = [
        args.c_server,
        args.c_www / "index.html",
        args.image,
        args.source_image,
        args.contract,
    ]
    opencv_executable = args.opencv_package / (
        "lw-ppocr-http-service.exe"
        if platform.system() == "Windows"
        else "lw-ppocr-http-service"
    )
    opencv_config = args.opencv_package / "http-service.json"
    required_files.extend((opencv_executable, opencv_config))
    for required in required_files:
        if not required.is_file():
            parser.error(f"required file does not exist: {required}")
    if not args.c_models.is_dir():
        parser.error(f"C model directory does not exist: {args.c_models}")

    try:
        contract = read_contract(
            args.contract,
            args.source_image,
            args.c_source_models,
            args.c_models,
            args.opencv_package,
        )
        opencv_config_contract = validate_opencv_config(opencv_config)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if not contract["source_sha256_match"]:
        parser.error("source image SHA-256 does not match the comparison contract")
    if not contract["c_source_models_match_contract"]:
        parser.error("pure-C source model hashes do not match the comparison contract")
    if not contract["opencv_manifest_matches_contract"]:
        parser.error("OpenCV model manifest hashes do not match the comparison contract")
    if not opencv_config_contract["matches_comparison_contract"]:
        parser.error("OpenCV HTTP service config does not match the comparison contract")

    source_ppm = args.image.read_bytes()
    image = ppm_to_bmp(source_ppm)
    c_port = free_port()
    opencv_port = free_port()
    while opencv_port == c_port:
        opencv_port = free_port()
    c_command = [
        str(args.c_server.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        str(c_port),
        "--models",
        str(args.c_models.resolve()),
        "--www",
        str(args.c_www.resolve()),
        "--ocr-workers",
        str(args.c_workers),
        "--rec-max-width",
        str(args.c_rec_max_width),
    ]
    opencv_environment = os.environ.copy()
    opencv_environment.update(
        {
            "LW_PPOCR_PORT": str(opencv_port),
            "LW_PPOCR_ENGINE_INSTANCES": "1",
            "LW_PPOCR_WORKER_THREADS": "4",
            "LW_PPOCR_LOGGING_ENABLED": "0",
        }
    )
    opencv_command = [
        str(opencv_executable.resolve()),
        "--config",
        str(opencv_config.resolve()),
    ]

    services: List[ServiceProcess] = []
    try:
        c_service = start_service(
            "lw.PPOCR.C",
            c_command,
            args.c_server.parent.resolve(),
            c_port,
            None,
            affinity,
            args.timeout,
        )
        services.append(c_service)
        opencv_service = start_service(
            "lw.PPOCR.OpenCVDNN",
            opencv_command,
            args.opencv_package.resolve(),
            opencv_port,
            opencv_environment,
            affinity,
            args.timeout,
        )
        services.append(opencv_service)

        references: Dict[str, List[Dict[str, object]]] = {}
        for key, service in (("c", c_service), ("opencv", opencv_service)):
            for _ in range(args.warmup):
                _, document = service.client.request(image)
            references[key] = normalize_result(document)

        rss_values = {
            "c": [process_rss_bytes(c_service.process.pid)],
            "opencv": [process_rss_bytes(opencv_service.process.pid)],
        }
        client_times: Dict[str, List[float]] = {"c": [], "opencv": []}
        server_times: Dict[str, List[float]] = {"c": [], "opencv": []}
        service_map = {"c": c_service, "opencv": opencv_service}
        for round_index in range(args.rounds):
            order = ("c", "opencv") if round_index % 2 == 0 else ("opencv", "c")
            for key in order:
                measure_batch(
                    service_map[key],
                    key,
                    image,
                    args.iterations_per_round,
                    client_times[key],
                    server_times[key],
                    rss_values[key],
                    references[key],
                )

        correctness = compare_results(
            references["c"], references["opencv"], args.coordinate_tolerance
        )
        engine_reports: Dict[str, object] = {}
        for key in ("c", "opencv"):
            engine_reports[key] = {
                "requests": len(client_times[key]),
                "client_latency_ms": summarize(client_times[key]),
                "server_latency_ms": summarize(server_times[key]),
                "throughput_rps": round(1000.0 / statistics.fmean(client_times[key]), 3),
                "rss_mib": {
                    "after_warmup": round(rss_values[key][0] / 1048576.0, 2),
                    "maximum_observed": round(max(rss_values[key]) / 1048576.0, 2),
                },
            }
        c_client_mean = statistics.fmean(client_times["c"])
        opencv_client_mean = statistics.fmean(client_times["opencv"])
        c_p95 = percentile(client_times["c"], 95)
        opencv_p95 = percentile(client_times["opencv"], 95)
        c_throughput = 1000.0 / c_client_mean
        opencv_throughput = 1000.0 / opencv_client_mean
        comparison = {
            "c_latency_reduction_percent": safe_percentage(
                opencv_client_mean - c_client_mean, opencv_client_mean
            ),
            "c_speedup": round(opencv_client_mean / c_client_mean, 3),
            "c_p95_reduction_percent": safe_percentage(opencv_p95 - c_p95, opencv_p95),
            "c_throughput_increase_percent": safe_percentage(
                c_throughput - opencv_throughput, opencv_throughput
            ),
        }
        version_path = args.opencv_package / "RELEASE_VERSION"
        document = {
            "schema_version": 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "reference_repository": REFERENCE_REPOSITORY,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "logical_cpus": os.cpu_count(),
                "python": platform.python_version(),
                "cpu_affinity": affinity,
            },
            "protocol": {
                "transport": "HTTP keep-alive over 127.0.0.1",
                "media_type": "image/bmp",
                "image_path": str(args.image.resolve()),
                "source_ppm_sha256": hashlib.sha256(source_ppm).hexdigest(),
                "image_sha256": hashlib.sha256(image).hexdigest(),
                "image_bytes": len(image),
                "warmup_per_engine": args.warmup,
                "rounds": args.rounds,
                "iterations_per_round": args.iterations_per_round,
                "alternating_order": True,
            },
            "configuration": {
                "c_workers": args.c_workers,
                "c_rec_max_width": args.c_rec_max_width,
                "opencv_engine_instances": 1,
                "opencv_worker_threads": 4,
                "opencv_version": version_path.read_text(encoding="utf-8").strip()
                if version_path.is_file()
                else None,
                "c_server_sha256": sha256_file(args.c_server),
                "opencv_server_sha256": sha256_file(opencv_executable),
            },
            "asset_contract": contract,
            "opencv_config_contract": opencv_config_contract,
            "correctness": correctness,
            "engines": engine_reports,
            "comparison": comparison,
        }
        rendered = json.dumps(document, ensure_ascii=False, indent=2)
        if args.output:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
            print(f"Created {output}")
        print(rendered)
        if not correctness["normalized_text_match"]:
            print("normalized OCR text differs; performance claim rejected", file=sys.stderr)
            return 3
        if not correctness["bounds_within_tolerance"]:
            print("OCR box bounds exceed tolerance; performance claim rejected", file=sys.stderr)
            return 4
        return 0
    finally:
        for service in reversed(services):
            service.stop()


if __name__ == "__main__":
    raise SystemExit(main())
