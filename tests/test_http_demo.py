from __future__ import annotations

import argparse
import base64
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple


def request_json(
    url: str, data: Optional[bytes] = None, content_type: Optional[str] = None
) -> Tuple[int, Dict[str, object]]:
    headers = {} if content_type is None else {"Content-Type": content_type}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--www", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    args = parser.parse_args()
    for required in (args.server, args.sample, args.www / "index.html"):
        if not required.is_file():
            raise AssertionError(f"missing HTTP Demo file: {required}")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [
            str(args.server),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--models", str(args.models),
            "--www", str(args.www),
        ],
        cwd=args.server.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while True:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise AssertionError(f"HTTP Demo exited early:\n{output}")
            try:
                status, health = request_json(base_url + "/health")
                if status == 200 and health.get("ok") is True:
                    break
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise AssertionError("HTTP Demo did not become ready")
            time.sleep(0.1)

        with urllib.request.urlopen(base_url + "/", timeout=10) as response:
            page = response.read().decode("utf-8")
            assert response.status == 200 and 'id="overlay"' in page

        image = args.sample.read_bytes()
        status, binary = request_json(
            base_url + "/api/ocr", image, "image/x-portable-pixmap"
        )
        assert status == 200 and binary.get("ok") is True
        result = binary.get("result")
        assert isinstance(result, list) and len(result) == 16
        assert result[0]["text"] == "纯臻营养护发素"
        assert binary.get("request_id")

        body = json.dumps(
            {"imageBase64": base64.b64encode(image).decode("ascii")}
        ).encode("utf-8")
        status, encoded = request_json(base_url + "/api/ocr", body, "application/json")
        assert status == 200 and encoded.get("ok") is True
        assert encoded["result"][0]["text"] == result[0]["text"]

        status, rejected = request_json(base_url + "/api/ocr", b"bad", "image/jpeg")
        assert status == 415 and rejected.get("error_code") == "unsupported_media_type"
        status, rejected = request_json(
            base_url + "/api/ocr", b"P6\n1 1\n255\n", "image/x-portable-pixmap"
        )
        assert status == 400 and rejected.get("error_code") == "bad_request"
        status, rejected = request_json(base_url + "/missing")
        assert status == 404 and rejected.get("error_code") == "not_found"
        print("native HTTP smoke passed: binary/json=16 lines, media=415, bad PPM=400, missing=404")
        return 0
    finally:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
