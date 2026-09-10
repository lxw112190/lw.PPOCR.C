"""Check the Legacy Web application layer stays within the ES2018 contract."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

FORBIDDEN = (
    (re.compile(r"\?\."), "optional chaining"),
    (re.compile(r"\?\?="), "nullish assignment"),
    (re.compile(r"\?\?"), "nullish coalescing"),
)


def application_layer(text: str) -> str:
    """Remove compiler-owned Emscripten runtime text before checking syntax."""
    marker = "(function(global) {"
    start = text.find(marker)
    if start >= 0:
        text = text[start:]
    start = text.find("const WORKER_RUNTIME_SOURCE = ")
    end = text.find("const READING_ORDER_VALUES", start)
    if start >= 0 and end >= 0:
        text = text[:start] + 'const WORKER_RUNTIME_SOURCE = "";\n  ' + text[end:]
    return text


def check(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "__LW_" in text:
        raise AssertionError(f"{path}: unresolved packaging placeholder")
    checked = application_layer(text)
    for pattern, label in FORBIDDEN:
        match = pattern.search(checked)
        if match:
            line = checked.count("\n", 0, match.start()) + 1
            raise AssertionError(f"{path}: {label} at application line {line}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.sdk, args.html):
        if not path.is_file():
            raise SystemExit(f"missing Legacy artifact: {path}")
        check(path)
    sdk = args.sdk.read_text(encoding="utf-8")
    if 'flavor: "legacy"' not in sdk:
        raise AssertionError("Legacy SDK buildInfo.flavor is missing")
    if "wasmSimd128: false" not in sdk or "pdf: false" not in sdk:
        raise AssertionError("Legacy SDK buildInfo flags are incorrect")
    print("legacy ES2018 application syntax gate: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
