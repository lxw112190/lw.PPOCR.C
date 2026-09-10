"""Check the Legacy Web artifacts stay within the ES2018 syntax contract."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

FORBIDDEN = (
    (re.compile(r"\?\."), "optional chaining"),
    (re.compile(r"\?\?="), "nullish assignment"),
    (re.compile(r"\?\?"), "nullish coalescing"),
)

def check(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "__LW_" in text:
        raise AssertionError(f"{path}: unresolved packaging placeholder")
    for pattern, label in FORBIDDEN:
        match = pattern.search(text)
        if match:
            line = text.count("\n", 0, match.start()) + 1
            raise AssertionError(f"{path}: {label} at line {line}")
    
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
    print("legacy ES2018 syntax gate: ok")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
