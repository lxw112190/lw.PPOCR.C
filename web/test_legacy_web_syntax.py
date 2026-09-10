"""Validate the Legacy Web packaging contract.

JavaScript grammar is checked by web/check_legacy_js_syntax.mjs. This test
only checks packaging metadata and unresolved placeholders.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def check(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "__LW_" in text:
        raise AssertionError(f"{path}: unresolved packaging placeholder")


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
    for marker in (
        'flavor: "legacy"',
        "wasmSimd128: false",
        "pdf: false",
        'jsTarget: "chrome70"',
    ):
        if marker not in sdk:
            raise AssertionError(f"Legacy SDK metadata is missing: {marker}")
    print("legacy packaging contract gate: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
