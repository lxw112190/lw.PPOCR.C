#!/usr/bin/env python3
"""Validate the complete top-level GitHub Release asset contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


VERSION_RE = re.compile(
    r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?$"
)
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ReleaseAssetError(ValueError):
    """Raised when the release directory violates the manifest contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expand_name(template: str, version: str) -> str:
    if not isinstance(template, str) or not template:
        raise ReleaseAssetError("asset name must be a non-empty string")
    try:
        parts = list(string.Formatter().parse(template))
    except ValueError as error:
        raise ReleaseAssetError(
            f"invalid asset name placeholder in {template!r}: {error}"
        ) from error
    fields = [field for _, field, _, _ in parts if field]
    formatted_fields = [
        field
        for _, field, format_spec, conversion in parts
        if field and (format_spec or conversion)
    ]
    if fields != ["version"] or formatted_fields:
        raise ReleaseAssetError(
            f"asset name must contain exactly one {{version}} placeholder: {template!r}"
        )
    try:
        name = template.format(version=version)
    except (KeyError, ValueError) as error:
        raise ReleaseAssetError(f"cannot expand asset name {template!r}: {error}") from error
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ReleaseAssetError(f"asset name must be a top-level filename: {name!r}")
    return name


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseAssetError(f"cannot read release manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ReleaseAssetError("release manifest root must be an object")
    if set(manifest) != {"schema_version", "required_assets"}:
        raise ReleaseAssetError(
            "release manifest must contain only schema_version and required_assets"
        )
    if manifest["schema_version"] != 1:
        raise ReleaseAssetError("unsupported release manifest schema_version")
    if not isinstance(manifest["required_assets"], list) or not manifest["required_assets"]:
        raise ReleaseAssetError("required_assets must be a non-empty array")
    return manifest


def release_plan(manifest_path: Path, version: str) -> tuple[set[str], dict[str, set[str]]]:
    if not VERSION_RE.fullmatch(version):
        raise ReleaseAssetError(f"invalid release version: {version!r}")
    manifest = load_manifest(manifest_path)
    assets: set[str] = set()
    folded_assets: set[str] = set()
    checksum_groups: dict[str, set[str]] = defaultdict(set)

    for index, entry in enumerate(manifest["required_assets"]):
        if not isinstance(entry, dict) or set(entry) != {"name", "checksum"}:
            raise ReleaseAssetError(
                f"required_assets[{index}] must contain only name and checksum"
            )
        name = expand_name(entry["name"], version)
        folded = name.casefold()
        if name in assets or folded in folded_assets:
            raise ReleaseAssetError(f"duplicate release asset name: {name}")
        assets.add(name)
        folded_assets.add(folded)

        checksum = entry["checksum"]
        if checksum == "sidecar":
            checksum_name = name + ".sha256"
        elif isinstance(checksum, dict) and set(checksum) == {"file"}:
            checksum_name = expand_name(checksum["file"], version)
        else:
            raise ReleaseAssetError(
                f"unsupported checksum contract for release asset {name}"
            )
        checksum_groups[checksum_name].add(name)

    checksum_names = set(checksum_groups)
    collisions = sorted(assets & checksum_names)
    if collisions:
        raise ReleaseAssetError(
            f"release assets collide with checksum filenames: {collisions}"
        )
    expected = assets | checksum_names
    folded_expected: dict[str, str] = {}
    for name in expected:
        folded = name.casefold()
        previous = folded_expected.get(folded)
        if previous is not None and previous != name:
            raise ReleaseAssetError(
                f"release filenames collide case-insensitively: {previous}, {name}"
            )
        folded_expected[folded] = name
    return assets, dict(checksum_groups)


def parse_checksum_file(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise ReleaseAssetError(f"cannot read checksum file {path.name}: {error}") from error
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+(\*?)(.+)", line)
        if match is None:
            raise ReleaseAssetError(
                f"invalid checksum line {path.name}:{line_number}"
            )
        digest, _, name = match.groups()
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ReleaseAssetError(
                f"checksum entry must use a top-level filename: {path.name}:{line_number}"
            )
        if name in entries:
            raise ReleaseAssetError(f"duplicate checksum entry in {path.name}: {name}")
        if not SHA256_RE.fullmatch(digest):
            raise ReleaseAssetError(f"invalid SHA-256 in {path.name}: {name}")
        entries[name] = digest.lower()
    if not entries:
        raise ReleaseAssetError(f"checksum file is empty: {path.name}")
    return entries


def verify_release_assets(
    manifest_path: Path,
    directory: Path,
    version: str,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    assets, checksum_groups = release_plan(manifest_path, version)
    directory = directory.resolve()
    if not directory.is_dir():
        raise ReleaseAssetError(f"release asset directory does not exist: {directory}")

    expected_files = assets | set(checksum_groups)
    for name in sorted(expected_files):
        path = directory / name
        if not path.is_file():
            raise ReleaseAssetError(f"required release file is missing: {name}")
        if path.stat().st_size <= 0:
            raise ReleaseAssetError(f"required release file is empty: {name}")

    for checksum_name, expected_entries in sorted(checksum_groups.items()):
        entries = parse_checksum_file(directory / checksum_name)
        if set(entries) != expected_entries:
            missing = sorted(expected_entries - set(entries))
            extra = sorted(set(entries) - expected_entries)
            raise ReleaseAssetError(
                f"checksum entries do not match {checksum_name}: "
                f"missing={missing}, extra={extra}"
            )
        for name, expected_digest in entries.items():
            actual_digest = sha256(directory / name)
            if actual_digest != expected_digest:
                raise ReleaseAssetError(
                    f"SHA-256 mismatch for {name}: "
                    f"expected {expected_digest}, got {actual_digest}"
                )

    actual_files = {entry.name for entry in directory.iterdir() if entry.is_file()}
    extra_files = sorted(actual_files - expected_files)
    if strict and extra_files:
        raise ReleaseAssetError(f"unknown release files: {extra_files}")
    warnings = [f"unknown release file: {name}" for name in extra_files]
    return {
        "schema_version": 1,
        "status": "ok",
        "version": version,
        "required_asset_count": len(assets),
        "verified_file_count": len(expected_files),
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = verify_release_assets(
            args.manifest, args.directory, args.version, strict=args.strict
        )
    except ReleaseAssetError as error:
        print(f"verify_release_assets.py: error: {error}", file=sys.stderr)
        return 1
    for warning in report["warnings"]:
        print(f"verify_release_assets.py: warning: {warning}", file=sys.stderr)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
