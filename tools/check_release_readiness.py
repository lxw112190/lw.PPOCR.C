#!/usr/bin/env python3
"""Check release metadata before publishing a preview or stable tag."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r"^project\(lw\.PPOCR\.C VERSION ([0-9]+\.[0-9]+\.[0-9]+) LANGUAGES C\)$", re.MULTILINE)
VALID_C_ABI_STATUSES = {"freeze-candidate", "frozen"}
VALID_LWM_STATUSES = {"freeze-candidate", "frozen"}
VALID_MODEL_STATUSES = {"primary", "supported", "analysis-only"}
VALID_SCOPE_STATUSES = {"proposal", "approved"}
VALID_LWM_POLICIES = {"frozen", "internal-preview"}
VALID_SCOPE_PLATFORMS = {
    "windows-x86_64",
    "linux-x86_64",
    "browser-wasm",
    "android-arm64",
    "linux-arm64",
    "loongarch64",
    "java-jni",
}
VALID_SCOPE_BINDINGS = {"c", "wasm", "java", "android", "http", "node"}


def load_json(path: Path, blockers: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        blockers.append(f"unable to read {path.relative_to(ROOT)}: {error}")
        return {}
    if not isinstance(value, dict):
        blockers.append(f"{path.relative_to(ROOT)} must contain a JSON object")
        return {}
    return value


def project_version(blockers: list[str]) -> str:
    try:
        text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        blockers.append(f"unable to read CMakeLists.txt: {error}")
        return ""
    match = VERSION_RE.search(text)
    if match is None:
        blockers.append("CMakeLists.txt does not declare a supported project version")
        return ""
    return match.group(1)


def check(mode: str, expected_version: str | None = None) -> dict[str, Any]:
    blockers: list[str] = []
    version = project_version(blockers)
    runtime = load_json(ROOT / "abi" / "runtime-contract-v1.json", blockers)
    candidate = load_json(ROOT / "abi" / "c-abi-v1-candidate.json", blockers)
    lwm = load_json(ROOT / "abi" / "lwm-v0.1-layout.json", blockers)
    wasm = load_json(ROOT / "abi" / "web-abi-v1-candidate.json", blockers)
    catalog = load_json(ROOT / "models" / "ppocrv6-models.json", blockers)
    release_assets = load_json(ROOT / "ci" / "release-assets.json", blockers)
    scope = load_json(ROOT / "ci" / "stable-release-scope.json", blockers)

    if expected_version is not None and version != expected_version:
        blockers.append(f"project version is {version or '<missing>'}, expected {expected_version}")
    if runtime.get("product_version") != version:
        blockers.append("runtime-contract-v1.json product_version does not match CMake")
    if candidate.get("status") not in VALID_C_ABI_STATUSES:
        blockers.append("C ABI manifest has an invalid status")
    if runtime.get("c_abi", {}).get("status") != candidate.get("status"):
        blockers.append("runtime-contract-v1.json and C ABI manifest statuses differ")
    if runtime.get("lwm_format_version") != f'{lwm.get("format_major")}.{lwm.get("format_minor")}':
        blockers.append("runtime-contract-v1.json and LWM layout versions differ")
    if runtime.get("lwm_layout_manifest") != "abi/lwm-v0.1-layout.json":
        blockers.append("runtime-contract-v1.json does not reference the LWM layout manifest")
    if lwm.get("status") not in VALID_LWM_STATUSES:
        blockers.append("LWM layout manifest has an invalid status")
    if wasm.get("abi_version") != runtime.get("wasm_host_abi_version"):
        blockers.append("WASM Host ABI manifest version does not match runtime contract")
    if wasm.get("status") not in VALID_C_ABI_STATUSES:
        blockers.append("WASM Host ABI manifest has an invalid status")
    if runtime.get("wasm_host_abi_manifest") != "abi/web-abi-v1-candidate.json":
        blockers.append("runtime contract does not reference the WASM Host ABI manifest")

    variants = catalog.get("variants")
    statuses: dict[str, str] = {}
    if not isinstance(variants, dict):
        blockers.append("model catalog variants are missing")
    else:
        for name in ("tiny", "small", "medium"):
            entry = variants.get(name)
            status = entry.get("runtime_status") if isinstance(entry, dict) else None
            if status not in VALID_MODEL_STATUSES:
                blockers.append(f"model catalog status is invalid for {name}")
            else:
                statuses[name] = status

    required_assets = release_assets.get("required_assets")
    if not isinstance(required_assets, list) or len(required_assets) != 19:
        blockers.append("release asset manifest must contain exactly 19 primary assets")
    release_line = ".".join(version.split(".")[:2])
    if not (ROOT / "docs" / f"release-readiness-v{release_line}.md").is_file():
        blockers.append(f"v{release_line} readiness checklist is missing")

    if scope.get("schema_version") != 1:
        blockers.append("stable release scope has an unsupported schema_version")
    if scope.get("release_version") != version:
        blockers.append("stable release scope version does not match CMake")
    if scope.get("status") not in VALID_SCOPE_STATUSES:
        blockers.append("stable release scope has an invalid status")
    if scope.get("lwm_policy") not in VALID_LWM_POLICIES:
        blockers.append("stable release scope has an invalid LWM policy")
    stable_models = scope.get("stable_models")
    preview_only_models = scope.get("preview_only_models")
    if not isinstance(stable_models, list) or not stable_models or not all(
        isinstance(name, str) and name for name in stable_models
    ):
        blockers.append("stable release scope must list at least one stable model")
        stable_models = []
    if not isinstance(preview_only_models, list) or not all(
        isinstance(name, str) and name for name in preview_only_models
    ):
        blockers.append("stable release scope preview_only_models is invalid")
        preview_only_models = []
    if set(stable_models) & set(preview_only_models):
        blockers.append("stable and preview-only model scopes overlap")
    stable_platforms = scope.get("stable_platforms")
    preview_only_platforms = scope.get("preview_only_platforms")
    for label, values in (
        ("stable_platforms", stable_platforms),
        ("preview_only_platforms", preview_only_platforms),
        ("stable_bindings", scope.get("stable_bindings")),
    ):
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            blockers.append(f"stable release scope {label} is invalid")
    if isinstance(stable_platforms, list) and isinstance(preview_only_platforms, list):
        if set(stable_platforms) & set(preview_only_platforms):
            blockers.append("stable and preview-only platform scopes overlap")
        unknown_platforms = sorted(
            (set(stable_platforms) | set(preview_only_platforms))
            - VALID_SCOPE_PLATFORMS
        )
        if unknown_platforms:
            blockers.append(
                "stable release scope names unknown platforms: "
                + ", ".join(unknown_platforms)
            )
    stable_bindings = scope.get("stable_bindings")
    if isinstance(stable_bindings, list):
        unknown_bindings = sorted(set(stable_bindings) - VALID_SCOPE_BINDINGS)
        if unknown_bindings:
            blockers.append(
                "stable release scope names unknown bindings: "
                + ", ".join(unknown_bindings)
            )
    if isinstance(variants, dict):
        unknown_scope_models = sorted(
            (set(stable_models) | set(preview_only_models)) - set(variants)
        )
        if unknown_scope_models:
            blockers.append(
                "stable release scope names unknown models: "
                + ", ".join(unknown_scope_models)
            )

    if mode == "stable":
        if not (ROOT / "docs" / f"release-notes-v{version}.md").is_file():
            blockers.append(f"v{version} release notes are missing")
        if candidate.get("status") != "frozen":
            blockers.append("C ABI is not frozen")
        if runtime.get("c_abi", {}).get("status") != "frozen":
            blockers.append("runtime contract does not mark C ABI as frozen")
        if (
            scope.get("lwm_policy") == "frozen"
            and lwm.get("status") != "frozen"
        ):
            blockers.append("LWM layout is not frozen")
        if wasm.get("status") != "frozen":
            blockers.append("WASM Host ABI is not frozen")
        if scope.get("status") != "approved":
            blockers.append("stable release scope is not approved")
        experimental = sorted(
            name
            for name in stable_models
            if statuses.get(name) == "analysis-only"
        )
        if experimental:
            blockers.append(
                "stable scope contains analysis-only model variants: "
                + ", ".join(experimental)
            )

    return {
        "schema_version": 1,
        "status": "ready" if not blockers else "blocked",
        "mode": mode,
        "project_version": version,
        "model_status": statuses,
        "stable_scope": scope,
        "blockers": blockers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preview", "stable"), default="preview")
    parser.add_argument("--version", help="expected CMake project version")
    args = parser.parse_args(argv)
    report = check(args.mode, args.version)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
