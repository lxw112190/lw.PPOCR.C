#!/usr/bin/env python3
"""Assemble a self-contained Java/JVM JNI smoke bundle."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn


PLATFORMS = {"windows-x64", "linux-x64"}
MODEL_FILES = ("det.lwm", "cls.lwm", "rec.lwm", "ppocr_keys.txt", "sample.jpg")
JAVA_FILES = ("NativeOcr.java", "OcrDemo.java")


def fail(message: str) -> NoReturn:
    raise SystemExit(f"package_java_jni.py: error: {message}")


def require_file(path: Path) -> Path:
    if not path.is_file():
        fail(f"required file is missing: {path}")
    return path


def copy_file(source: Path, destination: Path) -> None:
    require_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def copy_tree_files(source_dir: Path, destination_dir: Path, names: tuple[str, ...]) -> None:
    for name in names:
        copy_file(source_dir / name, destination_dir / name)


def copy_native(native_dir: Path, destination: Path, platform: str) -> None:
    if not native_dir.is_dir():
        fail(f"native directory is missing: {native_dir}")
    if platform == "windows-x64":
        entries = sorted(native_dir.glob("*.dll"))
        required = ("lw_ppocr_java.dll", "lw_ppocr_c.dll")
    else:
        entries = sorted(native_dir.glob("liblw_ppocr_java.so*"))
        entries += sorted(native_dir.glob("liblw_ppocr_c.so*"))
        required = ("liblw_ppocr_java.so",)
    names = {entry.name for entry in entries}
    for name in required:
        if name not in names:
            fail(f"native directory does not contain {name}: {native_dir}")
    if platform == "linux-x64" and not any(name.startswith("liblw_ppocr_c.so") for name in names):
        fail(f"native directory does not contain liblw_ppocr_c.so*: {native_dir}")
    for entry in entries:
        target = entry.resolve() if entry.is_symlink() else entry
        if not target.is_file():
            fail(f"native entry is not a regular file: {entry}")
        # Copy symlink targets as ordinary files, retaining the original name
        # as well so extracted bundles work on filesystems without symlinks.
        copy_file(target, destination / entry.name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(root: Path) -> None:
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt"),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in paths]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def java_version() -> str:
    java = shutil.which("java")
    if not java:
        return "8"
    try:
        result = subprocess.run([java, "-version"], capture_output=True, text=True, check=False)
    except OSError:
        return "8"
    text = result.stderr or result.stdout
    for line in text.splitlines():
        if 'version "' in line:
            version = line.split('version "', 1)[1].split('"', 1)[0]
            if version.startswith("1.8"):
                return "8"
            return version.split(".", 1)[0]
    return "8"


def write_build_info(root: Path, platform: str, commit: str | None) -> None:
    values = {
        "platform": platform,
        "commit": commit or os.environ.get("GITHUB_SHA", "local"),
        "branch": os.environ.get("GITHUB_REF_NAME") or os.environ.get("GITHUB_REF", "local"),
        "workflow": os.environ.get("GITHUB_WORKFLOW", "Desktop Java JNI OCR"),
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "java": java_version(),
        "runner": os.environ.get("RUNNER_OS", "local"),
        "architecture": os.environ.get("RUNNER_ARCH", "X64"),
    }
    (root / "BUILD-INFO.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=sorted(PLATFORMS), required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    native_dir = args.native_dir.resolve()
    stage_dir = args.stage_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        fail(f"output directory already exists; remove it explicitly before packaging: {output}")
    output.mkdir(parents=True)
    for name in ("native", "java", "models", "licenses"):
        (output / name).mkdir()

    copy_native(native_dir, output / "native", args.platform)
    copy_tree_files(stage_dir / "examples" / "java-jni" / "java", output / "java", JAVA_FILES)
    copy_tree_files(stage_dir / "models", output / "models", MODEL_FILES)
    copy_file(repo / "LICENSE", output / "LICENSE")
    copy_file(repo / "THIRD-PARTY-NOTICES.md", output / "THIRD-PARTY-NOTICES.md")
    copy_file(repo / "licenses" / "PaddleOCR-models-APACHE-2.0.txt",
              output / "licenses" / "PaddleOCR-models-APACHE-2.0.txt")
    copy_file(repo / "examples" / "java-jni" / "README.md", output / "README.md")
    copy_file(repo / "examples" / "java-jni" / "README.zh-CN.md", output / "README.zh-CN.md")
    write_build_info(output, args.platform, args.commit)
    write_checksums(output)
    print(f"created {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
