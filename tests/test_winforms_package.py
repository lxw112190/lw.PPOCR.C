"""Verify and run the exact WinForms ZIP on Windows (not a rebuilt substitute)."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import os
import re
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--csc", required=True, type=Path)
    parser.add_argument("--framework-dir", required=True, type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    sidecar = Path(str(archive) + ".sha256").read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[a-f0-9]{64}  " + re.escape(archive.name), sidecar) or digest(archive) != sidecar[:64]:
        raise ValueError("WinForms archive SHA-256 mismatch")
    with tempfile.TemporaryDirectory(prefix="lw-winforms-exact-package-") as temporary:
        destination = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.namelist()
            if len(members) != len(set(members)):
                raise ValueError("duplicate ZIP entry")
            for name in members:
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                    raise ValueError("unsafe ZIP entry")
                if path.parts[0] != archive.stem:
                    raise ValueError("unexpected archive root")
            if bundle.testzip() is not None:
                raise ValueError("ZIP CRC mismatch")
            bundle.extractall(destination)
        package = destination / archive.stem
        checksums = {}
        for line in (package / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            sha, name = line.split("  ", 1)
            path = PurePosixPath(name)
            if name in checksums or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError("unsafe or duplicate checksum path")
            if not re.fullmatch(r"[a-f0-9]{64}", sha) or digest(package / name) != sha:
                raise ValueError(f"package checksum mismatch: {name}")
            checksums[name] = sha
        files = {path.relative_to(package).as_posix() for path in package.rglob("*") if path.is_file()}
        if files != set(checksums) | {"SHA256SUMS.txt"}:
            raise ValueError("package inventory differs from checksum manifest")
        for name in ("lw.PPOCR.C.WinForms.exe", "lw.PPOCR.C.WinForms.exe.config", "lw_ppocr_c.dll",
                     "source/LwPpocrDemo.csproj", "README.md", "LICENSE", "BUILD-INFO.txt",
                     "models/det.onnx", "models/small/rec.onnx", "models/medium/rec.onnx"):
            if name not in checksums or (package / name).stat().st_size == 0:
                raise ValueError(f"missing required WinForms asset: {name}")
        references = ["mscorlib", "System", "System.Core", "System.Drawing", "System.Windows.Forms"]
        smoke = package / "winforms-package-smoke.exe"
        command = [str(args.csc.resolve()), "/nologo", "/noconfig", "/target:exe", "/platform:x64",
                   "/nostdlib+", "/warnaserror+", "/out:" + str(smoke),
                   "/reference:" + str(package / "lw.PPOCR.C.WinForms.exe")]
        command += ["/reference:" + str(args.framework_dir.resolve() / (name + ".dll")) for name in references]
        subprocess.run(command + [str(ROOT / "tests/winforms_package_smoke.cs")], check=True, timeout=90)
        environment = os.environ.copy()
        windows = Path(environment.get("SystemRoot", "C:/Windows"))
        environment["PATH"] = str(windows / "System32") + ";" + str(windows)
        subprocess.run([str(smoke), str(package), str(ROOT / "ci")], check=True,
                       cwd=package, env=environment, timeout=180)
        # The smoke must not rewrite the tested distribution's bytes.
        for name, sha in checksums.items():
            if digest(package / name) != sha:
                raise ValueError(f"smoke modified packaged file: {name}")
        print(f"exact WinForms ZIP, {len(checksums)} hashes and six OCR cases: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
