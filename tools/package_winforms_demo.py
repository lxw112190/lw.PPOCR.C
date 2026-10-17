#!/usr/bin/env python3
"""Build a .NET 4.8 x64 WinForms portable test ZIP from a static-CRT native build."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def copy(source: Path, destination: Path) -> None:
    if not source.is_file() or not source.stat().st_size:
        raise ValueError(f"missing or empty file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def resolve_package_version(version: str, release: str | None) -> str:
    value = version if release is None else release[1:] if release.startswith("v") else release
    if not re.fullmatch(re.escape(version) + r"(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?", value):
        raise ValueError("release version must match the CMake base version")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--csc", required=True, type=Path)
    parser.add_argument("--framework-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--release-version", help="Exact tag version for a CI release candidate (optional v prefix)")
    args = parser.parse_args()
    version = re.search(r"project\(lw\.PPOCR\.C VERSION (\S+) LANGUAGES C\)",
                        (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")).group(1)
    package_version = resolve_package_version(version, args.release_version)
    cache = {}
    for line in (args.build_dir / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([^#/:][^:]*):[^=]+=(.*)", line)
        if match:
            cache[match[1]] = match[2]
    if cache.get("CMAKE_PROJECT_VERSION") != version or cache.get("CMAKE_MSVC_RUNTIME_LIBRARY") != "MultiThreaded":
        raise ValueError("requires a current-version CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded build")
    if cache.get("CMAKE_GENERATOR_PLATFORM") != "x64":
        raise ValueError("native build must target x64")
    name = f"lw.PPOCR.C-{package_version}-winforms-windows-x64-net48"
    output = args.output_dir.resolve()
    staging = output / name
    archive = output / (name + ".zip")
    if staging.exists() or archive.exists():
        raise ValueError("output already exists; choose a fresh output directory")
    native = args.build_dir / "Release/lw_ppocr_c.dll"
    data = native.read_bytes()
    pe_offset = struct.unpack_from("<I", data, 0x3c)[0]
    if data[pe_offset:pe_offset + 4] != b"PE\0\0" or struct.unpack_from("<H", data, pe_offset + 4)[0] != 0x8664:
        raise ValueError("native DLL is not Windows AMD64")
    staging.mkdir(parents=True)
    copy(native, staging / "lw_ppocr_c.dll")
    sources = [ROOT / "examples/csharp-common" / name for name in ("NativeOcr.cs", "OcrModels.cs")]
    sources += [ROOT / "examples/csharp-winforms" / name for name in
                ("MainForm.cs", "OcrResultExporter.cs", "Program.cs", "Properties/AssemblyInfo.cs")]
    references = ["mscorlib", "System", "System.Core", "System.Drawing", "System.Web.Extensions", "System.Windows.Forms"]
    command = [str(args.csc.resolve()), "/nologo", "/noconfig", "/target:winexe", "/platform:x64",
               "/optimize+", "/warnaserror+", "/langversion:7.3", "/nostdlib+",
               "/out:" + str(staging / "lw.PPOCR.C.WinForms.exe")]
    command += ["/reference:" + str(args.framework_dir.resolve() / (name + ".dll")) for name in references]
    command += [str(path) for path in sources]
    subprocess.run(command, check=True)
    (staging / "lw.PPOCR.C.WinForms.exe.config").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<configuration>\n'
        '  <startup><supportedRuntime version="v4.0" sku=".NETFramework,Version=v4.8" /></startup>\n'
        '</configuration>\n', encoding="utf-8")
    catalog = json.loads((ROOT / "models/ppocrv6-models.json").read_text(encoding="utf-8"))
    for variant in ("tiny", "small", "medium"):
        model = catalog["variants"][variant]
        destination = staging / "models" if variant == "tiny" else staging / "models" / variant
        for key, filename in (("det", "det.onnx"), ("rec", "rec.onnx"),
                              ("cls", "cls.onnx"), ("dictionary", "ppocr_keys.txt")):
            asset = model[key]
            if isinstance(asset, str):
                asset = catalog["shared_assets"][asset.split(":", 1)[1]]
            source = ROOT / "models" / asset["path"]
            if digest(source) != asset["sha256"]:
                raise ValueError(f"model catalog SHA-256 mismatch: {source}")
            copy(source, destination / filename)
        copy(ROOT / "models/ppocrv6-tiny/sample.jpg", destination / "sample.jpg")
    for filename in ("LICENSE", "THIRD-PARTY-NOTICES.md", "sbom.cdx.json"):
        copy(ROOT / filename, staging / filename)
    copy(ROOT / "licenses/PaddleOCR-models-APACHE-2.0.txt", staging / "licenses/PaddleOCR-models-APACHE-2.0.txt")
    copy(ROOT / "docs/winforms-portable-demo.md", staging / "README.md")
    copy(ROOT / "models/ppocrv6-models.json", staging / "MODEL-CATALOG.json")
    for directory in ("csharp-common", "csharp-winforms"):
        for source in sorted((ROOT / "examples" / directory).rglob("*.cs")):
            copy(source, staging / "source" / directory / source.relative_to(ROOT / "examples" / directory))
    project_sources = [str(path.relative_to(ROOT / "examples")).replace("/", "\\") for path in sources]
    (staging / "source/LwPpocrDemo.csproj").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<Project ToolsVersion="15.0" DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">\n'
        '<PropertyGroup><OutputType>WinExe</OutputType><AssemblyName>lw.PPOCR.C.WinForms</AssemblyName>'
        '<TargetFrameworkVersion>v4.8</TargetFrameworkVersion><PlatformTarget>x64</PlatformTarget>'
        '<OutputPath>bin\\Release\\</OutputPath><Optimize>true</Optimize><LangVersion>7.3</LangVersion></PropertyGroup>\n'
        '<ItemGroup>' + ''.join(f'<Reference Include="{name}" />' for name in references if name != 'mscorlib')
        + '</ItemGroup>\n<ItemGroup>' + ''.join(f'<Compile Include="{name}" />' for name in project_sources)
        + '<None Include="App.config" /></ItemGroup>\n'
        '<Import Project="$(MSBuildToolsPath)\\Microsoft.CSharp.targets" />\n'
        '<Target Name="CopyNativeRuntime" AfterTargets="Build"><Copy '
        'SourceFiles="$(MSBuildProjectDirectory)\\..\\lw_ppocr_c.dll" DestinationFolder="$(OutDir)" /></Target>\n'
        '</Project>\n', encoding="utf-8")
    copy(staging / "lw.PPOCR.C.WinForms.exe.config", staging / "source/App.config")
    for variant, subdirectory in (("tiny", "models"), ("small", "models\\small"), ("medium", "models\\medium")):
        (staging / f"start-{variant}.cmd").write_text(
            '@echo off\r\ncd /d "%~dp0"\r\nstart "" "%~dp0lw.PPOCR.C.WinForms.exe" "%~dp0'
            + subdirectory + '"\r\n', encoding="ascii", newline="")
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    tracked_changes = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD"], check=False).returncode
    if tracked_changes not in (0, 1):
        raise RuntimeError("could not determine source modification state")
    (staging / "BUILD-INFO.txt").write_text(
        f"Product: lw.PPOCR.C {version}\nSource commit: {commit}\n"
        f"Package version: {package_version}\n"
        f"Tracked source modifications: {'yes' if tracked_changes else 'no'}\n"
        + ("Distribution: CI release candidate (publication requires release gates)\n"
           if args.release_version else "Distribution: local portable test build (not a tagged release)\n") +
        "OS/architecture: Windows 10/11 x64\nManaged: .NET Framework 4.8\n"
        "Native: Release, static MSVC CRT (/MT), default native dispatch\n"
        "REC maximum width: 960; Tiny default; Small/Medium Preview\n"
        "Source pixel policy: default 40 MP; configurable up to 200 MP on x64; crop limit 16 MP unchanged\n"
        "No GPU, server, Python, SDK installation or model downloads needed.\n",
        encoding="utf-8")
    files = sorted(path for path in staging.rglob("*") if path.is_file())
    checksums = "".join(f"{digest(path)}  {path.relative_to(staging).as_posix()}\n" for path in files)
    (staging / "SHA256SUMS.txt").write_text(checksums, encoding="utf-8")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output).as_posix())
    Path(str(archive) + ".sha256").write_text(f"{digest(archive)}  {archive.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "sha256": digest(archive)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
