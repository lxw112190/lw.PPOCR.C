"""Compile and run the actual C# wrapper/WinForms image-limit contract on Windows."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    args = parser.parse_args()
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    found = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-find",
                                     "MSBuild/Current/Bin/Roslyn/csc.exe"], text=True).strip().splitlines()
    if not found:
        raise RuntimeError("Visual Studio Roslyn csc.exe is required for the Windows managed contract")
    framework = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Reference Assemblies/Microsoft/Framework/.NETFramework/v4.8"
    references = ["mscorlib", "System", "System.Core", "System.Drawing", "System.Web.Extensions", "System.Windows.Forms"]
    sources = [ROOT / "examples/csharp-common" / name for name in ("NativeOcr.cs", "OcrModels.cs")]
    sources += [ROOT / "examples/csharp-winforms" / name for name in ("MainForm.cs", "OcrResultExporter.cs", "Program.cs")]
    sources += [ROOT / "tests/csharp_image_limits_contract.cs"]
    with tempfile.TemporaryDirectory(prefix="lw-csharp-limits-") as directory:
        staging = Path(directory)
        shutil.copyfile(args.native, staging / "lw_ppocr_c.dll")
        # Real sample enlarged inside a 41.6 MP canvas: exercises DET/crop/CLS/REC,
        # not just an empty image. This fixture is generated only in temp.
        large = staging / "large-41600000.png"
        with Image.new("RGB", (8000, 5200), "white") as image:
            with Image.open(ROOT / "models/ppocrv6-tiny/sample.jpg") as sample:
                with sample.resize((4000, 4000), Image.Resampling.BICUBIC) as enlarged:
                    image.paste(enlarged, (2000, 600))
            image.save(large)
        executable = staging / "image-limits-contract.exe"
        command = [found[0], "/nologo", "/noconfig", "/target:exe", "/platform:x64", "/optimize+",
                   "/warnaserror+", "/langversion:7.3", "/nostdlib+",
                   "/main:ImageLimitsContract", "/out:" + str(executable)]
        command += ["/reference:" + str(framework / (name + ".dll")) for name in references]
        subprocess.run(command + [str(path) for path in sources], check=True, timeout=90)
        subprocess.run([str(executable), str(ROOT / "models/ppocrv6-tiny"),
                        str(ROOT / "models/ppocrv6-tiny/sample.jpg"), str(large)],
                       check=True, cwd=staging, timeout=120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
