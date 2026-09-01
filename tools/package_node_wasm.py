"""Create the official, dependency-free Node.js/WASM release package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ASSETS = {
    "runtime": "runtime.cjs",
    "det": "det.lwm",
    "cls": "cls.lwm",
    "rec": "rec.lwm",
    "dictionary": "ppocr_keys.txt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_asset(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"missing Node/WASM package input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def package_readme(version: str) -> str:
    return f"""# lw.PPOCR.C Node/WASM runtime ({version})

This archive contains the standalone Emscripten runtime and the bundled
PP-OCRv6 tiny LWM assets. It has no npm runtime dependencies and does not
decode JPEG/PNG files. Applications should provide BGR8 pixels to the WASM
Host ABI.

## Requirements

- Node.js 18 or newer
- CommonJS `require()` support

## Minimal initialization

```javascript
const fs = require("node:fs");
const path = require("node:path");
const LwPpocrModule = require("./runtime.cjs");

async function main() {{
  const runtime = await LwPpocrModule({{}});
  runtime.FS.mkdir("/models");
  for (const name of ["det.lwm", "cls.lwm", "rec.lwm", "ppocr_keys.txt"]) {{
    runtime.FS.writeFile(`/models/${{name}}`,
      fs.readFileSync(path.join(__dirname, name)));
  }}
  const status = runtime._lw_web_init(1);
  if (status !== 0) throw new Error(`OCR initialization failed: ${{status}}`);
  console.log("lw.PPOCR.C WASM runtime ready");
  runtime._lw_web_shutdown();
}}

main().catch(error => {{ console.error(error); process.exitCode = 1; }});
```

The exported `lw_web_*` symbols are the stable WASM Host ABI v1. Use one
runtime instance from one request at a time; create separate Node Worker
instances when an application needs concurrency. See `manifest.json` and
`SHA256SUMS.txt` before loading assets.
"""


def deterministic_zip(staging: Path, archive: Path, root_name: str) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as output:
        for path in sorted(staging.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(staging).as_posix()
            info = zipfile.ZipInfo(f"{root_name}/{relative}")
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            output.writestr(info, path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--det", type=Path, required=True)
    parser.add_argument("--cls", type=Path, required=True)
    parser.add_argument("--rec", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--model-license", type=Path, required=True)
    parser.add_argument("--notices", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()

    staging = args.output_dir.resolve()
    archive = args.archive.resolve()
    if archive == staging or staging in archive.parents:
        raise SystemExit("archive must be outside the package staging directory")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    inputs = {
        "runtime": args.runtime,
        "det": args.det,
        "cls": args.cls,
        "rec": args.rec,
        "dictionary": args.dictionary,
    }
    for key, name in ASSETS.items():
        copy_asset(inputs[key], staging / name)
    copy_asset(args.license, staging / "LICENSE")
    copy_asset(args.notices, staging / "THIRD-PARTY-NOTICES.md")
    copy_asset(args.model_license, staging / "licenses" / "PaddleOCR-APACHE-2.0.txt")

    asset_hashes = {
        name: sha256(staging / name) for name in ASSETS.values()
    }
    manifest = {
        "schemaVersion": 1,
        "package": {"name": "lw.PPOCR.C-node-wasm", "version": args.version},
        "runtime": {
            "target": "node-wasm",
            "wasmHostAbiVersion": 1,
            "lwmVersion": "0.1",
            "backend": "wasm128",
            "threading": "single-threaded",
        },
        "compatibility": {"node": ">=18"},
        "features": {
            "det": True,
            "cls": True,
            "rec": True,
            "fullOcr": True,
            "adaptiveRecWidth": True,
        },
        "assets": {
            "runtime": {"path": ASSETS["runtime"], "sha256": asset_hashes[ASSETS["runtime"]]},
            "det": {"path": ASSETS["det"], "sha256": asset_hashes[ASSETS["det"]]},
            "cls": {"path": ASSETS["cls"], "sha256": asset_hashes[ASSETS["cls"]]},
            "rec": {"path": ASSETS["rec"], "sha256": asset_hashes[ASSETS["rec"]]},
            "dictionary": {
                "path": ASSETS["dictionary"],
                "sha256": asset_hashes[ASSETS["dictionary"]],
            },
        },
    }
    write_text(staging / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    write_text(staging / "README.md", package_readme(args.version))

    checksum_names = [
        "LICENSE",
        "THIRD-PARTY-NOTICES.md",
        "cls.lwm",
        "det.lwm",
        "manifest.json",
        "ppocr_keys.txt",
        "rec.lwm",
        "runtime.cjs",
        "README.md",
        "licenses/PaddleOCR-APACHE-2.0.txt",
    ]
    checksum_lines = [f"{sha256(staging / name)}  {name}" for name in checksum_names]
    write_text(staging / "SHA256SUMS.txt", "\n".join(checksum_lines) + "\n")

    root_name = f"lw.PPOCR.C-{args.version}-node-wasm"
    deterministic_zip(staging, archive, root_name)
    write_text(
        Path(f"{archive}.sha256"),
        f"{sha256(archive)}  {archive.name}\n",
    )
    print(f"wrote {archive} ({archive.stat().st_size} bytes)")
    print(f"wrote {archive}.sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
