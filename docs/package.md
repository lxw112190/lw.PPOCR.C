# Development package

The `0.1.x` archives are experimental development packages, not ABI-frozen 1.0
releases. Windows x64 and x86 archives are separate and must not be mixed.

## Contents

```text
bin/                         native Demos/runtime/HTTP server; optional WinForms
include/lw_infer.h           public C header
lib/                         static library and shared-library import library
lib/cmake/lw.PPOCR.C/        CMake package configuration
examples/                    native HTTP/C consumers and C# WinForms source
models/rec.lwm               converted PP-OCRv6 tiny REC model
models/cls.lwm               converted PP-OCRv6 tiny CLS model
models/det.lwm               converted DET model for the public detector
models/ppocr_keys.txt        UTF-8 recognition dictionary
models/sample-crop.ppm       dependency-free demo input
models/sample.ppm            full-image DET demo input
models/sample.jpg            C# WinForms image-decoding demo input
www/index.html               native HTTP OCR browser page
docs/                        API documentation and support image assets
LICENSE
README.md
README.zh-CN.md
THIRD-PARTY-NOTICES.md
dependencies.lock.json
sbom.cdx.json
BUILD-ENVIRONMENT.txt         CI toolchain/baseline report when provided
```

## Run the packaged demo

From an extracted Windows archive:

```powershell
.\bin\lw-recognize-ppm.exe `
  .\models\rec.lwm `
  .\models\ppocr_keys.txt `
  .\models\sample-crop.ppm
```

Expected text:

```text
text=纯臻营养护发素
```

The demo intentionally supports only binary P6 PPM. This keeps it pure C and
dependency-free while demonstrating the complete public recognizer API. The
core library accepts decoded BGR8 pixels; production applications may use their
own JPEG/PNG decoder.

Run the direction-classification demo against the same decoded crop:

```powershell
.\bin\lw-classify-ppm.exe `
  .\models\cls.lwm `
  .\models\sample-crop.ppm
```

It reports label `0`/`1`, orientation `0`/`180`, Softmax score, and resized
content width. It reports orientation but does not rotate the image.

Run the text-detection Demo against the full-image fixture:

```powershell
.\bin\lw-detect-ppm.exe `
  .\models\det.lwm `
  .\models\sample.ppm
```

It prints clockwise quadrilateral coordinates in the original image coordinate
system and one score per detected region. The package smoke test executes this
installed binary and requires a non-empty result.

Run the complete OCR Demo against the same full-image fixture:

```powershell
.\bin\lw-ocr-ppm.exe `
  .\models\det.lwm `
  .\models\cls.lwm `
  .\models\rec.lwm `
  .\models\ppocr_keys.txt `
  .\models\sample.ppm
```

It runs DET, pure-C perspective crop, CLS direction correction, and REC, then
prints UTF-8 text, all stage scores, applied rotation, and one original-image
quadrilateral per line. The staged-package smoke test builds the installed
consumer and requires the expected sample title from this installed binary.

The default package contains the cross-platform native
`lw.PPOCR.C.HttpServer` executable and `www/`. Its installed-package test
validates health, HTML, binary P6 PPM OCR, JSON/Base64 PPM OCR, and stable
400/415 error responses. On Windows, `-DLW_BUILD_CSHARP_DEMOS=ON` additionally
packages `lw.PPOCR.C.WinForms.exe` and the native DLL beside it.
See `managed-demos.md` for commands and security boundaries.

Start the native HTTP Demo from the extracted package root:

```powershell
.\bin\lw.PPOCR.C.HttpServer.exe
```

It resolves `models/` and `www/` relative to the executable and listens on
`http://127.0.0.1:8787/` by default. Linux/macOS packages use the same layout
and the executable name without `.exe`.

## Run the scalar benchmark

The packaged benchmark reuses one recognizer and emits machine-readable JSON:

```powershell
.\bin\lw-rec-benchmark.exe `
  .\models\rec.lwm `
  .\models\ppocr_keys.txt `
  .\models\sample-crop.ppm `
  3 20
```

The last two values select warm-up and measured iterations. Latency and RSS are
environment-dependent measurements, not release-wide performance guarantees.

The complete-pipeline benchmark reports reusable-handle DET latency, full OCR
latency, and the remaining crop/CLS/REC time. Its final argument selects the
line worker count:

```powershell
.\bin\lw-ocr-benchmark.exe `
  .\models\det.lwm `
  .\models\cls.lwm `
  .\models\rec.lwm `
  .\models\ppocr_keys.txt `
  .\models\sample.ppm `
  3 10 4 960
```

## Consume with CMake

Point `CMAKE_PREFIX_PATH` at the extracted package, then use either target:

```cmake
find_package(lw.PPOCR.C CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE lw.PPOCR.C::shared)
# or: lw.PPOCR.C::static
```

Consumers of `lw.PPOCR.C::shared` receive the required DLL import definition
through the imported target. Copy the DLL beside the application on Windows.

## Build a local archive

After configuring and building a Release tree:

```powershell
cpack --config .\build\CPackConfig.cmake -C Release -G ZIP
```

CPack writes the archive and its `.sha256` checksum under `build/packages`.
The package test installs to a clean staging directory, builds consumers using
the installed CMake package, and runs the installed Demos against installed
models, dictionary, and fixtures before packaging.

## Build customer Linux architecture packages

Run the manually dispatched `customer-linux-architectures` workflow from the
GitHub Actions page. It prepares the platform-independent LWM/PPM assets once,
then produces these independent CI artifacts:

- `lw.PPOCR.C-linux-amd64-ubuntu22.04` on a native amd64 runner;
- `lw.PPOCR.C-linux-arm64-ubuntu22.04` on a native ARM64 runner;
- `lw.PPOCR.C-linux-loongarch64-debian13-qemu-experimental` under QEMU.

The CPack filenames carry the same architecture/baseline suffix after the
project version, so extracted files remain identifiable after being downloaded
from the Actions artifact wrapper.

The architecture jobs do not need ONNX Runtime or converter wheels. They reuse
checksummed LWM assets, build both the static/shared runtime and native Demos,
install into a clean staging directory, compile an installed-package consumer,
and run real REC, DET, full OCR, and HTTP smoke tests. The resulting CPack
archive and its `.sha256` file are uploaded together. Each archive includes
`BUILD-ENVIRONMENT.txt` with its commit, compiler, CMake, glibc, architecture,
and whether execution was native or emulated.

The LoongArch64 artifact is an experimental customer-validation build. After
checking its `.sha256`, run it on the customer's actual system and record at
least the following before making a compatibility claim:

```bash
uname -m
cat /etc/os-release
getconf GNU_LIBC_VERSION
file ./bin/lw-ocr-ppm
ldd ./bin/lw-ocr-ppm
./bin/lw-ocr-ppm ./models/det.lwm ./models/cls.lwm \
  ./models/rec.lwm ./models/ppocr_keys.txt ./models/sample.ppm
```

The ARM64 package selects NEON and currently accelerates packed pointwise Conv,
regular 3x3 Conv, and 2x2 ConvTranspose. The LoongArch64 package detects LSX and
LASX through Linux HWCAP; LSX-capable CPUs use the LSX packed pointwise kernel,
LASX CPUs currently reuse that LSX implementation, and older CPUs remain on
the scalar fallback. The workflow verifies backend selection and direct Conv
correctness, but these packages are not performance-equivalent claims for the
amd64 build. Do not attach the manual artifacts to a tagged release until the
jobs are green and the intended support policy has been reviewed. LoongArch
performance claims additionally require the exact archive on physical customer
hardware.

## Publish a tagged release

Pushing a tag whose base version matches the CMake project version starts the
release workflow. Stable and prerelease suffixes are accepted, for example:

```bash
git tag -a v0.1.0-preview.1 -m "lw.PPOCR.C v0.1.0 preview 1"
git push origin v0.1.0-preview.1
```

The workflow rebuilds and tests the Windows x64, Linux x64, and browser WASM
packages from the tagged commit. A GitHub Release is created only after all
three jobs succeed. The Release contains the native archives, reusable
JavaScript SDK, offline HTML, and a SHA-256 file for every downloadable asset.
Tags with a `0.x` version or a prerelease suffix are automatically marked as
GitHub prereleases.
