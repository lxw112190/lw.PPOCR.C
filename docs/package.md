# Development package

The `0.x` preview archives are development packages, not ABI-frozen 1.0
releases. Platform and architecture archives are separate and must not be mixed.

## Contents

```text
bin/                         native Demos/runtime/HTTP server; optional WinForms
include/lw_infer.h           public C header
lib/                         static library and shared-library import library
lib/cmake/lw.PPOCR.C/        CMake package configuration
examples/                    native HTTP/C, C# WinForms, and Java/JVM JNI source
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
quadrilateral per line. The full-OCR demo defaults to `rec_max_width=960` and
adaptive line widths. To reproduce the legacy/default C API REC width, append
`320` as the final argument; valid values are `192`, `320`, `480`, `640`, and
`960`. The staged-package smoke test checks both the default `960` profile and
the explicit `320` override.

The default package contains the cross-platform native
`lw.PPOCR.C.HttpServer` executable and `www/`. Its installed-package test
validates health, HTML, binary P6 PPM OCR, JSON/Base64 PPM OCR, and stable
400/415 error responses. On Windows, `-DLW_BUILD_CSHARP_DEMOS=ON` additionally
packages `lw.PPOCR.C.WinForms.exe` and the native DLL beside it.
See `managed-demos.md` for commands and security boundaries.

The package also contains `examples/java-jni/`, a Java 8+ desktop consumer.
It builds a thin `lw_ppocr_java` JNI library against the installed CMake
package, uses Java `ImageIO` for JPEG/PNG/BMP input, and returns ordered text
lines. It is CI-verified on Windows x64, Linux x64, and macOS ARM64; it is not
an Android or Maven distribution. See the example's bilingual README for the
complete compile/run commands, Windows `PATH` requirement, and Linux/macOS
RPATH behavior.

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

## Java/JVM JNI CI bundles

The manually dispatchable `Desktop Java JNI OCR` workflow extends its Java 8
Windows/Linux smoke test by uploading these temporary Actions artifacts:

- `lw-ppocr-java-jni-windows-x64`;
- `lw-ppocr-java-jni-linux-x64`.

Each extracted bundle contains `native/` (the JNI library and its native
dependency closure), `java/`, the five checked-in OCR model assets under
`models/`, the two example READMEs, `BUILD-INFO.txt`, licenses, and
`SHA256SUMS.txt`. Linux shared-library symlinks are materialized as ordinary
files so the bundle also works after extraction on filesystems without symlink
support. The workflow verifies the checksum manifest and runs OCR from the
final bundle, not from the build directory. Windows additionally puts
`native/` on `PATH` because the Windows loader resolves dependent DLLs there;
Linux uses the `$ORIGIN` RPATH.

These are CI downloads retained for 14 days and are not themselves tagged
Release assets. The tagged `release.yml` workflow now waits for the same Java
workflow, verifies both bundle manifests again, and publishes:

- `lw.PPOCR.C-<version>-java-jni-windows-x64.zip` plus `.sha256`;
- `lw.PPOCR.C-<version>-java-jni-linux-x64.tar.gz` plus `.sha256`.

The published archives are still Preview integration bundles, not a stable
Java ABI promise.

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
release workflow. Stable and prerelease suffixes are accepted.
`v0.2.0-preview.1` has already been published and must not be recreated or
moved. Before creating another preview, update every product-version metadata
location and run `python -m unittest tests.test_versioning`. Then configure a
GPG, SSH, or S/MIME signing key recognized by GitHub and create the new version
as a signed tag. For example, after preparing `preview.2`:

```bash
git tag -s v0.2.0-preview.2 -m "lw.PPOCR.C v0.2.0 preview 2"
git verify-tag v0.2.0-preview.2
git push origin v0.2.0-preview.2
```

Do not fall back to replacing an existing public tag when signing is not
configured correctly. Fix the local signing setup and create the next version.

The tag rebuilds and tests native Windows/Linux packages, browser/Node WASM,
Android ARM64, Desktop Java/JNI, and the Tiny/Small/Medium runtime model packs.
The publish job runs only after all seven reusable jobs succeed. A prerelease
suffix automatically marks the GitHub Release as a prerelease; a stable tag is
marked latest.

The release asset contract is machine-readable in
[`ci/release-assets.json`](../ci/release-assets.json). It currently contains 18
primary downloads:

- Windows x64 and Linux x86_64 native archives;
- Tiny, Small, and Medium standalone HTML and browser SDK files;
- the raw Node.js/WASM package;
- Android ARM64 AAR and signed preview APK;
- Windows x64, Linux x64, and macOS ARM64 Desktop Java/JNI bundles;
- the source-model collection plus Tiny, Small, and Medium runtime model packs.

Every primary file has a SHA-256 record; the Android checksum file covers both
the AAR and APK. The final directory is checked in strict mode against the manifest,
so a missing, empty, misnamed, corrupted, or unexpected top-level asset blocks
publication. CI artifacts such as the Medium browser timing report are kept
outside this public Release directory.

For tags created from a commit containing the attestation-enabled workflow,
the publish job also creates signed GitHub build-provenance attestations for
all 18 primary downloads after strict manifest verification and before upload.
Verify a downloaded asset with GitHub CLI:

```bash
gh attestation verify PATH/TO/DOWNLOADED-ASSET \
  --repo lxw112190/lw.PPOCR.C
```

The published `v0.2.0-preview.1` tag predates this workflow enhancement and
continues to rely on its SHA-256 records and existing CI build history. An
attestation proves repository/workflow provenance; it does not replace the
checksum, SBOM review, malware scanning, or target-machine validation.

Small and Medium are opt-in preview variants. Their runtime model packs are
published only when Windows and Linux produce byte-identical archives. Their
browser SDK/HTML files are published only after model identity, 16-line
full-text golden OCR, same-engine repetition, destroy/recreate lifecycle, and
standalone-HTML OCR gates pass. PDF regression remains Tiny-only.

The Android demo APK is signed with a temporary CI key for preview testing, not
Play Store distribution. Uninstall an older preview before installing an APK
from a different CI run. The Node artifact is a raw package rather than an npm
package; Node 18/20/22 load its exact archive contents and run full OCR before
publication. See [Node/WASM distribution](NODE_WASM_DISTRIBUTION.md),
[Browser JavaScript SDK](web-sdk.md), and [runtime model packs](model-packs.md)
for their integration contracts.