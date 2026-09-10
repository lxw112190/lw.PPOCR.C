# lw.PPOCR.C

[简体中文](README.zh-CN.md) | English

Lightweight pure-C inference runtime for PP-OCR.

`lw.PPOCR.C` runs PP-OCR without Python, OpenCV, ONNX Runtime, OpenVINO,
TensorRT, protobuf, or any other deployment-time runtime dependency.

> This is not a general-purpose ONNX Runtime.

## Current preview: v0.2.0-preview.1

This preview release makes PP-OCRv6 Tiny, Small, and Medium available through one
pure-C Runtime. Tiny remains the default. Small and Medium are opt-in preview
variants with separate runtime model packs and self-contained browser files.
The C ABI and LWM v0.1 format are still not frozen.

Download from [GitHub Releases](https://github.com/lxw112190/lw.PPOCR.C/releases)
according to the integration target:

| Need | Release asset | Notes |
|---|---|---|
| Open an offline OCR page | `*-ocr-demo.html` | Tiny; recommended for phones and general use |
| Try a larger browser model | `*-ocr-demo-small.html` / `*-ocr-demo-medium.html` | Small is opt-in; Medium is desktop-first |
| Embed OCR in a Web page | `*-web-sdk.js` / `*-web-sdk-small.js` / `*-web-sdk-medium.js` | Same `LwPpocr` API, different embedded models |
| Native C/C++ | `*-windows-x64-msvc.zip` / `*-linux-x86_64.tar.gz` | Tiny is bundled by default |
| Android ARM64 | `*-android-arm64.aar` / `*-android-arm64-demo.apk` | `arm64-v8a`, `minSdk 21`, Tiny |
| Desktop Java/JNI | `*-java-jni-windows-x64.zip` / `*-java-jni-linux-x64.tar.gz` / `*-java-jni-macos-arm64.tar.gz` | Java 8+ console integration |
| Node.js | `*-node-wasm.zip` | Raw Node 18+ WASM package, Tiny |
| Change native model variant | `*-ppocrv6-{tiny,small,medium}-runtime.zip` | Namespaced model pack with manifest and hashes |

Download checksum files with the selected assets and verify them before use.
The Android `SHA256SUMS.txt` covers both its AAR and APK; other primary assets
have a sidecar `.sha256`. Do not mix binaries, SDK files, models, or dictionaries
from different releases. When upgrading from `0.1.x`, replace the complete
matching set and rebuild native/managed consumers because the ABI is not frozen.
See the [package guide](docs/package.md) and
[model selection matrix](docs/supported-models.md).

## Current milestone

The project currently provides:

- CI-gated PP-OCRv6 Tiny/Small/Medium runtime model packs, with Tiny as the
  default and Small/Medium as opt-in preview variants.
- Safe Conv + BatchNormalization folding during REC/CLS conversion.
- A bounds-checked C11 model loader, dynamic shape propagation, and reusable
  workspace planning.
- Public C APIs for recognition, fixed-batch direction classification,
  detection, and composed full OCR.
- Pure-C BGR preprocessing, perspective crop, UTF-8 CTC decoding, DB-style
  detection postprocessing, and reading-order quadrilateral output.
- FP32 CPU inference with scalar fallback and runtime-dispatched x86
  SSE2/AVX2, AArch64 NEON, and LoongArch LSX kernels. Full OCR can use a fixed
  DET operator pool and then recognize independent detected lines in parallel.
  Native builds derive the DET pool from the process CPU budget independently
  of the public CLS/REC line-worker setting; x86 and WebAssembly stay serial.
- Optional .NET Framework 3.5 WinForms, desktop Java/JVM JNI, native
  `cpp-httplib` HTTP/web, and offline single-file WebAssembly demos.

The WinForms and standalone browser Demos can copy recognized text and export
UTF-8 TXT or versioned JSON using the shared
[OCR result export schema](docs/ocr-export-schema.md).
The standalone browser Demo can select, drag-and-drop, or paste clipboard
images directly for OCR. It also opens local image-based PDFs, recognizes the
current page or all pages sequentially, and exports page-aware schema v2 JSON.

The deployment core accepts decoded BGR8 pixels. Image decoding remains an
application concern, so the core itself stays dependency-free.

### Platform focus

- Primary targets: Windows x64 and Linux x64.
- A manual customer workflow builds native Linux ARM64 and an experimental
  QEMU-validated Linux LoongArch64 package; physical customer hardware remains
  a separate validation gate.
- Windows 7 x86 compatibility is preserved by design.
- The LWM v0.1 format is custom and not yet frozen.

ARM64 uses NEON for packed pointwise Conv, regular 3x3 Conv, and 2x2
ConvTranspose. LoongArch64 detects LSX/LASX through Linux HWCAP, uses the LSX
packed pointwise kernel when available (including on LASX CPUs), and otherwise
falls back to scalar. See the [platform matrix](docs/platform-matrix.md) and
[development package guide](docs/package.md) before making compatibility or
performance claims; LoongArch performance still requires physical hardware.
The manual [ARM64 OCR performance workflow](docs/arm64-performance.md)
collects native operator and RSS profiles before new NEON kernels are added.
The manual [Native x64 OCR Performance workflow](docs/engine-comparison.md)
collects Windows AVX2 stage, operator, REC-width, and RSS profiles before
targeted x64 optimization.

### Performance snapshot

On the bundled 500×500 sample image, a Windows x64 release build measured the
following native baseline with `REC target_width = 320`:

| Full OCR mode | Mean latency |
|---|---:|
| 1 worker | 209.27 ms |
| 4 workers | 98.47 ms |

Four workers provide about **2.13×** throughput acceleration for this sample.
Results vary with CPU, compiler, image content, and system load. DET now uses a
separate CPU-topology-aware intra-op budget, capped at eight physical cores,
before the independently configured CLS/REC line workers begin. This keeps
`worker_count` focused on line-level memory and latency trade-offs.

Long-text clients such as the C full-OCR demo, offline HTML, Java/JNI, and C#
Demo use `REC target_width = 960` as a maximum to preserve wide-line detail.
Full OCR now selects 192/320/480/640/960 per detected line, sorts work by
width, and keeps at most two concrete REC sessions per worker. Standalone REC
and the public C ABI now default to 960; callers that need the historical faster
profile can explicitly request 320. In local fixed-960 versus adaptive-960 profiles,
the 16-line sample improved by 31.09% with one worker and 17.39% with four;
the long-line-heavy article sample improved by 13.38% and 5.61% respectively.
Both comparisons retained identical OCR text checksums.

## Build, convert, and test

Requirements:

- Python 3.9 or newer;
- a C11 compiler and, for the default HTTP Demo, a C++11 compiler;
- packages pinned in `requirements-converter.txt`.

```powershell
python -m pip install -r requirements-converter.txt
python converter/analyze_onnx.py `
  --json-output docs/ppocrv6-tiny-analysis.json `
  --markdown-output docs/SUPPORTED_OPS_V0.md
python -m unittest -v tests.test_analyze_onnx
```

Or through CMake:

```powershell
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The normal build creates `build/models/rec.lwm`, `build/models/cls.lwm`,
`build/models/det.lwm`, static and shared pure-C
libraries, the `lw-recognize-ppm`, `lw-detect-ppm`, and `lw-ocr-ppm`
public-API Demos, the machine-readable
`lw-rec-benchmark` and `lw-ocr-benchmark`, the native
`lw.PPOCR.C.HttpServer` plus browser page, and
`lwm-inspect`. Inspect the converted model with:

```powershell
.\build\Release\lwm-inspect.exe .\build\models\rec.lwm
```

With a single-configuration generator such as Ninja, omit the `Release`
subdirectory.

The checked PP-OCRv6 model inputs are catalogued in
[`models/ppocrv6-models.json`](models/ppocrv6-models.json). It includes Tiny,
Small, and Medium DET/REC assets, with one shared Tiny CLS and one shared
Small/Medium REC dictionary. Tiny remains the default in existing integrations;
Small and Medium are opt-in preview variants distributed as separate runtime
model packs and browser SDK/HTML artifacts. Their native and browser full-OCR
goldens are release gates, but the C ABI and LWM format are not yet frozen.
See the [model support and selection matrix](docs/supported-models.md) and the
project-owned [full-OCR corpus](docs/full-ocr-golden-corpus.md).

## Standalone browser OCR

Full PP-OCR can run entirely in the browser without a server. After activating
the Emscripten SDK, build one self-contained HTML file with:

```bash
emcmake cmake -S . -B build-wasm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLW_BUILD_HTTP_DEMO=OFF \
  -DLW_BUILD_CSHARP_DEMOS=OFF \
  -DBUILD_TESTING=OFF
cmake --build build-wasm --target lw-ocr-js lw-ocr-html
```

The build produces two self-contained browser artifacts:

- `build-wasm/lw-ppocr.js` is the reusable JavaScript SDK. It exposes
  `LwPpocr.create()`, accepts File, Blob, ImageData, or Canvas input, runs in a
  Worker when available, and returns versioned structured results.
- `build-wasm/ocr-demo.html` embeds that exact SDK plus the responsive example
  UI and a lazy, offline PDF.js frontend. It can be opened directly without the
  native HTTP Demo. The HTML also embeds PDF.js 6.3.289's `jbig2.wasm`,
  `openjpeg.wasm`, and `qcms_bg.wasm` helpers for supported scanned-PDF
  formats; PDF.js is not included in `lw-ppocr.js` or the C runtime.

Tiny is the default browser profile. Tagged releases also provide explicitly
named Small and Medium SDK/HTML files after variant-specific real-OCR and
lifecycle gates. Tiny is recommended for phones; Small is opt-in after testing
the target corpus/device, and Medium is a desktop-first preview because of its
substantially higher latency and memory cost. All three expose the same SDK API.

The HTML Demo accepts screenshots through the normal file picker, drag-and-drop,
or `Ctrl+V`/`⌘V` image paste. Pasting only prepares a local image preview; it
does not start OCR automatically, and clipboard images are never uploaded.

The standalone page supports selectable reading order: horizontal
left-to-right, vertical right-to-left for traditional books, and vertical
left-to-right. This changes result ordering only; coordinates, scores, and
model inference remain unchanged.

For restrictive mobile WebViews, PDF.js automatically falls back from its Blob
module Worker to main-thread parsing and from `Blob.arrayBuffer()` to
`FileReader`. A visible diagnostic report distinguishes initialization, file
read, parse, and render failures. Phone support claims still require testing
the exact artifact on the target device; desktop viewport emulation only gates
responsive layout.

The SDK queries the Web ABI for actual output capacities and reuses buffers
across repeated OCR runs. See the [Browser JavaScript SDK](docs/web-sdk.md),
[standalone HTML usage](docs/standalone-html.md), and
[OCR result export schema](docs/ocr-export-schema.md).

## Node.js WASM distribution

The Emscripten build also provides an official raw Node.js package. It contains
`runtime.cjs`, the PP-OCRv6 tiny LWM models, the dictionary, a manifest, and
checksums; it does not depend on the standalone HTML demo or any npm runtime
package:

```bash
cmake --build build-wasm --target lw-node-wasm-package
node tests/node/smoke.cjs \
  --package build-wasm/node-wasm \
  --sample build-wasm/models/sample.ppm
```

The generated `lw.PPOCR.C-<version>-node-wasm.zip` supports Node.js 18 or
newer. The runtime accepts BGR8 pixels through the existing WASM Host ABI;
applications provide their own JPEG/PNG decoder and create separate Worker
instances for concurrency. The manifest records whether the build used
`wasm128` or the scalar fallback. See [Node/WASM distribution](docs/NODE_WASM_DISTRIBUTION.md).

## Android ARM64 Native Preview

The repository includes an experimental Android Native SDK under `android/`.
The preview targets `arm64-v8a` with `minSdk 21`, bundles the offline PP-OCRv6
tiny LWM models, and exposes Kotlin and Java APIs over the existing pure-C runtime.
The demo is an offline image OCR workbench: choose an image, preview it, select
CLS and reading order, run OCR, inspect/highlight detection boxes, and copy,
share, or save TXT/JSON results. It uses the system media picker and requests
no camera, network, or storage permission. Camera/CameraX, PDF, video, and
batch processing are intentionally outside this preview.

The local machine does not need Android Studio, the Android SDK, the NDK, or an
emulator. GitHub Actions builds the Release AAR and a signed `demo-preview.apk`
and inspects the AArch64 ELF, native dependency closure, model assets, APK
permissions, and model manifest. CI also compiles the pure-Java `demo-java`
integration sample in Debug and Release; those Java APKs are not published. A
green CI build is not a physical-device OCR,
memory, thermal, or vendor-ROM validation; this preview has been separately
validated on an ARM64 device. See `android/README.md` for the SDK API, demo
workflow, artifact names, and upgrade limitations.

## Desktop Java/JVM JNI example

The development package also includes a deliberately small Java 8+ console
consumer under `examples/java-jni/`. It supports Windows x64, Linux x64, and
macOS ARM64, uses standard `ImageIO` for JPEG/PNG/BMP input, and calls the
existing C ABI through `lw_ppocr_java` without adding Java dependencies to the
core build.
It keeps a compatibility text-only API and also exposes immutable detailed
`OcrResult`/`OcrLine` values with source-image quadrilaterals and detection /
recognition scores. There is no UI, Android support, Maven artifact, automatic
native loader, or model download. See the bilingual
[`examples/java-jni/README.md`](examples/java-jni/README.md) and
[`examples/java-jni/README.zh-CN.md`](examples/java-jni/README.zh-CN.md).
The same workflow publishes SHA-256-checked Windows/Linux/macOS Java JNI bundles for
CI-verified integration testing. Tagged releases repackage those exact tested
bundles as versioned ZIP/TAR.GZ assets when the release workflow runs.

## Documentation

- [Supported operators and model analysis](docs/SUPPORTED_OPS_V0.md)
- [Supported model status](docs/supported-models.md) and
  [PP-OCRv6 Small analysis](docs/ppocrv6-small-analysis.md)
- [C API and ownership rules](docs/c-api.md)
- [Kernel scope and reference tests](docs/scalar-kernels.md)
- [REC, CLS, DET, and full-OCR pipelines](docs/rec-pipeline.md),
  [CLS](docs/cls-pipeline.md), [DET](docs/det-pipeline.md), and
  [full OCR](docs/full-ocr.md)
- [REC and full-OCR Golden corpora](docs/rec-golden-corpus.md),
  [full-OCR corpus](docs/full-ocr-golden-corpus.md), and
  [graph-executor gates](docs/graph-executor.md)
- [Performance baseline, profile, and optimization notes](docs/performance-baseline.md),
  [full-OCR profile](docs/full-ocr-profile.md), and
  [kernel optimization](docs/kernel-optimization.md)
- [Correctness-gated full-OCR comparison with OpenCV DNN](docs/opencv-dnn-comparison.md)
- [Paired C vs C# OCR engine comparison methodology](docs/engine-comparison.md)
- [Native x64 OCR profile workflow](docs/engine-comparison.md)
- [Browser JavaScript SDK](docs/web-sdk.md) and
  [standalone HTML usage](docs/standalone-html.md)
- [Node.js/WASM distribution](docs/NODE_WASM_DISTRIBUTION.md)
- [Development package and managed demos](docs/package.md) and
  [C#/HTTP/web integration](docs/managed-demos.md)
- [Desktop Java/JVM JNI example](examples/java-jni/README.md)

## Runtime dependency boundary

The `converter/` tool is allowed to use Python, ONNX, NumPy, and protobuf in a
development environment. The deployment loader under `src/` is C11 only and
does not link or import any of them.

`opencv-python-headless` is used only as an independent perspective-crop oracle
by the full-OCR reference test. It is not linked, imported, or packaged by the
runtime.

## Project direction

```text
PP-OCR ONNX
    -> development-time ModelC converter
    -> platform-independent .lwm
    -> pure-C runtime
    -> REC / CLS / DET / full OCR
```

Direct converter/model dependencies are recorded in `dependencies.lock.json`.
The pure-C libraries have no third-party deployment runtime dependency; the
optional native HTTP executable embeds the vendored header-only cpp-httplib.

## License

MIT
