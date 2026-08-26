# lw.PPOCR.C

[简体中文](README.zh-CN.md) | English

Tiny pure-C inference runtime for PP-OCR.

`lw.PPOCR.C` 是一个面向 PP-OCR 的轻量纯 C 推理 Runtime。部署端目标是不依赖
Python、OpenCV、ONNX Runtime、OpenVINO、TensorRT 或 protobuf。

> This is not a general-purpose ONNX Runtime.

## Current milestone

Exact PP-OCRv6 tiny model analysis, deterministic REC/CLS/DET-to-LWM v0.1
converters, the bounds-checked pure-C model loader, and runtime shape/workspace
planning are implemented. Reference-tested scalar kernels cover all converted
REC operators plus the static Reshape needed by CLS. A private, zero-allocation
graph executor matches the original ONNX REC output at two dynamic widths and
the fixed-batch CLS output. Pure-C BGR preprocessing, UTF-8 CTC decoding, and
direction classification now provide real cropped-text REC and CLS paths.
That path is regression-tested on ten real text-line crops against the original
ONNX model and remains a mandatory gate during runtime-only optimization. The
profile-directed scalar optimizations now cover general Conv address/bounds
simplification, a cache-contiguous pointwise path, a spatially local 3x3
downsampling path, and cache-contiguous row-blocked MatMul. Windows x64/x86 A/B
measurements retain unchanged recognition results. On x86/x64, MatMul,
pointwise Conv, ordinary stride-2 3x3 Conv, stride-1 3x3 Depthwise Conv, flat
Add/Mul/Div, and single-axis binary broadcasts now use runtime-detected AVX2 or
SSE2 with automatic scalar fallbacks.
The DET graph produces its full dynamic-shape probability map and matches the
original ONNX model at multiple input sizes. A public detector C API now owns
DET resize/normalize, graph execution, bounded DB-style postprocessing,
coordinate restoration, and reading-order quadrilateral output. Image-file
decoding remains outside the core: applications provide decoded BGR8 pixels.
A public full-OCR C API now composes detection, pure-C perspective cropping,
optional direction classification/180-degree correction, and recognition. It
returns one canonical quadrilateral per UTF-8 text line through caller-owned,
capacity-checked buffers. Native 64-bit full OCR defaults to four independent
CLS/REC workers after detection; x86 and WebAssembly default to one. The worker
count is configurable without making one OCR handle reentrant.
An optional .NET Framework 3.5 WinForms test tool demonstrates direct C# P/Invoke
with local image decoding, result overlays, selectable OCR workers, and reusable
mean/P95 performance history. A separate native C++
`cpp-httplib` Demo provides the cross-platform HTTP API and browser UI; the
browser converts selected images to P6 PPM before upload. The pure-C core
remains dependency-free and continues to accept decoded BGR8 pixels only.
An Emscripten build also packages the runtime, DET/CLS/REC models, dictionary,
and browser workbench into one offline HTML file for full local OCR without a
server.

Current scope:

- PP-OCRv6 tiny;
- public REC, fixed-batch CLS, DET, and composed full-OCR paths;
- FP32, CPU, scalar/SSE2/AVX2 runtime dispatch; DET and individual graph runs
  remain single-threaded, while full OCR can process independent lines in
  parallel;
- custom, non-frozen LWM v0.1 format;
- Windows x64 and Linux x64 first;
- Windows 7 x86 compatibility preserved by design.

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

## Standalone browser OCR

Full PP-OCR can run entirely in the browser without a server. After activating
the Emscripten SDK, build one self-contained HTML file with:

```bash
emcmake cmake -S . -B build-wasm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLW_BUILD_HTTP_DEMO=OFF \
  -DLW_BUILD_CSHARP_DEMOS=OFF \
  -DBUILD_TESTING=OFF
cmake --build build-wasm --target lw-ocr-html
```

`build-wasm/ocr-demo.html` embeds the WASM runtime, all three LWM models, the
dictionary, and the page assets. It can be opened directly without the native
HTTP Demo. The page queries the Web ABI for the actual output capacities and
reuses its input/output buffers across repeated OCR runs.

The human-readable result is in
[`docs/SUPPORTED_OPS_V0.md`](docs/SUPPORTED_OPS_V0.md). The JSON report is the
machine-readable source for future converter tests.

The experimental model/session API and ownership rules are documented in
[`docs/c-api.md`](docs/c-api.md).
The internal scalar-kernel scope and its test boundary are documented in
[`docs/scalar-kernels.md`](docs/scalar-kernels.md).
The private complete-graph execution gate is documented in
[`docs/graph-executor.md`](docs/graph-executor.md).
The private end-to-end REC preprocessing and decoding contract is documented in
[`docs/rec-pipeline.md`](docs/rec-pipeline.md).
The public CLS direction-classification contract and reference gates are documented in
[`docs/cls-pipeline.md`](docs/cls-pipeline.md).
The DET graph gate and public detection pipeline are documented in
[`docs/det-graph.md`](docs/det-graph.md) and
[`docs/det-pipeline.md`](docs/det-pipeline.md).
The composed DET/optional-CLS/REC pipeline, output ownership, and crop
correctness gates are documented in [`docs/full-ocr.md`](docs/full-ocr.md).
The ten-crop ONNX-versus-pure-C correctness gate is documented in
[`docs/rec-golden-corpus.md`](docs/rec-golden-corpus.md).
The optimization baseline and benchmark protocol are documented in
[`docs/performance-baseline.md`](docs/performance-baseline.md).
The full DET/CLS/REC stage, operator, Conv-class, and worker-dispatch profile is
documented in [`docs/full-ocr-profile.md`](docs/full-ocr-profile.md).
The profile-directed kernel optimizations and their Windows x64/x86 A/B
results are documented in
[`docs/kernel-optimization.md`](docs/kernel-optimization.md).
Development package contents and Demo commands are documented in
[`docs/package.md`](docs/package.md).
The C# WinForms and native HTTP/web integration, REST contract, build commands, and
compatibility boundary are documented in
[`docs/managed-demos.md`](docs/managed-demos.md).

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

The full, non-frozen design is recorded in
[`docs/PROJECT_DESIGN.md`](docs/PROJECT_DESIGN.md).

Direct converter/model dependencies are recorded in `dependencies.lock.json`.
The pure-C libraries have no third-party deployment runtime dependency; the
optional native HTTP executable embeds the vendored header-only cpp-httplib.

## License

MIT
