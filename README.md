# lw.PPOCR.C

[简体中文](README.zh-CN.md) | English

Tiny pure-C inference runtime for PP-OCR.

`lw.PPOCR.C` runs PP-OCR without Python, OpenCV, ONNX Runtime, OpenVINO,
TensorRT, protobuf, or any other deployment-time runtime dependency.

> This is not a general-purpose ONNX Runtime.

## Current milestone

The project currently provides:

- Deterministic PP-OCRv6 tiny REC/CLS/DET-to-LWM v0.1 conversion.
- Safe Conv + BatchNormalization folding during REC/CLS conversion.
- A bounds-checked C11 model loader, dynamic shape propagation, and reusable
  workspace planning.
- Public C APIs for recognition, fixed-batch direction classification,
  detection, and composed full OCR.
- Pure-C BGR preprocessing, perspective crop, UTF-8 CTC decoding, DB-style
  detection postprocessing, and reading-order quadrilateral output.
- FP32 CPU inference with scalar fallback and runtime-dispatched SSE2/AVX2
  kernels. Full OCR can use a fixed DET operator pool and then recognize
  independent detected lines in parallel.
- Optional .NET Framework 3.5 WinForms, native `cpp-httplib` HTTP/web, and
  offline single-file WebAssembly demos.

The WinForms and standalone browser Demos can copy recognized text and export
UTF-8 TXT or versioned JSON using the shared
[OCR result export schema](docs/ocr-export-schema.md).

The deployment core accepts decoded BGR8 pixels. Image decoding remains an
application concern, so the core itself stays dependency-free.

### Platform focus

- Primary targets: Windows x64 and Linux x64.
- Windows 7 x86 compatibility is preserved by design.
- The LWM v0.1 format is custom and not yet frozen.

### Performance snapshot

On the bundled 500×500 sample image, a Windows x64 release build measured the
following native baseline with `REC target_width = 320`:

| Full OCR mode | Median latency |
|---|---:|
| 1 worker | 264.12 ms |
| 4 workers | 105.77 ms |

Four workers provide about **2.50×** throughput acceleration for this sample.
Results vary with CPU, compiler, image content, and system load. The four-worker
mode applies DET output-channel parallelism before independent CLS/REC line
work.

Long-text clients such as the offline HTML and C# Demo use
`REC target_width = 960` as a maximum to preserve wide-line detail. Full OCR
now selects 192/320/480/640/960 per detected line, sorts work by width, and
keeps at most two concrete REC sessions per worker. Standalone REC and the
public C ABI remain unchanged. In local fixed-960 versus adaptive-960 profiles,
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
reuses its input/output buffers across repeated OCR runs. Completed results can
be copied as plain text or downloaded as UTF-8 TXT and versioned JSON; see the
[OCR result export schema](docs/ocr-export-schema.md).

## Documentation

- [Supported operators and model analysis](docs/SUPPORTED_OPS_V0.md)
- [C API and ownership rules](docs/c-api.md)
- [Kernel scope and reference tests](docs/scalar-kernels.md)
- [REC, CLS, DET, and full-OCR pipelines](docs/rec-pipeline.md),
  [CLS](docs/cls-pipeline.md), [DET](docs/det-pipeline.md), and
  [full OCR](docs/full-ocr.md)
- [Golden corpus and graph-executor gates](docs/rec-golden-corpus.md) and
  [graph executor](docs/graph-executor.md)
- [Performance baseline, profile, and optimization notes](docs/performance-baseline.md),
  [full-OCR profile](docs/full-ocr-profile.md), and
  [kernel optimization](docs/kernel-optimization.md)
- [Development package and managed demos](docs/package.md) and
  [C#/HTTP/web integration](docs/managed-demos.md)

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
