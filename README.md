# lw.PPOCR.C

Tiny pure-C inference runtime for PP-OCR.

`lw.PPOCR.C` 是一个面向 PP-OCR 的轻量纯 C 推理 Runtime。部署端目标是不依赖
Python、OpenCV、ONNX Runtime、OpenVINO、TensorRT 或 protobuf。

> This is not a general-purpose ONNX Runtime.

## Current milestone

Exact PP-OCRv6 tiny model analysis, the deterministic REC-to-LWM v0.1
converter, the bounds-checked pure-C model loader, and runtime REC shape/workspace
planning are implemented. Reference-tested scalar kernels now cover all 15
converted REC operator types and all 161 nodes. A private, zero-allocation graph
executor now binds LWM constants and planned workspace and matches the original
ONNX REC output at two dynamic widths. Private pure-C BGR preprocessing and
UTF-8 CTC decoding now complete a real cropped-text recognition golden path.
That path is regression-tested on ten real text-line crops against the original
ONNX model and remains a mandatory gate during runtime-only optimization. The
profile-directed scalar optimizations now cover general Conv address/bounds
simplification, a cache-contiguous pointwise path, a spatially local 3x3
downsampling path, and cache-contiguous row-blocked MatMul. Windows x64/x86 A/B
measurements retain unchanged recognition results. On x86/x64, MatMul,
pointwise Conv, stride-1 3x3 Depthwise Conv, flat Add/Mul/Div, and single-axis
binary broadcasts now use runtime-detected AVX2 or SSE2 with automatic scalar
fallbacks.
The public recognizer C API exposes that path with caller-owned UTF-8 output
buffers and bounded, preallocated inference memory. Image-file decoding remains
outside the core: applications currently provide decoded BGR8 pixels.

Current scope:

- PP-OCRv6 tiny;
- REC first;
- FP32, CPU, scalar/SSE2/AVX2 runtime dispatch, single-threaded; 15/15 REC operator
  types implemented;
- custom, non-frozen LWM v0.1 format;
- Windows x64 and Linux x64 first;
- Windows 7 x86 compatibility preserved by design.

## Build, convert, and test

Requirements:

- Python 3.9 or newer;
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

The normal build creates `build/models/rec.lwm`, static and shared pure-C
libraries, the `lw-recognize-ppm` public-API Demo, the machine-readable
`lw-rec-benchmark`, and `lwm-inspect`. Inspect the converted model with:

```powershell
.\build\Release\lwm-inspect.exe .\build\models\rec.lwm
```

With a single-configuration generator such as Ninja, omit the `Release`
subdirectory.

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
The ten-crop ONNX-versus-pure-C correctness gate is documented in
[`docs/rec-golden-corpus.md`](docs/rec-golden-corpus.md).
The optimization baseline and benchmark protocol are documented in
[`docs/performance-baseline.md`](docs/performance-baseline.md).
The profile-directed kernel optimizations and their Windows x64/x86 A/B
results are documented in
[`docs/kernel-optimization.md`](docs/kernel-optimization.md).
Development package contents and Demo commands are documented in
[`docs/package.md`](docs/package.md).

## Runtime dependency boundary

The `converter/` tool is allowed to use Python, ONNX, NumPy, and protobuf in a
development environment. The deployment loader under `src/` is C11 only and
does not link or import any of them.

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

Direct converter/model dependencies are recorded in `dependencies.lock.json`;
the intended deployment runtime dependency list is currently empty.

## License

MIT
