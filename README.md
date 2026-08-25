# lw.PPOCR.C

Tiny pure-C inference runtime for PP-OCR.

`lw.PPOCR.C` 是一个面向 PP-OCR 的轻量纯 C 推理 Runtime。部署端目标是不依赖
Python、OpenCV、ONNX Runtime、OpenVINO、TensorRT 或 protobuf。

> This is not a general-purpose ONNX Runtime.

## Current milestone

Exact PP-OCRv6 tiny model analysis, the deterministic REC-to-LWM v0.1
converter, the bounds-checked pure-C model loader, and runtime REC shape/workspace
planning are implemented. Reference-tested scalar kernels now cover Add, Mul,
Div, Relu, Erf, HardSigmoid, and Softmax (106 of the converted REC graph's 161
nodes). Full-graph inference is not implemented yet.

Current scope:

- PP-OCRv6 tiny;
- REC first;
- FP32, CPU, scalar, single-threaded;
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

The normal build creates `build/models/rec.lwm`, the pure-C static loader
library, and `lwm-inspect`. Inspect the converted model with:

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
