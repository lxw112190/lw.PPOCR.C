# lw.PPOCR.C

Tiny pure-C inference runtime for PP-OCR.

`lw.PPOCR.C` 是一个面向 PP-OCR 的轻量纯 C 推理 Runtime。部署端目标是不依赖
Python、OpenCV、ONNX Runtime、OpenVINO、TensorRT 或 protobuf。

> This is not a general-purpose ONNX Runtime.

## Current milestone

The project is in its first development gate: exact PP-OCRv6 tiny model
analysis. Runtime kernels are intentionally not implemented before the real
DET/CLS/REC operator and dynamic-shape surface is documented.

Current scope:

- PP-OCRv6 tiny;
- REC first;
- FP32, CPU, scalar, single-threaded;
- custom, non-frozen LWM v0 format next;
- Windows x64 and Linux x64 first;
- Windows 7 x86 compatibility preserved by design.

## Reproduce model analysis

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
cmake --build build --target analyze_models
ctest --test-dir build --output-on-failure
```

The human-readable result is in
[`docs/SUPPORTED_OPS_V0.md`](docs/SUPPORTED_OPS_V0.md). The JSON report is the
machine-readable source for future converter tests.

## Runtime dependency boundary

The `converter/` tool is allowed to use Python, ONNX, NumPy, and protobuf in a
development environment. The future deployment runtime under `src/` will be C
only and must not link or import any of them.

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
