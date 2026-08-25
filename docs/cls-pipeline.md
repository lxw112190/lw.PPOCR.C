# CLS direction-classification pipeline

The public CLS path supports the exact bundled PP-OCRv6 tiny classifier. It is
an additive capability beside REC; DET, DB postprocessing, crop extraction, and
the combined full-OCR API are not part of this milestone.

## Contract

- Development input: bundled `models/ppocrv6-tiny/cls.onnx`, exact SHA-256
  `dd8b2b61983d76ab230a58da9e0e0e84956b71c3877f2ce6e438fe22d74d2cf2`.
- Deployment model: deterministic `cls.lwm`, fixed input `[1,3,80,160]`, FP32.
- Source pixels: caller-owned interleaved BGR8 with explicit byte count and row
  stride; encoded image files are outside the runtime.
- Resize: height 80, aspect-ratio-preserving width rounded upward and capped at
  160, followed by black right padding.
- Normalize/layout: `(value / 255 - 0.5) * 2`, planar BGR CHW.
- Output: label `0`/`1`, orientation `0`/`180` degrees, selected Softmax score,
  and the actual resized width.
- The API reports orientation but never mutates or rotates caller-owned pixels.

The converter specializes batch size to one, removes the fixed metadata-only
Shape/Slice/Concat chain, rewrites GlobalAveragePool to the existing ReduceMean
kernel, removes Identity aliases, and emits one checked static Reshape node.
This transformation is restricted by the exact model hash; it is not general
ONNX support.

## Verification gates

- converter determinism and exact model-hash rejection;
- static Reshape scalar-kernel reference test;
- complete converted CLS graph versus ONNX Runtime;
- BGR preprocessing versus an independent NumPy bilinear reference;
- public classifier result versus original ONNX output;
- UTF-8 model paths on Windows;
- C structure sizes, initialized-structure rules, resource limits, and shared
  library export allowlist;
- Windows x64 and x86 builds and full regression suites.
