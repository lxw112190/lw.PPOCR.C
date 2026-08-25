# Scalar kernel milestone

This milestone introduces the first executable FP32 building blocks for the
exact converted PP-OCRv6 tiny REC graph. The functions are private runtime
interfaces; they are not part of the public C ABI yet.

## Implemented kernels

| LWM operator | REC nodes after Identity removal | Behavior |
|---|---:|---|
| Add | 52 | Contiguous FP32 with rank-aligned broadcasting |
| Mul | 25 | Contiguous FP32 with rank-aligned broadcasting |
| Div | 10 | Contiguous FP32 with rank-aligned broadcasting |
| Erf | 10 | C `erff` elementwise implementation |
| HardSigmoid | 5 | Configurable alpha/beta and `[0, 1]` clamp |
| Relu | 3 | Elementwise zero clamp |
| Softmax | 1 | Stable max-subtracted implementation on any valid axis |
| **Total** | **106 / 161** | Kernel available and reference-tested |

The binary kernels validate the expected output shape against NumPy/ONNX-style
broadcast rules and reject an output buffer that aliases either input. The
activation kernels and Softmax support in-place operation. Tensor rank is
limited to `LW_MAX_DIMS` (8), and all element-count and FP32 byte-size products
are checked before pointer indexing, including 32-bit builds.

## Correctness tests

`kernel-reference-driver` emits deterministic results for representative
three-dimensional broadcasts, activations, and a Softmax input containing
values near `+1000` and `-1000`. `tests/test_scalar_kernels_reference.py`
reconstructs the same values with NumPy and checks every output using a tight
FP32 tolerance. The driver also checks invalid shapes, invalid axes, forbidden
binary aliases, null inputs, and non-finite HardSigmoid parameters.

The same test is built and passes locally for Windows x64 and Windows x86.
Linux remains covered by the repository CI definition but is not claimed as
verified until that workflow runs remotely.

## Deliberate boundary

There is still no public inference call. The runtime does not execute the REC
graph until the remaining `Conv`, `BatchNormalization`, `MatMul`,
`ReduceMean`, `AveragePool`, `Squeeze`, `Unsqueeze`, and `Transpose` kernels
are implemented and the executor can reject unsupported nodes atomically.
