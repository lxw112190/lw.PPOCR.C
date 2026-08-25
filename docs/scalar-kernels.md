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
| ReduceMean | 3 | Multi-axis reduction with keep-dim and empty-axis behavior |
| AveragePool | 1 | NCHW 2D pooling with padding and include-pad behavior |
| Squeeze | 3 | Validated shape-only layout copy |
| Transpose | 3 | Contiguous rank-aware permutation |
| Unsqueeze | 2 | Validated shape-only layout copy |
| MatMul | 2 | Batched input matrices with one shared 2D weight matrix |
| **Total** | **120 / 161** | Kernel available and reference-tested |

The binary kernels validate the expected output shape against NumPy/ONNX-style
broadcast rules and reject an output buffer that aliases either input. The
activation kernels and Softmax support in-place operation. Tensor rank is
limited to `LW_MAX_DIMS` (8), and all element-count and FP32 byte-size products
are checked before pointer indexing, including 32-bit builds.

## Correctness tests

`kernel-reference-driver` emits deterministic results for representative
three-dimensional broadcasts, activations, and a Softmax input containing
values near `+1000` and `-1000`. `tensor-reference-driver` covers layout
changes, multi-axis reduction, padded AveragePool, and batched MatMul. Their
Python tests reconstruct the same values with NumPy and check every output
using a tight FP32 tolerance. The native drivers also check invalid shapes,
duplicate or invalid axes, forbidden aliases, null inputs, and non-finite
parameters.

The same test is built and passes locally for Windows x64 and Windows x86.
Linux remains covered by the repository CI definition but is not claimed as
verified until that workflow runs remotely.

## Deliberate boundary

There is still no public inference call. The runtime does not execute the REC
graph until the remaining 37 `Conv` and 4 `BatchNormalization` nodes are
implemented and the executor can reject unsupported nodes atomically. The
MatMul contract deliberately matches the supported REC graph: one or more
input matrices multiplied by one shared two-dimensional weight matrix.
