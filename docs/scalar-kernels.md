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
| Conv | 37 | NCHW normal, grouped, and Depthwise convolution |
| BatchNormalization | 4 | Inference-mode channel normalization |
| **Total** | **161 / 161** | Kernel available and reference-tested |

The binary kernels validate the expected output shape against NumPy/ONNX-style
broadcast rules and reject an output buffer that aliases either input. The
activation kernels and Softmax support in-place operation. Tensor rank is
limited to `LW_MAX_DIMS` (8), and all element-count and FP32 byte-size products
are checked before pointer indexing, including 32-bit builds.

## Correctness tests

`kernel-reference-driver` emits deterministic results for representative
three-dimensional broadcasts, activations, and a Softmax input containing
values near `+1000` and `-1000`. `tensor-reference-driver` covers layout
changes, multi-axis reduction, padded AveragePool, and batched MatMul.
`conv-reference-driver` covers normal, grouped, Depthwise, and asymmetric
dilated Conv, including padding-only output regions, plus in-place
BatchNormalization. The Conv/BN expectations come from ONNX's
opset-11 ReferenceEvaluator; the other Python tests use NumPy. Every output is
checked with an FP32 tolerance. The native drivers also check invalid shapes,
duplicate or invalid axes, forbidden aliases, null inputs, non-finite
parameters, and 32-bit buffer-size overflow.

The same test suite is built and passes locally for Windows x64 and Windows x86.
Linux remains covered by the repository CI definition but is not claimed as
verified until that workflow runs remotely.

## Deliberate boundary

The public recognizer API now wraps the private executor, preprocessing, and
CTC decoder without exposing kernel or tensor internals. The executor dispatches
all 161 converted REC nodes, binds constants/workspace, and passes
complete-output comparison. The MatMul contract deliberately
matches the supported REC graph: one
or more input matrices multiplied by one shared two-dimensional weight matrix.
Its cache-contiguous implementation initializes four output rows at a time,
then scans each shared weight row contiguously across columns. This reuses the
weight row across the block while preserving the inner-dimension accumulation
order for every output element. It adds no allocation or threads.

The executor now dispatches this MatMul contract through isolated AVX2 and SSE2
kernels on x86/x64 CPUs. The AVX2 and SSE2 loops process eight and four
independent columns per instruction respectively and use the same scalar tail.
AVX2 selection requires the CPUID AVX/OSXSAVE bits, XGETBV confirmation that the
operating system saves XMM/YMM state, and the CPUID AVX2 bit. x64 selects SSE2
as its minimum SIMD level; x86 validates SSE2 with CPUID; non-x86 and
unsupported x86 CPUs retain the scalar implementation. The AVX2 kernel uses
separate multiply and add instructions rather than FMA, preserving the inner
dimension accumulation order. All three paths are allocation-free and
single-threaded. On supported test hardware, the reference driver executes the
scalar, SSE2, AVX2, and automatic-dispatch paths and requires byte-identical
output before comparing it with NumPy.

The Conv implementation is a direct scalar loop with no im2col allocation. A
single grouped implementation covers the model's normal, grouped, and
Depthwise configurations, including its 1x5 Depthwise layer. Its first
profile-directed optimization hoists valid kernel bounds and reuses
row/channel pointers while preserving FP32 accumulation order; it still uses
no explicit SIMD or threads. A second portable-C fast path traverses the
spatial plane contiguously for 1x1, unit-stride, unit-dilation, zero-padding
Conv, including batched and grouped inputs, while retaining the input-channel
accumulation order. A third cache-local path covers ordinary 3x3, stride-2,
unit-dilation, pad-1 Conv and keeps the same addition order. These paths add no
allocation, explicit SIMD, or threads. The test-only executor profiler is
private, reports Conv and MatMul node shapes, and is not part of the installed
API or packages. See
[`kernel-optimization.md`](kernel-optimization.md) for the measured A/B result.
