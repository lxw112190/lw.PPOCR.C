# Profiled scalar-kernel optimizations

These milestones profile the exact bundled PP-OCRv6 tiny REC graph before
changing a kernel, optimize measured hotspots, and repeat the public end-to-end
benchmark under the same conditions. The figures below are local engineering
measurements, not a performance guarantee.

## Profiling method

An internal, test-only executor entry point accepts a monotonic-clock callback
and accumulates elapsed nanoseconds and invocation counts by LWM operator ID
and node index. The profiler reports Conv input, weight, output, group, kernel,
stride, dilation, and padding metadata, plus MatMul input, weight, output, and
derived matrix dimensions, so optimization targets are selected from evidence.
It does not change the public C ABI, exported symbols, installed headers, or
package contents. The normal executor does not read the clock.

On Windows x64, ten width-320 graph executions produced this initial profile:

| Operator | Time per graph | Share | Nodes per graph |
|---|---:|---:|---:|
| Conv | 570.928 ms | 95.44% | 37 |
| MatMul | 14.735 ms | 2.46% | 2 |
| Add | 3.829 ms | 0.64% | 52 |
| Erf | 3.200 ms | 0.53% | 10 |
| Mul | 2.710 ms | 0.45% | 25 |

The remaining ten operator types each accounted for less than one millisecond
per graph. Conv was therefore the first optimization target.

## First change: general Conv indexing

The scalar NCHW Conv implementation now:

- computes valid kernel-row and kernel-column ranges before the input-channel
  loop, instead of checking image bounds for every multiply;
- reuses group, channel, row, and weight pointers instead of rebuilding full
  64-bit tensor offsets in the innermost loop;
- preserves the original FP32 accumulation order: input channel, kernel row,
  then kernel column.

There is no im2col allocation, SIMD instruction, thread, or public-interface
change. Normal, grouped, Depthwise, dilated, asymmetric-padding, and
padding-only output regions remain covered by ONNX reference tests.

After the change, the same ten-run profile measured Conv at 350.485 ms per
graph, a 38.6% reduction. Conv remains the dominant hotspot at 92.89%, so
future optimization should continue there before broadening the runtime.

## End-to-end A/B result

Both measurements used the same AMD Ryzen 7 7735H host, Windows 10.0.19045,
MSVC 19.40 Release `/O2` build, scalar FP32 single-thread backend, bundled
model and 295x46 PPM crop, three warm-ups, and twenty measured recognitions.

| Process | Baseline mean | Optimized mean | Latency reduction | Speedup | Baseline throughput | Optimized throughput | RSS growth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Windows x64 | 570.752 ms | 381.206 ms | 33.21% | 1.497x | 1.752/s | 2.623/s | 0 B |
| Windows x86 | 1416.520 ms | 669.329 ms | 52.75% | 2.116x | 0.706/s | 1.494/s | 0 B |

Every call in both optimized runs returned exactly `纯臻营养护发素` with score
`0.998993874`. The complete REC graph remains reference-matched against ONNX,
and the ten-crop Golden Corpus remains unchanged.

## Second profile: pointwise Conv

The node-level profile after the first change showed that 25 ordinary
`1x1`, group-1, unit-stride, zero-padding Conv nodes consumed 341.150 ms per
graph, or 96.24% of all Conv time. Depthwise Conv was below 1% and was not an
appropriate next target.

The second change adds a general fast path for `1x1`, unit-stride,
unit-dilation, zero-padding Conv. It supports batch and groups. For each output
channel it initializes the contiguous output plane with bias, then visits input
channels in the original order while updating the whole spatial plane. This
changes memory traversal from channel-strided input reads to contiguous reads
and writes while preserving each output element's FP32 accumulation order.

The source uses portable C11 and adds no allocation, intrinsic, assembly,
thread, public ABI, or runtime CPU-dispatch requirement. An additional batched,
grouped 1x1 case is compared with ONNX ReferenceEvaluator.

On Windows x64, the ten-run profile after this change measured all Conv at
25.379 ms per graph. Pointwise Conv fell from 341.150 ms to 10.696 ms. The two
ordinary 3x3 nodes are now the largest Conv targets.

## Second end-to-end A/B result

The following 3+20 runs use the same protocol and fixture as the baseline and
first optimization:

| Process | First optimized mean | Pointwise optimized mean | Further reduction | Further speedup | Original baseline reduction | Original baseline speedup | Throughput | RSS growth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Windows x64 | 381.206 ms | 51.495 ms | 86.49% | 7.403x | 90.98% | 11.084x | 19.419/s | 0 B |
| Windows x86 | 669.329 ms | 139.297 ms | 79.19% | 4.805x | 90.17% | 10.169x | 7.179/s | 0 B |

Every measured call again returned exactly `纯臻营养护发素` with score
`0.998993874`. The ONNX complete-graph comparison, ten-crop Golden Corpus,
deterministic-repeat check, and x64/x86 reference tests remain green.

## Third profile: ordinary 3x3 downsampling Conv

After pointwise optimization, the remaining Conv profile identified two
ordinary `3x3`, stride-2, unit-dilation, pad-1, group-1 nodes. They consumed
about 10.76 ms per graph on Windows x64, or 43.6% of Conv time.

The third change adds a portable-C path for this shape. It initializes each
output plane once, then scans valid output positions in input-channel,
kernel-row, kernel-column order. Input and output traversal is spatially local,
while the FP32 addition order for each output element remains unchanged. The
existing ONNX reference case exercises the same path with an odd input width,
covering the right-edge boundary.

Across five repeated node profiles, the median measurement for those two nodes
was 5.601 ms after the change, a 47.94% reduction. Median total Conv time was
19.150 ms, 24.54% below the prior 25.379 ms measurement. The full operator
profile is now more balanced: median Conv share is 43.38% and MatMul is 29.33%,
so the two-node MatMul implementation becomes the next concentrated target.

## Third end-to-end A/B result

| Process | Pointwise optimized mean | 3x3 optimized mean | Further reduction | Further speedup | Original baseline speedup | Throughput | RSS growth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Windows x64 | 51.495 ms | 44.444 ms | 13.69% | 1.159x | 12.842x | 22.500/s | 0 B |
| Windows x86 | 139.297 ms | 126.773 ms | 8.99% | 1.099x | 11.174x | 7.888/s | 0 B |

The optimized mean and throughput values are medians from five repeated 3+20
runs. Every run retained the exact text and score, deterministic output, and
zero measured RSS growth.

## Fourth profile: shared-weight MatMul

Five repeated 20-iteration profiles identified the exact MatMul shapes:

| Node | Input | Shared weights | Output | Median time |
|---:|---|---|---|---:|
| 156 | `1x40x160` | `160x80` | `1x40x80` | 0.273 ms |
| 158 | `1x40x80` | `80x6906` | `1x40x6906` | 13.693 ms |

The original column-outer loop crossed the 6906-column weight stride for each
multiply. The fourth change initializes four output rows at a time, visits the
inner dimension in the original order, and scans the corresponding weight row
and output row contiguously across columns. Reusing that weight row across four
outputs reduced memory traffic without an allocation, intrinsic, assembly,
thread, public ABI, or CPU-dispatch change.

The existing batched `2x3x4` by `4x5` reference case covers multiple batches,
multiple rows, and a non-multiple-of-four column tail. Complete REC ONNX output,
ten-crop Golden Corpus, deterministic-repeat, alias validation, and x64/x86
tests remain unchanged.

Across five final x64 profiles, median MatMul time was 5.970 ms, down 57.23%
from 13.957 ms (2.338x). The large node fell to 5.844 ms. Median operator time
was 38.586 ms: Conv 19.308 ms, MatMul 5.970 ms, Add 4.263 ms, and Erf 3.129 ms.
MatMul is now 15.40% of measured operator time instead of 29.33%.

## Fourth end-to-end A/B result

| Process | 3x3 optimized mean | MatMul optimized mean | Further reduction | Further speedup | Original baseline speedup | Throughput | RSS growth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Windows x64 | 44.444 ms | 37.405 ms | 15.84% | 1.188x | 15.259x | 26.734/s | 0 B |
| Windows x86 | 126.773 ms | 115.728 ms | 8.71% | 1.095x | 12.240x | 8.641/s | 0 B |

These are medians from five repeated 3+20 runs after a clean Release build.
Every run returned exactly `纯臻营养护发素` with score `0.998993874`, was
bit-deterministic within the run, and reported zero RSS growth.

## Fifth profile: runtime-dispatched SSE2 MatMul

Release disassembly showed the portable MatMul column loop still used scalar
`mulss/addss` instructions under MSVC `/O2`. The fifth change therefore adds:

- architecture-isolated CPU detection under `src/simd`;
- x64 SSE2 selection as an architectural baseline and x86 CPUID feature-bit
  validation, with scalar fallback elsewhere;
- an explicit four-column `mulps/addps` MatMul loop plus the existing scalar
  tail;
- a dynamic `scalar` or `sse2` value in the benchmark JSON `backend` field.

There is no public C ABI, allocation, thread, model, workspace, or package
layout change. The tensor reference driver executes both scalar and dispatched
MatMul for batched rows and five columns, requires byte-identical output, and
then checks both against NumPy. Complete graph, ONNX, Golden Corpus,
deterministic-repeat, x64/x86, and installed-package gates remain mandatory.

Across five x64 profiles, median MatMul time fell from 5.970 ms to 2.845 ms,
a 52.34% reduction (2.098x). Release disassembly confirmed `mulps/addps` in
both x64 and x86 objects. Median measured operator time was 35.345 ms and
MatMul's share fell from 15.40% to about 8.05%.

## Fifth end-to-end A/B result

| Process | Scalar MatMul mean | SSE2 mean | Further reduction | Further speedup | Original baseline speedup | Throughput | RSS growth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Windows x64 | 37.405 ms | 35.344 ms | 5.51% | 1.058x | 16.148x | 28.293/s | 0 B |
| Windows x86 | 115.728 ms | 114.763 ms | 0.83% | 1.008x | 12.343x | 8.714/s | 0 B |

These values are medians from five repeated 3+20 runs. Every call retained the
exact text and score, every benchmark identified the selected backend as
`sse2`, and every run reported zero RSS growth.

## Sixth profile: runtime-dispatched AVX2 MatMul

The sixth change widens the explicit SIMD path while keeping dispatch safe on
older CPUs and operating systems:

- CPUID verifies AVX, OSXSAVE, and AVX2 support, while XGETBV verifies that the
  operating system saves both XMM and YMM state;
- an isolated eight-column AVX2 loop uses `vmulps/vaddps`, followed by the same
  scalar tail;
- SSE2 and scalar implementations remain the automatic fallbacks;
- the benchmark reports `avx2` when that implementation is selected.

There is no public C ABI, allocation, thread, model, workspace, or package
layout change. Direct scalar, SSE2, AVX2, and dispatched tensor checks require
byte-identical MatMul output on supported hardware before NumPy comparison.
Release x64 and x86 disassembly confirmed `vmulps/vaddps` and `vzeroupper` in
the isolated AVX2 object and confirmed that no fused multiply-add instruction
was emitted. The CPU-detection object contains no AVX instruction before the
runtime checks.

Across five x64 profiles, median MatMul time fell from 2.845 ms with SSE2 to
1.906 ms with AVX2, a 33.01% reduction (1.493x). Relative to the portable
row-blocked scalar implementation's 5.970 ms, the reduction is 68.07% (3.132x).
Median measured operator time was 34.036 ms and MatMul's share fell to 5.58%.
On x86, MatMul fell from 2.869 ms to 1.898 ms, a 33.84% reduction (1.511x).

## Sixth end-to-end A/B result

| Process | SSE2 mean | AVX2 mean | Further reduction | Further speedup | Original baseline speedup | Throughput | RSS growth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Windows x64 | 35.344 ms | 33.324 ms | 5.72% | 1.061x | 17.128x | 30.009/s | 0 B |
| Windows x86 | 114.763 ms | 112.594 ms | 1.89% | 1.019x | 12.581x | 8.881/s | 0 B |

These values are medians from five repeated 3+20 runs. Every call retained the
exact text and score, every benchmark identified the selected backend as
`avx2`, and every run reported zero RSS growth. With MatMul at 5.58% of the x64
operator profile, pointwise Conv SIMD is the next measured candidate.

## Seventh profile: runtime-dispatched pointwise Conv SIMD

Release disassembly of the portable pointwise loop showed partial SSE2
auto-vectorization on x64 but only scalar instructions on x86. Five pre-change
profiles measured median pointwise totals of 10.000 ms on x64 and 69.400 ms on
x86. The seventh change therefore adds:

- isolated four-spatial-element SSE2 and eight-spatial-element AVX2 loops;
- the existing CPUID/OSXSAVE/XGETBV runtime hierarchy and scalar fallback;
- support for batch and groups without changing the NCHW tensor layout;
- separate multiply and add operations that retain each spatial element's
  input-channel accumulation order.

There is no public C ABI, allocation, thread, model, workspace, or package
layout change. A grouped pointwise reference case uses a ten-element spatial
plane so both vector bodies and scalar tails execute. The test driver compares
scalar, direct SSE2, direct AVX2, and automatic dispatch byte for byte before
the ONNX reference comparison. Release disassembly confirmed `mulps/addps` in
the SSE2 objects and `vmulps/vaddps/vzeroupper` in the AVX2 objects for both x64
and x86, with no FMA instructions.

Across five final profiles, x64 pointwise time fell from 10.000 ms to 5.502 ms,
a 44.98% reduction (1.817x), while total Conv fell from 18.994 ms to 14.432 ms.
On x86, pointwise time fell from 69.400 ms to 14.350 ms, a 79.32% reduction
(4.836x), while total Conv fell from 79.867 ms to 25.914 ms.

## Seventh end-to-end A/B result

| Process | Prior mean | SIMD pointwise mean | Further reduction | Further speedup | Original baseline speedup | Throughput | RSS growth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Windows x64 | 34.164 ms | 29.504 ms | 13.64% | 1.158x | 19.345x | 33.893/s | 0 B |
| Windows x86 | 114.499 ms | 59.540 ms | 48.00% | 1.923x | 23.791x | 16.795/s | 0 B |

These values are medians from five repeated 3+20 runs. Every call retained the
exact text and score, every benchmark identified the selected backend as
`avx2`, and every run reported zero RSS growth. The remaining profile is now
spread across ordinary/depthwise Conv and elementwise operators, so another
target should be selected only after a fresh node-level comparison.

All results in this document describe one machine, compiler, model, and
fixture. They should be reproduced on target machines before being used for
capacity planning.
