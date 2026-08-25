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

All results in this document describe one machine, compiler, model, and
fixture. They should be reproduced on target machines before being used for
capacity planning.
