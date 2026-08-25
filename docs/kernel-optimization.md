# First profiled scalar-kernel optimization

This milestone profiles the exact bundled PP-OCRv6 tiny REC graph before
changing a kernel, optimizes the measured hotspot, and repeats the public
end-to-end benchmark under the same conditions. The figures below are local
engineering measurements, not a performance guarantee.

## Profiling method

An internal, test-only executor entry point accepts a monotonic-clock callback
and accumulates elapsed nanoseconds and invocation counts by LWM operator ID.
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

## Conv change

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

These results describe one machine, compiler, model, and fixture. They should
be reproduced on target machines before being used for capacity planning.
