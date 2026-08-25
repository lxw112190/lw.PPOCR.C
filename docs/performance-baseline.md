# Scalar REC performance baseline

`lw-rec-benchmark` establishes the measurement contract used before changing
the scalar kernels. It loads the public recognizer once, performs warm-up calls,
checks that every result is bit-deterministic, and emits one JSON object.

The report contains recognizer creation time, model size, planned workspace,
preallocated input/output bytes, mean/median/P95/min/max latency, throughput,
RSS after warm-up, final RSS, RSS growth, and peak RSS. RSS is an operational
signal only; it is not by itself proof that a process is leak-free.

## Reproduce

From a Release build tree:

```powershell
.\build-ninja-c\lw-rec-benchmark.exe `
  .\build-ninja-c\models\rec.lwm `
  .\models\ppocrv6-tiny\ppocr_keys.txt `
  .\build-ninja-c\models\sample-crop.ppm `
  3 20
```

The final two arguments are warm-up count and measured iteration count. Both
must be in `1..10000`. The output is UTF-8 JSON with `schema_version: 1`.
The `backend` field reports the selected `scalar`, `sse2`, or `avx2` kernel
level.

## Local baseline

The following is one local measurement, not a general performance promise:

- host: AMD Ryzen 7 7735H, 16 logical processors;
- OS: Windows 10.0.19045 x64;
- compiler: MSVC 19.40, Release `/O2`, C11;
- backend: scalar, FP32, single-threaded;
- model/image: bundled PP-OCRv6 tiny REC and 295x46 PPM crop;
- protocol: 3 warm-up calls followed by 20 measured calls.

| Process | Mean | Median | P95 | Min | Max | Throughput | RSS growth | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Windows x64 | 570.752 ms | 571.315 ms | 573.565 ms | 565.017 ms | 582.988 ms | 1.752/s | 0 B | 13.45 MiB |
| Windows x86 | 1416.520 ms | 1411.053 ms | 1438.057 ms | 1393.362 ms | 1457.315 ms | 0.706/s | -84 KiB | 14.75 MiB |

Both runs produced exactly `纯臻营养护发素` on every warm-up and measured call.
The runtime planned 2,948,160 workspace bytes and 1,289,280 preallocated
input/output bytes. Future optimization reports must use the same fixture and
protocol, retain the Golden Tests, and compare both latency and memory.

## First optimized result

The first profile-directed Scalar Conv change used the same machine, build,
fixture, and 3+20 protocol:

| Process | Baseline mean | Optimized mean | Reduction | Speedup | Optimized throughput | RSS growth |
|---|---:|---:|---:|---:|---:|---:|
| Windows x64 | 570.752 ms | 381.206 ms | 33.21% | 1.497x | 2.623/s | 0 B |
| Windows x86 | 1416.520 ms | 669.329 ms | 52.75% | 2.116x | 1.494/s | 0 B |

The profile, implementation boundary, correctness checks, and complete A/B
report are recorded in [`kernel-optimization.md`](kernel-optimization.md).

## Pointwise optimized result

Node-level profiling identified the 25 pointwise Conv nodes as 96.24% of Conv
time after the first change. A cache-contiguous 1x1 path produced this second
result under the same 3+20 protocol:

| Process | Original baseline | First optimized | Pointwise optimized | Overall speedup | Throughput | RSS growth |
|---|---:|---:|---:|---:|---:|---:|
| Windows x64 | 570.752 ms | 381.206 ms | 51.495 ms | 11.084x | 19.419/s | 0 B |
| Windows x86 | 1416.520 ms | 669.329 ms | 139.297 ms | 10.169x | 7.179/s | 0 B |

The full node profile and implementation constraints are recorded in
[`kernel-optimization.md`](kernel-optimization.md).

## 3x3 downsampling optimized result

The two remaining ordinary 3x3 downsampling nodes received a cache-local
portable-C path. Under the same 3+20 protocol:

| Process | Original baseline | Pointwise optimized | 3x3 optimized | Overall speedup | Throughput | RSS growth |
|---|---:|---:|---:|---:|---:|---:|
| Windows x64 | 570.752 ms | 51.495 ms | 44.444 ms | 12.842x | 22.500/s | 0 B |
| Windows x86 | 1416.520 ms | 139.297 ms | 126.773 ms | 11.174x | 7.888/s | 0 B |

The 3x3 result uses the median reported mean and throughput from five repeated
3+20 runs; all five runs had zero measured RSS growth.

## MatMul optimized result

The two REC MatMul nodes now scan each shared weight row contiguously and reuse
it across a four-row output block. Under the same five repeated 3+20 protocol:

| Process | 3x3 optimized | MatMul optimized | Further reduction | Further speedup | Original baseline speedup | Throughput | RSS growth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Windows x64 | 44.444 ms | 37.405 ms | 15.84% | 1.188x | 15.259x | 26.734/s | 0 B |
| Windows x86 | 126.773 ms | 115.728 ms | 8.71% | 1.095x | 12.240x | 8.641/s | 0 B |

The x64 node profile measured median MatMul time at 5.970 ms, down 57.23%
from 13.957 ms. Every measured call retained the exact text and score, and all
ten runs had zero measured RSS growth.

The latest operator profile is distributed across Conv, MatMul, and
elementwise work. Further specialization should therefore be selected using
CPU-dispatched SIMD measurements rather than adding another narrow scalar
shape path.

## SSE2 MatMul result

The first explicit SIMD milestone adds runtime CPU detection and a four-column
SSE2 MatMul loop while retaining the scalar implementation as the fallback.
Under the same five repeated 3+20 protocol:

| Process | Scalar MatMul stage | SSE2 stage | Further reduction | Further speedup | Original baseline speedup | Throughput | RSS growth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Windows x64 | 37.405 ms | 35.344 ms | 5.51% | 1.058x | 16.148x | 28.293/s | 0 B |
| Windows x86 | 115.728 ms | 114.763 ms | 0.83% | 1.008x | 12.343x | 8.714/s | 0 B |

The x64 MatMul operator median fell from 5.970 ms to 2.845 ms, a 52.34%
reduction. The x86 MatMul median fell from 5.860 ms to 2.869 ms, a 51.04%
reduction; its smaller end-to-end change reflects the remaining Conv and
elementwise cost and normal run-to-run variance. Every run reported `sse2`,
retained the exact text and score, and had zero measured RSS growth.

## AVX2 MatMul result

The next SIMD milestone adds OS-safe AVX2 detection and an eight-column MatMul
loop, while retaining SSE2 and scalar fallbacks. Under the same five repeated
3+20 protocol:

| Process | SSE2 stage | AVX2 stage | Further reduction | Further speedup | Original baseline speedup | Throughput | RSS growth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Windows x64 | 35.344 ms | 33.324 ms | 5.72% | 1.061x | 17.128x | 30.009/s | 0 B |
| Windows x86 | 114.763 ms | 112.594 ms | 1.89% | 1.019x | 12.581x | 8.881/s | 0 B |

The x64 MatMul operator median fell from 2.845 ms with SSE2 to 1.906 ms with
AVX2, a 33.01% reduction. The x86 MatMul median fell from 2.869 ms to 1.898 ms,
a 33.84% reduction. Every run reported `avx2`, retained the exact text and
score, and had zero measured RSS growth.

## SIMD pointwise Conv result

The same OS-safe runtime dispatch now selects eight-lane AVX2, four-lane SSE2,
or scalar pointwise Conv. Under the same five repeated 3+20 protocol:

| Process | AVX2 MatMul stage | SIMD pointwise stage | Further reduction | Further speedup | Original baseline speedup | Throughput | RSS growth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Windows x64 | 34.164 ms | 29.504 ms | 13.64% | 1.158x | 19.345x | 33.893/s | 0 B |
| Windows x86 | 114.499 ms | 59.540 ms | 48.00% | 1.923x | 23.791x | 16.795/s | 0 B |

The x64 pointwise Conv node total fell from 10.000 ms to 5.502 ms, a 44.98%
reduction. The x86 total fell from 69.400 ms to 14.350 ms, a 79.32% reduction.
The larger x86 result reflects that its prior MSVC build kept this loop scalar,
while the x64 compiler had already generated some SSE2 instructions. Every run
reported `avx2`, retained the exact text and score, and had zero measured RSS
growth.
