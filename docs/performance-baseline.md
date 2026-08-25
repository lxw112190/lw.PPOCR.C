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
