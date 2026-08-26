# Full OCR profiling

The private `full-ocr-profile-driver` measures the exact bundled
DET/CLS/REC composition without changing the public C ABI. It exists to select
CPU optimization work from evidence rather than from isolated REC results.

The profiler is built when `BUILD_TESTING=ON`. A Ninja Release invocation is:

```powershell
.\build-ninja-c\full-ocr-profile-driver.exe `
  .\build-ninja-c\models\det.lwm `
  .\build-ninja-c\models\cls.lwm `
  .\build-ninja-c\models\rec.lwm `
  .\models\ppocrv6-tiny\ppocr_keys.txt `
  .\build-ninja-c\models\sample.ppm `
  5 `
  4
```

The last two arguments are measured iterations and OCR line workers. Run one
ordinary warm-up before the measured iterations so dynamic DET session sizing,
crop storage, and reusable workspaces do not distort the report.

## Measurement semantics

The JSON report separates latency from accumulated worker work:

- `wall_nanoseconds.total` is observed full-request latency;
- DET preprocess, graph, and postprocess are mutually sequential wall stages;
- `crop` measures perspective pixel extraction;
- `line_workers` is the wall time spent processing CLS/REC batches;
- `line_worker_critical` is the sum of the slowest worker in each batch;
- `line_dispatch_overhead` is `line_workers - line_worker_critical`, an estimate
  of thread creation, scheduling, joining, and caller-side dispatch overhead;
- `line_work_nanoseconds` sums CLS/REC work across workers and can exceed wall
  time when workers overlap;
- operator and Conv-class time is accumulated work time, not request wall time.

Profiling calls the clock around every graph node. Use it to rank hotspots, not
as the release latency benchmark. `lw-ocr-benchmark` remains the source for
mean/P95 latency, throughput, deterministic output, and RSS measurements.

The operator table covers all current IDs 1 through 21, including the six DET
and CLS operators that the older REC-only profile did not need. Conv work is
also classified as 1x1, ordinary 3x3, Depthwise 3x3, ordinary stride-2 3x3, or
other Conv.

## First Windows x64 result

On the local AVX2 development host, five profiled runs of the bundled 500x500
image produced 16 text lines:

| Stage | 1 worker | 4 workers |
|---|---:|---:|
| Full OCR wall | 856.530 ms | 629.267 ms |
| DET graph | 499.568 ms | 504.278 ms |
| DET preprocess + postprocess | 5.448 ms | 5.422 ms |
| Crop | 4.857 ms | 4.945 ms |
| CLS/REC worker wall | 346.615 ms | 114.591 ms |
| Worker dispatch overhead | 0.011 ms | 0.914 ms |

For the single-worker run, the largest accumulated graph operators were:

| Operator | Time | Graph work |
|---|---:|---:|
| Conv | 503.758 ms | 60.56% |
| ConvTranspose | 106.270 ms | 12.78% |
| Erf | 81.781 ms | 9.83% |
| Resize | 33.603 ms | 4.04% |
| MatMul | 30.809 ms | 3.70% |

These are instrumented profiles rather than uninstrumented A/B claims. The
important decision is nevertheless clear: on this sample, persistent line
worker creation accounts for less than one millisecond, while the single DET
graph takes about 500 milliseconds. A persistent pool remains desirable for
architecture and tail stability, but it is not the next largest latency win.

The report now includes every DET Conv/ConvTranspose node with its exact input,
weight, output, kernel, stride, dilation, padding, group, calls, and time. The
first node-level run identified four missing fast paths rather than a general
threading problem:

- ordinary 3x3 stride-1/pad-1 Conv;
- 2x2 stride-1 Conv with bottom/right padding;
- 2x2 stride-2 ConvTranspose without overlap;
- Depthwise 5x5 stride-1/pad-2 Conv.

SSE2 and AVX2 kernels were added for the three Conv families. The
non-overlapping ConvTranspose path was reordered by output channel so it writes
contiguous planes and retains the original per-output accumulation order.

## Specialized-kernel A/B

The following numbers are uninstrumented Release measurements from the same
AVX2 host, model files, 500x500 image, two warm-ups, eight measured iterations,
and deterministic 16-line result. The baseline executable and DLL were kept in
a separate build directory before changing the kernels.

| Metric | Baseline | Specialized kernels | Change |
|---|---:|---:|---:|
| DET mean | 494.960 ms | 168.194 ms | -66.0% |
| Full OCR mean, 1 worker | 845.416 ms | 484.385 ms | -42.7% |
| Full OCR mean, 4 workers | 618.641 ms | 256.651 ms | -58.5% |
| Full OCR throughput, 4 workers | 1.616/s | 3.896/s | 2.41x |
| Post-DET work, 4 workers | 114.312 ms | 91.054 ms | -20.3% |

The largest individual changes in the instrumented node report were:

| DET node/path | Before | After |
|---|---:|---:|
| node 234, regular 3x3 stride-1 | 140.54 ms | 13.38 ms |
| node 236, ConvTranspose 2x2 stride-2 | 82.94 ms | 4.89 ms |
| node 2, Conv 2x2 stride-1 | 41.21 ms | 2.84 ms |
| node 4, Conv 2x2 stride-1 | 39.60 ms | 2.64 ms |
| node 221, Depthwise 5x5 | 19.80 ms | 2.50 ms |

These results favor shape-specific SIMD and cache-friendly loop order before
operator-level threading. The next experiment should examine the remaining
stride-2 3x3 nodes and then compare output-channel parallelism at 1/2/4/8
threads. Nested operator threads must stay disabled while CLS/REC line workers
are active unless an oversubscription benchmark proves a benefit.

The automated `full_ocr_operator_profile` test runs both one and four workers,
requires deterministic 16-line output, verifies all 21 operator IDs, checks the
exact 4,946 node invocations per request, and confirms that Conv-class counts
sum to the Conv operator count. It also verifies the DET node metadata and
requires every Conv/ConvTranspose invocation to have a concrete 4D shape.
