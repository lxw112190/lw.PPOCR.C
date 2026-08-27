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

## Session-prepared pointwise weights

The next 1x1 experiment keeps the public NCHW tensors and canonical LWM OIHW
weights unchanged. During session creation, eligible group-1 weights are
copied once into `[Cout/4][Cin][4]` blocks. The execution microkernel loads one
input vector and updates four output-channel accumulators, reducing repeated
input traffic without changing each output element's input-channel addition
order. Separate scalar, SSE2, and AVX2 implementations share the packed format;
FMA remains disabled.

The selector is deliberately conservative. It requires SSE2 or AVX2, an output
channel count divisible by four, and the long feature-map geometry used by the
CLS/REC stages. Large square DET maps measured slower with four distant output
streams and remain on the existing kernel. Non-x86 and scalar-only targets do
not allocate packed weights.

The checked analysis behind this decision can be reproduced with:

```powershell
python converter/analyze_conv_shapes.py
```

It reports every Conv/MatMul/Gemm node, its concrete shape and FLOP share, plus
deduplicated kernel families for the bundled REC, CLS, and DET models.

One paired local Windows x64 Release run used the preserved pre-change binary,
the same DLL build settings, bundled 500x500/16-line fixture, three warm-ups,
and twelve measured calls:

| Metric | Existing kernel | Prepared 4-output kernel | Change |
|---|---:|---:|---:|
| Full OCR, 1 worker | 598.517 ms | 562.491 ms | -6.02% |
| Post-DET, 1 worker | 389.860 ms | 355.069 ms | -8.92% |
| Throughput, 1 worker | 1.671/s | 1.778/s | +6.41% |
| Steady RSS, 1 worker | 67.61 MiB | 69.68 MiB | +2.07 MiB |
| Full OCR, 4 workers | 361.050 ms | 338.231 ms | -6.32% |
| Post-DET, 4 workers | 155.578 ms | 132.231 ms | -15.01% |
| Throughput, 4 workers | 2.770/s | 2.957/s | +6.75% |
| Steady RSS, 4 workers | 98.14 MiB | 106.38 MiB | +8.25 MiB |

The extra memory is per-session packed weight storage, so it scales with the
number of CLS/REC workers. This tradeoff should be remeasured on each target
CPU and worker count. Direct tail-block tests require scalar, SSE2, AVX2, and
automatic dispatch to be byte-identical before ONNX reference comparison; the
complete x64 suite and x86 ABI, graph, package, and full-OCR gates also pass.

## Integer nearest-neighbor Resize

The DET feature pyramid contains six nearest-neighbor Resize nodes per graph
execution. The original general-rank kernel decoded every output element back
to all source coordinates with repeated modulo, division, and `floor` calls.
For the model's exact NCHW integer upscales, that coordinate work dominated the
actual copies.

The specialized path requires rank 4, unchanged N/C dimensions, scale 1 on N/C,
and exact positive integer height/width factors consistent with the resolved
output shape. It expands one source row contiguously, then copies that completed
row for the remaining vertical repetitions. Fractional scales, downsampling,
and all other ranks still use the original path. No lookup table, persistent
allocation, model-format change, or public API change is introduced.

On the local three-iteration operator profile, Resize fell from 99.240 ms to
1.775 ms (-98.21%), and the instrumented DET graph fell from 495.248 ms to
389.256 ms (-21.40%). A follow-up uninstrumented 3+12 run measured:

| Metric | Prepared-pointwise stage | Integer Resize stage | Change |
|---|---:|---:|---:|
| DET, 1 worker | 207.421 ms | 135.541 ms | -34.65% |
| Full OCR, 1 worker | 562.491 ms | 436.159 ms | -22.46% |
| DET, 4 workers | 206.000 ms | 139.311 ms | -32.37% |
| Full OCR, 4 workers | 338.231 ms | 224.049 ms | -33.76% |
| Throughput, 4 workers | 2.957/s | 4.463/s | +50.94% |

These are sequential local stage measurements, not a portable capacity claim.
Steady RSS remained at the prepared-pointwise stage level. The tensor reference
suite covers different height/width integer factors across multiple batches and
channels, plus a fractional-scale case that must remain on the general path.

## Exact spatial reduction and pooling

The next exact optimization targets two remaining coordinate-heavy operators
without changing their public or model contracts:

- NCHW ReduceMean over spatial axes 2 and 3 now sums each contiguous channel
  plane directly. Its input traversal and floating-point addition order remain
  row-major and identical to the general implementation.
- The detector's exact 2x2, stride-1, bottom/right-padded SAME_UPPER MaxPool
  uses direct row pointers for the interior and separate right/bottom border
  loops. Comparison order remains row-major, including the existing NaN
  behavior.

All other ReduceMean axes, keep-dimension modes, kernels, strides, padding and
ceil modes retain the original general paths. Neither specialization allocates
memory or adds session state.

On the local Windows x64 Release operator profiler, using the bundled 500x500
fixture, the targeted work per full OCR request changed as follows:

| Operator | General path | Exact specialized path | Change |
|---|---:|---:|---:|
| MaxPool | 12.115 ms | 0.651 ms | -94.62% |
| ReduceMean | 16.736 ms | 3.819 ms | -77.18% |

An uninstrumented 3-warm-up/12-measurement observation after both changes
reported 230.426 ms full-OCR mean and 4.340 requests/s with four line workers.
Machine load produced visible run-to-run latency noise, so that observation is
recorded as a local result rather than a paired end-to-end improvement claim.
The tensor reference suite covers multi-batch/multi-channel spatial reduction,
the specialized pool interior and borders, and the pre-existing general paths.

## Contiguous-axis Softmax

The recognition output Softmax processes 6,906 classes on a contiguous final
axis. A direct-row path removes the general inner-stride multiplication and
offset reconstruction, but deliberately retains `expf`, maximum selection,
summation order and element-wise division. It supports separate and in-place
output buffers; Softmax over a genuinely strided axis remains on the general
path.

In one five-iteration Windows x64 Release operator profile, Softmax work per
full OCR request fell from 15.848 ms to 13.967 ms (-11.87%). The scalar
reference suite covers both contiguous and strided axes and verifies that the
contiguous in-place result is byte-identical to separate-output execution.

A four-output-row MatMul experiment was also measured and rejected. Although
it loaded each wide weight vector once for four output rows, the four concurrent
6,906-column output streams expanded the active output working set. MatMul rose
from 32.105 ms to 33.735 ms per request (+5.08%) on the same class of profile,
so the experiment was removed rather than retained behind a heuristic.

Two exactness-gated Erf/GELU experiments were also rejected. An Abramowitz and
Stegun single-precision Erf approximation stayed within the numerical and OCR
reference gates, but its `expf` and division work raised median Erf time from
80.94 ms to 125.40 ms (+54.9%) on MSVC/UCRT. An exact five-node
`Div -> Erf -> Add -> Mul -> Mul` fusion retained `erff` and the original
floating-point operation order, but paired 3+12 full-OCR measurements showed no
repeatable gain: 412.13 ms became 413.18 ms with one worker, and 216.08 ms
became 216.18 ms with four workers. Both implementations were removed. Future
Erf work therefore needs a genuinely vectorized approximation plus an explicit
error corpus; eliminating intermediate tensor passes alone is not sufficient.
