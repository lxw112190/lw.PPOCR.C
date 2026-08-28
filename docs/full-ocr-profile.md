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

## Four-output stride-2 3x3 Conv

The next AVX2 experiment targets the stride-2 3x3 Conv family used by DET, CLS
and REC. The previous kernel streamed one output channel at a time, so every
output plane repeated the same strided input loads and even-lane shuffles. The
new path keeps four independent output vectors in registers and applies four
output-channel weights to each gathered input vector. Every output still adds
input channels and the nine kernel positions in the original order, and FMA
remains disabled, so the result is byte-identical to the scalar reference.

The selector requires an output-channel count divisible by four. Other shapes
retain the previous output-plane streaming kernel; a small-input-channel path
also covers non-multiple-of-four outputs. No packed weights, session memory,
public ABI or LWM model change is introduced.

An alternating local A/B used independent Release builds from commit `7fb8881`
and the candidate source, the same bundled 500x500/16-line fixture, three
profiled iterations and one line worker. Across four operator-profile runs, the
accumulated stride-2 3x3 work per request changed as follows:

| Component | Existing AVX2 path | Four-output path | Change |
|---|---:|---:|---:|
| All DET/CLS/REC stride-2 3x3 | 57.98 ms | 35.40 ms | -38.9% |
| DET | 27.61 ms | 19.71 ms | -28.6% |
| CLS | 1.93 ms | 0.85 ms | -56.0% |
| REC | 28.44 ms | 14.85 ms | -47.8% |

The first DET layer (`Cin=3`, `Cout=16`) fell from about 6.08 ms to 1.75 ms in
the alternating measurements. The direct Conv reference driver includes odd
spatial dimensions and a four-output-channel case, and requires the AVX2 result
to be byte-identical to the scalar kernel before graph and OCR gates run.

The public reusable-handle benchmark was also run in alternating order for the
complete OCR latency. Each entry below is the mean of three independent
processes; every process used three warm-up calls followed by eight measured
calls on the same 500x500/16-line fixture.

| Line workers | Existing AVX2 path | Four-output path | Latency change | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 401.195 ms | 382.943 ms | -4.55% | 1.048x |
| 4 | 212.031 ms | 203.781 ms | -3.89% | 1.040x |

Throughput increased by 4.77% with one worker and 4.03% with four workers. The
end-to-end gain is smaller than the targeted operator reduction because image
pre/post-processing and the other graph operators are unchanged.

## Shape-aware x64 pointwise microkernel

The next pointwise experiment retains the existing four-output packed-weight
format but doubles the x64 AVX2 spatial tile from 8 to 16 values. Eight output
accumulators remain live while each packed weight broadcast is shared by two
input vectors. The wider tile is x64-only because 32-bit x86 exposes too few
vector registers and would spill; x86, SSE2, non-x86, and spatial tails retain
their existing kernels.

The session selector continues to prepare long CLS/REC feature maps. On an x64
AVX2 host it now also prepares square pointwise maps with at least 256 spatial
positions, covering the useful DET maps while excluding tiny attention
tensors. The canonical LWM weights remain unchanged. Prepared weights are
private session state, every output retains its original input-channel
addition order, FMA remains disabled, and the public ABI is unchanged.

Four alternating three-iteration operator profiles compared the independent
Release binary from commit `fcbb37e` with the final candidate:

| Profiled work per request | Existing 4x8 path | Shape-aware 4x16 path | Change |
|---|---:|---:|---:|
| All 1x1 Conv | 112.244 ms | 84.396 ms | -24.81% |
| DET 1x1 Conv | 30.370 ms | 22.156 ms | -27.05% |
| Instrumented full OCR wall | 399.979 ms | 381.103 ms | -4.72% |

The uninstrumented reusable-handle benchmark used the same 500x500/16-line
fixture. Two alternating one-worker pairs, each with five warm-ups and twenty
measured calls, changed full OCR from 389.570 ms to 359.945 ms (-7.60%, 1.082x)
and throughput by +8.22%. Five alternating four-worker pairs, each with three
warm-ups and twelve measured calls, averaged 213.683 ms versus 199.140 ms
(-6.80%); the paired reduction median was 5.92%, with a 1.67% to 15.31% range
under visible host-load variation.

Preparing the additional DET weights raised steady RSS by about 2.8 MiB per OCR
handle. This is one detector-side cost and does not multiply with the number of
line workers. The complete Windows x64 suite passed 33/33 tests, while the x86
ABI, export, Conv, DET graph, full-OCR, and staged-package gates passed 6/6 and
continued to use the previous 4x8 path.

## Piecewise AVX2 Erf

The successful follow-up to the rejected scalar Erf experiments uses three
piecewise degree-8 single-precision polynomials. They cover `|x| < 1`,
`1 <= |x| < 2`, and `2 <= |x| < 4`; larger finite magnitudes saturate to one,
then the input sign bit is restored. The implementation evaluates eight values
at a time without FMA, preserves positive and negative zero, maps infinities to
positive and negative one, and propagates NaNs. A scalar `erff` tail handles
non-multiples of eight. The offline Emscripten build evaluates the same regions
four values at a time with WASM SIMD128; other non-AVX2 targets continue to
execute the existing scalar kernel.

The direct kernel gate samples 4,099 evenly spaced values from -6 through 6,
requires maximum absolute error no greater than `5e-7`, checks monotonicity and
checks the special-value contract. REC, CLS and DET graph references, the full
OCR pipeline reference, and the ten-crop OCR Golden corpus remain mandatory.
SIMD capability is detected once per graph execution so the runtime does not
repeat CPUID/XGETBV for every Erf node. No public ABI, LWM model, or caller-owned
buffer contract changes.

Four alternating three-iteration operator profiles compared the frozen
shape-aware pointwise binary with this candidate on the bundled
500x500/16-line fixture:

| Profiled work per request | Scalar `erff` | Piecewise AVX2 | Change |
|---|---:|---:|---:|
| All Erf | 86.599 ms | 15.248 ms | -82.39% |
| DET Erf | 20.586 ms | 3.626 ms | -82.39% |
| REC Erf | 66.013 ms | 11.622 ms | -82.39% |
| Instrumented full OCR wall | 390.733 ms | 307.389 ms | -21.33% |

The uninstrumented reusable-handle benchmark then used five alternating pairs.
One worker used five warm-ups and twenty measured calls per process; four
workers used three warm-ups and twelve measured calls. Because this host showed
visible run-to-run load variation, the paired median and full range are
reported instead of selecting the fastest run:

| Line workers | Paired latency reduction median | Pair range | Throughput gain at median pair |
|---:|---:|---:|---:|
| 1 | 19.61% | 3.73% to 39.95% | 24.40% |
| 4 | 16.50% | 3.65% to 18.56% | 19.76% |

Steady RSS was effectively unchanged. The complete Windows x64 suite passed
33/33 tests and the complete x86 suite passed 32/32 tests, including numerical,
graph, full-OCR, Golden-corpus, ABI/export and staged-package coverage.

## REC resized-width distribution

The private full-OCR profile now records the actual aspect-ratio-preserving REC
width before right padding. Its JSON report includes the sample count, resized
and target width sums, mean resized width, mean padding ratio, and stable
64/96/128/160/192/256/320/overflow histogram buckets. Worker-local counters are
merged after joining, so profiling remains race-free. Ordinary OCR calls do not
collect these values, and the public C ABI is unchanged.

On the bundled 500x500/16-line fixture, three repeated requests produced 48
samples with the same distribution for one and four workers:

| Metric | Observation |
|---|---:|
| Mean resized width | 299.5625 |
| Mean right-padding ratio | 6.39% |
| Width <= 256 | 9 / 48 |
| 256 < width <= 320 | 39 / 48 |
| Width <= 192 | 0 / 48 |

The ten-crop REC Golden corpus is similarly wide: its estimated mean resized
width is 313.7 and mean padding ratio is about 1.97%. These fixtures therefore
do not support immediately adding multiple REC width sessions: their maximum
theoretical width-work reduction is small compared with the extra session and
workspace memory. Broader customer-image profiling should precede a width-bucket
implementation; the current evidence instead keeps DET parallelism as the next
high-potential latency experiment.

## Fixed-pool DET output-channel parallelism

The full-OCR worker budget now also drives a session-owned DET thread pool.
Eligible group-1 and depthwise convolutions split disjoint output-channel
ranges only when their estimated multiply-add count reaches eight million.
Packed 1x1 slices remain aligned to four output channels. Threads are created
once with the dynamic DET session and reused across graph nodes; small kernels
remain serial. DET and CLS/REC execute in separate phases, so operator workers
are never nested inside line workers. Standalone detector and session APIs keep
their previous single-thread behavior, and the public C ABI is unchanged.

On the local Windows x64 Release AVX2 build, the bundled 500x500/16-line fixture
was warmed up and then measured in seven independent processes with five OCR
requests per process. Median per-request results were:

| Metric | 1 worker | 4 workers | Change |
|---|---:|---:|---:|
| Complete OCR | 291.35 ms | 121.20 ms | -58.40% |
| DET graph | 90.66 ms | 49.15 ms | -45.78% |
| CLS/REC line phase | 190.84 ms | 61.36 ms | -67.85% |

Compared with the immediately preceding four-worker build, whose repeated
profile was about 175 ms per request, the combined fixed-pool DET change brings
the complete pipeline to about 121 ms on this machine. The profile regression
also compares a deterministic text checksum between one- and four-worker runs.

The uninstrumented reusable-handle benchmark used five independent processes,
each with three warm-ups and twenty measured calls. Its median process means
represent the user-visible full OCR latency without per-node profile clocks:

| Workers | Mean OCR | P95 OCR | Throughput | Steady RSS |
|---:|---:|---:|---:|---:|
| 1 | 295.14 ms | 298.13 ms | 3.39/s | 72.46 MiB |
| 4 | 122.46 ms | 130.17 ms | 8.17/s | 109.22 MiB |

Four workers therefore reduced complete OCR latency by 58.51%, delivered a
2.41x speedup, and increased throughput by about 141%. The additional memory is
primarily the existing independent CLS/REC worker sessions; the DET pool shares
the detector model, weights, workspace, input, and output buffers.

## Known-capacity CTC and AVX2 GELU fusion

Full OCR allocates each line's text slot from
`lw_recognizer_info.max_text_capacity`. The recognizer now uses that guarantee
to collapse greedy CTC output once, writing UTF-8 text while computing the exact
used capacity and score. Public size queries and calls with smaller buffers
retain the original exact-capacity two-pass path. On the 16-line fixture, the
accumulated CTC work fell from about 7.45 ms to 3.71 ms, but four-worker wall
latency improved by only about 0.38 ms in the initial comparison. This is a
small, low-risk cleanup rather than the main route to the 100 ms target.

A more aggressive REC terminal-head prototype tiled
`MatMul -> Softmax -> CTC` and avoided the full probability tensor. Six
alternating 20-request pairs found only about 0.6 ms median wall improvement:
the Softmax denominator still requires all 6,906 exponentials. The prototype
was removed rather than retaining a second graph executor for that result.

The retained superkernel instead targets the ten REC and three DET GELU chains:

```text
x / sqrt(2) -> Erf -> +1 -> *x -> *0.5
```

Fusion is enabled only on AVX2 when five consecutive nodes have the exact
`sqrt(2)`, `1`, and `0.5` constants, identical FP32 tensor sizes, and four
intermediate tensors whose final consumers are inside the chain. Every other
model and backend continues through the ordinary node dispatcher. The kernel
uses the existing three-region AVX2 Erf polynomial and preserves division,
addition, and multiplication order. A dense 4,099-value test requires fused,
unfused, separate-output, and in-place results to be byte-identical. Graph,
pipeline, Golden-corpus, ABI, HTTP, and staged-package tests remain mandatory.

Six alternating 20-request four-worker pairs used the same Release configuration,
model files, and 500x500/16-line fixture. The baseline included known-capacity
CTC but not GELU fusion:

| Metric | Baseline median | GELU median | Change |
|---|---:|---:|---:|
| Complete OCR | 110.29 ms | 105.77 ms | -4.10% |
| DET graph | 43.25 ms | 41.10 ms | -4.98% |
| CLS/REC line wall | 55.76 ms | 53.55 ms | -3.96% |
| Accumulated REC graph work | 155.84 ms | 150.59 ms | -3.37% |

Every pair retained output checksum `f7bf2108d8c44764`; paired complete-OCR
improvement ranged from 2.01 ms to 5.57 ms, with a 4.76 ms median. The complete
Windows x64 Release suite passed 34/34 tests. In execution profiles, fused work
is attributed to Erf because the stable public profile schema has no GELU
operator; the four skipped semantic nodes retain invocation markers.

## AVX2 regular 3x3 four-output blocking

The detector's node 234 is a group-1 `64 -> 16` convolution with a 3x3 kernel,
unit stride, pad one, and a 128x128 output. The former AVX2 implementation
completed one output plane at a time, so it loaded the same input vectors once
for every output channel. The new path keeps four independent accumulators and
reuses each input vector across four output planes. It applies to any validated
unit-stride 3x3 shape whose output-channel count is divisible by four; other
shapes retain the previous kernel. A dedicated reference case covers vector
interior pixels, scalar borders, bias, and byte-identical output.

Four alternating pairs of 50-request, four-worker profiles were run under the
same Release configuration. The longer run intentionally reports medians
because the host was under higher sustained load than the earlier GELU test:

| Metric | Previous kernel | Four-output kernel | Change |
|---|---:|---:|---:|
| Node 234 per invocation | 5.13 ms | 2.95 ms | -42.45% |
| DET graph per request | 48.07 ms | 45.64 ms | -5.06% |
| Complete OCR per request | 139.09 ms | 136.61 ms | -1.78% |

The paired complete-OCR saving had a 2.15 ms median. A separate four-pair,
12-request one-worker profile reduced node 234 from 14.00 ms to 7.48 ms and
complete OCR from 291.24 ms to 286.10 ms. All baseline and candidate runs kept
checksum `f7bf2108d8c44764`. These results justify retaining the general kernel,
while also showing that four-worker wall latency is now dominated by the
parallel REC phase rather than this DET node alone.

## AVX2 long-axis Softmax

REC Softmax uses a contiguous final axis of 6,906 classes, while CLS Softmax
uses only a short class axis. The new AVX2 path is therefore enabled only for
contiguous axes with at least 256 values. It subtracts the row maximum, uses a
range-reduced FP32 exponential with a Taylor polynomial over the reduced
interval, and keeps the original left-to-right scalar sum order before the
normalization divide. This keeps the recognizer score contract stable while
vectorizing the expensive exponential and normalization passes. Strided and
short-axis Softmax calls retain the original implementation.

Four alternating 20-request, four-worker profiles compared binaries that both
included the regular 3x3 four-output kernel; only the long-axis Softmax path
differed:

| Metric | Scalar Softmax | AVX2 Softmax | Change |
|---|---:|---:|---:|
| Complete OCR | 103.67 ms | 101.57 ms | -2.03% |
| CLS/REC line wall | 53.19 ms | 50.97 ms | -4.17% |
| Accumulated REC graph work | 148.50 ms | 141.12 ms | -4.97% |

All eight runs retained output checksum `f7bf2108d8c44764`. REC graph reference,
REC pipeline reference, full-OCR reference/profile, scalar kernel, and the
complete Windows x64 Release suite passed after enabling the path. The profile
still reports Softmax as the same public operator, so no ABI or profile schema
change is required.
## REC four-row tiled MatMul

The AVX2 MatMul backend now processes four REC rows together in eight-column tiles for wide
terminal projections. This keeps one weight vector live while applying four broadcast inputs,
reducing repeated weight loads without changing accumulation order.

On the Windows x64 Release sample (20 iterations, four OCR workers), alternating A/B runs were:

| configuration | run 1 | run 2 | run 3 | run 4 | median |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous MatMul path | 106.53 ms | 109.64 ms | 109.22 ms | 108.17 ms | 109.22 ms |
| tiled four-row path | 103.90 ms | 104.34 ms | 103.93 ms | 106.39 ms | 104.34 ms |

The median end-to-end improvement is approximately **4.5%**. The output checksum remained
`f7bf2108d8c44764`; the full Windows x64 Release suite passed 34/34 tests.
