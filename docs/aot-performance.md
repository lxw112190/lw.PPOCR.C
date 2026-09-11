# PP-OCR AOT performance work

This document defines the staged, evidence-driven path for using additional
memory and model-side prepared data to reduce OCR latency. It does not change
the public C ABI or the LWM v0.1 format.

## Runtime profiles

The existing runtime remains the compact path: it shares model weights, keeps a
bounded adaptive REC session cache, and limits persistent workspace. A future
performance path may retain concrete REC plans for 192, 320, 480, 640, and 960
pixels, but it must remain opt-in until its latency/RSS tradeoff is measured.

The first gate for a performance experiment is the current Tiny/AVX2 baseline
at REC width 960. A candidate is retained only when it preserves the complete
OCR checksum, Golden corpus, line count, and result structure. As a practical
screen, a full-OCR improvement below 2% with more than 50 MiB additional RSS
is not sufficient by itself.

## Offline REC pattern analysis

The first implementation step is analysis-only:

```powershell
python converter/analyze_rec_aot_patterns.py `
  --model models/ppocrv6-tiny/rec.onnx `
  --json-output build/rec-aot-patterns.json `
  --markdown-output build/rec-aot-patterns.md
```

The report records:

- exact ONNX heavy-node shapes and estimated FLOPs;
- repeated Conv families, including node indexes and aggregate work;
- the terminal recognition MatMul candidate;
- candidate reasons for later packed-layout or block-fusion A/B tests.

The analyzer uses ONNX indexes. They are not LWM node indexes and are not a
runtime dispatch contract. It emits no model or cache files.

For the bundled Tiny REC graph, the report currently identifies two repeated
pointwise families:

- `160 -> 320`, output `[1, 320, 3, 80]`, three nodes;
- `320 -> 160`, output `[1, 160, 3, 80]`, three nodes.

It also identifies the terminal `[1, 40, 80] x [80, 6906]` MatMul. These are
candidates for isolated benchmarks, not changes that are enabled automatically.

## Required experiment sequence

1. Measure an isolated kernel or graph-node A/B using the same model, width,
   worker count, and ISA.
2. Measure uninstrumented full OCR mean/P95 and peak RSS.
3. Run the REC graph/pipeline reference tests and the Golden corpus.
4. Repeat on the generated 100-image corpus before making a product-profile
   decision.

Only after a candidate clears those gates should the project consider persistent
multi-width plans, larger worker-private scratch arenas, AOT prepared layouts,
or fused REC blocks. Compact behavior remains the fallback for WASM, low-memory
ARM, and other constrained targets.
## Experimental resident-width profile

Native builds can opt into the first space-for-time experiment with:

```powershell
cmake -S . -B build-performance-vs `
  -DLW_REC_RESIDENT_WIDTHS=ON `
  -DBUILD_TESTING=ON
```

This is deliberately off by default. With the bundled Tiny/AVX2 500x500
fixture, one local five-request smoke comparison measured:

| Profile | Workers | OCR mean | Peak RSS |
|---|---:|---:|---:|
| Compact | 1 | 275.569 ms | 81.7 MiB |
| Resident widths | 1 | 279.086 ms | 97.0 MiB |
| Compact | 4 | 124.498 ms | 128.7 MiB |
| Resident widths | 4 | 111.101 ms | 175.0 MiB |

The resident variant prepared 192/320/480/640/960 sessions for every worker
and switched between them without session construction. Both variants returned
16 lines and checksum `0ebf8b448ab7df47`. These are local smoke measurements,
not portable performance claims; the resident mode is still experimental and
requires the uninstrumented benchmark, full OCR profile, Golden corpus, and
100-image comparison before it can become a user-facing option.

## Reproducible compact/resident comparison

The repository includes `tools/compare_rec_runtime_profiles.py` for paired
benchmark runs. It invokes the compact and resident executables with identical
models, input image, REC width, worker count, and detector thread count, then
reports OCR mean/P95, peak RSS, speedup, and the output contract (line count and
FNV-1a text checksum). Example:

```powershell
python tools/compare_rec_runtime_profiles.py `
  --compact-driver build/Release/full-ocr-intra-benchmark.exe `
  --performance-driver build-performance-vs/Release/full-ocr-intra-benchmark.exe `
  --det build/models/det.lwm --cls build/models/cls.lwm `
  --rec build/models/rec.lwm --dictionary models/ppocrv6-tiny/ppocr_keys.txt `
  --image build/models/sample.ppm --warmup 1 --iterations 5 `
  --workers 4 --target-width 960 --det-threads 4 `
  --json-output build/compact-vs-performance.json `
  --markdown-output build/compact-vs-performance.md
```

The local Tiny/AVX2 4-worker smoke comparison measured 122.460 ms versus
112.691 ms (1.087x speedup, -7.98% mean latency, -8.43% P95) and an additional
45.949 MiB peak RSS. Both runs returned 16 lines with checksum
`0ebf8b448ab7df47`. The numbers are a reproducibility check, not a cross-machine
claim; the resident option remains opt-in until the 100-image paired corpus
clears the same contract and memory gates.

When `LW_REC_RESIDENT_WIDTHS=ON`, the existing `full_ocr_operator_profile` CTest
is additionally run with `--expect-resident`; it requires zero REC session-cache misses
and zero reconfigurations for both one-worker and four-worker cases. The default
Compact build keeps the original cache assertions.

The manual `Native x64 OCR Performance` workflow now includes a `Native x64 Resident A/B` job. It builds both modes from the same prepared assets, runs the Resident zero-reconfiguration gate, and uploads the JSON/Markdown comparison artifact.

## Terminal CTC MatMul benchmark

The terminal REC projection is now covered by an isolated benchmark at its
actual Tiny geometry: `batch=1`, `rows=40`, `inner=80`, and `columns=6906`.
The benchmark compares the packed scalar reference (including bias and
argmax) with the AVX2 fused `MatMul + bias + argmax` path. It is a measurement
and result-contract test only; it does not change dispatch or the public ABI.

Run it after a native build with:

```powershell
ctest --test-dir build-performance-vs -C Release -R packed_matmul_benchmark_smoke --output-on-failure
build-performance-vs/Release/packed-matmul-benchmark-driver.exe 3
```

On the local x64 AVX2 build, three iterations measured `4.680667 ms` for the
scalar reference and `0.794200 ms` for the fused AVX2 path, or `5.893562x`
speedup. Both paths matched output checksum `0x0f75ad457f88c673` and argmax
checksum `0x4facc659e45064c3`. These values are machine-specific; the CTest
contract checks geometry, finite positive timings, and deterministic output
checksums without imposing a latency threshold.

On non-AVX2 targets the driver reports `supported=false` and exits successfully,
so ARM, LoongArch, scalar, and WASM builds do not execute an unsafe AVX2 call.
This benchmark is the first gate for any further terminal projection kernel
work; a proposed change must preserve both checksums before it is compared in
full OCR.

## REC node 6 Conv3x3 stride-2 measurement

The full profile maps its largest REC Conv node (`LWM node 6`) to the second
3x3 stride-2 convolution, `24 -> 48` channels. The existing Conv3x3 benchmark
now includes this exact geometry at both REC widths:

- REC width 320: input `[1,24,24,160]`, output `[1,48,12,80]`;
- REC width 960: input `[1,24,24,480]`, output `[1,48,12,240]`.

The local AVX2 run at width 320 measured `0.794367 ms` for the dispatched path
and `0.625333 ms` for the packed path (`1.270x` packed improvement). At width
960 it measured `1.968733 ms` versus `1.799200 ms` (`1.094x`). Both cases
preserved their scalar-reference checksums. The gain is real but modest, so
this node is now covered by a stable A/B baseline before any further
specialization is considered.

An opt-in AVX2+FMA candidate was then measured for this exact Node 6 geometry.
The candidate is enabled only by `LW_EXPERIMENTAL_AVX2_FMA_DISPATCH=ON` and only
for the two REC widths above; the default build and every other Conv3x3 shape
remain unchanged. On the local AVX2 host, the packed kernel measured `0.619220`
ms (320) and `1.839400` ms (960) in the default build, versus `0.561200` ms
and `1.519500` ms in the experimental build. The corresponding isolated
packed-kernel ratios were `1.103x` and `1.210x`; both builds passed the complete
Conv3x3 benchmark smoke, and the FMA path stayed within a `1.0e-2` maximum
absolute-error bound. A one-shot full-OCR check at width 960 kept checksum
`c9c15dc8d3fe01ab` and showed no working-set increase, but this is not yet a
promotion gate: paired CI measurements across Tiny, Small, and Medium are still
required before enabling the candidate outside the experimental build.
## Latest x64 FMA CI checkpoint

The paired native x64 workflow confirmed the Node 6 FMA candidate at both
production REC widths: `1.141x` for width 320 and `1.134x` for width 960. The
full OCR profiles improved by `3.44%` (1 worker) and `4.05%` (4 workers), while
keeping 16 lines and checksum `0ebf8b448ab7df47`. Peak RSS deltas were `+0.023`
and `+0.037 MiB`, respectively. These are strong experimental results, but the
working-set gate is not yet satisfied, so the candidate stays behind the native
FMA experiment switch.

The same report shows why Conv1x1 remains shape-gated: its median AVX2/FMA ratio
was `1.139x` at width 320 and `1.070x` at width 960, while the slowest shapes
were `0.978x` and `0.981x`. Unknown and regressing shapes must continue using
regular AVX2 until more runner data justifies a narrower or revised allowlist.
## REC Erf measurement

REC profile attribution shows Erf as the next large operator family. The new
`erf-benchmark-driver` uses the actual dynamic shapes at target widths 320 and
960 and checks the existing maximum absolute error contract of `5e-7` against
`erff`.

At target width 960, the AVX2 approximation measured approximately `18.0x` to
`19.2x` faster than scalar `erff` across the six REC shapes. The largest case,
`[1,320,3,240]` (230,400 elements), measured `3.790333 ms` scalar versus
`0.204400 ms` AVX2, with maximum absolute error `2.98023224e-7`. This makes
Erf an important accumulated operator in the graph, but not a promising first
kernel rewrite target: the existing AVX2 path is already efficient and within
its accuracy contract.

## REC pointwise Conv measurement

The repeated REC pointwise families are also covered at their exact graph
shapes: `48↔96` at height 12, `96↔192` at height 6, and `160↔320` at height
3. At target width 960, the current packed AVX2 path measured `24.3x` to
`34.4x` faster than the scalar reference across these six cases; the
`160→320` case measured `24.955667 ms` scalar versus `0.785900 ms` packed.
All outputs preserved their checksums. These results make the existing packed
pointwise kernel another low-priority rewrite target; future work should focus
on graph scheduling, workspace reuse, or end-to-end overhead rather than
replacing this inner loop without new evidence.