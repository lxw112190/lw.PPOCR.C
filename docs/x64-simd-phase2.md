# x64 SIMD Phase 2B — AVX2+FMA Conv1x1

Phase 2B hardens the measurement and dispatch-safety foundation for optional
AVX2+FMA kernels. It does not change the default OCR kernel selection.

## Runtime capability model

lw_simd_level remains the mutually exclusive backend selector. The internal
lw_cpu_capabilities snapshot adds:

- has_fma: hardware FMA, AVX and OSXSAVE are present and XMM/YMM state is
  enabled by the operating system;
- has_avx2_fma: the active backend is AVX2 and FMA is usable.

The SIMD level is cached by the runtime probe, and each session stores one
capability snapshot. Existing low-level wrappers therefore no longer issue
repeated CPUID/XGETBV probes during a graph run. The public C ABI, LWM format,
and model files are unchanged.

## FMA candidate

src/simd/avx2_fma_packed_conv1x1.c is an isolated Conv1x1 candidate.
It uses the same PACKED4 weight layout and geometry as the current AVX2
Conv1x1 kernel, but uses explicit _mm256_fmadd_ps instructions. The second
candidate, src/simd/avx2_fma_packed_matmul.c, covers only the Tiny terminal
CTC projection shape `[1,40,80] x [80,6906]` and keeps the existing packed
weight layout. Both are compiled with AVX2+FMA target attributes on GCC/Clang
and `/arch:AVX2` on MSVC.

The candidates are deliberately not connected to the default production dispatch.
This keeps non-FMA hosts safe and preserves the current deterministic OCR path.
A native-only experimental build may opt into a shape-aware dispatch policy
while paired A/B data is collected.

## Benchmark contract

The packed Conv1x1 benchmark invokes the regular AVX2 and FMA entry points directly; it does not use the dispatch wrapper as the AVX2 baseline. Each case records the requested batch (currently 1), input geometry including width, and seven interleaved ABBA rounds. The report exposes median, minimum, maximum, and p90 timings for both kernels, plus AVX2/FMA ratio and absolute/relative FMA error.

The packed Conv1x1 benchmark reports optional fields when the host supports
AVX2+FMA. On x64 it also measures the isolated 8-output × 8-spatial
lw_avx2_fma_packed_conv1x1_8x8_f32 candidate with a second interleaved
ABBA sequence. The packed MatMul benchmark separately measures the terminal
projection shape and checks its argmax contract:

- fma_ms;
- fma_speedup;
- fma_max_abs_error;
- fma_checksum;
- fma8_ms, fma8_vs_fma, fma8_max_abs_error, and fma8_checksum (x64 only);
- avx2_ms, avx2_min_ms, avx2_max_ms, avx2_p90_ms, and fma_vs_avx2.

The candidates must remain finite and within the current exploratory absolute
error bound of `1e-2`. The smoke tests only validate that they are measurable,
machine-readable, and numerically bounded; they do not promote them to the
default backend.

Run locally after configuring a Release build:

    cmake --build build --target packed-conv1x1-benchmark-driver packed-matmul-benchmark-driver --config Release
    build\Release\packed-conv1x1-benchmark-driver.exe 960 20
    build\Release\packed-matmul-benchmark-driver.exe 20
    ctest --test-dir build -C Release -R "packed_(conv1x1|matmul)_benchmark_smoke" --output-on-failure

## End-to-end experiment build

The candidate can be evaluated in a separate native build without changing the
normal dispatch. Configure that build with:

    cmake -S . -B build-fma -G "Visual Studio 17 2022" -A x64 -DLW_EXPERIMENTAL_AVX2_FMA_DISPATCH=ON

Then run the same `full-ocr-intra-benchmark` command against the default and
`build-fma` binaries. The experimental option is native-only, defaults to OFF,
and does not change the public ABI or model files. The performance workflow also
builds the experimental Conv1x1 and terminal MatMul drivers and runs both smoke
tests. It is intended for paired latency, checksum, and RSS measurements only.

The Native x64 OCR Performance workflow runs this experiment at 1 worker/1 DET
thread and 4 workers/4 DET threads. The JSON and Markdown outputs are uploaded
as the `lw-ppocr-x64-fma-ocr-results-*` artifact.
The Conv1x1 summary also lists the three slowest and three fastest shapes,
which is the input for a future shape-aware dispatch policy.

## Shape-aware experimental dispatch

`LW_EXPERIMENTAL_AVX2_FMA_DISPATCH=ON` enables a native-only policy that routes
only measured beneficial shapes to the FMA candidates. The current explicit
8x8 candidate allowlist is intentionally small:

- Medium 512 -> 1024 at height 6;
- late 768 -> 384 at height 3;
- late 1536 -> 768 at height 3.
The same experimental build now contains a separate FMA candidate for the REC
Node 6 stride-2 Conv3x3 shape `24 -> 48`. It is gated to batch 1, input height
24, output height 12, and widths `160 -> 80` or `480 -> 240` (REC target widths
320 and 960). It reuses the existing packed 3x3 layout and is selected only
when the cached capability snapshot reports AVX2+FMA. All other stride-2
Conv3x3 shapes continue to use the regular AVX2 path. The candidate is still
opt-in and is not part of the default build.
The 384 -> 768 late shape remains on the regular four-output FMA path; the
8x8 candidate is intentionally not used for it. Unknown shapes, 1024 -> 512,
and all other medium/late shapes remain on regular AVX2. The terminal Tiny MatMul
candidate continues to use its separate exact-shape gate. The default build keeps the
existing AVX2 dispatch and is unchanged.

The experimental benchmarks accept the small FMA rounding difference with a
`1.0e-2` maximum absolute error bound; the default benchmarks remain byte-exact.
The terminal MatMul candidate improves the isolated local benchmark by about 1.4x
over the existing AVX2 path. Full OCR remains checksum-identical in local paired
runs; promotion still requires the gates below on the full corpus.

## 100-image quality parity checkpoint

The project-owned generated corpus was replayed locally with the default AVX2
driver and the experimental FMA driver using the same Tiny DET/CLS/REC assets,
REC target width `960`, and manifest (`seed=20260907`, `614` reference lines).
The reports were compared with `tools/compare_ocr_dataset_reports.py`:

| Metric | Default AVX2 | Experimental FMA | Delta |
|---|---:|---:|---:|
| Detection F1 | 99.3517% | 99.3517% | 0.0000 pp |
| Exact reference-line rate | 58.1433% | 58.1433% | 0.0000 pp |
| CER on matched lines | 3.6458% | 3.6458% | 0.0000 pp |

Detection precision, recall, mean matched IoU, matched-line exact rate, missing
lines, and extra lines were also identical. This is a quality-parity checkpoint,
not a release gate: generated images remain local-only, and the FMA dispatch stays
opt-in until repeated runner measurements confirm the performance and working-set
gates below.

## Latest x64 CI checkpoint

The latest native x64 FMA run confirms that the shape-gated experiment is useful
but is not yet a blanket replacement for AVX2. The paired Conv3x3 Node 6
candidate measured `1.141x` at REC width 320 and `1.134x` at width 960. The
Conv1x1 direct A/B median was `1.139x` at width 320 and `1.070x` at width 960,
but the slowest measured shapes were `0.978x` and `0.981x`, respectively. This
supports keeping explicit shape allowlists rather than routing every Conv1x1
shape to FMA.

The same run measured complete OCR mean latency reductions of `3.44%` for the
1-worker profile and `4.05%` for the 4-worker profile. Both profiles retained
checksum `0ebf8b448ab7df47` and 16 lines. Peak RSS changed by only `+0.023 MiB`
and `+0.037 MiB`, but the current promotion policy requires no stable working-set
increase, so the candidate remains opt-in pending another paired run and the
Tiny/Small/Medium corpus gate.

## Medium 1024 -> 512 exclusion checkpoint

A paired local A/B was run against the Medium REC profile at target width 960
with two OCR iterations, using the same Windows x64 host and the same checksum
contract. The direct Conv1x1 benchmark had suggested that the `1024 -> 512`,
height-6 shape might benefit from FMA, so it was tested both with the regular
four-output FMA kernel and with the eight-output-by-eight-spatial candidate.
Neither variant improved the complete OCR pipeline:

| Candidate | 1 worker total | 4 workers total | checksum |
|---|---:|---:|---|
| Existing shape-gated FMA | 16182.5 ms | 11568.3 ms | `c9c15dc8d3fe01ab` |
| Enable 1024 -> 512 four-output FMA | 16219.0 ms (+0.23%) | 12000.2 ms (+3.73%) | identical |
| Enable 1024 -> 512 8x8 FMA | 16252.3 ms (+0.43%) | 11908.2 ms (+2.94%) | identical |

The result is a useful negative checkpoint: isolated-kernel speedups are not
sufficient evidence for promotion when FMA frequency effects and worker-level
scheduling are included. The shape remains on regular AVX2, and the 8x8
allowlist is unchanged.

## Small 384 -> 192 candidate checkpoint

The Small REC profile contains repeated `384 -> 192`, height-6 Conv1x1 nodes.
The isolated benchmark was added as `middle-384x192` and measured AVX2/FMA
ratios of `1.008x` at REC width 320 and `1.091x` at width 960. A paired
end-to-end Small OCR check with three iterations produced:

| REC width | 1 worker AVX2/FMA | 4 workers AVX2/FMA | checksum |
|---:|---:|---:|---|
| 320 | 1.052x | 1.075x | identical |
| 960 | 1.021x | 0.994x | identical |

Because the 960-width multi-worker result is neutral and the 320-width kernel
ratio is close to parity, this shape is not added to the experimental runtime
allowlist yet. The benchmark case remains so future CI runs can re-evaluate it
with more replicas and the same FMA contract.

## Promotion gate

Before enabling FMA in the production dispatch, collect paired measurements on
the real Tiny, Small, and Medium REC shapes at widths 320 and 960. Require:

1. at least 5% median improvement in the target kernel family;
2. no stable shape regression above 2%;
3. complete OCR paired median improvement of at least 2% on the 100-image corpus;
4. identical text, line count, and reading order;
5. detection geometry and scores within an explicitly recorded tolerance;
6. no peak working-set increase.

If the candidate does not meet these gates, remove it and retain the current
non-FMA AVX2 path. The shape-gated 8x8 kernel remains an x64 benchmark candidate only
until the same shape-aware and end-to-end gates are met; it is not part of the
production dispatcher. AVX512 remains a later, profile-driven experiment.

## Promoted narrow Conv3x3 FMA dispatch

The measured REC Node 6 Conv3x3 path is now controlled by the independent
`LW_AVX2_FMA_CONV3X3_DISPATCH` option, which defaults to `ON` for native builds
and remains disabled for WebAssembly. Runtime dispatch still requires AVX2+FMA
capability and an exact Node 6 geometry (`24 -> 48` channels, input height 24,
REC widths 320 or 960). All other Conv3x3 shapes continue to use the regular
AVX2 or scalar packed kernel.

This promotion is intentionally separate from
`LW_EXPERIMENTAL_AVX2_FMA_DISPATCH`: the broader Conv1x1 and terminal MatMul FMA
candidates remain opt-in because their shape matrix still contains regressions.
The promoted path passed the x64 full-OCR A/B checksum and 100-image quality
parity gates with no measurable RSS increase.
