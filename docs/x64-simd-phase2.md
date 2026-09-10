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
AVX2+FMA. The packed MatMul benchmark separately measures the terminal
projection shape and checks its argmax contract:

- fma_ms;
- fma_speedup;
- fma_max_abs_error;
- fma_checksum;
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
only the measured beneficial Tiny/REC Conv1x1 shapes and the terminal Tiny MatMul
shape to the FMA candidates. Unknown shapes and the measured Conv1x1 regressions
(the large medium/late shapes) remain on regular AVX2.
The default build keeps the existing AVX2 dispatch and is unchanged.

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
non-FMA AVX2 path. AVX512 and small-plane 8x8 kernels remain later,
profile-driven experiments.
