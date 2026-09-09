# x64 SIMD Phase 2A

Phase 2A establishes the measurement and dispatch-safety foundation for optional
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

src/simd/avx2_fma_packed_conv1x1.c is an isolated candidate implementation.
It uses the same PACKED4 weight layout and geometry as the current AVX2
Conv1x1 kernel, but uses explicit _mm256_fmadd_ps instructions. It is
compiled with AVX2+FMA target attributes on GCC/Clang and /arch:AVX2 on
MSVC.

The candidate is deliberately not connected to production dispatch yet. This
keeps non-FMA hosts safe and preserves the current deterministic OCR path while
the A/B data is collected.

## Benchmark contract

The existing packed Conv1x1 benchmark reports optional fields when the host
supports AVX2+FMA:

- fma_ms;
- fma_speedup;
- fma_max_abs_error;
- fma_checksum.

The candidate must remain finite and within the current exploratory absolute
error bound of 1e-2. The smoke test only validates that it is measurable,
machine-readable, and numerically bounded; it does not promote it to the
default backend.

Run locally after configuring a Release build:

    cmake --build build --target packed-conv1x1-benchmark-driver --config Release
    build\Release\packed-conv1x1-benchmark-driver.exe 960 20
    ctest --test-dir build -C Release -R packed_conv1x1_benchmark_smoke --output-on-failure

## End-to-end experiment build

The candidate can be evaluated in a separate native build without changing the
normal dispatch. Configure that build with:

    cmake -S . -B build-fma -G "Visual Studio 17 2022" -A x64 `
      -DLW_EXPERIMENTAL_AVX2_FMA_DISPATCH=ON

Then run the same `full-ocr-intra-benchmark` command against the default and
`build-fma` binaries. The experimental option is native-only, defaults to OFF,
and does not change the public ABI or model files. It is intended for paired
latency, checksum, and RSS measurements only.
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
