# Native x64 finer REC widths (experimental)

`LW_NATIVE_FINE_REC_WIDTHS=ON` reduces recognition padding in compiled native
x64 OCR. It is **OFF by default**, independent of `LW_WASM_FINE_REC_WIDTHS`.
WASM, ARM64 and canonical native builds are unchanged. Configuration rejects
enabling this option without the native x64 fast path.

The existing buckets are 192/320/480/640/960. The experiment uses
192/256/320/384/448/512/576/640/704/768/832/896/960. The minimum remains 192
and maximum remains 960; detection resolution, CLS, reading order and model
weights do not change. Widths above 960 still follow the existing cap. Session
and packed-weight reuse remain in effect; this does not enable resident widths.

Less padding means less REC computation. However, padding also participates in
the model's attention computations: this is a shape-policy change, **not a
numerically equivalent kernel optimization**. Review text quality before
enabling it for a product. Production golden checksums are not relaxed.

## Local sample measurements

Baseline code: `21bc5d8`; candidate: the same source plus this option enabled.
Windows x64, AMD Ryzen 7 7735H (16 logical CPUs), MSVC 2019 Release, official
ONNX assets from `models/ppocrv6-models.json`, shared Tiny CLS enabled,
500x500 bundled PPM, REC max 960. Each case uses three alternating fresh-process
pairs, two warm-ups and five measured OCR calls per process. Values are medians
of per-process means; speedup is the median of paired before/after ratios.

| Model | Workers | Before ms | Fine ms | Paired speedup (range) | Before peak MiB | Fine peak MiB | Full text |
|---|---:|---:|---:|---|---:|---:|---|
| Tiny | 1 | 111.49 | 108.31 | 1.035x (1.027–1.057) | 84.2 | 84.9 | Same |
| Tiny | 4 | 54.41 | 50.27 | 1.082x (1.001–1.113) | 101.6 | 100.3 | Same |
| Small | 1 | 370.14 | 352.00 | 1.052x (1.013–1.125) | 198.4 | 199.6 | Different |
| Small | 4 | 284.05 | 210.81 | 1.347x (1.108–1.467) | 234.1 | 229.8 | Different |
| Medium | 1 | 1214.48 | 1101.61 | 1.116x (1.053–1.169) | 745.1 | 746.1 | Same |
| Medium | 4 | 976.96 | 875.01 | 1.105x (1.095–1.117) | 764.5 | 764.0 | Same |

Sample SHA-256:
`a694ab9def7845b53470aab9f3b819d6469a7b652f820fa30799d918cede4743`.
Peak working set includes initialization and the benchmark's extra standalone
detector, not just an OCR handle. It is not retained allocation size. The extra
CLI text collection runs after timing and is excluded from these measurements.

Small changes one line from `OEMODM` to `OEM ODM`. It is reported as a quality
experiment, not an output-equivalent speedup. Small 4-worker timings show
substantial variance: do not treat 1.347x as a stable corpus-wide gain. Tiny and
Medium retain their existing sample text checksums. A single sample is not a
quality corpus; broader CER/Exact Line Rate and long/short line coverage are
required before promoting this width policy.

An additional one-pair confirmation measured Small 4 workers at 205.56 vs
182.65 ms (1.125x), and Medium 1/4 workers at 1.124x/1.111x. This supports the
direction of the gain, not a universal Small 4-worker improvement of 34.7%.

Several pointwise tiling, cache blocking, spin-wait and Erf experiments were
also measured; none produced a convincing all-model full-OCR benefit. Those
runtime/kernel changes were removed, rather than left enabled on microbenchmarks.

## Reproduce

Build both configurations from **one source revision** with one compiler:

```powershell
cmake -S . -B build/default -A x64 -DLW_EXPERIMENTAL_AVX2_FAST_PATH=ON -DLW_NATIVE_FINE_REC_WIDTHS=OFF -DLW_REC_RESIDENT_WIDTHS=OFF
cmake -S . -B build/fine -A x64 -DLW_EXPERIMENTAL_AVX2_FAST_PATH=ON -DLW_NATIVE_FINE_REC_WIDTHS=ON -DLW_REC_RESIDENT_WIDTHS=OFF
cmake --build build/default --config Release --target lw-ocr-benchmark lw-ocr-ppm
cmake --build build/fine --config Release --target lw-ocr-benchmark lw-ocr-ppm
python tools/compare_onnx_e2e.py --baseline build/default/Release/lw-ocr-benchmark.exe --candidate build/fine/Release/lw-ocr-benchmark.exe --sample build/default/models/sample.ppm --rounds 3 --iterations 5 --report-text-differences --collect-text --output build/onnx-width-comparison
```

The manual GitHub Actions workflow **Native x64 ONNX fine REC widths A/B** runs
the same comparison. Default production builds must pass `native_onnx_import`
and all existing golden contracts first. The experimental comparison records
full text differences without overwriting any golden. Run/worker nondeterminism,
invalid results and incorrect model identities remain hard failures. Latency
and memory on hosted runners are informational.

Reports include binary/model/sample SHA identities, selected CMake options,
individual process measurements, paired ratio ranges, and untimed full text
whose buffer checksum must match the measured benchmark. All generated outputs
stay under ignored `build/` directories.
