# Local standalone HTML A/B

v1.2.0 release CI selects the compiled SIMD128 / 2×16 / lazy-fallback profile
for Tiny/Small modern HTML and SDK only; global compiled REC remains OFF.
Medium and Node remain canonical. Measurements below are historical local
A/B evidence, not tagged v1.2.0-v1.1.0 timings or current ONNX import heap
claims. Fine widths and FFN tiling are not enabled in the release profile.

For the subsequent Tiny/Small thirteen-bucket experiment, its reviewed Small
text difference and strict A/B commands, see
[fine-grained WASM REC widths](wasm-fine-rec-widths.md).

2026-09-30: Emscripten 4.0.15, Chromium 151, bundled 500x500 JPEG,
CLS enabled, adaptive REC capped at 960. The same browser runs each build
in a fresh context; revision order alternates. Three warm-ups precede five
measured calls. Image decode and engine initialization are excluded; timings
include the full `recognize()` call and Demo result materialization/UI update.

| Model | Canonical ms | Compiled ms | Paired speedup | Canonical heap MiB | Compiled heap MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny | 962.10 | 610.90 | 1.575x | 70.50 | 59.25 |
| Small | 4395.30 | 2663.50 | 1.684x | 134.44 | 132.38 |

Tiny used two alternating rounds and Small three. Latencies above are medians
of round medians; speedup is the median of paired ratios. Every call matched
the variant's checked-in Web golden. Linear heap capacity is **not** browser
process RSS. These local numbers are not comparable to hosted-runner Node PPM
measurements, nor evidence of mobile browser performance.

The improvement comes from enabling the existing compiled SIMD128 / 2x16 /
lazy-fallback backend, not a newly improved kernel. Global
`LW_WASM_COMPILED_REC` remains OFF. A forced two-IC Pointwise unroll regressed
Tiny full OCR (0.877x) and was discarded. LTO was inconclusive (1.021x) and
is not part of the recommended configuration.

## Reproduce

Activate Emscripten and a Python environment containing converter dependencies,
Playwright and Ninja. On this Windows setup:

```powershell
. ..\emsdk\emsdk_env.ps1
. .\build\wasm-tools\Scripts\Activate.ps1
```

Configure canonical and compiled builds explicitly:

```powershell
emcmake cmake -S . -B build/wasm-canonical -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DLW_BUILD_HTTP_DEMO=OFF `
  -DLW_WASM_COMPILED_REC=OFF -DLW_BUILD_WEB_MODEL_VARIANTS=ON
emcmake cmake -S . -B build/wasm-fast -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DLW_BUILD_HTTP_DEMO=OFF `
  -DLW_WASM_COMPILED_REC=ON -DLW_WASM_REC_LAZY_FALLBACK=ON `
  -DLW_WASM_POINTWISE_ROWS=2 -DLW_BUILD_WEB_MODEL_VARIANTS=ON
cmake --build build/wasm-canonical --target lw-ocr-html lw-ocr-html-small --parallel 8
cmake --build build/wasm-fast --target lw-ocr-html lw-ocr-html-small --parallel 8
python web/benchmark_ocr_html.py `
  --baseline build/wasm-canonical/ocr-demo.html `
  --candidate build/wasm-fast/ocr-demo.html `
  --sample models/ppocrv6-tiny/sample.jpg --golden ci/web-ppocrv6-tiny.json `
  --output build/wasm-fast/tiny-browser-ab.json
```

For Small use `ocr-demo-small.html` in both directories and
`ci/web-ppocrv6-small.json`. Default benchmark settings are three paired rounds,
three warm-ups and nine measured calls. Golden text/line count and repeatability
are gates; latency is informational. JSON retains every measured time and heap.

HTML now depends on the generated SDK file as well as its build target,
including model variants. Incremental runtime changes therefore update the
embedded WASM instead of silently leaving a stale HTML. Successful compiled
status lines emitted on stderr are matched strictly by browser regression
tests; unknown console errors and JavaScript page errors still fail tests.

## Erf/GELU optimization on top of the compiled baseline

The next local round optimized actual SIMD computation, not backend selection.
The Node profile identified REC as about 77% of Tiny full OCR, with Pointwise
post-bias/GELU the main REC hotspot. Instrumented timings guided the work;
the results below come from separate uninstrumented Chromium HTML A/B runs.

Changes:

- Skip middle/large Erf polynomials when all four SIMD lanes have magnitude
  below 1. Mixed intervals, infinities and NaNs keep the previous selection path.
- Share the original polynomial coefficients and GELU division/add/multiply
  sequence in a private inline header. OC16 epilogues consume vector values
  directly instead of storing to a stack block and calling/reloading GELU.
- Preserve standard SIMD128, non-FMA arithmetic, scalar tails, and public ABI.
  No relaxed SIMD, fast-math, reciprocal substitution, or new approximation.

Three alternating rounds, three warm-ups each:

| Configuration | Before ms | After ms | Paired speedup | Heap before/after MiB |
| --- | ---: | ---: | ---: | ---: |
| Tiny compiled, 7 measured calls/round | 589.80 | 566.50 | 1.036x | 59.25 / 59.25 |
| Small compiled, 5 measured calls/round | 2551.80 | 2438.50 | 1.058x | 132.38 / 132.38 |
| Tiny canonical, 5 measured calls/round | 1036.90 | 963.90 | 1.068x | 70.50 / 70.50 |

Latencies are medians of round medians; speedups are medians of paired ratios,
so dividing the displayed median latencies need not produce the paired ratio.
All calls matched the unchanged goldens. The combined compiled HTML grows
about 11.5 KiB; the canonical Tiny HTML grows about 0.7 KiB. Neither allocates
additional runtime buffers. These sample results are not mobile benchmarks.
The first small-interval-only experiment measured 1.065x for Tiny on a separate
run; it must not be added to or multiplied by the combined result above.

`wasm-erf-contract` compares the new Erf/GELU vectors against the pre-optimization
unconditional selection path over 65,536 deterministic vector blocks, uniform
and mixed boundary inputs, signed zero, subnormals, infinities and NaNs. It also
checks direct register GELU, in-place calls, and scalar tails. Finite results
and Erf outputs are bitwise checked; arithmetic GELU NaN payloads are not
specified by WebAssembly and are checked as NaN. The contract preserves the
existing polynomial approximation, not equality to mathematical `erf`.

```powershell
cmake --build build/wasm-fast --target wasm-erf-contract --parallel 8
node build/wasm-fast/wasm-erf-contract.js
```

The SIMD128 contract runs in browser-WASM CI. Native/scalar builds retain their
existing scalar `erff` implementation; the compiled REC global default stays OFF.
