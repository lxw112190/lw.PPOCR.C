# Experimental fine-grained WASM REC widths

This experiment targets Tiny/Small standalone HTML full OCR. It changes the
adaptive REC padding policy, not the C ABI, model weights, DET resolution,
SIMD arithmetic, or CTC algorithm. `LW_WASM_FINE_REC_WIDTHS` defaults to OFF
and requires compiled SIMD128 WASM REC. `LW_WASM_COMPILED_REC` also remains
OFF globally. Native, canonical WASM, and Legacy builds retain five buckets.

v1.2.0 release Tiny/Small HTML and SDK opt into compiled SIMD128 but still
keep this fine-width experiment OFF: their release policy also has five
buckets. The thirteen-bucket timings/text policy below remain experimental.

| Policy | Width buckets (pixels) |
| --- | --- |
| Existing | 192, 320, 480, 640, 960 |
| Fine | 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960 |

The existing selection logic uses the smallest supported bucket that fits the
resized text line, capped by the configured maximum. The default maximum
`target_width` remains 960. All thirteen compiled slots share the existing
constant ownership and lazy canonical-fallback mechanisms. This is not lazy
creation of the compiled slots themselves.

DET's default `limit_side_length` is 960, not a fixed 512 input. The bundled
500x500 image happens to round to 512x512 through `lw_det_compute_size()`.
This experiment does not lower DET resolution or change that resize policy.

## Local standalone HTML results

2026-09-30, Windows local Chromium 151 / Emscripten 4.0.15, bundled 500x500
JPEG, CLS enabled, compiled SIMD128 / 2x16 / lazy fallback ON. Both builds use
the same source; only the fine-width option differs. Three fresh-context
paired rounds alternate individual OCR calls, with no simultaneous inference;
each build has three warm-ups per round. Tiny has seven measured calls per
round and Small five.

| Model | Five widths ms | Fine widths ms | Paired speedup | Five widths heap MiB | Fine widths heap MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny | 537.50 | 487.20 | 1.105x | 59.25 | 59.88 |
| Small | 2460.30 | 2139.90 | 1.140x | 132.38 | 133.44 |

Latency includes the complete Demo `recognize()` call and result/UI update,
but excludes initial engine loading and image decoding. Latencies are medians
of round medians; speedup is the median of within-round paired ratios, so it
need not equal division of the displayed median latencies. Heap is WASM linear
memory capacity, **not browser process RSS**. These are local sample results,
not mobile benchmarks or evidence of general corpus accuracy.

Startup and first-call measurements are kept separately:

| Model / metric | Five widths ms | Fine widths ms |
| --- | ---: | ---: |
| Tiny, page navigation to ready | 444.83 | 451.42 |
| Tiny, CLS engine rebuild | 273.95 | 274.94 |
| Tiny, first OCR | 583.40 | 541.00 |
| Small, page navigation to ready | 1137.57 | 1111.93 |
| Small, CLS engine rebuild | 782.09 | 796.88 |
| Small, first OCR | 2535.40 | 2240.20 |

These startup figures include browser/loading effects and are diagnostic;
they do not establish an initialization speedup.

Separate Node profile runs reuse the existing REC width counters. Padding is
`(target_width_sum - resized_width_sum) / target_width_sum`, weighted over
recognized lines, not an unweighted average of per-line ratios:

| Model | Resized width sum | Five target sum | Fine target sum | Five padding | Fine padding |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny | 8344 | 9920 | 8640 | 15.89% | 3.43% |
| Small | 8006 | 9792 | 8448 | 18.24% | 5.23% |

Both policies executed compiled DET/CLS/REC without fallback on this sample.
These instrumented Node runs are not the browser latency measurements.

## Text contract: reviewed difference, not silent parity

Tiny retains its existing golden: sixteen lines and checksum
`4341424a06f714666cd3adfe8698beda3dbea45de3695d794b29d79f9a801dcf`.

Small has one reviewed difference, at zero-based line 4:

- Five widths: `【品牌】:代加工方式/OEMODM`
- Fine widths: `【品牌】:代加工方式/OEM ODM`

The user confirmed that **OEM ODM with the space is correct**. The existing
five-width Small golden is unchanged. Fine widths use a separate explicit
[reviewed golden](../ci/web-ppocrv6-small-fine-widths.json), checksum
`9cd560aaff37f1013cf10ebd9c616f4e2446b800985a9d15aa408d4010cdba95`.
Both produce sixteen deterministic lines and pass their respective strict
contracts. The report explicitly records `texts_identical: false`; this is
not a byte-identical A/B. No whitespace normalization or allowed-hash list
masks the difference. Adding 480 back as a fourteenth bucket did not eliminate
it, and that trial was not retained.

Changing padding can affect model output even with unchanged weights and
arithmetic. Wider corpus and mobile-device checks remain necessary before
promoting the fine policy to a default. The sample alone is insufficient.

## Reproduce

Activate Emscripten and a Python environment with converter dependencies,
Playwright and Ninja as described in [local browser A/B](wasm-browser-benchmark.md).
Build both configurations explicitly:

```powershell
foreach ($mode in @('baseline', 'candidate')) {
    $fine = if ($mode -eq 'candidate') { 'ON' } else { 'OFF' }
    emcmake cmake -S . -B "build/fine-$mode" -G Ninja `
      -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF `
      -DLW_BUILD_HTTP_DEMO=OFF -DLW_BUILD_WEB_MODEL_VARIANTS=ON `
      -DLW_WASM_COMPILED_REC=ON -DLW_WASM_REC_LAZY_FALLBACK=ON `
      -DLW_WASM_REC_SIMD_KERNELS=ON -DLW_WASM_POINTWISE_ROWS=2 `
      "-DLW_WASM_FINE_REC_WIDTHS=$fine"
    cmake --build "build/fine-$mode" --parallel 8 `
      --target lw-ocr-html lw-ocr-html-small wasm-rec-lazy-contract wasm-pointwise-contract
    node "build/fine-$mode/wasm-rec-lazy-contract.js"
    node "build/fine-$mode/wasm-pointwise-contract.js"
}
python web/benchmark_ocr_html.py `
  --baseline build/fine-baseline/ocr-demo.html `
  --candidate build/fine-candidate/ocr-demo.html `
  --sample models/ppocrv6-tiny/sample.jpg --golden ci/web-ppocrv6-tiny.json `
  --baseline-widths 5 --candidate-widths 13 --interleave `
  --iterations 7 --rounds 3 --output build/fine-candidate/tiny-ab.json
python web/benchmark_ocr_html.py `
  --baseline build/fine-baseline/ocr-demo-small.html `
  --candidate build/fine-candidate/ocr-demo-small.html `
  --sample models/ppocrv6-tiny/sample.jpg --golden ci/web-ppocrv6-small.json `
  --candidate-golden ci/web-ppocrv6-small-fine-widths.json `
  --baseline-widths 5 --candidate-widths 13 --interleave `
  --iterations 5 --rounds 3 --output build/fine-candidate/small-ab.json
python tools/summarize_wasm_fine_widths.py --directory build/fine-candidate
```

## CI and verification

The manual **WASM Tiny Small fine REC widths A-B** workflow builds both policies
from one revision, runs the private lazy-fallback and Pointwise contracts,
Tiny HTML/SDK regression, Small reviewed-golden HTML regression, browser A/B,
and separate Node padding/fallback diagnostics. It publishes a job summary
and JSON/log artifacts even when a later step fails:

```powershell
gh workflow run wasm-fine-widths.yml --ref main
```

The lazy contract checks 5/5 or 13/13 slots on source and clones, exercises all
bucket boundaries and forced canonical reconstruction, and compares text.
Partial or missing compiled-width coverage fails validation. The Pointwise
contract covers 792 bitwise 2x16/4x16 cases, activations, channel/pixel tails,
bias/residual/in-place behavior and output bounds. Tiny's full HTML and SDK
tests cover exports, lifecycle and repeated-run heap behavior. The existing
Erf/GELU contract also passes 65,536 vector blocks.

Timing is informational, not a hosted-runner gate. Neither this workflow nor
its reviewed Small golden changes the normal five-width release tests. Medium
is deliberately outside this browser optimization experiment.

See [WASM REC physical-op profile](wasm-rec-op-profile.md) for the next experiment.
