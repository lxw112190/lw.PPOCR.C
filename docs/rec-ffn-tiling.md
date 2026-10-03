# Experimental REC FFN cache-local scheduling

## Decision

This experiment is **not promoted to a production default**. Cache-local
Pointwise expansion / exact GELU / Pointwise projection scheduling passed
correctness checks, but did not deliver a repeatable Tiny/Small HTML speedup.
`LW_EXPERIMENTAL_REC_FFN_TILING` remains OFF. Global compiled REC, Fine REC
widths, model assets, golden text and public C ABI defaults are unchanged.

The baseline was `11a0e19024153ece35d1dc3379f9121dfe7e2aba`. Local measurements
used Windows x64, Ryzen 7 7735H and the bundled 500x500 sample, with CLS enabled
and the standard 192/320/480/640/960 width policy. These results are not a
comparison against the preceding thirteen-Fine-width experiment and are not
a 100-image corpus quality or performance claim.

## Implementation and lifetime

The compiler marks only adjacent, non-fallback NHWC Pointwise pairs with a
single-consumer intermediate, exact GELU on expansion, and no activation on
projection. Model outputs and the CTC activation cannot become intermediates.
CLS is excluded. The first input's physical lifetime is extended through
projection **before arena placement**, including existing alias-root lifetime
propagation. Pair flags are reconstructed on the second lowering pass.

The executor runs expansion then projection on 12, 48 or 96 pixels, using the
existing kernels and epilogues. Blocks and native worker boundaries preserve
the supported row-tile alignment. There is no new hot-path allocation, weight
packing format, multiplication reassociation, approximate GELU or CTC rewrite.
The full planned intermediate slot remains available to one-op diagnostics:
this is scheduling, not intermediate output-elision or reduced workspace.

The private pair hook checks shapes, kernel eligibility, overflow, arena bounds,
input/intermediate/output disjointness and residual overlap before any writes.
Detailed WASM one-op tracing deliberately executes the original separate ops;
ordinary timing must run without that diagnostic. The aggregate REC profile
adds `REC_FFN width=... pairs=... elapsed=... tile=...` when pairs execute.

## Full OCR measurements

Native: three alternating paired rounds, two warm-ups and five measured
iterations per fresh process. Both revisions used identical official ONNX
models, dictionaries and PPM input. Speedup is the median paired baseline /
candidate ratio, not the quotient of the displayed latency medians.
Native candidate blocks were 96 pixels.

| Model | Workers | Baseline ms | Candidate ms | Paired speedup (range) | Baseline / candidate peak WS MiB |
| --- | ---: | ---: | ---: | --- | --- |
| Tiny | 1 | 113.69 | 110.87 | 1.025x (1.009-1.040) | 84.5 / 84.5 |
| Tiny | 4 | 56.32 | 55.11 | 1.021x (0.994-1.023) | 101.7 / 101.6 |
| Small | 1 | 378.84 | 378.82 | 1.009x (0.993-1.016) | 198.5 / 198.3 |
| Small | 4 | 221.22 | 226.59 | 0.976x (0.873-1.010) | 234.0 / 234.0 |
| Medium | 1 | 1323.57 | 1316.15 | 0.981x (0.979-1.011) | 745.1 / 745.2 |
| Medium | 4 | 1053.47 | 1045.24 | 1.011x (0.995-1.025) | 764.8 / 764.7 |

All paired full-text checksums matched. Peak WS includes model initialization
and the benchmark's additional standalone detector. Except Tiny/1, paired
ranges cross 1.0; Tiny/1's modest gain does not justify enabling the candidate
for the intended Tiny/Small browser workload.

Browser: Chromium 151.0.7922.34, Emscripten 4.0.15, three paired rounds, three
warm-ups and five measured OCR runs per round. Both sides use compiled SIMD128,
Pointwise 2x16, lazy fallback and the same five-width policy. The 12-pixel Small
experiment alternates individual OCR calls between two contexts; the other
experiments alternate whole context runs. Timings cover the existing HTML
benchmark's OCR path, not startup. Heap is retained linear memory after runs,
not process RSS. Forty-eight-pixel blocks were not benchmarked.

| Model | Block pixels | Baseline ms | Candidate ms | Paired speedup | Baseline / candidate heap MiB |
| --- | ---: | ---: | ---: | ---: | --- |
| Tiny | 96 | 537.9 | 538.8 | 0.997x | 59.25 / 59.25 |
| Small | 96 | 2348.7 | 2424.7 | 0.969x | 138.75 / 138.75 |
| Tiny | 12 | 548.3 | 551.0 | 0.995x | 59.25 / 59.25 |
| Small | 12 | 2286.0 | 2326.0 | 0.999x | 138.75 / 138.75 |

All browser runs preserved the existing model-specific 16-line golden checksum
and repeat determinism. Small/96 had substantial round-to-round timing drift;
the interleaved Small/12 check still showed no useful gain. Increasing locality
alone did not remove the dominant computation. No memory improvement is claimed.

Raw local reports are deliberately ignored build artifacts:
`build/rec-ffn-candidate/e2e-ab/report.json`, and
`build/wasm-ffn-candidate/{tiny,small,tiny-12,small-12}-browser-ab.json`.

## Validation and reproduction

The opt-in `rec_ffn_contract_<variant>` tests use actual Tiny/Small/Medium REC
graphs, all five widths and serial/four-worker execution. Each matched pair's
intermediate and projected output is compared bitwise with separate execution.
Tail prefixes 1/2/3/5/11/12/13/95/96/97, rejected malformed pairs without writes,
final backbone activations, CTC indices and probabilities are checked. Tests
require real matched pairs (Tiny 9, Small 13, Medium 14 per graph), so an inactive
candidate cannot silently pass.

Configure a native candidate (choose the installed native generator):

```sh
cmake -S . -B build/rec-ffn-candidate -DBUILD_TESTING=ON -DLW_BUILD_HTTP_DEMO=OFF -DLW_EXPERIMENTAL_REC_FFN_TILING=ON -DLW_REC_FFN_TILE_PIXELS=96 -DLW_NATIVE_FINE_REC_WIDTHS=OFF
cmake --build build/rec-ffn-candidate --config Release --target rec-ffn-contract-driver
ctest --test-dir build/rec-ffn-candidate -C Release -R "^rec_ffn_contract_(tiny|small|medium)$" --output-on-failure
```

After activating Emscripten and installing the existing converter and browser
test dependencies, build both HTML configurations. For the baseline, use a
separate build directory and set only the experimental FFN flag to OFF:

```sh
emcmake cmake -S . -B build/wasm-ffn-candidate -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DLW_BUILD_HTTP_DEMO=OFF -DLW_BUILD_WEB_MODEL_VARIANTS=ON -DLW_DEPLOY_ONNX_MODELS=ON -DLW_WASM_COMPILED_REC=ON -DLW_WASM_REC_LAZY_FALLBACK=ON -DLW_WASM_POINTWISE_ROWS=2 -DLW_WASM_FINE_REC_WIDTHS=OFF -DLW_EXPERIMENTAL_REC_FFN_TILING=ON -DLW_REC_FFN_TILE_PIXELS=12
cmake --build build/wasm-ffn-candidate --target lw-ocr-html lw-ocr-html-small
python web/benchmark_ocr_html.py --baseline build/wasm-ffn-baseline/ocr-demo.html --candidate build/wasm-ffn-candidate/ocr-demo.html --sample models/ppocrv6-tiny/sample.jpg --golden ci/web-ppocrv6-tiny.json --rounds 3 --iterations 5 --output build/wasm-ffn-candidate/tiny-12-browser-ab.json
python web/benchmark_ocr_html.py --baseline build/wasm-ffn-baseline/ocr-demo-small.html --candidate build/wasm-ffn-candidate/ocr-demo-small.html --sample models/ppocrv6-tiny/sample.jpg --golden ci/web-ppocrv6-small.json --rounds 3 --iterations 5 --interleave --output build/wasm-ffn-candidate/small-12-browser-ab.json
```

Local native contracts, Tiny full-OCR/REC golden corpora, report/quality Python
contracts and Tiny/Small browser A/B checks passed. ASan/UBSan validation is
configured as a separate opt-in candidate build in `runtime-sanitizers.yml`;
it has not been run locally and still requires the remote CI run after pushing.
The cross-platform service skill informed this isolated sanitizer gate; its
OpenCV dependency preflight is not applicable to this dependency-free C runtime.

Next arithmetic-preserving work should target the measured Pointwise kernels'
compute/packing or actual intermediate traffic, with the same correctness and
end-to-end A/B contracts. Simply shortening the scheduler block is not an
established performance improvement.
