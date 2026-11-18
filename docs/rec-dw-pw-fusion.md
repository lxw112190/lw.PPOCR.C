# Native REC DW3x3 to Pointwise experiment

## Decision

`LW_EXPERIMENTAL_REC_DW_PW_FUSION` remains **OFF**. The implementation preserves
bitwise results, but the full OCR experiment establishes neither a repeatable
latency improvement nor a peak memory reduction. Do not promote it as a release
optimization. It is separate from the [FFN experiment](rec-ffn-fusion.md); both
FFN options must be OFF for this experiment.

The baseline is `db93832` plus the preceding uncommitted FFN experiment with
all fusion flags disabled. Control and candidate use the same source tree,
compiler, official ONNX assets and dictionary; only DW/PW fusion differs.
This is native x64 only. C ABI, model files, width policy and Web defaults
remain unchanged. No upstream source was copied.

## What the actual models permit

The PP-OCRv6 REC backbone uses DW3x3, not DW5x5. Tiny also has an axis 1x5
depthwise near its head; Small/Medium have axis 1x7 depthwise. They are outside
this experiment. SE/reduction branches and values escaping as graph outputs,
aliases or the CTC activation cannot be elided.

Inspection of the lowered physical graphs is essential: many apparently
adjacent DW/PW operations also retain the DW result as a later FFN residual.
Those are multi-consumer values and **must not** be removed by a two-op fusion.
A DW/Add/PW matcher was tried during development, but hit no actual model
paths and was removed. It is not a supported feature. Fusing an entire
DW/FFN/residual block would need a separate lifetime proof and A/B experiment;
the current flag deliberately does not broaden its coverage that way.

The actual eligible pair counts at width 960 are Tiny 2, Small 2, Medium 3.
For diagnosis, the contract driver accepts `<rec.onnx> --dump` and prints DW
physical op indices, semantic indices and the next physical op kinds.

## Execution

The compiler requires an adjacent, non-fallback NHWC DW3x3/PW pair and a private
single-consumer intermediate. It discovers the pair before arena placement,
extends the DW input's physical lifetime through PW, and removes the DW output
from arena allocation. Lowering pass two reconstructs and validates the sentinel
offsets, pair kinds, scratch requirement and eliminated-value count.

The range primitive reads the original image, including halo outside the
current output tile, and writes a compact flattened pixel range starting at
scratch[0]. Output addresses are computed by subtracting integer pixel indices;
it never constructs a pointer before the scratch allocation. Both ordinary
and range execution use the same ascending-tap FMA and post-bias operations.
Interior pairs retain the existing two-pixel AVX2 kernel. Range boundaries,
row crossings, padding and channel tails use the existing single-pixel kernel.

PW consumes that tile with its existing selected kernel and epilogue. Worker
boundaries align to 12 pixels; the configured tile is 12, 48 or 96 pixels.
Each worker uses independent reusable scratch. Allocation failure falls back
to the preallocated serial workspace. Input/output and residual/output overlap
are rejected: the image halo makes input overwrite unsafe here. Single-op
diagnostics reject the eliminated hidden address; use a control build when
requiring a materialized intermediate.

`LW_X64_REC_PROFILE=1` emits `REC_DW_PW` elapsed/call/tile diagnostics separately
from ordinary Pointwise/Depthwise counters, avoiding attribution of both ops
to DW alone. The normal profiling-disabled execution retains the existing
cached environment check behavior.

## Full OCR A/B

Windows x64, Ryzen 7 7735H, MSVC 19.29 Release, bundled 500x500 sample. Same
official ONNX models, dictionary and PPM input; CLS enabled; REC maximum width
960 and the existing 192/320/480/640/960 policy. Three alternating paired rounds,
two warm-ups and five measured iterations per fresh process; 48-pixel tiles.
No build or correctness tests ran concurrently with timed measurements.

| Model | Workers | Control ms | Fusion ms | Paired speedup (range) | Control / fusion peak WS MiB |
| --- | ---: | ---: | ---: | --- | --- |
| Tiny | 1 | 110.39 | 109.02 | 1.013x (1.007-1.060) | 84.4 / 85.0 |
| Tiny | 4 | 51.75 | 50.58 | 1.010x (1.002-1.035) | 101.8 / 102.4 |
| Small | 1 | 428.89 | 405.93 | 1.057x (0.957-1.212) | 199.0 / 200.0 |
| Small | 4 | 217.09 | 242.73 | 0.923x (0.855-0.997) | 234.2 / 235.8 |
| Medium | 1 | 1,616.31 | 1,595.73 | 1.013x (0.860-1.029) | 745.4 / 745.4 |
| Medium | 4 | 980.80 | 965.70 | 1.011x (1.008-1.020) | 764.7 / 768.0 |

Every paired full-text checksum matched, and the untimed CLI text sideband
was checked against it. Speedups are medians of within-round ratios, not
quotients of the displayed medians. Tiny and Medium/4 show about a one-percent
gain in this final run, while Small/4 regresses across all three pairs. Small/1
and Medium/1 ranges cross 1.0. Peak WS includes initialization and the benchmark's
additional standalone detector. Final raw reports are ignored artifacts under
`build/rec-dw-pw-fusion/e2e-final-ab/`.

The initial run is retained under `e2e-ab/`: its paired speedups were
Tiny 1.003x/0.997x, Small 1.006x/0.999x and Medium 1.040x/0.993x for 1/4 workers;
all ranges crossed 1.0 and peak memory did not improve. The final run used
the completed binaries after cleanup and range validation. Taken together,
these runs do not justify promoting the experiment, especially for Small/4.

Removing the intermediate did not shrink the measured process peak: retained
models/constants, other live ranges, arena placement and worker scratch still
contribute. The width-960 candidate capacities are:

| Model | REC arena bytes | Serial scratch bytes |
| --- | ---: | ---: |
| Tiny | 2,764,800 | 18,432 |
| Small | 9,400,320 | 36,864 |
| Medium | 12,349,440 | 98,304 |

These are individual REC program capacities, not total engine RSS.

## Existing profiler evidence for the next step

One diagnostic control process per model used one worker, one warm-up and
one measured OCR call. Only `X64REC` calls with a nonzero CTC time were summed,
excluding CLS. The denominator is backbone op work plus CTC projection time.
Each model contributed 32 REC line calls. Instrumentation affects absolute
latency, so these percentages are hotspot indicators, not end-to-end speedups.

| Model | Pointwise | Dense | DW | Attention MatMul | CTC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny | 44.38% | 8.45% | 18.16% | 1.64% | 16.47% |
| Small | 34.36% | 6.58% | 9.78% | 13.47% | 16.35% |
| Medium | 62.64% | 3.07% | 6.14% | 7.03% | 8.84% |

Other operators account for the remaining work. Only a fraction of DW is
covered by this fusion. Further work should target measured model/shape
hotspots; expanding planner coverage alone is not evidence of faster OCR.
Raw diagnostics are in the ignored `build/rec-dw-pw-fusion/profile/` directory.

## Validation and reproduction

Eight selected CTest tests passed: three model bitwise contracts; public full
OCR and REC Golden Corpus; portable depthwise; comparison-tool contracts; and
source newlines. Model tests cover five widths and 1/4 workers, independently
materializing the complete DW plane for the reference, then checking final
CTC indices and probabilities bitwise. Range tests cover padding, row-crossing
tiles, stride 1 and 2, non-32-aligned channel tails, bias/post-bias, canaries,
and invalid ranges. Runtime tests reject missing pair flags, overlap, escaping
arena offsets, insufficient scratch and invalid single-op access.

Shared depthwise, REC compiler and REC executor sources also passed local
Emscripten SIMD128 strict-warning compile checks. This was a source compilation
check, not a browser OCR benchmark or a complete WASM release build.
The preceding FFN experiment's three model contracts were also rebuilt and
passed after these shared-source changes.

```powershell
cmake -S . -B build/dw-pw-fusion -DBUILD_TESTING=ON -DLW_BUILD_HTTP_DEMO=OFF -DLW_EXPERIMENTAL_REC_DW_PW_FUSION=ON -DLW_REC_DW_PW_TILE_PIXELS=48 -DLW_EXPERIMENTAL_REC_FFN_FUSION=OFF -DLW_EXPERIMENTAL_REC_FFN_TILING=OFF -DLW_NATIVE_FINE_REC_WIDTHS=OFF
cmake --build build/dw-pw-fusion --config Release --target rec-dw-pw-contract-driver lw-ocr-benchmark lw-ocr-ppm ocr-pipeline-driver rec-pipeline-driver nhwc-depthwise-portable-driver
ctest --test-dir build/dw-pw-fusion -C Release -R '^(rec_dw_pw_contract_.*|nhwc_depthwise_portable|full_ocr_golden_corpus|rec_golden_corpus)$' --output-on-failure
```

Build a separate control directory with DW/PW fusion OFF; compare with
`tools/compare_onnx_e2e.py --baseline <control-driver> --candidate <fusion-driver>
--sample <sample.ppm> --rounds 3 --iterations 5 --collect-text --output <report-dir>`.
The `Native REC DW-PW fusion experiment` CI runs Windows/Linux x64 contracts,
full OCR A/B, summary tables and raw-log artifacts. Text/numerics gate; timings
and memory do not. Remote CI results are pending submission.
