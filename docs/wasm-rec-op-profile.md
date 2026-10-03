# WASM REC physical-op profile

Scope: Tiny/Small compiled SIMD128, Pointwise 2x16, lazy fallback and the same
thirteen Fine REC widths. Global compiled REC and Fine defaults remain unchanged.
DET still limits its side to 960; the 500x500 sample rounds to 512x512.

## Prepared Dense decision

Three interleaved Chromium A/B rounds used three warm-ups and seven measured
Tiny runs or five Small runs per round. Both sides used Fine widths13.

| Model | Median paired baseline/candidate ratio | Baseline/candidate heap MiB |
| --- | --- | --- |
| Tiny | 1.015x | 59.875 / 59.875 |
| Small | 1.008x | 133.4375 / 133.625 |

These are paired per-round ratios, not ratios of pooled latency medians.
Every run preserved its text checksum. Both gains fell below the design's 2%
threshold, so production Prepared Dense flags, metadata, compiler and executor
integration were removed. Only the standalone 1,296-case bitwise kernel
contract remains. This commit does not claim a retained runtime speedup.

## Reproduce the physical profile

The manual workflow builds packages, validates ordinary browser HTML/SDK
separately, then profiles Node with one warm-up and three measured iterations:

    gh workflow run wasm-rec-op-profile.yml --ref main

Node diagnostics require these environment variables set to 1:
LW_WASM_REC_OP_PROFILE, LW_X64_REC_PROFILE, LW_WASM_OCR_PROFILE and
LW_REC_MEMORY_PROFILE. Summarize the captured stderr:

    python tools/summarize_wasm_rec_ops.py --log tiny-rec-ops.log --output tiny-ops.json --markdown tiny-ops.md --stage-output tiny-stages.json --variant tiny --skip-ocr-runs 1

Tiny uses its existing golden; Small uses the reviewed Fine-width golden
containing OEM ODM. Reports and logs are uploaded as workflow artifacts.
Ordinary browser runs do not enable the Node-only diagnostic.

BEGIN/OP/END records identify invocation, width and sequential physical index.
Malformed, failed and truncated invocations are rejected. CLS is excluded.
Warm-up removal uses complete OCR boundaries. Operators, pairs and triples
are ranked by accumulated elapsed time without crossing invocation boundaries.
Overlapping windows are not estimates of fusion savings.

## Local measured hotspots

Three retained OCR runs produced 48 REC invocations per model:

| Model | Backbone total ms | Pointwise ms / backbone share | Separate CTC-head ms |
| --- | --- | --- | --- |
| Tiny | 851.280 | 678.260 / 79.68% | 192.516 |
| Small | 4607.195 | 3568.427 / 77.45% | 768.382 |

Instrumented totals are not ordinary browser latency. Backbone shares exclude
the CTC head. The top width640 adjacent pair was Pointwise expansion with
exact GELU followed by Pointwise projection with residual:

| Model | Physical indices | Semantic indices | Channels | Pair total ms / calls |
| --- | --- | --- | --- | --- |
| Tiny | 46,47 | 127,134 | 160 -> 320 -> 160 | 22.267 / 9 |
| Small | 83,84 | 201,208 | 384 -> 768 -> 384 | 80.049 / 6 |

Both operate on 3x160 pixels and include post-bias. Exact GELU prevents simply
multiplying the weight matrices. The subsequent cache-local FFN scheduling
experiment verified dataflow, single-consumer ownership and physical lifetime,
but found no useful Tiny/Small HTML speedup and remains default-OFF. See
[the implementation, contracts and A/B results](rec-ffn-tiling.md).
No DW->PW fusion or expanded operator coverage was added.

## Integration and validation

The existing stage parser now accepts Fine widths and the original 480 bucket;
independent coverage gates still require exactly 5 or 13 slots.
Node packaging now depends on the generated runtime file, preventing stale
runtime.cjs after incremental builds.

Local validation passed: 18 Python unit tests, 1,296 Dense bitwise cases,
Tiny full HTML/SDK, Small browser text contract, both Node profiles with no
canonical fallback, and Windows native core compilation.
No Medium, pthread, relaxed SIMD, fast-math, DET-resolution change or CTC
rewrite is included. Timing remains informational.
