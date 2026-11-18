# Native REC tile-local FFN experiment

The follow-up [DW3x3/Pointwise experiment](rec-dw-pw-fusion.md) documents actual
model coverage, independent end-to-end measurements and the next hotspot data.

## Decision

`LW_EXPERIMENTAL_REC_FFN_FUSION` is OFF by default. The implementation and
bitwise contracts pass, but full OCR A/B does **not** support promoting it:
most paired latency ranges cross 1.0 and peak working set usually increases.
Removing one intermediate is not sufficient evidence of lower process memory.
No release speedup or memory reduction is claimed.

This is a separate native x64 experiment based on `db93832`, following the
[earlier full-hidden scheduling experiment](rec-ffn-tiling.md). The existing
tiling option, WASM defaults, model assets, REC width policy and public ABI
are unchanged. This implementation uses the project's own packed NHWC
Pointwise kernels and exact GELU; no upstream source was copied.

## Execution and physical lifetime

The compiler marks adjacent NHWC Pointwise/GELU/Pointwise pairs only when the
hidden tensor has one consumer, no aliases and no graph-output/CTC escape.
Before arena placement it removes that hidden value from allocation candidates
and extends the first input's physical lifetime through projection. The second
lowering pass reconstructs the pair and validates its sentinel offsets and
scratch requirement. Separate one-op execution deliberately rejects elided
hidden addresses.

The FFN primitive expands and projects a compact pixel tile. Kernel selection
happens once outside its tile loop. It retains the existing packed weights,
ascending-channel FMA, post-bias/residual order and exact GELU implementation.
Each worker has independent scratch, reused through the existing dense shard
scratch capacity. Serial scratch participates in the existing shared workspace.
Parallel allocation failure uses the already allocated serial workspace.

Projection can reuse the input only when channel counts and storage sizes
match, the root is neither constant nor caller input, and all aliases reach
their last use by projection without escaping as graph outputs. Each worker
finishes expansion for its entire tile before overwriting those input rows.
Partial overlaps and unapproved aliases are rejected before writing.

At width 960, the actual model contract tests report:

| Model | Pairs | In-place pairs | Arena bytes | Serial scratch bytes |
| --- | ---: | ---: | ---: | ---: |
| Tiny | 9 | 7 | 2,764,800 | 122,880 |
| Small | 13 | 11 | 9,400,320 | 294,912 |
| Medium | 14 | 11 | 12,349,440 | 589,824 |

These are individual compiled REC program capacities, not full OCR RSS or
peak working set. They exclude models, packed constants, DET/CLS and other
worker instances. In particular, a different live-range/placement pattern
can increase the arena high-water mark despite eliminating hidden storage.

## Local full OCR A/B

Windows x64, Ryzen 7 7735H (16 logical CPUs), MSVC 19.29 Release. Identical
official ONNX models, dictionary, 500x500 sample and CLS enabled; maximum REC
width 960 with the existing five-width adaptive policy. Three alternating
paired rounds, fresh processes, two warm-ups and five measured OCR runs each.
Both binaries were built from the same working tree with only fusion enabled
on the candidate. FFN tiles were 96 pixels. No concurrent build was running.

| Model | Workers | Control ms | Fusion ms | Paired speedup (range) | Control / fusion peak WS MiB |
| --- | ---: | ---: | ---: | --- | --- |
| Tiny | 1 | 143.55 | 140.44 | 1.022x (0.992-1.212) | 84.9 / 87.4 |
| Tiny | 4 | 68.98 | 74.86 | 0.913x (0.825-1.014) | 101.9 / 105.9 |
| Small | 1 | 498.44 | 498.91 | 1.016x (0.993-1.020) | 198.8 / 207.4 |
| Small | 4 | 295.61 | 271.34 | 1.108x (0.998-1.115) | 234.8 / 245.3 |
| Medium | 1 | 2,194.10 | 1,690.25 | 1.039x (1.035-1.391) | 745.3 / 745.3 |
| Medium | 4 | 1,717.45 | 1,665.98 | 1.000x (0.994-1.043) | 764.7 / 785.2 |

Speedup is the median of paired ratios, not the quotient of displayed latency
medians. Runner drift is visible, especially Medium/1; its displayed medians
must not be interpreted as a 1.30x matched gain. Peak WS includes initialization
and the benchmark's extra standalone detector. Every paired full-text checksum
matched; the untimed CLI text sideband was also checked against that checksum.

The preceding scratch-only version also failed to establish a broad win.
Raw reports remain ignored local artifacts under
`build/rec-ffn-fusion/{e2e-ab,e2e-inplace-ab}/`. Nothing in these measurements
justifies changing the release defaults.

## Validation and reproduction

Seven targeted CTest tests passed: three model FFN contracts, full OCR and REC
Golden Corpus, benchmark comparison contracts, and source final newlines.
The FFN tests cover five widths and 1/4 workers; independent full hidden
reference execution; four native row kernels; non-aligned channels and pixel
tails; bias/post-bias/residual; in-place projection; canaries; and invalid
overlap, arena bounds, missing flags and insufficient scratch rejection.

```powershell
cmake -S . -B build/ffn-fusion -DBUILD_TESTING=ON -DLW_BUILD_HTTP_DEMO=OFF -DLW_EXPERIMENTAL_REC_FFN_FUSION=ON -DLW_EXPERIMENTAL_REC_FFN_TILING=OFF -DLW_REC_FFN_TILE_PIXELS=96 -DLW_NATIVE_FINE_REC_WIDTHS=OFF
cmake --build build/ffn-fusion --config Release --target rec-ffn-contract-driver lw-ocr-benchmark lw-ocr-ppm ocr-pipeline-driver rec-pipeline-driver
ctest --test-dir build/ffn-fusion -C Release -R '^(rec_ffn_contract_.*|full_ocr_golden_corpus|rec_golden_corpus)$' --output-on-failure
```

Build a separate control directory with fusion OFF, then use
`tools/compare_onnx_e2e.py --baseline <control-driver> --candidate <fusion-driver>
--sample <sample.ppm> --rounds 3 --iterations 5 --collect-text --output <report-dir>`.

The `Native REC FFN fusion experiment` workflow runs the contracts and complete
three-model 1/4-worker A/B on Windows x64 and Linux x64. It is available manually
and for relevant pull requests; latency/memory are informational, while text
and numerical correctness are gating. Single-op backend diagnostic suites
requiring a materialized hidden plane should use the control build.
