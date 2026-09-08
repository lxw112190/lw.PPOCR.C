# PP-OCR AOT performance work

This document defines the staged, evidence-driven path for using additional
memory and model-side prepared data to reduce OCR latency. It does not change
the public C ABI or the LWM v0.1 format.

## Runtime profiles

The existing runtime remains the compact path: it shares model weights, keeps a
bounded adaptive REC session cache, and limits persistent workspace. A future
performance path may retain concrete REC plans for 192, 320, 480, 640, and 960
pixels, but it must remain opt-in until its latency/RSS tradeoff is measured.

The first gate for a performance experiment is the current Tiny/AVX2 baseline
at REC width 960. A candidate is retained only when it preserves the complete
OCR checksum, Golden corpus, line count, and result structure. As a practical
screen, a full-OCR improvement below 2% with more than 50 MiB additional RSS
is not sufficient by itself.

## Offline REC pattern analysis

The first implementation step is analysis-only:

```powershell
python converter/analyze_rec_aot_patterns.py `
  --model models/ppocrv6-tiny/rec.onnx `
  --json-output build/rec-aot-patterns.json `
  --markdown-output build/rec-aot-patterns.md
```

The report records:

- exact ONNX heavy-node shapes and estimated FLOPs;
- repeated Conv families, including node indexes and aggregate work;
- the terminal recognition MatMul candidate;
- candidate reasons for later packed-layout or block-fusion A/B tests.

The analyzer uses ONNX indexes. They are not LWM node indexes and are not a
runtime dispatch contract. It emits no model or cache files.

For the bundled Tiny REC graph, the report currently identifies two repeated
pointwise families:

- `160 -> 320`, output `[1, 320, 3, 80]`, three nodes;
- `320 -> 160`, output `[1, 160, 3, 80]`, three nodes.

It also identifies the terminal `[1, 40, 80] x [80, 6906]` MatMul. These are
candidates for isolated benchmarks, not changes that are enabled automatically.

## Required experiment sequence

1. Measure an isolated kernel or graph-node A/B using the same model, width,
   worker count, and ISA.
2. Measure uninstrumented full OCR mean/P95 and peak RSS.
3. Run the REC graph/pipeline reference tests and the Golden corpus.
4. Repeat on the generated 100-image corpus before making a product-profile
   decision.

Only after a candidate clears those gates should the project consider persistent
multi-width plans, larger worker-private scratch arenas, AOT prepared layouts,
or fused REC blocks. Compact behavior remains the fallback for WASM, low-memory
ARM, and other constrained targets.
## Experimental resident-width profile

Native builds can opt into the first space-for-time experiment with:

```powershell
cmake -S . -B build-performance-vs `
  -DLW_REC_RESIDENT_WIDTHS=ON `
  -DBUILD_TESTING=ON
```

This is deliberately off by default. With the bundled Tiny/AVX2 500x500
fixture, one local five-request smoke comparison measured:

| Profile | Workers | OCR mean | Peak RSS |
|---|---:|---:|---:|
| Compact | 1 | 275.569 ms | 81.7 MiB |
| Resident widths | 1 | 279.086 ms | 97.0 MiB |
| Compact | 4 | 124.498 ms | 128.7 MiB |
| Resident widths | 4 | 111.101 ms | 175.0 MiB |

The resident variant prepared 192/320/480/640/960 sessions for every worker
and switched between them without session construction. Both variants returned
16 lines and checksum `0ebf8b448ab7df47`. These are local smoke measurements,
not portable performance claims; the resident mode is still experimental and
requires the uninstrumented benchmark, full OCR profile, Golden corpus, and
100-image comparison before it can become a user-facing option.

## Reproducible compact/resident comparison

The repository includes `tools/compare_rec_runtime_profiles.py` for paired
benchmark runs. It invokes the compact and resident executables with identical
models, input image, REC width, worker count, and detector thread count, then
reports OCR mean/P95, peak RSS, speedup, and the output contract (line count and
FNV-1a text checksum). Example:

```powershell
python tools/compare_rec_runtime_profiles.py `
  --compact-driver build/Release/full-ocr-intra-benchmark.exe `
  --performance-driver build-performance-vs/Release/full-ocr-intra-benchmark.exe `
  --det build/models/det.lwm --cls build/models/cls.lwm `
  --rec build/models/rec.lwm --dictionary models/ppocrv6-tiny/ppocr_keys.txt `
  --image build/models/sample.ppm --warmup 1 --iterations 5 `
  --workers 4 --target-width 960 --det-threads 4 `
  --json-output build/compact-vs-performance.json `
  --markdown-output build/compact-vs-performance.md
```

The local Tiny/AVX2 4-worker smoke comparison measured 122.460 ms versus
112.691 ms (1.087x speedup, -7.98% mean latency, -8.43% P95) and an additional
45.949 MiB peak RSS. Both runs returned 16 lines with checksum
`0ebf8b448ab7df47`. The numbers are a reproducibility check, not a cross-machine
claim; the resident option remains opt-in until the 100-image paired corpus
clears the same contract and memory gates.

When `LW_REC_RESIDENT_WIDTHS=ON`, the existing `full_ocr_operator_profile` CTest
is additionally run with `--expect-resident`; it requires zero REC session-cache misses
and zero reconfigurations for both one-worker and four-worker cases. The default
Compact build keeps the original cache assertions.

The manual `Native x64 OCR Performance` workflow now includes a `Native x64 Resident A/B` job. It builds both modes from the same prepared assets, runs the Resident zero-reconfiguration gate, and uploads the JSON/Markdown comparison artifact.
