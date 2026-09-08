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