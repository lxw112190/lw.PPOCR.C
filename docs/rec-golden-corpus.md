# REC golden corpus

The v0.1 correctness gate contains ten real text-line crops from the bundled
PP-OCR sample image. Crop coordinates and expected UTF-8 recognition text are
versioned in `tests/fixtures/rec-golden-corpus.json`; the manifest also pins the
source image SHA-256 so coordinates cannot silently target different pixels.

For every crop, `rec_golden_corpus` performs two independent checks:

1. the original ONNX model, run by ONNX Runtime during development, must produce
   the versioned expected text;
2. the pure-C `.lwm` recognizer must match the ONNX text, character count, and
   confidence score within the existing FP32 tolerance.

The corpus covers Chinese text, ASCII, digits, punctuation, mixed scripts, and
different aspect ratios. ONNX Runtime and Pillow remain test-only dependencies;
neither is linked into or shipped with the C runtime.

These ten crops improve regression coverage but do not constitute a general OCR
accuracy benchmark. Future corpus additions should use independently sourced,
redistributable images and preserve the manifest schema and source hashes.
