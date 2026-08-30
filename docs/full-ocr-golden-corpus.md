# Full-OCR golden corpus

The v0.1 full-OCR regression gate is versioned in
`tests/fixtures/ocr-golden-corpus.json`. The manifest pins the SHA-256 of the
source image, DET/CLS/REC ONNX models, and recognition dictionary so a model or
fixture change cannot silently reuse stale expectations.

`full_ocr_golden_corpus` runs seven deterministic cases through the public
three-model OCR pipeline:

- the original 500x500 image with direction classification enabled;
- the same image with direction classification disabled;
- integer-defined nearest-neighbor resizing to 375x375 and 750x750;
- the source on a 500x600 white canvas;
- a byte-exact 90-degree counterclockwise rotation;
- a 640x192 white image with no text.

The derived pixels do not depend on Pillow or OpenCV resizing implementations.
The test defines nearest-neighbor indices with integer arithmetic and uses NumPy
only to apply those indices, rotations, and canvas operations. Pillow decodes
the one bundled JPEG, as it does in the existing reference tests.

Every case freezes the exact UTF-8 result and reading order, detected count,
DET resize shape, classifier labels, and applied rotations. Scores use explicit
lower bounds instead of exact floating-point comparison. Every quadrilateral
must remain within the source image and have nonzero area. The original image
also compares all eight coordinates of all 16 boxes with a two-pixel tolerance.
The blank case freezes the valid zero-line/zero-text result.

This corpus broadens pipeline regression coverage across scale, aspect ratio,
empty input, classifier options, and rotated text, but it is derived from one
redistributable source image. It is not a general OCR accuracy benchmark.
Future additions should prefer independently sourced redistributable images,
record their hashes, and review changed expectations separately from runtime
optimizations.
