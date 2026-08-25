# Private REC preprocessing and CTC milestone

The private PP-OCR REC path now recognizes a real cropped text region without
OpenCV, ONNX Runtime, or another deployment library. It is a correctness gate,
not a public ABI. ONNX Runtime and Pillow are development-only test oracles.

## Input and preprocessing contract

`lw_rec_preprocess_bgr_u8` accepts decoded, interleaved BGR unsigned-byte
pixels. It does not decode JPEG, PNG, or another image-file format. The caller
provides the source byte count, width, height, and row stride so padded rows are
handled safely.

The output is contiguous FP32 CHW data with shape `[1, 3, 48, W]`:

1. compute the resized width as `min(W, ceil(48 * source_width / source_height))`;
2. resize with half-pixel bilinear coordinates and clamped borders;
3. retain BGR channel order and normalize each value with `value * 2 / 255 - 1`;
4. fill the unused right side with source value 128 before normalization.

Interpolation is performed directly in floating point. It is intentionally not
defined in terms of OpenCV's implementation or an intermediate resized `uint8`
image, so it remains reproducible without OpenCV. The native result is compared
against an independent NumPy implementation for odd sizes, padded input rows,
and right padding.

All source and destination size calculations are checked before indexing. The
current milestone fixes batch size and height to the exact bundled REC model.

## Dictionary and CTC contract

The dictionary loader accepts a UTF-8 text file, an optional UTF-8 BOM, and LF
or CRLF line endings. Invalid UTF-8, embedded NUL bytes, an empty file, and an
all-empty dictionary are rejected. Interior empty labels are preserved because
the production PP-OCRv6 dictionary intentionally contains one.

For a dictionary with `N` lines, the decoder expects exactly `N + 2` classes:

- class `0` is the CTC blank;
- classes `1..N` map to dictionary lines in file order;
- class `N + 1` is an implicit ASCII space.

At each time step, the earliest maximum wins ties. Blank classes are skipped,
and adjacent repeated classes collapse unless separated by a blank. The score
is the arithmetic mean of the selected maximum probability for emitted
non-blank classes. No allocation occurs during decoding.

Text output follows a caller-owned buffer contract. A null output buffer queries
the required UTF-8 byte capacity, including the trailing NUL. An undersized
buffer reports the same required capacity without returning partial text.

## Real golden test

`rec_pipeline_reference` crops `[left=20, top=30, right=315, bottom=76]` from
`models/ppocrv6-tiny/sample.jpg`, converts the crop to BGR, and preprocesses it
to width 320. It then runs both:

- the independent Python preprocessing + original ONNX model + Python CTC path;
- the pure-C preprocessing + private LWM executor + pure-C CTC path.

Both paths must produce exactly `纯臻营养护发素` with seven decoded characters;
the C score must also match the oracle within the test tolerance. Synthetic
tests additionally cover BOM/CRLF, Chinese, ASCII, multi-byte UTF-8, implicit
space, blank-separated repeats, output-buffer sizing, invalid UTF-8, all-empty
dictionaries, and non-finite probabilities.

The same seventeen-test suite passes locally on Windows x64 and Windows x86,
including shared-library exports, the public C Demo, and staged-package smoke.
Linux CI and physical Windows 7 validation remain separate platform claims.

The extended ten-crop correctness gate is documented in
[`rec-golden-corpus.md`](rec-golden-corpus.md).

## Public wrapper and deliberate boundary

The low-level functions remain under `src/ppocr/rec_internal.h`. Applications
use the opaque `lw_recognizer` API in `include/lw_infer.h`, which owns the model,
dictionary, session, and preallocated input/output tensors. Callers own source
pixels and the UTF-8 text buffer. A recognizer must not be called concurrently;
separate recognizers have independent mutable state and may run in parallel.

Encoded image decoding is deliberately not part of this milestone. A future
convenience layer may decode JPEG/PNG, but the dependency-free core contract
continues to accept decoded BGR8 pixels.
