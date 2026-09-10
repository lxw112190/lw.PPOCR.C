# C ABI v1 freeze candidate

`v0.2.0-preview.2` starts the C ABI v1 soft-freeze period. This is a
compatibility candidate, not a permanent ABI guarantee yet. A later release
will promote this contract only after the candidate client, package, and
cross-platform compatibility tests have passed.

## Stable candidate scope

The candidate scope is the high-level, decoded-pixel OCR API in
`include/lw_infer.h`:

- common status and error helpers;
- `lw_recognizer_*` for one pre-cropped text line;
- `lw_classifier_*` for 0/180 degree orientation classification;
- `lw_detector_*` for DB-style text detection and reading order;
- `lw_ocr_*` for composed detection, optional classification, and recognition.

The exact exported symbol list is maintained in
`abi/exports-v1-candidate.txt`. The machine-readable summary is
`abi/c-abi-v1-candidate.json`. It is installed under docs/abi/ in the
development package.

## Not frozen by this candidate

The following remain experimental in the 0.2.x preview line:

- `lw_model_*` and `lw_session_*` low-level planning APIs;
- the internal graph executor and tensor scheduling details;
- the LWM v0.1 file format;
- the separate WebAssembly Host ABI.

Applications should use the recognizer, classifier, detector, and full-OCR
handles for integration. The low-level model/session API is useful for tests
and experiments but is not covered by the candidate compatibility promise.

## Contract rules

The current preview validates the complete known `struct_size` for public calls. Prefix-compatible input/output handling is a requirement to verify before the permanent freeze; this candidate does not claim that older shorter structures are already accepted.

- All public structures begin with `struct_size` and must be initialized with
  their matching `_init` function.
- Existing structure prefixes and enum numeric values are retained after the
  eventual freeze. Additive fields can only be appended under the documented
  size/version rules.
- Strings crossing the boundary are UTF-8.
- Input pixels are caller-owned interleaved BGR8. JPEG/PNG decoding remains an
  application responsibility.
- Output buffers are caller-owned. An insufficient capacity returns
  `LW_STATUS_OUT_OF_BOUNDS` without copying a partial result.
- Every successful create has one matching free; freeing `NULL` is safe.
- A handle must not be used concurrently unless the API explicitly documents
  otherwise. Use independent handles for parallel calls.

## Recognition-only integration

For a caller that already has a cropped, single-line text image:

```text
decode image to BGR8
        |
lw_recognizer_create
        |
lw_recognizer_recognize_bgr_u8
        |
UTF-8 text + score
```

This path does not run detection and does not return coordinates. It also does
not run CLS automatically. If orientation is unknown, call
`lw_classifier_classify_bgr_u8`, rotate the caller-owned crop when required,
then call the recognizer.

Use `lw_recognizer_get_info` to allocate `max_text_capacity` once for the
normal one-pass path. The two-pass `NULL`/zero-capacity call is available when
an exact output allocation is required.

## Release gate

The candidate is considered ready for an RC only when all of the following
remain green:

1. structure-size and enum-value checks in `tests/test_abi.c`;
2. exact legacy export checks in `abi/exports-v0.txt`;
3. candidate stable-symbol checks in `abi/exports-v1-candidate.txt`;
4. recognition-only buffer, error, and lifecycle tests;
5. candidate client tests against the staged Windows and Linux packages;
6. Tiny, Small, and Medium model-pack compatibility tests.

The ABI may still receive corrections during the preview line. Once the final
freeze is announced, removing or changing an existing candidate symbol,
structure prefix, enum value, ownership rule, or error semantic requires a new
ABI major version.
