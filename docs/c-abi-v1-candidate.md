# C ABI v1 frozen contract

The C ABI v1 contract is frozen for the approved 1.0 scope and is shipped by
the `v1.0.0` stable package; v1.2.0 preserves it unchanged. The historical
`*-candidate` filenames are
retained so existing package and documentation paths remain valid.

## Frozen scope

The frozen scope is the high-level, decoded-pixel OCR API in
`include/lw_infer.h`:

- common status and error helpers;
- `lw_recognizer_*` for one pre-cropped text line;
- `lw_classifier_*` for 0/180 degree orientation classification;
- `lw_detector_*` for DB-style text detection and reading order;
- `lw_ocr_*` for composed detection, optional classification, and recognition.

The exact exported symbol list is maintained in
`abi/exports-v1-candidate.txt` (the filename is retained for path
compatibility). The machine-readable summary is
`abi/c-abi-v1-candidate.json`. Cross-product version and compatibility
metadata is recorded in `abi/runtime-contract-v1.json` and checked by the
versioning test. It is installed under docs/abi/ in the
development package.

## Not frozen by this contract

The following remain experimental and are outside the stable v1 contract:

- `lw_model_*`, `lw_session_*`, and `lw_tensor_desc_init` low-level planning APIs;
- the internal graph executor and tensor scheduling details;
- LWM v0.1 remains an internal Preview format; the WebAssembly Host ABI is
  frozen separately in `abi/web-abi-v1-candidate.json`.

Applications should use the recognizer, classifier, detector, and full-OCR
handles for integration. The low-level model/session API is useful for tests
and experiments but is not covered by the frozen compatibility promise.

## Contract rules

Output metadata queries now accept a caller-advertised prefix: the `*_get_info`
functions and `lw_session_get_output_desc` copy only the common bytes and leave
the caller's original `struct_size` intact. A larger caller structure is also
accepted; fields unknown to this library are ignored. This behavior is covered
by `c_abi_prefix_compatibility`.

Stable high-level input options (`lw_recognizer_options`,
`lw_classifier_options`, `lw_detector_options`, and `lw_ocr_options`) also
accept a caller-advertised prefix. Missing fields retain the documented
defaults, including missing nested OCR options. Recognition result structures
(`lw_recognition_result`, `lw_classification_result`,
`lw_detection_result`, and `lw_ocr_result`) use a local full result and copy
only the caller's prefix on every success, capacity error, and pipeline error
after result fields become available. Bytes beyond the declared prefix are not
written. Experimental model/session options remain exact-size APIs.

- Public option/info/result structures begin with struct_size and must be initialized with
  their matching _init function. Array element records lw_detection_box and lw_ocr_line intentionally omit struct_size; their capacities and layout are governed by the surrounding result structures.
- Existing structure prefixes and enum numeric values are permanent after the
  freeze. Additive fields can only be appended under the documented
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

The frozen contract is maintained only while all of the following remain
green:

1. structure-size and enum-value checks in `tests/test_abi.c`;
2. prefix-compatible output metadata checks in `tests/test_abi_prefix.c`;
3. exact legacy export checks in `abi/exports-v0.txt`;
4. frozen stable-symbol checks in `abi/exports-v1-candidate.txt`;
5. recognition-only buffer, error, and lifecycle tests;
   The staged package also configures and runs the abi_v1_client.c example against the installed CMake package and shared library;
6. client tests against the staged Windows and Linux packages;
7. Tiny, Small, and Medium model-pack compatibility tests.

Removing or changing an existing frozen symbol, structure prefix, enum value,
ownership rule, or error semantic requires a new ABI major version.
