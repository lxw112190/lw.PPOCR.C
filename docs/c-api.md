# Experimental C API

The public API in `include/lw_infer.h` is available for integration experiments
but is not ABI-frozen before 1.0. It covers low-level model/session planning and
complete recognize-only and direction-classification APIs for decoded BGR8
pixels. The recognizer hides preprocessing, graph execution, dictionary
indexing, and UTF-8 CTC decoding. The classifier hides resize/pad/normalize,
graph execution, and the two-class 0/180-degree decision.

## Ownership and thread model

- `lw_model_load` creates a read-only model handle; `lw_model_free` destroys it.
- `lw_session_create` creates an independent resolved tensor table and workspace.
- A model must outlive every session created from it.
- Separate sessions do not share mutable workspace and may later run in parallel.
- A single session is not promised to be safe for concurrent calls.
- `lw_recognizer_create` owns its model, dictionary, session, and reusable
  input/output buffers until `lw_recognizer_free`.
- Source BGR pixels and output text remain caller-owned.
- Recognition performs no heap allocation after successful creation.
- A single recognizer must not be called concurrently. Separate recognizers
  own independent mutable state and may run in parallel.
- `lw_classifier_create` owns its model, session, and reusable input/output
  buffers until `lw_classifier_free`; classification performs no heap
  allocation after successful creation.
- A single classifier must not be called concurrently. Separate classifiers
  own independent mutable state and may run in parallel.
- Every successful create has one matching free; all free functions accept null.

Paths passed to the library are UTF-8. The Windows implementation converts to
UTF-16 before opening a file.

## REC session planning

The exact v0.1 REC model has one FP32 input. The descriptor must be:

```text
[N, 3, 48, W]
```

`N` and `W` are positive runtime values. A width that collapses below a valid
spatial output is rejected with `LW_STATUS_INVALID_SHAPE`.

Session creation performs all of the following before returning:

1. validates descriptor size, dtype, rank, dimensions, and unused fields;
2. propagates concrete shapes through all 161 nodes;
3. checks every resolved tensor byte multiplication and `max_tensor_size`;
4. computes tensor birth and last-use nodes;
5. assigns reusable 64-byte-aligned workspace ranges with first-fit allocation;
6. checks `max_workspace_size` and allocates one aligned workspace block.

Default limits are 256 MiB for one tensor and 512 MiB for a session workspace.
Passing zero for either initialized option keeps its default.

## Minimal planning example

```c
lw_model* model = NULL;
lw_session* session = NULL;
lw_error error;
lw_tensor_desc input;
lw_tensor_desc output;
lw_session_info info;

lw_error_init(&error);
if (lw_model_load("rec.lwm", NULL, &model, &error) != LW_STATUS_OK) {
    /* error.code and error.message are available here */
}

lw_tensor_desc_init(&input);
input.dtype = LW_DTYPE_F32;
input.rank = 4;
input.dimensions[0] = 1;
input.dimensions[1] = 3;
input.dimensions[2] = 48;
input.dimensions[3] = 320;

if (lw_session_create(model, &input, 1, NULL, &session, &error) == LW_STATUS_OK) {
    lw_session_info_init(&info);
    lw_tensor_desc_init(&output);
    lw_session_get_info(session, &info);
    lw_session_get_output_desc(session, 0, &output);
    /* output is [1, 40, 6906] for width 320 */
}

lw_session_free(session);
lw_model_free(model);
```

The low-level session planner still accepts no input data pointer. Applications
should use the recognizer API for completed REC inference; the internal executor
is not an integration contract and may change without ABI notice.

## Public REC recognizer

`lw_recognizer_create` accepts UTF-8 paths to one compatible REC `.lwm` model
and dictionary. Its options have these defaults:

| Field | Default | Meaning |
|---|---:|---|
| `target_width` | 320 | Model input width; input height is fixed at 48 |
| `max_model_file_size` | 1 GiB | Model loader limit |
| `max_workspace_size` | 512 MiB | Planned session workspace limit |
| `max_tensor_size` | 256 MiB | Per-tensor limit |
| `max_image_pixels` | 40,000,000 | Decoded source pixel limit |

Zero option values retain their defaults. Reserved fields must remain zero.
Creation rejects a dictionary whose class count does not match the model.

`lw_recognizer_recognize_bgr_u8` accepts interleaved BGR unsigned-byte pixels,
the accessible source byte count, width, height, and row stride. It does not
accept JPEG/PNG file bytes. The source byte count and stride allow the runtime
to reject truncated or invalid layouts before reading pixels.

Text is NUL-terminated UTF-8 in a caller-owned buffer. Call
`lw_recognizer_get_info` and allocate `max_text_capacity` once for the usual
single-pass path. Alternatively, pass a null text pointer and zero capacity;
recognition still runs and returns the exact `required_text_capacity`, so a
second call is required to obtain text. An undersized buffer returns
`LW_STATUS_OUT_OF_BOUNDS`, never partial text.

`lw_recognition_result` reports the emitted CTC entry count, mean score, actual
resized content width, output time steps, and required text capacity. Initialize
every public structure with its matching `_init` function before use.

```c
lw_recognizer* recognizer = NULL;
lw_recognizer_options options;
lw_recognizer_info info;
lw_recognition_result result;
lw_error error;
char* text;

lw_recognizer_options_init(&options);
lw_error_init(&error);
if (lw_recognizer_create("rec.lwm", "ppocr_keys.txt", &options,
                         &recognizer, &error) != LW_STATUS_OK) {
    /* error.code and error.message */
}

lw_recognizer_info_init(&info);
lw_recognizer_get_info(recognizer, &info);
text = (char*)malloc((size_t)info.max_text_capacity);

lw_recognition_result_init(&result);
lw_error_init(&error);
if (lw_recognizer_recognize_bgr_u8(
        recognizer, bgr, bgr_byte_count, width, height, stride,
        text, info.max_text_capacity, &result, &error) == LW_STATUS_OK) {
    /* text is UTF-8; result.score and result.emitted_count are valid */
}

free(text);
lw_recognizer_free(recognizer);
```

## Public CLS classifier

`lw_classifier_create` accepts a UTF-8 path to the compatible fixed-batch CLS
`.lwm` model. Its input is fixed at `[1,3,80,160]`. The options expose the same
model, workspace, tensor, and decoded-image limits as the recognizer; zero
values retain initialized defaults and reserved fields must remain zero.

`lw_classifier_classify_bgr_u8` accepts interleaved BGR unsigned-byte pixels,
accessible byte count, width, height, and row stride. It preserves aspect ratio,
resizes to height 80, right-pads with zero-valued BGR pixels, normalizes to
`[-1,1]`, and executes the two-class model. It does not decode JPEG/PNG bytes.

`lw_classification_result.label` is `0` for upright and `1` for 180 degrees;
`orientation_degrees` exposes the same decision as `0` or `180`. `score` is the
selected Softmax probability and `resized_width` reports the non-padding width.
The classifier reports orientation only—it does not rotate caller-owned pixels.

```c
lw_classifier* classifier = NULL;
lw_classifier_options options;
lw_classification_result result;
lw_error error;

lw_classifier_options_init(&options);
lw_error_init(&error);
if (lw_classifier_create("cls.lwm", &options, &classifier, &error) == LW_STATUS_OK) {
    lw_classification_result_init(&result);
    if (lw_classifier_classify_bgr_u8(
            classifier, bgr, bgr_byte_count, width, height, stride,
            &result, &error) == LW_STATUS_OK) {
        /* result.orientation_degrees is 0 or 180; result.score is valid. */
    }
}
lw_classifier_free(classifier);
```

## Errors

Programs should branch on `lw_status`, not parse diagnostic text. Relevant
planning errors are:

- `LW_STATUS_INVALID_ARGUMENT`: null pointer, wrong count, or bad options ABI;
- `LW_STATUS_INVALID_SHAPE`: incompatible dtype/rank/dimension/operator shape;
- `LW_STATUS_MEMORY_LIMIT`: tensor/workspace limit or size overflow;
- `LW_STATUS_OUT_OF_MEMORY`: host allocation failure;
- model load errors defined by the LWM loader.

Recognition and classification additionally return `LW_STATUS_INVALID_SHAPE` for a truncated BGR
layout, `LW_STATUS_OUT_OF_BOUNDS` for an insufficient text buffer, and
`LW_STATUS_MEMORY_LIMIT` when decoded dimensions exceed `max_image_pixels`.
