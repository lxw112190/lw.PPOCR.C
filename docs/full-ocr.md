# Full OCR pipeline

The public full-OCR path composes the exact PP-OCRv6 tiny models while keeping
the deployment runtime C11-only and dependency-free:

```text
decoded BGR8 image
  -> DET resize/normalize + graph + DB postprocess
  -> reading-order quadrilaterals
  -> pure-C perspective crop (tall crop: rotate 90 degrees clockwise)
  -> optional CLS (confident odd label: rotate 180 degrees)
  -> REC resize/normalize + graph + UTF-8 CTC decode
  -> caller-owned line records and shared text buffer
```

JPEG, PNG, camera, PDF, and other encoded inputs remain application concerns.
The core accepts decoded interleaved BGR8 bytes with explicit accessible byte
count, dimensions, and row stride.

## Geometry contract

Each detected quadrilateral is clockwise and begins near its top-left point.
The crop width is the rounded top-edge length and the crop height is the rounded
left-edge length. A destination-to-source homography and bilinear sampling with
replicated image borders produce the crop. If unrotated height is at least 1.5
times its width, the crop is rotated 90 degrees clockwise before CLS/REC.

The runtime implementation does not use OpenCV. Tests compare horizontal and
tall crops with an independent OpenCV `warpPerspective`/`rotate` oracle, allowing
a maximum per-channel byte difference of 6 and mean difference below 0.8 for
the deterministic fixture. This tolerance covers small rounding and interpolation
differences without weakening the full OCR text Golden.

## Result and allocation contract

`lw_ocr_line` embeds exactly one `lw_detection_box`; no duplicate flattened and
array coordinate forms are returned. It also reports REC/CLS scores, classifier
label, applied rotation, emitted CTC entry count, and the offset/length of its
UTF-8 text. Strings share a caller-owned byte buffer and are individually
NUL-terminated; `text_length` excludes the NUL.

Use `lw_ocr_get_info` to allocate maximum line/text capacities for one inference
pass. Passing null outputs and zero capacities instead performs an exact query,
but still runs DET/CLS/REC and therefore requires a second pass to obtain data.
Supplying only one null output, a nonzero capacity for a null pointer, or an
undersized buffer is rejected. Capacity failure never exposes partial output.

`detected_count` can exceed `line_count` only when a degenerate box cannot form
a valid crop. Valid output preserves detector reading order.

## Limits and memory

The full-OCR handle owns separate component sessions and reusable maximum-sized
line/text scratch. After DET, independent CLS/REC worker pairs process batches
of text-line crops in parallel. The crop buffer holds at most one crop per
active worker, grows to the largest batch seen by that handle, and is then
reused. The default maximum is 16,000,000 pixels per crop; exceeding it returns
`LW_STATUS_MEMORY_LIMIT`. DET and each nested model retain their own image,
tensor, workspace, and model-size limits.

Native 64-bit builds default to four workers; x86 and WebAssembly default to
one. `lw_ocr_options.worker_count` accepts `1..16`. The same budget is used by
the detector's fixed thread pool for sufficiently large output-channel-parallel
convolutions; DET completes before the independent CLS/REC line workers start,
so these phases do not oversubscribe each other. More workers trade model,
workspace, crop, and thread resources for lower latency, so applications should
benchmark `1`, `2`, and `4` on their target CPU and memory budget.

The detector's DB postprocessor may allocate bounded transient scratch. The
full-OCR path therefore promises bounded resources and buffer reuse, not zero
allocations per call. Internal line workers do not make the public handle
reentrant: concurrent requests still require separate OCR handles.

## Correctness gates

`full_ocr_pipeline_reference` covers:

- perspective crops against the OpenCV test oracle;
- the exact 16-line UTF-8 result and reading order on the bundled full image;
- stage scores, coordinate bounds, classifier metadata, and 180-degree rotation;
- exact capacity query and all-or-nothing capacity failure;
- direction classification disabled with a null CLS path;
- nested option validation and crop-pixel resource rejection.

`full_ocr_golden_corpus` separately runs seven versioned full-pipeline cases
covering scale, non-square input, a 90-degree image rotation, classifier on/off,
and a valid zero-line blank image. It pins the source/model/dictionary hashes,
freezes exact text and ordering, and checks score thresholds, classifier
metadata, coordinate bounds, polygon area, and tolerant original-image boxes.
See [`full-ocr-golden-corpus.md`](full-ocr-golden-corpus.md).

`c_demo_full_ocr` executes the public PPM Demo. `staged_package` installs the
project into a clean directory, builds the standalone installed consumer, and
runs the installed binary with installed models and data. These tests prove the
current exact model/sample contract, not broad OCR accuracy across arbitrary
documents or languages. The derived full-OCR corpus still has only one source
image and does not change that limitation.
