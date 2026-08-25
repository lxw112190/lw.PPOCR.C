# Public DET pipeline

The experimental detector API accepts decoded interleaved BGR8 pixels and
returns caller-owned clockwise quadrilaterals in original-image coordinates.
It covers aspect-preserving resize to multiples of 32, PP-OCR normalization,
dynamic DET session execution, thresholding, optional 2x2 dilation, component
extraction, convex-hull minimum rectangles, DB-style expansion, score
filtering, coordinate restoration, and reading-order sorting.

The default options are a 960-pixel long-side limit, bitmap threshold 0.3, box
threshold 0.6, unclip ratio 1.6, no dilation, and at most 1,000 candidates.
Source images are limited to 40,000,000 decoded pixels by default. Model,
workspace, and tensor limits use the same bounded runtime defaults as REC and
CLS.

This pure-C postprocessor intentionally does not claim bit-for-bit identity
with OpenCV `findContours` or Clipper. It uses 8-connected components, convex
hulls, minimum-area rectangles, rectangle scores, and rectangle-offset
expansion. The public contract is the returned quadrilateral and score, not an
OpenCV contour implementation detail.

## Capacity contract

For one-pass use, allocate `lw_detector_info.max_candidates` boxes and call
`lw_detector_detect_bgr_u8` once. For exact allocation, pass a null box pointer
and zero capacity; the call performs detection and returns
`required_box_capacity`. A second call obtains the boxes. An undersized buffer
returns `LW_STATUS_OUT_OF_BOUNDS`, reports the required capacity, and never
writes a partial result.

One detector owns its model, dynamic session, and reusable inference tensors.
It is not safe for concurrent calls. A size change replaces its planned
session transactionally. DB postprocessing uses bounded transient scratch
memory for the current probability map.

## Correctness gates

- deterministic preprocessing is compared element-by-element with an
  independent NumPy reference, including padded source stride;
- the isolated DET graph output is compared with ONNX Runtime at dynamic
  shapes;
- synthetic postprocessing checks score, geometry, exact-capacity query, and
  non-partial insufficient-capacity behavior;
- the public API runs the real bundled image and model, validates all scores
  and original-image coordinates, and exercises dynamic session reuse;
- the installed package runs `lw-detect-ppm` against the installed model and
  full-image PPM fixture.
