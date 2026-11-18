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
session transactionally. DB postprocessing reuses scratch buffers owned by
the detector.

## Contour workspace

The DB scan keeps the leftmost and rightmost contour points of each row in a
component. The segment between those endpoints contains the row's other
points, so the same convex hull can be built from at most two points per row.
Foreground runs already give those endpoints. Hole contours use the original
boundary coordinates and reduce them as they are collected.

The row and point buffers grow with map height. The hull buffer grows with
the number of points passed to the hull, including space for the expanded
quadrilateral. At 960×960, the former interior bitmap, contour points and hull
reserved 23,040,000 bytes together. Their replacements use at most 53,760 bytes
when the largest map seen by the detector is 960×960. Other DB buffers and
inference tensors contribute separately to process memory.

On Windows x64 with a Ryzen 7 7735H and MSVC Release, a standalone DB run
compared the previous and compact workspaces on four 960×960 probability
maps. Each version ran 40 calls per map in each of two runs, with the version
order reversed on the second run.

| Probability map | Previous mean ms | Compact mean ms |
| --- | ---: | ---: |
| Empty | 3.06 | 2.29 |
| Filled | 4.51 | 3.58 |
| Text-like strips | 6.87 | 4.12 |
| Enclosed holes | 8.82 | 5.23 |

An additional 312-map differential check matched detection counts, order,
coordinate bits and score bits between the two builds. It covered variable
sizes, dilation, candidate limits and non-finite input.

The native full-OCR benchmark used the same 100 generated images, official
ONNX models, CLS, DET960 and REC width limit 960 on both builds. Each process
ran one full warm-up pass and one measured pass with the listed line workers.
Output text checksums and line counts matched within all six pairs.

| Model | Workers | Mean OCR ms before → after | P95 ms before → after | Peak working set MiB before → after |
| --- | ---: | ---: | ---: | ---: |
| Tiny | 1 | 112.20 → 114.05 | 144.85 → 149.39 | 116.5 → 115.8 |
| Tiny | 4 | 83.12 → 81.28 | 110.43 → 110.01 | 135.2 → 132.8 |
| Small | 1 | 344.91 → 348.67 | 468.70 → 457.90 | 227.3 → 224.5 |
| Small | 4 | 268.12 → 267.31 | 347.95 → 359.39 | 262.9 → 261.8 |
| Medium | 1 | 1175.34 → 1174.30 | 1528.47 → 1543.59 | 739.0 → 738.6 |
| Medium | 4 | 1077.71 → 1085.50 | 1416.81 → 1427.53 | 784.7 → 783.9 |

These single-pair runs establish output parity and show mixed changes in
end-to-end latency. The observed reduction in process peak working set was
0.4–2.8 MiB. The allocation capacity above accounts for the contour
workspace alone, and the Windows working set counts resident pages.

## Correctness gates

- deterministic preprocessing is compared element-by-element with an
  independent NumPy reference, including padded source stride;
- the isolated DET graph output is compared with ONNX Runtime at dynamic
  shapes;
- synthetic postprocessing checks score, geometry, exact-capacity query,
  non-partial insufficient-capacity behavior, enclosed holes, dilation and
  workspace reuse after a smaller or invalid probability map;
- the public API runs the real bundled image and model, validates all scores
  and original-image coordinates, and exercises dynamic session reuse;
- the installed package runs `lw-detect-ppm` against the installed model and
  full-image PPM fixture.
