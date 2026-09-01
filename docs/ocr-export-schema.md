# OCR result export schema

The standalone browser OCR page and .NET Framework WinForms Demo can copy
recognized text and save UTF-8 TXT or JSON files. TXT contains one recognized
line per line, in the same reading order shown by the application. JSON uses
the versioned application-level contract below; it does not change or extend
the public C ABI.

## Version 1

```json
{
  "schema_version": 1,
  "source": "article.png",
  "image": {"width": 1920, "height": 1080},
  "options": {"use_cls": true, "reading_order": "horizontal-ltr"},
  "elapsed_ms": 123.456,
  "lines": [
    {
      "index": 0,
      "text": "recognized text",
      "box": [10.0, 20.0, 100.0, 20.0, 100.0, 50.0, 10.0, 50.0],
      "det_score": 0.97,
      "rec_score": 0.99,
      "cls_score": 0.98,
      "cls_label": 0,
      "rotation_degrees": 0
    }
  ]
}
```

- `source` is the selected file name only; it never contains a local path.
- `image` is the decoded original image size. The browser page may downscale a
  large image before inference, but exported coordinates are restored to this
  size.
- `elapsed_ms` is application wall-clock time for producing the displayed
  result. It includes image decoding, OCR, and result materialization/rendering,
  but excludes model initialization and the later clipboard/file operation.
- `lines` preserves detector reading order. `index` is zero-based and `text` is
  UTF-8 when serialized to the downloaded file.
- `box` is the only coordinate representation. It contains
  `[x1, y1, x2, y2, x3, y3, x4, y4]` in original-image pixels, clockwise and
  beginning near the top-left point. The coordinate origin is the image's
  top-left corner.
- `det_score` and `rec_score` are the detector and recognizer confidence scores.
- `cls_score`, `cls_label`, and `rotation_degrees` are present only when
  `options.use_cls` is `true`.
- A successful run with no recognized text has an empty `lines` array.

The browser writes LF line endings. WinForms uses Windows line endings. Both
TXT and JSON downloads are UTF-8 without a byte-order mark.

Version 1 may gain optional fields, but existing field names, types, and
semantics must remain compatible. A breaking change requires a new
`schema_version`.

## Version 2: PDF documents

The standalone HTML uses version 2 for PDF recognition. Version 1 remains the
image contract; a PDF result is not forced into the single-image shape.

```json
{
  "schema_version": 2,
  "source_type": "pdf",
  "source": "document.pdf",
  "document": {"page_count": 2, "processed_pages": 2},
  "options": {
    "use_cls": false,
    "reading_order": "horizontal-ltr",
    "pdf_dpi": 180,
    "pdf_max_pixels": 5000000
  },
  "timing": {
    "pdf_load_ms": 18.25,
    "render_ms": 42.75,
    "inference_ms": 310.5,
    "ui_ms": 3.1,
    "total_ms": 374.6
  },
  "pages": [
    {
      "page_number": 1,
      "pdf": {"width_pt": 595.276, "height_pt": 841.89, "rotation": 0},
      "image": {"width": 1488, "height": 2105},
      "timing": {
        "render_ms": 20.5,
        "inference_ms": 150.25,
        "total_ms": 170.75
      },
      "lines": [
        {
          "index": 0,
          "text": "recognized text",
          "box": [25, 40, 250, 40, 250, 85, 25, 85],
          "pdf_box": [10, 825.89, 100, 825.89, 100, 807.89, 10, 807.89],
          "det_score": 0.97,
          "rec_score": 0.99
        }
      ]
    }
  ]
}
```

- `source_type` is always `pdf`; `source` is the selected file name without a
  local path.
- `document.page_count` is the PDF page count. `processed_pages` is the number
  of completed page objects in `pages`. It can be smaller after the user stops
  an all-page run. Pages are stored in processing order and `page_number` is
  one-based.
- `pdf_dpi` is the requested rasterization density. `pdf_max_pixels` is the
  per-page safety cap; the renderer may reduce the effective scale to stay
  within it. `image` records the actual Canvas pixel size used for OCR.
- `pdf.width_pt` and `pdf.height_pt` are page dimensions in PDF user-space
  points (72 points per inch). `rotation` is the page rotation applied by the
  PDF viewport.
- `box` has the version 1 point order, but its coordinates are pixels in the
  page's exported `image`, with a top-left origin.
- `pdf_box` contains the same four points converted through the PDF.js viewport
  into PDF user space. Its origin and axes follow the PDF page coordinate
  system (commonly bottom-left); page rotation is already accounted for.
- Page `render_ms` measures PDF rasterization. Page `inference_ms` is the SDK
  total for that page, including pixel preparation and native OCR. Page
  `total_ms` is their sum.
- Document `pdf_load_ms`, `render_ms`, `inference_ms`, and `ui_ms` expose useful
  phases. Document `total_ms` is application wall-clock time including PDF
  opening and the selected-page run; it can be greater than the phase sum due
  to scheduling and browser rendering.
- Optional CLS fields on each line follow version 1 and appear only when
  `options.use_cls` is true.

PDF TXT export uses LF endings, keeps line reading order within each page, and
adds `===== Page N / M =====` headings. JSON is UTF-8 without a byte-order mark.

Version 2 may gain optional fields, but its existing field names, types, and
semantics must remain compatible. A future breaking PDF contract requires a
new `schema_version`.
