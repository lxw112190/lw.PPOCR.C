# Offline OCR export schema

The standalone browser OCR page can copy recognized text and download UTF-8
TXT or JSON files. TXT contains one recognized line per line, in the same
reading order shown by the page. JSON uses the versioned application-level
contract below; it does not change or extend the public C ABI.

## Version 1

```json
{
  "schema_version": 1,
  "source": "article.png",
  "image": {"width": 1920, "height": 1080},
  "options": {"use_cls": true},
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
- `image` is the decoded original image size. The page may downscale a large
  image before inference, but exported coordinates are restored to this size.
- `elapsed_ms` is browser wall-clock time from starting the run through result
  rendering. It includes image decoding and pixel conversion as well as OCR.
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

Version 1 may gain optional fields, but existing field names, types, and
semantics must remain compatible. A breaking change requires a new
`schema_version`.
