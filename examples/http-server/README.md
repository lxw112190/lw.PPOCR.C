# Native HTTP OCR Demo

This cross-platform C++ Demo uses `cpp-httplib` and the public `lw_ocr_*` C
ABI. It has no .NET, OpenCV, or general-purpose JSON dependency.

The pure-C runtime accepts decoded BGR8 pixels rather than encoded image files.
The browser page therefore decodes common image formats with Canvas and sends a
binary P6 PPM request. Native clients may send P6 PPM directly or place its
Base64 representation in the JSON `imageBase64` field.

```http
POST /api/ocr
Content-Type: image/x-portable-pixmap

<P6 PPM bytes>
```

```http
POST /api/ocr
Content-Type: application/json

{"imageBase64":"<Base64 encoded P6 PPM>"}
```

Run `lw.PPOCR.C.HttpServer --help` for command-line options. It defaults to
`127.0.0.1:8787` and resolves `models/` and `www/` next to the packaged `bin/`
directory. `--ocr-workers 1..16` controls independent CLS/REC line workers;
native x64 uses the online logical-processor count capped at 8 and x86 uses 1.
The server still serializes requests through one OCR handle, while lines inside
that request can run in parallel.
