# C# WinForms and native HTTP/web Demos

The two Demos deliberately use different hosts:

```text
C# WinForms (.NET Framework 3.5, Windows only)
  -> System.Drawing decodes JPEG/PNG/BMP/GIF/TIFF
  -> shared P/Invoke wrapper
  -> lw_ocr_run_bgr_u8 C ABI

Browser
  -> Canvas decodes the selected image and emits binary P6 PPM
  -> C++ cpp-httplib HTTP server (Windows/Linux/macOS)
  -> PPM RGB-to-BGR conversion
  -> lw_ocr_run_bgr_u8 C ABI
```

The pure-C runtime still has no encoded-image, HTTP, .NET, OpenCV, or JSON
runtime dependency. The native HTTP executable statically embeds only the
vendored `cpp-httplib` header and links the project's static OCR library.

## Build

The native HTTP Demo is enabled by default:

```powershell
cmake -S . -B build -G Ninja -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build -R http_demo_smoke --output-on-failure
```

On Windows, enable the optional .NET Framework 3.5 WinForms Demo with:

```powershell
cmake -S . -B build -G Ninja `
  -DBUILD_TESTING=ON `
  -DLW_BUILD_CSHARP_DEMOS=ON
cmake --build build
```

For x86, use an x86 compiler environment and a separate build directory. Do
not mix x64 and x86 managed executables, native DLLs, or models. The first C#
build may restore the pinned
`Microsoft.NETFramework.ReferenceAssemblies.net35` build-only package.

## C# WinForms Demo

Run:

```powershell
.\build\managed\winforms\lw.PPOCR.C.WinForms.exe `
  .\build\managed\models
```

The application supports common `System.Drawing` image formats, file drag and
drop, a selectable model directory, optional classification, background OCR,
original-image quadrilateral drawing, per-line text/scores/rotation, and
complete JSON output. After a successful run, the result tab can copy plain
recognized text or save UTF-8 TXT and versioned JSON using the same
[OCR export schema](ocr-export-schema.md) as the standalone browser page. A
new image or OCR run invalidates the previous export snapshot until recognition
succeeds. The status bar reports the process architecture and the actual loaded
`lw_ppocr_c.dll` path, which helps diagnose x86/x64 or stale-DLL problems.

For line-worker A/B tests, select `1`, `2`, or `4` workers, set warm-up and
measured iteration counts, and click **性能测试**. The image is decoded once;
only native DET/CLS/REC calls are timed. The **性能记录** tab retains worker
count, CLS state, line count, mean, P95, minimum, maximum, and decode time so
multiple configurations can be compared in one session. Every repeated result
must have identical text and coordinates or the measurement is rejected.

The P/Invoke wrapper checks native structure sizes, explicitly marshals UTF-8
model paths, validates all returned text ranges, and serializes access to the
native handle. The six-argument `NativeOcr` constructor additionally accepts
`workerCount`; zero keeps the platform default. `WorkerCount` reports the
effective native value, while the internal decoded-image path lets the Demo
benchmark native OCR without repeatedly timing `System.Drawing` decoding.

## Native HTTP and browser Demo

Run on Windows:

```powershell
.\build\bin\lw.PPOCR.C.HttpServer.exe `
  --host 127.0.0.1 --port 8787 `
  --models .\build\models --www .\build\www `
  --ocr-workers 4
```

On Linux/macOS, use the same arguments with
`./build/bin/lw.PPOCR.C.HttpServer`. Open
`http://127.0.0.1:8787/`, select a common browser-supported image, and the
page will show recognized text and scaled quadrilaterals.

The browser performs encoded-image decoding because the core accepts decoded
pixels by contract. It converts Canvas RGBA to P6 PPM RGB; the server validates
the header, dimensions, exact payload size, and 40,000,000-pixel limit before
converting to BGR.

### API

Health:

```http
GET /health
```

Efficient binary OCR:

```http
POST /api/ocr
Content-Type: image/x-portable-pixmap

<P6 PPM bytes>
```

JSON/Base64 OCR, compatible in shape with the referenced HTTP Demo:

```http
POST /api/ocr
Content-Type: application/json

{"imageBase64":"<Base64 encoded P6 PPM>"}
```

The alias `image_base64` is also accepted. A successful response uses one
canonical flattened coordinate form:

```json
{
  "ok": true,
  "api_version": 1,
  "request_id": "...",
  "image_width": 640,
  "image_height": 480,
  "detected_count": 1,
  "result": [
    {
      "text": "example",
      "score": 0.98,
      "det_score": 0.91,
      "cls_label": 0,
      "cls_score": 0.99,
      "rotation": 0,
      "x1": 10.0, "y1": 20.0,
      "x2": 100.0, "y2": 20.0,
      "x3": 100.0, "y3": 50.0,
      "x4": 10.0, "y4": 50.0
    }
  ],
  "timing_ms": {"server_total": 31.5}
}
```

Errors contain the same API version/request ID plus stable `error_code` and
`error` fields. x64 requests are limited to 50 MiB and x86 requests to 10
MiB. OCR access is serialized because one native OCR handle is reused, while
cpp-httplib can process unrelated HTTP work on its worker pool. Inside one OCR
request, `--ocr-workers 1..16` controls independent CLS/REC line workers; x64
defaults to 4 and x86 defaults to 1.

The Demo defaults to loopback and provides no TLS, authentication, CORS, rate
limit, or production access log. Do not bind it to an untrusted network.
Production exposure requires an authenticated HTTPS reverse proxy and normal
operational controls.

## Compatibility status

The vendored `cpp-httplib` file is the Windows 7-compatible copy from the
referenced `lw.OpenCVDNN.PPOCR` repository, and the WinForms implementation
uses its .NET 3.5/Cdecl/fixed-layout interop pattern. The new
`lw.PPOCR.C` binaries are locally tested on Windows 10 x64/x86. They remain
compatibility targets—not new physical Windows 7 verification claims—until
these exact packages are retested on that machine.
