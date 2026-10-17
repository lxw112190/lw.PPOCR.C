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

For line-worker A/B tests, select `1`, `2`, `4`, or `8` workers, set warm-up and
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

### Large-image pixel policy / 大图像素上限

The existing five/six-argument constructors retain the 40,000,000-pixel
default. The new seven-argument overload sets the decoded-image and native
DET source limit together; this is an instance policy, not a process-global
setting. Both encoded input and the internal predecoded benchmark path enforce
it. No C ABI, model, preprocessing or export-schema change is required.

```csharp
using (var ocr = new NativeOcr(detPath, clsPath, recPath, dictionaryPath,
    true, 4u, 100000000UL)) // maxImagePixels: 100 million, not bytes
{
    OcrResponse result = ocr.RecognizeFile(imagePath);
}
```

`MaxImagePixels` reports the selected value. Zero is invalid (not unlimited).
This managed wrapper accepts 1..200,000,000 pixels on x64, and at most the
original 40,000,000 on x86. These are policy bounds, not a guarantee that the
machine or GDI+ can decode every image below the cap. Width, height, total
pixels and the selected limit appear in dimension failures. The separate
native `max_crop_pixels=16,000,000` is unchanged: a huge detected text crop
may still fail even if the original image is allowed.

WinForms exposes **原图上限(百万像素)**, default **40**. For an ordinary
image above 40 MP, set **100** before selecting it or running OCR. The setting
is disabled while busy and changing it causes the next initialization/run to
rebuild the engine. Preview checks the same bound before making Bitmap copies;
an over-limit selection leaves the previous preview intact. No automatic
resize, tiling or change to original coordinates occurs.

The displayed memory estimate is **BGR only**, not total process memory.
100 MP needs about 300 MB for BGR alone, plus encoded data, GDI+ decode,
preview copies, models and OCR workspace. Restrict simultaneous large-image
requests; the native handle lock does not stop different threads decoding
images before inference. The encoded-image decoder may allocate before size
validation, so this is not a hardened decoder for untrusted image bombs.
For dense long screenshots, prefer an application-level tiled pipeline rather
than blindly raising this limit. HTTP, Web/WASM and Android limits are not
changed by this C# option.

中文：这是可配置的内存保护，不是“支持任意大图”的承诺。默认4000万不变，x64界面可按需
提高到1亿等值；单位为百万像素，100代表100,000,000像素。单框1600万限制不变，
不会自动跳过异常框。界面提示仅估算BGR，真实内存还需加上预览、模型和工作区。

Windows x64 CTest runs `csharp_image_limits_contract` with Visual Studio's
Roslyn compiler and the .NET 4.8 targeting pack. It checks the old constructor,
exact/invalid bounds, native DET synchronization, preview rejection/recovery,
busy controls and engine rebuilding, and generates a temporary 41.6 MP image
containing the bundled sample for real full OCR. Generated images are not
committed. The original WinForms sources also compile against .NET 3.5.
This change does not claim compatibility with untested customer codecs or
machines. Generic OpenCV/HTTP-service preflight requirements (OpenCV version,
web assets and service configuration) do not apply to the pure-C WinForms
portable package; its `.exe.config`, model hashes, DLL dependency closure and
actual extracted-package OCR checks are the relevant gates. Cached files in
ignored `build-*` trees are not package inputs or tracked source.

## Native HTTP and browser Demo

Run on Windows:

```powershell
.\build\bin\lw.PPOCR.C.HttpServer.exe `
  --host 127.0.0.1 --port 8787 `
  --models .\build\models --www .\build\www `
  --ocr-workers 4 --rec-max-width 960
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

Native clients may also send uncompressed 24-bit BMP with `Content-Type:
image/bmp`. This lossless common format is used by the
[OpenCV DNN comparison](opencv-dnn-comparison.md).

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
uses the online logical-processor count capped at 8 and x86 uses 1.
`--rec-max-width 192|320|480|640|960` caps adaptive REC width and defaults to
320.

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
