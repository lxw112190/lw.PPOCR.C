# Tiny Legacy Web distribution

lw.PPOCR.C provides a separate compatibility build for older Chromium and
Android WebView environments. It is intentionally independent from the normal
Modern Web build.

| Distribution | Model | WASM | PDF | Baseline |
| --- | --- | --- | --- | --- |
| ocr-demo.html | PP-OCRv6 Tiny/optional variants | SIMD128 by default | Yes | Modern browsers |
| ocr-demo-legacy.html | PP-OCRv6 Tiny | Scalar WASM | No | Chrome/Chromium WebView 70+ |

Legacy is a compatibility build, not a performance build. It does not change
the C ABI, model format, Modern Web artifact, Node package, Small/Medium
variants, or PDF support.

## Build locally or in CI

The build requires the same pinned Emscripten release as the normal Web build:

~~~bash
emcmake cmake -S . -B build-wasm-legacy -G Ninja   -DCMAKE_BUILD_TYPE=Release   -DLW_BUILD_HTTP_DEMO=OFF   -DLW_BUILD_CSHARP_DEMOS=OFF   -DBUILD_TESTING=OFF   -DLW_WEB_LEGACY=ON   -DLW_WASM_SIMD128=OFF   -DLW_WEB_PDF=OFF   -DLW_BUILD_WEB_MODEL_VARIANTS=OFF

cmake --build build-wasm-legacy --target lw-ocr-js lw-ocr-html --parallel
~~~

The output names are deliberately distinct:

~~~text
build-wasm-legacy/lw-ppocr-legacy.js
build-wasm-legacy/ocr-demo-legacy.html
~~~

LW_WEB_LEGACY=ON fails configuration if SIMD, PDF, or Web model variants are
also enabled. The Legacy link uses MIN_CHROME_VERSION=70 and enables the
web, webview, and worker Emscripten environments. LEGACY_VM_SUPPORT is not
used because it disables WebAssembly.

## Runtime metadata and diagnostics

The SDK exposes:

~~~javascript
LwPpocr.buildInfo
// {
//   flavor: "legacy",
//   wasmSimd128: false,
//   pdf: false,
//   minChromeVersion: 70
// }
~~~

The standalone page labels itself as the compatibility build. If JavaScript
or WASM initialization fails, the page displays a copyable diagnostics panel
containing the build flavor, user agent, capabilities, bootstrap state, and
last error. It never includes image bytes, OCR text, or local file paths.

Legacy supports image OCR only. PDF controls are not included in this
distribution. CI first verifies the paired Modern Tiny HTML against the exact
Tiny golden checksum, then compares the scalar Legacy result with that Modern
result on the same runner and browser. The Legacy result must be deterministic
across two runs, retain all 16 lines and the expected first line, keep at least
75% exact lines, and remain below 5% character error rate. This allows bounded
floating-point backend differences without reducing the test to a smoke check.

The CI workflow builds and tests this artifact in an independent legacy job
and uploads it as lw.PPOCR.C-browser-legacy-<commit>. It is upload-only for
now; Release workflow publication waits for physical validation on the target
realme Q2/older WebView device.
