# Third-party notices

The pure-C OCR libraries have no large third-party runtime dependency. The
repository also contains an optional HTTP Demo, development tools, and model
inputs with their own licenses.

## Optional HTTP Demo

The cross-platform native HTTP OCR Demo embeds `cpp-httplib` 0.48.0 under the
MIT license. Its vendored header is the Windows 7-compatible copy from the
referenced `lw.OpenCVDNN.PPOCR` repository. See
`licenses/cpp-httplib-MIT.txt`. This dependency belongs to the optional C++
HTTP executable; it is not linked into the pure-C OCR libraries.

## PP-OCRv6 tiny model assets

The ONNX model files, dictionary, and sample image under
`models/ppocrv6-tiny/` are derived from the PP-OCR/PaddleOCR ecosystem and are
included for converter development and reproducible analysis under the Apache
License 2.0. See `licenses/PaddleOCR-models-APACHE-2.0.txt`.

## Converter-only Python dependencies

- ONNX — Apache-2.0
- ONNX Runtime — MIT
- NumPy — BSD-3-Clause
- Pillow — HPND
- protobuf — BSD-3-Clause
- ml_dtypes — Apache-2.0
- typing_extensions — PSF-2.0

These packages are development/converter/test dependencies. They are not linked
into the pure-C deployment runtime.

## WASM build and browser-test dependencies

- Emscripten 4.0.15 — MIT and University of Illinois/NCSA Open Source License;
- Playwright for Python 1.62.0 — Apache-2.0;
- Chromium downloaded by Playwright — CI smoke-test browser under its upstream
  licenses and third-party notices.

Emscripten is the build toolchain for the standalone HTML artifact. Playwright
and Chromium run only in CI to open that artifact, execute real OCR repeatedly,
and check that WASM memory remains stable. They are not embedded in the HTML or
linked into the native runtime.

The default standalone HTML also embeds the legacy browser build of Mozilla
PDF.js 6.3.289 under the Apache License 2.0. PDF.js converts PDF pages to Canvas
pixels in the browser; it is not part of `lw-ppocr.js`, the native runtime, or
the public C ABI. The vendored distribution and license are under
`web/vendor/pdfjs/`. Configure with `LW_WEB_PDF=OFF` to omit it.

## Windows runtime files

Windows binary archives include the Microsoft Visual C++ Runtime and Universal
C Runtime app-local files selected by CMake from the installed Visual Studio
toolchain. Their redistribution and use are governed by the applicable
Microsoft Visual Studio license terms. They are packaging dependencies, not
source dependencies of the pure-C OCR runtime.
