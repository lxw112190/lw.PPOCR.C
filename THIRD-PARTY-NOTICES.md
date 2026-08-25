# Third-party notices

The deployment runtime is intended to have no large third-party runtime
dependency. The current repository nevertheless contains development tools and
model inputs with their own licenses.

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

## Windows runtime files

Windows binary archives include the Microsoft Visual C++ Runtime and Universal
C Runtime app-local files selected by CMake from the installed Visual Studio
toolchain. Their redistribution and use are governed by the applicable
Microsoft Visual Studio license terms. They are packaging dependencies, not
source dependencies of the pure-C OCR runtime.
