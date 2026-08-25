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
- NumPy — BSD-3-Clause
- protobuf — BSD-3-Clause
- ml_dtypes — Apache-2.0
- typing_extensions — PSF-2.0

These packages are development/converter dependencies. They are not permitted
in the future pure-C deployment runtime.
