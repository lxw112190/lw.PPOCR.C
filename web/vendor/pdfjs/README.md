# Vendored PDF.js browser build

This directory contains the legacy browser build from Mozilla PDF.js
`pdfjs-dist` 6.3.289. The standalone OCR HTML embeds `pdf.min.mjs` and
`pdf.worker.min.mjs` as Base64 and creates Blob URLs only when a PDF is first
opened. The reusable `lw-ppocr.js` SDK does not contain PDF.js.

`LW_WEB_PDF=OFF` excludes these files from the generated HTML. Runtime PDF
loading disables PDF.js WebAssembly helpers so the self-contained page never
requests optional `.wasm`, CMap, or standard-font resources from the network.

The source archive and file hashes are recorded in `VERSION`. PDF.js is
licensed under Apache-2.0; see `LICENSE` and the repository's
`THIRD-PARTY-NOTICES.md`.
