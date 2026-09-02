# Vendored PDF.js browser build

This directory contains the legacy browser build from Mozilla PDF.js
`pdfjs-dist` 6.3.289. The standalone OCR HTML embeds `pdf.min.mjs`,
`pdf.worker.min.mjs`, and the PDF.js image/color WASM helpers as Base64. It
creates Blob URLs only when a PDF is first opened. The reusable `lw-ppocr.js`
SDK does not contain PDF.js.

`LW_WEB_PDF=OFF` excludes these files from the generated HTML. With PDF support
enabled, `BinaryDataFactory` serves `jbig2.wasm`, `openjpeg.wasm`, and
`qcms_bg.wasm` from the HTML itself with `useWorkerFetch=false`; the adapter
does not fetch optional PDF resources from the network. CMaps and standard
fonts are intentionally outside this image-only PDF frontend's scope.

The source archive and file hashes are recorded in `VERSION`. PDF.js is
licensed under Apache-2.0; see `LICENSE`, the per-codec license files
(`LICENSE_JBIG2`, `LICENSE_OPENJPEG`, `LICENSE_QCMS` and their PDF.js wrapper
notices), and the repository's `THIRD-PARTY-NOTICES.md`.
