# PDF frontend boundary

`lw_pdf_adapter.js` is the standalone Demo's PDF document frontend. It lazily
loads the PDF.js core and Worker embedded by `package_ocr_html.py`, opens a local
`File` or `Blob`, and renders one page at a time to a Canvas.

It deliberately does not import or call `LwPpocr`, define OCR result schemas,
extract PDF text layers, or retain page bitmaps. `ocr-demo-ui.js` passes the
returned Canvas to the unchanged image-only browser SDK and calls `release()`
after each page.

The public adapter surface is `LwPdf` API version 1:

```javascript
const documentHandle = await LwPdf.open(file);
const page = await documentHandle.renderPage(1, {
  dpi: 180,
  maxPixels: 5_000_000
});

await engine.recognize(page.canvas);
page.release();
await documentHandle.close();
```

The adapter is injected only into `ocr-demo.html`. It is not part of
`lw-ppocr.js`, the native runtime, or the C ABI.
