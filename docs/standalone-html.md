# Standalone HTML usage

**ocr-demo.html** is the ready-made offline OCR application. It embeds the
browser SDK, WebAssembly runtime, DET/CLS/REC models, dictionary, PDF.js,
user interface, and support image in one file. Selected images and PDFs remain
local and are not uploaded.

For a custom browser application, use the separate **lw-ppocr.js** artifact and
the public **LwPpocr.create()** API described in the
[Browser JavaScript SDK guide](web-sdk.md). Do not load an HTML file as a
script.

## Use the page

Download the release HTML or build the **lw-ocr-html** CMake target, then open
**ocr-demo.html** directly in a modern browser.

1. On desktop, select an image/PDF or drag it onto the page. You can also
   press `Ctrl+V`/`⌘V` to paste an image or screenshot from the clipboard.
2. On a phone, choose **拍照识别** for a new image or **选择图片 / PDF** for
   an existing file.
3. Enable CLS when orientation classification is needed.
4. Choose a reading order: **Horizontal LTR** for normal horizontal pages,
   **Vertical RTL** for traditional right-to-left columns, or **Vertical LTR**
   for left-to-right vertical columns. The default is Horizontal LTR.
5. For PDF input, select the current page or all pages and choose 144, 180, or
   220 DPI. Higher DPI can improve small text, but needs more time and memory.
6. Select **开始识别**. All-page OCR runs sequentially and can be stopped; if
   native OCR has started for a page, stopping takes effect after that page.
7. Copy or share the text, or export UTF-8 TXT or structured JSON.

The PDF preview can move between pages without retaining every rendered page.
The Demo keeps only the current page bitmap plus structured results, and caps
an OCR render at 5 megapixels. Selecting another file invalidates the old
export snapshot until recognition succeeds again.
Pasting an image follows the same lifecycle as selecting a file: it replaces
the previous source, clears old results immediately, shows a preview, and waits
for the user to start OCR. Plain-text paste is not intercepted. Only the first
image item is used when the clipboard contains multiple images; PDFs are not
accepted through paste.

Inference normally runs inside a Blob Worker so the controls remain
responsive. If local browser policy rejects Blob Workers, the embedded SDK
uses its compatible main-thread fallback. The default build enables
WebAssembly SIMD128.

PDF.js also prefers its own Blob module Worker. The standalone page constructs
that Worker directly, avoiding PDF.js's nested Blob wrapper on `file://` pages.
If a mobile browser or embedded WebView rejects module Workers, PDF parsing and
rendering automatically use a main-thread compatibility mode. OCR keeps its
already initialized backend. Older browsers without `Blob.arrayBuffer()` use a
`FileReader` fallback. The PDF frontend also supplies
`Promise.withResolvers()` in both the page and PDF Worker when an older
Chromium-based browser does not provide it.

## Online deployment

The PDF-enabled standalone HTML is a large self-contained file because the
WASM runtime, OCR models, PDF.js, and both Workers are embedded. Its small
loading timer starts before those payloads have finished downloading, so a
phone can distinguish network transfer from WASM initialization.

Enable HTTP compression when serving the file. Without `Content-Encoding:
gzip` or Brotli, the browser must transfer the full Base64 HTML before the
scripts at the end of the document can initialize OCR. A minimal nginx setup
is:

~~~nginx
# Put these directives inside the nginx http or server block.
gzip on;
gzip_min_length 1024;
gzip_comp_level 6;
gzip_vary on;
gzip_types application/javascript application/json application/wasm
           text/css text/plain image/svg+xml;

# Match the public URL used by your deployment. No filesystem path is needed
# in this example.
location = /ocr-demo.html {
    add_header Cache-Control "no-cache" always;
}
~~~

nginx compresses `text/html` when gzip is enabled, so it does not need to be
listed in `gzip_types`. Use HTTPS for production deployment. If releases keep
the same URL, retain ETag/Last-Modified validation or use `no-cache` as above
so phones do not keep an old standalone artifact. Validate and reload nginx,
then verify the public response:

~~~bash
nginx -t
nginx -s reload
curl -I -H "Accept-Encoding: gzip" https://example.com/ocr-demo.html
~~~

The response should include `Content-Encoding: gzip`,
`Vary: Accept-Encoding`, and `Cache-Control: no-cache`. Replace the example
URL and location with the deployment's public URL; do not publish private
filesystem paths in shared configuration examples.

## Mobile PDF troubleshooting

Open the HTML with the current system Chrome or Safari rather than a file
manager, messaging-app, or office-app preview WebView. When PDF opening fails,
the page shows a **PDF 打开失败 · 查看诊断信息** panel. Copy that report when
filing an issue; it contains the stable error code, failure phase, PDF worker
backend, browser capabilities, protocol, and user agent, but no PDF contents or
local path.

The most useful error codes are:

- `LW_PDF_INIT_FAILED`: PDF.js or its compatibility backend could not start;
- `LW_PDF_READ_FAILED`: the selected browser could not read the local file;
- `LW_PDF_LOAD_FAILED`: PDF parsing failed;
- `LW_PDF_RENDER_FAILED`: the PDF opened but the current page could not render;
- `LW_PDF_PASSWORD_REQUIRED`: encrypted PDFs are not supported by this UI.

CI opens the final artifact through `file://`, blocks Worker, removes
`Blob.arrayBuffer()` and `Promise.withResolvers()`, verifies the compatibility
fallbacks, and checks a 390-pixel layout. That is not a physical Android/iOS
browser certification; release compatibility claims still require testing the
exact artifact on the target phone and browser.

## Customize the Demo

The source is deliberately split into four layers:

~~~text
web/lw_ppocr_sdk.template.js  reusable OCR SDK
             |
web/pdf/lw_pdf_adapter.js      PDF.js -> page Canvas (standalone only)
             |
web/ocr-demo-ui.js            DOM, preview, export, mobile interaction
             |
web/ocr-demo.template.html    markup and responsive styles
~~~

The build first produces **lw-ppocr.js**, then injects that exact artifact,
PDF.js, its Worker, and the UI script into **ocr-demo.html**. PDF.js and its
Worker are decoded into Blob URLs only when the first PDF is opened. No model,
PDF, page image, or optional PDF.js resource is fetched over the network.

The boundary remains: PDF.js renders a page to Canvas, then `LwPpocr` recognizes
that Canvas. The packaged page includes PDF.js 6.3.289's `jbig2.wasm`,
`openjpeg.wasm`, and `qcms_bg.wasm` helpers through an embedded
`BinaryDataFactory`, so supported CCITT/JBIG2, JPX/JPEG2000, and ICC paths do
not need network resources. CMaps and standard-font files remain outside this
image-only frontend's scope. Integrations should keep image decoding, WASM
pointers, buffer ownership, and native calls inside the SDK. Demo changes should
operate on structured OCR results. The reusable **lw-ppocr.js** remains
image/Canvas-only and does not contain PDF.js.

The page still exposes **window.lwPpocrDemo** version 1 as a compatibility
adapter for existing customizations:

~~~javascript
await window.lwPpocrDemo.ready();
const result = await window.lwPpocrDemo.recognize(file, {useCls: true});
const text = window.lwPpocrDemo.getPlainText();
~~~

It updates the built-in preview, result list, and export state. New
applications should use **LwPpocr.create()** directly because it is independent
of Demo markup and is tested as a standalone SDK.

The Demo also dispatches:

- **lwppocr:ready** with **{backend}**;
- **lwppocr:result** with the successful export object;
- **lwppocr:error** with **{phase, code, message, diagnostics}** for PDF errors
  (non-PDF failures may omit `code` and `diagnostics`).

The test hook, Worker messages, embedded model variables, Emscripten members,
and DOM state are implementation details, not public API.

## Export format

Image exports keep `schema_version: 1`. PDF exports use `schema_version: 2` and
contain document/page metadata, per-page timings, rendered-pixel boxes, and
PDF-coordinate boxes. PDF TXT separates pages with visible headings. See
[OCR result export schema](ocr-export-schema.md).

Encrypted PDFs are reported as unsupported in this first version. Configure
with `-DLW_WEB_PDF=OFF` to build a smaller image-only standalone HTML. This
option changes only the application artifact; **lw-ppocr.js** and the C ABI are
identical.

The browser SDK and the Demo export are application-layer contracts. Neither
changes the public C ABI.
