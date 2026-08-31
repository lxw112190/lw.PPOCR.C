# Standalone HTML usage

**ocr-demo.html** is the ready-made offline OCR application. It embeds the
browser SDK, WebAssembly runtime, DET/CLS/REC models, dictionary, user
interface, and support image in one file. Selected images remain local and are
not uploaded.

For a custom browser application, use the separate **lw-ppocr.js** artifact and
the public **LwPpocr.create()** API described in the
[Browser JavaScript SDK guide](web-sdk.md). Do not load an HTML file as a
script.

## Use the page

Download the release HTML or build the **lw-ocr-html** CMake target, then open
**ocr-demo.html** directly in a modern browser.

1. On desktop, select an image or drag it onto the page.
2. On a phone, choose **拍照识别** or **从相册选择**.
3. Enable CLS when orientation classification is needed.
4. Select **开始识别**.
5. Copy or share the text, or export UTF-8 TXT or structured JSON.

The page keeps one current image and one result. Selecting another image
invalidates the old export snapshot until recognition succeeds again.

Inference normally runs inside a Blob Worker so the controls remain
responsive. If local browser policy rejects Blob Workers, the embedded SDK
uses its compatible main-thread fallback. The default build enables
WebAssembly SIMD128.

## Customize the Demo

The source is deliberately split into three layers:

~~~text
web/lw_ppocr_sdk.template.js  reusable OCR SDK
             |
web/ocr-demo-ui.js            DOM, preview, export, mobile interaction
             |
web/ocr-demo.template.html    markup and responsive styles
~~~

The build first produces **lw-ppocr.js**, then injects that exact artifact and
the UI script into **ocr-demo.html**. Integrations should keep image decoding,
WASM pointers, buffer ownership, and native calls inside the SDK. Demo changes
should operate on structured OCR results.

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
- **lwppocr:error** with **{phase, message}**.

The test hook, Worker messages, embedded model variables, Emscripten members,
and DOM state are implementation details, not public API.

## Export format

The Demo export contains original image dimensions, OCR options, elapsed time,
reading-order lines, original-image quadrilaterals, and confidence scores. See
[OCR result export schema](ocr-export-schema.md).

The browser SDK and the Demo export are application-layer contracts. Neither
changes the public C ABI.
