# Standalone HTML usage and JavaScript API

The standalone OCR page packages the WebAssembly runtime, DET/CLS/REC models,
dictionary, user interface, and support assets into one `ocr-demo.html` file.
It runs OCR locally in the browser and does not upload the selected image.

This document covers both direct end-user operation and integration code added
to the standalone page.

## Use the page directly

Download `ocr-demo.html` from a Release or build it with the
`lw-ocr-html` CMake target. Open the file directly in a modern browser; a
native server is not required.

1. On desktop, select an image or drag it onto the page.
2. On mobile, use **拍照识别** or **从相册选择**.
3. Enable CLS when text orientation classification is needed.
4. Select **开始识别**.
5. Copy or share the recognized text, or export UTF-8 TXT or JSON.

The page keeps a single current image and result. Selecting another image or
starting another recognition invalidates the previous export snapshot until
the new run succeeds.

The generated page uses a Blob Web Worker so inference does not block normal UI
interaction. If a browser or local security policy rejects Blob Workers, the
page falls back to a compatible main-thread backend. The WebAssembly module
itself does not use pthreads and retains one internal OCR line worker.

The default build enables WebAssembly SIMD128. Use a browser with WebAssembly
SIMD support, or build with `-DLW_WASM_SIMD128=OFF` when an older scalar-only
browser must be supported.

## Integration boundary

`ocr-demo.html` is a complete application, not a JavaScript library. Do not
load it with:

```html
<!-- Not supported -->
<script src="ocr-demo.html"></script>
```

The public `window.lwPpocrDemo` API is intended for code added after the
existing application script in `web/ocr-demo.template.html`, or for other
customizations made to the generated standalone page. It updates the built-in
preview, status, result list, and export state.

Cross-page iframe access is not a supported integration contract. In
particular, local `file://` iframe behavior varies by browser and origin
policy. Applications that need an independent HTTP API should use the native
HTTP Demo described in [managed demos](managed-demos.md).

## Basic JavaScript integration

Add custom markup where appropriate in the template:

```html
<input id="my-image" type="file" accept="image/*">
<button id="my-ocr" type="button">Recognize</button>
<pre id="my-result"></pre>
```

Add the following script after the existing OCR application script:

```html
<script>
  const imageInput = document.getElementById("my-image");
  const resultNode = document.getElementById("my-result");

  document.getElementById("my-ocr").addEventListener("click", async () => {
    const file = imageInput.files[0];
    if (!file) {
      resultNode.textContent = "Select an image first.";
      return;
    }

    try {
      await window.lwPpocrDemo.ready();
      const result = await window.lwPpocrDemo.recognize(file, {
        useCls: true
      });
      resultNode.textContent = JSON.stringify(result, null, 2);
    } catch (error) {
      resultNode.textContent = "OCR failed: " + error.message;
    }
  });
</script>
```

The normal call sequence is:

```text
page load
  -> await lwPpocrDemo.ready()
  -> obtain a browser File or decodable Blob
  -> await lwPpocrDemo.recognize(file, options)
  -> consume the schema_version: 1 result
```

Calls to `recognize` must be serialized. Await one call before starting the
next one.

## API reference

### `apiVersion`

The JavaScript integration API version. Its current value is `1`. This is
separate from the OCR JSON `schema_version`.

### `ready()`

```javascript
await window.lwPpocrDemo.ready();
```

Waits for the current WASM/model initialization attempt. It rejects when the
engine did not initialize successfully. It can also be used after changing CLS
while the page reinitializes the engine.

### `selectImage(file)`

```javascript
await window.lwPpocrDemo.selectImage(file);
```

Decodes and prepares a browser `File` or decodable `Blob`, updates the
built-in preview, and invalidates the previous result. It does not run OCR.
Ignore its return value.

### `recognize(file?, options?)`

```javascript
const result = await window.lwPpocrDemo.recognize(file, {
  useCls: false
});
```

Runs the complete DET -> optional CLS -> REC pipeline and returns the same
structured object used by JSON export.

- `file` is optional. When omitted, the current prepared image is used.
- `options.useCls` is optional. Pass an explicit Boolean when an integration
  must not depend on the current checkbox state.
- Passing a file prepares it before recognition.
- Omitting `file` when no current image is prepared rejects the Promise.
- Starting another call while recognition is running rejects the Promise.
- The Promise rejects if initialization, image decoding, or recognition fails.
- A successful image with no detected text returns an empty `lines` array.

### `getResult()`

```javascript
const result = window.lwPpocrDemo.getResult();
```

Returns a deep copy of the last successful structured result, or `null` when
no current result is available. Mutating this copy does not alter page state.

### `getPlainText()`

```javascript
const text = window.lwPpocrDemo.getPlainText();
```

Returns recognized lines joined with LF characters in display/reading order.
It returns an empty string when there is no current result.

### `getStatus()`

```javascript
const status = window.lwPpocrDemo.getStatus();
console.log(status.ready, status.backend, status.heapBytes);
```

Returns a diagnostic snapshot used for troubleshooting and automated tests.
Useful fields include:

| Field | Meaning |
|---|---|
| `ready` | The engine is ready to recognize an image |
| `backend` | `worker`, `main-thread`, `loading`, or `failed` |
| `runCount` | Number of successful OCR runs |
| `heapBytes` | Current WebAssembly linear-memory high-water mark |
| `sourceCapacity` | Reusable BGR input capacity in bytes |
| `prepareCount` | Number of images decoded and prepared |
| `prepared` | A current image is ready |
| `hasResults` | A successful current result exists |
| `exportEnabled` | Copy/TXT/JSON actions are enabled |

Diagnostic objects may gain optional fields without changing `apiVersion`.

## Document events

Integrations may observe page activity without replacing the built-in
controls:

```javascript
document.addEventListener("lwppocr:ready", event => {
  console.log("OCR backend:", event.detail.backend);
});

document.addEventListener("lwppocr:result", event => {
  console.log("OCR result:", event.detail);
});

document.addEventListener("lwppocr:error", event => {
  console.error(event.detail.phase, event.detail.message);
});
```

| Event | `detail` |
|---|---|
| `lwppocr:ready` | `{backend}`; emitted after initial setup and CLS reinitialization |
| `lwppocr:result` | The successful `schema_version: 1` result |
| `lwppocr:error` | `{phase, message}`, where phase is `initialize` or `recognize` |

Register listeners before calling `recognize`. Code that must observe the
initial ready event should be placed before the page calls `boot()`; otherwise
call `ready()` to query the current state reliably.

## Result format

The structured result contains the original image dimensions, options, elapsed
time, reading-order text lines, original-image quadrilaterals, and confidence
scores. See [OCR result export schema](ocr-export-schema.md) for the versioned
field contract.

The JavaScript API and JSON export are application-level contracts. They do not
change or extend the public C ABI.

## Internal names are not public API

Do not depend on the Worker message protocol, `MODEL_B64`, `RUNTIME_JS`,
WASM pointers, buffer-management helpers, DOM state variables, or
`window.__lwOcrTest`. The last name exists only for browser automation and
may change with the tests.
