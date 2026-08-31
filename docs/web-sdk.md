# Browser JavaScript SDK

**lw-ppocr.js** is the reusable browser SDK for full local OCR. It contains the
Emscripten runtime, WebAssembly, DET/CLS/REC models, and dictionary in one
JavaScript file. Images stay in the browser; using the SDK does not require the
native HTTP Demo or a model download at runtime.

The SDK is separate from **ocr-demo.html**: applications load the JavaScript
SDK, while people who only need the ready-made interface can open the
standalone HTML directly.

## Quick start

Place the release SDK beside the application page:

~~~html
<input id="image" type="file" accept="image/*">
<button id="recognize" type="button">Recognize</button>
<pre id="output"></pre>

<script src="./lw-ppocr.js"></script>
<script>
  let ocr;

  async function getOcr() {
    if (!ocr) {
      ocr = await LwPpocr.create({
        useCls: false,
        maxImageSide: 1600
      });
    }
    return ocr;
  }

  document.getElementById("recognize").addEventListener("click", async () => {
    const file = document.getElementById("image").files[0];
    if (!file) return;

    const button = document.getElementById("recognize");
    button.disabled = true;
    try {
      const engine = await getOcr();
      const result = await engine.recognize(file);
      document.getElementById("output").textContent =
        result.lines.map(line => line.text).join("\n");
    } catch (error) {
      console.error(error.code, error.stage, error.message);
    } finally {
      button.disabled = false;
    }
  });

  window.addEventListener("beforeunload", () => {
    if (ocr) ocr.destroy();
  });
</script>
~~~

The normal lifecycle is:

~~~text
load lw-ppocr.js
  -> await LwPpocr.create(options)
  -> await engine.recognize(source)
  -> consume result_version: 1
  -> engine.destroy()
~~~

Keep and reuse one engine when processing multiple images. Model
initialization is relatively expensive, while image and output buffers grow to
a high-water mark and are reused by later calls.

## Public namespace

The script defines one frozen global object:

~~~javascript
LwPpocr.version;       // SDK package version, for example "0.1.0"
LwPpocr.webAbiVersion; // low-level Web ABI used by this SDK; currently 1
LwPpocr.Error;         // error class
LwPpocr.create;        // async factory
~~~

Do not call Emscripten module functions, Web ABI functions, the virtual file
system, or heap views directly. They are implementation details and may change
independently of the JavaScript SDK contract.

## Create an engine

~~~javascript
const engine = await LwPpocr.create({
  useCls: true,
  maxImageSide: 1600
});
~~~

| Option | Type | Default | Meaning |
|---|---:|---:|---|
| **useCls** | Boolean | false | Run text-orientation classification before recognition |
| **maxImageSide** | integer | 1600 | Scale larger images down before OCR; zero disables this limit |

Options belong to the engine. To change **useCls**, destroy the old engine and
create another one. This keeps buffer ownership and concurrent-call behavior
unambiguous.

The SDK normally runs the Emscripten module in a Blob Worker so inference does
not block the page. If the browser or its security policy rejects Blob Workers,
the SDK falls back to a main-thread backend. Inspect
**engine.getStatus().backend** when diagnostics need to distinguish
**worker** from **main-thread**.

## Recognize an image

The async **engine.recognize(source)** method accepts:

- File;
- any decodable image Blob;
- ImageData;
- HTMLCanvasElement.

~~~javascript
const result = await engine.recognize(file);
~~~

Calls on the same engine must be serialized. Starting another recognition
before the first Promise settles rejects with **LW_OCR_BUSY**. Use separate
engine instances only when the extra model and WebAssembly memory is acceptable.

For File input, **result.source** is the filename. Sources without a filename
use **image**. When **maxImageSide** scales an image internally,
**result.image** and every box coordinate are restored to the source image
dimensions.

## Result contract

The SDK returns a versioned object:

~~~json
{
  "result_version": 1,
  "source": "article.png",
  "image": {"width": 1920, "height": 1080},
  "options": {"use_cls": false},
  "timing": {
    "decode_ms": 4.125,
    "inference_ms": 86.75,
    "total_ms": 91.203
  },
  "lines": [
    {
      "index": 0,
      "text": "识别文字",
      "box": [10, 20, 200, 20, 200, 60, 10, 60],
      "det_score": 0.97,
      "rec_score": 0.99
    }
  ]
}
~~~

The box is the reading-order quadrilateral
**[x1,y1,x2,y2,x3,y3,x4,y4]** in original-image pixels. When **useCls** is
true, each line also includes:

~~~json
{
  "cls_score": 0.998,
  "cls_label": 0,
  "rotation_degrees": 0
}
~~~

**decode_ms** includes browser decoding, optional scaling, and RGBA-to-BGR
conversion. **inference_ms** measures the native DET/CLS/REC call.
**total_ms** includes both and JavaScript result conversion.

The standalone Demo adapts this object to the downloadable
**schema_version: 1** format documented in
[OCR result export schema](ocr-export-schema.md). The C ABI is unchanged.

## State, status, and errors

An engine follows:

~~~text
CREATING -> READY -> RUNNING -> READY -> DESTROYED
~~~

**engine.getStatus()** returns diagnostic data including state, backend, ready,
runCount, heapBytes, and reusable buffer capacities. Applications normally only
need state, backend, and ready.

Errors are instances of **LwPpocr.Error** with a stable machine-readable
**code**, a **stage**, and a human-readable **message**.

| Code | Meaning |
|---|---|
| **LW_OCR_OPTIONS** | Invalid create options |
| **LW_OCR_INIT_FAILED** | Runtime or model initialization failed |
| **LW_OCR_BUSY** | The same engine already has a running recognition |
| **LW_OCR_FAILED** | Browser preparation or inference failed |
| **LW_OCR_DESTROYED** | The engine has already been destroyed |

Native numeric status codes can also appear for low-level allocation, ABI, or
inference failures. Application logic should use **code**, not parse localized
message text.

**destroy()** is idempotent. It terminates the Worker or shuts down the fallback
module and releases SDK-owned buffers. A destroyed engine cannot be reused.

## Build and test

After activating Emscripten:

~~~bash
emcmake cmake -S . -B build-wasm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLW_BUILD_HTTP_DEMO=OFF \
  -DLW_BUILD_CSHARP_DEMOS=OFF \
  -DBUILD_TESTING=OFF

cmake --build build-wasm --target lw-ocr-js
cmake --build build-wasm --target lw-ocr-html
~~~

Outputs:

- **build-wasm/lw-ppocr.js**: reusable JavaScript SDK;
- **build-wasm/ocr-demo.html**: standalone application containing that SDK and
  the example UI.

The SDK packager concatenates the generated Emscripten runtime directly with
the wrapper. It does not use eval or new Function. The HTML packager then
inlines the exact SDK artifact plus **web/ocr-demo-ui.js**.

CI runs both browser suites:

~~~bash
python web/test_ocr_sdk.py \
  --sdk build-wasm/lw-ppocr.js \
  --sample models/ppocrv6-tiny/sample.jpg

python web/test_ocr_html.py \
  --html build-wasm/ocr-demo.html \
  --sample models/ppocrv6-tiny/sample.jpg
~~~

The SDK test covers the public namespace, File/Canvas/ImageData input,
structured results, busy/destroyed behavior, CLS on/off, repeated inference,
and the post-warm-up WebAssembly heap high-water mark. The HTML test separately
covers the example UI, exports, compatibility adapter, and phone layout.
