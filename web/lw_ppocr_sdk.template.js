(function(global) {
  "use strict";

  const SDK_VERSION = __LW_SDK_VERSION__;
  const WEB_ABI_VERSION = 1;
  const MODEL_B64 = {
    det: "__LW_DET_MODEL_BASE64__",
    cls: "__LW_CLS_MODEL_BASE64__",
    rec: "__LW_REC_MODEL_BASE64__",
    dict: "__LW_DICTIONARY_BASE64__"
  };
  // The runtime is concatenated directly before this wrapper. This copy is
  // only used to construct the optional offline Worker without eval/new Function.
  const WORKER_RUNTIME_SOURCE = __LW_RUNTIME_JS_JSON__;

  class LwPpocrError extends Error {
    constructor(message, code, stage) {
      super(message);
      this.name = "LwPpocrError";
      this.code = code;
      this.stage = stage;
    }
  }

  /*
   * Worker entry point. The SDK creates a Blob containing:
   *   generated Emscripten runtime + this function + model payload
   * The whole WASM instance and its buffers therefore stay off the UI thread.
   */
  function workerMain(models) {
    const ABI_VERSION = 1;
    let module = null;
    let sourceCapacity = 0;
    let inputId = 0;
    let maxLines = 0;
    let maxText = 0;
    let lineSize = 0;
    let resultSize = 0;
    const pointers = {source: 0, lines: 0, text: 0, result: 0};

    function decode(value) {
      const raw = atob(value);
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; ++i) bytes[i] = raw.charCodeAt(i);
      return bytes;
    }
    function u32(pointer) { return module.HEAPU32[pointer >> 2] >>> 0; }
    function f32(pointer) { return module.HEAPF32[pointer >> 2]; }
    function utf8(pointer, length) {
      return new TextDecoder().decode(module.HEAPU8.subarray(pointer, pointer + length));
    }
    function allocate(size, label) {
      const pointer = module._lw_web_malloc(size);
      if (!pointer) throw new Error("无法分配" + label + "（" + size + " 字节）");
      return pointer;
    }
    function releaseOutputs() {
      for (const name of ["lines", "text", "result"]) {
        if (pointers[name]) module._lw_web_free(pointers[name]);
        pointers[name] = 0;
      }
    }
    function ensureSource(required) {
      if (required <= sourceCapacity) return pointers.source;
      const replacement = allocate(required, "输入图像缓冲区");
      if (pointers.source) module._lw_web_free(pointers.source);
      pointers.source = replacement;
      sourceCapacity = required;
      return replacement;
    }
    function snapshot() {
      return {
        heapBytes: module ? module.HEAPU8.buffer.byteLength : 0,
        sourceCapacity,
        maxLineCapacity: maxLines,
        maxTextCapacity: maxText,
        lineSize,
        resultSize
      };
    }
    async function initialize(useCls) {
      if (!module) {
        module = await LwPpocrModule({});
        module.FS.mkdir("/models");
        for (const [name, key] of [
          ["det.lwm", "det"], ["cls.lwm", "cls"],
          ["rec.lwm", "rec"], ["ppocr_keys.txt", "dict"]
        ]) {
          module.FS.writeFile("/models/" + name, decode(models[key]));
          models[key] = "";
        }
      }
      if (pointers.lines) {
        releaseOutputs();
        module._lw_web_shutdown();
      }
      const status = module._lw_web_init(useCls ? 1 : 0);
      if (status !== 0) throw new Error("初始化失败：" + status);
      const infoSize = module._lw_web_info_size();
      if (infoSize !== 20) throw new Error("不支持的 Web ABI 信息大小：" + infoSize);
      const info = allocate(infoSize, "Web ABI 信息");
      try {
        if (module._lw_web_get_info(info) !== 0) throw new Error("读取引擎容量失败");
        if (u32(info) !== ABI_VERSION) throw new Error("不支持的 Web ABI：" + u32(info));
        maxLines = u32(info + 4);
        maxText = u32(info + 8);
        lineSize = u32(info + 12);
        resultSize = u32(info + 16);
        if (!maxLines || !maxText || lineSize !== 60 || resultSize !== 16) {
          throw new Error("引擎返回了无效的输出容量");
        }
      } finally {
        module._lw_web_free(info);
      }
      pointers.lines = allocate(maxLines * lineSize, "识别行缓冲区");
      pointers.text = allocate(maxText, "文本缓冲区");
      pointers.result = allocate(resultSize, "结果缓冲区");
      return {type: "ready", snapshot: snapshot()};
    }
    function run(message) {
      if (!pointers.lines) throw new Error("OCR 引擎尚未初始化");
      if (message.pixels) {
        const pixels = new Uint8Array(message.pixels);
        const source = ensureSource(pixels.byteLength);
        module.HEAPU8.set(pixels, source);
        inputId = message.inputId;
      }
      if (!inputId || inputId !== message.inputId) throw new Error("OCR 输入图像已经失效");
      const started = performance.now();
      const status = module._lw_web_run(
        pointers.source, message.byteLength, message.width, message.height,
        message.width * 3, pointers.lines, maxLines, pointers.text, maxText,
        pointers.result
      );
      if (status !== 0) throw new Error("推理失败：" + status);
      const count = u32(pointers.result);
      if (count > maxLines) throw new Error("引擎返回的文字行数量超出容量");
      const lines = [];
      for (let i = 0; i < count; ++i) {
        const pointer = pointers.lines + i * lineSize;
        const box = [];
        for (let k = 0; k < 8; ++k) box.push(f32(pointer + k * 4));
        const offset = u32(pointer + 52);
        const length = u32(pointer + 56);
        if (offset > maxText || length > maxText - offset) {
          throw new Error("引擎返回的文本范围无效");
        }
        lines.push({
          box,
          text: utf8(pointers.text + offset, length),
          detection: f32(pointer + 32),
          recognition: f32(pointer + 36),
          classification: f32(pointer + 40),
          classificationLabel: u32(pointer + 44),
          rotationDegrees: u32(pointer + 48)
        });
      }
      return {
        type: "result",
        lines,
        inferenceMilliseconds: performance.now() - started,
        snapshot: snapshot()
      };
    }
    self.onmessage = async event => {
      const message = event.data;
      try {
        let response;
        if (message.type === "initialize") response = await initialize(message.useCls);
        else if (message.type === "run") response = run(message);
        else throw new Error("未知的 Worker 请求：" + message.type);
        self.postMessage({requestId: message.requestId, ok: true, ...response});
      } catch (error) {
        self.postMessage({
          requestId: message.requestId,
          ok: false,
          error: error && error.message ? error.message : String(error)
        });
      }
    };
  }

  class LwPpocrEngine {
    constructor(options) {
      this._options = {
        useCls: Boolean(options.useCls),
        maxImageSide: options.maxImageSide === undefined ? 1600 : options.maxImageSide
      };
      this._state = "CREATING";
      this._backend = "loading";
      this._module = null;
      this._worker = null;
      this._workerUrl = null;
      this._pending = new Map();
      this._nextRequestId = 1;
      this._buffers = {source: 0, lines: 0, text: 0, result: 0};
      this._sourceCapacity = 0;
      this._maxLineCapacity = 0;
      this._maxTextCapacity = 0;
      this._lineSize = 0;
      this._resultSize = 0;
      this._runCount = 0;
      this._lastHeapBytes = 0;
    }

    async initialize() {
      if (!global.Worker || !global.Blob || !global.URL.createObjectURL) {
        await this._initializeDirect();
        this._backend = "main-thread";
        this._state = "READY";
        return;
      }
      try {
        await this._initializeWorker();
        this._backend = "worker";
      } catch (workerError) {
        this._closeWorker();
        await this._initializeDirect();
        this._backend = "main-thread";
      }
      this._state = "READY";
    }

    async _initializeWorker() {
      const source = [
        WORKER_RUNTIME_SOURCE,
        "\n(", workerMain.toString(), ")(",
        JSON.stringify(MODEL_B64), ");"
      ].join("");
      this._workerUrl = global.URL.createObjectURL(new Blob([source], {type: "text/javascript"}));
      const worker = new global.Worker(this._workerUrl);
      this._worker = worker;
      worker.onmessage = event => {
        const message = event.data;
        const request = this._pending.get(message.requestId);
        if (!request) return;
        this._pending.delete(message.requestId);
        if (message.ok) request.resolve(message);
        else request.reject(new Error(message.error || "Worker 请求失败"));
      };
      worker.onerror = event => {
        const error = new Error(event.message || "OCR Worker 启动失败");
        for (const request of this._pending.values()) request.reject(error);
        this._pending.clear();
      };
      const ready = await this._workerRequest({type: "initialize", useCls: this._options.useCls});
      this._applySnapshot(ready.snapshot);
      global.URL.revokeObjectURL(this._workerUrl);
      this._workerUrl = null;
    }

    _workerRequest(message, transfer = []) {
      if (!this._worker) return Promise.reject(new Error("OCR Worker 不可用"));
      const requestId = this._nextRequestId++;
      return new Promise((resolve, reject) => {
        this._pending.set(requestId, {resolve, reject});
        this._worker.postMessage({...message, requestId}, transfer);
      });
    }

    async _initializeDirect() {
      this._module = await LwPpocrModule({});
      this._module.FS.mkdir("/models");
      for (const [name, key] of [
        ["det.lwm", "det"], ["cls.lwm", "cls"],
        ["rec.lwm", "rec"], ["ppocr_keys.txt", "dict"]
      ]) {
        this._module.FS.writeFile("/models/" + name, this._decodeBase64(MODEL_B64[key]));
      }
      const status = this._module._lw_web_init(this._options.useCls ? 1 : 0);
      if (status !== 0) throw new LwPpocrError("初始化失败：" + status, status, "initialize");
      this._allocateDirectOutputs();
      this._lastHeapBytes = this._module.HEAPU8.buffer.byteLength;
    }

    _decodeBase64(value) {
      const raw = atob(value);
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; ++i) bytes[i] = raw.charCodeAt(i);
      return bytes;
    }

    _allocateDirect(size, label) {
      const pointer = this._module._lw_web_malloc(size);
      if (!pointer) throw new LwPpocrError("无法分配" + label + "（" + size + " 字节）", 12, "allocation");
      return pointer;
    }

    _allocateDirectOutputs() {
      const infoSize = this._module._lw_web_info_size();
      if (infoSize !== 20) throw new LwPpocrError("不支持的 Web ABI 信息大小：" + infoSize, 95, "initialize");
      const info = this._allocateDirect(infoSize, "Web ABI 信息");
      try {
        if (this._module._lw_web_get_info(info) !== 0) {
          throw new LwPpocrError("读取引擎容量失败", 1, "initialize");
        }
        const u32 = pointer => this._module.HEAPU32[pointer >> 2] >>> 0;
        if (u32(info) !== WEB_ABI_VERSION) {
          throw new LwPpocrError("不支持的 Web ABI：" + u32(info), 95, "initialize");
        }
        this._maxLineCapacity = u32(info + 4);
        this._maxTextCapacity = u32(info + 8);
        this._lineSize = u32(info + 12);
        this._resultSize = u32(info + 16);
        if (!this._maxLineCapacity || !this._maxTextCapacity ||
            this._lineSize !== 60 || this._resultSize !== 16) {
          throw new LwPpocrError("引擎返回了无效的输出容量", 95, "initialize");
        }
      } finally {
        this._module._lw_web_free(info);
      }
      this._buffers.lines = this._allocateDirect(this._maxLineCapacity * this._lineSize, "识别行缓冲区");
      this._buffers.text = this._allocateDirect(this._maxTextCapacity, "文本缓冲区");
      this._buffers.result = this._allocateDirect(this._resultSize, "结果缓冲区");
    }

    _applySnapshot(snapshot) {
      this._lastHeapBytes = snapshot.heapBytes;
      this._sourceCapacity = snapshot.sourceCapacity;
      this._maxLineCapacity = snapshot.maxLineCapacity;
      this._maxTextCapacity = snapshot.maxTextCapacity;
      this._lineSize = snapshot.lineSize;
      this._resultSize = snapshot.resultSize;
    }

    async _decodeSource(source) {
      if (typeof ImageData !== "undefined" && source instanceof ImageData) {
        return {image: source, width: source.width, height: source.height, close: null};
      }
      if (typeof HTMLCanvasElement !== "undefined" && source instanceof HTMLCanvasElement) {
        const context = source.getContext("2d");
        if (!context) throw new LwPpocrError("无法读取 Canvas", 22, "decode");
        return {
          image: context.getImageData(0, 0, source.width, source.height),
          width: source.width,
          height: source.height,
          close: null
        };
      }
      if (typeof Blob === "undefined" || !(source instanceof Blob)) {
        throw new LwPpocrError("recognize() 需要 File、Blob、ImageData 或 HTMLCanvasElement", 22, "decode");
      }
      if (global.createImageBitmap) {
        const image = await global.createImageBitmap(source);
        return {image, width: image.width, height: image.height, close: image.close ? () => image.close() : null};
      }
      if (typeof document === "undefined") {
        throw new LwPpocrError("当前浏览器没有可用的图片解码器", 22, "decode");
      }
      const url = global.URL.createObjectURL(source);
      try {
        const image = new Image();
        image.decoding = "async";
        image.src = url;
        await image.decode();
        return {image, width: image.naturalWidth, height: image.naturalHeight, close: null};
      } finally {
        global.URL.revokeObjectURL(url);
      }
    }

    async _prepare(source) {
      const started = performance.now();
      const decoded = await this._decodeSource(source);
      try {
        const originalWidth = decoded.width;
        const originalHeight = decoded.height;
        const limit = this._options.maxImageSide;
        const scale = limit ? Math.min(1, limit / Math.max(originalWidth, originalHeight)) : 1;
        const width = Math.max(1, Math.round(originalWidth * scale));
        const height = Math.max(1, Math.round(originalHeight * scale));
        let imageData;
        if (typeof ImageData !== "undefined" && decoded.image instanceof ImageData &&
            width === originalWidth && height === originalHeight) {
          imageData = decoded.image;
        } else {
          if (typeof document === "undefined") {
            throw new LwPpocrError("当前环境无法缩放图片", 22, "decode");
          }
          const canvas = document.createElement("canvas");
          canvas.width = width;
          canvas.height = height;
          const context = canvas.getContext("2d", {willReadFrequently: true});
          context.drawImage(decoded.image, 0, 0, width, height);
          imageData = context.getImageData(0, 0, width, height);
        }
        const rgba = imageData.data;
        const pixels = new Uint8Array(width * height * 3);
        for (let i = 0, j = 0; i < rgba.length; i += 4) {
          pixels[j++] = rgba[i + 2];
          pixels[j++] = rgba[i + 1];
          pixels[j++] = rgba[i];
        }
        return {
          originalWidth, originalHeight, width, height,
          pixels, byteLength: pixels.byteLength,
          prepareMilliseconds: performance.now() - started,
          source: source && source.name ? source.name : "image"
        };
      } finally {
        if (decoded.close) decoded.close();
      }
    }

    _readDirectLines() {
      const u32 = pointer => this._module.HEAPU32[pointer >> 2] >>> 0;
      const f32 = pointer => this._module.HEAPF32[pointer >> 2];
      const utf8 = (pointer, length) =>
        new TextDecoder().decode(this._module.HEAPU8.subarray(pointer, pointer + length));
      const count = u32(this._buffers.result);
      if (count > this._maxLineCapacity) throw new LwPpocrError("文字行数量超出容量", 75, "inference");
      const lines = [];
      for (let i = 0; i < count; ++i) {
        const pointer = this._buffers.lines + i * this._lineSize;
        const box = [];
        for (let k = 0; k < 8; ++k) box.push(f32(pointer + k * 4));
        const offset = u32(pointer + 52);
        const length = u32(pointer + 56);
        if (offset > this._maxTextCapacity || length > this._maxTextCapacity - offset) {
          throw new LwPpocrError("文本范围无效", 75, "inference");
        }
        lines.push({
          box, text: utf8(this._buffers.text + offset, length),
          detection: f32(pointer + 32), recognition: f32(pointer + 36),
          classification: f32(pointer + 40),
          classificationLabel: u32(pointer + 44),
          rotationDegrees: u32(pointer + 48)
        });
      }
      return lines;
    }

    _ensureDirectSource(required) {
      if (required <= this._sourceCapacity) return this._buffers.source;
      const replacement = this._allocateDirect(required, "输入图像缓冲区");
      if (this._buffers.source) this._module._lw_web_free(this._buffers.source);
      this._buffers.source = replacement;
      this._sourceCapacity = required;
      return replacement;
    }

    async recognize(source) {
      if (this._state === "DESTROYED") {
        throw new LwPpocrError("OCR 实例已经销毁", "LW_OCR_DESTROYED", "destroyed");
      }
      if (this._state !== "READY") {
        throw new LwPpocrError("OCR 实例正忙，请等待当前识别完成", "LW_OCR_BUSY", "busy");
      }
      this._state = "RUNNING";
      const started = performance.now();
      try {
        const prepared = await this._prepare(source);
        let response;
        if (this._worker) {
          response = await this._workerRequest({
            type: "run",
            inputId: this._runCount + 1,
            width: prepared.width,
            height: prepared.height,
            byteLength: prepared.byteLength,
            pixels: prepared.pixels.buffer
          }, [prepared.pixels.buffer]);
          this._applySnapshot(response.snapshot);
        } else {
          const pointer = this._ensureDirectSource(prepared.byteLength);
          this._module.HEAPU8.set(prepared.pixels, pointer);
          const inferenceStarted = performance.now();
          const status = this._module._lw_web_run(
            pointer, prepared.byteLength, prepared.width, prepared.height,
            prepared.width * 3, this._buffers.lines, this._maxLineCapacity,
            this._buffers.text, this._maxTextCapacity, this._buffers.result
          );
          if (status !== 0) throw new LwPpocrError("推理失败：" + status, status, "inference");
          response = {
            lines: this._readDirectLines(),
            inferenceMilliseconds: performance.now() - inferenceStarted,
            snapshot: {
              heapBytes: this._module.HEAPU8.buffer.byteLength,
              sourceCapacity: this._sourceCapacity,
              maxLineCapacity: this._maxLineCapacity,
              maxTextCapacity: this._maxTextCapacity,
              lineSize: this._lineSize,
              resultSize: this._resultSize
            }
          };
          this._applySnapshot(response.snapshot);
        }
        const xScale = prepared.originalWidth / prepared.width;
        const yScale = prepared.originalHeight / prepared.height;
        const lines = response.lines.map((line, index) => {
          const exported = {
            index,
            text: line.text,
            box: line.box.map((coordinate, coordinateIndex) =>
              coordinate * (coordinateIndex % 2 ? yScale : xScale)),
            det_score: line.detection,
            rec_score: line.recognition
          };
          if (this._options.useCls) {
            exported.cls_score = line.classification;
            exported.cls_label = line.classificationLabel;
            exported.rotation_degrees = line.rotationDegrees;
          }
          return exported;
        });
        this._runCount += 1;
        return {
          result_version: 1,
          source: prepared.source,
          image: {width: prepared.originalWidth, height: prepared.originalHeight},
          options: {use_cls: this._options.useCls},
          timing: {
            decode_ms: Number(prepared.prepareMilliseconds.toFixed(3)),
            inference_ms: Number(response.inferenceMilliseconds.toFixed(3)),
            total_ms: Number((performance.now() - started).toFixed(3))
          },
          lines
        };
      } catch (error) {
        if (error instanceof LwPpocrError) throw error;
        throw new LwPpocrError(String(error), "LW_OCR_FAILED", "recognize");
      } finally {
        if (this._state !== "DESTROYED") this._state = "READY";
      }
    }

    getStatus() {
      return {
        state: this._state,
        backend: this._backend,
        ready: this._state === "READY",
        runCount: this._runCount,
        heapBytes: this._lastHeapBytes || (this._module ? this._module.HEAPU8.buffer.byteLength : 0),
        sourceCapacity: this._sourceCapacity,
        maxLineCapacity: this._maxLineCapacity,
        maxTextCapacity: this._maxTextCapacity,
        lineSize: this._lineSize,
        resultSize: this._resultSize
      };
    }

    _closeWorker() {
      for (const request of this._pending.values()) {
        request.reject(new LwPpocrError("OCR Worker 已关闭", "LW_OCR_DESTROYED", "destroyed"));
      }
      this._pending.clear();
      if (this._worker) this._worker.terminate();
      if (this._workerUrl) global.URL.revokeObjectURL(this._workerUrl);
      this._worker = null;
      this._workerUrl = null;
    }

    destroy() {
      if (this._state === "DESTROYED") return;
      this._closeWorker();
      if (this._module) {
        for (const name of ["source", "lines", "text", "result"]) {
          if (this._buffers[name]) this._module._lw_web_free(this._buffers[name]);
          this._buffers[name] = 0;
        }
        this._module._lw_web_shutdown();
        this._module = null;
      }
      this._state = "DESTROYED";
    }
  }

  async function create(options = {}) {
    if (!options || typeof options !== "object") {
      throw new LwPpocrError("create() options 必须是对象", "LW_OCR_OPTIONS", "initialize");
    }
    const maxImageSide = options.maxImageSide === undefined ? 1600 : options.maxImageSide;
    if (!Number.isInteger(maxImageSide) || maxImageSide < 0) {
      throw new LwPpocrError("maxImageSide 必须是非负整数", "LW_OCR_OPTIONS", "initialize");
    }
    const engine = new LwPpocrEngine({...options, maxImageSide});
    try {
      await engine.initialize();
      return engine;
    } catch (error) {
      engine.destroy();
      if (error instanceof LwPpocrError) throw error;
      throw new LwPpocrError(String(error), "LW_OCR_INIT_FAILED", "initialize");
    }
  }

  global.LwPpocr = Object.freeze({
    version: SDK_VERSION,
    webAbiVersion: WEB_ABI_VERSION,
    Error: LwPpocrError,
    create
  });
})(globalThis);
