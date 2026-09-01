/*
 * PDF document frontend for the standalone OCR Demo.
 *
 * This adapter owns PDF.js loading and PDF page -> Canvas rendering only. It
 * deliberately knows nothing about LwPpocr or OCR result schemas.
 */
(function(global) {
  "use strict";

  const PDFJS_VERSION = "__LW_PDFJS_VERSION__";
  const PDFJS_CORE_BASE64 = "__LW_PDFJS_CORE_BASE64__";
  const PDFJS_WORKER_BASE64 = "__LW_PDFJS_WORKER_BASE64__";
  const WORKER_START_TIMEOUT_MS = 2500;
  let modulePromise = null;
  let coreUrl = null;
  let workerUrl = null;
  let sharedWorkerPort = null;
  let state = "not-loaded";
  let workerBackend = "not-loaded";
  let workerFallbackReason = null;
  let lastError = null;

  class LwPdfError extends Error {
    constructor(message, code, phase, cause) {
      super(message);
      this.name = "LwPdfError";
      this.code = code;
      this.phase = phase;
      if (cause !== undefined) this.cause = cause;
    }
  }

  function errorSummary(error) {
    if (!error) return null;
    const cause = error.cause;
    return {
      code: error.code || "LW_PDF_UNKNOWN",
      phase: error.phase || "unknown",
      message: error.message || String(error),
      cause_name: cause && cause.name ? String(cause.name) : null,
      cause_message: cause && cause.message ? String(cause.message) : null
    };
  }

  function rememberError(error) {
    lastError = errorSummary(error);
    return error;
  }

  function environmentSnapshot() {
    return {
      protocol: global.location ? global.location.protocol : "unknown",
      user_agent: global.navigator ? global.navigator.userAgent : "unknown",
      has_worker: typeof global.Worker === "function",
      has_file_reader: typeof global.FileReader === "function",
      has_blob_array_buffer: typeof global.Blob === "function" &&
        typeof global.Blob.prototype.arrayBuffer === "function"
    };
  }

  function getStatus() {
    return {
      api_version: 1,
      pdfjs_version: PDFJS_VERSION,
      state,
      worker_backend: workerBackend,
      worker_fallback_reason: workerFallbackReason,
      environment: environmentSnapshot(),
      last_error: lastError ? {...lastError} : null
    };
  }

  function base64BlobUrl(encoded) {
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let offset = 0; offset < binary.length; offset += 1) {
      bytes[offset] = binary.charCodeAt(offset);
    }
    return URL.createObjectURL(new Blob([bytes], {type: "text/javascript"}));
  }

  function closeWorkerPort() {
    if (sharedWorkerPort) sharedWorkerPort.terminate();
    sharedWorkerPort = null;
  }

  async function createModuleWorker() {
    if (typeof global.Worker !== "function") {
      return {port: null, reason: "Worker API unavailable"};
    }
    return new Promise(resolve => {
      let port;
      try {
        // Construct the original worker Blob directly. This avoids PDF.js
        // wrapping it in a second module Blob when the page has a null file://
        // origin, a path rejected by some mobile WebViews.
        port = new global.Worker(workerUrl, {type: "module"});
      } catch (error) {
        resolve({port: null, reason: error.message || String(error)});
        return;
      }
      let settled = false;
      const finish = (result) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        port.removeEventListener("message", onMessage);
        port.removeEventListener("error", onError);
        if (!result.port) port.terminate();
        resolve(result);
      };
      const onMessage = () => finish({port, reason: null});
      const onError = event => finish({
        port: null,
        reason: event && event.message ? event.message : "module Worker failed"
      });
      const timeout = setTimeout(() => finish({
        port: null,
        reason: "module Worker startup timed out"
      }), WORKER_START_TIMEOUT_MS);
      port.addEventListener("message", onMessage);
      port.addEventListener("error", onError);
    });
  }

  async function configurePdfWorker(pdfjs) {
    pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;
    const moduleWorker = await createModuleWorker();
    if (moduleWorker.port) {
      sharedWorkerPort = moduleWorker.port;
      pdfjs.GlobalWorkerOptions.workerPort = sharedWorkerPort;
      workerBackend = "module-worker";
      workerFallbackReason = null;
      return;
    }

    // PDF.js can run its worker protocol on the main thread. Importing the
    // worker module publishes globalThis.pdfjsWorker, which makes that fallback
    // deterministic instead of relying on a failed Worker to trigger it.
    workerFallbackReason = moduleWorker.reason || "module Worker unavailable";
    const workerModule = await import(workerUrl);
    global.pdfjsWorker = global.pdfjsWorker || workerModule;
    pdfjs.GlobalWorkerOptions.workerPort = null;
    workerBackend = "main-thread";
  }

  function resetModuleAssets() {
    closeWorkerPort();
    if (coreUrl) URL.revokeObjectURL(coreUrl);
    if (workerUrl) URL.revokeObjectURL(workerUrl);
    coreUrl = null;
    workerUrl = null;
  }

  async function loadPdfJs() {
    if (!modulePromise) {
      state = "loading";
      modulePromise = (async () => {
        coreUrl = base64BlobUrl(PDFJS_CORE_BASE64);
        workerUrl = base64BlobUrl(PDFJS_WORKER_BASE64);
        const pdfjs = await import(coreUrl);
        await configurePdfWorker(pdfjs);
        state = "ready";
        lastError = null;
        return pdfjs;
      })().catch(error => {
        resetModuleAssets();
        modulePromise = null;
        state = "failed";
        workerBackend = "failed";
        throw rememberError(new LwPdfError(
          "PDF 组件初始化失败",
          "LW_PDF_INIT_FAILED",
          "initialize",
          error
        ));
      });
    }
    return modulePromise;
  }

  function normalizeError(error, pdfjs, phase) {
    if (error instanceof LwPdfError) return error;
    if (error && error.name === "PasswordException") {
      const invalid = pdfjs && pdfjs.PasswordResponses &&
        error.code === pdfjs.PasswordResponses.INCORRECT_PASSWORD;
      return new LwPdfError(
        invalid ? "PDF 密码错误" : "此 PDF 已加密，当前版本暂不支持密码输入",
        invalid ? "LW_PDF_PASSWORD_INVALID" : "LW_PDF_PASSWORD_REQUIRED",
        "open",
        error
      );
    }
    if (error && error.name === "RenderingCancelledException") {
      return new LwPdfError("PDF 页面渲染已取消", "LW_PDF_CANCELLED", "render", error);
    }
    return new LwPdfError(
      phase === "render" ? "PDF 页面渲染失败" : "无法打开 PDF，文件可能损坏或格式不受支持",
      phase === "render" ? "LW_PDF_RENDER_FAILED" : "LW_PDF_LOAD_FAILED",
      phase,
      error
    );
  }

  async function readBlob(source) {
    if (!source || typeof source !== "object" || typeof source.size !== "number") {
      throw new LwPdfError("PDF 输入必须是 File 或 Blob", "LW_PDF_INPUT_REQUIRED", "read");
    }
    let nativeError = null;
    if (typeof source.arrayBuffer === "function") {
      try {
        return await source.arrayBuffer();
      } catch (error) {
        nativeError = error;
      }
    }
    if (typeof global.FileReader !== "function") {
      throw new LwPdfError(
        "当前浏览器无法读取所选 PDF",
        "LW_PDF_READ_FAILED",
        "read",
        nativeError
      );
    }
    return new Promise((resolve, reject) => {
      const reader = new global.FileReader();
      reader.onload = () => {
        if (reader.result instanceof ArrayBuffer) resolve(reader.result);
        else reject(new LwPdfError(
          "当前浏览器无法读取所选 PDF",
          "LW_PDF_READ_FAILED",
          "read"
        ));
      };
      reader.onerror = () => reject(new LwPdfError(
        "当前浏览器无法读取所选 PDF",
        "LW_PDF_READ_FAILED",
        "read",
        reader.error || nativeError
      ));
      reader.onabort = () => reject(new LwPdfError(
        "PDF 文件读取已取消",
        "LW_PDF_READ_CANCELLED",
        "read"
      ));
      try {
        reader.readAsArrayBuffer(source);
      } catch (error) {
        reject(new LwPdfError(
          "当前浏览器无法读取所选 PDF",
          "LW_PDF_READ_FAILED",
          "read",
          error
        ));
      }
    });
  }

  function assertPageNumber(pageNumber, pageCount) {
    if (!Number.isInteger(pageNumber) || pageNumber < 1 || pageNumber > pageCount) {
      throw new LwPdfError("PDF 页码超出范围", "LW_PDF_PAGE_FAILED", "render");
    }
  }

  async function open(source, options = {}) {
    const pdfjs = await loadPdfJs();
    let loadingTask = null;
    try {
      const data = new Uint8Array(await readBlob(source));
      loadingTask = pdfjs.getDocument({
        data,
        password: options.password || undefined,
        isEvalSupported: false,
        useWasm: false
      });
      const pdfDocument = await loadingTask.promise;
      let closed = false;
      let activeRenderTask = null;
      lastError = null;

      return Object.freeze({
        pageCount: pdfDocument.numPages,
        async renderPage(pageNumber, renderOptions = {}) {
          if (closed) {
            throw rememberError(new LwPdfError(
              "PDF 文档已经关闭", "LW_PDF_CLOSED", "render"));
          }
          assertPageNumber(pageNumber, pdfDocument.numPages);
          const dpi = renderOptions.dpi === undefined ? 180 : Number(renderOptions.dpi);
          const maxPixels = renderOptions.maxPixels === undefined ? 5000000 :
            Number(renderOptions.maxPixels);
          if (!Number.isFinite(dpi) || dpi <= 0 || !Number.isFinite(maxPixels) ||
              maxPixels < 1) {
            throw rememberError(new LwPdfError(
              "PDF 渲染参数无效", "LW_PDF_OPTIONS", "render"));
          }
          const page = await pdfDocument.getPage(pageNumber);
          const baseViewport = page.getViewport({scale: 1});
          const pdfWidth = Math.abs(page.view[2] - page.view[0]);
          const pdfHeight = Math.abs(page.view[3] - page.view[1]);
          const dpiScale = dpi / 72;
          const pixelScale = Math.sqrt(maxPixels /
            Math.max(1, baseViewport.width * baseViewport.height));
          const scale = Math.min(dpiScale, pixelScale);
          const viewport = page.getViewport({scale});
          const width = Math.max(1, Math.round(viewport.width));
          const height = Math.max(1, Math.round(viewport.height));
          const canvas = document.createElement("canvas");
          canvas.width = width;
          canvas.height = height;
          const context = canvas.getContext("2d", {alpha: false});
          context.save();
          context.fillStyle = "#fff";
          context.fillRect(0, 0, width, height);
          context.restore();
          let released = false;
          try {
            const renderTask = page.render({canvasContext: context, viewport});
            activeRenderTask = renderTask;
            await renderTask.promise;
            lastError = null;
          } catch (error) {
            canvas.width = 1;
            canvas.height = 1;
            page.cleanup();
            throw rememberError(normalizeError(error, pdfjs, "render"));
          } finally {
            activeRenderTask = null;
          }
          return Object.freeze({
            canvas,
            width,
            height,
            rotation: viewport.rotation,
            scale,
            pdfWidth,
            pdfHeight,
            toPdfPoint(x, y) {
              return viewport.convertToPdfPoint(x, y);
            },
            release() {
              if (released) return;
              released = true;
              canvas.width = 1;
              canvas.height = 1;
              page.cleanup();
            }
          });
        },
        cancelRender() {
          if (activeRenderTask) activeRenderTask.cancel();
        },
        async close() {
          if (closed) return;
          closed = true;
          if (activeRenderTask) activeRenderTask.cancel();
          await loadingTask.destroy();
        }
      });
    } catch (error) {
      if (loadingTask) {
        try { await loadingTask.destroy(); } catch (_) {}
      }
      throw rememberError(normalizeError(error, pdfjs, error.phase || "open"));
    }
  }

  function dispose() {
    resetModuleAssets();
    modulePromise = null;
    state = "disposed";
    workerBackend = "disposed";
  }

  global.LwPdf = Object.freeze({
    apiVersion: 1,
    pdfjsVersion: PDFJS_VERSION,
    Error: LwPdfError,
    open,
    getStatus,
    dispose
  });
})(typeof globalThis !== "undefined" ? globalThis : window);
