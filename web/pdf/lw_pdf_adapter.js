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
  let modulePromise = null;
  let coreUrl = null;
  let workerUrl = null;

  class LwPdfError extends Error {
    constructor(message, code, phase, cause) {
      super(message);
      this.name = "LwPdfError";
      this.code = code;
      this.phase = phase;
      if (cause !== undefined) this.cause = cause;
    }
  }

  function base64BlobUrl(encoded) {
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let offset = 0; offset < binary.length; offset += 1) {
      bytes[offset] = binary.charCodeAt(offset);
    }
    return URL.createObjectURL(new Blob([bytes], {type: "text/javascript"}));
  }

  async function loadPdfJs() {
    if (!modulePromise) {
      modulePromise = (async () => {
        coreUrl = base64BlobUrl(PDFJS_CORE_BASE64);
        workerUrl = base64BlobUrl(PDFJS_WORKER_BASE64);
        const pdfjs = await import(coreUrl);
        pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;
        return pdfjs;
      })().catch(error => {
        if (coreUrl) URL.revokeObjectURL(coreUrl);
        if (workerUrl) URL.revokeObjectURL(workerUrl);
        coreUrl = null;
        workerUrl = null;
        modulePromise = null;
        throw new LwPdfError(
          "PDF 组件初始化失败",
          "LW_PDF_INIT_FAILED",
          "initialize",
          error
        );
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

  function assertPageNumber(pageNumber, pageCount) {
    if (!Number.isInteger(pageNumber) || pageNumber < 1 || pageNumber > pageCount) {
      throw new LwPdfError("PDF 页码超出范围", "LW_PDF_PAGE_FAILED", "render");
    }
  }

  async function open(source, options = {}) {
    const pdfjs = await loadPdfJs();
    if (!source || typeof source.arrayBuffer !== "function") {
      throw new LwPdfError("PDF 输入必须是 File 或 Blob", "LW_PDF_INPUT_REQUIRED", "open");
    }
    let loadingTask = null;
    try {
      const data = new Uint8Array(await source.arrayBuffer());
      loadingTask = pdfjs.getDocument({
        data,
        password: options.password || undefined,
        isEvalSupported: false,
        useWasm: false
      });
      const pdfDocument = await loadingTask.promise;
      let closed = false;
      let activeRenderTask = null;

      return Object.freeze({
        pageCount: pdfDocument.numPages,
        async renderPage(pageNumber, renderOptions = {}) {
          if (closed) {
            throw new LwPdfError("PDF 文档已经关闭", "LW_PDF_CLOSED", "render");
          }
          assertPageNumber(pageNumber, pdfDocument.numPages);
          const dpi = renderOptions.dpi === undefined ? 180 : Number(renderOptions.dpi);
          const maxPixels = renderOptions.maxPixels === undefined ? 5000000 :
            Number(renderOptions.maxPixels);
          if (!Number.isFinite(dpi) || dpi <= 0 || !Number.isFinite(maxPixels) ||
              maxPixels < 1) {
            throw new LwPdfError("PDF 渲染参数无效", "LW_PDF_OPTIONS", "render");
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
          } catch (error) {
            canvas.width = 1;
            canvas.height = 1;
            page.cleanup();
            throw normalizeError(error, pdfjs, "render");
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
      throw normalizeError(error, pdfjs, "open");
    }
  }

  function dispose() {
    if (coreUrl) URL.revokeObjectURL(coreUrl);
    if (workerUrl) URL.revokeObjectURL(workerUrl);
    coreUrl = null;
    workerUrl = null;
    modulePromise = null;
  }

  global.LwPdf = Object.freeze({
    apiVersion: 1,
    pdfjsVersion: PDFJS_VERSION,
    Error: LwPdfError,
    open,
    dispose
  });
})(typeof globalThis !== "undefined" ? globalThis : window);
