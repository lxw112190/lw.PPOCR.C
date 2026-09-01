/*
 * UI glue for the standalone example.
 *
 * LwPpocr remains an image-only OCR SDK. LwPdf is an optional document
 * frontend that renders one PDF page at a time to Canvas. This file joins both
 * application-layer APIs and owns preview, progress, and export behavior.
 */
(function() {
  "use strict";

  const IMAGE_MAX_SIDE = 1600;
  const PDF_MAX_PIXELS = 5000000;
  const PDF_PREVIEW_MAX_PIXELS = 1500000;
  const PDF_PREVIEW_DPI = 96;

  const fileInput = document.getElementById("file");
  const cameraInput = document.getElementById("camera");
  const dropzone = document.getElementById("dropzone");
  const runButton = document.getElementById("run");
  const clsInput = document.getElementById("use-cls");
  const statusNode = document.getElementById("status");
  const statsNode = document.getElementById("stats");
  const canvas = document.getElementById("canvas");
  const overlay = document.getElementById("overlay");
  const workspace = document.getElementById("workspace");
  const previewTitle = document.getElementById("preview-title");
  const showImageButton = document.getElementById("show-image");
  const showResultsButton = document.getElementById("show-results");
  const resultsNode = document.getElementById("results");
  const copyTextButton = document.getElementById("copy-text");
  const shareResultButton = document.getElementById("share-result");
  const exportTxtButton = document.getElementById("export-txt");
  const exportJsonButton = document.getElementById("export-json");
  const pdfControls = document.getElementById("pdf-controls");
  const pdfMeta = document.getElementById("pdf-meta");
  const pdfScope = document.getElementById("pdf-scope");
  const pdfDpi = document.getElementById("pdf-dpi");
  const pdfPrev = document.getElementById("pdf-prev");
  const pdfNext = document.getElementById("pdf-next");
  const pdfPageLabel = document.getElementById("pdf-page-label");
  const pdfProgress = document.getElementById("pdf-progress");
  const pdfProgressBar = document.getElementById("pdf-progress-bar");
  const pdfProgressLabel = document.getElementById("pdf-progress-label");
  const pdfDiagnostics = document.getElementById("pdf-diagnostics");
  const pdfDiagnosticsText = document.getElementById("pdf-diagnostics-text");
  const copyPdfDiagnosticsButton = document.getElementById("copy-pdf-diagnostics");
  const pdfResultTabs = document.getElementById("pdf-result-tabs");
  const showPageResultButton = document.getElementById("show-page-result");
  const showFullResultButton = document.getElementById("show-full-result");
  const exportButtons = [copyTextButton, exportTxtButton, exportJsonButton];
  const resultActionButtons = [...exportButtons, shareResultButton];

  let engine = null;
  let enginePromise = null;
  let source = null;
  let preparedSource = null;
  let originalWidth = 0;
  let originalHeight = 0;
  let prepareMilliseconds = 0;
  let prepareCount = 0;
  let runCount = 0;
  let previewSequence = 0;
  let running = false;
  let pdfPreviewRunning = false;
  let lastResults = null;
  let lastTimingBreakdown = null;
  let pdfResultView = "page";

  if (!window.LwPdf) {
    fileInput.accept = "image/*";
    const dropTitle = dropzone.querySelector("strong");
    const dropHint = dropzone.querySelector("span");
    const galleryLabel = document.querySelector('.source-button[for="file"]');
    if (dropTitle) dropTitle.textContent = "选择或拖入图片";
    if (dropHint) dropHint.textContent = "支持 JPG、PNG、BMP、WebP，所有文件仅在本机处理";
    if (galleryLabel) galleryLabel.textContent = "从相册选择";
    const headerHint = document.querySelector("header p");
    if (headerHint) {
      headerHint.textContent = "纯浏览器本地推理 · 模型和 WASM 已内嵌 · 图片不会上传到网络";
    }
  }

  function sourceKind(file) {
    return file && (file.type === "application/pdf" || /\.pdf$/i.test(file.name || "")) ?
      "pdf" : "image";
  }
  function pdfStatus() {
    return window.LwPdf && typeof LwPdf.getStatus === "function" ?
      LwPdf.getStatus() : null;
  }
  function clearPdfDiagnostics() {
    pdfDiagnostics.hidden = true;
    pdfDiagnostics.open = false;
    pdfDiagnosticsText.textContent = "";
  }
  function pdfErrorMessage(error) {
    const code = error && error.code ? error.code : "LW_PDF_UNKNOWN";
    const messages = {
      LW_PDF_INIT_FAILED: "当前浏览器无法初始化 PDF 组件，请更新浏览器或改用系统 Chrome/Safari 打开。",
      LW_PDF_INPUT_REQUIRED: "浏览器没有提供可读取的 PDF 文件。",
      LW_PDF_READ_FAILED: "当前浏览器无法读取所选 PDF，请尝试系统 Chrome/Safari 或重新选择文件。",
      LW_PDF_READ_CANCELLED: "PDF 文件读取已取消。",
      LW_PDF_PASSWORD_REQUIRED: "此 PDF 已加密，当前版本暂不支持密码输入。",
      LW_PDF_PASSWORD_INVALID: "PDF 密码错误。",
      LW_PDF_RENDER_FAILED: "PDF 已打开，但首页渲染失败；浏览器兼容性或可用内存可能不足。",
      LW_PDF_LOAD_FAILED: "PDF 解析失败：文件可能损坏，或当前浏览器不支持该 PDF。"
    };
    return (messages[code] || "PDF 打开失败，请查看诊断信息。") + "（" + code + "）";
  }
  function showPdfDiagnostics(error) {
    const status = pdfStatus() || {};
    const report = {
      error_code: error && error.code ? error.code : "LW_PDF_UNKNOWN",
      error_phase: error && error.phase ? error.phase : "unknown",
      pdf: status,
      ocr_backend: engine ? engine.getStatus().backend : "not-ready"
    };
    pdfDiagnosticsText.textContent = JSON.stringify(report, null, 2);
    pdfDiagnostics.hidden = false;
    pdfDiagnostics.open = true;
    return report;
  }
  function setExportEnabled(enabled) {
    for (const button of resultActionButtons) button.disabled = !enabled;
  }
  function resultLineCount(result = lastResults) {
    if (!result) return 0;
    if (result.schema_version === 2) {
      return result.pages.reduce((total, page) => total + page.lines.length, 0);
    }
    return result.lines.length;
  }
  function clearLastResults() {
    lastResults = null;
    lastTimingBreakdown = null;
    setExportEnabled(false);
    pdfResultTabs.hidden = true;
    overlay.replaceChildren();
  }
  function plainTextResult() {
    if (!lastResults) return "";
    if (lastResults.schema_version === 2) {
      return lastResults.pages.map(page =>
        "===== Page " + page.page_number + " / " + lastResults.document.page_count + " =====\n\n" +
        page.lines.map(line => line.text).join("\n")
      ).join("\n\n");
    }
    return lastResults.lines.map(line => line.text).join("\n");
  }
  function escapeHtml(value) {
    return String(value).replace(/[&<>]/g, character =>
      ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[character]));
  }
  function exportBaseName(value) {
    const leaf = String(value || "").replace(/[\\/]/g, "_");
    const dot = leaf.lastIndexOf(".");
    const stem = (dot > 0 ? leaf.slice(0, dot) : leaf)
      .replace(/[<>:"|?*\u0000-\u001f]/g, "_").trim().replace(/[. ]+$/g, "");
    return (stem || "ocr-result") + "-ocr";
  }
  function downloadResult(content, mimeType, extension) {
    if (!lastResults) return;
    const url = URL.createObjectURL(new Blob([content], {type: mimeType}));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = exportBaseName(lastResults.source) + extension;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
  async function copyTextValue(text) {
    try {
      if (!navigator.clipboard || !navigator.clipboard.writeText) {
        throw new Error("Clipboard API unavailable");
      }
      await navigator.clipboard.writeText(text);
    } catch (clipboardError) {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      const copied = document.execCommand("copy");
      textarea.remove();
      if (!copied) throw clipboardError;
    }
  }
  async function copyPlainText() {
    if (!lastResults) return;
    const result = lastResults;
    const text = plainTextResult();
    await copyTextValue(text);
    if (lastResults === result) {
      statusNode.textContent = "已复制 " + resultLineCount(result) + " 行文本。";
    }
  }
  async function shareResult() {
    if (!lastResults || !navigator.share) return;
    const result = lastResults;
    await navigator.share({
      title: exportBaseName(result.source) + " 识别结果",
      text: plainTextResult()
    });
    if (lastResults === result) {
      statusNode.textContent = "已分享 " + resultLineCount(result) + " 行文本。";
    }
  }
  function exportTxt() {
    if (!lastResults) return;
    const text = plainTextResult();
    downloadResult(text ? text + "\n" : "", "text/plain;charset=utf-8", ".txt");
    statusNode.textContent = "已导出 " + resultLineCount() + " 行 TXT。";
  }
  function exportJson() {
    if (!lastResults) return;
    downloadResult(JSON.stringify(lastResults, null, 2) + "\n",
      "application/json;charset=utf-8", ".json");
    statusNode.textContent = "已导出 " + resultLineCount() + " 行 JSON。";
  }
  function setMobilePanel(panel) {
    workspace.dataset.mobilePanel = panel;
    const showingImage = panel === "image";
    showImageButton.classList.toggle("active", showingImage);
    showResultsButton.classList.toggle("active", !showingImage);
    showImageButton.setAttribute("aria-pressed", String(showingImage));
    showResultsButton.setAttribute("aria-pressed", String(!showingImage));
  }
  function drawPreview(image, width, height) {
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(image, 0, 0, width, height);
    overlay.setAttribute("viewBox", "0 0 " + width + " " + height);
    overlay.replaceChildren();
  }
  function drawResults(lines, width, height, xScale = 1, yScale = 1) {
    const namespace = "http://www.w3.org/2000/svg";
    overlay.setAttribute("viewBox", "0 0 " + width + " " + height);
    overlay.replaceChildren();
    lines.forEach((line, index) => {
      const box = line.box.map((coordinate, coordinateIndex) =>
        coordinate * (coordinateIndex % 2 ? yScale : xScale));
      const polygon = document.createElementNS(namespace, "polygon");
      polygon.dataset.lineIndex = String(index);
      polygon.setAttribute("points", [
        box[0] + "," + box[1], box[2] + "," + box[3],
        box[4] + "," + box[5], box[6] + "," + box[7]
      ].join(" "));
      polygon.setAttribute("stroke-width", String(Math.max(2, width / 500)));
      overlay.appendChild(polygon);
      const label = document.createElementNS(namespace, "text");
      label.setAttribute("x", String(box[0]));
      label.setAttribute("y", String(Math.max(16, box[1] - 3)));
      label.textContent = line.text;
      overlay.appendChild(label);
    });
  }
  function selectResultLine(index) {
    for (const node of resultsNode.querySelectorAll("[data-line-index]")) {
      node.classList.toggle("active", Number(node.dataset.lineIndex) === index);
    }
    for (const node of overlay.querySelectorAll("[data-line-index]")) {
      node.classList.toggle("active", Number(node.dataset.lineIndex) === index);
    }
  }
  function renderResults(lines, pageNumber = null, pageCount = null) {
    const heading = pageNumber === null ? "" :
      '<div class="page-heading">Page ' + pageNumber + " / " + pageCount + "</div>";
    resultsNode.innerHTML = heading + (lines.length ? lines.map((line, index) =>
      '<div class="line" data-line-index="' + index + '"><div class="index">' +
      String(index + 1).padStart(2, "0") + '</div><div class="text"><b>' +
      escapeHtml(line.text) + '</b><div class="score">检测 ' +
      (line.det_score * 100).toFixed(1) + "% · 识别 " +
      (line.rec_score * 100).toFixed(1) + "%</div></div></div>"
    ).join("") : '<div class="empty">未检测到文本。</div>');
  }

  function findPdfPageResult(pageNumber) {
    return lastResults && lastResults.schema_version === 2 ?
      lastResults.pages.find(page => page.page_number === pageNumber) || null : null;
  }
  function renderPdfResultView() {
    if (!source || source.kind !== "pdf" || !lastResults || lastResults.schema_version !== 2) {
      return;
    }
    pdfResultTabs.hidden = false;
    const pageMode = pdfResultView === "page";
    showPageResultButton.classList.toggle("active", pageMode);
    showFullResultButton.classList.toggle("active", !pageMode);
    if (!pageMode) {
      resultsNode.innerHTML = '<div class="full-text">' + escapeHtml(plainTextResult()) + "</div>";
      return;
    }
    const page = findPdfPageResult(source.currentPage);
    renderResults(page ? page.lines : [], source.currentPage, source.pageCount);
  }
  async function decodeImagePreview(file) {
    const started = performance.now();
    let image;
    let close = null;
    if (window.createImageBitmap) {
      image = await createImageBitmap(file);
      close = image.close ? () => image.close() : null;
    } else {
      const url = URL.createObjectURL(file);
      try {
        image = new Image();
        image.decoding = "async";
        image.src = url;
        await image.decode();
      } finally {
        URL.revokeObjectURL(url);
      }
    }
    try {
      originalWidth = image.width || image.naturalWidth;
      originalHeight = image.height || image.naturalHeight;
      const scale = Math.min(1, IMAGE_MAX_SIDE / Math.max(originalWidth, originalHeight));
      const width = Math.max(1, Math.round(originalWidth * scale));
      const height = Math.max(1, Math.round(originalHeight * scale));
      drawPreview(image, width, height);
      preparedSource = canvas;
      prepareMilliseconds = performance.now() - started;
      prepareCount += 1;
      return {width, height};
    } finally {
      if (close) close();
    }
  }
  function updatePdfControls() {
    const pdf = source && source.kind === "pdf" ? source : null;
    pdfControls.hidden = !pdf;
    previewTitle.textContent = pdf ? "PDF 页面预览与检测框" : "图像预览与检测框";
    if (!pdf) return;
    pdfMeta.textContent = (pdf.file.name || "document.pdf") + " · " + pdf.pageCount + " 页";
    pdfPageLabel.textContent = pdf.currentPage + " / " + pdf.pageCount;
    pdfPrev.disabled = running || pdfPreviewRunning || pdf.currentPage <= 1;
    pdfNext.disabled = running || pdfPreviewRunning || pdf.currentPage >= pdf.pageCount;
    pdfScope.disabled = running;
    pdfDpi.disabled = running;
  }
  async function renderPdfPreview(pageNumber) {
    if (!source || source.kind !== "pdf" || pdfPreviewRunning) return;
    const pdfSource = source;
    const sequence = ++previewSequence;
    pdfPreviewRunning = true;
    pdfSource.currentPage = pageNumber;
    updatePdfControls();
    statusNode.textContent = "正在渲染第 " + pageNumber + " / " + pdfSource.pageCount + " 页预览…";
    let rendered = null;
    try {
      rendered = await pdfSource.document.renderPage(pageNumber, {
        dpi: PDF_PREVIEW_DPI,
        maxPixels: PDF_PREVIEW_MAX_PIXELS
      });
      if (sequence !== previewSequence || source !== pdfSource) return;
      drawPreview(rendered.canvas, rendered.width, rendered.height);
      const pageResult = findPdfPageResult(pageNumber);
      if (pageResult) {
        drawResults(pageResult.lines, rendered.width, rendered.height,
          rendered.width / pageResult.image.width,
          rendered.height / pageResult.image.height);
      }
      renderPdfResultView();
      const compatibility = pdfStatus() &&
        pdfStatus().worker_backend === "main-thread" ? " · PDF 兼容模式" : "";
      statusNode.textContent = pageResult ?
        "第 " + pageNumber + " 页已识别，共 " + pageResult.lines.length + " 行" +
          compatibility + "。" :
        "PDF 已准备好" + compatibility + "，点击“开始识别”。";
    } catch (error) {
      if (!(error && error.code === "LW_PDF_CANCELLED")) throw error;
    } finally {
      if (rendered) rendered.release();
      pdfPreviewRunning = false;
      updatePdfControls();
    }
  }
  async function disposeSource(oldSource) {
    if (oldSource && oldSource.kind === "pdf") {
      oldSource.cancelled = true;
      oldSource.document.cancelRender();
      await oldSource.document.close();
    }
  }
  async function selectFile(file) {
    if (running) return null;
    const sequence = ++previewSequence;
    const previous = source;
    source = null;
    preparedSource = null;
    clearLastResults();
    clearPdfDiagnostics();
    runButton.disabled = true;
    setMobilePanel("image");
    updatePdfControls();
    await disposeSource(previous);
    if (!file) {
      statusNode.textContent = "就绪，请选择图片或 PDF。";
      resultsNode.innerHTML = '<div class="empty">选择文件后，这里会显示识别文本。</div>';
      return null;
    }
    if (sourceKind(file) === "image") {
      try {
        statusNode.textContent = "正在准备图片…";
        const prepared = await decodeImagePreview(file);
        if (sequence !== previewSequence) return null;
        source = {
          kind: "image",
          file,
          preparedCanvas: canvas,
          originalWidth,
          originalHeight
        };
        resultsNode.innerHTML = '<div class="empty">图片已准备好，点击“开始识别”执行 OCR。</div>';
        statusNode.textContent = "已选择：" + (file.name || "image") + " · 处理尺寸 " +
          prepared.width + "×" + prepared.height + "，点击“开始识别”。";
        runButton.disabled = !engine;
        return preparedSource;
      } catch (error) {
        if (sequence === previewSequence) {
          source = null;
          preparedSource = null;
          statusNode.textContent = "图片预览失败：" + error;
        }
        throw error;
      }
    }
    if (!window.LwPdf) {
      statusNode.textContent = "当前 HTML 构建未包含 PDF 支持。";
      throw new Error("PDF support is disabled");
    }
    const started = performance.now();
    statusNode.textContent = "正在打开 PDF…";
    let documentHandle = null;
    try {
      documentHandle = await LwPdf.open(file);
      if (sequence !== previewSequence) {
        await documentHandle.close();
        return null;
      }
      prepareMilliseconds = performance.now() - started;
      prepareCount += 1;
      source = {
        kind: "pdf",
        file,
        document: documentHandle,
        pageCount: documentHandle.pageCount,
        currentPage: 1,
        cancelled: false,
        pdfLoadMilliseconds: prepareMilliseconds
      };
      updatePdfControls();
      resultsNode.innerHTML = '<div class="empty">PDF 已准备好，点击“开始识别”执行 OCR。</div>';
      await renderPdfPreview(1);
      runButton.disabled = !engine;
      return source;
    } catch (error) {
      if (documentHandle) {
        try {
          await documentHandle.close();
        } catch (_) {
          // Keep the original open/render error as the user-visible failure.
        }
      }
      if (sequence === previewSequence) {
        source = null;
        updatePdfControls();
        const message = pdfErrorMessage(error);
        const diagnostics = showPdfDiagnostics(error);
        statusNode.textContent = message;
        document.dispatchEvent(new CustomEvent("lwppocr:error", {
          detail: {
            phase: error && error.phase ? error.phase : "pdf-open",
            code: error && error.code,
            message,
            diagnostics
          }
        }));
      }
      throw error;
    }
  }

  function adaptImageResult(result) {
    const imageSource = source;
    const xScale = imageSource.originalWidth / result.image.width;
    const yScale = imageSource.originalHeight / result.image.height;
    return {
      schema_version: 1,
      source: imageSource.file.name || result.source,
      image: {width: imageSource.originalWidth, height: imageSource.originalHeight},
      options: result.options,
      elapsed_ms: 0,
      lines: result.lines.map(line => ({
        ...line,
        box: line.box.map((coordinate, index) =>
          coordinate * (index % 2 ? yScale : xScale))
      }))
    };
  }
  function adaptPdfPageResult(pageNumber, rendered, result, renderMilliseconds) {
    const xScale = rendered.width / result.image.width;
    const yScale = rendered.height / result.image.height;
    const lines = result.lines.map(line => {
      const box = line.box.map((coordinate, index) =>
        coordinate * (index % 2 ? yScale : xScale));
      const pdfBox = [];
      for (let index = 0; index < box.length; index += 2) {
        const point = rendered.toPdfPoint(box[index], box[index + 1]);
        pdfBox.push(Number(point[0].toFixed(3)), Number(point[1].toFixed(3)));
      }
      return {...line, box, pdf_box: pdfBox};
    });
    return {
      page_number: pageNumber,
      pdf: {
        width_pt: Number(rendered.pdfWidth.toFixed(3)),
        height_pt: Number(rendered.pdfHeight.toFixed(3)),
        rotation: rendered.rotation
      },
      image: {width: rendered.width, height: rendered.height},
      timing: {
        render_ms: Number(renderMilliseconds.toFixed(3)),
        inference_ms: Number(result.timing.total_ms.toFixed(3)),
        total_ms: 0
      },
      lines
    };
  }
  function updateStats(status, result) {
    if (source && source.kind === "pdf" && lastResults && lastResults.schema_version === 2) {
      const timing = lastResults.timing;
      statsNode.textContent = lastResults.document.processed_pages + " / " +
        lastResults.document.page_count + " 页 · " + resultLineCount() + " 行 · PDF 打开 " +
        timing.pdf_load_ms.toFixed(0) + " ms · 页面渲染 " +
        timing.render_ms.toFixed(0) + " ms · OCR " +
        timing.inference_ms.toFixed(0) + " ms · 总计 " + timing.total_ms.toFixed(0) + " ms";
    } else if (result) {
      statsNode.textContent = "第 " + runCount + " 次 · " +
        (status.backend === "worker" ? "后台线程" : "兼容模式") +
        " · 图像准备 " + prepareMilliseconds.toFixed(0) + " ms · 推理 " +
        result.timing.inference_ms.toFixed(0) + " ms · 总计 " +
        result.timing.total_ms.toFixed(0) + " ms · CLS " +
        (result.options.use_cls ? "开启" : "关闭");
    } else {
      statsNode.textContent = "引擎已就绪 · " +
        (status.backend === "worker" ? "后台线程" : "兼容模式") +
        " · 输出 " + status.maxLineCapacity + " 行/" +
        status.maxTextCapacity + " 字节 · CLS " +
        (clsInput.checked ? "开启" : "关闭");
    }
  }
  async function createEngine() {
    clsInput.disabled = true;
    statusNode.textContent = "正在加载 WASM 和模型…";
    try {
      const instance = await LwPpocr.create({
        useCls: clsInput.checked,
        // The Demo owns image and PDF page sizing. The reusable SDK default
        // remains 1600 for third-party callers.
        maxImageSide: 0
      });
      engine = instance;
      updateStats(instance.getStatus());
      runButton.disabled = !source;
      statusNode.textContent = source ?
        "文件已准备好，点击“开始识别”。" : "就绪，请选择图片或 PDF。";
      document.dispatchEvent(new CustomEvent("lwppocr:ready", {
        detail: {backend: instance.getStatus().backend}
      }));
      return instance;
    } finally {
      clsInput.disabled = false;
    }
  }
  async function reconfigureCls() {
    if (!engine || running) return;
    runButton.disabled = true;
    engine.destroy();
    engine = null;
    enginePromise = createEngine();
    await enginePromise;
  }
  function setRunningState(value, canStop = false) {
    running = value;
    fileInput.disabled = value;
    cameraInput.disabled = value;
    clsInput.disabled = value;
    runButton.disabled = value && !canStop;
    runButton.classList.toggle("stop", value && canStop);
    runButton.textContent = value && canStop ? "停止" : "开始识别";
    document.body.toggleAttribute("aria-busy", value);
    updatePdfControls();
  }
  async function runImageOcr() {
    if (!engine || !source || source.kind !== "image" || running) return null;
    setRunningState(true);
    clearLastResults();
    statusNode.textContent = "正在识别，页面仍可正常操作…";
    try {
      const result = await engine.recognize(source.preparedCanvas);
      const uiStarted = performance.now();
      drawResults(result.lines, result.image.width, result.image.height);
      lastResults = adaptImageResult(result);
      renderResults(lastResults.lines);
      setExportEnabled(true);
      runCount += 1;
      setMobilePanel("results");
      const uiMilliseconds = performance.now() - uiStarted;
      lastTimingBreakdown = {
        prepareMilliseconds,
        sdkTotalMilliseconds: result.timing.total_ms,
        uiMilliseconds
      };
      lastResults.elapsed_ms = Number((prepareMilliseconds +
        result.timing.total_ms + uiMilliseconds).toFixed(3));
      updateStats(engine.getStatus(), result);
      statusNode.textContent = "完成：" + lastResults.lines.length + " 行";
      document.dispatchEvent(new CustomEvent("lwppocr:result", {detail: lastResults}));
      return lastResults;
    } catch (error) {
      statusNode.textContent = "识别失败：" + error;
      document.dispatchEvent(new CustomEvent("lwppocr:error", {
        detail: {phase: "recognize", message: String(error)}
      }));
      console.error(error);
      throw error;
    } finally {
      setRunningState(false);
      runButton.disabled = !(engine && source);
    }
  }
  async function nextAnimationFrame() {
    await new Promise(resolve => requestAnimationFrame(resolve));
  }

  async function runPdfOcr() {
    if (!engine || !source || source.kind !== "pdf" || running) return null;
    const pdfSource = source;
    pdfSource.cancelled = false;
    const pages = pdfScope.value === "current" ? [pdfSource.currentPage] :
      Array.from({length: pdfSource.pageCount}, (_, index) => index + 1);
    const dpi = Number(pdfDpi.value);
    const runStarted = performance.now();
    let renderTotal = 0;
    let inferenceTotal = 0;
    let uiTotal = 0;
    clearLastResults();
    lastResults = {
      schema_version: 2,
      source_type: "pdf",
      source: pdfSource.file.name || "document.pdf",
      document: {page_count: pdfSource.pageCount, processed_pages: 0},
      options: {use_cls: clsInput.checked, pdf_dpi: dpi, pdf_max_pixels: PDF_MAX_PIXELS},
      timing: {
        pdf_load_ms: Number(pdfSource.pdfLoadMilliseconds.toFixed(3)),
        render_ms: 0,
        inference_ms: 0,
        ui_ms: 0,
        total_ms: 0
      },
      pages: []
    };
    pdfResultTabs.hidden = false;
    pdfResultView = "page";
    pdfProgress.hidden = false;
    pdfProgressBar.max = pages.length;
    pdfProgressBar.value = 0;
    pdfProgressLabel.textContent = "0 / " + pages.length;
    setRunningState(true, true);
    setExportEnabled(false);
    try {
      for (let pageIndex = 0; pageIndex < pages.length; pageIndex += 1) {
        if (pdfSource.cancelled || source !== pdfSource) break;
        const pageNumber = pages[pageIndex];
        pdfSource.currentPage = pageNumber;
        updatePdfControls();
        statusNode.textContent = "正在处理第 " + pageNumber + " / " +
          pdfSource.pageCount + " 页：渲染…";
        let rendered = null;
        try {
          const renderStarted = performance.now();
          rendered = await pdfSource.document.renderPage(pageNumber, {
            dpi,
            maxPixels: PDF_MAX_PIXELS
          });
          const renderMilliseconds = performance.now() - renderStarted;
          renderTotal += renderMilliseconds;
          if (pdfSource.cancelled || source !== pdfSource) break;
          statusNode.textContent = "正在处理第 " + pageNumber + " / " +
            pdfSource.pageCount + " 页：OCR…";
          const ocrResult = await engine.recognize(rendered.canvas);
          inferenceTotal += ocrResult.timing.total_ms;
          if (source !== pdfSource) break;
          const uiStarted = performance.now();
          const pageResult = adaptPdfPageResult(
            pageNumber, rendered, ocrResult, renderMilliseconds);
          drawPreview(rendered.canvas, rendered.width, rendered.height);
          drawResults(pageResult.lines, rendered.width, rendered.height);
          pageResult.timing.total_ms = Number((renderMilliseconds +
            ocrResult.timing.total_ms).toFixed(3));
          lastResults.pages.push(pageResult);
          lastResults.document.processed_pages = lastResults.pages.length;
          pdfResultView = "page";
          renderPdfResultView();
          uiTotal += performance.now() - uiStarted;
          lastResults.timing.render_ms = Number(renderTotal.toFixed(3));
          lastResults.timing.inference_ms = Number(inferenceTotal.toFixed(3));
          lastResults.timing.ui_ms = Number(uiTotal.toFixed(3));
          lastResults.timing.total_ms = Number((pdfSource.pdfLoadMilliseconds +
            renderTotal + inferenceTotal + uiTotal).toFixed(3));
          setExportEnabled(true);
          pdfProgressBar.value = pageIndex + 1;
          pdfProgressLabel.textContent = (pageIndex + 1) + " / " + pages.length;
          updateStats(engine.getStatus());
          await nextAnimationFrame();
        } catch (error) {
          if (pdfSource.cancelled && error && error.code === "LW_PDF_CANCELLED") break;
          throw error;
        } finally {
          if (rendered) rendered.release();
        }
      }
      runCount += 1;
      setMobilePanel("results");
      lastResults.timing.total_ms = Number((pdfSource.pdfLoadMilliseconds +
        performance.now() - runStarted).toFixed(3));
      lastTimingBreakdown = JSON.parse(JSON.stringify(lastResults.timing));
      const stopped = pdfSource.cancelled;
      statusNode.textContent = stopped ?
        "已停止：完成 " + lastResults.document.processed_pages + " / " +
          pdfSource.pageCount + " 页。" :
        "完成：" + lastResults.document.processed_pages + " 页，共 " +
          resultLineCount() + " 行。";
      if (lastResults.pages.length) {
        document.dispatchEvent(new CustomEvent("lwppocr:result", {detail: lastResults}));
      } else {
        clearLastResults();
      }
      return lastResults;
    } catch (error) {
      statusNode.textContent = "PDF 识别失败：" + error;
      document.dispatchEvent(new CustomEvent("lwppocr:error", {
        detail: {phase: "pdf-recognize", code: error && error.code, message: String(error)}
      }));
      console.error(error);
      throw error;
    } finally {
      pdfProgress.hidden = true;
      setRunningState(false);
      runButton.disabled = !(engine && source);
      updatePdfControls();
    }
  }
  async function runOcr() {
    if (!source) return null;
    return source.kind === "pdf" ? runPdfOcr() : runImageOcr();
  }
  function cancelPdfOcr() {
    if (!running || !source || source.kind !== "pdf") return;
    source.cancelled = true;
    source.document.cancelRender();
    runButton.disabled = true;
    statusNode.textContent = "正在停止；若当前页已进入 OCR，将在本页完成后停止…";
  }
  async function navigatePdf(delta) {
    if (!source || source.kind !== "pdf" || running || pdfPreviewRunning) return;
    const pageNumber = Math.max(1, Math.min(source.pageCount, source.currentPage + delta));
    if (pageNumber !== source.currentPage) await renderPdfPreview(pageNumber);
  }
  function snapshot() {
    const status = engine ? engine.getStatus() : {ready: false, backend: "loading"};
    const pdf = source && source.kind === "pdf" ? source : null;
    const documentStatus = pdfStatus();
    return {
      ...status,
      ready: Boolean(engine && status.ready),
      runCount,
      prepareCount,
      prepared: Boolean(source && (preparedSource || pdf)),
      sourceKind: source ? source.kind : null,
      pdfPageCount: pdf ? pdf.pageCount : 0,
      pdfCurrentPage: pdf ? pdf.currentPage : 0,
      pdfWorkerBackend: documentStatus ? documentStatus.worker_backend : null,
      pdfErrorCode: documentStatus && documentStatus.last_error ?
        documentStatus.last_error.code : null,
      processedPages: lastResults && lastResults.schema_version === 2 ?
        lastResults.document.processed_pages : 0,
      hasResults: Boolean(lastResults),
      exportEnabled: exportButtons.every(button => !button.disabled)
    };
  }

  fileInput.addEventListener("change", event =>
    selectFile(event.target.files[0] || null).catch(() => {}));
  cameraInput.addEventListener("change", event =>
    selectFile(event.target.files[0] || null).catch(() => {}));
  clsInput.addEventListener("change", () => reconfigureCls().catch(error => {
    statusNode.textContent = "切换 CLS 失败：" + error;
  }));
  ["dragenter", "dragover"].forEach(type => dropzone.addEventListener(type, event => {
    event.preventDefault();
    dropzone.classList.add("drag");
  }));
  ["dragleave", "drop"].forEach(type => dropzone.addEventListener(type, event => {
    event.preventDefault();
    dropzone.classList.remove("drag");
  }));
  dropzone.addEventListener("drop", event => {
    if (event.dataTransfer.files.length) {
      selectFile(event.dataTransfer.files[0]).catch(() => {});
    }
  });
  runButton.addEventListener("click", () => {
    if (running) cancelPdfOcr();
    else runOcr().catch(() => {});
  });
  pdfPrev.addEventListener("click", () => navigatePdf(-1).catch(console.error));
  pdfNext.addEventListener("click", () => navigatePdf(1).catch(console.error));
  copyTextButton.addEventListener("click", () => copyPlainText().catch(error => {
    statusNode.textContent = "复制失败：" + error;
  }));
  shareResultButton.addEventListener("click", () => shareResult().catch(error => {
    if (error.name !== "AbortError") statusNode.textContent = "分享失败：" + error;
  }));
  exportTxtButton.addEventListener("click", exportTxt);
  exportJsonButton.addEventListener("click", exportJson);
  copyPdfDiagnosticsButton.addEventListener("click", () =>
    copyTextValue(pdfDiagnosticsText.textContent).then(() => {
      statusNode.textContent = "PDF 诊断信息已复制。";
    }).catch(error => {
      statusNode.textContent = "复制诊断信息失败：" + error;
    }));
  showImageButton.addEventListener("click", () => setMobilePanel("image"));
  showResultsButton.addEventListener("click", () => setMobilePanel("results"));
  showPageResultButton.addEventListener("click", () => {
    pdfResultView = "page";
    renderPdfResultView();
  });
  showFullResultButton.addEventListener("click", () => {
    pdfResultView = "full";
    renderPdfResultView();
  });
  resultsNode.addEventListener("click", event => {
    const line = event.target.closest("[data-line-index]");
    if (line) selectResultLine(Number(line.dataset.lineIndex));
  });
  shareResultButton.hidden = !navigator.share;

  // Compatibility adapter for existing image-only Demo automation. PDF support
  // remains a UI/document-frontend feature and does not change this v1 API.
  window.lwPpocrDemo = Object.freeze({
    apiVersion: 1,
    ready: async () => {
      await enginePromise;
      if (!engine) throw new Error("OCR 引擎初始化失败");
    },
    selectImage: async file => {
      if (file && sourceKind(file) !== "image") {
        throw new LwPpocr.Error("selectImage() 仅接受图片", "LW_OCR_DECODE", "decode");
      }
      return selectFile(file);
    },
    recognize: async (file, options = {}) => {
      await window.lwPpocrDemo.ready();
      if (running) {
        throw new LwPpocr.Error("OCR 实例正忙，请等待当前识别完成", "LW_OCR_BUSY", "busy");
      }
      if (file && sourceKind(file) !== "image") {
        throw new LwPpocr.Error("recognize() 仅接受图片", "LW_OCR_DECODE", "decode");
      }
      if (typeof options.useCls === "boolean" && options.useCls !== clsInput.checked) {
        clsInput.checked = options.useCls;
        await reconfigureCls();
      }
      if (file) await selectFile(file);
      if (!source || source.kind !== "image" || !preparedSource) {
        throw new LwPpocr.Error("请先选择图片", "LW_OCR_INPUT_REQUIRED", "decode");
      }
      return runImageOcr();
    },
    getResult: () => lastResults ? JSON.parse(JSON.stringify(lastResults)) : null,
    getPlainText: plainTextResult,
    getStatus: snapshot
  });
  window.__lwOcrTest = {
    snapshot,
    plainTextResult,
    structuredResult: () => lastResults,
    timingBreakdown: () => lastTimingBreakdown ?
      JSON.parse(JSON.stringify(lastTimingBreakdown)) : null,
    pdfStatus: () => pdfStatus(),
    selectFile,
    runOcr,
    cancelPdfOcr
  };
  window.addEventListener("beforeunload", () => {
    if (source && source.kind === "pdf") source.document.close();
    if (engine) engine.destroy();
    if (window.LwPdf) LwPdf.dispose();
  });
  enginePromise = createEngine().catch(error => {
    engine = null;
    statsNode.textContent = "引擎加载失败";
    statusNode.textContent = "WASM 初始化失败：" + error;
    document.dispatchEvent(new CustomEvent("lwppocr:error", {
      detail: {phase: "initialize", message: String(error)}
    }));
    console.error(error);
  });
})();
