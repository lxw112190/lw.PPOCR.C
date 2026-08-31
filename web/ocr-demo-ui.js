/*
 * UI glue for the standalone example.
 *
 * Image decoding, RGBA -> BGR conversion, WASM calls, memory ownership, and
 * structured results belong to LwPpocr in lw-ppocr.js. This file only connects
 * DOM controls to that public SDK and adapts results for display/export.
 */
(function() {
  "use strict";

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
  const showImageButton = document.getElementById("show-image");
  const showResultsButton = document.getElementById("show-results");
  const resultsNode = document.getElementById("results");
  const copyTextButton = document.getElementById("copy-text");
  const shareResultButton = document.getElementById("share-result");
  const exportTxtButton = document.getElementById("export-txt");
  const exportJsonButton = document.getElementById("export-json");
  const exportButtons = [copyTextButton, exportTxtButton, exportJsonButton];
  const resultActionButtons = [...exportButtons, shareResultButton];

  let engine = null;
  let enginePromise = null;
  let selectedFile = null;
  let preparedSource = null;
  let originalWidth = 0;
  let originalHeight = 0;
  let prepareMilliseconds = 0;
  let prepareCount = 0;
  let runCount = 0;
  let previewSequence = 0;
  let running = false;
  let lastResults = null;

  function setExportEnabled(enabled) {
    for (const button of resultActionButtons) button.disabled = !enabled;
  }
  function clearLastResults() {
    lastResults = null;
    setExportEnabled(false);
    overlay.replaceChildren();
  }
  function plainTextResult() {
    return lastResults ? lastResults.lines.map(line => line.text).join("\n") : "";
  }
  function escapeHtml(value) {
    return value.replace(/[&<>]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));
  }
  function exportBaseName(source) {
    const leaf = String(source || "").replace(/[\\/]/g, "_");
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
  async function copyPlainText() {
    if (!lastResults) return;
    const result = lastResults;
    const text = plainTextResult();
    try {
      if (!navigator.clipboard || !navigator.clipboard.writeText) throw new Error("Clipboard API unavailable");
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
    if (lastResults === result) statusNode.textContent = "已复制 " + result.lines.length + " 行文本。";
  }
  async function shareResult() {
    if (!lastResults || !navigator.share) return;
    const result = lastResults;
    await navigator.share({title: exportBaseName(result.source) + " 识别结果", text: plainTextResult()});
    if (lastResults === result) statusNode.textContent = "已分享 " + result.lines.length + " 行文本。";
  }
  function exportTxt() {
    if (!lastResults) return;
    const text = plainTextResult();
    downloadResult(text ? text + "\n" : "", "text/plain;charset=utf-8", ".txt");
    statusNode.textContent = "已导出 " + lastResults.lines.length + " 行 TXT。";
  }
  function exportJson() {
    if (!lastResults) return;
    downloadResult(JSON.stringify(lastResults, null, 2) + "\n", "application/json;charset=utf-8", ".json");
    statusNode.textContent = "已导出 " + lastResults.lines.length + " 行 JSON。";
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
  function drawResults(lines, width, height) {
    const namespace = "http://www.w3.org/2000/svg";
    overlay.setAttribute("viewBox", "0 0 " + width + " " + height);
    overlay.replaceChildren();
    lines.forEach((line, index) => {
      const polygon = document.createElementNS(namespace, "polygon");
      polygon.dataset.lineIndex = String(index);
      polygon.setAttribute("points", [
        line.box[0] + "," + line.box[1], line.box[2] + "," + line.box[3],
        line.box[4] + "," + line.box[5], line.box[6] + "," + line.box[7]
      ].join(" "));
      polygon.setAttribute("stroke-width", String(Math.max(2, width / 500)));
      overlay.appendChild(polygon);
      const label = document.createElementNS(namespace, "text");
      label.setAttribute("x", String(line.box[0]));
      label.setAttribute("y", String(Math.max(16, line.box[1] - 3)));
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
  function renderResults(lines) {
    resultsNode.innerHTML = lines.length ? lines.map((line, index) =>
      '<div class="line" data-line-index="' + index + '"><div class="index">' +
      String(index + 1).padStart(2, "0") + '</div><div class="text"><b>' +
      escapeHtml(line.text) + '</b><div class="score">检测 ' +
      (line.det_score * 100).toFixed(1) + '% · 识别 ' +
      (line.rec_score * 100).toFixed(1) + '%</div></div></div>'
    ).join("") : '<div class="empty">未检测到文本。</div>';
  }
  async function decodePreview(file) {
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
      const scale = Math.min(1, 1600 / Math.max(originalWidth, originalHeight));
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
  async function selectFile(file) {
    const sequence = ++previewSequence;
    selectedFile = file || null;
    preparedSource = null;
    clearLastResults();
    runButton.disabled = true;
    setMobilePanel("image");
    if (!file) {
      statusNode.textContent = "就绪，请选择图片。";
      resultsNode.innerHTML = '<div class="empty">选择图片后，这里会显示识别文本。</div>';
      return null;
    }
    try {
      statusNode.textContent = "正在准备图片…";
      const prepared = await decodePreview(file);
      if (sequence !== previewSequence) return null;
      resultsNode.innerHTML = '<div class="empty">图片已准备好，点击“开始识别”执行 OCR。</div>';
      statusNode.textContent = "已选择：" + (file.name || "image") + " · 处理尺寸 " +
        prepared.width + "×" + prepared.height + "，点击“开始识别”。";
      runButton.disabled = !(engine && preparedSource);
      return preparedSource;
    } catch (error) {
      if (sequence === previewSequence) {
        selectedFile = null;
        preparedSource = null;
        statusNode.textContent = "图片预览失败：" + error;
      }
      throw error;
    }
  }
  function adaptResult(result) {
    const xScale = originalWidth / result.image.width;
    const yScale = originalHeight / result.image.height;
    return {
      schema_version: 1,
      source: selectedFile && selectedFile.name ? selectedFile.name : result.source,
      image: {width: originalWidth, height: originalHeight},
      options: result.options,
      elapsed_ms: result.timing.total_ms,
      lines: result.lines.map(line => ({
        ...line,
        box: line.box.map((coordinate, index) =>
          coordinate * (index % 2 ? yScale : xScale))
      }))
    };
  }
  function updateStats(status, result) {
    if (result) {
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
    statusNode.textContent = "正在加载 WASM 和模型…";
    const instance = await LwPpocr.create({useCls: clsInput.checked, maxImageSide: 1600});
    engine = instance;
    updateStats(instance.getStatus());
    runButton.disabled = !preparedSource;
    statusNode.textContent = preparedSource ? "图片已准备好，点击“开始识别”。" : "就绪，请选择图片。";
    document.dispatchEvent(new CustomEvent("lwppocr:ready", {
      detail: {backend: instance.getStatus().backend}
    }));
    return instance;
  }
  async function reconfigureCls() {
    if (!engine || running) return;
    runButton.disabled = true;
    engine.destroy();
    engine = null;
    enginePromise = createEngine();
    await enginePromise;
  }
  async function runOcr() {
    if (!engine || !preparedSource || running) return null;
    running = true;
    runButton.disabled = true;
    clsInput.disabled = true;
    clearLastResults();
    document.body.setAttribute("aria-busy", "true");
    statusNode.textContent = "正在识别，页面仍可正常操作…";
    try {
      const result = await engine.recognize(preparedSource);
      // SDK coordinates are relative to the prepared canvas. Draw those
      // coordinates over the preview, then export coordinates restored to the
      // original image dimensions.
      drawResults(result.lines, result.image.width, result.image.height);
      lastResults = adaptResult(result);
      renderResults(lastResults.lines);
      setExportEnabled(true);
      runCount += 1;
      updateStats(engine.getStatus(), result);
      statusNode.textContent = "完成：" + lastResults.lines.length + " 行";
      setMobilePanel("results");
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
      document.body.removeAttribute("aria-busy");
      running = false;
      clsInput.disabled = false;
      runButton.disabled = !(engine && preparedSource);
    }
  }
  function snapshot() {
    const status = engine ? engine.getStatus() : {ready: false, backend: "loading"};
    return {
      ...status,
      ready: Boolean(engine && status.ready),
      runCount,
      prepareCount,
      prepared: Boolean(preparedSource),
      hasResults: Boolean(lastResults),
      exportEnabled: exportButtons.every(button => !button.disabled)
    };
  }

  fileInput.addEventListener("change", event => selectFile(event.target.files[0] || null).catch(console.error));
  cameraInput.addEventListener("change", event => selectFile(event.target.files[0] || null).catch(console.error));
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
    if (event.dataTransfer.files.length) selectFile(event.dataTransfer.files[0]).catch(console.error);
  });
  runButton.addEventListener("click", () => runOcr().catch(() => {}));
  copyTextButton.addEventListener("click", () => copyPlainText().catch(error => {
    statusNode.textContent = "复制失败：" + error;
  }));
  shareResultButton.addEventListener("click", () => shareResult().catch(error => {
    if (error.name !== "AbortError") statusNode.textContent = "分享失败：" + error;
  }));
  exportTxtButton.addEventListener("click", exportTxt);
  exportJsonButton.addEventListener("click", exportJson);
  showImageButton.addEventListener("click", () => setMobilePanel("image"));
  showResultsButton.addEventListener("click", () => setMobilePanel("results"));
  resultsNode.addEventListener("click", event => {
    const line = event.target.closest("[data-line-index]");
    if (line) selectResultLine(Number(line.dataset.lineIndex));
  });
  shareResultButton.hidden = !navigator.share;

  // Compatibility adapter for existing demo automation. New applications
  // should call LwPpocr.create() directly from the standalone SDK.
  window.lwPpocrDemo = Object.freeze({
    apiVersion: 1,
    ready: async () => {
      await enginePromise;
      if (!engine) throw new Error("OCR 引擎初始化失败");
    },
    selectImage: selectFile,
    recognize: async (file, options = {}) => {
      await window.lwPpocrDemo.ready();
      if (running) {
        throw new LwPpocr.Error(
          "OCR 实例正忙，请等待当前识别完成",
          "LW_OCR_BUSY",
          "busy"
        );
      }
      if (typeof options.useCls === "boolean" && options.useCls !== clsInput.checked) {
        clsInput.checked = options.useCls;
        await reconfigureCls();
      }
      if (file) await selectFile(file);
      if (!preparedSource) {
        throw new LwPpocr.Error(
          "请先选择图片",
          "LW_OCR_INPUT_REQUIRED",
          "decode"
        );
      }
      return runOcr();
    },
    getResult: () => lastResults ? JSON.parse(JSON.stringify(lastResults)) : null,
    getPlainText: plainTextResult,
    getStatus: snapshot
  });
  window.__lwOcrTest = {snapshot, plainTextResult, structuredResult: () => lastResults};
  window.addEventListener("beforeunload", () => {
    if (engine) engine.destroy();
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
